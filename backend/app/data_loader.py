from __future__ import annotations

import csv
import json
import os
import re
from functools import lru_cache
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    from together import Together
except Exception:  # pragma: no cover - optional dependency
    Together = None


APP_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = APP_DIR.parents[1]
MATERIALS_DIR = WORKSPACE_ROOT / "materials"
PART1_DIR = MATERIALS_DIR / "project_part1"
PART2_DIR = MATERIALS_DIR / "project_part2"
PART3_DIR = MATERIALS_DIR / "Project_part_3"

CORPUS_DIR = PART1_DIR / "data" / "open_ragbench" / "pdf" / "arxiv" / "corpus"
QUERIES_PATH = PART1_DIR / "data" / "open_ragbench" / "pdf" / "arxiv" / "queries.json"
QRELS_PATH = PART1_DIR / "data" / "open_ragbench" / "pdf" / "arxiv" / "qrels.json"
ANSWERS_PATH = PART1_DIR / "data" / "open_ragbench" / "pdf" / "arxiv" / "answers.json"
CHUNK_STATS_PATH = PART1_DIR / "data" / "processed" / "chunk_stats.json"
GENERATION_SUMMARY_PATH = PART3_DIR / "evaluation_summary.csv"
DEMO_CHUNK_PATH = PART1_DIR / "data" / "processed" / "chunks_section_aware.jsonl"
DEMO_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEMO_EMBEDDING_CACHE = PART2_DIR / "chunks_section_aware_BAAI_bge-base-en-v1.5_fp16.npy"
DEMO_GENERATOR_MODEL = os.getenv("ACADEMIC_RAG_GENERATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo")

TOP_K_FILES = {
    3: PART2_DIR / "top_K_3.csv",
    5: PART2_DIR / "top_K_5.csv",
    10: PART2_DIR / "top_K_10.csv",
    15: PART2_DIR / "top_K_15.csv",
}

WORD_PATTERN = re.compile(r"[A-Za-z0-9\-]{2,}")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _clean_doc_id(doc_id: str) -> str:
    return doc_id.split("v")[0] if "v" in doc_id else doc_id


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_PATTERN.findall(text)]


def _score_text(text: str, question_tokens: set[str]) -> float:
    text_tokens = _tokenize(text)
    if not text_tokens or not question_tokens:
        return 0.0
    text_token_set = set(text_tokens)
    overlap = question_tokens & text_token_set
    density = sum(1 for token in text_tokens if token in question_tokens) / len(text_tokens)
    return len(overlap) + density


def _pick_sentences(text: str, question_tokens: set[str], max_sentences: int = 2) -> list[str]:
    sentences = SENTENCE_SPLIT_PATTERN.split(text.strip())
    ranked: list[tuple[float, str]] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 40:
            continue
        ranked.append((_score_text(sentence, question_tokens), sentence))
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen = [sentence for score, sentence in ranked[:max_sentences] if score > 0]
    if chosen:
        return chosen
    return [sentences[0].strip()] if sentences and sentences[0].strip() else []


def _looks_like_noise_chunk(text: str) -> bool:
    stripped = text.strip()
    if "....." in stripped:
        return True
    if stripped.startswith("# Appendices"):
        return True
    return False


def _is_summary_question(question: str) -> bool:
    question_lower = question.lower()
    keywords = ["summary", "summarize", "main contribution", "main idea", "overview", "what does this paper do"]
    return any(keyword in question_lower for keyword in keywords)


def _answer_from_retrieved_chunks(question: str, chunks: list[dict], strategy: str = "Structured") -> str:
    if _is_summary_question(question):
        for source_index, chunk in enumerate(chunks, start=1):
            if chunk.get("section_id") not in {-1, "abstract"}:
                continue
            sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(chunk.get("text", "")) if len(sentence.strip()) >= 40]
            if sentences:
                selected = sentences[:2]
                if strategy == "Naive":
                    return " ".join(selected)
                return " ".join(f"{sentence} [Source {source_index}]" for sentence in selected)

    question_tokens = set(_tokenize(question))
    candidates: list[tuple[float, str, int]] = []

    for source_index, chunk in enumerate(chunks, start=1):
        for sentence in SENTENCE_SPLIT_PATTERN.split(chunk.get("text", "")):
            sentence = sentence.strip()
            if len(sentence) < 45:
                continue
            score = _score_text(sentence, question_tokens)
            candidates.append((score, sentence, source_index))

    candidates.sort(key=lambda item: item[0], reverse=True)
    top_sentences = [(sentence, source_index) for score, sentence, source_index in candidates[:3] if score > 0]
    if not top_sentences:
        return "Insufficient information."

    if strategy == "Naive":
        return " ".join(sentence for sentence, _ in top_sentences[:2])

    if strategy == "Structured":
        return " ".join(f"{sentence} [Source {source_index}]" for sentence, source_index in top_sentences[:2])

    evidence = "; ".join(f"Source {source_index}" for _, source_index in top_sentences[:2])
    final = " ".join(f"{sentence} [Source {source_index}]" for sentence, source_index in top_sentences[:2])
    return f"Relevant evidence comes from {evidence}. Final answer: {final}"


def _demo_prompt(question: str, context: str) -> str:
    return f"""You are an academic research assistant. Answer the user's question using ONLY the provided context.

Rules:
- Cite evidence with [Source N]
- If the context is insufficient, say "Insufficient information"
- Keep the answer concise and grounded

Context:
{context}

Question: {question}

Answer:"""


def _join_demo_context(chunks: list[dict], max_chars: int = 750) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        label = chunk.get("label") or chunk.get("title") or f"Section {chunk.get('section_id')}"
        text = (chunk.get("text") or "")[:max_chars]
        parts.append(f"[Source {index} | {label}]\n{text}")
    return "\n\n".join(parts)


def _sanitize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.nan_to_num(embeddings.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return embeddings / norms


def _sanitize_query_embedding(query_embedding: np.ndarray) -> np.ndarray:
    query_embedding = np.nan_to_num(query_embedding.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(query_embedding))
    if norm <= 1e-12:
        return query_embedding
    return query_embedding / norm


class DemoGenerator:
    def __init__(self, model_name: str):
        if Together is None:
            raise RuntimeError("The 'together' package is not installed.")
        api_key = os.getenv("TOGETHER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY is not set.")
        self.client = Together(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=220,
        )
        return response.choices[0].message.content.strip()


@lru_cache(maxsize=1)
def load_chunk_stats() -> dict:
    return _load_json(CHUNK_STATS_PATH)


@lru_cache(maxsize=1)
def load_retrieval_rows() -> list[dict]:
    rows: list[dict] = []
    for top_k, path in TOP_K_FILES.items():
        with open(path, "r", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                metric_name = next(key for key in row if key.startswith("ChunkHit@"))
                rows.append(
                    {
                        "top_k": top_k,
                        "chunking": row["Chunking"],
                        "embedding": row["Embedding"],
                        "hit_rate": float(row[metric_name]),
                    }
                )
    return rows


def load_generation_summary() -> list[dict]:
    if not GENERATION_SUMMARY_PATH.exists():
        return []

    rows: list[dict] = []
    with open(GENERATION_SUMMARY_PATH, "r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append(
                {
                    "strategy": row["Strategy"],
                    "rouge_l_mean": float(row["ROUGE-L_mean"]),
                    "bertscore_f1_mean": float(row["BERTScore_F1_mean"]),
                    "faithfulness_mean": float(row["Faithfulness_mean"]),
                    "latency_s_mean": float(row["Latency_s_mean"]),
                    "hallucination_count": int(row["Hallucination_count"]),
                    "too_generic_count": int(row["Too_Generic_count"]),
                }
            )
    return rows


@lru_cache(maxsize=1)
def load_queries() -> dict:
    return _load_json(QUERIES_PATH)


@lru_cache(maxsize=1)
def load_qrels() -> dict:
    return _load_json(QRELS_PATH)


@lru_cache(maxsize=1)
def load_answers() -> dict:
    return _load_json(ANSWERS_PATH)


@lru_cache(maxsize=1)
def load_paper_index() -> list[dict]:
    question_counts = load_question_counts()
    papers: list[dict] = []
    for path in sorted(CORPUS_DIR.glob("*.json")):
        doc = _load_json(path)
        paper_id = doc.get("id", path.stem)
        papers.append(
            {
                "id": paper_id,
                "title": doc.get("title", path.stem),
                "abstract": doc.get("abstract", ""),
                "categories": doc.get("categories", []),
                "authors": doc.get("authors", []),
                "section_count": len(doc.get("sections", [])),
                "question_count": question_counts.get(_clean_doc_id(paper_id), 0),
            }
        )
    return papers


@lru_cache(maxsize=256)
def load_paper(paper_id: str) -> dict:
    paper_path = CORPUS_DIR / f"{paper_id}.json"
    return _load_json(paper_path)


def get_papers(query: str | None = None, limit: int = 40) -> list[dict]:
    papers = load_paper_index()
    if query:
        query_lower = query.lower()
        papers = [
            paper
            for paper in papers
            if query_lower in paper["title"].lower()
            or any(query_lower in category.lower() for category in paper["categories"])
        ]
    papers.sort(key=lambda paper: (-paper["question_count"], paper["title"]))
    return papers[:limit]


def get_paper_detail(paper_id: str) -> dict:
    doc = load_paper(paper_id)
    sections = doc.get("sections", [])
    preview_sections = []
    for index, section in enumerate(sections[:8]):
        preview_sections.append(
            {
                "section_id": index,
                "label": f"Section {index + 1}",
                "snippet": section.get("text", "")[:280],
            }
        )
    return {
        "id": doc.get("id", paper_id),
        "title": doc.get("title", paper_id),
        "abstract": doc.get("abstract", ""),
        "authors": doc.get("authors", []),
        "categories": doc.get("categories", []),
        "section_count": len(sections),
        "preview_sections": preview_sections,
    }


def get_sample_questions(paper_id: str, limit: int = 6) -> list[dict]:
    queries = load_queries()
    qrels = load_qrels()
    answers = load_answers()
    paper_base = _clean_doc_id(paper_id)

    matches: list[dict] = []
    for query_id, rel in qrels.items():
        doc_id = rel["doc_id"] if isinstance(rel, dict) else ""
        if _clean_doc_id(str(doc_id)) != paper_base:
            continue
        query_payload = queries.get(query_id, {})
        question = query_payload.get("query") or query_payload.get("text") or query_payload.get("question")
        if not question:
            continue
        matches.append(
            {
                "query_id": query_id,
                "question": question,
                "reference_answer": answers.get(query_id, ""),
                "query_type": query_payload.get("type", "unknown"),
                "source": query_payload.get("source", "text"),
            }
        )
    return matches[:limit]


@lru_cache(maxsize=1)
def load_question_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for rel in load_qrels().values():
        if isinstance(rel, dict):
            doc_id = str(rel.get("doc_id", ""))
            if doc_id:
                key = _clean_doc_id(doc_id)
                counts[key] = counts.get(key, 0) + 1
    return counts


@lru_cache(maxsize=1)
def load_demo_chunks() -> list[dict]:
    if not DEMO_CHUNK_PATH.exists():
        return []
    rows = _load_jsonl(DEMO_CHUNK_PATH)
    cleaned: list[dict] = []
    for row in rows:
        text = (row.get("text") or "").strip()
        section_id = row.get("section_id")
        label = "Abstract" if section_id in {-1, "abstract"} else f"Section {section_id}"
        cleaned.append(
            {
                "chunk_id": row.get("chunk_id"),
                "doc_id": row.get("doc_id"),
                "section_id": section_id,
                "label": label,
                "title": row.get("title"),
                "text": text,
            }
        )
    return cleaned


@lru_cache(maxsize=1)
def load_demo_embedder() -> SentenceTransformer | None:
    if SentenceTransformer is None:
        return None
    return SentenceTransformer(DEMO_EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def load_demo_chunk_embeddings() -> np.ndarray | None:
    if np is None:
        return None
    if DEMO_EMBEDDING_CACHE.exists():
        return _sanitize_embeddings(np.load(DEMO_EMBEDDING_CACHE).astype(np.float32))

    model = load_demo_embedder()
    chunks = load_demo_chunks()
    if model is None or not chunks:
        return None

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    embeddings = _sanitize_embeddings(embeddings)
    np.save(DEMO_EMBEDDING_CACHE, embeddings.astype(np.float16))
    return embeddings


@lru_cache(maxsize=1)
def load_demo_generator() -> DemoGenerator | None:
    try:
        return DemoGenerator(DEMO_GENERATOR_MODEL)
    except Exception:
        return None


def _demo_backend_status() -> str:
    retrieval = "embedding retrieval" if load_demo_embedder() is not None else "lexical retrieval fallback"
    generation = "Together model-backed generation" if load_demo_generator() is not None else "local grounded fallback generation"
    return f"{retrieval} + {generation}"


def _paper_sections_for_demo(paper_id: str) -> list[dict]:
    doc = load_paper(paper_id)
    sections = []

    abstract = (doc.get("abstract") or "").strip()
    if abstract:
        sections.append({"section_id": "abstract", "label": "Abstract", "text": abstract})

    for index, section in enumerate(doc.get("sections", [])):
        text = (section.get("text") or "").strip()
        if len(text) < 40:
            continue
        sections.append(
            {
                "section_id": index,
                "label": f"Section {index + 1}",
                "text": text,
            }
        )
    return sections


def _retrieve_demo_chunks(paper_id: str, question: str, top_k: int) -> tuple[list[dict], str]:
    paper_base = _clean_doc_id(paper_id)
    chunks = load_demo_chunks()
    model = load_demo_embedder()
    embeddings = load_demo_chunk_embeddings()
    question_tokens = set(_tokenize(question))
    wants_summary = _is_summary_question(question)

    if model is not None and embeddings is not None and len(chunks) == len(embeddings):
        paper_indices = [
            index
            for index, chunk in enumerate(chunks)
            if _clean_doc_id(str(chunk.get("doc_id", ""))) == paper_base and len((chunk.get("text") or "").strip()) >= 40
        ]
        non_noise_indices = [index for index in paper_indices if not _looks_like_noise_chunk(chunks[index]["text"])]
        if non_noise_indices:
            paper_indices = non_noise_indices
        if paper_indices:
            query_embedding = model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)[0]
            query_embedding = _sanitize_query_embedding(query_embedding)
            paper_embeddings = embeddings[paper_indices]
            scores = np.einsum("ij,j->i", paper_embeddings, query_embedding, dtype=np.float32)
            scores = np.nan_to_num(scores, nan=-1.0, posinf=1.0, neginf=-1.0)
            lexical_scores = np.array(
                [_score_text(chunks[index]["text"], question_tokens) for index in paper_indices],
                dtype=np.float32,
            )
            hybrid_scores = scores + (0.03 * lexical_scores)
            sorted_local_indices = np.argsort(hybrid_scores)[::-1]

            retrieved = []
            seen_texts: set[str] = set()
            if wants_summary:
                for global_index in paper_indices:
                    chunk = chunks[global_index]
                    if chunk["section_id"] in {-1, "abstract"}:
                        text_key = re.sub(r"\s+", " ", chunk["text"]).strip()[:220]
                        seen_texts.add(text_key)
                        abstract_chunk = dict(chunk)
                        abstract_chunk["score"] = round(float(hybrid_scores[paper_indices.index(global_index)]) + 0.05, 4)
                        retrieved.append(abstract_chunk)
                        break
            for local_index in sorted_local_indices:
                global_index = paper_indices[int(local_index)]
                chunk = dict(chunks[global_index])
                text_key = re.sub(r"\s+", " ", chunk["text"]).strip()[:220]
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)
                chunk["score"] = round(float(hybrid_scores[int(local_index)]), 4)
                retrieved.append(chunk)
                if len(retrieved) >= top_k:
                    break
            return retrieved, "embedding"

    sections = _paper_sections_for_demo(paper_id)
    ranked = []
    for section in sections:
        score = _score_text(section["text"], question_tokens)
        if score <= 0:
            continue
        ranked.append(
            {
                "section_id": section["section_id"],
                "label": section["label"],
                "score": round(score, 4),
                "text": section["text"],
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return (ranked[:top_k] if ranked else sections[:top_k]), "lexical"


def ask_paper(paper_id: str, question: str, top_k: int = 3) -> dict:
    retrieved, retrieval_backend = _retrieve_demo_chunks(paper_id=paper_id, question=question, top_k=top_k)
    context = _join_demo_context(retrieved)
    generator = load_demo_generator()

    if generator is not None:
        try:
            answer = generator.generate(_demo_prompt(question, context))
            generation_backend = "together"
        except Exception:
            answer = _answer_from_retrieved_chunks(question, retrieved, strategy="Structured")
            generation_backend = "local-fallback"
    else:
        answer = _answer_from_retrieved_chunks(question, retrieved, strategy="Structured")
        generation_backend = "local-fallback"

    if not answer.strip():
        answer = "I could not find enough direct evidence inside the selected paper to answer that question confidently."

    return {
        "mode": f"{retrieval_backend}-retrieval + {generation_backend}-generation",
        "paper_id": paper_id,
        "question": question,
        "answer": answer,
        "retrieval_backend": retrieval_backend,
        "generation_backend": generation_backend,
        "retrieved_sections": [
            {
                "section_id": item["section_id"],
                "label": item["label"],
                "score": item["score"],
                "snippet": item["text"][:700],
            }
            for item in retrieved
        ],
    }


def build_dashboard_payload() -> dict:
    retrieval_rows = load_retrieval_rows()
    generation_rows = load_generation_summary()

    best_by_top_k = []
    for top_k in sorted({row["top_k"] for row in retrieval_rows}):
        candidates = [row for row in retrieval_rows if row["top_k"] == top_k]
        best_row = max(candidates, key=lambda row: row["hit_rate"])
        best_by_top_k.append(best_row)

    return {
        "chunk_stats": load_chunk_stats(),
        "retrieval_rows": retrieval_rows,
        "best_retrieval_by_top_k": best_by_top_k,
        "generation_summary": generation_rows,
        "integration_notes": {
            "current_demo_backend": _demo_backend_status(),
            "part3_rerun_needed": False,
            "part3_rerun_note": "Part 3 has been rerun against answers.json. Demo now uses embedding retrieval and upgrades to Together generation when TOGETHER_API_KEY is available.",
        },
    }
