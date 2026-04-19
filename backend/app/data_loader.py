from __future__ import annotations

import csv
import io
import json
import os
import re
import uuid
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

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


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
DEMO_CHUNKING_OPTIONS = ("fixed", "recursive", "section_aware")
DEMO_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEMO_GENERATOR_MODEL = os.getenv("ACADEMIC_RAG_GENERATOR_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo")
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_COMPAT_MODEL = os.getenv("OPENAI_MODEL", os.getenv("ACADEMIC_RAG_OPENAI_MODEL", "gemini-2.5-pro")).strip()

TOP_K_FILES = {
    3: PART2_DIR / "top_K_3.csv",
    5: PART2_DIR / "top_K_5.csv",
    10: PART2_DIR / "top_K_10.csv",
    15: PART2_DIR / "top_K_15.csv",
}

WORD_PATTERN = re.compile(r"[A-Za-z0-9\-]{2,}")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
UPLOADED_DOCS: dict[str, dict] = {}


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


def _demo_chunk_path(chunking: str) -> Path:
    return PART1_DIR / "data" / "processed" / f"chunks_{chunking}.jsonl"


def _demo_embedding_cache_path(chunking: str) -> Path:
    return PART2_DIR / f"chunks_{chunking}_{DEMO_EMBEDDING_MODEL.replace('/', '_')}_fp16.npy"


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


def _chunk_uploaded_page_text(page_texts: list[tuple[int, str]], max_chars: int = 1200) -> list[dict]:
    chunks: list[dict] = []
    for page_number, page_text in page_texts:
        cleaned = re.sub(r"\n{2,}", "\n\n", page_text).strip()
        if len(cleaned) < 80:
            continue
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
        buffer = ""
        chunk_index = 0
        for paragraph in paragraphs:
            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
            if len(candidate) <= max_chars:
                buffer = candidate
                continue
            if buffer:
                chunks.append(
                    {
                        "chunk_id": f"page-{page_number}-chunk-{chunk_index}",
                        "doc_id": f"upload-page-{page_number}",
                        "section_id": page_number,
                        "label": f"Page {page_number}",
                        "text": buffer,
                    }
                )
                chunk_index += 1
            buffer = paragraph[:max_chars]
        if buffer:
            chunks.append(
                {
                    "chunk_id": f"page-{page_number}-chunk-{chunk_index}",
                    "doc_id": f"upload-page-{page_number}",
                    "section_id": page_number,
                    "label": f"Page {page_number}",
                    "text": buffer,
                }
            )
    return chunks


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


def _rank_chunks_for_question(
    chunks: list[dict],
    question: str,
    top_k: int,
    embeddings: np.ndarray | None = None,
    use_summary_bias: bool = False,
) -> tuple[list[dict], str]:
    question_tokens = set(_tokenize(question))
    wants_summary = _is_summary_question(question)

    if (
        np is not None
        and embeddings is not None
        and len(chunks) == len(embeddings)
        and load_demo_embedder() is not None
    ):
        model = load_demo_embedder()
        query_embedding = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]
        query_embedding = _sanitize_query_embedding(query_embedding)
        scores = np.einsum("ij,j->i", embeddings, query_embedding, dtype=np.float32)
        scores = np.nan_to_num(scores, nan=-1.0, posinf=1.0, neginf=-1.0)
        lexical_scores = np.array([_score_text(chunk["text"], question_tokens) for chunk in chunks], dtype=np.float32)
        hybrid_scores = scores + (0.03 * lexical_scores)
        sorted_indices = np.argsort(hybrid_scores)[::-1]

        retrieved = []
        seen_texts: set[str] = set()
        if wants_summary and use_summary_bias:
            for index, chunk in enumerate(chunks):
                if chunk.get("section_id") not in {-1, "abstract"}:
                    continue
                text_key = re.sub(r"\s+", " ", chunk["text"]).strip()[:220]
                seen_texts.add(text_key)
                abstract_chunk = dict(chunk)
                abstract_chunk["score"] = round(float(hybrid_scores[index]) + 0.05, 4)
                retrieved.append(abstract_chunk)
                break

        for index in sorted_indices:
            chunk = dict(chunks[int(index)])
            text_key = re.sub(r"\s+", " ", chunk["text"]).strip()[:220]
            if text_key in seen_texts:
                continue
            if _looks_like_noise_chunk(chunk["text"]):
                continue
            seen_texts.add(text_key)
            chunk["score"] = round(float(hybrid_scores[int(index)]), 4)
            retrieved.append(chunk)
            if len(retrieved) >= top_k:
                break
        if retrieved:
            return retrieved, "embedding"

    ranked = []
    for chunk in chunks:
        score = _score_text(chunk["text"], question_tokens)
        if score <= 0:
            continue
        ranked.append({**chunk, "score": round(score, 4)})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return (ranked[:top_k] if ranked else chunks[:top_k]), "lexical"


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


class OpenAICompatibleGenerator:
    def __init__(self, model_name: str, api_key: str, base_url: str):
        if OpenAI is None:
            raise RuntimeError("The 'openai' package is not installed.")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a grounded academic research assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=220,
            stream=False,
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


@lru_cache(maxsize=len(DEMO_CHUNKING_OPTIONS))
def load_demo_chunks(chunking: str = "section_aware") -> list[dict]:
    chunk_path = _demo_chunk_path(chunking)
    if not chunk_path.exists():
        return []
    rows = _load_jsonl(chunk_path)
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


@lru_cache(maxsize=len(DEMO_CHUNKING_OPTIONS))
def load_demo_chunk_embeddings(chunking: str = "section_aware") -> np.ndarray | None:
    if np is None:
        return None
    cache_path = _demo_embedding_cache_path(chunking)
    if cache_path.exists():
        return _sanitize_embeddings(np.load(cache_path).astype(np.float32))

    model = load_demo_embedder()
    chunks = load_demo_chunks(chunking)
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
    np.save(cache_path, embeddings.astype(np.float16))
    return embeddings


@lru_cache(maxsize=1)
def load_demo_generator() -> OpenAICompatibleGenerator | DemoGenerator | None:
    if OPENAI_COMPAT_API_KEY and OPENAI_COMPAT_BASE_URL:
        try:
            return OpenAICompatibleGenerator(
                model_name=OPENAI_COMPAT_MODEL,
                api_key=OPENAI_COMPAT_API_KEY,
                base_url=OPENAI_COMPAT_BASE_URL,
            )
        except Exception:
            pass
    try:
        return DemoGenerator(DEMO_GENERATOR_MODEL)
    except Exception:
        return None


def _demo_backend_status(chunking: str = "section_aware") -> str:
    retrieval = "embedding retrieval" if load_demo_embedder() is not None else "lexical retrieval fallback"
    if OPENAI_COMPAT_API_KEY and OPENAI_COMPAT_BASE_URL and load_demo_generator() is not None:
        generation = f"OpenAI-compatible model generation ({OPENAI_COMPAT_MODEL})"
    elif load_demo_generator() is not None:
        generation = "Together model-backed generation"
    else:
        generation = "local grounded fallback generation"
    return f"{chunking} chunking + {retrieval} + {generation}"


def upload_pdf_document(filename: str, file_bytes: bytes) -> dict:
    if PdfReader is None:
        raise RuntimeError("PDF upload requires the 'pypdf' package.")

    reader = PdfReader(io.BytesIO(file_bytes))
    page_texts: list[tuple[int, str]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            page_texts.append((page_index, text))

    if not page_texts:
        raise ValueError("No extractable text was found in this PDF. Please upload a text-based PDF.")

    chunks = _chunk_uploaded_page_text(page_texts)
    if not chunks:
        raise ValueError("The uploaded PDF did not produce enough readable text for retrieval.")

    embedder = load_demo_embedder()
    embeddings = None
    if np is not None and embedder is not None:
        embeddings = embedder.encode(
            [chunk["text"] for chunk in chunks],
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        embeddings = _sanitize_embeddings(embeddings)

    upload_id = uuid.uuid4().hex[:12]
    title = Path(filename).stem or "Uploaded PDF"
    preview = " ".join(page_texts[0][1].split())[:320]
    UPLOADED_DOCS[upload_id] = {
        "id": upload_id,
        "title": title,
        "filename": filename,
        "page_count": len(reader.pages),
        "chunks": chunks,
        "embeddings": embeddings,
        "preview": preview,
    }
    return {
        "id": upload_id,
        "title": title,
        "filename": filename,
        "page_count": len(reader.pages),
        "chunk_count": len(chunks),
        "preview": preview,
    }


def get_uploaded_document(upload_id: str) -> dict:
    if upload_id not in UPLOADED_DOCS:
        raise FileNotFoundError(f"Uploaded PDF not found: {upload_id}")
    return UPLOADED_DOCS[upload_id]


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


def _retrieve_demo_chunks(paper_id: str, question: str, top_k: int, chunking: str = "section_aware") -> tuple[list[dict], str]:
    paper_base = _clean_doc_id(paper_id)
    chunks = load_demo_chunks(chunking)
    embeddings = load_demo_chunk_embeddings(chunking)

    paper_chunks = [
        chunk
        for chunk in chunks
        if _clean_doc_id(str(chunk.get("doc_id", ""))) == paper_base and len((chunk.get("text") or "").strip()) >= 40
    ]
    paper_embeddings = None
    if np is not None and embeddings is not None and len(chunks) == len(embeddings):
        paper_indices = [
            index
            for index, chunk in enumerate(chunks)
            if _clean_doc_id(str(chunk.get("doc_id", ""))) == paper_base and len((chunk.get("text") or "").strip()) >= 40
        ]
        if paper_indices:
            paper_embeddings = embeddings[paper_indices]
    if paper_chunks:
        return _rank_chunks_for_question(
            paper_chunks,
            question=question,
            top_k=top_k,
            embeddings=paper_embeddings,
            use_summary_bias=True,
        )

    sections = _paper_sections_for_demo(paper_id)
    return _rank_chunks_for_question(sections, question=question, top_k=top_k, embeddings=None, use_summary_bias=True)


def ask_paper(paper_id: str, question: str, top_k: int = 3, chunking: str = "section_aware") -> dict:
    if chunking not in DEMO_CHUNKING_OPTIONS:
        chunking = "section_aware"
    retrieved, retrieval_backend = _retrieve_demo_chunks(
        paper_id=paper_id,
        question=question,
        top_k=top_k,
        chunking=chunking,
    )
    context = _join_demo_context(retrieved)
    generator = load_demo_generator()

    if generator is not None:
        try:
            answer = generator.generate(_demo_prompt(question, context))
            generation_backend = "openai-compatible" if isinstance(generator, OpenAICompatibleGenerator) else "together"
        except Exception:
            answer = _answer_from_retrieved_chunks(question, retrieved, strategy="Structured")
            generation_backend = "local-fallback"
    else:
        answer = _answer_from_retrieved_chunks(question, retrieved, strategy="Structured")
        generation_backend = "local-fallback"

    if not answer.strip():
        answer = "I could not find enough direct evidence inside the selected paper to answer that question confidently."

    return {
        "mode": f"{chunking}-chunking + {retrieval_backend}-retrieval + {generation_backend}-generation",
        "paper_id": paper_id,
        "question": question,
        "chunking": chunking,
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


def ask_uploaded_document(upload_id: str, question: str, top_k: int = 3) -> dict:
    uploaded = get_uploaded_document(upload_id)
    retrieved, retrieval_backend = _rank_chunks_for_question(
        uploaded["chunks"],
        question=question,
        top_k=top_k,
        embeddings=uploaded.get("embeddings"),
        use_summary_bias=False,
    )
    context = _join_demo_context(retrieved)
    generator = load_demo_generator()

    if generator is not None:
        try:
            answer = generator.generate(_demo_prompt(question, context))
            generation_backend = "openai-compatible" if isinstance(generator, OpenAICompatibleGenerator) else "together"
        except Exception:
            answer = _answer_from_retrieved_chunks(question, retrieved, strategy="Structured")
            generation_backend = "local-fallback"
    else:
        answer = _answer_from_retrieved_chunks(question, retrieved, strategy="Structured")
        generation_backend = "local-fallback"

    return {
        "mode": f"uploaded-pdf + {retrieval_backend}-retrieval + {generation_backend}-generation",
        "upload_id": upload_id,
        "title": uploaded["title"],
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
        "demo_chunking_options": list(DEMO_CHUNKING_OPTIONS),
        "retrieval_rows": retrieval_rows,
        "best_retrieval_by_top_k": best_by_top_k,
        "generation_summary": generation_rows,
        "integration_notes": {
            "current_demo_backend": _demo_backend_status(),
            "part3_rerun_needed": False,
            "part3_rerun_note": "Part 3 has been rerun against answers.json. Demo now uses embedding retrieval and upgrades to OpenAI-compatible or Together generation when API credentials are available.",
        },
    }
