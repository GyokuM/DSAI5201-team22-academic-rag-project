from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path


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


@lru_cache(maxsize=1)
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


def ask_paper(paper_id: str, question: str, top_k: int = 3) -> dict:
    sections = _paper_sections_for_demo(paper_id)
    question_tokens = set(_tokenize(question))
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
    retrieved = ranked[:top_k] if ranked else sections[:top_k]

    answer_fragments = []
    for index, section in enumerate(retrieved, start=1):
        for sentence in _pick_sentences(section["text"], question_tokens):
            answer_fragments.append(f"{sentence} [S{index}]")
        if len(answer_fragments) >= 3:
            break

    answer = " ".join(answer_fragments[:3]).strip()
    if not answer:
        answer = "I could not find enough direct evidence inside the selected paper to answer that question confidently."

    return {
        "mode": "lexical-grounded-summary",
        "paper_id": paper_id,
        "question": question,
        "answer": answer,
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
            "current_demo_backend": "Lexical retrieval fallback for local demo stability",
            "part3_rerun_needed": True,
            "part3_rerun_note": "Rerun evaluation against answers.json and export stable generation/evaluation results.",
        },
    }
