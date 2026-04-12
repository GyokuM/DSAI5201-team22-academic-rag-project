from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from bert_score import score as bert_score_fn
except Exception:  # pragma: no cover - optional dependency
    bert_score_fn = None

try:
    from together import Together
except Exception:  # pragma: no cover - optional dependency
    Together = None


WORD_PATTERN = re.compile(r"[A-Za-z0-9\-]{2,}")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
SOURCE_PATTERN = re.compile(r"\[Source\s+(\d+)\]")


@dataclass
class Paths:
    project_root: Path
    materials_root: Path
    part1_root: Path
    part2_root: Path
    part3_root: Path
    outputs_dir: Path
    queries_path: Path
    qrels_path: Path
    answers_path: Path
    chunk_file: Path
    embedding_cache: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean rerun for Part 3 generation and evaluation.")
    parser.add_argument("--chunking", default="section_aware", choices=["fixed", "recursive", "section_aware"])
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-limit", type=int, default=120)
    parser.add_argument("--generator-mode", default="lexical", choices=["lexical", "together"])
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-7B-Instruct-Turbo")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--faithfulness-mode", default="lexical", choices=["lexical", "together"])
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Paths:
    part3_root = Path(__file__).resolve().parent
    materials_root = part3_root.parent
    project_root = materials_root.parent
    part1_root = materials_root / "project_part1"
    part2_root = materials_root / "project_part2"

    chunk_file = part1_root / "data" / "processed" / f"chunks_{args.chunking}.jsonl"
    model_safe_name = args.embedding_model.replace("/", "_")
    embedding_cache = part2_root / f"chunks_{args.chunking}_{model_safe_name}_fp16.npy"
    outputs_dir = part3_root / "rerun_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    return Paths(
        project_root=project_root,
        materials_root=materials_root,
        part1_root=part1_root,
        part2_root=part2_root,
        part3_root=part3_root,
        outputs_dir=outputs_dir,
        queries_path=part1_root / "data" / "open_ragbench" / "pdf" / "arxiv" / "queries.json",
        qrels_path=part1_root / "data" / "open_ragbench" / "pdf" / "arxiv" / "qrels.json",
        answers_path=part1_root / "data" / "open_ragbench" / "pdf" / "arxiv" / "answers.json",
        chunk_file=chunk_file,
        embedding_cache=embedding_cache,
    )


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_query_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        return payload.get("query") or payload.get("text") or payload.get("question") or str(payload)
    return str(payload)


def clean_doc_id(doc_id: str) -> str:
    return doc_id.split("v")[0] if "v" in doc_id else doc_id


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_PATTERN.findall(text)]


def lexical_overlap(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)


def join_context(chunks: list[dict], max_chars: int = 650) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")[:max_chars]
        parts.append(f"[Source {index} | doc={chunk.get('doc_id')} sec={chunk.get('section_id')}]\n{text}")
    return "\n\n".join(parts)


def answer_from_context(question: str, chunks: list[dict], strategy: str) -> str:
    question_tokens = set(tokenize(question))
    candidates: list[tuple[float, str, int]] = []

    for source_index, chunk in enumerate(chunks, start=1):
        for sentence in SENTENCE_SPLIT.split(chunk.get("text", "")):
            sentence = sentence.strip()
            if len(sentence) < 45:
                continue
            score = lexical_overlap(sentence, question) + sum(token in sentence.lower() for token in question_tokens) * 0.05
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


def prompt_naive(question: str, context: str) -> str:
    return f"""Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""


def prompt_structured(question: str, context: str) -> str:
    return f"""You are an academic assistant. Answer the question strictly based on the provided context.

Rules:
- ONLY use information from the context below
- If the context does not contain enough information, say "Insufficient information"
- Cite sources using [Source N] format

Context:
{context}

Question: {question}

Answer:"""


def prompt_cot(question: str, context: str) -> str:
    return f"""You are an academic research assistant. Answer the question using ONLY the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Identify the relevant sources
2. Reason step by step using only supported facts
3. Provide a concise final answer with [Source N] citations
4. If context is insufficient, say "Insufficient information"

Answer:"""


PROMPT_BUILDERS = {
    "Naive": prompt_naive,
    "Structured": prompt_structured,
    "CoT": prompt_cot,
}


class TogetherGenerator:
    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        if Together is None:
            raise RuntimeError("The 'together' package is not installed.")
        api_key = os.getenv("TOGETHER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY is not set.")
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()


def compute_or_load_embeddings(
    chunk_texts: list[str],
    model: SentenceTransformer,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        print(f"Loading cached chunk embeddings from {cache_path.name}")
        embeddings = np.load(cache_path).astype(np.float32)
        return sanitize_embeddings(embeddings)

    print(f"Computing chunk embeddings for {len(chunk_texts)} chunks")
    embeddings = model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    embeddings = sanitize_embeddings(embeddings)
    np.save(cache_path, embeddings.astype(np.float16))
    return embeddings


def sanitize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.nan_to_num(embeddings.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return embeddings / norms


def sanitize_query_embedding(query_embedding: np.ndarray) -> np.ndarray:
    query_embedding = np.nan_to_num(query_embedding.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(query_embedding))
    if norm <= 1e-12:
        return query_embedding
    return query_embedding / norm


def retrieve_chunks(
    query: str,
    chunk_embeddings: np.ndarray,
    model: SentenceTransformer,
    chunks: list[dict],
    top_k: int,
) -> list[dict]:
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)[0]
    query_embedding = sanitize_query_embedding(query_embedding)
    scores = np.einsum("ij,j->i", chunk_embeddings, query_embedding, dtype=np.float32)
    scores = np.nan_to_num(scores, nan=-1.0, posinf=1.0, neginf=-1.0)
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    sorted_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    results: list[dict] = []
    for index in sorted_indices:
        chunk = dict(chunks[int(index)])
        chunk["retrieval_score"] = float(scores[int(index)])
        results.append(chunk)
    return results


def lexical_faithfulness(answer: str, context: str) -> int:
    overlap = lexical_overlap(answer, context)
    if overlap >= 0.85:
        return 5
    if overlap >= 0.65:
        return 4
    if overlap >= 0.45:
        return 3
    if overlap >= 0.25:
        return 2
    return 1


def llm_faithfulness(generator: TogetherGenerator, question: str, context: str, answer: str) -> int:
    prompt = f"""You are an evaluation judge. Rate the faithfulness of the answer to the given context on a scale of 1 to 5.

Question:
{question}

Context:
{context[:1600]}

Answer:
{answer[:900]}

Return only a JSON object like {{"score": 4}}"""
    response = generator.generate(prompt)
    match = re.search(r"\{.*?\}", response, re.DOTALL)
    if not match:
        return lexical_faithfulness(answer, context)
    try:
        payload = json.loads(match.group())
        score = int(payload.get("score", 3))
        return max(1, min(5, score))
    except Exception:
        return lexical_faithfulness(answer, context)


def classify_error(rouge_l: float, faithfulness: int, answer: str, reference_answer: str, context: str) -> str:
    answer = answer or ""
    if "insufficient information" in answer.lower():
        return "Retrieval Miss"
    if faithfulness <= 2:
        return "Hallucination"
    if len(answer.split()) < 8:
        return "Too Generic"
    if rouge_l < 0.12 and lexical_overlap(reference_answer, context) > 0.15:
        return "Incomplete Answer"
    return "OK"


def summarize_results(results_df: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    rows = []
    for strategy in ["Naive", "Structured", "CoT"]:
        subset = results_df[results_df["strategy"] == strategy]
        rouge_vals = subset["rouge_l"].dropna().tolist()
        bert_vals = subset["bert_score_f1"].dropna().tolist()
        faith_vals = subset["faithfulness"].dropna().tolist()
        lat_vals = subset["latency_s"].dropna().tolist()

        rows.append(
            {
                "Strategy": strategy,
                "ROUGE-L_mean": mean(rouge_vals) if rouge_vals else math.nan,
                "ROUGE-L_std": pstdev(rouge_vals) if len(rouge_vals) > 1 else 0.0,
                "BERTScore_F1_mean": mean(bert_vals) if bert_vals else math.nan,
                "BERTScore_F1_std": pstdev(bert_vals) if len(bert_vals) > 1 else 0.0,
                "Faithfulness_mean": mean(faith_vals) if faith_vals else math.nan,
                "Faithfulness_std": pstdev(faith_vals) if len(faith_vals) > 1 else 0.0,
                "Latency_s_mean": mean(lat_vals) if lat_vals else math.nan,
                "Latency_s_std": pstdev(lat_vals) if len(lat_vals) > 1 else 0.0,
                "Hallucination_count": int((subset["error_type"] == "Hallucination").sum()),
                "Too_Generic_count": int((subset["error_type"] == "Too Generic").sum()),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_path, index=False, float_format="%.4f")
    return summary_df


def save_visualizations(results_df: pd.DataFrame, summary_df: pd.DataFrame, output_root: Path) -> None:
    strategy_order = ["Naive", "Structured", "CoT"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics = [
        ("rouge_l", "ROUGE-L", "#60a5fa"),
        ("bert_score_f1", "BERTScore F1", "#34d399"),
        ("faithfulness", "Faithfulness", "#fb923c"),
        ("latency_s", "Latency (s)", "#f87171"),
    ]

    for ax, (column, title, color) in zip(axes, metrics):
        data = [results_df[results_df["strategy"] == strategy][column].dropna().tolist() for strategy in strategy_order]
        ax.boxplot(data, tick_labels=strategy_order, patch_artist=True, boxprops={"facecolor": color, "alpha": 0.55})
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_root / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    error_order = ["OK", "Hallucination", "Retrieval Miss", "Incomplete Answer", "Too Generic"]
    color_map = {
        "OK": "#16a34a",
        "Hallucination": "#dc2626",
        "Retrieval Miss": "#ea580c",
        "Incomplete Answer": "#2563eb",
        "Too Generic": "#7c3aed",
    }
    for ax, strategy in zip(axes, strategy_order):
        subset = results_df[results_df["strategy"] == strategy]
        counts = subset["error_type"].value_counts()
        labels = [label for label in error_order if label in counts]
        sizes = [counts[label] for label in labels]
        colors = [color_map[label] for label in labels]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title(strategy)

    plt.tight_layout()
    plt.savefig(output_root / "error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    radar_metrics = ["ROUGE-L_mean", "BERTScore_F1_mean", "Faithfulness_mean"]
    radar_values = summary_df[radar_metrics].fillna(0.0).copy()
    if not radar_values.empty:
        max_faith = max(radar_values["Faithfulness_mean"].max(), 1.0)
        radar_values["Faithfulness_mean"] = radar_values["Faithfulness_mean"] / max_faith

    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    for _, row in summary_df.iterrows():
        values = [
            float(row["ROUGE-L_mean"] or 0),
            float(row["BERTScore_F1_mean"] or 0),
            float((row["Faithfulness_mean"] or 0) / max(summary_df["Faithfulness_mean"].max(), 1.0)),
        ]
        values += values[:1]
        ax.plot(angles, values, label=row["Strategy"])
        ax.fill(angles, values, alpha=0.08)
    ax.set_thetagrids(np.degrees(angles[:-1]), ["ROUGE-L", "BERTScore", "Faithfulness"])
    ax.set_title("Prompt Strategy Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.savefig(output_root / "radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    ensure_exists(paths.chunk_file, "Chunk file")
    ensure_exists(paths.queries_path, "queries.json")
    ensure_exists(paths.qrels_path, "qrels.json")
    ensure_exists(paths.answers_path, "answers.json")

    print(f"Project root: {paths.project_root}")
    print(f"Chunk file: {paths.chunk_file}")
    print(f"Embedding cache: {paths.embedding_cache}")
    print(f"Generator mode: {args.generator_mode}")

    chunks = load_jsonl(paths.chunk_file)
    chunk_texts = [chunk["text"] for chunk in chunks]
    queries_raw = load_json(paths.queries_path)
    qrels_raw = load_json(paths.qrels_path)
    answers_raw = load_json(paths.answers_path)

    query_ids = list(queries_raw.keys())[: args.query_limit]
    query_texts = [extract_query_text(queries_raw[query_id]) for query_id in query_ids]

    model = SentenceTransformer(args.embedding_model)
    chunk_embeddings = compute_or_load_embeddings(chunk_texts, model, paths.embedding_cache)

    generator = None
    if args.generator_mode == "together":
        generator = TogetherGenerator(
            model_name=args.generator_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    generation_rows: list[dict] = []

    for query_id, question in tqdm(zip(query_ids, query_texts), total=len(query_ids), desc="Generating answers"):
        retrieved_chunks = retrieve_chunks(
            query=question,
            chunk_embeddings=chunk_embeddings,
            model=model,
            chunks=chunks,
            top_k=args.top_k,
        )
        context = join_context(retrieved_chunks)
        reference_answer = answers_raw.get(query_id, "")

        for strategy, prompt_builder in PROMPT_BUILDERS.items():
            prompt = prompt_builder(question, context)
            start = time.time()

            if generator is not None:
                answer = generator.generate(prompt)
            else:
                answer = answer_from_context(question, retrieved_chunks, strategy)

            latency_s = round(time.time() - start, 3)
            generation_rows.append(
                {
                    "query_id": query_id,
                    "question": question,
                    "strategy": strategy,
                    "prompt": prompt,
                    "answer": answer,
                    "reference_answer": reference_answer,
                    "context": context,
                    "num_chunks": len(retrieved_chunks),
                    "retrieved_docs": [chunk.get("doc_id") for chunk in retrieved_chunks],
                    "retrieved_section_ids": [chunk.get("section_id") for chunk in retrieved_chunks],
                    "latency_s": latency_s,
                }
            )

    generation_path = paths.outputs_dir / "generation_results.jsonl"
    dump_jsonl(generation_path, generation_rows)
    print(f"Saved generation results to {generation_path}")

    results_df = pd.DataFrame(generation_rows)
    results_df["rouge_l"] = results_df.apply(
        lambda row: scorer.score(str(row["reference_answer"]), str(row["answer"]))["rougeL"].fmeasure
        if str(row["reference_answer"]).strip()
        else math.nan,
        axis=1,
    )

    if bert_score_fn is not None:
        valid_mask = results_df["reference_answer"].astype(str).str.strip().astype(bool)
        if valid_mask.any():
            _, _, f1 = bert_score_fn(
                results_df.loc[valid_mask, "answer"].tolist(),
                results_df.loc[valid_mask, "reference_answer"].tolist(),
                lang="en",
                verbose=False,
            )
            results_df.loc[valid_mask, "bert_score_f1"] = f1.numpy()
        else:
            results_df["bert_score_f1"] = math.nan
    else:
        print("bert_score not installed; BERTScore will be NaN")
        results_df["bert_score_f1"] = math.nan

    if args.faithfulness_mode == "together" and generator is not None:
        results_df["faithfulness"] = [
            llm_faithfulness(generator, row.question, row.context, row.answer)
            for row in tqdm(results_df.itertuples(index=False), total=len(results_df), desc="Faithfulness")
        ]
    else:
        results_df["faithfulness"] = results_df.apply(
            lambda row: lexical_faithfulness(str(row["answer"]), str(row["context"])),
            axis=1,
        )

    results_df["error_type"] = results_df.apply(
        lambda row: classify_error(
            rouge_l=float(row["rouge_l"]) if not pd.isna(row["rouge_l"]) else 0.0,
            faithfulness=int(row["faithfulness"]),
            answer=str(row["answer"]),
            reference_answer=str(row["reference_answer"]),
            context=str(row["context"]),
        ),
        axis=1,
    )

    evaluation_path = paths.outputs_dir / "evaluation_results.jsonl"
    dump_jsonl(evaluation_path, results_df.to_dict(orient="records"))
    print(f"Saved evaluation results to {evaluation_path}")

    summary_path = paths.part3_root / "evaluation_summary.csv"
    summary_df = summarize_results(results_df, summary_path)
    print(f"Saved summary CSV to {summary_path}")

    save_visualizations(results_df, summary_df, paths.part3_root)
    print(f"Updated plot files under {paths.part3_root}")

    details_path = paths.outputs_dir / "run_config.json"
    with open(details_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "chunking": args.chunking,
                "embedding_model": args.embedding_model,
                "top_k": args.top_k,
                "query_limit": args.query_limit,
                "generator_mode": args.generator_mode,
                "generator_model": args.generator_model,
                "faithfulness_mode": args.faithfulness_mode,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved run config to {details_path}")

    print("\nSummary preview:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
