"""
一键运行 JSON corpus 处理脚本
输出路径：project/data/processed/
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Iterable

# ── 路径配置（绝对化，避免 cwd 歧义）──────────────────────────────────────
THIS_FILE   = Path(__file__).resolve()           # project/code/run_processing.py
CODE_DIR    = THIS_FILE.parent                   # project/code/
PROJECT_DIR = CODE_DIR.parent                    # project/
DATA_ROOT   = PROJECT_DIR / "data" / "open_ragbench" / "pdf" / "arxiv"
CORPUS_DIR  = DATA_ROOT / "corpus"
OUTPUT_DIR  = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_PATH        = OUTPUT_DIR / "cleaned_sections.jsonl"
FIXED_PATH          = OUTPUT_DIR / "chunks_fixed.jsonl"
RECURSIVE_PATH      = OUTPUT_DIR / "chunks_recursive.jsonl"
SECTION_AWARE_PATH  = OUTPUT_DIR / "chunks_section_aware.jsonl"
STATS_PATH          = OUTPUT_DIR / "chunk_stats.json"

print(f"PROJECT_DIR : {PROJECT_DIR}")
print(f"CORPUS_DIR  : {CORPUS_DIR}")
print(f"OUTPUT_DIR  : {OUTPUT_DIR}")
assert DATA_ROOT.exists(),  f"DATA_ROOT not found: {DATA_ROOT}"
assert CORPUS_DIR.exists(), f"CORPUS_DIR not found: {CORPUS_DIR}"

# ── 正则（正确转义，无双重反斜杠）──────────────────────────────────────────
REFERENCE_HEADING_PATTERN = re.compile(
    r"^(references|bibliography|acknowledg(e)?ments?|appendix)$", re.IGNORECASE
)
ARXIV_NOISE_PATTERN  = re.compile(r"arxiv:\s*\d{4}\.\d{4,5}(v\d+)?", re.IGNORECASE)
PAGE_NUMBER_PATTERN  = re.compile(r"^\s*\d+\s*$")
WHITESPACE_PATTERN   = re.compile(r"[ \t]+")
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
INLINE_NEWLINE_PATTERN = re.compile(r"(?<!\n)\n(?!\n)")
HYPHEN_BREAK_PATTERN  = re.compile(r"(\w+)-\n(\w+)")


# ── 文本清洗函数 ─────────────────────────────────────────────────────────────
def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    text = MULTI_NEWLINE_PATTERN.sub("\n\n", text)
    return text.strip()

def fix_hyphenation(text: str) -> str:
    return HYPHEN_BREAK_PATTERN.sub(r"\1\2", text)

def merge_inline_newlines(text: str) -> str:
    return INLINE_NEWLINE_PATTERN.sub(" ", text)

def remove_noise_lines(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if PAGE_NUMBER_PATTERN.fullmatch(stripped):
            continue
        if ARXIV_NOISE_PATTERN.search(stripped):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def is_reference_heading(text: str) -> bool:
    candidate = normalize_whitespace(text).strip(":. ")
    return bool(REFERENCE_HEADING_PATTERN.fullmatch(candidate))

def clean_text(text: str, merge_lines: bool = False) -> str:
    text = fix_hyphenation(text)
    text = remove_noise_lines(text)
    if merge_lines:
        text = merge_inline_newlines(text)
    text = normalize_whitespace(text)
    return text

def first_reference_index(section_texts: Iterable[str]) -> int | None:
    for idx, text in enumerate(section_texts):
        if is_reference_heading(text):
            return idx
    return None


# ── Chunking 策略 ────────────────────────────────────────────────────────────
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False
    RecursiveCharacterTextSplitter = None  # type: ignore

class FixedChunker:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

class RecursiveChunker:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        if _HAS_LANGCHAIN:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        else:
            self._splitter = None

    def split(self, text: str) -> list[str]:
        if self._splitter is None:
            return FixedChunker(self.chunk_size, self.chunk_overlap).split(text)
        return self._splitter.split_text(text)

class SectionAwareChunker:
    def __init__(self, max_section_chars: int = 1600,
                 chunk_size: int = 1200, chunk_overlap: int = 150):
        self.max_section_chars  = max_section_chars
        self.recursive_chunker  = RecursiveChunker(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)

    def split(self, text: str) -> list[str]:
        if len(text) <= self.max_section_chars:
            return [text]
        return self.recursive_chunker.split(text)

def build_chunk_records(doc_id, section_id, title, strategy, chunks, extra_metadata=None):
    extra_metadata = extra_metadata or {}
    return [
        {
            "chunk_id":    f"{doc_id}_sec{section_id}_chunk{i}_{strategy}",
            "doc_id":      doc_id,
            "section_id":  section_id,
            "chunk_index": i,
            "strategy":    strategy,
            "title":       title,
            "text":        chunk,
            "metadata":    extra_metadata,
        }
        for i, chunk in enumerate(chunks)
    ]

fixed_chunker        = FixedChunker()
recursive_chunker    = RecursiveChunker()
section_aware_chunker = SectionAwareChunker()


# ── 读取 & 清洗 ─────────────────────────────────────────────────────────────
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_jsonl(path: Path, records: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def extract_cleaned_sections(doc: dict) -> list[dict]:
    title    = doc.get("title", "")
    doc_id   = doc.get("id", "")
    sections = doc.get("sections", [])

    section_texts = [s.get("text", "") for s in sections]
    ref_idx = first_reference_index(section_texts)
    if ref_idx is not None:
        sections = sections[:ref_idx]

    cleaned: list[dict] = []

    abstract = clean_text(doc.get("abstract", ""), merge_lines=True)
    if len(abstract) >= 50:
        cleaned.append({
            "doc_id":        doc_id,
            "section_id":    -1,
            "section_label": "abstract",
            "title":         title,
            "text":          abstract,
            "categories":    doc.get("categories", []),
            "source":        "open_ragbench_json",
        })

    for idx, section in enumerate(sections):
        text = clean_text(section.get("text", ""), merge_lines=True)
        if len(text) < 50:
            continue
        cleaned.append({
            "doc_id":        doc_id,
            "section_id":    idx,
            "section_label": "section",
            "title":         title,
            "text":          text,
            "categories":    doc.get("categories", []),
            "source":        "open_ragbench_json",
        })
    return cleaned


# ── 主流程 ──────────────────────────────────────────────────────────────────
def main():
    corpus_files = sorted(CORPUS_DIR.glob("*.json"))
    print(f"Found {len(corpus_files)} corpus JSON files.")

    # Step 1: 清洗 sections
    cleaned_sections: list[dict] = []
    for fp in corpus_files:
        doc = load_json(fp)
        cleaned_sections.extend(extract_cleaned_sections(doc))
    dump_jsonl(CLEANED_PATH, cleaned_sections)
    print(f"[OK] cleaned_sections → {CLEANED_PATH}  ({len(cleaned_sections)} records)")

    # Step 2: 三种 chunking
    def build_all_chunks(strategy: str) -> list[dict]:
        all_chunks: list[dict] = []
        for sec in cleaned_sections:
            text = sec["text"]
            if strategy == "fixed":
                splits = fixed_chunker.split(text)
            elif strategy == "recursive":
                splits = recursive_chunker.split(text)
            else:
                splits = section_aware_chunker.split(text)
            records = build_chunk_records(
                doc_id=sec["doc_id"],
                section_id=sec["section_id"],
                title=sec["title"],
                strategy=strategy,
                chunks=splits,
                extra_metadata={
                    "section_label": sec["section_label"],
                    "categories":    sec["categories"],
                    "source":        sec["source"],
                },
            )
            all_chunks.extend(records)
        return all_chunks

    fixed_chunks        = build_all_chunks("fixed")
    recursive_chunks    = build_all_chunks("recursive")
    section_aware_chunks = build_all_chunks("section_aware")

    dump_jsonl(FIXED_PATH,        fixed_chunks)
    dump_jsonl(RECURSIVE_PATH,    recursive_chunks)
    dump_jsonl(SECTION_AWARE_PATH, section_aware_chunks)
    print(f"[OK] chunks_fixed        → {FIXED_PATH}  ({len(fixed_chunks)} records)")
    print(f"[OK] chunks_recursive    → {RECURSIVE_PATH}  ({len(recursive_chunks)} records)")
    print(f"[OK] chunks_section_aware→ {SECTION_AWARE_PATH}  ({len(section_aware_chunks)} records)")

    # Step 3: 统计
    def summarize(chunks):
        lengths = [len(c["text"]) for c in chunks]
        return {
            "num_chunks":      len(chunks),
            "avg_chunk_chars": round(mean(lengths), 2) if lengths else 0,
            "max_chunk_chars": max(lengths) if lengths else 0,
            "min_chunk_chars": min(lengths) if lengths else 0,
        }

    stats = {
        "cleaned_sections": len(cleaned_sections),
        "fixed":            summarize(fixed_chunks),
        "recursive":        summarize(recursive_chunks),
        "section_aware":    summarize(section_aware_chunks),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[OK] chunk_stats         → {STATS_PATH}")
    print()
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
