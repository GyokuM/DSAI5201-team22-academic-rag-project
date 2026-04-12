"""
OCR 处理脚本：下载 50 篇 raw PDF，提取 markdown，切分 sections，与 JSON corpus 对比
输出路径：
  project/data/rawpdf/raw_pdfs/          ← 原始 PDF
  project/data/rawpdf/ocr_processed/     ← 每篇 OCR 解析 JSON
  project/data/rawpdf/comparison/        ← 每篇对比结果 + summary.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import requests
from tqdm import tqdm

# ── 路径配置（绝对化）─────────────────────────────────────────────────────────
THIS_FILE   = Path(__file__).resolve()
CODE_DIR    = THIS_FILE.parent
PROJECT_DIR = CODE_DIR.parent
DATA_ROOT   = PROJECT_DIR / "data" / "open_ragbench" / "pdf" / "arxiv"
PDF_URLS_PATH   = DATA_ROOT / "pdf_urls.json"
JSON_CORPUS_DIR = DATA_ROOT / "corpus"

RAWPDF_DIR         = PROJECT_DIR / "data" / "rawpdf"
RAW_PDF_DIR        = RAWPDF_DIR / "raw_pdfs"
OCR_OUTPUT_DIR     = RAWPDF_DIR / "ocr_processed"
COMPARE_OUTPUT_DIR = RAWPDF_DIR / "comparison"

for d in [RAW_PDF_DIR, OCR_OUTPUT_DIR, COMPARE_OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"PROJECT_DIR    : {PROJECT_DIR}")
print(f"PDF_URLS_PATH  : {PDF_URLS_PATH}")
print(f"OCR_OUTPUT_DIR : {OCR_OUTPUT_DIR}")

assert DATA_ROOT.exists(),       f"DATA_ROOT not found: {DATA_ROOT}"
assert PDF_URLS_PATH.exists(),   f"PDF_URLS_PATH not found: {PDF_URLS_PATH}"
assert JSON_CORPUS_DIR.exists(), f"JSON_CORPUS_DIR not found: {JSON_CORPUS_DIR}"

# ── 正则 ──────────────────────────────────────────────────────────────────────
ARXIV_NOISE_PATTERN    = re.compile(r"arxiv:\s*\d{4}\.\d{4,5}(v\d+)?", re.IGNORECASE)
PAGE_NUMBER_PATTERN    = re.compile(r"^\s*\d+\s*$")
WHITESPACE_PATTERN     = re.compile(r"[ \t]+")
MULTI_NEWLINE_PATTERN  = re.compile(r"\n{3,}")
INLINE_NEWLINE_PATTERN = re.compile(r"(?<!\n)\n(?!\n)")
HYPHEN_BREAK_PATTERN   = re.compile(r"(\w+)-\n(\w+)")


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


def clean_text(text: str, merge_lines: bool = False) -> str:
    text = fix_hyphenation(text)
    text = remove_noise_lines(text)
    if merge_lines:
        text = merge_inline_newlines(text)
    text = normalize_whitespace(text)
    return text


# ── PDF 下载 ──────────────────────────────────────────────────────────────────
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_pdf(paper_id: str, url: str) -> Path:
    out_path = RAW_PDF_DIR / f"{paper_id}.pdf"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    return out_path


# ── OCR 提取 ──────────────────────────────────────────────────────────────────
def extract_markdown(pdf_path: Path) -> str:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=False,
        write_images=False,
        force_text=False,
    )


def markdown_to_sections(md_text: str) -> list[dict]:
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(md_text))

    if not matches:
        cleaned = clean_text(md_text, merge_lines=False)
        return [{"heading": "full_document", "text": cleaned}] if cleaned else []

    sections = []
    for idx, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(md_text)
        body = md_text[start:end].strip()
        cleaned = clean_text(body, merge_lines=False)
        if len(cleaned) < 50:
            continue
        sections.append({"heading": heading, "text": cleaned})
    return sections


def parse_pdf_to_json(pdf_path: Path, paper_id: str) -> dict:
    out_path = OCR_OUTPUT_DIR / f"{paper_id}.json"
    if out_path.exists():
        return load_json(out_path)

    markdown = extract_markdown(pdf_path)
    sections = markdown_to_sections(markdown)
    parsed = {
        "id": paper_id,
        "source": "raw_pdf_pymupdf4llm_markdown",
        "markdown": markdown,
        "sections": sections,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    return parsed


# ── 对比 ──────────────────────────────────────────────────────────────────────
def compare_with_reference(paper_id: str, ocr_doc: dict) -> dict:
    ref_path = JSON_CORPUS_DIR / f"{paper_id}.json"
    if not ref_path.exists():
        return {"paper_id": paper_id, "status": "missing_reference"}

    ref_doc = load_json(ref_path)
    ref_sections = [
        clean_text(s.get("text", ""), merge_lines=True)
        for s in ref_doc.get("sections", [])
    ]
    ref_sections = [t for t in ref_sections if len(t) >= 50]

    ocr_sections = [
        s["text"]
        for s in ocr_doc.get("sections", [])
        if len(s.get("text", "")) >= 50
    ]

    ref_chars = sum(len(t) for t in ref_sections)
    ocr_chars = sum(len(t) for t in ocr_sections)
    ratio = round(ocr_chars / ref_chars, 4) if ref_chars else 0.0

    comparison = {
        "paper_id": paper_id,
        "reference_sections": len(ref_sections),
        "ocr_sections": len(ocr_sections),
        "reference_chars": ref_chars,
        "ocr_chars": ocr_chars,
        "char_coverage_ratio": ratio,
    }

    out_path = COMPARE_OUTPUT_DIR / f"{paper_id}_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    return comparison


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    pdf_urls  = load_json(PDF_URLS_PATH)
    paper_ids = list(pdf_urls.keys())[:50]
    print(f"选择了 {len(paper_ids)} 篇论文")

    # Step 1: 下载 PDF
    downloaded_paths: list[Path] = []
    for paper_id in tqdm(paper_ids, desc="下载 PDF"):
        pdf_path = download_pdf(paper_id, pdf_urls[paper_id])
        downloaded_paths.append(pdf_path)
    print(f"[OK] PDF 下载完成，共 {len(downloaded_paths)} 个")

    # Step 2: OCR 解析
    ocr_docs: list[dict] = []
    for pdf_path, paper_id in tqdm(
        list(zip(downloaded_paths, paper_ids)), desc="OCR 解析"
    ):
        ocr_doc = parse_pdf_to_json(pdf_path, paper_id)
        ocr_docs.append(ocr_doc)
    print(f"[OK] OCR 解析完成，共 {len(ocr_docs)} 个文档")

    # Step 3: 与 JSON corpus 对比
    all_results: list[dict] = []
    for paper_id, ocr_doc in zip(paper_ids, ocr_docs):
        result = compare_with_reference(paper_id, ocr_doc)
        all_results.append(result)

    summary_path = COMPARE_OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[OK] 对比摘要 → {summary_path}")

    # 打印汇总统计
    valid = [r for r in all_results if "char_coverage_ratio" in r]
    if valid:
        avg_ratio = sum(r["char_coverage_ratio"] for r in valid) / len(valid)
        avg_ref   = sum(r["reference_sections"] for r in valid) / len(valid)
        avg_ocr   = sum(r["ocr_sections"] for r in valid) / len(valid)
        print(f"\n===== OCR vs JSON 对比汇总 =====")
        print(f"  有效对比论文数  : {len(valid)}")
        print(f"  平均 reference sections : {avg_ref:.1f}")
        print(f"  平均 OCR sections       : {avg_ocr:.1f}")
        print(f"  平均字符覆盖率          : {avg_ratio:.4f}")


if __name__ == "__main__":
    main()
