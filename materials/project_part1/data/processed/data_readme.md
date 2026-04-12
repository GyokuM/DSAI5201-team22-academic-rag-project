# Data README — `project/data/processed/`

本文档说明 `project/data/processed/` 目录下所有数据文件的来源、格式与字段含义，以及对应 `project/data/rawpdf/` 目录下 OCR 处理产物的结构说明。

---

## 目录结构总览

```
project/data/
├── open_ragbench/pdf/arxiv/
│   ├── corpus/            ← 1 000 篇 arXiv 论文 JSON（原始来源）
│   ├── pdf_urls.json      ← 论文 arXiv ID → PDF 下载地址映射
│   ├── queries.json       ← 问题列表
│   ├── qrels.json         ← 问题→文档相关性标注
│   └── answers.json       ← 参考答案
│
├── processed/             ← JSON corpus 处理产物（本文档说明范围）
│   ├── cleaned_sections.jsonl
│   ├── chunks_fixed.jsonl
│   ├── chunks_recursive.jsonl
│   ├── chunks_section_aware.jsonl
│   └── chunk_stats.json
│
└── rawpdf/                ← raw PDF OCR 处理产物
    ├── raw_pdfs/          ← 下载的原始 PDF 文件
    ├── ocr_processed/     ← 每篇论文的 OCR 解析结果 JSON
    └── comparison/        ← OCR vs JSON corpus 对比结果
```

---

## 一、原始数据来源

### `open_ragbench/pdf/arxiv/corpus/`

- **来源**：[Open RAG Benchmark](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000)，包含 1 000 篇 arXiv 论文的结构化 JSON 解析版本。
- **格式**：每篇一个 JSON 文件，命名为 `{arxiv_id}.json`。
- **字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | arXiv 论文 ID（如 `2401.01872v2`） |
| `title` | str | 论文标题 |
| `abstract` | str | 摘要全文 |
| `authors` | list[str] | 作者列表 |
| `categories` | list[str] | arXiv 分类标签 |
| `sections` | list[dict] | 章节列表（见下） |

每个 section 的字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `section_id` | str | 章节编号或标题标识 |
| `text` | str | 章节正文文本 |
| `tables` | list | 章节内的表格（可为空） |
| `images` | list | 章节内的图片（可为空） |

---

## 二、处理流程概述

处理脚本：[`project/code/run_processing.py`](../code/run_processing.py) 和 [`project/code/json_corpus_processing.ipynb`](../code/json_corpus_processing.ipynb)

```
corpus/*.json
    │
    ▼ Step 1: 文本清洗（正则去噪 + 连字符修复 + 空白规范化）
cleaned_sections.jsonl
    │
    ├─ Step 2a: Fixed-size chunking   → chunks_fixed.jsonl
    ├─ Step 2b: Recursive chunking    → chunks_recursive.jsonl
    └─ Step 2c: Section-aware chunking → chunks_section_aware.jsonl
```

### 清洗规则（适用于全部三种切分策略）

| 规则 | 正则 / 方法 | 说明 |
|------|-------------|------|
| arXiv 噪声行 | `arxiv:\s*\d{4}\.\d{4,5}(v\d+)?` (IGNORECASE) | 删除 arXiv citation 标注行 |
| 纯数字页码行 | `^\s*\d+\s*$` | 删除仅包含页码的行 |
| 连字符换行 | `(\w+)-\n(\w+)` → `\1\2` | 合并 PDF 行末连字符断词 |
| 多余空白 | `[ \t]+` → 单空格，`\n{3,}` → `\n\n` | 规范化空白 |
| 不可见空格 | `\xa0` → ` ` | 替换非断行空格 |

---

## 三、`processed/` 目录各文件说明

### 3.1 `cleaned_sections.jsonl`

- **描述**：对 1 000 篇论文中每个 section 进行清洗后的完整文本单元，是后续三种切分策略的共同输入。
- **格式**：JSONL（每行一个 JSON 对象）
- **统计**：**19 816 条**记录（约等于 1 000 篇 × 平均 19.8 个章节）
- **文件大小**：约 91.9 MB

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `paper_id` | str | 原始论文 arXiv ID |
| `section_id` | str | 章节编号/标识 |
| `text` | str | 清洗后的章节正文文本 |

---

### 3.2 `chunks_fixed.jsonl`

- **描述**：固定长度切分策略产出的文本块，对应论文中的"Contribution 1: Chunking Ablation"的 baseline 策略。
- **格式**：JSONL
- **统计**：
  - **块数**：87 769 条
  - **平均字符数**：1 074.42
  - **最大字符数**：1 200
  - **最小字符数**：51
- **文件大小**：约 128.4 MB
- **切分参数**：
  - `chunk_size = 1200`
  - `chunk_overlap = 150`
  - 按字符数硬切，不考虑语义边界

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `paper_id` | str | 来源论文 arXiv ID |
| `section_id` | str | 来源章节编号 |
| `chunk_id` | int | 本 section 内的块序号（从 0 开始） |
| `text` | str | 切分后的文本块内容 |
| `strategy` | str | 固定值 `"fixed"` |

---

### 3.3 `chunks_recursive.jsonl`

- **描述**：LangChain `RecursiveCharacterTextSplitter` 策略产出的文本块，按段落/句子/词边界递归切分，保留更多语义完整性。
- **格式**：JSONL
- **统计**：
  - **块数**：102 471 条
  - **平均字符数**：841.89
  - **最大字符数**：1 200
  - **最小字符数**：3
- **文件大小**：约 126.0 MB
- **切分参数**：
  - `chunk_size = 1200`
  - `chunk_overlap = 150`
  - 分隔符优先级：`["\n\n", "\n", ". ", " ", ""]`

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `paper_id` | str | 来源论文 arXiv ID |
| `section_id` | str | 来源章节编号 |
| `chunk_id` | int | 本 section 内的块序号（从 0 开始） |
| `text` | str | 切分后的文本块内容 |
| `strategy` | str | 固定值 `"recursive"` |

---

### 3.4 `chunks_section_aware.jsonl`

- **描述**：Section-aware 切分策略产出的文本块。优先将每个 section 作为独立单元保留，当 section 长度超过 max_chunk_chars 时再递归切分，从而保留章节边界语义。
- **格式**：JSONL
- **统计**：
  - **块数**：99 773 条
  - **平均字符数**：864.07
  - **最大字符数**：1 600
  - **最小字符数**：3
- **文件大小**：约 125.8 MB
- **切分参数**：
  - `max_chunk_chars = 1600`（section 整体不超过此值则保留完整）
  - 超出时使用 `RecursiveCharacterTextSplitter(chunk_size=1600, overlap=200)` 二次切分

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `paper_id` | str | 来源论文 arXiv ID |
| `section_id` | str | 来源章节编号 |
| `chunk_id` | int | 本 section 内的块序号（从 0 开始） |
| `text` | str | 切分后的文本块内容 |
| `strategy` | str | 固定值 `"section_aware"` |

---

### 3.5 `chunk_stats.json`

- **描述**：三种切分策略的汇总统计，用于快速对比各策略产出规模。
- **格式**：单个 JSON 对象

```json
{
  "cleaned_sections": 19816,
  "fixed": {
    "num_chunks": 87769,
    "avg_chunk_chars": 1074.42,
    "max_chunk_chars": 1200,
    "min_chunk_chars": 51
  },
  "recursive": {
    "num_chunks": 102471,
    "avg_chunk_chars": 841.89,
    "max_chunk_chars": 1200,
    "min_chunk_chars": 3
  },
  "section_aware": {
    "num_chunks": 99773,
    "avg_chunk_chars": 864.07,
    "max_chunk_chars": 1600,
    "min_chunk_chars": 3
  }
}
```

---

## 四、`rawpdf/` 目录说明

处理脚本：[`project/code/run_ocr_processing.py`](../code/run_ocr_processing.py) 和 [`project/code/raw_pdf_ocr_processing.ipynb`](../code/raw_pdf_ocr_processing.ipynb)

本目录存放从 arXiv 直接下载原始 PDF 并经 `pymupdf4llm` 解析后的产物，以及与 JSON corpus 的对比结果。处理对象为从 `pdf_urls.json` 中选取的前 50 篇论文。

```
rawpdf/
├── raw_pdfs/         ← 原始 PDF 文件
├── ocr_processed/    ← 每篇论文的 markdown + section 解析结果
└── comparison/       ← 每篇论文的 OCR vs JSON 统计对比 + 汇总
```

### 4.1 `rawpdf/raw_pdfs/`

- **描述**：从 arXiv 直接下载的原始 PDF 文件。
- **命名规则**：`{arxiv_id}.pdf`（如 `2407.01528v3.pdf`）
- **数量**：50 个文件

---

### 4.2 `rawpdf/ocr_processed/`

- **描述**：每篇论文经 `pymupdf4llm.to_markdown()` 提取 markdown 后，再分割为 sections 的结构化 JSON。
- **命名规则**：`{arxiv_id}.json`
- **数量**：50 个文件

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 论文 arXiv ID |
| `source` | str | 固定值 `"raw_pdf_pymupdf4llm_markdown"` |
| `markdown` | str | 完整的 markdown 原文（未切分） |
| `sections` | list[dict] | 按 `#`～`######` 标题切分的 section 列表 |

每个 section 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `heading` | str | 章节标题文字 |
| `text` | str | 章节正文（经过清洗：去噪声行 + 连字符修复 + 空白规范化） |

---

### 4.3 `rawpdf/comparison/`

- **描述**：每篇论文的 OCR 解析结果与官方 JSON corpus 的基础统计对比，以及全部 50 篇的汇总文件。
- **命名规则**：`{arxiv_id}_comparison.json` + `summary.json`
- **数量**：50 个单篇文件 + 1 个 `summary.json`

**每篇对比 JSON 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `paper_id` | str | 论文 arXiv ID |
| `reference_sections` | int | JSON corpus 中有效章节数（text ≥ 50 字符） |
| `ocr_sections` | int | OCR 解析得到的有效章节数（text ≥ 50 字符） |
| `reference_chars` | int | JSON corpus 所有有效章节的总字符数 |
| `ocr_chars` | int | OCR 解析所有有效章节的总字符数 |
| `char_coverage_ratio` | float | `ocr_chars / reference_chars`，表征 OCR 文本覆盖率 |

**`summary.json`**：以上 50 篇对比结果的 JSON 数组，可直接用于 chunking/retrieval 实验的数据质量分析。

**汇总统计（50 篇，全部有效）**：

| 指标 | 数值 |
|------|------|
| 有效对比论文数 | 50 |
| 平均 reference_sections（JSON corpus） | 19.1 |
| 平均 ocr_sections（pymupdf4llm） | 19.9 |
| 平均字符覆盖率（char_coverage_ratio） | **0.9753** |

> 字符覆盖率 0.975 表明 pymupdf4llm markdown 提取方案能覆盖 JSON corpus 约 97.5% 的文本量，解析质量可靠，适合作为 PDF 解析基线。

---

## 五、三种切分策略对比摘要

| 策略 | 参数 | 块数 | 平均字符数 | 特点 |
|------|------|------|------------|------|
| Fixed | size=1200, overlap=150 | 87 769 | 1 074.42 | 硬切，块长最均匀，不考虑语义边界 |
| Recursive | size=1200, overlap=150 | 102 471 | 841.89 | 按段落/句子递归切分，产出块数最多 |
| Section-aware | max=1600, overlap=200 | 99 773 | 864.07 | 优先保留章节完整性，超长再递归切 |

> **实验用途**：上述三份 JSONL 文件分别用于构建三个独立 FAISS 索引，通过 Recall@k 对比验证不同切分策略对 RAG 检索质量的影响（Contribution 1: Chunking Strategy Ablation）。

---

## 六、数据用途与下游实验对应

| 数据文件 | 对应实验 |
|----------|----------|
| `cleaned_sections.jsonl` | 章节级检索基线；section-aware chunking 前置步骤 |
| `chunks_fixed.jsonl` | Chunking ablation — Fixed baseline |
| `chunks_recursive.jsonl` | Chunking ablation — Recursive strategy |
| `chunks_section_aware.jsonl` | Chunking ablation — Section-aware strategy（推荐） |
| `chunk_stats.json` | 报告 Table 1：切分统计对比表 |
| `rawpdf/ocr_processed/*.json` | PDF parsing quality 分析；OCR pipeline 验证 |
| `rawpdf/comparison/summary.json` | PDF解析 vs JSON corpus 字符覆盖率分析 |

---

## 七、生成方式

| 脚本 | 功能 | 输出 |
|------|------|------|
| [`run_processing.py`](../code/run_processing.py) | 从 JSON corpus 清洗并三种策略切分 | `processed/` 下全部 5 个文件 |
| [`json_corpus_processing.ipynb`](../code/json_corpus_processing.ipynb) | 同上，notebook 版本，含可视化步骤说明 | 同上 |
| [`run_ocr_processing.py`](../code/run_ocr_processing.py) | 下载50篇PDF → OCR解析 → 与JSON对比 | `rawpdf/` 下全部文件 |
| [`raw_pdf_ocr_processing.ipynb`](../code/raw_pdf_ocr_processing.ipynb) | 同上，notebook 版本 | 同上 |

---

*最后更新：2026-04-03*
