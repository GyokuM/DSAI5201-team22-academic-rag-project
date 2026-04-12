# 学术论文 RAG 数据处理流水线设计文档

本目录包含两条并行数据处理流水线，分别对应两种论文文本来源：

1. **JSON corpus 流水线**：基于 Open RAG Benchmark 提供的结构化 JSON 语料，完成清洗与三种切分策略
2. **Raw PDF OCR 流水线**：从 arXiv 直接下载原始 PDF，用 `pymupdf4llm` 提取 markdown，与 JSON corpus 做对比

---

## 目录结构

```text
project/
├── code/
│   ├── json_corpus_processing.ipynb     ← JSON 语料处理（notebook 版）
│   ├── raw_pdf_ocr_processing.ipynb     ← OCR 处理（notebook 版）
│   ├── run_processing.py                ← JSON 语料处理（独立脚本版）
│   ├── run_ocr_processing.py            ← OCR 处理（独立脚本版）
│   └── PIPELINE_DESIGN.md              ← 本文档
│
└── data/
    ├── open_ragbench/pdf/arxiv/
    │   ├── corpus/                      ← 1 000 篇论文 JSON（原始来源）
    │   ├── pdf_urls.json                ← arXiv ID → PDF 下载链接
    │   ├── queries.json                 ← 评测问题集
    │   ├── qrels.json                   ← 问题-文档相关性标注
    │   └── answers.json                 ← 参考答案
    │
    ├── processed/                       ← JSON corpus 处理产物
    │   ├── cleaned_sections.jsonl       ← 清洗后的章节（19 816 条）
    │   ├── chunks_fixed.jsonl           ← 固定长度切分（87 769 块）
    │   ├── chunks_recursive.jsonl       ← 递归切分（102 471 块）
    │   ├── chunks_section_aware.jsonl   ← Section-aware 切分（99 773 块）
    │   ├── chunk_stats.json             ← 三种切分统计汇总
    │   └── data_readme.md               ← 数据字段说明文档
    │
    └── rawpdf/                          ← OCR 处理产物
        ├── raw_pdfs/                    ← 下载的原始 PDF（50 个）
        ├── ocr_processed/               ← 每篇 OCR 解析 JSON（50 个）
        └── comparison/                  ← 对比结果（50 篇 + summary.json）
```

---

## 一、JSON Corpus 处理流水线

### 目标

将 `corpus/*.json` 转换为统一的清洗文本，并生成三种切分变体，为后续 FAISS 索引构建和 Recall@k 消融实验提供输入。

### 运行方式

```bash
# 推荐：独立脚本，路径自动绝对化
python project/code/run_processing.py

# 或在 VSCode 中打开 notebook 逐 cell 运行
# project/code/json_corpus_processing.ipynb
```

### 处理步骤

```
corpus/*.json
    │
    ▼ Step 1: 读取所有 section，过滤 text < 50 字符的空节
    ▼ Step 2: 文本清洗（见下表）
cleaned_sections.jsonl
    │
    ├─ Step 3a: Fixed-size chunking   → chunks_fixed.jsonl
    ├─ Step 3b: Recursive chunking    → chunks_recursive.jsonl
    └─ Step 3c: Section-aware chunking → chunks_section_aware.jsonl
    │
    ▼ Step 4: 统计写入 chunk_stats.json
```

### 文本清洗规则

| 规则 | 方法 | 说明 |
|------|------|------|
| arXiv 噪声行 | 正则 `arxiv:\s*\d{4}\.\d{4,5}(v\d+)?` (IGNORECASE) | 删除 arXiv citation 标注行 |
| 纯数字页码行 | 正则 `^\s*\d+\s*$` | 删除仅含页码的行 |
| 连字符换行 | `(\w+)-\n(\w+)` → `\1\2` | 合并 PDF 行末断词 |
| 多余空白 | `[ \t]+` → 单空格；`\n{3,}` → `\n\n` | 规范化空白 |
| 不可见空格 | `\xa0` → ` ` | 替换非断行空格 |
| 内联换行（JSON 版） | 单个 `\n` → 空格 | JSON corpus 专用，消除段落内换行 |

### 三种切分策略

#### 策略一：Fixed（固定长度切分）

- **参数**：`chunk_size=1200`，`chunk_overlap=150`
- **方法**：按字符数硬切，不考虑语义边界
- **产出**：87 769 块，平均 1 074.42 字符/块
- **定位**：Chunking ablation 的 baseline 方案

#### 策略二：Recursive（递归切分）

- **参数**：`chunk_size=1200`，`chunk_overlap=150`
- **方法**：`LangChain.RecursiveCharacterTextSplitter`，分隔符优先级 `["\n\n", "\n", ". ", " ", ""]`
- **产出**：102 471 块，平均 841.89 字符/块（块数最多，块长最短）
- **定位**：兼顾段落/句子语义边界

#### 策略三：Section-aware（章节感知切分）

- **参数**：`max_chunk_chars=1600`，`chunk_overlap=200`
- **方法**：section 整体长度 ≤ 1600 字符时直接保留完整章节；超出则使用 `RecursiveCharacterTextSplitter` 二次切分
- **产出**：99 773 块，平均 864.07 字符/块，最大 1 600 字符
- **定位**：最适合学术论文结构，**推荐实验中重点分析**

### 切分策略对比

| 策略 | 参数 | 块数 | 均值（字符） | 特点 |
|------|------|------|-------------|------|
| Fixed | size=1200, overlap=150 | 87 769 | 1 074.42 | 硬切，块长最均匀 |
| Recursive | size=1200, overlap=150 | 102 471 | 841.89 | 语义感知，块数最多 |
| Section-aware | max=1600, overlap=200 | 99 773 | 864.07 | 保留章节完整性 |

### 输出字段（各 chunks_*.jsonl）

| 字段 | 类型 | 说明 |
|------|------|------|
| `paper_id` | str | 来源论文 arXiv ID |
| `section_id` | str | 来源章节编号 |
| `chunk_id` | int | 章节内块序号（从 0 起） |
| `text` | str | 切分后文本内容 |
| `strategy` | str | `"fixed"` / `"recursive"` / `"section_aware"` |

---

## 二、Raw PDF OCR 处理流水线

### 目标

从 arXiv 直接下载 50 篇原始 PDF，用 `pymupdf4llm.to_markdown()` 提取结构化 markdown，与 JSON corpus 做基础统计对比，验证 PDF 解析质量。

### 运行方式

```bash
# 推荐：独立脚本
python project/code/run_ocr_processing.py

# 或在 VSCode 中打开 notebook 运行
# project/code/raw_pdf_ocr_processing.ipynb
```

### 依赖安装

```bash
pip install requests tqdm pymupdf pymupdf4llm
# 可选：pip install paddleocr paddlepaddle
```

### 关于 pymupdf4llm

当前使用 `pymupdf4llm.to_markdown(...)` 作为 PDF → markdown 提取入口，版本 v1.27.2.2。该工具基于 PyMuPDF，对学术论文排版有良好适配性。若后续需替换为纯 PaddleOCR / PP-Structure 方案，只需替换 `extract_markdown()` 函数，其余流程保持不变。

### 处理步骤

```
pdf_urls.json（前 50 篇）
    │
    ▼ Step 1: 下载原始 PDF → rawpdf/raw_pdfs/{paper_id}.pdf
    ▼ Step 2: pymupdf4llm 提取 markdown → 按 # 标题切分 sections
    ▼ Step 3: 文本清洗（同 JSON corpus 清洗规则，merge_lines=False）
    ▼ Step 4: 写入 rawpdf/ocr_processed/{paper_id}.json
    ▼ Step 5: 与 JSON corpus 做统计对比 → rawpdf/comparison/{paper_id}_comparison.json
    ▼ Step 6: 汇总写入 rawpdf/comparison/summary.json
```

### OCR 解析 JSON 字段（`ocr_processed/*.json`）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 论文 arXiv ID |
| `source` | str | 固定值 `"raw_pdf_pymupdf4llm_markdown"` |
| `markdown` | str | 完整 markdown 原文（未切分） |
| `sections` | list[dict] | 按 `#`～`######` 标题切分的 section 列表 |

每个 section：`{"heading": str, "text": str}`

### 实测运行结果（50 篇）

| 指标 | 数值 |
|------|------|
| 下载 PDF | 50 篇 |
| OCR 解析成功 | 50 篇 |
| 有效对比 | 50 篇 |
| 平均 reference_sections（JSON corpus） | 19.1 |
| 平均 ocr_sections（pymupdf4llm） | 19.9 |
| **平均字符覆盖率** | **0.9753** |

> 字符覆盖率 0.975 表明 pymupdf4llm 方案可覆盖 JSON corpus 约 97.5% 的文本量，解析质量可靠，适合作为 PDF 解析基线。

---

## 三、对比协议

### 当前指标

| 字段 | 计算方式 | 含义 |
|------|----------|------|
| `reference_sections` | JSON corpus 有效章节数（text ≥ 50 字符） | 参考基准章节数 |
| `ocr_sections` | OCR 有效章节数（text ≥ 50 字符） | OCR 解析章节数 |
| `reference_chars` | JSON corpus 所有有效章节总字符数 | 参考文本量 |
| `ocr_chars` | OCR 所有有效章节总字符数 | OCR 文本量 |
| `char_coverage_ratio` | `ocr_chars / reference_chars` | OCR 文本覆盖率（越接近 1 越好） |

### 报告中建议补充的指标

- 相同 query set 下的检索命中率（Recall@k / Hit Rate）
- 与 `qrels.json` 的 section 级别对齐率
- 各切分策略下的平均块长和块数对比
- 对 10 篇抽样论文的人工错误分类：
  - 段落缺失
  - 标题切分错误
  - 公式字符损坏
  - 参考文献泄漏
  - 表格/图像缺失

---

## 四、实验建议顺序

1. 运行 `run_processing.py`，产出 `processed/` 下的 4 个 JSONL 文件
2. 对三种 chunk 文件分别构建 FAISS 索引
3. 用 `queries.json` + `qrels.json` 评测 Recall@k，对比三种切分策略
4. 运行 `run_ocr_processing.py`，产出 `rawpdf/` 下的 OCR 文件
5. 对 OCR 文本同样应用三种切分，构建对比 FAISS 索引
6. 对比"JSON corpus 来源"与"raw PDF OCR 来源"在检索和问答质量上的差异

---

## 五、注意事项

- **路径**：两个脚本均使用 `Path(__file__).resolve()` 绝对化路径，可在任意目录直接运行，无需 `cd` 到特定目录。
- **notebook 路径**：两个 notebook 使用 VSCode 注入的 `__vsc_ipynb_file__` 全局变量定位项目根目录；若不在 VSCode 中运行，回退策略为通过 `Path.cwd()` 推断。
- **数据来源优先级**：第一阶段以 JSON corpus 为主要数据源；Raw PDF OCR 作为对照实验，不作为默认 baseline。
- **切分策略推荐**：对学术论文 RAG，`section_aware` 切分最能保留章节语义完整性，是报告中最值得重点分析的贡献点。
- **可扩展性**：若后续需接入 PaddleOCR / PP-Structure，只需替换 `extract_markdown()` 函数，流水线其余部分（清洗、切分、对比）无需修改。
