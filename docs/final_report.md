# DSAI5201 Final Report

## Lightweight Academic Paper QA with an Integrated RAG System

### Team Project Report

---

## Abstract

This project studies how to build a lightweight retrieval-augmented generation (RAG) system for academic paper question answering, with a particular focus on the impact of document preprocessing, chunking, retrieval configuration, and grounded answer generation. Rather than training a new large language model from scratch, we treat academic PDF QA as a systems and empirical evaluation problem: the core challenge is not only to generate an answer, but to retrieve the correct evidence from long, structured research papers and keep the answer faithful to that evidence.

Our pipeline uses the Open RAG Benchmark academic paper collection as the main data source, together with a raw-PDF OCR comparison pipeline. We compare three chunking strategies (fixed, recursive, and section-aware), two embedding models (`all-MiniLM-L6-v2` and `BAAI/bge-base-en-v1.5`), multiple Top-K retrieval settings, and three prompt styles for answer generation (`Naive`, `Structured`, and `CoT`). The strongest retrieval results consistently come from `BAAI/bge-base-en-v1.5`, while the best chunking choice depends on retrieval depth: Section-Aware is best at Top-3 and Top-15, Recursive is best at Top-5, and Fixed is best at Top-10. In the rerun prompt evaluation, `Structured` achieves the highest faithfulness score (4.8083), while `Naive` achieves the best ROUGE-L and BERTScore. These results suggest that academic RAG quality depends on careful retrieval and prompting trade-offs rather than a single universally best configuration.

We further integrate the project into a unified deliverable, `Academic RAG Studio`, with a FastAPI backend and a React frontend. The final system supports benchmark paper QA, uploaded single-PDF QA, evidence display, chunking selection for benchmark papers, session history, and a report-ready experiment dashboard. The result is a complete lightweight academic RAG workflow that combines data processing, ablation experiments, evaluation, and a live demo in one coherent project.

---

## 1. Introduction

Question answering over academic papers is a valuable but difficult real-world task. Students, researchers, and practitioners often need to identify methods, assumptions, datasets, limitations, and findings from long technical documents under time pressure. A useful academic QA system should therefore do more than produce fluent text: it should retrieve the right parts of the paper, ground answers in document evidence, and expose the source of its claims.

Academic PDF QA is harder than standard short-context question answering for several reasons. First, research papers are long and highly structured, with abstracts, methods, experiments, appendices, tables, references, and formulas. Second, PDF parsing is noisy: text may be broken across lines, headings may be misdetected, and OCR or PDF extraction can damage structure. Third, retrieval errors compound generation errors: if evidence is missing or poorly chunked, even a strong language model may hallucinate. Finally, academic use cases demand higher faithfulness than casual chat applications because users often need answers that can be traced back to paper sections.

This project addresses these challenges by building and evaluating a lightweight RAG system for academic paper QA. Instead of treating the task as a pure “chatbot” problem, we frame it as a pipeline design problem with measurable system choices. Our goal is to answer the following questions:

1. How much does PDF/text preprocessing affect downstream retrieval?
2. Which chunking strategy is most suitable for academic papers?
3. Which retrieval configuration provides the best trade-off between effectiveness and simplicity?
4. How do prompt styles affect faithfulness and answer quality?
5. Can these ideas be integrated into a lightweight, presentation-ready demo system?

The project therefore makes contributions at both the experimental and systems levels. Experimentally, we compare chunking strategies, embedding models, retrieval depths, and prompt formats. At the systems level, we consolidate all components into one integrated repository and one interactive frontend/backend demo.

---

## 2. Related Background

### 2.1 Retrieval-Augmented Generation

RAG augments a language model with external retrieved context. Instead of asking a model to answer solely from parametric memory, the system first retrieves relevant document chunks from a corpus and then conditions answer generation on those retrieved chunks. In academic QA, this is especially useful because the target knowledge is document-specific and often too recent or too niche to rely on model memory alone.

However, the effectiveness of RAG depends on retrieval quality. If the retrieved chunks do not contain the necessary information, the generator may produce vague or fabricated answers. For that reason, RAG pipelines are often sensitive to preprocessing, chunk size, embedding model choice, and retrieval depth.

### 2.2 Vector Retrieval and FAISS-Style Indexing

Dense retrieval maps text chunks and user questions into vector embeddings. Similarity search then returns the chunks whose embeddings are closest to the question embedding. This project uses sentence-transformer style embeddings and cached chunk vectors as a lightweight retrieval backbone. While the original course direction suggests FAISS as a natural implementation choice, our practical project emphasis is on retrieval configuration and chunk evaluation rather than on building a specialized indexing algorithm from scratch. The same dense retrieval logic remains central: chunk text is embedded, indexed, and ranked by similarity to the query.

### 2.3 PDF Parsing and Chunking

Academic papers differ from ordinary web text because their structure matters. A paper’s section boundaries, headings, method descriptions, and evaluation results often align with how users ask questions. If chunking breaks these boundaries too aggressively, retrieval may return incomplete or noisy snippets. If chunks are too large, unrelated content may dilute the embedding and hurt matching.

This makes chunking a primary design decision rather than a preprocessing detail. We therefore treat chunking as a first-class ablation dimension. In particular, we compare:

- Fixed-size chunking
- Recursive chunking with semantic separators
- Section-aware chunking that tries to preserve academic document structure

### 2.4 Grounded Generation

Grounded generation refers to answer generation that is explicitly constrained by retrieved evidence. In academic QA, grounding is essential because users expect answers to reflect the paper rather than generic background knowledge. Our prompt comparison therefore asks not only which prompt gives the highest lexical overlap with references, but also which prompt remains most faithful to retrieved evidence.

---

## 3. Baseline System

### 3.1 Data Source

The project uses the academic-paper portion of the Open RAG Benchmark as the main corpus. The processed JSON corpus contains 1,000 arXiv papers, associated queries, relevance labels, and reference answers. The baseline data processing pipeline produces 19,816 cleaned document sections for downstream chunking and retrieval.

### 3.2 Parser and Preprocessing

We implement two parallel document-processing paths:

1. A **JSON corpus pipeline**, which uses the benchmark’s structured paper data as the main experimental source.
2. A **raw PDF OCR pipeline**, which downloads original arXiv PDFs and extracts markdown-like text for comparison.

The JSON corpus path is the primary baseline because it is more stable and easier to analyze consistently. The OCR path is used as a comparison to evaluate whether raw-PDF extraction can approximate the benchmark corpus quality.

Preprocessing includes:

- removing arXiv noise lines
- removing pure page-number lines
- merging hyphenated line breaks
- normalizing whitespace
- replacing non-breaking spaces
- collapsing inline line breaks for the JSON corpus

### 3.3 Chunking

The baseline pipeline produces three chunk variants:

- **Fixed**: `chunk_size=1200`, `chunk_overlap=150`
- **Recursive**: `chunk_size=1200`, `chunk_overlap=150`
- **Section-aware**: keep full sections up to 1600 characters, otherwise recursively split with overlap 200

The processed corpus statistics are:

| Strategy | Number of Chunks | Average Chunk Length | Max Length |
|---|---:|---:|---:|
| Fixed | 87,769 | 1,074.42 chars | 1,200 |
| Recursive | 102,471 | 841.89 chars | 1,200 |
| Section-aware | 99,773 | 864.07 chars | 1,600 |

These variants form the basis for retrieval ablation.

### 3.4 Embedding and Retrieval

The retrieval baseline compares two embedding models:

- `all-MiniLM-L6-v2`, a lighter sentence embedding model
- `BAAI/bge-base-en-v1.5`, a stronger mid-sized embedding model

The system evaluates chunk hit rate at Top-3, Top-5, Top-10, and Top-15. The retrieval workstream also tested reranking, but the project notes conclude that reranking is not necessary for the final lightweight system, because gains were limited relative to added complexity.

### 3.5 Generation

For generation, the project compares prompt styles rather than training a new language model. The original notebook-based generation path was later rerun in a cleaner script pipeline using `answers.json` as reference. Three prompt strategies are compared:

- `Naive`
- `Structured`
- `CoT`

The final integrated system supports two deployment styles:

- local retrieval plus lightweight grounded fallback generation
- local retrieval plus API-based model generation through an OpenAI-compatible endpoint

### 3.6 Baseline Deliverable

The baseline system therefore includes the full RAG chain:

1. paper ingestion and cleaning
2. chunking
3. dense retrieval
4. answer generation
5. evidence display

This baseline is the foundation for the contributions described next.

---

## 4. Our Contributions

This project is not merely a reproduction of an existing academic QA pipeline. Our main contributions lie in systematic ablation, practical system optimization, and integrated deployment.

### 4.1 Parsing and Chunking Ablation

Our first contribution is a structured comparison of preprocessing and chunking strategies for academic paper QA. We do not assume that a generic splitter is sufficient; instead, we explicitly test whether preserving document structure improves retrieval outcomes. We compare fixed, recursive, and section-aware chunking, and we also build a raw-PDF OCR comparison path to measure how closely OCR-extracted text matches the benchmark corpus.

The OCR pipeline achieves an average character coverage ratio of **0.9753** over 50 sampled PDFs, suggesting that the selected extraction method is a viable text baseline. More importantly, chunking ablation reveals that retrieval quality changes noticeably across chunk strategies, confirming that chunk design is a meaningful contribution rather than an implementation detail.

### 4.2 Retrieval Optimization

Our second contribution is retrieval configuration optimization. We compare two embedding models across four Top-K settings and three chunking strategies. This allows us to ask whether a lightweight embedding backbone is already enough, when stronger embeddings become beneficial, and whether retrieval depth changes the preferred chunking strategy.

The strongest result at each retrieval depth is:

| Top-K | Best Configuration | Chunk Hit Rate |
|---|---|---:|
| Top-3 | Section-Aware + BGE | 0.770 |
| Top-5 | Recursive + BGE | 0.825 |
| Top-10 | Fixed + BGE | 0.895 |
| Top-15 | Section-Aware + BGE | 0.920 |

Across all settings, `BAAI/bge-base-en-v1.5` consistently outperforms `all-MiniLM-L6-v2`, while the best chunking strategy varies with retrieval depth. This provides a concrete new insight: there is no universally best chunking method, but section-aware and recursive strategies are especially strong in lower-depth retrieval where semantic precision matters more.

### 4.3 Grounded Answering and Prompt Comparison

Our third contribution is grounded answer generation evaluation. We compare prompt styles instead of treating generation as a fixed black box. The rerun evaluation uses `answers.json` as the reference target and summarizes performance with ROUGE-L, BERTScore, faithfulness, latency, hallucination count, and over-generic response count.

The main prompt findings are:

| Strategy | ROUGE-L | BERTScore F1 | Faithfulness |
|---|---:|---:|---:|
| Naive | 0.2192 | 0.8595 | 4.7917 |
| Structured | 0.2077 | 0.8545 | 4.8083 |
| CoT | 0.1832 | 0.8485 | 4.3750 |

These results suggest a trade-off. `Naive` produces the strongest overlap metrics, but `Structured` produces the highest faithfulness. `CoT` is less effective in this setup, possibly because extra reasoning scaffolding adds verbosity without improving retrieval grounding. This is useful for academic RAG, where faithfulness may matter more than stylistic richness.

### 4.4 Lightweight Deployment and Integrated Deliverable

Our fourth contribution is turning the experimental pipeline into a complete lightweight deliverable. We integrate the three original project parts into one repository, one backend, and one frontend application:

- merged project materials under one workspace
- unified FastAPI backend APIs
- polished React demo interface
- benchmark paper QA
- uploaded single-PDF QA
- evidence display and session history
- experiment dashboard for report and presentation use

This matters because the course emphasizes an end-to-end practical pipeline rather than isolated notebooks. The final system therefore demonstrates not only analysis, but also deployment-minded integration.

---

## 5. Experimental Setup

### 5.1 Dataset

The main experiments use the academic-paper split of the Open RAG Benchmark. The processed corpus contains 1,000 papers and 19,816 cleaned sections. The benchmark also provides:

- `queries.json` for evaluation questions
- `qrels.json` for relevance labels
- `answers.json` for reference answers

For OCR comparison, we additionally process 50 raw arXiv PDFs through a markdown extraction pipeline.

### 5.2 Question Set

The retrieval experiments are aligned with the benchmark query-relevance format. The rerun generation evaluation uses a subset of **120 queries**, as recorded in the rerun configuration.

### 5.3 Retrieval Settings

We compare:

- chunking: `Fixed`, `Recursive`, `Section-Aware`
- embedding model: `all-MiniLM-L6-v2`, `BAAI/bge-base-en-v1.5`
- retrieval depth: Top-3, Top-5, Top-10, Top-15

The generation rerun uses:

- chunking: `section_aware`
- embedding model: `BAAI/bge-base-en-v1.5`
- Top-K: `5`

### 5.4 Prompt Settings

Three prompt strategies are evaluated:

- `Naive`
- `Structured`
- `CoT`

The rerun script records the generator mode as `lexical` for the evaluation pipeline, which keeps the experiment reproducible locally while preserving the retrieval and prompt-comparison structure. The integrated demo later supports API-backed generation for live presentation.

### 5.5 Evaluation Metrics

We use the following metrics:

- **Chunk Hit@K** for retrieval effectiveness
- **ROUGE-L** for reference overlap
- **BERTScore F1** for semantic similarity
- **Faithfulness** for grounded answer reliability
- **Latency** for response cost
- **Hallucination count**
- **Too generic count**

### 5.6 Environment

The project was developed as a lightweight pipeline using Python-based preprocessing and backend services together with a React frontend. The final application is designed to run on a single machine with local retrieval and optional API-based generation. This setup supports both reproducible local evaluation and stable live demonstration without requiring large-model training infrastructure.

---

## 6. Results and Analysis

### 6.1 Parsing and Preprocessing Findings

The processing pipeline confirms that academic QA performance begins with document quality. The JSON corpus path yields 19,816 cleaned sections, providing a stable basis for chunking experiments. The OCR comparison over 50 papers achieves an average character coverage ratio of 0.9753, indicating that raw-PDF extraction can recover most benchmark text content. However, because OCR and PDF extraction can still introduce structural noise, the JSON corpus remains the main baseline for controlled evaluation.

This justifies the project’s text-first strategy: we first optimize retrieval and grounded answering on stable extracted text before extending the system toward more complex multimodal paper content.

### 6.2 Retrieval Results

The retrieval experiments show two consistent patterns.

First, `BAAI/bge-base-en-v1.5` is stronger than `all-MiniLM-L6-v2` at every Top-K. The gap is modest at lower K but remains consistent, making BGE the safer default for the final integrated system.

Second, the best chunking strategy depends on retrieval depth:

- At **Top-3**, Section-Aware performs best, suggesting that preserving local academic structure helps when only a few chunks are retrieved.
- At **Top-5**, Recursive becomes best, likely because it balances semantic boundaries and chunk granularity.
- At **Top-10**, Fixed performs best, possibly because broader retrieval depth reduces the harm of hard boundaries.
- At **Top-15**, Section-Aware again becomes best, suggesting that document-aware chunking remains valuable when recall is prioritized.

These results imply that chunking should not be chosen blindly. For academic QA, the right chunking strategy depends on how many evidence chunks the system plans to use downstream.

### 6.3 Prompt Strategy Results

The rerun prompt evaluation shows that generation quality is also configuration-sensitive.

`Naive` achieves the best ROUGE-L and BERTScore, which means it aligns most closely with the lexical and semantic content of the references. However, `Structured` achieves the highest faithfulness score. This is an important practical finding: explicitly structured prompting appears to keep answers better aligned with retrieved evidence, even if the resulting text is not the closest lexical match to the reference answer.

`CoT` underperforms both `Naive` and `Structured` in this setting. A likely explanation is that when retrieval context is already narrow and evidence-based, extra reasoning structure may encourage verbose or indirect answers rather than more faithful ones.

### 6.4 Case Study

A representative example from the rerun outputs concerns the question:

> *What are the challenges in estimating output impedance in inverter-based grids?*

In the evaluated outputs, the three prompt strategies all generate short evidence-centered answers, but `Structured` explicitly anchors claims to cited evidence markers while remaining concise. This is consistent with the higher faithfulness score observed in the summary table. Although the local rerun mode does not always preserve fully expanded retrieved section metadata in the saved results, the answer patterns still illustrate the intended grounded-answer behavior.

### 6.5 Error Analysis

Several recurring error modes appear across the project:

1. **Parsing noise**
   Some PDF-origin text contains formatting artifacts, merged lines, or noisy references.

2. **Chunking mismatch**
   Hard chunk boundaries can separate the answer-bearing sentence from its local context.

3. **Retrieval miss**
   The correct paper may be identified, but the specific chunk selected is only partially relevant.

4. **Evidence underuse**
   Even when retrieval is approximately correct, generation may produce overly generic answers.

5. **Grounding drift**
   More elaborate prompting can sometimes reduce clarity or alignment instead of improving it.

These findings support the project’s main thesis: academic RAG quality is governed by pipeline design choices, not only by the generator model.

### 6.6 Best Configuration Summary

From the combined experiments and integrated demo perspective, the strongest practical configuration is:

- text-first JSON corpus baseline
- section-aware chunking as the default demo configuration
- `BAAI/bge-base-en-v1.5` as the embedding backbone
- Top-K evidence display with moderate retrieval depth
- structured, grounded answering for presentation and evidence transparency

This configuration is also the most suitable for live demonstration because it balances retrieval quality, interpretability, and implementation simplicity.

---

## 7. Limitations and Future Work

This project still has several limitations.

First, the current system is **text-first**. Although the benchmark can include tables and images, our implemented analysis and integrated demo primarily focus on textual paper content. This is appropriate for a lightweight first version but does not yet address multimodal academic QA fully.

Second, the uploaded-PDF workflow currently uses a lightweight temporary text chunking path rather than the full benchmark ablation setup. This keeps the demo responsive, but it means uploaded PDFs are not evaluated with the same fixed/recursive/section-aware experiment assets as benchmark papers.

Third, the generation rerun is intentionally lightweight and reproducible. While this is useful for evaluation consistency, future work could extend the study with stronger model-backed evaluation under the same experimental framework.

Fourth, OCR comparison is currently limited to baseline statistics and controlled parsing analysis. A deeper comparison between OCR-origin retrieval and JSON-corpus retrieval on the full benchmark query set would strengthen the conclusions further.

Future work can therefore proceed in four directions:

- extend the system to table and figure-aware multimodal QA
- apply unified chunking ablation to uploaded user PDFs
- evaluate more generators and rerankers under the same framework
- expand error analysis with richer manually annotated failure categories

---

## 8. Conclusion

This project presented a lightweight academic paper QA system built around retrieval-augmented generation and grounded answering. Rather than training a new model, we focused on the practical decisions that determine whether academic RAG works well in real use: preprocessing quality, chunking strategy, retrieval configuration, and prompt design.

Our experiments show that:

- chunking matters and should be treated as a real ablation dimension
- `BAAI/bge-base-en-v1.5` is the strongest retrieval backbone among the tested embeddings
- the best chunking strategy depends on retrieval depth
- structured prompting improves answer faithfulness
- a lightweight, integrated academic QA system can be built without full-scale model training

Finally, we integrated the full project into a single repository and a live demo application, turning three separate workstreams into one coherent system. This makes the project not only analytically strong, but also practical and presentation-ready.

---

## 9. Statement of Contribution

> Replace the names, student IDs, and contribution wording below with the team’s official submission details before exporting the final PDF.

| Member | Student ID | Main Responsibility | Concrete Work |
|---|---|---|---|
| Member 1 | `[ID]` | Data processing and chunking pipeline | Built JSON corpus preprocessing, three chunking strategies, OCR comparison pipeline, and chunk statistics outputs |
| Member 2 | `[ID]` | Retrieval experiments | Compared embedding models and Top-K retrieval settings, produced retrieval result tables, and evaluated reranking practicality |
| Member 3 | `[ID]` | Generation and evaluation | Implemented prompt-based generation experiments, initial evaluation notebooks, and result visualizations |
| Zheng Yan | `[ID]` | Integration, rerun, demo, and report | Merged all workstreams into one repository, rebuilt Part 3 evaluation as a cleaner rerun, developed the integrated frontend/backend demo, added uploaded-PDF QA and session history, and wrote the final report |

---

## References

1. Open RAG Benchmark academic-paper dataset and benchmark assets.
2. Project preprocessing and chunking pipeline documentation.
3. Retrieval experiment CSV outputs for Top-3, Top-5, Top-10, and Top-15.
4. Rerun evaluation summary and integrated application deliverable.

