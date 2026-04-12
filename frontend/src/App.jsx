import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { api } from "./api";

const EMBEDDING_COLORS = {
  "BAAI/bge-base-en-v1.5": "#e36a2c",
  "all-MiniLM-L6-v2": "#1f7a72"
};

const VIEW_META = {
  demo: {
    title: "Paper Demo",
    eyebrow: "Interactive Experience",
    description: "Ask questions about a selected paper, inspect retrieved evidence, and show the grounded answer path."
  },
  results: {
    title: "Results Board",
    eyebrow: "Experiment Synthesis",
    description: "Summarize the strongest retrieval settings and compare the prompt strategies used in the rerun evaluation."
  },
  system: {
    title: "Integration Notes",
    eyebrow: "Project Story",
    description: "Explain how the three workstreams were merged into one deliverable and what remains as optional polish."
  }
};

function StatusPill({ children, tone = "neutral" }) {
  return <span className={`status-pill status-pill-${tone}`}>{children}</span>;
}

function StatCard({ label, value, detail, tone = "neutral" }) {
  return (
    <article className={`stat-card stat-card-${tone}`}>
      <span className="eyebrow">{label}</span>
      <strong>{value}</strong>
      <p>{detail}</p>
    </article>
  );
}

function InsightCard({ title, body, meta }) {
  return (
    <article className="insight-card">
      <h3>{title}</h3>
      <p>{body}</p>
      {meta ? <span className="insight-meta">{meta}</span> : null}
    </article>
  );
}

function SectionCard({ item, index }) {
  return (
    <article className="evidence-card">
      <div className="evidence-head">
        <span className="source-label">Source {index + 1}</span>
        <span>{item.label}</span>
        {"score" in item ? <span className="score-pill">{item.score}</span> : null}
      </div>
      <p>{item.snippet}</p>
    </article>
  );
}

function HeroMetric({ label, value }) {
  return (
    <div className="hero-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function HistoryCard({ item, onSelect }) {
  return (
    <button className="history-card" onClick={onSelect}>
      <div className="history-head">
        <span className="eyebrow">{item.sourceLabel}</span>
        <span>{item.chunkingLabel}</span>
      </div>
      <strong>{item.question}</strong>
      <p>{item.answerPreview}</p>
    </button>
  );
}

export default function App() {
  const [view, setView] = useState("demo");
  const [dashboard, setDashboard] = useState(null);
  const [papers, setPapers] = useState([]);
  const [selectedPaperId, setSelectedPaperId] = useState("");
  const [paperDetail, setPaperDetail] = useState(null);
  const [sampleQuestions, setSampleQuestions] = useState([]);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(3);
  const [chunking, setChunking] = useState("section_aware");
  const [demoResult, setDemoResult] = useState(null);
  const [sourceType, setSourceType] = useState("paper");
  const [uploadedDoc, setUploadedDoc] = useState(null);
  const [uploadingPdf, setUploadingPdf] = useState(false);
  const [history, setHistory] = useState([]);
  const [loadingAsk, setLoadingAsk] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api.dashboard(), api.papers()])
      .then(([dashboardData, paperData]) => {
        setDashboard(dashboardData);
        setPapers(paperData.items);
        if (paperData.items.length > 0) {
          setSelectedPaperId(paperData.items[0].id);
        }
      })
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    if (!selectedPaperId) return;
    Promise.all([api.paper(selectedPaperId), api.paperQuestions(selectedPaperId)])
      .then(([paper, questionData]) => {
        setPaperDetail(paper);
        setSampleQuestions(questionData.items);
      })
      .catch((err) => setError(String(err)));
  }, [selectedPaperId]);

  const retrievalChartData = useMemo(() => {
    if (!dashboard) return [];
    return dashboard.best_retrieval_by_top_k.map((item) => ({
      topK: `Top-${item.top_k}`,
      hitRate: Number((item.hit_rate * 100).toFixed(1)),
      label: `${item.chunking} / ${item.embedding}`
    }));
  }, [dashboard]);

  const embeddingComparison = useMemo(() => {
    if (!dashboard) return [];
    const grouped = new Map();
    dashboard.retrieval_rows.forEach((row) => {
      const key = `${row.top_k}-${row.embedding}`;
      const current = grouped.get(key);
      if (!current || row.hit_rate > current.hit_rate) {
        grouped.set(key, row);
      }
    });
    return [...grouped.values()]
      .sort((a, b) => a.top_k - b.top_k)
      .map((row) => ({
        topK: row.top_k,
        embedding: row.embedding,
        hitRate: Number((row.hit_rate * 100).toFixed(1))
      }));
  }, [dashboard]);

  const promptSummary = useMemo(() => dashboard?.generation_summary ?? [], [dashboard]);
  const chunkingOptions = dashboard?.demo_chunking_options ?? ["fixed", "recursive", "section_aware"];

  const bestRetrieval = useMemo(() => {
    if (!dashboard?.best_retrieval_by_top_k?.length) return null;
    return [...dashboard.best_retrieval_by_top_k].sort((a, b) => b.hit_rate - a.hit_rate)[0];
  }, [dashboard]);

  const bestPrompt = useMemo(() => {
    if (!promptSummary.length) return null;
    return [...promptSummary].sort((a, b) => b.faithfulness_mean - a.faithfulness_mean)[0];
  }, [promptSummary]);

  const currentMode = dashboard?.integration_notes?.current_demo_backend ?? "Loading backend status...";

  const activeSourceLabel =
    sourceType === "upload" && uploadedDoc ? uploadedDoc.title : (paperDetail?.title ?? "Selected paper");

  const handleSearch = async () => {
    setError("");
    try {
      const result = await api.papers(searchTerm);
      setPapers(result.items);
      if (result.items[0]) {
        setSelectedPaperId(result.items[0].id);
      }
    } catch (err) {
      setError(String(err));
    }
  };

  const handleAsk = async (questionOverride) => {
    const nextQuestion = typeof questionOverride === "string" ? questionOverride.trim() : question.trim();
    const hasSource = sourceType === "upload" ? Boolean(uploadedDoc?.id) : Boolean(selectedPaperId);
    if (!hasSource || !nextQuestion) return;
    setLoadingAsk(true);
    setError("");
    try {
      const result =
        sourceType === "upload" && uploadedDoc
          ? await api.askUploaded(uploadedDoc.id, { question: nextQuestion, top_k: topK, chunking })
          : await api.askPaper(selectedPaperId, { question: nextQuestion, top_k: topK, chunking });
      setDemoResult(result);
      setHistory((current) => [
        {
          id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
          question: nextQuestion,
          answerPreview: result.answer.slice(0, 160),
          sourceType,
          sourceLabel: sourceType === "upload" && uploadedDoc ? uploadedDoc.title : (paperDetail?.title ?? "Paper"),
          chunkingLabel: sourceType === "upload" ? "uploaded pdf" : chunking,
          result
        },
        ...current
      ].slice(0, 8));
    } catch (err) {
      setError(String(err));
    } finally {
      setLoadingAsk(false);
    }
  };

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploadingPdf(true);
    setError("");
    try {
      const result = await api.uploadPdf(file);
      setUploadedDoc(result);
      setSourceType("upload");
      setQuestion("");
      setDemoResult(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setUploadingPdf(false);
      event.target.value = "";
    }
  };

  const switchSourceType = (nextSourceType) => {
    setSourceType(nextSourceType);
    setDemoResult(null);
    setQuestion("");
  };

  const clearUploadedDoc = () => {
    setUploadedDoc(null);
    setDemoResult(null);
    setQuestion("");
    setSourceType("paper");
  };

  return (
    <div className="app-shell">
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />

      <header className="hero glass-panel">
        <div className="hero-copy-block">
          <p className="hero-kicker">DSAI5201 · Final Integrated Demonstration</p>
          <h1>Academic RAG Studio</h1>
          <p className="hero-copy">
            A presentation-ready system that unifies data processing, retrieval experiments, prompt evaluation,
            and a live paper question-answering demo into one polished interface.
          </p>
        </div>

        <div className="hero-side">
          <div className="hero-badges">
            <StatusPill tone="teal">Part 1 · Pipeline</StatusPill>
            <StatusPill tone="amber">Part 2 · Retrieval</StatusPill>
            <StatusPill tone="ink">Part 3 · Evaluation</StatusPill>
          </div>

          <div className="hero-metrics">
            <HeroMetric label="Corpus Units" value={dashboard ? dashboard.chunk_stats.cleaned_sections.toLocaleString() : "—"} />
            <HeroMetric label="Best Hit Rate" value={bestRetrieval ? `${(bestRetrieval.hit_rate * 100).toFixed(1)}%` : "—"} />
            <HeroMetric label="Best Prompt" value={bestPrompt?.strategy ?? "—"} />
          </div>
        </div>
      </header>

      <section className="view-header">
        <nav className="view-switcher">
          {Object.entries(VIEW_META).map(([key, meta]) => (
            <button
              key={key}
              className={view === key ? "active" : ""}
              onClick={() => setView(key)}
            >
              {meta.title}
            </button>
          ))}
        </nav>

        <div className="view-summary glass-panel">
          <span className="eyebrow">{VIEW_META[view].eyebrow}</span>
          <h2>{VIEW_META[view].title}</h2>
          <p>{VIEW_META[view].description}</p>
        </div>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      {view === "demo" ? (
        <section className="demo-layout">
          <div className="demo-column">
            <div className="panel glass-panel workspace-panel stage-panel">
              <div className="panel-head">
                <span className="eyebrow">Paper Workspace</span>
                <h2>Ask a Paper</h2>
                <p>Select a paper, choose a benchmark-like question, or compose your own prompt for the live demo.</p>
              </div>

              <div className="mode-banner">
                <div>
                  <span className="eyebrow">Backend Status</span>
                  <strong>{currentMode}</strong>
                </div>
                <StatusPill tone={currentMode.includes("model generation") ? "teal" : "amber"}>
                  {currentMode.includes("model generation") ? "Live Model" : "Fallback Ready"}
                </StatusPill>
              </div>

              <div className="source-toggle">
                <button
                  className={sourceType === "paper" ? "active" : ""}
                  onClick={() => switchSourceType("paper")}
                >
                  Benchmark Papers
                </button>
                <button
                  className={sourceType === "upload" ? "active" : ""}
                  onClick={() => switchSourceType("upload")}
                >
                  Uploaded PDF
                </button>
              </div>

              {sourceType === "paper" ? (
                <>
                  <label className="field-label">Find a paper</label>
                  <div className="search-row">
                    <input
                      value={searchTerm}
                      onChange={(event) => setSearchTerm(event.target.value)}
                      placeholder="Search by title or category"
                    />
                    <button onClick={handleSearch}>Search</button>
                  </div>

                  <label className="field-label">Choose a paper</label>
                  <select value={selectedPaperId} onChange={(event) => setSelectedPaperId(event.target.value)}>
                    {papers.map((paper) => (
                      <option key={paper.id} value={paper.id}>
                        {paper.title}
                      </option>
                    ))}
                  </select>

                  {paperDetail ? (
                    <div className="paper-summary">
                      <div className="paper-summary-head">
                        <div>
                          <span className="eyebrow">Selected Paper</span>
                          <h3>{paperDetail.title}</h3>
                        </div>
                        <StatusPill tone="neutral">{paperDetail.section_count} sections</StatusPill>
                      </div>
                      <p>{paperDetail.abstract.slice(0, 310)}...</p>
                      <div className="tag-row">
                        {paperDetail.categories.slice(0, 4).map((tag) => (
                          <span key={tag}>{tag}</span>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  <label className="field-label">Sample benchmark questions</label>
                  <div className="sample-list">
                    {sampleQuestions.map((item) => (
                      <button
                        key={item.query_id}
                        className="sample-chip"
                        onClick={() => {
                          setQuestion(item.question);
                          void handleAsk(item.question);
                        }}
                      >
                        {item.question}
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                <>
                  <div className="upload-box">
                    <div>
                      <span className="eyebrow">Upload PDF</span>
                      <strong>Single text-based PDF</strong>
                      <p>Upload one PDF for temporary retrieval and question answering during this session.</p>
                    </div>
                    <label className="upload-button">
                      {uploadingPdf ? "Uploading..." : "Choose PDF"}
                      <input type="file" accept="application/pdf" onChange={handleUpload} hidden />
                    </label>
                  </div>

                  {uploadedDoc ? (
                    <div className="uploaded-summary">
                      <div className="paper-summary-head">
                        <div>
                          <span className="eyebrow">Uploaded Document</span>
                          <h3>{uploadedDoc.title}</h3>
                        </div>
                        <div className="uploaded-actions">
                          <StatusPill tone="neutral">{uploadedDoc.chunk_count} chunks</StatusPill>
                          <button className="ghost-button" onClick={clearUploadedDoc}>
                            Remove
                          </button>
                        </div>
                      </div>
                      <p>{uploadedDoc.preview}...</p>
                    </div>
                  ) : (
                    <div className="empty-state compact-empty">
                      <div>
                        <span className="eyebrow">No PDF Uploaded</span>
                        <p>Upload a text-based PDF to start a temporary document question-answering session.</p>
                      </div>
                    </div>
                  )}
                </>
              )}

              <label className="field-label">Question</label>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask about the selected paper"
              />

              <div className="ask-row">
                <label className="control-inline">
                  <span>Top-K Evidence</span>
                  <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
                    {[1, 2, 3, 4, 5].map((value) => (
                      <option key={value} value={value}>
                        {value}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="control-inline">
                  <span>{sourceType === "upload" ? "Document Processing" : "Chunking"}</span>
                  {sourceType === "upload" ? (
                    <div className="static-control">Temporary text chunking</div>
                  ) : (
                    <select
                      value={chunking}
                      onChange={(event) => setChunking(event.target.value)}
                    >
                      {chunkingOptions.map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  )}
                </label>
                <button className="primary-button" onClick={() => void handleAsk()} disabled={loadingAsk}>
                  {loadingAsk ? "Generating..." : "Run Demo"}
                </button>
              </div>
            </div>
          </div>

          <div className="demo-column">
            <div className="panel glass-panel answer-panel accent-panel">
              <div className="panel-head">
                <span className="eyebrow">Answer Workspace</span>
                <h2>Grounded Output</h2>
                <p>The live answer stays paired with the retrieved evidence used by the backend.</p>
              </div>

              {demoResult ? (
                <>
                  <div className="answer-card">
                    <div className="answer-head">
                      <StatusPill tone="teal">{demoResult.mode}</StatusPill>
                      <StatusPill tone="neutral">{demoResult.retrieved_sections?.length ?? 0} evidence chunks</StatusPill>
                    </div>
                    <p>{demoResult.answer}</p>
                  </div>

                  <div className="evidence-header">
                    <div>
                      <span className="eyebrow">Retrieved Evidence</span>
                      <h3>Why The System Answered This Way</h3>
                    </div>
                    <p>{activeSourceLabel}</p>
                  </div>

                  <div className="evidence-list">
                    {demoResult.retrieved_sections?.map((item, index) => (
                      <SectionCard item={item} index={index} key={`${item.section_id}-${index}`} />
                    ))}
                  </div>
                </>
              ) : (
                <div className="empty-state">
                  <div>
                    <span className="eyebrow">Live Demo Ready</span>
                    <p>Pick a paper, ask a question, and the system will retrieve evidence before producing a grounded answer.</p>
                  </div>
                </div>
              )}
            </div>

            <div className="panel glass-panel history-panel">
              <div className="panel-head">
                <span className="eyebrow">Session History</span>
                <h2>Recent Questions</h2>
                <p>Review earlier answers from this session and reopen their evidence trail.</p>
              </div>
              <div className="history-list">
                {history.length ? (
                  history.map((item) => (
                    <HistoryCard
                      key={item.id}
                      item={item}
                      onSelect={() => {
                        setQuestion(item.question);
                        setDemoResult(item.result);
                        setSourceType(item.sourceType);
                      }}
                    />
                  ))
                ) : (
                  <div className="empty-state compact-empty">
                    <div>
                      <span className="eyebrow">No History Yet</span>
                      <p>Asked questions will appear here so you can revisit the answer and evidence.</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      ) : null}

      {view === "results" && dashboard ? (
        <section className="results-layout">
          <div className="stats-row">
            <StatCard
              label="Processed Corpus Units"
              value={dashboard.chunk_stats.cleaned_sections.toLocaleString()}
              detail="Section-aware text units generated in Part 1."
              tone="neutral"
            />
            <StatCard
              label="Strongest Retrieval"
              value={bestRetrieval ? `${(bestRetrieval.hit_rate * 100).toFixed(1)}%` : "—"}
              detail={bestRetrieval ? `${bestRetrieval.chunking} + ${bestRetrieval.embedding}` : "Best hit rate across Top-K settings."}
              tone="amber"
            />
            <StatCard
              label="Best Prompt Strategy"
              value={bestPrompt?.strategy ?? "—"}
              detail={bestPrompt ? `Faithfulness ${bestPrompt.faithfulness_mean.toFixed(2)}` : "Prompt comparison from rerun Part 3."}
              tone="teal"
            />
            <StatCard
              label="Rerun Status"
              value="Completed"
              detail="Part 3 was rerun against answers.json and is now feeding this dashboard."
              tone="ink"
            />
          </div>

          <div className="chart-grid">
            <div className="panel glass-panel chart-panel retrieval-panel">
              <div className="panel-head">
                <span className="eyebrow">Retrieval Comparison</span>
                <h2>Best Retrieval By Top-K</h2>
                <p>Best chunking and embedding combination at each retrieval depth.</p>
              </div>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={retrievalChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(23, 31, 38, 0.12)" />
                    <XAxis dataKey="topK" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="hitRate" stroke="#e36a2c" strokeWidth={3} dot={{ r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="mini-list">
                {retrievalChartData.map((item) => (
                  <div key={item.topK} className="mini-row">
                    <span>{item.topK}</span>
                    <strong>{item.hitRate}%</strong>
                    <span>{item.label}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="panel glass-panel chart-panel embedding-panel">
              <div className="panel-head">
                <span className="eyebrow">Embedding Trend</span>
                <h2>Embedding Comparison</h2>
                <p>Best-performing chunking per embedding at each Top-K.</p>
              </div>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={embeddingComparison}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(23, 31, 38, 0.12)" />
                    <XAxis dataKey="topK" />
                    <YAxis />
                    <Tooltip />
                    {Object.keys(EMBEDDING_COLORS).map((embedding) => (
                      <Line
                        key={embedding}
                        type="monotone"
                        dataKey="hitRate"
                        data={embeddingComparison.filter((item) => item.embedding === embedding)}
                        stroke={EMBEDDING_COLORS[embedding]}
                        strokeWidth={3}
                        dot={{ r: 4 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="legend-row">
                {Object.entries(EMBEDDING_COLORS).map(([name, color]) => (
                  <span key={name}>
                    <i style={{ background: color }} />
                    {name}
                  </span>
                ))}
              </div>
            </div>

            <div className="panel glass-panel wide-panel chart-panel prompt-panel">
              <div className="panel-head">
                <span className="eyebrow">Generation Snapshot</span>
                <h2>Prompt Strategy Comparison</h2>
                <p>Rerun Part 3 summary highlighting answer faithfulness, latency, and hallucination counts.</p>
              </div>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={promptSummary}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(23, 31, 38, 0.12)" />
                    <XAxis dataKey="strategy" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="faithfulness_mean" radius={[8, 8, 0, 0]}>
                      {promptSummary.map((entry) => (
                        <Cell
                          key={entry.strategy}
                          fill={entry.strategy === "Naive" ? "#1f7a72" : entry.strategy === "Structured" ? "#e36a2c" : "#202733"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="prompt-grid">
                {promptSummary.map((item) => (
                  <div key={item.strategy} className="prompt-card">
                    <span className="eyebrow">{item.strategy}</span>
                    <strong>Faithfulness {item.faithfulness_mean.toFixed(2)}</strong>
                    <p>ROUGE-L {item.rouge_l_mean.toFixed(2)}</p>
                    <p>Latency {item.latency_s_mean.toFixed(2)}s</p>
                    <p>Hallucinations {item.hallucination_count}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      ) : null}

      {view === "system" && dashboard ? (
        <section className="system-layout">
          <div className="panel glass-panel story-panel">
            <div className="panel-head">
              <span className="eyebrow">Integrated Narrative</span>
              <h2>What The Final Project Now Includes</h2>
              <p>The three original workstreams now read as one coherent pipeline rather than separate notebooks.</p>
            </div>

            <div className="timeline-grid">
              <div className="timeline-card">
                <span className="eyebrow">Part 1</span>
                <h3>Data & Processing</h3>
                <p>Corpus cleaning, chunking strategies, OCR comparison, and documented preprocessing outputs.</p>
              </div>
              <div className="timeline-card">
                <span className="eyebrow">Part 2</span>
                <h3>Retrieval Experiments</h3>
                <p>Chunking, embedding, and Top-K comparisons to identify strong retrieval settings and deprioritize reranking.</p>
              </div>
              <div className="timeline-card">
                <span className="eyebrow">Part 3</span>
                <h3>Generation Evaluation</h3>
                <p>Prompt strategy rerun against answers.json with refreshed summary outputs and presentation-ready figures.</p>
              </div>
              <div className="timeline-card emphasis-card">
                <span className="eyebrow">Integration</span>
                <h3>System Deliverable</h3>
                <p>One repository, one frontend, one backend, one dashboard, and one demo flow for presentation and report use.</p>
              </div>
            </div>
          </div>

          <div className="system-grid">
            <div className="panel glass-panel next-panel">
              <div className="panel-head">
                <span className="eyebrow">Demo Configuration</span>
                <h2>Current System Configuration</h2>
              </div>
              <div className="note-stack">
                <div className="note-card">
                  <strong>Chunking selector</strong>
                  <p>The demo can switch between fixed, recursive, and section-aware chunking using the existing experiment assets.</p>
                </div>
                <div className="note-card">
                  <strong>Retrieval and generation</strong>
                  <p>The current deployment uses local retrieval and upgrades generation through an OpenAI-compatible endpoint when credentials are present.</p>
                </div>
                <div className="note-card">
                  <strong>Evaluation artifacts</strong>
                  <p>The results board reads from the rerun evaluation summary generated against answers.json.</p>
                </div>
              </div>
            </div>

            <div className="panel glass-panel">
              <div className="panel-head">
                <span className="eyebrow">Best Findings</span>
                <h2>Current Best-Performing Setup</h2>
              </div>
              <div className="note-stack">
                <div className="note-card">
                  <strong>Embedding model</strong>
                  <p>`BAAI/bge-base-en-v1.5` remains the strongest retrieval backbone in the current comparison results.</p>
                </div>
                <div className="note-card">
                  <strong>Retrieval default</strong>
                  <p>The integrated demo defaults to section-aware chunking and Top-K evidence display because that is the most presentation-stable setting.</p>
                </div>
                <div className="note-card">
                  <strong>Prompt comparison</strong>
                  <p>The prompt strategy summary shown on the results page is sourced from the rerun Part 3 evaluation outputs.</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      ) : null}
    </div>
  );
}
