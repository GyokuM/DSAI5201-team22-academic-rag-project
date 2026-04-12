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
  "BAAI/bge-base-en-v1.5": "#f97316",
  "all-MiniLM-L6-v2": "#0f766e"
};

function StatCard({ label, value, detail }) {
  return (
    <div className="stat-card">
      <span className="eyebrow">{label}</span>
      <strong>{value}</strong>
      {detail ? <p>{detail}</p> : null}
    </div>
  );
}

function SectionCard({ item, index }) {
  return (
    <article className="evidence-card">
      <div className="evidence-head">
        <span>Source {index + 1}</span>
        <span>{item.label}</span>
        {"score" in item ? <span className="score-pill">{item.score}</span> : null}
      </div>
      <p>{item.snippet}</p>
    </article>
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
  const [demoResult, setDemoResult] = useState(null);
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
    const rows = dashboard.retrieval_rows;
    const grouped = new Map();
    rows.forEach((row) => {
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
    const nextQuestion = (questionOverride ?? question).trim();
    if (!selectedPaperId || !nextQuestion) return;
    setLoadingAsk(true);
    setError("");
    try {
      const result = await api.askPaper(selectedPaperId, { question: nextQuestion, top_k: topK });
      setDemoResult(result);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoadingAsk(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="background-orb orb-a" />
      <div className="background-orb orb-b" />

      <header className="hero">
        <div>
          <p className="hero-kicker">DSAI5201 • Integrated Deliverable</p>
          <h1>Academic RAG Studio</h1>
          <p className="hero-copy">
            A cleaner project shell that turns the three notebook-based parts into one demo,
            one dashboard, and one report-ready story.
          </p>
        </div>
        <div className="hero-badges">
          <span>Part 1: Pipeline</span>
          <span>Part 2: Retrieval</span>
          <span>Part 3: Evaluation</span>
        </div>
      </header>

      <nav className="view-switcher">
        {[
          ["demo", "Paper Demo"],
          ["results", "Results Board"],
          ["system", "Integration Notes"]
        ].map(([key, label]) => (
          <button
            key={key}
            className={view === key ? "active" : ""}
            onClick={() => setView(key)}
          >
            {label}
          </button>
        ))}
      </nav>

      {error ? <div className="error-banner">{error}</div> : null}

      {view === "demo" ? (
        <section className="panel-grid">
            <div className="panel glass-panel control-panel">
              <div className="panel-head">
                <h2>Ask A Paper</h2>
                <p>Current backend mode: {dashboard?.integration_notes?.current_demo_backend ?? "loading demo backend..."}</p>
              </div>

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
            <select
              value={selectedPaperId}
              onChange={(event) => setSelectedPaperId(event.target.value)}
            >
              {papers.map((paper) => (
                <option key={paper.id} value={paper.id}>
                  {paper.title}
                </option>
              ))}
            </select>

            {paperDetail ? (
              <div className="paper-summary">
                <h3>{paperDetail.title}</h3>
                <p>{paperDetail.abstract.slice(0, 260)}...</p>
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

            <label className="field-label">Question</label>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask about the selected paper"
            />

            <div className="ask-row">
              <label>
                Top-K
                <select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
                  {[1, 2, 3, 4, 5].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <button className="primary-button" onClick={handleAsk} disabled={loadingAsk}>
                {loadingAsk ? "Searching..." : "Run Demo"}
              </button>
            </div>
          </div>

          <div className="panel glass-panel result-panel">
            <div className="panel-head">
              <h2>Grounded Output</h2>
              <p>Embedding retrieval stays local; generation upgrades to Together when `TOGETHER_API_KEY` is set.</p>
            </div>

            {demoResult ? (
              <>
                <div className="answer-card">
                  <span className="eyebrow">{demoResult.mode}</span>
                  <p>{demoResult.answer}</p>
                </div>

                <div className="evidence-list">
                  {demoResult.retrieved_sections?.map((item, index) => (
                    <SectionCard item={item} index={index} key={`${item.section_id}-${index}`} />
                  ))}
                </div>
              </>
            ) : (
              <div className="empty-state">
                <p>Pick a paper, ask a question, and the app will retrieve evidence and draft a grounded summary.</p>
              </div>
            )}
          </div>
        </section>
      ) : null}

      {view === "results" && dashboard ? (
        <section className="results-layout">
          <div className="stats-row">
            <StatCard
              label="Cleaned Sections"
              value={dashboard.chunk_stats.cleaned_sections.toLocaleString()}
              detail="Part 1 processed corpus units"
            />
            <StatCard
              label="Best Hit@15"
              value={`${Math.max(...dashboard.best_retrieval_by_top_k.map((row) => row.hit_rate * 100)).toFixed(1)}%`}
              detail="Top retrieval result from Part 2"
            />
            <StatCard
              label="Best Embedding"
              value="BAAI/bge-base-en-v1.5"
              detail="Consistently strongest retrieval model"
            />
            <StatCard
              label="Part 3 Status"
              value="Needs rerun"
              detail="Use answers.json as final reference"
            />
          </div>

          <div className="chart-grid">
            <div className="panel glass-panel">
              <div className="panel-head">
                <h2>Best Retrieval By Top-K</h2>
                <p>Best chunking/embedding combination at each retrieval depth.</p>
              </div>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={retrievalChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(20, 24, 28, 0.12)" />
                    <XAxis dataKey="topK" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="hitRate" stroke="#f97316" strokeWidth={3} />
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

            <div className="panel glass-panel">
              <div className="panel-head">
                <h2>Embedding Comparison</h2>
                <p>Best-performing chunking per embedding at each top-k.</p>
              </div>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={embeddingComparison}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(20, 24, 28, 0.12)" />
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

            <div className="panel glass-panel wide-panel">
              <div className="panel-head">
                <h2>Prompt Strategy Snapshot</h2>
                <p>Current Part 3 summary kept visible, but should be rerun cleanly before final submission.</p>
              </div>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={promptSummary}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(20, 24, 28, 0.12)" />
                    <XAxis dataKey="strategy" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="faithfulness_mean" radius={[8, 8, 0, 0]}>
                      {promptSummary.map((entry) => (
                        <Cell
                          key={entry.strategy}
                          fill={entry.strategy === "Naive" ? "#0f766e" : entry.strategy === "Structured" ? "#f97316" : "#111827"}
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
          <div className="panel glass-panel">
            <div className="panel-head">
              <h2>What This Project Already Has</h2>
            </div>
            <div className="note-stack">
              <div className="note-card">
                <span className="eyebrow">Part 1</span>
                <p>Corpus cleaning, chunking ablation inputs, OCR comparison pipeline, and data documentation.</p>
              </div>
              <div className="note-card">
                <span className="eyebrow">Part 2</span>
                <p>Retrieval leaderboard across chunking, embedding, and top-k, with reranker already deprioritized.</p>
              </div>
              <div className="note-card">
                <span className="eyebrow">Part 3</span>
                <p>Prompt strategy comparison has been rerun against answers.json, and the refreshed summary now feeds this dashboard.</p>
              </div>
            </div>
          </div>

          <div className="panel glass-panel">
            <div className="panel-head">
              <h2>What You Can Improve Next</h2>
            </div>
            <ol className="ordered-notes">
              <li>Replace the fallback answer generator with a model-backed generation path.</li>
              <li>Surface retrieved chunk citations and prompt strategy choices directly in the demo panel.</li>
              <li>Add upload-PDF ingestion after the final baseline demo is stable.</li>
              <li>Use this frontend as the report and presentation screenshot source.</li>
            </ol>
          </div>
        </section>
      ) : null}
    </div>
  );
}
