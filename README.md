# Academic RAG Studio

Academic RAG Studio is the final integrated repository for the DSAI5201 project on lightweight academic paper question answering. The project studies academic PDF QA as an end-to-end retrieval-augmented generation (RAG) pipeline problem, with emphasis on preprocessing, chunking, retrieval configuration, grounded answering, and final system integration.

The repository consolidates the outputs of three experimental workstreams and turns them into one unified deliverable:

- a documented data-processing and chunking pipeline
- retrieval ablation results across chunking strategies, embedding models, and Top-K settings
- generation and evaluation outputs from the rerun Part 3 pipeline
- an integrated FastAPI backend and React frontend for live demonstration

## Project Scope

The system is designed for lightweight academic paper QA rather than full-scale model training. Instead of training a new language model from scratch, the project evaluates how design choices within the RAG pipeline affect retrieval quality and answer faithfulness. The final application supports:

- benchmark paper selection and question answering
- retrieved evidence display
- benchmark chunking selection
- uploaded single-PDF question answering
- session history
- experiment dashboards for report and presentation use

## Repository Structure

- `materials/`
  Consolidated source materials from the three original project parts.
- `backend/`
  FastAPI application serving paper, dashboard, upload, and QA endpoints.
- `frontend/`
  React + Vite application for the integrated demo interface.
- `docs/`
  Final report, project notes, and supplementary submission assets.

## Included Assets

This repository includes the code and lightweight deliverables required to reproduce the final project structure:

- processed corpus statistics
- retrieval CSV summaries
- prompt-evaluation summaries and plots
- frontend and backend application code
- final report and presentation materials

Large raw datasets and local cache artifacts are intentionally excluded from normal Git tracking. See [docs/data-assets.md](./docs/data-assets.md) for details.

## Local Development

### Backend

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

By default, the frontend proxies `/api/*` requests to `http://localhost:8000`.

## Optional Model-Backed Generation

The integrated demo supports API-based answer generation through an OpenAI-compatible endpoint. If no model API is configured, the system falls back to a local grounded-answer mode so that the project remains runnable for testing and presentation.

Recommended environment variables for model-backed generation:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_MODEL=...
```

## Deployment Note

For live presentation, the frontend can be exposed through a fixed Cloudflare Tunnel hostname while the backend remains local. This makes it possible to share a stable public URL during demonstrations without permanently hosting the application.

## Final Deliverable Context

This repository is intended as the formal submission version of the project. It reflects the final integrated state of the codebase, experiment outputs, documentation, and presentation materials used for the course deliverables.
