# Academic RAG Studio

Academic RAG Studio is the cleaned-up repository workspace for the DSAI5201 academic paper QA project.
It combines the useful outputs from three parallel workstreams and wraps them in one integrated app structure.

## Repository Structure

- `materials/`
  Copied-in working materials from the original three project parts.
- `backend/`
  FastAPI backend that reads the merged materials and serves demo/dashboard APIs.
- `frontend/`
  React + Vite frontend for the paper demo, experiment dashboard, and integration notes.
- `docs/`
  Project notes, including the suggested Git commit roadmap.

## What Is Already Included

- Part 1 pipeline outputs, processed corpus data, and OCR comparison artifacts
- Part 2 retrieval CSV results and cached embedding artifacts
- Part 3 evaluation summary CSV and current figures
- A runnable integrated frontend/backend shell

## Current Limitation

The current demo uses a lightweight lexical retrieval plus extractive summary fallback so the app can run locally without the original Colab/Together setup.

That means the repository is already suitable for:

- showing the merged project structure
- demoing retrieval-grounded interaction
- presenting experiment dashboards

But Part 3 still needs a clean rerun if you want the final model-backed generation path in the app.

## Local Run

### Backend

```bash
cd backend
python3 -m uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend proxies `/api/*` to `http://localhost:8000`.

## Suggested Git Strategy

1. First commit the merged materials
2. Then commit the integrated app shell
3. Then continue with incremental reruns and feature work

See [docs/git-roadmap.md](./docs/git-roadmap.md) for the recommended commit sequence.
