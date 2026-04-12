from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .data_loader import (
    ask_paper,
    build_dashboard_payload,
    get_paper_detail,
    get_papers,
    get_sample_questions,
)


app = FastAPI(title="Academic RAG Studio API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskPayload(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard() -> dict:
    return build_dashboard_payload()


@app.get("/api/papers")
def papers(query: str | None = None, limit: int = 40) -> dict:
    return {"items": get_papers(query=query, limit=limit)}


@app.get("/api/papers/{paper_id}")
def paper_detail(paper_id: str) -> dict:
    try:
        return get_paper_detail(paper_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}") from error


@app.get("/api/papers/{paper_id}/questions")
def paper_questions(paper_id: str, limit: int = 6) -> dict:
    return {"items": get_sample_questions(paper_id, limit=limit)}


@app.post("/api/papers/{paper_id}/ask")
def paper_ask(paper_id: str, payload: AskPayload) -> dict:
    try:
        return ask_paper(paper_id=paper_id, question=payload.question, top_k=payload.top_k)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}") from error
