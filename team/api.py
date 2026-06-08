"""FastAPI backend for the Vite RAG chatbot UI."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from team.src.pipeline import DEFAULT_TOP_K, RETRIEVAL_MODES, SAMPLE_QUESTIONS, answer_question
from team.src.vector_store import vector_store_status


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=8)
    retrieval_mode: str = Field("hybrid_vector")
    use_reranking: bool = True
    use_openai: bool = True


class Source(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    metadata: dict[str, Any]


app = FastAPI(title="Day08 RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Health endpoint used by the Vite UI."""
    return {
        "status": "ok",
        "retrieval_modes": RETRIEVAL_MODES,
        "sample_questions": SAMPLE_QUESTIONS,
    }


@app.get("/api/vector-store")
def get_vector_store_status() -> dict[str, Any]:
    """Return persisted local vector database status."""
    return vector_store_status()


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict[str, Any]:
    """Run the RAG chatbot pipeline."""
    retrieval_mode = request.retrieval_mode
    if retrieval_mode not in RETRIEVAL_MODES:
        retrieval_mode = "hybrid_vector"

    result = answer_question(
        query=request.query,
        top_k=request.top_k,
        use_openai=request.use_openai,
        retrieval_mode=retrieval_mode,
        use_reranking=request.use_reranking,
    )
    result.setdefault("metadata", {})
    result["metadata"].setdefault("retrieval_mode", retrieval_mode)
    result["metadata"].setdefault("use_reranking", request.use_reranking)
    return result

