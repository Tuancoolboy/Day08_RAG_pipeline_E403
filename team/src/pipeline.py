"""End-to-end team RAG pipeline."""

from __future__ import annotations

from typing import Any

from .generation import generate_with_openai
from .retrieval import retrieve


DEFAULT_TOP_K = 5


def _citation(metadata: dict[str, Any]) -> str:
    source = metadata.get("source") or metadata.get("title") or "Unknown source"
    year = metadata.get("year") or "n.d."
    return f"[{source}, {year}]"


def _local_answer(query: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "I cannot verify this information."

    top_sources = sources[:2]
    paragraphs = []
    for source in top_sources:
        content = source["content"].strip()
        citation = _citation(source.get("metadata") or {})
        paragraphs.append(f"{content} {citation}")

    return "\n\n".join(paragraphs)


def answer_question(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    use_openai: bool = True,
    retrieval_mode: str = "hybrid",
    use_reranking: bool = True,
) -> dict[str, Any]:
    """Run retrieval and generation for the chatbot UI and evaluation."""
    sources = retrieve(
        query=query,
        top_k=top_k,
        mode=retrieval_mode,
        use_reranking=use_reranking,
    )

    if not sources:
        return {
            "answer": "I cannot verify this information.",
            "sources": [],
            "metadata": {
                "mode": "no_context",
                "retrieval_mode": retrieval_mode,
                "use_reranking": use_reranking,
            },
        }

    if use_openai:
        result = generate_with_openai(query=query, sources=sources)
        error = result.get("metadata", {}).get("error")
        if not error:
            result["metadata"].update(
                {
                    "mode": "openai",
                    "retrieval_mode": retrieval_mode,
                    "use_reranking": use_reranking,
                }
            )
            return result

    return {
        "answer": _local_answer(query, sources),
        "sources": sources,
        "metadata": {
            "mode": "local_fallback",
            "retrieval_mode": retrieval_mode,
            "use_reranking": use_reranking,
        },
    }


SAMPLE_QUESTIONS = [
    "Điều 249 quy định gì về tội tàng trữ trái phép chất ma túy?",
    "Các hình thức cai nghiện ma túy theo Luật Phòng, chống ma túy 2021 là gì?",
    "Điều 251 quy định khung hình phạt cơ bản cho mua bán trái phép chất ma túy thế nào?",
    "Khi đọc tin nghệ sĩ liên quan đến ma túy cần kiểm chứng nguồn tin ra sao?",
]


RETRIEVAL_MODES = [
    "hybrid_vector",
    "vector",
    "hybrid",
    "dense_only",
    "lexical_only",
]
