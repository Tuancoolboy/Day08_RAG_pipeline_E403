"""Task 10 — RAG generation with citations."""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv

try:
    from .common import tokenize
    from .task9_retrieval_pipeline import retrieve
except ImportError:  # pragma: no cover
    from common import tokenize
    from task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3


SYSTEM_PROMPT = """Answer in Vietnamese using only the provided context.
Every factual claim must include a citation like [Source, Year].
If the context is insufficient, say: Tôi không thể xác minh thông tin này từ nguồn hiện có."""


def reorder_for_llm(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Place strong chunks at the beginning and end to reduce lost-in-the-middle."""

    if len(chunks) <= 2:
        return list(chunks)

    reordered: list[dict[str, Any]] = []
    for index in range(0, len(chunks), 2):
        reordered.append(chunks[index])

    last_even = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for index in range(last_even, 0, -2):
        reordered.append(chunks[index])
    return reordered


def _citation(metadata: dict[str, Any]) -> str:
    source = metadata.get("title") or metadata.get("source") or metadata.get("filename") or "Nguồn"
    year = metadata.get("year")
    if not year:
        text = " ".join(str(value) for value in metadata.values())
        for token in text.replace("-", " ").split():
            if token.isdigit() and len(token) == 4:
                year = token
                break
    return f"[{source}, {year or 'n.d.'}]"


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks with source labels for citation."""

    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        parts.append(
            f"[Document {index} | Source: {metadata.get('source', 'unknown')} | "
            f"Title: {metadata.get('title', 'unknown')} | Type: {metadata.get('type', 'unknown')}]\n"
            f"{chunk.get('content', '').strip()}"
        )
    return "\n\n---\n\n".join(parts)


def _local_generate(query: str, chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    sentences: list[str] = []
    query_terms = set(tokenize(query))
    for chunk in chunks[:3]:
        content = _clean_content(chunk.get("content", ""))
        if not content:
            continue
        candidate_sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", content)
            if len(sentence.strip()) >= 40
        ]
        if not candidate_sentences:
            candidate_sentences = [content[:260].strip()]
        first_sentence = max(
            candidate_sentences,
            key=lambda sentence: len(query_terms & set(tokenize(sentence))),
        )
        citation = _citation(chunk.get("metadata") or {})
        sentences.append(f"{first_sentence.rstrip('.')}. {citation}")

    if not sentences:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    return " ".join(sentences)


def _clean_content(content: str) -> str:
    lines: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line == "---":
            continue
        if line.startswith("#"):
            continue
        if line.startswith("**") and line.endswith("**"):
            continue
        if line.lower().startswith(("title:", "source url:", "year:", "**source file:**", "**url:**", "**source:**", "**crawled:**")):
            continue
        lines.append(line)
    return " ".join(lines)


def _openai_generate(query: str, chunks: list[dict[str, Any]]) -> str | None:
    """Optional OpenAI call, disabled unless INDIVIDUAL_USE_OPENAI=1."""

    if os.getenv("INDIVIDUAL_USE_OPENAI") != "1" or not os.getenv("OPENAI_API_KEY"):
        return None

    from openai import OpenAI

    context = format_context(reorder_for_llm(chunks))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")))
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content or None


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict[str, Any]:
    """End-to-end retrieval + generation with citation."""

    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)

    try:
        answer = _openai_generate(query, reordered) or _local_generate(query, reordered)
    except Exception:
        answer = _local_generate(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "context": format_context(reordered),
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
        "generation_mode": "openai" if os.getenv("INDIVIDUAL_USE_OPENAI") == "1" else "local_fallback",
    }


if __name__ == "__main__":
    for question in [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy?",
        "Khi đọc tin nghệ sĩ liên quan ma túy cần kiểm chứng gì?",
    ]:
        result = generate_with_citation(question)
        print(f"\nQ: {question}\nA: {result['answer']}")
