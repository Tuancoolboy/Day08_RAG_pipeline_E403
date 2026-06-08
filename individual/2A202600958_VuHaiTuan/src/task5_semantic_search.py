"""Task 5 — Semantic search over the local vector store."""

from __future__ import annotations

from typing import Any

try:
    from .common import VECTOR_INDEX_PATH, cosine_similarity, hashed_embedding, read_json
    from .task4_chunking_indexing import EMBEDDING_DIM, build_index
except ImportError:  # pragma: no cover
    from common import VECTOR_INDEX_PATH, cosine_similarity, hashed_embedding, read_json
    from task4_chunking_indexing import EMBEDDING_DIM, build_index


def _load_records() -> list[dict[str, Any]]:
    index = read_json(VECTOR_INDEX_PATH, default=None)
    if not index or not index.get("records"):
        index = build_index(force=True)
    return list(index.get("records", []))


def semantic_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Return top chunks by cosine similarity between local dense vectors."""

    if top_k <= 0:
        return []

    query_embedding = hashed_embedding(query, EMBEDDING_DIM)
    results: list[dict[str, Any]] = []
    for record in _load_records():
        score = cosine_similarity(query_embedding, record.get("embedding", []))
        if score <= 0:
            continue
        results.append(
            {
                "content": record["content"],
                "score": round(float(score), 6),
                "metadata": dict(record.get("metadata") or {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
