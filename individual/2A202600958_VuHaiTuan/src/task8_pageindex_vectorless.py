"""Task 8 — PageIndex vectorless RAG fallback.

The real PageIndex SDK requires an external account. This module exposes the
same function contract and provides a local vectorless fallback based on document
structure and lexical overlap, so the individual pipeline remains demonstrable
without credentials.
"""

from __future__ import annotations

from typing import Any

try:
    from .common import PAGEINDEX_MANIFEST_PATH, STANDARDIZED_DIR, tokenize, write_json
    from .task4_chunking_indexing import chunk_documents, load_documents
except ImportError:  # pragma: no cover
    from common import PAGEINDEX_MANIFEST_PATH, STANDARDIZED_DIR, tokenize, write_json
    from task4_chunking_indexing import chunk_documents, load_documents


def upload_documents() -> dict[str, Any]:
    """Create a local manifest representing uploaded PageIndex documents."""

    documents = load_documents()
    manifest = {
        "provider": "local_pageindex_fallback",
        "source_dir": str(STANDARDIZED_DIR),
        "document_count": len(documents),
        "documents": [document["metadata"] for document in documents],
    }
    write_json(PAGEINDEX_MANIFEST_PATH, manifest)
    return manifest


def _vectorless_score(query: str, text: str, title: str = "") -> float:
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    text_terms = set(tokenize(f"{title} {text}"))
    overlap = len(query_terms & text_terms) / len(query_terms)
    title_bonus = len(query_terms & set(tokenize(title))) / len(query_terms)
    return overlap + 0.25 * title_bonus


def pageindex_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Vectorless retrieval using local document structure and token overlap."""

    if top_k <= 0:
        return []

    documents = load_documents()
    chunks = chunk_documents(documents)
    results: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        score = _vectorless_score(query, chunk["content"], str(metadata.get("title", "")))
        if score <= 0:
            continue
        results.append(
            {
                "content": chunk["content"],
                "score": round(float(score), 6),
                "metadata": dict(metadata),
                "source": "pageindex",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    upload_documents()
    for result in pageindex_search("hình phạt sử dụng ma túy", top_k=3):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
