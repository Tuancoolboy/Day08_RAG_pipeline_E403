"""Task 9 — Complete retrieval pipeline with hybrid search and fallback."""

from __future__ import annotations

from typing import Any

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:  # pragma: no cover
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def _mark_source(results: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    for result in results:
        item = dict(result)
        item["metadata"] = dict(item.get("metadata") or {})
        item["source"] = source
        marked.append(item)
    return marked


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict[str, Any]]:
    """Run semantic + lexical retrieval, RRF merge, rerank and fallback."""

    if top_k <= 0:
        return []

    dense_results = semantic_search(query, top_k=top_k * 3)
    sparse_results = lexical_search(query, top_k=top_k * 3)

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 3)
    merged = _mark_source(merged, "hybrid")

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = _mark_source(final_results, "hybrid")
    else:
        final_results = merged[:top_k]

    if not final_results or float(final_results[0].get("score", 0.0)) < score_threshold:
        return pageindex_search(query, top_k=top_k)

    return final_results[:top_k]


if __name__ == "__main__":
    for question in [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy",
        "Tin nghệ sĩ liên quan ma túy cần kiểm chứng thế nào",
        "Luật phòng chống ma túy quy định gì về cai nghiện",
    ]:
        print(f"\nQuery: {question}")
        for result in retrieve(question, top_k=3):
            print(f"  [{result['score']:.3f}] [{result['source']}] {result['content'][:90]}...")
