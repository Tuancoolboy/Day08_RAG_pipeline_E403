"""Task 7 — Reranking module.

Default reranker is a local cross-encoder-style heuristic. It scores the full
query against each candidate using token overlap, phrase hits, source/title hits
and the original retrieval score. This keeps the assignment runnable without an
external Jina/Qwen API while preserving the reranking contract.
"""

from __future__ import annotations

from typing import Any

try:
    from .common import cosine_similarity, hashed_embedding, result_key, tokenize
    from .task4_chunking_indexing import EMBEDDING_DIM
except ImportError:  # pragma: no cover
    from common import cosine_similarity, hashed_embedding, result_key, tokenize
    from task4_chunking_indexing import EMBEDDING_DIM


def _overlap_score(query: str, text: str) -> float:
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    text_terms = set(tokenize(text))
    return len(query_terms & text_terms) / len(query_terms)


def rerank_cross_encoder(query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Local reranker with cross-encoder-like full text features."""

    scored: list[dict[str, Any]] = []
    query_norm = " ".join(tokenize(query))
    query_numbers = {token for token in tokenize(query) if token.isdigit()}

    for candidate in candidates:
        metadata = candidate.get("metadata") or {}
        content = candidate.get("content", "")
        searchable = " ".join(
            [
                str(metadata.get("title", "")),
                str(metadata.get("source", "")),
                str(metadata.get("filename", "")),
                content,
            ]
        )
        overlap = _overlap_score(query, searchable)
        phrase_bonus = 0.15 if query_norm and query_norm in " ".join(tokenize(searchable)) else 0.0
        number_bonus = 0.2 if query_numbers & set(tokenize(searchable)) else 0.0
        original = float(candidate.get("score", 0.0))
        original_norm = original / (1.0 + abs(original))
        score = 0.62 * overlap + 0.23 * original_norm + phrase_bonus + number_bonus

        item = dict(candidate)
        item["metadata"] = dict(metadata)
        item["score"] = round(float(score), 6)
        scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict[str, Any]]:
    """Maximal Marginal Relevance for diversity-aware ranking."""

    remaining = list(range(len(candidates)))
    selected: list[int] = []
    embeddings = [
        candidate.get("embedding") or hashed_embedding(candidate.get("content", ""), EMBEDDING_DIM)
        for candidate in candidates
    ]

    while remaining and len(selected) < top_k:
        best_idx = remaining[0]
        best_score = float("-inf")
        for idx in remaining:
            relevance = cosine_similarity(query_embedding, embeddings[idx])
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(cosine_similarity(embeddings[idx], embeddings[chosen]) for chosen in selected)
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    results: list[dict[str, Any]] = []
    for idx in selected:
        item = dict(candidates[idx])
        item["score"] = round(float(cosine_similarity(query_embedding, embeddings[idx])), 6)
        results.append(item)
    return results


def rerank_rrf(ranked_lists: list[list[dict[str, Any]]], top_k: int = 5, k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion over semantic and lexical result lists."""

    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = result_key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key] = item

    fused: list[dict[str, Any]] = []
    for key, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True):
        item = dict(items[key])
        item["metadata"] = dict(item.get("metadata") or {})
        item["score"] = round(float(score), 6)
        fused.append(item)
    return fused[:top_k]


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict[str, Any]]:
    """Unified reranking interface."""

    if not candidates or top_k <= 0:
        return []
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        return rerank_mmr(hashed_embedding(query, EMBEDDING_DIM), candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 249: Tội tàng trữ trái phép chất ma túy", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ bị nhắc đến trong tin liên quan chất cấm", "score": 0.7, "metadata": {}},
        {"content": "Python programming", "score": 0.4, "metadata": {}},
    ]
    for result in rerank("hình phạt tàng trữ ma túy", dummy_candidates, top_k=2):
        print(f"[{result['score']:.3f}] {result['content']}")
