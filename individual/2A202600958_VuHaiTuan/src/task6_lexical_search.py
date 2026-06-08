"""Task 6 — Lexical search using a local BM25 implementation."""

from __future__ import annotations

import math
from typing import Any

try:
    from .common import tokenize
    from .task4_chunking_indexing import build_index
except ImportError:  # pragma: no cover
    from common import tokenize
    from task4_chunking_indexing import build_index


def _load_corpus() -> list[dict[str, Any]]:
    return list(build_index(force=False).get("records", []))


def build_bm25_index(corpus: list[dict[str, Any]]) -> dict[str, Any]:
    """Build enough BM25 statistics for deterministic local search."""

    tokenized = [tokenize(item["content"]) for item in corpus]
    doc_freq: dict[str, int] = {}
    for tokens in tokenized:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    avgdl = sum(len(tokens) for tokens in tokenized) / max(len(tokenized), 1)
    return {"tokenized": tokenized, "doc_freq": doc_freq, "avgdl": avgdl, "n": len(corpus)}


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], index: dict[str, Any]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    k1 = 1.5
    b = 0.75
    n_docs = max(index["n"], 1)
    avgdl = index["avgdl"] or 1.0
    doc_len = len(doc_tokens)
    frequencies: dict[str, int] = {}
    for token in doc_tokens:
        frequencies[token] = frequencies.get(token, 0) + 1

    score = 0.0
    for token in query_tokens:
        tf = frequencies.get(token, 0)
        if tf == 0:
            continue
        df = index["doc_freq"].get(token, 0)
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
        score += idf * numerator / denominator
    return score


def lexical_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Search chunks using BM25 lexical relevance."""

    if top_k <= 0:
        return []

    corpus = _load_corpus()
    index = build_bm25_index(corpus)
    query_tokens = tokenize(query)

    results: list[dict[str, Any]] = []
    for item, doc_tokens in zip(corpus, index["tokenized"]):
        score = _bm25_score(query_tokens, doc_tokens, index)
        if score <= 0:
            continue
        results.append(
            {
                "content": item["content"],
                "score": round(float(score), 6),
                "metadata": dict(item.get("metadata") or {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in lexical_search("Điều 249 tàng trữ trái phép chất ma túy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
