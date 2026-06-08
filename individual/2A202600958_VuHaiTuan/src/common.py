"""Shared helpers for Tuan's individual RAG pipeline.

The individual assignment should run in class without external services, so the
default implementation uses local files plus deterministic hashed embeddings.
Optional online providers can be plugged in later without changing the public
task function contracts used by the tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
STANDARDIZED_DIR = DATA_DIR / "standardized"
INDEX_DIR = DATA_DIR / "index"
VECTOR_INDEX_PATH = INDEX_DIR / "vector_store.json"
PAGEINDEX_MANIFEST_PATH = INDEX_DIR / "pageindex_manifest.json"

TOKEN_RE = re.compile(r"[0-9a-zA-ZÀ-ỹĐđ]+")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(strip_accents(text).lower())


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def hashed_embedding(text: str, dim: int = 384) -> list[float]:
    """Create a deterministic local embedding without model downloads.

    Tokens are hashed into a fixed-size signed bag-of-words vector and then L2
    normalized. It is lightweight but still behaves like a dense vector store for
    semantic-ish similarity in this assignment.
    """

    vector = [0.0] * dim
    counts: dict[str, int] = {}
    for token in tokenize(text):
        counts[token] = counts.get(token, 0) + 1

    for token, count in counts.items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign * (1.0 + math.log(count))

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def result_key(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    return "|".join(
        str(part)
        for part in (
            metadata.get("id"),
            metadata.get("source"),
            metadata.get("chunk_index"),
            item.get("content", "")[:120],
        )
        if part is not None
    )
