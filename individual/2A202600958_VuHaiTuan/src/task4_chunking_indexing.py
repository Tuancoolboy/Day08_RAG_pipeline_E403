"""Task 4 — Chunk markdown documents and build a local vector store."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

try:
    from .common import (
        STANDARDIZED_DIR,
        VECTOR_INDEX_PATH,
        ensure_dir,
        hashed_embedding,
        read_json,
        write_json,
    )
    from .task3_convert_markdown import convert_all
except ImportError:  # pragma: no cover
    from common import (
        STANDARDIZED_DIR,
        VECTOR_INDEX_PATH,
        ensure_dir,
        hashed_embedding,
        read_json,
        write_json,
    )
    from task3_convert_markdown import convert_all


# Recursive character chunking is stable for mixed legal/news markdown. 800 chars
# keeps context readable for citation while 120 chars overlap preserves article
# and legal references split near chunk boundaries.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "recursive_character"

# Deterministic local embedding. It avoids model downloads in tests but keeps the
# same vector-store workflow: embed chunks, persist vectors, cosine search later.
EMBEDDING_MODEL = "local-hashed-tfidf-384"
EMBEDDING_DIM = 384
VECTOR_STORE = "local_json"


def _bootstrap_markdown() -> None:
    if not STANDARDIZED_DIR.exists() or not list(STANDARDIZED_DIR.rglob("*.md")):
        convert_all()


def _doc_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip() or fallback
    return fallback


def _doc_year(content: str, fallback: str = "") -> str:
    for pattern in (r"Year:\s*(\d{4})", r"\b(20\d{2}|19\d{2})\b"):
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    return fallback


def load_documents() -> list[dict[str, Any]]:
    """Read markdown files from `data/standardized/`."""

    _bootstrap_markdown()
    documents: list[dict[str, Any]] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in path.parts else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "id": path.stem,
                    "source": str(path.relative_to(STANDARDIZED_DIR)),
                    "filename": path.name,
                    "title": _doc_title(content, path.stem),
                    "year": _doc_year(content),
                    "type": doc_type,
                },
            }
        )
    return documents


def _split_long_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return [chunk for chunk in chunks if chunk]


def _recursive_split(text: str) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > CHUNK_SIZE:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_text(paragraph))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            chunks.append(current.strip())
            overlap_text = current[-CHUNK_OVERLAP:].strip()
            current = f"{overlap_text}\n\n{paragraph}".strip()
            if len(current) > CHUNK_SIZE:
                chunks.extend(_split_long_text(current))
                current = ""

    if current:
        chunks.append(current.strip())
    return [chunk[:CHUNK_SIZE] for chunk in chunks if chunk.strip()]


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split documents into overlapping chunks."""

    chunks: list[dict[str, Any]] = []
    for document in documents:
        for index, text in enumerate(_recursive_split(document["content"])):
            chunks.append(
                {
                    "content": text,
                    "metadata": {
                        **document["metadata"],
                        "chunk_index": index,
                        "chunking": CHUNKING_METHOD,
                    },
                }
            )
    return chunks


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach deterministic local embeddings to chunks."""

    embedded: list[dict[str, Any]] = []
    for chunk in chunks:
        item = {**chunk, "embedding": hashed_embedding(chunk["content"], EMBEDDING_DIM)}
        embedded.append(item)
    return embedded


def index_to_vectorstore(chunks: list[dict[str, Any]]) -> Path:
    """Persist chunks and embeddings to a local JSON vector store."""

    payload = {
        "vector_store": VECTOR_STORE,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "built_at_unix": int(time.time()),
        "record_count": len(chunks),
        "records": chunks,
    }
    write_json(VECTOR_INDEX_PATH, payload)
    return VECTOR_INDEX_PATH


def build_index(force: bool = False) -> dict[str, Any]:
    """Build or load the local vector index."""

    if VECTOR_INDEX_PATH.exists() and not force:
        existing = read_json(VECTOR_INDEX_PATH, default={})
        if existing and existing.get("records"):
            return existing

    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded = embed_chunks(chunks)
    index_to_vectorstore(embedded)
    return read_json(VECTOR_INDEX_PATH, default={"records": []})


def run_pipeline(force: bool = True) -> Path:
    """Run load -> chunk -> embed -> index."""

    ensure_dir(VECTOR_INDEX_PATH.parent)
    build_index(force=force)
    return VECTOR_INDEX_PATH


if __name__ == "__main__":
    path = run_pipeline()
    index = read_json(path, default={})
    print(f"Indexed {index.get('record_count', 0)} chunks into {path}")
