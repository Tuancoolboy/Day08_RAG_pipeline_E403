"""Task 3 — Convert raw legal/news files to markdown."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from markitdown import MarkItDown
except Exception:  # pragma: no cover - optional dependency
    MarkItDown = None

try:
    from .common import LANDING_DIR, STANDARDIZED_DIR, ensure_dir, read_text_file
    from .task1_collect_legal_docs import collect_legal_docs
    from .task2_crawl_news import write_offline_articles
except ImportError:  # pragma: no cover
    from common import LANDING_DIR, STANDARDIZED_DIR, ensure_dir, read_text_file
    from task1_collect_legal_docs import collect_legal_docs
    from task2_crawl_news import write_offline_articles


OUTPUT_DIR = STANDARDIZED_DIR


def _convert_with_markitdown(path: Path) -> str:
    if MarkItDown is None:
        return read_text_file(path)
    try:
        result = MarkItDown().convert(str(path))
        return result.text_content or read_text_file(path)
    except Exception:
        return read_text_file(path)


def convert_legal_docs() -> list[Path]:
    """Convert raw PDF/DOC/DOCX legal files to markdown.

    The offline source fixtures are UTF-8 text files with legal file extensions;
    real PDF/DOCX files are passed through MarkItDown when available.
    """

    legal_dir = LANDING_DIR / "legal"
    if not legal_dir.exists() or not list(legal_dir.glob("*")):
        collect_legal_docs()

    output_dir = ensure_dir(OUTPUT_DIR / "legal")
    converted: list[Path] = []
    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        content = _convert_with_markitdown(path).strip()
        output_path = output_dir / f"{path.stem}.md"
        output_path.write_text(
            f"# {path.stem.replace('-', ' ').title()}\n\n"
            f"**Source file:** {path.name}\n\n---\n\n{content}\n",
            encoding="utf-8",
        )
        converted.append(output_path)
    return converted


def convert_news_articles() -> list[Path]:
    """Convert crawled JSON article files to markdown."""

    news_dir = LANDING_DIR / "news"
    if not news_dir.exists() or len(list(news_dir.glob("*.json"))) < 5:
        write_offline_articles()

    output_dir = ensure_dir(OUTPUT_DIR / "news")
    converted: list[Path] = []
    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        title = data.get("title") or path.stem
        content = data.get("content_markdown") or data.get("markdown") or data.get("content") or ""
        output_path = output_dir / f"{path.stem}.md"
        output_path.write_text(
            f"# {title}\n\n"
            f"**URL:** {data.get('url', 'N/A')}\n\n"
            f"**Source:** {data.get('source_name', 'N/A')}\n\n"
            f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n"
            f"---\n\n{content.strip()}\n",
            encoding="utf-8",
        )
        converted.append(output_path)
    return converted


def convert_all() -> list[Path]:
    """Convert all raw sources and return output markdown paths."""

    ensure_dir(OUTPUT_DIR)
    return convert_legal_docs() + convert_news_articles()


if __name__ == "__main__":
    files = convert_all()
    print(f"Converted {len(files)} files to {OUTPUT_DIR}")
