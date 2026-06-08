"""Configuration helpers for the team RAG app."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    def load_dotenv(*args, **kwargs):
        return False


PLACEHOLDER_VALUES = {
    "",
    "sk-xxx",
    "xxx",
    "your-api-key",
    "your_api_key",
    "your_openai_api_key",
}


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    model: str
    timeout_seconds: float


def _clean(value: str | None) -> str:
    return value.strip() if value else ""


def _is_placeholder(value: str) -> bool:
    return value.lower() in PLACEHOLDER_VALUES


def _parse_timeout(value: str | None) -> float:
    if not value:
        return 30.0

    try:
        timeout = float(value)
    except ValueError as exc:
        raise ConfigError("OPENAI_TIMEOUT_SECONDS must be a number.") from exc

    if timeout <= 0:
        raise ConfigError("OPENAI_TIMEOUT_SECONDS must be greater than 0.")

    return timeout


def load_openai_settings() -> OpenAISettings:
    """Load OpenAI settings from .env or process environment."""
    load_dotenv()

    api_key = _clean(os.getenv("OPENAI_API_KEY"))
    if _is_placeholder(api_key):
        raise ConfigError(
            "Missing OPENAI_API_KEY. Copy .env.example to .env and set a real key."
        )

    model = _clean(os.getenv("OPENAI_MODEL")) or "gpt-4o-mini"
    timeout_seconds = _parse_timeout(os.getenv("OPENAI_TIMEOUT_SECONDS"))

    return OpenAISettings(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )

