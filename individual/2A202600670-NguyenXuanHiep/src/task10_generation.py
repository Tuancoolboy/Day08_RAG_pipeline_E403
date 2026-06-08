"""
Task 10 — Generation Có Citation.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env
for _p in [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent.parent.parent / ".env",
]:
    if _p.exists():
        load_dotenv(_p, override=True)
        break
else:
    load_dotenv()

from .task9_retrieval_pipeline import retrieve

# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K       = 5
TOP_P       = 0.9
TEMPERATURE = 0.3

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY",    "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY",  "")

MISTRAL_MODELS = [
    "mistral-small-latest",
    "open-mistral-7b",
]

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về luật phòng chống ma tuý Việt Nam.
Hãy trả lời câu hỏi bằng tiếng Việt, dựa HOÀN TOÀN vào context được cung cấp.

Quy tắc bắt buộc:
1. Mọi thông tin đều PHẢI có citation dạng [Document X | tên nguồn]
2. Nếu context không đủ → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
3. KHÔNG bịa đặt thông tin ngoài context
4. Cấu trúc câu trả lời rõ ràng, có đoạn văn
"""


# =============================================================================
# HELPERS
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    if len(chunks) <= 2:
        return chunks
    return chunks[::2] + chunks[1::2][::-1]


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        parts.append(
            f"[Document {i} | {meta.get('source', f'Source {i}')} "
            f"| type={meta.get('type','unknown')} "
            f"| score={chunk.get('score', 0.0):.4f}]\n"
            f"{chunk['content'].strip()}"
        )
    return "\n\n---\n\n".join(parts)


def _truncate_prompt(prompt: str, max_chars: int = 6000) -> str:
    """Cắt context nếu quá dài — tránh Groq 413."""
    if len(prompt) <= max_chars:
        return prompt

    q_marker = "Câu hỏi:"
    q_pos    = prompt.rfind(q_marker)

    if q_pos > 0:
        question  = prompt[q_pos:]
        allowed   = max_chars - len(question) - 50
        truncated = prompt[:allowed] + "\n\n[... truncated ...]\n\n"
        result    = truncated + question
    else:
        result = prompt[:max_chars]

    print(f"    [Truncate] {len(prompt)} → {len(result)} chars")
    return result


# =============================================================================
# LLM CALLERS
# =============================================================================

def _call_mistral(prompt: str) -> str:
    """Mistral SDK v2.x"""
    # ✅ Import trực tiếp, không dùng cache
    if "mistralai" in sys.modules:
        del sys.modules["mistralai"]

    import importlib
    mistralai = importlib.import_module("mistralai")

    print(f"    [Mistral] SDK path: {mistralai.__file__}")
    print(f"    [Mistral] Available: {[x for x in dir(mistralai) if not x.startswith('_')]}")

    if not hasattr(mistralai, "Mistral"):
        raise ImportError(
            f"mistralai.Mistral không tồn tại.\n"
            f"Available: {[x for x in dir(mistralai) if not x.startswith('_')]}"
        )

    client     = mistralai.Mistral(api_key=MISTRAL_API_KEY)
    last_error = None

    for model in MISTRAL_MODELS:
        try:
            print(f"    [Mistral] Trying: {model} ...")
            resp = client.chat.complete(
                model    = model,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature = TEMPERATURE,
                top_p       = TOP_P,
                max_tokens  = 1024,
            )
            print(f"    [Mistral] ✓ {model} OK")
            return resp.choices[0].message.content.strip()

        except Exception as e:
            err = str(e)
            if any(x in err for x in ["429", "rate", "quota", "limit", "404"]):
                print(f"    [Mistral] ⚠ {model}: {err[:80]}")
                last_error = e
                continue
            raise

    raise RuntimeError(f"Mistral thất bại: {last_error}")


def _call_groq(prompt: str) -> str:
    """Groq API với truncate + retry."""
    from groq import Groq

    prompt     = _truncate_prompt(prompt, max_chars=6000)
    client     = Groq(api_key=GROQ_API_KEY)
    last_error = None

    for model in GROQ_MODELS:
        for attempt in range(2):
            try:
                print(f"    [Groq] {model} (attempt {attempt+1}) ...")
                resp = client.chat.completions.create(
                    model       = model,
                    messages    = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature = TEMPERATURE,
                    top_p       = TOP_P,
                    max_tokens  = 512,
                )
                print(f"    [Groq] ✓ {model} OK")
                return resp.choices[0].message.content.strip()

            except Exception as e:
                err = str(e)
                if "decommissioned" in err or "404" in err:
                    print(f"    [Groq] ✗ {model} decommissioned → skip")
                    break
                if any(x in err for x in ["429", "413", "rate", "quota", "limit"]):
                    wait = 5 * (attempt + 1)
                    print(f"    [Groq] ⚠ {err[:60]} → sleep {wait}s ...")
                    time.sleep(wait)
                    last_error = e
                    continue
                raise

    raise RuntimeError(f"Groq thất bại: {last_error}")


def _call_gemini(prompt: str) -> str:
    """Gemini API fallback."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    models     = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash-8b"]
    last_error = None

    for model_name in models:
        try:
            print(f"    [Gemini] Trying: {model_name} ...")
            m    = genai.GenerativeModel(
                model_name         = model_name,
                system_instruction = SYSTEM_PROMPT,
                generation_config  = genai.GenerationConfig(
                    temperature=TEMPERATURE, top_p=TOP_P, max_output_tokens=1024
                ),
            )
            resp = m.generate_content(prompt)
            print(f"    [Gemini] ✓ {model_name} OK")
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if any(x in err for x in ["429", "404", "quota", "not found"]):
                print(f"    [Gemini] ⚠ {model_name}: {err[:60]}")
                last_error = e
                continue
            raise

    raise RuntimeError(f"Gemini thất bại: {last_error}")


def _call_llm(prompt: str) -> str:
    """Mistral → Groq → Gemini."""
    if MISTRAL_API_KEY:
        try:
            return _call_mistral(prompt)
        except Exception as e:
            print(f"  ⚠ Mistral: {e} → thử Groq ...")

    if GROQ_API_KEY:
        try:
            return _call_groq(prompt)
        except Exception as e:
            print(f"  ⚠ Groq: {e} → thử Gemini ...")

    if GEMINI_API_KEY:
        return _call_gemini(prompt)

    raise ValueError("Cần MISTRAL_API_KEY hoặc GROQ_API_KEY hoặc GEMINI_API_KEY")


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    print(f"  [Debug] MISTRAL : {'SET ✓' if MISTRAL_API_KEY else 'EMPTY ✗'}")
    print(f"  [Debug] GROQ    : {'SET ✓' if GROQ_API_KEY    else 'EMPTY ✗'}")
    print(f"  [Debug] GEMINI  : {'SET ✓' if GEMINI_API_KEY  else 'EMPTY ✗'}")

    print(f"  [1/4] Retrieving top {top_k} chunks ...")
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer"          : "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources"         : [],
            "retrieval_source": "none",
        }

    retrieval_source = chunks[0].get("source", "hybrid")
    reordered        = reorder_for_llm(chunks)
    context          = format_context(reordered)
    user_message     = f"Context:\n\n{context}\n\n{'─'*60}\n\nCâu hỏi: {query}"

    print(f"  [4/4] Calling LLM ...")
    answer = _call_llm(user_message)
    print(f"  ✓ Generated {len(answer)} ký tự.")

    return {
        "answer"          : answer,
        "sources"         : chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    result = generate_with_citation("Hình phạt tàng trữ ma tuý?")
    print(f"\nA:\n{result['answer']}")