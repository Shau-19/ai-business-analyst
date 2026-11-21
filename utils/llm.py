# utils/llm.py
"""
One place to build chat models, so the provider is a config switch.

.env:
  LLM_PROVIDER=groq | openai      (default: groq)
  GROQ_API_KEY=...                LLM_MODEL=openai/gpt-oss-120b (Groq model id)
  OPENAI_API_KEY=...              OPENAI_MODEL=gpt-4.1-mini

Automatic fallback: when both keys are set, the primary provider falls back to
the other on errors such as rate limits. Groq's free tier allows 200K tokens
per day for gpt-oss-120b, roughly 100-150 questions, after which every query
would otherwise fail until the window resets.
"""

import os
from typing import Optional

from config import settings
from utils.logger import logger

PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_KEY = settings.GROQ_API_KEY


def _groq(temperature: float, max_tokens: Optional[int], reasoning_low: bool):
    from langchain_groq import ChatGroq
    extra = {}
    if reasoning_low and "gpt-oss" in settings.LLM_MODEL.lower():
        extra["model_kwargs"] = {"reasoning_effort": "low"}
    return ChatGroq(api_key=GROQ_KEY, model=settings.LLM_MODEL, temperature=temperature,
                    max_tokens=max_tokens, **extra)


def _openai(temperature: float, max_tokens: Optional[int]):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError("LLM_PROVIDER=openai needs the langchain-openai package: "
                          "pip install langchain-openai==0.2.12") from e
    return ChatOpenAI(api_key=OPENAI_KEY, model=OPENAI_MODEL, temperature=temperature,
                      max_tokens=max_tokens, timeout=60, max_retries=2)


def make_llm(temperature: float = 0.1, max_tokens: Optional[int] = None, reasoning_low: bool = False):
    """
    Chat model for the configured provider, with a cross-provider fallback
    when both API keys are present.

    reasoning_low: for tiny classification calls on reasoning models, keep
    hidden reasoning short so it doesn't consume the whole token budget.
    """
    builders = {
        "groq": (lambda: _groq(temperature, max_tokens, reasoning_low)) if GROQ_KEY else None,
        "openai": (lambda: _openai(temperature, max_tokens)) if OPENAI_KEY else None,
    }
    order = ["openai", "groq"] if PROVIDER == "openai" else ["groq", "openai"]
    available = [p for p in order if builders[p]]
    if not available:
        raise ValueError("No LLM API key configured. Set GROQ_API_KEY or OPENAI_API_KEY in .env.")

    models = []
    for p in available:
        try:
            models.append(builders[p]())
        except ImportError as e:
            logger.warning(f"⚠️  {e}")
    if not models:
        raise ValueError("No LLM provider could be initialised; see the warnings above.")
    primary, rest = models[0], models[1:]
    return primary.with_fallbacks(rest) if rest else primary


def describe() -> str:
    names = {"groq": f"Groq {settings.LLM_MODEL}", "openai": f"OpenAI {OPENAI_MODEL}"}
    order = ["openai", "groq"] if PROVIDER == "openai" else ["groq", "openai"]
    keys = {"groq": bool(GROQ_KEY), "openai": bool(OPENAI_KEY)}
    active = [names[p] for p in order if keys[p]]
    return " → fallback ".join(active) if active else "no provider configured"
