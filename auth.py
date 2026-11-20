# auth.py - API Key Auth + Rate Limiting + Daily Quota
#
# .env settings:
#   API_KEYS=key1,key2          Access keys. Unset = open demo (no key needed).
#   DAILY_QUERY_LIMIT=500       Max LLM-backed queries per UTC day, across all
#                               users (protects your Groq credit). 0/unset = no cap.
#
# Keys are NEVER shipped in frontend code: the UI asks the user for a key
# (stored in their browser) only when /auth/check says one is required.

import os
import threading
from datetime import datetime, timezone

from fastapi import Header, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

# ── Rate limiter ─────────────────────────────────────────────────────
# Uses client IP as the key. Shared instance imported by main.py.
limiter = Limiter(key_func=get_remote_address)

# ── API key auth ─────────────────────────────────────────────────────
# Header expected: X-Api-Key: <key>

def _load_keys() -> set:
    raw = os.getenv("API_KEYS", "")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    if not keys:
        import warnings
        warnings.warn("⚠️  API_KEYS not set in .env — auth is DISABLED (open demo mode)", stacklevel=2)
    return keys

VALID_KEYS = _load_keys()


def auth_required() -> bool:
    return bool(VALID_KEYS)


def key_is_valid(key: str | None) -> bool:
    return not VALID_KEYS or (bool(key) and key in VALID_KEYS)


async def verify_key(x_api_key: str = Header(default=None, alias="X-Api-Key")):
    """FastAPI dependency — raises 401 if key missing or invalid."""
    if not key_is_valid(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Daily query quota ────────────────────────────────────────────────
# A global cap on LLM-backed calls per UTC day. Per-IP rate limits alone
# don't bound total spend on a public deployment (many IPs, or one patient
# client). In-memory: resets on restart — fine for a single-process demo.

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default

DAILY_QUERY_LIMIT = _int_env("DAILY_QUERY_LIMIT", 0)

_quota_lock = threading.Lock()
_quota_state = {"day": None, "count": 0}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def quota_status() -> dict:
    with _quota_lock:
        used = _quota_state["count"] if _quota_state["day"] == _today() else 0
    return {
        "limit":     DAILY_QUERY_LIMIT or None,
        "used":      used,
        "remaining": (DAILY_QUERY_LIMIT - used) if DAILY_QUERY_LIMIT else None,
    }


async def enforce_daily_quota():
    """FastAPI dependency — raises 429 once today's query budget is spent."""
    if not DAILY_QUERY_LIMIT:
        return
    with _quota_lock:
        today = _today()
        if _quota_state["day"] != today:
            _quota_state["day"], _quota_state["count"] = today, 0
        if _quota_state["count"] >= DAILY_QUERY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Daily query limit reached for this demo. Please try again tomorrow.",
            )
        _quota_state["count"] += 1
