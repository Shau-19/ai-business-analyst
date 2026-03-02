# auth.py - API Key Auth + Rate Limiting
# Drop this file next to main.py. Zero changes to existing logic.
#
# Requires:
#   pip install slowapi
#   Add to .env:  API_KEYS=key1,key2,key3

import os
from fastapi import Header, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

# ── Rate limiter ─────────────────────────────────────────────────────
# Uses client IP as the key. Shared instance imported by main.py.
limiter = Limiter(key_func=get_remote_address)

# ── API key auth ─────────────────────────────────────────────────────
# Keys stored in .env as comma-separated string: API_KEYS=abc123,xyz789
# Header expected: X-Api-Key: <key>

def _load_keys() -> set:
    raw = os.getenv("API_KEYS", "")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    if not keys:
        # Dev fallback — warns loudly so you don't forget to set keys
        import warnings
        warnings.warn("⚠️  API_KEYS not set in .env — auth is DISABLED", stacklevel=2)
    return keys

VALID_KEYS = _load_keys()

async def verify_key(x_api_key: str = Header(default=None, alias="X-Api-Key")):
    """FastAPI dependency — raises 401 if key missing or invalid."""
    if not VALID_KEYS:
        return  # Auth disabled (dev mode — API_KEYS not set)
    if not x_api_key or x_api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")