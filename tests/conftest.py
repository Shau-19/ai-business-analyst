"""
Shared fixtures.

Live-server tests (tests/api, tests/llm) talk to ANALYST_URL and authenticate
with ANALYST_KEY, falling back to the first key in API_KEYS from .env. They
create conversations under the user id "pytest_qa" and delete them afterwards.
"""

import os
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # modules use ./data, ./uploads relative paths

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

BASE = os.getenv("ANALYST_URL", "http://127.0.0.1:8000").rstrip("/")
KEY = os.getenv("ANALYST_KEY") or next(
    (k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()), "")
QA_USER = "pytest_qa"


# ── unit-level fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """A DatabaseManager over a throwaway SQLite file with two tables."""
    from database.db_manager import DatabaseManager
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE upload_aaaa1111_sales (region TEXT, "Unit Price ($)" REAL, units INTEGER, created TEXT);
        INSERT INTO upload_aaaa1111_sales VALUES
            ('North', 10.5, 3, '2024-01-05'), ('South', 20.0, 1, '2024-02-10'),
            ('North', 5.0, 7, '2024-02-11'),  ('East', NULL, 2, '2024-03-01');
        CREATE TABLE upload_bbbb2222_secret (owner TEXT, salary INTEGER);
        INSERT INTO upload_bbbb2222_secret VALUES ('someone else', 999999);
        CREATE TABLE employees (name TEXT, salary INTEGER);
        INSERT INTO employees VALUES ('Engineering lead', 123456);
    """)
    con.commit()
    con.close()
    return DatabaseManager({"type": "sqlite", "path": str(path)})


@pytest.fixture(scope="session")
def messy_csv():
    return FIXTURES / "messy_orders.csv"


def _num(s):
    if pd.isna(s):
        return None
    s = (str(s).strip().lower().replace("usd", "").replace("$", "").replace(",", "")
         .replace("units", "").replace("%", "").strip())
    if s in ("", "n/a", "none", "-"):
        return None
    try:
        return float(s.split("/")[0])
    except ValueError:
        return None


@pytest.fixture(scope="session")
def messy_expected(messy_csv):
    """
    Answers a careful analyst would compute by hand from the messy file,
    written independently of tools/data_clean.py so the two can disagree.
    """
    raw = pd.read_csv(messy_csv, dtype=str, keep_default_na=False)
    ref = raw.apply(lambda c: c.str.strip()).drop_duplicates()
    typos = {"Nrth": "North", "Sout": "South", "Est": "East", "Wst": "West", "Centrl": "Central"}
    region = ref["Region"].str.title().replace(typos).replace("", None)
    units = ref["Units"].map(_num)
    price = ref["Unit Price ($)"].map(_num)
    disc_raw = ref["Discount"]
    disc = [0.0 if str(s).strip().lower() in ("", "none") else
            (lambda v: None if v is None else (v / 100 if ("%" in str(s) or v > 1) else v))(_num(s))
            for s in disc_raw]
    rating = ref["Customer Rating"].map(_num)
    dates = pd.to_datetime(ref["Order Date"], errors="coerce", format="mixed", dayfirst=True)
    revenue = units * price * (1 - pd.Series(disc, index=ref.index).fillna(0))
    return {
        "rows_in_file": len(raw),
        "rows": len(ref),
        "duplicates": len(raw) - len(ref),
        "regions": region.value_counts().to_dict(),
        "avg_rating": pd.Series(rating.values, index=region.values).groupby(level=0).mean().round(2).to_dict(),
        "revenue_all": round(float(revenue.sum()), 2),
        "returns": int((units < 0).sum()),
        "dec_2024": int(((dates.dt.year == 2024) & (dates.dt.month == 12)).sum()),
        "damaged": int(ref["Notes"].str.lower().str.contains("damaged").sum()),
    }


# ── live-server fixtures ──────────────────────────────────────────────────────

def _server_up() -> bool:
    import requests
    try:
        return requests.get(f"{BASE}/auth/check", timeout=3).ok
    except Exception:
        return False


@pytest.fixture(scope="session")
def server():
    if not _server_up():
        pytest.skip(f"No server at {BASE}; start one (python main.py) or set ANALYST_URL")
    return BASE


@pytest.fixture(scope="session")
def http(server):
    import requests
    s = requests.Session()
    s.headers.update({"X-Api-Key": KEY, "X-User-ID": QA_USER})
    return s


def upload_raw(http, cid, files, headers=None):
    h = {} if cid is None else {"X-Conversation-ID": cid}
    h.update(headers or {})
    for _ in range(12):
        r = http.post(f"{BASE}/upload", headers=h, files=files)
        if r.status_code != 429:
            return r
        time.sleep(10)
    return r


class Conversations:
    """Creates conversations and deletes all of them at the end of the session."""

    def __init__(self, http):
        self.http, self.ids = http, []

    def new(self, title="pytest"):
        r = self.http.post(f"{BASE}/conversations", json={"user_id": QA_USER, "title": title})
        r.raise_for_status()
        cid = r.json()["conversation_id"]
        self.ids.append(cid)
        return cid

    def post_files(self, cid, files):
        """POST /upload, waiting out the 10/minute rate limit instead of failing."""
        return upload_raw(self.http, cid, files)

    def upload(self, cid, paths, timeout=600):
        files = [("files", (Path(p).name, Path(p).read_bytes())) for p in paths]
        r = self.post_files(cid, files)
        r.raise_for_status()
        return self.wait(cid, timeout)

    def wait(self, cid, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.http.get(f"{BASE}/conversations/{cid}/processing-status").json()
            if st["status"] != "processing":
                return st
            time.sleep(0.5)
        raise TimeoutError(f"indexing {cid} didn't finish in {timeout}s")

    def cleanup(self):
        for cid in self.ids:
            try:
                self.http.delete(f"{BASE}/conversations/{cid}")
            except Exception:
                pass


@pytest.fixture(scope="session")
def convs(http):
    c = Conversations(http)
    yield c
    c.cleanup()


@pytest.fixture(scope="session")
def messy_cid(convs, messy_csv):
    """One conversation with the messy 1,000-row file, shared by api + llm tests."""
    cid = convs.new("pytest messy")
    st = convs.upload(cid, [messy_csv])
    assert st["status"] == "ready", st
    return cid


def ask(http, cid, question, **extra):
    r = http.post(f"{BASE}/query/silent", json={"question": question, "conversation_id": cid, **extra},
                  timeout=300)
    assert r.status_code == 200, r.text[:300]
    return r.json()
