"""
HTTP contract against a running server. No LLM tokens spent.

Upload requests are kept to six per run: /upload is limited to 10/minute per IP.
"""

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from conftest import BASE, FIXTURES, upload_raw

pytestmark = pytest.mark.api


@pytest.fixture(scope="module")
def anon(server):
    return requests.Session()


# ── health + auth ─────────────────────────────────────────────────────────────

def test_health(server, http):
    assert http.get(f"{BASE}/").ok
    st = http.get(f"{BASE}/status").json()
    assert st and "sql_agent" in st


def test_auth_check_reports_key_state(anon, http):
    a = anon.get(f"{BASE}/auth/check").json()
    if not a["required"]:
        pytest.skip("server runs in open demo mode (API_KEYS unset)")
    assert a["valid"] is False
    assert http.get(f"{BASE}/auth/check").json()["valid"] is True
    assert anon.get(f"{BASE}/auth/check", headers={"X-Api-Key": "wrong"}).json()["valid"] is False


PROTECTED = [
    ("post", "/query", {"json": {"question": "hi"}}),
    ("post", "/query/silent", {"json": {"question": "hi", "conversation_id": "abc"}}),
    ("post", "/query/stream", {"json": {"question": "hi"}}),
    ("post", "/upload", {"files": [("files", ("a.txt", b"x"))], "headers": {"X-Conversation-ID": "abc"}}),
    ("delete", "/conversations/abc", {}),
    ("patch", "/conversations/abc", {"json": {"title": "x"}}),
    ("post", "/conversations/abc/dashboard", {"json": {"state": {}}}),
    ("post", "/mcp/tools/call", {"json": {"name": "get_status", "arguments": {}}}),
]


@pytest.mark.parametrize("method, path, kw", PROTECTED, ids=[p for _, p, _ in PROTECTED])
@pytest.mark.parametrize("key", [None, "wrong-key"], ids=["no-key", "bad-key"])
def test_protected_endpoints_need_a_valid_key(anon, method, path, kw, key):
    if not anon.get(f"{BASE}/auth/check").json()["required"]:
        pytest.skip("open demo mode")
    kw = dict(kw)
    headers = dict(kw.pop("headers", {}))
    if key:
        headers["X-Api-Key"] = key
    r = getattr(anon, method)(f"{BASE}{path}", headers=headers, **kw)
    assert r.status_code == 401, r.text[:200]


# ── input validation ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("cid", ["..", "a.b", "a b", "x" * 65, "%2E%2E", "abc;drop"])
def test_bad_conversation_ids_rejected(http, cid):
    for path in ("documents", "profile", "schema", "processing-status", "dashboard"):
        r = http.get(f"{BASE}/conversations/{cid}/{path}")
        assert r.status_code in (400, 404), (path, r.status_code)


@pytest.mark.parametrize("cid", ["../x", "..", "a/b", "a\\b"])
def test_bad_conversation_id_headers_and_bodies_rejected(http, cid):
    r = upload_raw(http, cid, [("files", ("a.txt", b"x"))])
    assert r.status_code == 400
    r = http.post(f"{BASE}/query/silent", json={"question": "hi", "conversation_id": cid})
    assert r.status_code == 400


def test_upload_needs_conversation_header(http):
    r = upload_raw(http, None, [("files", ("a.txt", b"x"))])
    assert r.status_code == 400


@pytest.mark.parametrize("body", [
    {},
    {"question": ""},
    {"question": "   "},
    {"question": "x" * 2001},
    {"question": None},
    {"question": ["list"]},
], ids=["missing", "empty", "blank", "too-long", "null", "wrong-type"])
def test_bad_questions_rejected_before_the_llm(http, body):
    r = http.post(f"{BASE}/query/silent", json={**body, "conversation_id": "abc"})
    assert r.status_code == 422, r.text[:200]


def test_malformed_json_rejected(http):
    r = http.post(f"{BASE}/query/silent", data="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# ── uploads ───────────────────────────────────────────────────────────────────

def test_upload_size_cap(http, convs):
    cid = convs.new("pytest big")
    limit_mb = int(os.getenv("MAX_UPLOAD_MB", "25"))
    big = io.BytesIO(b"a,b\n" + b"1,2\n" * ((limit_mb * 1024 * 1024) // 4 + 10))
    r = upload_raw(http, cid, [("files", ("big.csv", big.getvalue()))])
    assert r.status_code == 413
    assert not (Path("uploads") / cid / "big.csv").exists()


def test_bad_files_fail_softly_and_never_escape_the_session_folder(http, convs):
    cid = convs.new("pytest bad files")
    files = [
        ("files", ("../../pytest_escape.txt", b"traversal probe text that should stay inside")),
        ("files", ("..\\pytest_escape2.txt", b"windows traversal probe")),
        ("files", ("empty.csv", b"")),
        ("files", ("garbage.csv", b"\x00\x01\x02\x03garbage\x00")),
        ("files", ("tool.exe", b"MZ\x90\x00")),
    ]
    r = upload_raw(http, cid, files)
    assert r.status_code == 200, r.text
    st = convs.wait(cid, 120)
    # The two text files are valid documents; the rest must be listed as errors
    assert st["status"] == "ready", st
    err = st["error"] or ""
    for name in ("empty.csv", "garbage.csv", "tool.exe"):
        assert name in err, (name, err)
    assert not any(Path(".").glob("pytest_escape*")) and not any(Path("uploads").glob("pytest_escape*"))
    assert (Path("uploads") / cid / "pytest_escape.txt").exists()
    assert http.get(f"{BASE}/").ok           # server survived


def test_messy_upload_is_cleaned_and_profiled(http, messy_cid, messy_expected):
    t = time.time()
    p = http.get(f"{BASE}/conversations/{messy_cid}/profile").json()
    assert time.time() - t < 5, "profile should be instant (no LLM)"
    assert p["success"], p
    ds = p["datasets"][0]
    assert ds["rows"] == messy_expected["rows"]
    assert ds["source"] == "messy_orders.csv"
    cols = {c["name"]: c for c in ds["columns"]}
    assert cols["Region"]["distinct"] == 5
    assert cols["Units"]["role"] == "numeric" and cols["Unit Price ($)"]["role"] == "numeric"
    assert ds["quality"]["duplicates_removed"] == messy_expected["duplicates"]
    assert ds["kpis"] and ds["charts"]


def test_cleaned_columns_have_real_sql_types(http, messy_cid):
    sch = http.get(f"{BASE}/conversations/{messy_cid}/schema").json()
    text = str(sch)
    assert "INTEGER" in text or "REAL" in text, text[:400]
    nc = http.get(f"{BASE}/conversations/{messy_cid}/numeric-columns").json()
    assert {"Units", "Unit Price ($)"} <= set(nc["columns"])


def test_awkward_csv_formats_import(http, convs, tmp_path):
    rows = "Name,City,Sales\nChloé,Paris,10\nJosé,Madrid,20\nAsha,Pune,30\n"
    variants = {"excel_cp1252.csv": rows.encode("cp1252"), "euro_semicolon.csv": rows.replace(",", ";").encode(),
                "bom.csv": rows.encode("utf-8-sig"), "UPPER.CSV": rows.encode()}
    cid = convs.new("pytest formats")
    files = [("files", (n, b)) for n, b in variants.items()]
    r = upload_raw(http, cid, files)
    assert r.ok
    st = convs.wait(cid, 300)
    assert st["status"] == "ready" and not st["error"], st
    p = http.get(f"{BASE}/conversations/{cid}/profile").json()
    assert len(p["datasets"]) == 4
    for ds in p["datasets"]:
        names = [c["name"] for c in ds["columns"]]
        assert names == ["Name", "City", "Sales"], (ds["source"], names)
        assert "Chloé" in str(ds["sample"]), ds["source"]


def test_reupload_replaces_instead_of_duplicating(http, convs):
    cid = convs.new("pytest reupload")
    csv = FIXTURES / "Sales_Report_H1_2025.csv"
    convs.upload(cid, [csv])
    before = http.get(f"{BASE}/conversations/{cid}/documents").json()
    convs.upload(cid, [csv])
    after = http.get(f"{BASE}/conversations/{cid}/documents").json()
    assert after["documents"] == before["documents"]
    assert after["total_vectors"] == before["total_vectors"]


# ── conversations, isolation, delete ──────────────────────────────────────────

def test_profiles_are_isolated_between_conversations(http, convs, messy_cid):
    other = convs.new("pytest empty")
    p = http.get(f"{BASE}/conversations/{other}/profile").json()
    assert p["datasets"] == []
    assert "messy" not in str(p)


def test_rename(http, convs):
    cid = convs.new("pytest rename")
    r = http.patch(f"{BASE}/conversations/{cid}", json={"title": "  Q3   review  "})
    assert r.ok and r.json()["title"] == "Q3 review"
    assert http.patch(f"{BASE}/conversations/{cid}", json={"title": "   "}).status_code == 400
    assert len(http.patch(f"{BASE}/conversations/{cid}", json={"title": "x" * 300}).json()["title"]) == 80
    assert http.patch(f"{BASE}/conversations/{cid}", json={"title": "x" * 501}).status_code == 422
    assert http.patch(f"{BASE}/conversations/nope-not-real", json={"title": "x"}).status_code == 404


def test_listing_only_shows_own_conversations(http, convs):
    cid = convs.new("pytest list")
    mine = http.get(f"{BASE}/conversations", params={"limit": 200}).json()["conversations"]
    assert cid in [c["conversation_id"] for c in mine]
    theirs = requests.get(f"{BASE}/conversations", headers={"X-User-ID": "someone_else_pytest"}).json()
    assert cid not in str(theirs)


def test_dashboard_roundtrip_and_cap(http, convs):
    cid = convs.new("pytest dash")
    state = {"panels": [{"q": "revenue by month", "wide": True}], "v": 1}
    assert http.post(f"{BASE}/conversations/{cid}/dashboard", json={"state": state}).json()["success"]
    assert http.get(f"{BASE}/conversations/{cid}/dashboard").json()["state"] == state
    huge = {"blob": "x" * (6 * 1024 * 1024)}
    assert http.post(f"{BASE}/conversations/{cid}/dashboard", json={"state": huge}).json()["success"] is False


def test_delete_removes_everything(http, convs):
    cid = convs.new("pytest delete")
    convs.upload(cid, [FIXTURES / "Sales_Report_H1_2025.csv"])
    assert http.get(f"{BASE}/conversations/{cid}/profile").json()["datasets"]
    http.post(f"{BASE}/conversations/{cid}/dashboard", json={"state": {"a": 1}})
    assert http.delete(f"{BASE}/conversations/{cid}").json()["success"]
    assert http.get(f"{BASE}/conversations/{cid}").status_code == 404
    assert http.get(f"{BASE}/conversations/{cid}/profile").json()["datasets"] == []
    assert http.get(f"{BASE}/conversations/{cid}/dashboard").json()["state"] is None
    assert not (Path("uploads") / cid).exists()
    assert not (Path("data/vector_stores") / cid).exists()
    import sqlite3
    con = sqlite3.connect("data/business.db")
    left = [t for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if t.lower().startswith(f"upload_{cid[:8].lower()}")]
    con.close()
    assert left == []
    assert http.delete(f"{BASE}/conversations/{cid}").ok     # idempotent


def test_unknown_conversation_status(http):
    assert http.get(f"{BASE}/conversations/never-made-123/processing-status").json()["status"] == "unknown"


# ── load ──────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_parallel_reads_are_consistent(http, messy_cid):
    def get(_):
        r = http.get(f"{BASE}/conversations/{messy_cid}/profile")
        return r.status_code, r.json()["datasets"][0]["rows"]
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(get, range(36)))
    assert {s for s, _ in res} == {200}
    assert len({n for _, n in res}) == 1


# ── protocol surfaces ─────────────────────────────────────────────────────────

def test_mcp_and_a2a_discovery(http):
    tools = http.get(f"{BASE}/mcp/tools/list").json()
    assert "tools" in str(tools).lower()
    reg = http.get(f"{BASE}/a2a/registry").json()
    assert "sql" in str(reg).lower()
