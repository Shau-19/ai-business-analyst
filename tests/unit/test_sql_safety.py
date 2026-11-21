"""
The SQL the LLM writes is untrusted. These are the guards between it and
the database: read-only allow-list, read-only connection, session table
isolation, and a time limit.
"""

import threading
import time

import pytest

from database import db_manager as dbm
from database.db_manager import ForbiddenTableError, QueryTimeoutError, _assert_read_only_sql

T = "upload_aaaa1111_sales"
OTHER = "upload_bbbb2222_secret"


# ── statement allow-list ──────────────────────────────────────────────────────

@pytest.mark.parametrize("sql", [
    f"SELECT * FROM {T}",
    f"select region, count(*) from {T} group by 1",
    f"WITH x AS (SELECT * FROM {T}) SELECT * FROM x",
    f"SELECT * FROM {T};",
    f"EXPLAIN QUERY PLAN SELECT * FROM {T}",
    f"PRAGMA table_info({T})",
    f"SELECT SUM(CAST(REPLACE(REPLACE(\"Unit Price ($)\", '$', ''), ',', '') AS REAL)) FROM {T}",
    f"SELECT * FROM {T} WHERE region = 'drop table users; --'",
    f"SELECT * FROM {T} WHERE region LIKE '%update%'",
    f"SELECT region AS \"delete\" FROM {T}",
    f"SELECT created AS update_date FROM {T}",
    f"SELECT 'it''s; fine' FROM {T}",
])
def test_read_only_statements_pass(sql):
    _assert_read_only_sql(sql)


@pytest.mark.parametrize("sql", [
    f"INSERT INTO {T} VALUES (1)",
    f"UPDATE {T} SET units = 0",
    f"DELETE FROM {T}",
    f"DROP TABLE {T}",
    f"ALTER TABLE {T} ADD x",
    "CREATE TABLE x (a)",
    f"REPLACE INTO {T} VALUES (1)",
    "ATTACH DATABASE 'x.db' AS x",
    "DETACH DATABASE x",
    "VACUUM",
    "PRAGMA writable_schema = 1",
    "PRAGMA journal_mode",
    f"SELECT * FROM {T}; DROP TABLE {T}",
    f"SELECT 1; SELECT 2",
    f"WITH d AS (DELETE FROM {T} RETURNING *) SELECT * FROM d",
    f"EXPLAIN DELETE FROM {T}",
    f"select * from {T} where 1=1 ; delete from {T}",
    f"SeLeCt 1; InSeRt into {T} values (1)",
    "  ",
    "",
])
def test_writes_and_stacked_statements_rejected(sql):
    with pytest.raises(ValueError):
        _assert_read_only_sql(sql)


def test_connection_is_read_only_even_if_guard_bypassed(tmp_db, monkeypatch):
    monkeypatch.setattr(dbm, "_assert_read_only_sql", lambda q: None)
    with pytest.raises(Exception, match="(?i)readonly|read-only"):
        tmp_db.execute_query(f"INSERT INTO {T} VALUES ('x', 1, 1, 'x')")
    assert len(tmp_db.execute_query(f"SELECT * FROM {T}")) == 4


def test_extensions_cannot_be_loaded(tmp_db):
    with pytest.raises(Exception):
        tmp_db.execute_query("SELECT load_extension('evil')")


# ── session isolation (SQLite authorizer) ─────────────────────────────────────

@pytest.mark.parametrize("sql", [
    f"SELECT * FROM {OTHER}",
    f"SELECT * FROM main.{OTHER}",
    f"SELECT * FROM '{OTHER}'",
    f"SELECT * FROM [{OTHER}]",
    f'SELECT * FROM "{OTHER}"',
    f"SELECT * FROM /**/ {OTHER}",
    f"SELECT * FROM {OTHER.upper()}",
    f"SELECT * FROM {T} WHERE units IN (SELECT salary FROM {OTHER})",
    f"SELECT (SELECT owner FROM {OTHER}) FROM {T}",
    f"SELECT region FROM {T} UNION SELECT owner FROM {OTHER}",
    f"SELECT * FROM {T} JOIN {OTHER} ON 1=1",
    f"SELECT * FROM {T}, {OTHER}",
    f"WITH x AS (SELECT * FROM {OTHER}) SELECT * FROM x",
    "SELECT * FROM employees",
    "SELECT name, sql FROM sqlite_master",
    "SELECT * FROM sqlite_schema",
    f"SELECT * FROM pragma_table_info('{OTHER}')",
    f"PRAGMA table_info({OTHER})",
])
def test_other_tables_are_unreadable(tmp_db, sql):
    with pytest.raises(ForbiddenTableError):
        tmp_db.execute_query(sql, allowed_tables=[T])


@pytest.mark.parametrize("sql", [
    f"SELECT * FROM {T}",
    f"SELECT * FROM main.{T}",
    f"SELECT * FROM {T.upper()}",
    f"WITH employees AS (SELECT * FROM {T}) SELECT * FROM employees",   # CTE shadows a real name
    f"SELECT a.region FROM {T} a JOIN {T} b ON a.region = b.region",
    f"SELECT * FROM pragma_table_info('{T}')",
    f"PRAGMA table_info({T})",
    f"SELECT strftime('%Y', created), count(*) FROM {T} GROUP BY 1",
])
def test_own_tables_are_readable(tmp_db, sql):
    assert len(tmp_db.execute_query(sql, allowed_tables=[T])) >= 1


def test_no_allow_list_means_internal_call(tmp_db):
    """Internal callers (profile, forecast) pass no list and read normally."""
    assert len(tmp_db.execute_query(f"SELECT * FROM {OTHER}")) == 1


def test_executor_reports_forbidden_cleanly(tmp_db):
    from tools.sql_executor import SQLExecutor
    res = SQLExecutor(tmp_db).execute_query(f"SELECT * FROM {OTHER}", allowed_tables=[T])
    assert res["success"] is False and res.get("forbidden") is True
    assert "999999" not in str(res)


# ── time limit ────────────────────────────────────────────────────────────────

RUNAWAY = [
    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c",
    f"SELECT count(*) FROM {T} a, {T} b, {T} c, {T} d, {T} e, {T} f, {T} g, {T} h, {T} i, {T} j, {T} k, {T} l, {T} m",
]


@pytest.mark.parametrize("sql", RUNAWAY)
def test_runaway_query_is_stopped(tmp_db, sql):
    out = {}

    def run():
        try:
            tmp_db.execute_query(sql, allowed_tables=[T], timeout_s=1)
        except Exception as e:
            out["err"] = e

    t0 = time.monotonic()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(10)
    assert not th.is_alive(), "query was not interrupted"
    assert isinstance(out.get("err"), QueryTimeoutError)
    assert time.monotonic() - t0 < 5


def test_executor_reports_timeout_cleanly(tmp_db, monkeypatch):
    from tools.sql_executor import SQLExecutor
    monkeypatch.setattr(dbm, "SQL_TIMEOUT_S", 1.0)
    res = SQLExecutor(tmp_db).execute_query(RUNAWAY[0], allowed_tables=[T])
    assert res["success"] is False and res.get("timeout") is True


# ── executor output is always JSON-safe ───────────────────────────────────────

def test_nan_and_inf_become_none(tmp_db):
    import json
    from tools.sql_executor import SQLExecutor
    res = SQLExecutor(tmp_db).execute_query(
        f'SELECT region, "Unit Price ($)" AS p, 1.0 / 0 AS div, 1e999 AS inf, -1e999 AS ninf FROM {T}',
        allowed_tables=[T])
    assert res["success"], res
    json.dumps(res, allow_nan=False)   # raises on NaN / inf
    east = next(r for r in res["data"] if r["region"] == "East")
    assert east["p"] is None


def test_code_fences_and_comments_stripped(tmp_db):
    from tools.sql_executor import SQLExecutor
    res = SQLExecutor(tmp_db).execute_query(
        f"```sql\nSELECT count(*) AS n FROM {T} -- all rows\n/* block */;\n```", allowed_tables=[T])
    assert res["success"] and res["data"][0]["n"] == 4


def test_result_caps_at_100_rows(tmp_db):
    from tools.sql_executor import SQLExecutor
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c LIMIT 250) SELECT x FROM c"
    res = SQLExecutor(tmp_db).execute_query(sql)
    assert res["row_count"] == 250 and len(res["data"]) == 100 and res["truncated"] is True
