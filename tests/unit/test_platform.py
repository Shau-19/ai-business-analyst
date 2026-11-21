"""LLM provider factory, user-facing error text, auth/quota, value rewriter, forecast helpers."""

import asyncio

import pandas as pd
import pytest


# ── LLM factory ───────────────────────────────────────────────────────────────

@pytest.fixture
def llm_mod(monkeypatch):
    import utils.llm as L
    return L


def _kind(m):
    return type(m).__name__


def test_no_keys_is_a_clear_error(llm_mod, monkeypatch):
    monkeypatch.setattr(llm_mod, "GROQ_KEY", "")
    monkeypatch.setattr(llm_mod, "OPENAI_KEY", "")
    with pytest.raises(ValueError, match="No LLM API key"):
        llm_mod.make_llm()


def test_groq_primary_openai_fallback(llm_mod, monkeypatch):
    monkeypatch.setattr(llm_mod, "GROQ_KEY", "g")
    monkeypatch.setattr(llm_mod, "OPENAI_KEY", "o")
    monkeypatch.setattr(llm_mod, "PROVIDER", "groq")
    m = llm_mod.make_llm()
    assert _kind(m) == "RunnableWithFallbacks"
    assert _kind(m.runnable) == "ChatGroq" and _kind(m.fallbacks[0]) == "ChatOpenAI"


def test_openai_primary_when_configured(llm_mod, monkeypatch):
    monkeypatch.setattr(llm_mod, "GROQ_KEY", "g")
    monkeypatch.setattr(llm_mod, "OPENAI_KEY", "o")
    monkeypatch.setattr(llm_mod, "PROVIDER", "openai")
    m = llm_mod.make_llm()
    assert _kind(m.runnable) == "ChatOpenAI" and _kind(m.fallbacks[0]) == "ChatGroq"


def test_single_provider_has_no_fallback_wrapper(llm_mod, monkeypatch):
    monkeypatch.setattr(llm_mod, "GROQ_KEY", "")
    monkeypatch.setattr(llm_mod, "OPENAI_KEY", "o")
    assert _kind(llm_mod.make_llm()) == "ChatOpenAI"


def test_reasoning_low_only_for_gpt_oss(llm_mod, monkeypatch):
    monkeypatch.setattr(llm_mod, "GROQ_KEY", "g")
    monkeypatch.setattr(llm_mod, "OPENAI_KEY", "")
    m = llm_mod.make_llm(reasoning_low=True)
    expect = "gpt-oss" in llm_mod.settings.LLM_MODEL.lower()
    assert (m.model_kwargs.get("reasoning_effort") == "low") is expect


def test_describe_never_leaks_keys(llm_mod, monkeypatch):
    monkeypatch.setattr(llm_mod, "GROQ_KEY", "gsk_SECRET")
    monkeypatch.setattr(llm_mod, "OPENAI_KEY", "sk-SECRET")
    assert "SECRET" not in llm_mod.describe()


# ── provider errors shown to users ────────────────────────────────────────────

@pytest.mark.parametrize("raw, expect", [
    ("Error code: 429 - rate_limit_exceeded tokens per day", "rate limit"),
    ("Rate limit reached for model", "rate limit"),
    ("Request timed out.", "too long"),
    ("ReadTimeout", "too long"),
    ("Error code: 401 - invalid_api_key", "API key"),
    ("Invalid API Key provided", "API key"),
])
def test_friendly_llm_error(raw, expect):
    from agents.sql_analyst import friendly_llm_error
    msg = friendly_llm_error(Exception(raw))
    assert expect.lower() in msg.lower()
    assert "Error code" not in msg


def test_friendly_llm_error_names_no_single_provider():
    """With OpenAI as fallback, telling users to fix GROQ_API_KEY can be wrong."""
    from agents.sql_analyst import friendly_llm_error
    msg = friendly_llm_error(Exception("401 invalid_api_key"))
    assert ("GROQ_API_KEY" in msg) == ("OPENAI_API_KEY" in msg)


def test_unknown_errors_pass_through():
    from agents.sql_analyst import friendly_llm_error
    assert friendly_llm_error(Exception("no such column: foo")) == "no such column: foo"


# ── auth + daily quota ────────────────────────────────────────────────────────

@pytest.fixture
def auth(monkeypatch):
    import auth as A
    monkeypatch.setattr(A, "VALID_KEYS", {"k1", "k2"})
    monkeypatch.setattr(A, "_quota_state", {"day": None, "count": 0})
    return A


def test_key_checks(auth):
    assert auth.auth_required()
    assert auth.key_is_valid("k1") and auth.key_is_valid("k2")
    for bad in (None, "", "k3", "K1", " k1"):
        assert not auth.key_is_valid(bad)


def test_open_mode_when_no_keys(auth, monkeypatch):
    monkeypatch.setattr(auth, "VALID_KEYS", set())
    assert not auth.auth_required() and auth.key_is_valid(None)


def test_verify_key_raises_401(auth):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        asyncio.run(auth.verify_key("nope"))
    assert e.value.status_code == 401
    asyncio.run(auth.verify_key("k1"))


def test_daily_quota(auth, monkeypatch):
    from fastapi import HTTPException
    monkeypatch.setattr(auth, "DAILY_QUERY_LIMIT", 3)
    for _ in range(3):
        asyncio.run(auth.enforce_daily_quota())
    assert auth.quota_status() == {"limit": 3, "used": 3, "remaining": 0}
    with pytest.raises(HTTPException) as e:
        asyncio.run(auth.enforce_daily_quota())
    assert e.value.status_code == 429
    monkeypatch.setattr(auth, "_today", lambda: "2999-01-01")      # next day resets
    asyncio.run(auth.enforce_daily_quota())
    assert auth.quota_status()["used"] == 1


def test_quota_is_thread_safe(auth, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from fastapi import HTTPException
    monkeypatch.setattr(auth, "DAILY_QUERY_LIMIT", 50)

    def hit(_):
        try:
            asyncio.run(auth.enforce_daily_quota())
            return 1
        except HTTPException:
            return 0
    with ThreadPoolExecutor(16) as ex:
        assert sum(ex.map(hit, range(200))) == 50


# ── value rewriter (fuzzy-matches literals to real values) ────────────────────

@pytest.fixture
def vindex():
    from tools.column_value_index import ColumnValueIndex
    idx = ColumnValueIndex()
    idx.build_from_dataframe(pd.DataFrame({
        "Region": ["North", "South", "East"], "Units": [1, 2, 3], "Month": ["January", "February", "March"],
    }), "t")
    return idx


def test_rewriter_fixes_case_of_literals(vindex):
    out = vindex.rewrite_sql("SELECT * FROM t WHERE Region = 'north'", "t")
    assert "'North'" in out


def test_rewriter_never_touches_identifiers(vindex):
    sql = 'SELECT SUM("Units") FROM t WHERE "Region" = \'North\''
    assert vindex.rewrite_sql(sql, "t") == sql


def test_rewriter_leaves_numbers_and_unknown_tables(vindex):
    sql = "SELECT * FROM t WHERE Units = '2'"
    assert vindex.rewrite_sql(sql, "t") == sql
    assert vindex.rewrite_sql("SELECT 'north'", "other") == "SELECT 'north'"


@pytest.fixture
def dated_index():
    from tools.column_value_index import ColumnValueIndex
    idx = ColumnValueIndex()
    idx.build_from_dataframe(pd.DataFrame({
        "Order Date": ["2024-12-28", "2024-11-03", "2023-01-15"],
        "City": ["Mumbai", "Navi Mumbai", "Pune"],
    }), "t")
    return idx


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM t WHERE strftime('%Y-%m', \"Order Date\") = '2024-12'",   # month prefix
    "SELECT COUNT(*) FROM t WHERE \"Order Date\" >= '2024-12-01'",                  # date bound
    "SELECT COUNT(*) FROM t WHERE \"Order Date\" LIKE '2024-12%'",
    "SELECT COUNT(*) FROM t WHERE substr(\"Order Date\", 1, 4) = '2024'",
])
def test_rewriter_never_alters_dates_or_patterns(dated_index, sql):
    """'2024-12' was fuzzy-matched to '2024-12-28': December counted one day."""
    assert dated_index.rewrite_sql(sql, "t") == sql


def test_rewriter_does_not_turn_a_short_value_into_a_longer_one(dated_index):
    sql = "SELECT * FROM t WHERE City = 'Mumbai'"
    assert dated_index.rewrite_sql(sql, "t") == sql
    assert "'Pune'" in dated_index.rewrite_sql("SELECT * FROM t WHERE City = 'pune'", "t")


def test_rewriter_keeps_like_patterns(vindex):
    sql = "SELECT * FROM t WHERE Region LIKE '%damaged%'"
    assert vindex.rewrite_sql(sql, "t") == sql


# ── forecast: month-name columns ──────────────────────────────────────────────

def _fa():
    from agents.forecast_agent import ForecastAgent
    return ForecastAgent.__new__(ForecastAgent)


def test_month_names_take_year_from_table_name():
    df = pd.DataFrame({"Month": ["Nov", "Dec", "Jan", "Feb"], "v": [1, 2, 3, 4]})
    df.attrs["table"] = "upload_ab_sales_report_2024"
    out, start, guessed = _fa()._dates_from_month_names(df, "Month")
    assert start == 2024 and not guessed
    assert [d.strftime("%Y-%m") for d in out["__month_date"]] == ["2024-11", "2024-12", "2025-01", "2025-02"]


def test_month_names_without_year_are_flagged_as_guessed():
    df = pd.DataFrame({"Month": ["January", "February", "March"], "v": [1, 2, 3]})
    _, _, guessed = _fa()._dates_from_month_names(df, "Month")
    assert guessed


def test_month_name_column_detection():
    df = pd.DataFrame({"m": ["Jan", "Feb", "Mar"], "x": ["a", "b", "c"]})
    assert _fa()._month_name_column(df) == "m"
    assert _fa()._month_name_column(pd.DataFrame({"x": ["a", "b"]})) is None


# ── forecast: one value per period ────────────────────────────────────────────

def _series(dates, values):
    return pd.DataFrame({"ds": pd.to_datetime(dates), "y": values})


def test_rows_per_channel_are_totalled_per_month():
    """4 channel rows per month used to look like daily data and forecast 6 days."""
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    pdf = _series(list(months.repeat(4)), [100.0] * 48)
    out, freq, notes = _fa()._to_period_series(pdf, "Revenue")
    assert freq == "MS" and len(out) == 12 and out["y"].tolist() == [400.0] * 12
    assert "Totalled per month: 48 rows → 12 months." in notes[0]


def test_rates_are_averaged_not_summed():
    pdf = _series(["2024-01-05", "2024-01-06", "2024-02-03", "2024-02-21", "2024-03-02", "2024-03-30",
                   "2024-04-04", "2024-04-05", "2024-05-02", "2024-05-29"], [4, 2] * 5)
    out, _, notes = _fa()._to_period_series(pdf, "Customer Rating")
    assert out["y"].tolist() == [3.0] * 5 and notes[0].startswith("Averaged per month")


def test_incomplete_last_month_and_typos_left_out():
    days = pd.date_range("2024-01-01", "2024-12-03", freq="D")
    pdf = _series(list(days) * 2, [10.0] * (2 * len(days)))
    pdf.loc[5, "y"] = 9999
    out, _, notes = _fa()._to_period_series(pdf, "Units")
    assert out["ds"].max() == pd.Timestamp("2024-11-01")
    assert out["y"].max() < 1000
    assert any("December 2024 is incomplete" in n for n in notes)
    assert any("1 extreme value left out" in n for n in notes)


def test_clean_monthly_series_is_untouched():
    pdf = _series(pd.date_range("2024-01-01", periods=6, freq="MS"), [1, 2, 3, 4, 5, 6])
    out, freq, notes = _fa()._to_period_series(pdf, "Revenue")
    assert freq == "MS" and out["y"].tolist() == [1, 2, 3, 4, 5, 6] and notes == []


def test_tickets_with_clock_times_are_aggregated():
    times = pd.date_range("2024-01-01 09:00", periods=300, freq="7h")
    out, freq, notes = _fa()._to_period_series(_series(times, [5.0] * 300), "Resolution Hours")
    assert freq in ("W-MON", "MS") and len(out) < 30 and notes[0].startswith("Averaged per")
