"""Overview profile (KPIs, roles, auto charts) and chart specs from SQL results."""

import json

import numpy as np
import pandas as pd
import pytest

from tools.data_clean import clean_frame
from tools.data_profile import classify, profile_table
from utils.plot_builder import PlotBuilder


# ── column roles ──────────────────────────────────────────────────────────────

def test_roles_on_a_typical_sales_table():
    df = pd.DataFrame({
        "order_id": [f"O{i}" for i in range(40)],
        "date": pd.date_range("2024-01-01", periods=40, freq="W").strftime("%Y-%m-%d"),
        "region": ["North", "South", "East", "West"] * 10,
        "revenue": np.linspace(100, 500, 40),
        "notes": [f"free text note number {i} with detail" for i in range(40)],
    })
    roles = classify(df)
    assert roles == {"order_id": "id", "date": "date", "region": "categorical",
                     "revenue": "numeric", "notes": "text"}


@pytest.mark.parametrize("name", ["Amount Paid", "Bid", "Prepaid", "valid_amount"])
def test_words_ending_in_id_are_not_ids(name):
    """'Amount Paid' ends in 'id' but is money, not an identifier."""
    df = pd.DataFrame({name: np.random.default_rng(1).uniform(10, 999, 50).round(2)})
    assert classify(df)[name] == "numeric"


@pytest.mark.parametrize("name", ["id", "customer_id", "Customer ID", "CustomerID", "OrderId", "invoice number"])
def test_real_id_columns_are_ids(name):
    df = pd.DataFrame({name: range(1000, 1050)})
    assert classify(df)[name] == "id"


def test_year_column_is_time_not_a_measure():
    df = pd.DataFrame({"Year": [2020, 2021, 2022, 2023], "Sales": [1, 2, 3, 4]})
    prof = profile_table(df, "t")
    assert [k["column"] for k in prof["kpis"]] == ["Sales"]


def test_month_name_column_detected():
    df = pd.DataFrame({"Month": ["Jan", "Feb", "Mar", "Apr"], "Revenue": [1, 2, 3, 4]})
    assert classify(df)["Month"] == "month"


# ── KPIs ──────────────────────────────────────────────────────────────────────

def test_kpis_sum_additive_and_average_rates():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=6, freq="MS").strftime("%Y-%m-%d"),
        "revenue": [100, 200, 300, 400, 500, 600],
        "churn_rate": [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
        "unit_price": [10, 10, 10, 10, 10, 10],
    })
    kpis = {k["column"]: k for k in profile_table(df, "t")["kpis"]}
    assert kpis["revenue"]["agg"] == "Total" and kpis["revenue"]["value"] == 2100
    assert kpis["churn_rate"]["agg"] == "Average" and kpis["churn_rate"]["value"] == pytest.approx(0.15)
    assert list(kpis)[0] == "revenue"                         # additive measure leads
    assert kpis["revenue"]["change_pct"] == 20.0              # May 500 → Jun 600 (latest vs previous)
    assert len(kpis["revenue"]["spark"]) == 6


def test_missing_dates_never_become_a_period():
    """Rows with no date showed up as a final 'NaT' month and in '2023-01 → NaT'."""
    dates = pd.date_range("2024-01-01", periods=30, freq="W").strftime("%Y-%m-%d").tolist() + [None] * 5
    df = pd.DataFrame({"date": dates, "sales": range(35)})
    prof = profile_table(df, "t")
    k = prof["kpis"][0]
    assert k["to_label"] != "NaT" and "NaT" not in k["from_label"]
    trend = next(c for c in prof["charts"] if c["kind"] == "trend")
    assert "NaT" not in trend["plot"]["x"] and "None" not in trend["plot"]["x"]


def test_charts_leave_out_extreme_typos_and_say_so():
    """Three 9,999-unit typos made a ~13-unit series spike to 10K."""
    n = 300
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "region": (["North", "South", "East"] * n)[:n],
        "units": [10 + (i % 7) for i in range(n)],
    })
    df.loc[[5, 100, 200], "units"] = 9999
    prof = profile_table(df, "t")
    k = prof["kpis"][0]
    assert k["value"] == df["units"].sum()                     # headline keeps every row
    assert "3 extreme values" in k["trend_note"]
    assert max(k["spark"]) < 1000
    trend = next(c for c in prof["charts"] if c["kind"] == "trend")
    assert max(v for v in trend["plot"]["y"] if v is not None) < 1000
    bars = next(c for c in prof["charts"] if c["kind"] == "breakdown")
    assert max(bars["plot"]["y"]) < 1.2 * min(bars["plot"]["y"])  # regions comparable again
    for c in prof["charts"]:
        assert c["note"] == "Excludes 3 extreme values (9,999)"


def test_no_note_when_data_has_no_extremes():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60, freq="D").strftime("%Y-%m-%d"),
                       "units": [10 + (i % 7) for i in range(60)]})
    prof = profile_table(df, "t")
    assert all(c.get("note") is None for c in prof["charts"])
    assert "trend_note" not in prof["kpis"][0]


def test_kpi_change_skipped_when_previous_period_is_zero():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-02-01", "2024-03-01"], "sales": [5, 0, 9]})
    k = profile_table(df, "t")["kpis"][0]
    assert "change_pct" not in k


# ── whole profile stays valid JSON on awkward data ────────────────────────────

@pytest.mark.parametrize("df", [
    pd.DataFrame({"a": []}),
    pd.DataFrame({"only_text": ["x", "y", "z"]}),
    pd.DataFrame({"n": [np.nan, np.nan, np.nan]}),
    pd.DataFrame({"n": [1.0, np.inf, -np.inf, 2.0]}),
    pd.DataFrame({"d": ["2024-01-01", None, "2024-03-01"], "v": [1, None, 3]}),
    pd.DataFrame({"flag": [True, False, True], "v": [1, 2, 3]}),
    pd.DataFrame({"city": ["दिल्ली", "मुंबई", "दिल्ली"], "v": [1, 2, 3]}),
])
def test_profile_never_crashes_and_is_json(df):
    import math

    def json_safe(o):  # same contract as main._json_safe, which the endpoint applies
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, dict):
            return {k: json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [json_safe(v) for v in o]
        return o

    prof = profile_table(df, "t")
    json.dumps(json_safe(prof), allow_nan=False)


def test_profile_of_messy_file(messy_csv, messy_expected):
    raw = pd.read_csv(messy_csv, dtype=str, keep_default_na=False)
    df, _ = clean_frame(raw)
    prof = profile_table(df, "upload_x_messy_orders", "messy_orders.csv")
    roles = {c["name"]: c["role"] for c in prof["columns"]}
    assert roles["Order ID"] == "id"
    assert roles["Order Date"] == "date"
    assert roles["Region"] == "categorical"
    assert roles["Units"] == "numeric" and roles["Unit Price ($)"] == "numeric"
    assert roles["Notes"] == "text"
    region = next(c for c in prof["columns"] if c["name"] == "Region")
    assert region["distinct"] == 5
    assert {c["kind"] for c in prof["charts"]} == {"trend", "breakdown"}
    assert prof["kpis"][0]["column"] == "Units"


# ── chart specs from SQL results ──────────────────────────────────────────────

pb = PlotBuilder()


def _res(rows, row_count=None):
    return {"success": True, "data": rows, "data_preview": rows[:10], "row_count": row_count or len(rows)}


@pytest.mark.parametrize("q, kind", [
    ("show revenue as a bar chart", "bar"),
    ("revenue trend", "line"),
    ("pie chart of revenue by region", "pie"),
    ("revenue share by region", "pie"),
    ("revenue by region", "bar"),
    ("online sales by region", "bar"),      # "online" is not "line"
    ("how many pieces per region", "bar"),  # "pieces" is not "pie"
])
def test_chart_type_inference(q, kind):
    rows = [{"region": r, "revenue": v} for r, v in [("N", 1), ("S", 2), ("E", 3)]]
    assert pb.build_plot_spec(_res(rows), q)["type"] == kind


def test_time_axis_defaults_to_line():
    rows = [{"month": m, "revenue": v} for m, v in [("2024-01", 1), ("2024-02", 2), ("2024-03", 3)]]
    assert pb.build_plot_spec(_res(rows), "revenue per month")["type"] == "line"


def test_all_rows_used_not_just_preview():
    rows = [{"product": f"P{i}", "revenue": i} for i in range(25)]
    spec = pb.build_plot_spec(_res(rows), "revenue by product")
    assert len(spec["x"]) == 25 and spec["truncated"] is False


def test_truncation_flagged():
    rows = [{"product": f"P{i}", "revenue": i} for i in range(100)]
    assert pb.build_plot_spec(_res(rows, row_count=340), "revenue by product")["truncated"] is True


def test_nulls_are_gaps_not_zero():
    rows = [{"m": "a", "v": 1}, {"m": "b", "v": None}, {"m": "c", "v": 3}]
    assert pb.build_plot_spec(_res(rows), "v by m")["y"] == [1, None, 3]


def test_multiple_series_and_id_columns_skipped():
    rows = [{"month": f"2024-0{i}", "customer_id": i, "revenue": i * 10, "cost": i * 4} for i in range(1, 5)]
    spec = pb.build_plot_spec(_res(rows), "revenue and cost by month")
    assert [d["label"] for d in spec["datasets"]] == ["revenue", "cost"]


def test_numeric_strings_plot_as_numbers():
    rows = [{"r": "N", "v": "1.5"}, {"r": "S", "v": "2"}]
    assert pb.build_plot_spec(_res(rows), "v by r")["y"] == [1.5, 2.0]


@pytest.mark.parametrize("res", [
    {"success": False, "error": "x"},
    _res([]),
    _res([{"region": "N", "revenue": 1}]),                 # one row
    _res([{"a": "x", "b": "y"}, {"a": "z", "b": "w"}]),    # no numbers
])
def test_no_chart_when_nothing_to_plot(res):
    assert pb.build_plot_spec(res, "chart it") is None


def test_title_is_human():
    rows = [{"month": "Jan", "new_customers": 3}, {"month": "Feb", "new_customers": 5}]
    assert pb.build_plot_spec(_res(rows), "new customers by month")["title"] == "New customers by month"


def test_unfinished_last_month_left_out_of_trends():
    """Orders ending 19 Dec made every trend dive at the end and skewed the KPI change."""
    days = pd.date_range("2024-01-01", "2024-12-19", freq="D")
    df = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "units": 10})
    prof = profile_table(df, "t")
    k = prof["kpis"][0]
    assert k["to_label"] == "2024-11" and k["from_label"] == "2024-10"
    trend = next(c for c in prof["charts"] if c["kind"] == "trend")
    assert trend["plot"]["x"][-1] == "2024-11"


def test_month_stamped_data_keeps_its_last_month():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m-%d"),
                       "revenue": range(12)})
    assert profile_table(df, "t")["kpis"][0]["to_label"] == "2024-12"
