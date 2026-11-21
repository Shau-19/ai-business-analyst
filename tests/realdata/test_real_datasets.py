"""
The analyst on public datasets it was not built around.

Every expected value here is computed with plain pandas inside the test, and
each checks a documented finding about that dataset:

  Telco churn     fiber customers churn more; they pay more and lack add-ons
  IBM HR          Sales Representatives drive Sales attrition; they are junior and low-paid
  Superstore      tables are the loss-makers inside Furniture
  Clothing        Trend has the lowest ratings; reviews there talk about fit and returns

Run:  python tests/realdata/fetch.py  then  pytest -m realdata
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from parsers.hybrid_csv_processor import HybridCSVProcessor
from tools.data_clean import clean_frame
from tools.data_profile import profile_table
from tools.driver_analysis import analyze

pytestmark = pytest.mark.realdata
DATA = Path(__file__).resolve().parent / "data"


def load(name):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} missing: run python tests/realdata/fetch.py")
    raw = HybridCSVProcessor._read_frame(str(path))
    df, report = clean_frame(raw)
    return raw, df, report


def factors(r):
    return [d["factor"] for d in r["data"]]


def headline_pct(answer):
    return float(re.search(r":\s*([\d.]+)%\*\*", answer).group(1))


# ── reading + cleaning real exports ───────────────────────────────────────────

@pytest.mark.parametrize("name, rows, cols", [("Telco-Customer-Churn.csv", 7043, 21), ("emp_attrition.csv", 1470, 35),
                                              ("superstoreSales.csv", 8399, 21),
                                              ("womens_clothing_reviews.csv", 23486, 11)])
def test_files_read_with_the_right_shape(name, rows, cols):
    raw, df, report = load(name)
    assert raw.shape == (rows, cols)           # superstore has old Mac \r line endings
    assert len(df) == rows and report["duplicates_removed"] == 0


def test_cleaner_fixes_telco_total_charges():
    """TotalCharges is text in the export (blank for 11 brand-new customers)."""
    raw, df, _ = load("Telco-Customer-Churn.csv")
    assert raw["TotalCharges"].dtype == object
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == (raw["TotalCharges"].str.strip() == "").sum()


def test_cleaner_leaves_clean_categories_alone():
    raw, df, _ = load("emp_attrition.csv")
    for c in ("Department", "JobRole", "EducationField", "OverTime"):
        assert sorted(df[c].unique()) == sorted(raw[c].unique()), c


def test_profiles_build_on_every_dataset():
    for name in ("Telco-Customer-Churn.csv", "emp_attrition.csv", "superstoreSales.csv", "womens_clothing_reviews.csv"):
        _, df, _ = load(name)
        prof = profile_table(df, "t", name)
        assert prof["kpis"] and prof["columns"], name


# ── Telco churn ───────────────────────────────────────────────────────────────

def test_telco_fiber_churn(embedder):
    _, df, _ = load("Telco-Customer-Churn.csv")
    r = analyze(df, "Why is churn higher for Fiber optic customers?", embed=embedder)
    fiber = df.InternetService == "Fiber optic"
    truth = (df.loc[fiber, "Churn"] == "Yes").mean() * 100
    assert headline_pct(r["answer"]) == pytest.approx(truth, abs=0.05)
    f = factors(r)
    assert "Monthly Charges" in f                                    # fiber costs more
    assert any(x in f for x in ("Online Security: No", "Tech Support: No", "Contract: Two year",
                                "Contract: Month-to-month"))
    assert not any("No internet service" in x for x in f)            # true by definition, not a cause
    charges = next(d for d in r["data"] if d["factor"] == "Monthly Charges")
    assert charges["segment_avg"] == pytest.approx(df.loc[fiber, "MonthlyCharges"].mean(), rel=1e-3)


def test_telco_contract_tenure(embedder):
    _, df, _ = load("Telco-Customer-Churn.csv")
    r = analyze(df, "Why is churn higher for month-to-month contracts?", embed=embedder)
    assert "Tenure" in factors(r)


# ── IBM HR attrition ──────────────────────────────────────────────────────────

def test_hr_sales_attrition_comes_from_sales_reps(embedder):
    _, df, _ = load("emp_attrition.csv")
    r = analyze(df, "Why is attrition higher in Sales?", embed=embedder)
    rep = df.JobRole == "Sales Representative"
    truth = (df.loc[rep, "Attrition"] == "Yes").mean() * 100
    top = r["data"][0]
    assert top["factor"] == "Job Role: Sales Representative"
    assert top["metric_with"] * 100 == pytest.approx(truth, abs=0.05)
    assert "a group found only in Sales" in r["answer"]
    assert not any("Manufacturing Director" in x for x in factors(r))   # never in Sales: structure


def test_hr_sales_reps_are_junior_and_low_paid(embedder):
    _, df, _ = load("emp_attrition.csv")
    r = analyze(df, "Why is attrition higher for Sales Representatives?", embed=embedder)
    assert r["analysis"]["segment_column"] == "JobRole"               # plural matched the value
    rep = df.JobRole == "Sales Representative"
    f = {d["factor"]: d for d in r["data"]}
    assert {"Monthly Income", "Job Level", "Total Working Years"} & set(f)
    if "Monthly Income" in f:
        assert f["Monthly Income"]["segment_avg"] == pytest.approx(df.loc[rep, "MonthlyIncome"].mean(), rel=1e-3)
    assert "employees" in r["answer"]                                  # EmployeeNumber names the rows


# ── Superstore ────────────────────────────────────────────────────────────────

def test_superstore_tables_lose_money(embedder):
    _, df, _ = load("superstoreSales.csv")
    r = analyze(df, "Why is profit lower for Furniture?", embed=embedder)
    furn = df["Product Category"] == "Furniture"
    tables = df["Product Name"].str.contains("table", case=False)
    assert df.loc[furn & tables, "Profit"].mean() < 0                    # the documented finding
    top = r["data"][0]
    assert "table" in top["factor"].lower() and top["metric_with"] < 0
    assert "−$" in r["answer"]                                           # negative money formatted


def test_superstore_false_premise_is_corrected(embedder):
    _, df, _ = load("superstoreSales.csv")
    best = df.groupby("Region")["Profit"].sum().idxmax()
    r = analyze(df, f"Why is profit lower in {best}?", embed=embedder)
    assert r["answer"].startswith(f"**{best} has the highest")


# ── Clothing reviews (free text) ──────────────────────────────────────────────

def test_reviews_trend_department(embedder):
    _, df, _ = load("womens_clothing_reviews.csv")
    lowest = df.groupby("Department Name")["Rating"].mean().idxmin()
    assert lowest == "Trend"
    r = analyze(df, "Why are ratings lower in the Trend department?", embed=embedder)
    f = factors(r)
    assert any(x.startswith("Review Text:") for x in f)                 # the cause is in the reviews
    assert not any(x == "Class Name: Trend" for x in f)                 # 100% of Trend: structure
    assert not any(x.startswith("Recommended") for x in f)              # outcome proxy, not a cause
    text = next(d for d in r["data"] if d["factor"].startswith("Review Text:"))
    assert text["metric_with"] < text["metric_without"]
    assert "&#39;" not in r["answer"]                                    # HTML entities decoded
