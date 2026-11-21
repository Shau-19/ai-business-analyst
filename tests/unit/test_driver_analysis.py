"""
"Why" questions answered from the data. Each synthetic table has a cause
planted in it, so the test knows what a correct answer must find.
"""

import numpy as np
import pandas as pd
import pytest

from tools.data_clean import clean_frame
from tools.driver_analysis import analyze, is_why_question, parse_question

rng = np.random.default_rng(7)


def planted(n=900, region_effect="West", problem_rate=(0.35, 0.10)):
    """Ratings drop when a note reports a problem; West gets more problems."""
    regions = rng.choice(["North", "South", "East", "West"], n)
    bad = np.where(regions == region_effect, rng.random(n) < problem_rate[0], rng.random(n) < problem_rate[1])
    good_notes = ["Delivered on time, happy.", "Smooth order, no issues.", "Repeat buyer."]
    bad_notes = ["Arrived damaged, wants refund.", "Late delivery, complained.", "Wrong size shipped."]
    notes = [rng.choice(bad_notes) if b else rng.choice(good_notes) for b in bad]
    rating = np.where(bad, rng.choice([1, 2, 2, 3], n), rng.choice([4, 4, 5, 5, 3], n)).astype(float)
    return pd.DataFrame({
        "order_id": [f"O{i}" for i in range(n)],
        "order_date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "region": regions,
        "category": rng.choice(["A", "B", "C"], n),
        "units": rng.integers(1, 20, n),
        "unit_price": rng.uniform(5, 100, n).round(2),
        "customer_rating": rating,
        "notes": notes,
    })


# ── detection + parsing ───────────────────────────────────────────────────────

@pytest.mark.parametrize("q", ["Why are ratings lower in West?", "what explains the drop in March",
                               "How come North sells more?", "What caused revenue to fall?",
                               "reasons behind the low score", "What drove the spike in June?"])
def test_why_questions_detected(q):
    assert is_why_question(q)


@pytest.mark.parametrize("q", ["How many orders are there?", "Show revenue by region", "whyte goods total"])
def test_other_questions_not_detected(q):
    assert not is_why_question(q)


def test_parse_metric_segment_and_direction():
    spec = parse_question(planted(), "Why are customer ratings lower in the West region?")
    assert (spec.metric, spec.segment_col, spec.segment_val, spec.direction, spec.agg) == \
        ("customer_rating", "region", "West", "lower", "mean")


def test_parse_derives_revenue_from_units_and_price():
    spec = parse_question(planted(), "Why is revenue higher in North?")
    assert spec.metric == "__revenue" and spec.derived_from[:2] == ["units", "unit_price"]
    assert spec.agg == "sum"


def test_parse_period_question():
    spec = parse_question(planted(), "Why did revenue drop in March 2024?")
    assert spec.period == "2024-03" and spec.segment_col is None


def test_unrelated_question_returns_none():
    assert analyze(planted(), "Why is the sky blue?") is None


# ── free text: words and word themes ──────────────────────────────────────────

@pytest.mark.parametrize("text, words", [
    ("Entregó tarde, caja dañada", ["entregó", "tarde", "caja", "dañada"]),
    ("डिलीवरी देर से हुई", ["डिलीवरी", "देर", "से", "हुई"]),
    ("Box crushed; screen cracked 😞", ["box", "crushed", "screen", "cracked"]),
])
def test_words_split_in_any_script(text, words):
    from tools.driver_analysis import _words
    assert _words(text) == words


def test_no_keyword_list_in_the_analyst():
    """Themes come from the data and the embedder, never from a built-in word list."""
    import tools.driver_analysis as da
    assert not any(hasattr(da, n) for n in ("PROBLEM_STEMS", "PROBLEM_WORDS", "_problem_hits"))


# ── segment gaps ──────────────────────────────────────────────────────────────

def test_finds_the_planted_cause_and_explains_the_gap(embedder):
    r = analyze(planted(), "Why are customer ratings lower in the West region?", embed=embedder)
    assert r and r["analysis"]["kind"] == "segment"
    kinds = [d["kind"] for d in r["data"]]
    assert kinds and set(kinds) <= {"text", "theme"}, r["data"]     # the notes, nothing random
    top = r["data"][0]
    assert top["segment_share"] > top["others_share"] and top["metric_with"] < top["metric_without"]
    assert "of the gap" in r["answer"]


def test_messy_file_west_ratings(messy_csv, embedder):
    df, _ = clean_frame(pd.read_csv(messy_csv, dtype=str, keep_default_na=False))
    r = analyze(df, "Why are customer ratings lower in the West region?", embed=embedder)
    assert r["answer"].startswith("**West has the lowest average customer rating: 3.51**")
    words = " ".join(d["factor"] for d in r["data"]).lower()
    assert "damaged" in words or "cracked" in words
    assert r["analysis"]["explained"] >= 0.9


def test_works_without_an_embedder(messy_csv):
    df, _ = clean_frame(pd.read_csv(messy_csv, dtype=str, keep_default_na=False))
    r = analyze(df, "Why are customer ratings lower in the West region?")
    assert r and all(d["kind"] in ("text", "category", "numeric") for d in r["data"])


def test_structural_features_are_not_causes():
    """A value that never (or always) occurs in the segment is structure, not a reason."""
    df = planted()
    df["region_group"] = np.where(df.region == "West", "Pacific", "Other")       # 100% of West
    df["service"] = np.where(df.region == "West", "Standard", rng.choice(["Standard", "None"], len(df)))
    r = analyze(df, "Why are customer ratings lower in the West region?")
    factors = [d["factor"] for d in r["data"]]
    assert not any("Region Group" in f or "Service: None" in f for f in factors), factors


def test_plural_segment_name_matches():
    df = planted()
    df["role"] = rng.choice(["Sales Representative", "Engineer", "Manager"], len(df))
    spec = parse_question(df, "Why are customer ratings lower for Sales Representatives?")
    assert (spec.segment_col, spec.segment_val) == ("role", "Sales Representative")


def test_yes_no_column_becomes_a_rate():
    df = planted()
    df["churned"] = np.where(rng.random(len(df)) < np.where(df.region == "West", 0.4, 0.1), "Yes", "No")
    r = analyze(df, "Why is churn higher in the West?")
    assert r["analysis"]["metric"] == "Churned rate"
    assert r["answer"].startswith("**West has the highest churned rate:")
    assert "%" in r["answer"].splitlines()[0]


def test_outcome_proxies_are_not_drivers():
    """'Recommended' moves in lockstep with 'Rating': the same outcome, not a cause."""
    df = planted()
    df["recommended"] = (df.customer_rating >= 4).astype(int)
    r = analyze(df, "Why are customer ratings lower in the West region?")
    assert not any(d["factor"] == "Recommended" for d in r["data"])


def test_no_real_gap_says_so_instead_of_inventing_causes():
    df = planted(problem_rate=(0.12, 0.12))
    r = analyze(df, "Why are customer ratings lower in the West region?")
    assert "about the same" in r["answer"] and "no real gap" in r["answer"]
    assert r["data"] == []


def test_volume_vs_value_for_totals():
    df = planted()
    extra = df[df.region == "North"].copy()
    extra["order_id"] = extra["order_id"] + "x"
    df = pd.concat([df, extra], ignore_index=True)        # North: twice the orders, same order size
    r = analyze(df, "Why is North revenue the highest?")
    assert "volume, not order size" in r["answer"]


def test_category_driver():
    df = planted(problem_rate=(0.1, 0.1))
    west = df.region == "West"
    df.loc[west, "category"] = np.where(rng.random(west.sum()) < 0.7, "C", "A")
    df.loc[df.category == "C", "customer_rating"] -= 1.5
    r = analyze(df, "Why are customer ratings lower in the West?")
    assert any(d["kind"] == "category" and d["factor"] == "Category: C" for d in r["data"]), r["data"]


def test_extreme_values_are_left_out_and_reported():
    df = planted()
    df.loc[[3, 50, 400], "units"] = 9999
    r = analyze(df, "Why is revenue higher in North?")
    assert "3 extreme values left out" in r["answer"]


def test_answers_never_mention_internals():
    r = analyze(planted(), "Why are customer ratings lower in the West region?")
    for word in ("SQL", "document", "dataframe", "None", "nan"):
        assert word not in r["answer"]


# ── period changes ────────────────────────────────────────────────────────────

def test_month_name_table_period_change():
    df = pd.DataFrame({"Month": ["Jan", "Feb", "Mar", "Apr"], "Revenue": [450, 520, 480, 550],
                       "New_Customers": [12, 15, 11, 18], "Churn_Rate": [2.1, 1.8, 2.3, 1.5]})
    r = analyze(df, "Why did revenue drop in March?")
    assert r["answer"].startswith("**Total revenue fell 7.7% in March")
    assert "New Customers" in r["answer"] and "Churn Rate" in r["answer"]


def test_period_contribution_by_category():
    n = 400
    df = pd.DataFrame({"date": np.repeat(["2024-01-15", "2024-02-15"], n // 2),
                       "product": np.tile(["Pen", "Ink", "Desk", "Lamp"], n // 4),
                       "revenue": 10.0})
    df.loc[(df.date == "2024-02-15") & (df["product"] == "Desk"), "revenue"] = 1.0
    r = analyze(df, "Why did revenue fall in February 2024?")
    assert "Product “Desk”" in r["answer"]


def test_first_period_has_nothing_to_compare():
    df = planted()
    assert analyze(df, "Why was revenue low in January 2024?") is None


# ── wording ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cols, noun", [(["Order ID", "x"], "orders"), (["Employee ID"], "employees"),
                                        (["customer_id"], "customers"), (["TicketID"], "tickets"),
                                        (["EmployeeNumber"], "employees"), (["Row ID", "Order ID"], "orders"),
                                        (["Phone Number", "x"], "rows"), (["id"], "rows")])
def test_unit_noun_from_id_column(cols, noun):
    from tools.driver_analysis import _unit_noun
    df = pd.DataFrame({c: [f"{c}{i}" for i in range(20)] for c in cols})
    assert _unit_noun(df) == noun


def test_unit_noun_ignores_ids_that_repeat():
    """'Clothing ID' on a reviews table names the item, not the row."""
    from tools.driver_analysis import _unit_noun
    df = pd.DataFrame({"Clothing ID": [1, 1, 2, 2, 3], "Review Text": list("abcde")})
    assert _unit_noun(df) == "rows"


def test_acronyms_keep_their_case():
    from tools.driver_analysis import _lc
    assert _lc("CSAT") == "CSAT" and _lc("Customer Rating") == "customer rating" and _lc("NPS Score") == "NPS score"
