"""Import-time cleaning: numbers, dates, categories, duplicates, outliers."""

import numpy as np
import pandas as pd
import pytest

from tools.data_clean import clean_frame, parse_number


# ── parse_number ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, value, pct", [
    ("$1,299.00", 1299.0, False),
    ("1,299", 1299.0, False),
    ("USD 42", 42.0, False),
    ("42 usd", 42.0, False),
    ("₹ 1,00,000", 100000.0, False),
    ("€5", 5.0, False),
    ("12 units", 12.0, False),
    ("5kg", 5.0, False),
    ("4.5/5", 4.5, False),
    ("(12.50)", -12.5, False),
    ("−3", -3.0, False),           # unicode minus
    ("-3", -3.0, False),
    ("15%", 15.0, True),
    ("15 %", 15.0, True),
    (" 7 ", 7.0, False),
    ("1e3", 1000.0, False),
    (".5", 0.5, False),
    (42, 42.0, False),
    (3.5, 3.5, False),
    (np.int64(9), 9.0, False),
])
def test_parse_number_formats(raw, value, pct):
    assert parse_number(raw) == (value, pct)


@pytest.mark.parametrize("raw", ["", "  ", "N/A", "n/a", "NA", "none", "null", "-", "--", "?", "unknown", "#N/A",
                                 None, float("nan")])
def test_parse_number_null_tokens(raw):
    assert parse_number(raw)[0] is None


@pytest.mark.parametrize("raw", ["abc", "twelve", "1.2.3", "12-34-56", "$", "%", "1 2 3 4 5 units of 9"])
def test_parse_number_rejects_garbage(raw):
    assert parse_number(raw)[0] is None


def test_parse_number_bool_is_not_a_number():
    assert parse_number(True)[0] is None


# ── numeric columns ───────────────────────────────────────────────────────────

def test_currency_column_becomes_float_and_is_reported():
    df, rep = clean_frame(pd.DataFrame({"Price": ["$10", "$20.50", "30", "n/a", "$1,000"]}))
    assert df["Price"].dtype == "float64"
    assert df["Price"].tolist()[:3] == [10.0, 20.5, 30.0]
    assert pd.isna(df["Price"][3]) and df["Price"][4] == 1000.0
    col = next(c for c in rep["columns"] if c["column"] == "Price")
    assert col["kind"] == "numeric" and any("converted to numbers" in t for t in col["changes"])


def test_whole_numbers_become_nullable_int():
    df, _ = clean_frame(pd.DataFrame({"Units": ["1", "2", "3 units", "", "5"]}))
    assert str(df["Units"].dtype) == "Int64"
    assert df["Units"].tolist()[:3] == [1, 2, 3]


def test_mixed_percent_column_becomes_fractions():
    df, rep = clean_frame(pd.DataFrame({"Discount": ["10%", "0.2", "15", "5 %", "0"]}))
    assert df["Discount"].round(4).tolist() == [0.1, 0.2, 0.15, 0.05, 0.0]


@pytest.mark.parametrize("name", ["order_id", "Customer ID", "id", "Zip", "PIN code", "Phone", "SKU", "Account"])
def test_identifier_columns_stay_text(name):
    df, _ = clean_frame(pd.DataFrame({name: ["00123", "00456", "10789", "20001"]}))
    assert df[name].tolist() == ["00123", "00456", "10789", "20001"]


def test_leading_zero_codes_stay_text_even_without_id_name():
    df, _ = clean_frame(pd.DataFrame({"Ref": ["0012", "0045", "0078", "1000", "0099"]}))
    assert df["Ref"].tolist()[0] == "0012"


def test_mostly_text_column_is_not_forced_numeric():
    df, _ = clean_frame(pd.DataFrame({"Mixed": ["1", "2", "abc", "def", "ghi"]}))
    assert df["Mixed"].tolist() == ["1", "2", "abc", "def", "ghi"]


def test_already_numeric_column_untouched():
    df, rep = clean_frame(pd.DataFrame({"x": [1.5, 2.5, 3.5]}))
    assert df["x"].tolist() == [1.5, 2.5, 3.5]
    assert rep["changed_columns"] == 0


def test_year_column_is_number_not_date():
    df, _ = clean_frame(pd.DataFrame({"Year": ["2021", "2022", "2023", "2024"]}))
    assert pd.api.types.is_integer_dtype(df["Year"])


# ── dates ─────────────────────────────────────────────────────────────────────

def test_mixed_date_formats_unified_day_first_inferred():
    raw = ["2024-01-15", "Jan 20, 2024", "23/02/2024", "03/03/2024", "15 Mar 2024"]
    df, rep = clean_frame(pd.DataFrame({"Order Date": raw}))
    assert df["Order Date"].tolist() == ["2024-01-15", "2024-01-20", "2024-02-23", "2024-03-03", "2024-03-15"]
    changes = next(c for c in rep["columns"] if c["column"] == "Order Date")["changes"]
    assert any("day/month/year" in t for t in changes)


def test_month_first_dates_inferred():
    df, _ = clean_frame(pd.DataFrame({"date": ["12/25/2024", "01/02/2024", "03/04/2024"]}))
    assert df["date"].tolist() == ["2024-12-25", "2024-01-02", "2024-03-04"]


def test_unparseable_dates_left_empty_and_reported():
    df, rep = clean_frame(pd.DataFrame({"Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "soon"]}))
    assert df["Date"].iloc[4] is None
    changes = next(c for c in rep["columns"] if c["column"] == "Date")["changes"]
    assert any("couldn’t be read as dates" in t for t in changes)


def test_text_with_occasional_dates_is_not_a_date_column():
    notes = ["call back on 2024-01-05", "ok", "great", "fine", "late", "n/a"]
    df, _ = clean_frame(pd.DataFrame({"Notes": notes}))
    assert df["Notes"].iloc[0] == "call back on 2024-01-05"


# ── categories ────────────────────────────────────────────────────────────────

def _col(counts):
    return [v for v, n in counts for _ in range(n)]


def test_case_space_and_typo_variants_merge():
    vals = _col([("North", 50), ("north", 5), ("NORTH ", 3), ("Nrth", 2), ("South", 40), ("Sout", 2),
                 ("East", 30), ("Est", 2), ("West", 30), ("Wst", 1)])
    df, rep = clean_frame(pd.DataFrame({"Region": vals, "n": range(len(vals))}))
    assert df["Region"].value_counts().to_dict() == {"North": 60, "South": 42, "East": 32, "West": 31}
    changes = next(c for c in rep["columns"] if c["column"] == "Region")["changes"]
    assert any("typos fixed" in t for t in changes)


def test_est_goes_to_east_not_west():
    vals = _col([("East", 40), ("West", 40), ("Est", 2), ("North", 40)])
    df, _ = clean_frame(pd.DataFrame({"Region": vals, "n": range(len(vals))}))
    assert df["Region"].value_counts()["East"] == 42


@pytest.mark.parametrize("common, rare", [("Male", "Female"), ("Mumbai", "Navi Mumbai"), ("Pro", "Pro Max"),
                                          ("Delhi", "New Delhi"), ("Paid", "Unpaid")])
def test_distinct_real_categories_never_merge(common, rare):
    vals = _col([(common, 60), (rare, 3), ("Other", 30)])
    df, _ = clean_frame(pd.DataFrame({"c": vals, "n": range(len(vals))}))
    assert df["c"].value_counts().get(rare) == 3


def test_one_letter_substitution_typo_merges():
    vals = _col([("Finance", 30), ("Finanse", 1), ("Sales", 40), ("Support", 35)])
    df, _ = clean_frame(pd.DataFrame({"Department": vals, "n": range(len(vals))}))
    assert df["Department"].value_counts().to_dict() == {"Sales": 40, "Support": 35, "Finance": 31}


@pytest.mark.parametrize("common, rare", [("Class A", "Class B"), ("Grade 10", "Grade 11"), ("Size M", "Size L"),
                                          ("Plan X1", "Plan X2")])
def test_short_or_numeric_token_differences_never_merge(common, rare):
    vals = _col([(common, 60), (rare, 3), ("Something else", 30)])
    df, _ = clean_frame(pd.DataFrame({"c": vals, "n": range(len(vals))}))
    assert df["c"].value_counts().get(rare) == 3


def test_placeholders_become_empty():
    vals = ["A", "B", "N/A", "-", "A", "B", "none", "A"]
    df, rep = clean_frame(pd.DataFrame({"Tier": vals, "n": range(len(vals))}))
    assert df["Tier"].isna().sum() == 3
    changes = next(c for c in rep["columns"] if c["column"] == "Tier")["changes"]
    assert any("placeholder" in t for t in changes)


def test_free_text_column_untouched():
    notes = [f"customer {i} said something different" for i in range(50)]
    df, rep = clean_frame(pd.DataFrame({"Notes": notes}))
    assert df["Notes"].tolist() == notes
    assert not any(c["column"] == "Notes" and c.get("changes") for c in rep["columns"])


def test_unicode_categories_survive():
    vals = _col([("दिल्ली", 20), ("मुंबई", 20), ("Chloé", 10)])
    df, _ = clean_frame(pd.DataFrame({"city": vals, "n": range(len(vals))}))
    assert set(df["city"]) == {"दिल्ली", "मुंबई", "Chloé"}


# ── rows, outliers, edge shapes ───────────────────────────────────────────────

def test_exact_duplicates_removed_after_trimming():
    df, rep = clean_frame(pd.DataFrame({"Order ID": ["A1", " A1", "A2"], "b": ["1", "1", "2"]}))
    assert rep["duplicates_removed"] == 1 and rep["rows_in"] == 3 and rep["rows_out"] == 2
    assert len(df) == 2


def test_near_duplicates_kept():
    _, rep = clean_frame(pd.DataFrame({"a": ["x", "x"], "b": ["1", "2"]}))
    assert rep["duplicates_removed"] == 0


def test_identical_rows_without_a_key_are_kept_and_reported():
    """Two identical sales in a log with no order ID can both be real."""
    raw = pd.DataFrame({"product": ["Pen", "Pen", "Ink"], "amount": ["5", "5", "9"]})
    df, rep = clean_frame(raw)
    assert len(df) == 3 and rep["duplicates_removed"] == 0
    assert rep.get("identical_rows_kept") == 1


def test_identical_rows_with_a_key_are_removed():
    raw = pd.DataFrame({"order_id": ["A1", "A1", "A2"], "amount": ["5", "5", "9"]})
    df, rep = clean_frame(raw)
    assert len(df) == 2 and rep["duplicates_removed"] == 1


def test_outlier_flagged_not_changed():
    vals = [10 + (i % 7) for i in range(100)] + [100000]
    df, rep = clean_frame(pd.DataFrame({"Units": vals}))
    assert df["Units"].max() == 100000
    col = next(c for c in rep["columns"] if c["column"] == "Units")
    assert col["warnings"] and "extreme value" in col["warnings"][0]


def test_column_names_trimmed():
    df, _ = clean_frame(pd.DataFrame({"  Revenue ": [1, 2, 3]}))
    assert list(df.columns) == ["Revenue"]


@pytest.mark.parametrize("frame", [
    pd.DataFrame(),
    pd.DataFrame({"a": []}),
    pd.DataFrame({"a": [None, None, None]}),
    pd.DataFrame({"a": ["only one"]}),
    pd.DataFrame({"a": [np.nan] * 5, "b": ["", "", "", "", ""]}),
    pd.DataFrame({"flag": [True, False, True]}),
])
def test_degenerate_frames_do_not_crash(frame):
    df, rep = clean_frame(frame)
    assert rep["rows_out"] == len(df)


def test_cleaning_is_idempotent(messy_csv):
    raw = pd.read_csv(messy_csv, dtype=str, keep_default_na=False)
    once, _ = clean_frame(raw)
    twice, rep2 = clean_frame(once)
    assert rep2["duplicates_removed"] == 0
    pd.testing.assert_frame_equal(once.astype(str), twice.astype(str))


def test_clean_sample_file_is_left_alone():
    from conftest import FIXTURES
    raw = pd.read_csv(FIXTURES / "Sales_Report_H1_2025.csv")
    df, rep = clean_frame(raw)
    assert rep["duplicates_removed"] == 0 and rep["changed_columns"] == 0
    pd.testing.assert_frame_equal(df, raw)


# ── the messy 1,000-row file end to end ───────────────────────────────────────

def test_messy_file_matches_hand_computed_answers(messy_csv, messy_expected):
    raw = pd.read_csv(messy_csv, dtype=str, keep_default_na=False)
    df, rep = clean_frame(raw)
    assert rep["rows_in"] == messy_expected["rows_in_file"]
    assert rep["duplicates_removed"] == messy_expected["duplicates"]
    assert len(df) == messy_expected["rows"]
    assert df["Region"].value_counts().to_dict() == messy_expected["regions"]
    for c in ["Units", "Unit Price ($)", "Discount", "Customer Rating"]:
        assert pd.api.types.is_numeric_dtype(df[c]), c
    assert int((df["Units"] < 0).sum()) == messy_expected["returns"]
    assert df["Order Date"].notna().mean() > 0.95
    revenue = (df["Units"] * df["Unit Price ($)"] * (1 - df["Discount"].fillna(0))).sum()
    assert round(float(revenue), 2) == pytest.approx(messy_expected["revenue_all"], abs=0.01)
    assert df["Order ID"].tolist()[0].startswith("ORD-")


def test_html_entities_decoded():
    df, _ = clean_frame(pd.DataFrame({"Review": ["it&#39;s great", "R&amp;D team", "fine"]}))
    assert df["Review"].tolist() == ["it's great", "R&D team", "fine"]
