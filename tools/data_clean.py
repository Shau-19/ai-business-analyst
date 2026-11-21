# tools/data_clean.py
"""
Ingest-time cleaning for uploaded CSV / Excel files.

Real exports are messy: numbers stored as "$1,299.00" or "12 units", dates in
five formats, "North" / "north " / "NORTH" / "Nrth" in one column, duplicate
rows. Imported as-is, every column lands in SQLite as TEXT, so SQL math
silently returns 0 for "$842.79", GROUP BY splits one region into five, and
date filters miss most rows.

clean_frame() fixes what it can be confident about, never deletes data
except exact duplicate rows, and returns a report of every change so the UI
can show a "Data health" summary. Things it can't fix safely (outliers,
unparseable values) are reported, not altered.
"""

import difflib
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

NULL_TOKENS = {"", "na", "n/a", "n.a.", "null", "none", "nan", "nil", "-", "--", "?", "unknown", "#n/a"}
CURRENCY_RE = re.compile(r"^(usd|inr|eur|gbp|rs\.?|us\$|\$|₹|€|£)\s*|\s*(usd|inr|eur|gbp)$", re.IGNORECASE)
TRAILING_UNIT_RE = re.compile(r"^([-+]?\d[\d.,]*)\s*[a-zA-Z]{1,12}$")      # "12 units", "5kg"
RATIO_RE = re.compile(r"^([-+]?\d+(?:\.\d+)?)\s*/\s*\d+(?:\.\d+)?$")          # "4.5/5"
NUMBER_RE = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)(e[-+]?\d+)?$", re.IGNORECASE)
KEEP_AS_TEXT_NAME = re.compile(r"(^id$|_id$|\bid\b|code|zip|pin|postal|phone|mobile|isbn|sku|account|gstin|pan\b)", re.IGNORECASE)
DATE_NAME = re.compile(r"(date|time|day|period|created|updated|timestamp|\bdt\b|month|year|week)", re.IGNORECASE)
DATE_SHAPE = re.compile(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
                        r"\d{1,2}[-\s][A-Za-z]{3,9}[-\s,]+\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}")


def _is_null(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip().lower() in NULL_TOKENS


def parse_number(v) -> Tuple[Optional[float], bool]:
    """Parse a messy numeric string. Returns (value or None, had_percent_sign)."""
    if _is_null(v):
        return None, False
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v), False
    s = str(v).strip().replace("−", "-").replace(" ", " ")
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1].strip()
    pct = "%" in s
    s = s.replace("%", "").strip()
    s = CURRENCY_RE.sub("", s).strip()
    m = RATIO_RE.match(s)
    if m:
        s = m.group(1)
    m = TRAILING_UNIT_RE.match(s)
    if m:
        s = m.group(1)
    s = s.replace(",", "").replace(" ", "")
    if not NUMBER_RE.match(s):
        return None, pct
    f = float(s)
    return (-f if neg else f), pct


def _try_numeric(col: pd.Series, name: str) -> Optional[Tuple[pd.Series, Dict[str, Any]]]:
    if KEEP_AS_TEXT_NAME.search(name):
        return None
    vals = col.dropna().astype(str).str.strip()
    vals = vals[~vals.str.lower().isin(NULL_TOKENS)]
    if len(vals) < 3:
        return None
    # Leading zeros ("00123") are codes, not quantities
    if (vals.str.match(r"^0\d+$")).mean() > 0.2:
        return None
    parsed = [parse_number(v) for v in col]
    ok = sum(1 for v, _ in parsed if v is not None)
    if ok / len(vals) < 0.9:
        return None
    nums = [v for v, _ in parsed]
    pct_flags = [p for _, p in parsed]
    notes = []
    if any(pct_flags):
        # Mixed "10%", "0.10", "15": a percent sign means /100; bare values
        # above 1 in a percent column are percents too; ≤ 1 are fractions.
        nums = [None if v is None else (v / 100 if (p or v > 1) else v) for v, p in zip(nums, pct_flags)]
        notes.append("percentages converted to fractions (10% → 0.10)")
    text_like = vals[~vals.str.match(r"^[-+]?\d+(\.\d+)?$")]
    if len(text_like):
        examples = list(dict.fromkeys(text_like.head(50)))[:3]
        notes.insert(0, f"{len(text_like)} values written as text ({', '.join(examples)}) converted to numbers")
    unparsed = len(vals) - ok
    if unparsed:
        notes.append(f"{unparsed} values couldn’t be read as numbers and were left empty")
    out = pd.Series(nums, index=col.index, dtype="float64")
    if out.dropna().apply(float.is_integer).all():
        out = out.astype("Int64")
    return out, {"kind": "numeric", "changes": notes}


def _dayfirst(vals: pd.Series) -> Optional[bool]:
    """Infer day/month order from unambiguous numeric dates like 23/05/2024."""
    first_big = second_big = 0
    for s in vals:
        m = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.]\d{2,4}$", s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            first_big += a > 12
            second_big += b > 12
    if first_big and not second_big:
        return True
    if second_big and not first_big:
        return False
    return None


def _try_date(col: pd.Series, name: str) -> Optional[Tuple[pd.Series, Dict[str, Any]]]:
    if pd.api.types.is_numeric_dtype(col):
        return None
    vals = col.dropna().astype(str).str.strip()
    vals = vals[~vals.str.lower().isin(NULL_TOKENS)]
    if len(vals) < 3:
        return None
    shaped = vals.str.contains(DATE_SHAPE).mean()
    if shaped < 0.8 and not (DATE_NAME.search(name) and shaped >= 0.5):
        return None
    dayfirst = _dayfirst(vals)
    parsed = pd.to_datetime(col.where(~col.astype(str).str.strip().str.lower().isin(NULL_TOKENS)),
                            errors="coerce", format="mixed", dayfirst=bool(dayfirst))
    ok = parsed.notna().sum()
    if ok / len(vals) < 0.8:
        return None
    formats = Counter()
    for s in vals.head(400):
        formats[re.sub(r"[A-Za-z]+", "Mon", re.sub(r"\d", "9", s))] += 1
    notes = []
    if len(formats) > 1:
        notes.append(f"{len(formats)} date formats unified to YYYY-MM-DD")
    if dayfirst is not None:
        notes.append("read as day/month/year" if dayfirst else "read as month/day/year")
    elif any(re.match(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$", s) for s in vals.head(400)):
        notes.append("ambiguous dates like 05/06 read as month/day")
    bad = len(vals) - ok
    if bad:
        notes.append(f"{bad} values couldn’t be read as dates and were left empty")
    out = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), None)
    return out, {"kind": "date", "changes": notes}


def _normalize_categories(col: pd.Series, name: str) -> Optional[Tuple[pd.Series, Dict[str, Any]]]:
    s = col.astype("string").str.strip()
    s = s.where(~s.str.lower().isin(NULL_TOKENS), None)
    non_null = s.dropna()
    if non_null.empty or non_null.str.len().mean() > 40 or KEEP_AS_TEXT_NAME.search(name):
        return None
    key = non_null.str.casefold().str.replace(r"\s+", " ", regex=True)
    if key.nunique() > 60 or key.nunique() > 0.5 * len(non_null):
        return None                                            # free text / high-cardinality
    # 1) case/space variants → the most common spelling of each key
    canon = {}
    for k, grp in non_null.groupby(key):
        canon[k] = grp.value_counts().idxmax()
    counts = key.value_counts()
    # 2) rare typos → a much more common spelling they're clearly a slip of.
    #    Conservative on purpose: the rare spelling must be the common one with
    #    letters dropped ("Nrth" ⊂ "North") or ≥ 90% similar, and ≥ 4× rarer.
    #    "Male" vs "Female" (80% similar, not a subsequence) never merges.
    def is_subsequence(short, long):
        # Typos keep their first letter: "Est" is East, not West.
        it = iter(long)
        return len(short) >= 3 and short[0] == long[0] and all(ch in it for ch in short)

    merges = {}
    top = counts.iloc[0]
    for k, c in counts.items():
        if c > 0.1 * top:
            continue
        candidates = [t for t, tc in counts.items() if t != k and tc >= 4 * c and t not in merges]
        best = None
        for t in candidates:
            ratio = difflib.SequenceMatcher(None, k, t).ratio()
            if (is_subsequence(k, t) and ratio >= 0.7) or ratio >= 0.9:
                if best is None or ratio > best[1]:
                    best = (t, ratio)
        if best:
            merges[k] = best[0]
    final_key = key.map(lambda k: merges.get(k, k))
    out = s.copy()
    out.loc[final_key.index] = final_key.map(canon).values

    before = non_null.nunique()
    after = out.dropna().nunique()
    raw = col.astype("string").str.strip()
    placeholders = int((raw.notna() & (raw != "") & raw.str.lower().isin(NULL_TOKENS)).sum())
    notes = []
    if after < before:
        detail = f"{before} spellings merged into {after} values"
        if merges:
            src = {k: non_null[key == k].iloc[0] for k in merges}
            detail += " (typos fixed: " + ", ".join(f"{src[k]} → {canon[v]}" for k, v in list(merges.items())[:4]) + ")"
        notes.append(detail)
    if placeholders:
        notes.append(f"{placeholders} placeholder values (N/A, -, none) set to empty")
    if not notes:
        return None
    return out.astype(object).where(out.notna(), None), {"kind": "category", "changes": notes}


def outlier_mask(col: pd.Series) -> pd.Series:
    """
    True for values more than 12 robust SDs (MAD) from the median — typos like
    9,999 units in a column whose typical value is 13. Empty mask when there
    are too few values, or so many extremes (> 2%) that they're a real pattern.
    """
    v = pd.to_numeric(col, errors="coerce")
    none = pd.Series(False, index=col.index)
    vals = v.dropna()
    if len(vals) < 20:
        return none
    med = vals.median()
    mad = (vals - med).abs().median() or (vals.std() / 1.4826 if vals.std() else 0)
    if not mad:
        return none
    mask = ((v - med).abs() > 12 * 1.4826 * mad).fillna(False)
    if not mask.any() or mask.sum() > 0.02 * len(vals):
        return none
    return mask


def _outliers(col: pd.Series) -> Optional[str]:
    mask = outlier_mask(col)
    if not mask.any():
        return None
    v = pd.to_numeric(col, errors="coerce").dropna()
    med = v.median()
    extreme = pd.to_numeric(col, errors="coerce")[mask]
    shown = ", ".join(f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}" for x in sorted(extreme.unique(), reverse=True)[:3])
    return f"{len(extreme)} extreme value{'s' if len(extreme) > 1 else ''} ({shown}) far from the typical {med:,.4g}; left as-is"


def _has_row_key(df: pd.DataFrame) -> bool:
    """True if some column identifies rows: unique across the de-duplicated frame."""
    distinct = df.drop_duplicates()
    if len(distinct) < 2:
        return False
    for c in distinct.columns:
        col = distinct[c].dropna()
        if len(col) < 0.98 * len(distinct) or col.nunique() != len(col):
            continue
        if KEEP_AS_TEXT_NAME.search(str(c)) or re.search(r"(^|\s|_)(no|number|ref|invoice|txn)\b", str(c), re.I):
            return True
        # Text keys like "ORD-00355" or timestamps; bare numbers are measures
        if col.dtype == object:
            s = col.astype(str)
            if s.str.contains(r"\d").all() and not s.str.match(r"^[-+$₹€£]?[\d.,\s%]+$").any():
                return True
    return False


def clean_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    report: Dict[str, Any] = {"rows_in": int(len(df)), "columns": []}

    # Trim whitespace in text cells before looking for duplicates
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    # Identical rows are only provably duplicates when the file has a key
    # (an order/customer ID that is unique once repeats are dropped). In a
    # log without one, two identical rows can be two real sales; removing
    # them would silently change every SUM, so report them instead.
    dup = int(df.duplicated().sum())
    report["duplicates_removed"] = 0
    if dup:
        if _has_row_key(df):
            df = df.drop_duplicates().reset_index(drop=True)
            report["duplicates_removed"] = dup
        else:
            report["identical_rows_kept"] = dup

    for c in df.columns:
        col = df[c]
        entry = None
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            for attempt in (_try_date, _try_numeric, _normalize_categories):
                res = attempt(col, c)
                if res:
                    df[c], entry = res
                    break
        if pd.api.types.is_numeric_dtype(df[c]):
            note = _outliers(df[c])
            if note:
                entry = entry or {"kind": "numeric", "changes": []}
                entry.setdefault("warnings", []).append(note)
        empty = int(pd.Series(df[c]).isna().sum())
        if entry or empty:
            entry = entry or {"kind": "unchanged", "changes": []}
            entry["column"] = c
            entry["empty"] = empty
            report["columns"].append(entry)

    report["rows_out"] = int(len(df))
    report["changed_columns"] = sum(1 for e in report["columns"] if e.get("changes"))
    return df, report
