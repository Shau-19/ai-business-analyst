# tools/data_profile.py
"""
Deterministic dataset profile for the BI overview — no LLM involved.

Given a session's uploaded tables, returns for each one:
  - column roles (date / month / numeric / categorical / id / text) + stats
  - up to 3 KPI cards (total or average, with a per-period sparkline and change)
  - up to 2 auto charts (trend over time, breakdown by category) as plot specs
    in the same shape PlotBuilder produces ({type, x, y, datasets, ...})
  - suggested questions built from the real column names
  - the first rows, for a preview table

Everything is plain pandas, so the overview loads instantly and costs nothing.
"""

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MONTH_INDEX = {m: i for i, m in enumerate(MONTHS)}
MONTH_INDEX.update({
    "january": 0, "february": 1, "march": 2, "april": 3, "june": 5, "july": 6,
    "august": 7, "september": 8, "sept": 8, "october": 9, "november": 10, "december": 11,
})

DATE_HINTS = ("date", "time", "day", "week", "period", "timestamp", "created", "updated", "year")
# "customer_id", "Customer ID", "CustomerID", "OrderId", "invoice number" — but not
# "Amount Paid", "Bid" or "Prepaid", which merely end in the letters "id".
ID_HINT = re.compile(r"(^id$|[_\s\-.]id$|code$|key$|uuid|^no\.?$|[_\s]no\.?$|number$)", re.IGNORECASE)
ID_CAMEL = re.compile(r"[a-z0-9](ID|Id)$")


def _id_name(col) -> bool:
    return bool(ID_HINT.search(str(col).lower()) or ID_CAMEL.search(str(col)))
MEAN_HINT = re.compile(r"(rate|pct|percent|ratio|avg|average|mean|score|churn|margin|age|price|"
                       r"size|per_|_per|lpa|days|duration|rating|discount|temperature|tenure)", re.IGNORECASE)
MEASURE_PRIORITY = ("revenue", "sales", "amount", "profit", "cost", "spend", "price", "total",
                    "units", "quantity", "qty", "orders", "customers", "value", "salary")


def humanize(col: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(col)).replace("_", " ").strip()
    return s[:1].upper() + s[1:]


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _month_col(series: pd.Series) -> bool:
    vals = series.dropna().astype(str).str.strip().str.lower()
    return len(vals) > 0 and vals.isin(MONTH_INDEX.keys()).mean() >= 0.9


def _date_col(name: str, series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype != object:
        return False
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    # Require something date-shaped so plain words/IDs never parse as dates
    if not sample.str.contains(r"\d{4}|\d{1,2}[/-]\d{1,2}", regex=True).mean() >= 0.8:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.8


def classify(df: pd.DataFrame) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        name = str(col).lower()
        if _month_col(s):
            roles[col] = "month"
        elif _date_col(name, s) or (any(h in name for h in DATE_HINTS) and _date_col(name, s.astype(str))):
            roles[col] = "date"
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            is_idlike = _id_name(col) and s.nunique() >= 0.9 * n
            roles[col] = "id" if is_idlike else "numeric"
        elif _id_name(col) and s.nunique() >= 0.5 * n:
            roles[col] = "id"
        elif s.nunique(dropna=True) <= max(30, int(0.05 * n)) and s.nunique(dropna=True) < n:
            roles[col] = "categorical"
        else:
            roles[col] = "text"
    return roles


def _measures(df: pd.DataFrame, roles: Dict[str, str]) -> List[str]:
    nums = [c for c, r in roles.items() if r == "numeric"]

    def rank(c):
        n = str(c).lower()
        for i, key in enumerate(MEASURE_PRIORITY):
            if key in n:
                return i
        return len(MEASURE_PRIORITY)
    # Additive measures (revenue, units) make better headline KPIs than
    # per-unit ones (price, rating), so they lead.
    return sorted(nums, key=lambda c: (_agg_name(c) == "mean", rank(c), list(df.columns).index(c)))


def _agg_name(col: str) -> str:
    return "mean" if MEAN_HINT.search(str(col)) else "sum"


def _period_frame(df: pd.DataFrame, roles: Dict[str, str]):
    """Return (period_key Series, label list, sort order) for the time axis, or None."""
    month = next((c for c, r in roles.items() if r == "month"), None)
    if month:
        key = df[month].astype(str).str.strip().str.lower().map(MONTH_INDEX)
        labels = {i: df[month][key == i].iloc[0] for i in key.dropna().unique()}
        return key, (lambda i: str(labels.get(i, i))), month
    date = next((c for c, r in roles.items() if r == "date"), None)
    if date:
        d = pd.to_datetime(df[date], errors="coerce", format="mixed")
        if d.notna().sum() < 2:
            return None
        span_days = (d.max() - d.min()).days
        if span_days > 62:
            key = d.dt.to_period("M").astype(str)       # YYYY-MM sorts lexically
        else:
            key = d.dt.strftime("%Y-%m-%d")
        # astype(str) turns a missing date into the text "NaT", which then
        # showed up as the last month on charts and in "2023-01 → NaT".
        key = key.where(d.notna())
        return key, (lambda k: k), date
    return None


def _series_by_period(df, roles, measure):
    pf = _period_frame(df, roles)
    if not pf:
        return None
    key, label, axis_col = pf
    agg = _agg_name(measure)
    g = df.assign(_k=key).dropna(subset=["_k"]).groupby("_k")[measure].agg(agg).sort_index()
    if len(g) < 2:
        return None
    return {"axis": axis_col, "labels": [label(k) for k in g.index],
            "values": [_num(v) for v in g.values], "agg": agg}


def _typical(df: pd.DataFrame, measure: str):
    """
    (df with extreme values of `measure` blanked, how many, the largest).
    Three typo rows of 9,999 units turned a monthly trend of ~300 into three
    10K spikes and made one region look 9x the others. Charts use typical
    values and say what they left out; headline totals keep every row so
    they agree with what a SQL question returns.
    """
    from tools.data_clean import outlier_mask
    mask = outlier_mask(df[measure])
    if not mask.any():
        return df, 0, None
    vals = pd.to_numeric(df[measure], errors="coerce")
    out = df.copy()
    out[measure] = vals.where(~mask)
    return out, int(mask.sum()), _num(vals[mask].abs().max())


def _excluded_note(n: int, biggest) -> Optional[str]:
    if not n:
        return None
    size = f"{biggest:,.0f}" if biggest is not None and float(biggest).is_integer() else f"{biggest:,.2f}"
    return f"Excludes {n} extreme value{'s' if n > 1 else ''} ({size})"


def _kpis(df, roles, measures) -> List[Dict[str, Any]]:
    out = []
    for m in measures[:3]:
        agg = _agg_name(m)
        col = pd.to_numeric(df[m], errors="coerce")
        value = col.mean() if agg == "mean" else col.sum()
        kpi = {"column": m, "label": humanize(m), "agg": "Average" if agg == "mean" else "Total",
               "value": _num(value)}
        typical, n_ext, biggest = _typical(df, m)
        trend = _series_by_period(typical, roles, m)
        if trend and trend["values"][0] not in (None, 0) and trend["values"][-1] is not None:
            first, last = trend["values"][0], trend["values"][-1]
            kpi.update({
                "spark": trend["values"],
                "change_pct": round((last - first) / abs(first) * 100, 1),
                "from_label": trend["labels"][0], "to_label": trend["labels"][-1],
                "last": last,
            })
            if n_ext:
                kpi["trend_note"] = _excluded_note(n_ext, biggest)
        out.append(kpi)
    return out


def _charts(df, roles, measures) -> List[Dict[str, Any]]:
    charts = []
    if not measures:
        return charts
    m = measures[0]
    full = df
    df, n_ext, biggest = _typical(df, m)
    note = _excluded_note(n_ext, biggest)
    trend = _series_by_period(df, roles, m)
    if trend:
        label = f"{'Average' if trend['agg'] == 'mean' else 'Total'} {humanize(m).lower()}"
        charts.append({
            "kind": "trend",
            "note": note,
            "question": f"Show {humanize(m).lower()} over time",
            "plot": {"type": "line" if len(trend["labels"]) > 8 else "bar",
                     "x": trend["labels"], "y": trend["values"],
                     "datasets": [{"label": label, "data": trend["values"]}],
                     "x_label": humanize(trend["axis"]), "y_label": label,
                     "title": f"{humanize(m)} by {humanize(trend['axis']).lower()}"},
        })
    cats = [c for c, r in roles.items() if r == "categorical" and 2 <= full[c].nunique() <= 25]
    if cats:
        c = min(cats, key=lambda x: abs(full[x].nunique() - 6))   # prefer a readable 4–8 groups
        agg = _agg_name(m)
        g = (df.assign(_v=pd.to_numeric(df[m], errors="coerce"))
               .groupby(c)["_v"].agg(agg).sort_values(ascending=False).head(10))
        label = f"{'Average' if agg == 'mean' else 'Total'} {humanize(m).lower()}"
        charts.append({
            "kind": "breakdown",
            "note": note,
            "question": f"What is the {label.lower()} by {humanize(c).lower()}?",
            "plot": {"type": "bar", "x": [str(k) for k in g.index], "y": [_num(v) for v in g.values],
                     "datasets": [{"label": label, "data": [_num(v) for v in g.values]}],
                     "x_label": humanize(c), "y_label": label,
                     "title": f"{humanize(m)} by {humanize(c).lower()}"},
        })
    return charts


def _questions(roles, measures) -> List[str]:
    cats = [c for c, r in roles.items() if r == "categorical"]
    has_time = any(r in ("date", "month") for r in roles.values())
    qs = []
    if measures and cats:
        qs.append(f"Which {humanize(cats[0]).lower()} has the highest {humanize(measures[0]).lower()}?")
    if measures and has_time:
        qs.append(f"How did {humanize(measures[0]).lower()} change over time?")
    if len(measures) > 1:
        qs.append(f"Is there a relationship between {humanize(measures[0]).lower()} and {humanize(measures[1]).lower()}?")
    if measures and has_time:
        qs.append(f"Forecast {humanize(measures[0]).lower()} for the next 6 periods")
    qs.append("Summarize the key trends in this data")
    return qs[:5]


def _column_stats(df, roles) -> List[Dict[str, Any]]:
    cols = []
    for c in df.columns:
        s, role = df[c], roles[c]
        info = {"name": c, "label": humanize(c), "role": role,
                "nulls": int(s.isna().sum()), "distinct": int(s.nunique(dropna=True))}
        if role == "numeric":
            v = pd.to_numeric(s, errors="coerce")
            info.update({"min": _num(v.min()), "max": _num(v.max()), "mean": _num(v.mean()),
                         "sum": _num(v.sum())})
        elif role == "categorical":
            info["top"] = [{"value": str(k), "count": int(n)} for k, n in s.value_counts().head(5).items()]
        elif role == "date":
            d = pd.to_datetime(s, errors="coerce", format="mixed")
            info.update({"min": str(d.min().date()) if d.notna().any() else None,
                         "max": str(d.max().date()) if d.notna().any() else None})
        cols.append(info)
    return cols


def _why_question(df, roles, measures) -> Optional[str]:
    """'Why is customer rating lower in West?' for the weakest group on an average-type measure."""
    rated = [m for m in measures if _agg_name(m) == "mean" and not re.search(r"price|cost|age|discount", m, re.I)]
    rated.sort(key=lambda m: not re.search(r"rating|score|satisf|nps|csat|margin|conversion", m, re.I))
    cats = [c for c, r in roles.items() if r == "categorical" and 3 <= df[c].nunique() <= 12]
    if not rated or not cats:
        return None
    m, c = rated[0], min(cats, key=lambda x: abs(df[x].nunique() - 5))
    means = pd.to_numeric(df[m], errors="coerce").groupby(df[c]).mean().dropna()
    if len(means) < 3:
        return None
    return f"Why is {humanize(m).lower()} lower in {means.idxmin()}?"


def profile_table(df: pd.DataFrame, table: str, source: Optional[str] = None) -> Dict[str, Any]:
    roles = classify(df)
    measures = _measures(df, roles)
    questions = _questions(roles, measures)
    why = _why_question(df, roles, measures)
    if why:
        questions = (questions[:1] + [why] + questions[1:])[:5]
    return {
        "table": table,
        "source": source,
        "rows": int(len(df)),
        "columns": _column_stats(df, roles),
        "kpis": _kpis(df, roles, measures),
        "charts": _charts(df, roles, measures),
        "questions": questions,
        "sample": json.loads(df.head(25).to_json(orient="records", date_format="iso")),
    }


def build_profile(db_manager, tables: List[str], files: List[str], quality: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    quality = quality or {}
    datasets = []
    for t in tables:
        try:
            df = db_manager.execute_query(f'SELECT * FROM "{t}"')
        except Exception as e:
            datasets.append({"table": t, "error": str(e)})
            continue
        # Match the table back to its uploaded file name for display
        stem = re.sub(r"[^a-z0-9_]", "_", t.split("_", 2)[-1].lower())
        source = next((f for f in files if re.sub(r"[^a-z0-9_]", "_", f.rsplit(".", 1)[0].lower()) == stem), None)
        prof = profile_table(df, t, source)
        prof["quality"] = quality.get(t)
        datasets.append(prof)
    structured = {d.get("source") for d in datasets}
    documents = [f for f in files if f not in structured and not f.lower().endswith((".csv", ".xlsx", ".xls"))]
    return {"datasets": datasets, "documents": documents, "files": files}
