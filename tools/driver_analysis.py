# tools/driver_analysis.py
"""
Answers "why" questions from the data itself.

"Why are ratings lower in the West?" is not a SQL question: one GROUP BY
shows *that* West is lower, not *why*. An analyst would compare West orders
with the rest on everything else that was recorded (product mix, sales rep,
discount, words in the notes) and keep the differences that line up with
the gap. That is what this module does, with plain statistics:

  1. Parse the question into a metric, a segment (Region = West) or a period
     (March 2025), and a direction. Heuristics first; an LLM only when the
     wording is unusual, and its answer is validated against real columns.
  2. Measure the gap (segment vs the rest, or period vs the previous one).
  3. Score candidate drivers. For a yes/no feature f (a category value or a
     word in a text column):
         effect = (share of f in segment - share in rest)
                  x (avg metric with f - avg metric without f)
     i.e. how much of the gap f's over-representation accounts for.
     Numeric columns use the same idea with a regression slope.
     Only drivers that point the same way as the gap, have enough rows and a
     share difference that isn't noise (|z| >= 2) are kept.
  4. Check the top drivers counterfactually: remove those rows from both
     sides and see how much of the gap is left.

Every number in the answer is computed here; nothing is written by an LLM.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tools.data_clean import outlier_mask
from tools.data_profile import MEAN_HINT, MONTHS, classify, humanize
from utils.logger import logger

WHY_RE = re.compile(
    r"\b(why|how come|what (?:explains|drives|drove|caused|causes|led to|is behind|made)|"
    r"reasons? (?:for|behind|why)|root cause|explain (?:the|why))\b", re.IGNORECASE)
LOWER_WORDS = re.compile(r"\b(lower|low|less|fewer|drop(?:ped)?|fell|fall(?:en|ing)?|declin\w*|decreas\w*|"
                         r"worse|poor\w*|down|dip(?:ped)?|weak\w*|bad|underperform\w*|slump\w*)\b", re.I)
HIGHER_WORDS = re.compile(r"\b(higher|high|more|rise|rose|risen|increas\w*|grew|grow\w*|better|up|"
                          r"jump\w*|spike\w*|strong\w*|outperform\w*|best|top)\b", re.I)
MONEY_RE = re.compile(r"(revenue|sales|amount|price|cost|profit|spend|value|income|\$|usd|inr|₹)", re.I)
FRACTION_NAME = re.compile(r"(discount|rate|pct|percent|share|margin|churn|ratio)", re.I)
REVENUE_WORDS = re.compile(r"\b(revenue|sales|turnover|income|order value|gmv)\b", re.I)
SYNONYMS = {
    "rating": ("rating", "score", "stars", "satisfaction", "csat", "nps"),
    "revenue": ("revenue", "sales", "turnover", "income", "amount"),
    "units": ("units", "quantity", "qty", "volume"),
    "price": ("price", "cost"),
}
STOPWORDS = set("""
a an the and or but if then than so to of in on at by for with without from into onto over under
is are was were be been being am do does did done have has had having it its this that these those
i we you he she they them our your their my me us his her very just also too not no nor only
customer customers order orders item items product products asked said says wants want will would
could should can may might per via about after before again all any each more most other some such
own same few both here there when where which who whom what how why one two three
""".split())

MONTH_NAMES = {m: i + 1 for i, m in enumerate(MONTHS)}
FULL_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
               "september", "october", "november", "december"]


def is_why_question(question: str) -> bool:
    return bool(WHY_RE.search(question or ""))


# ── question parsing ──────────────────────────────────────────────────────────

@dataclass
class Spec:
    metric: str                       # column name, or "__revenue" when derived
    metric_label: str
    agg: str                          # "mean" or "sum"
    segment_col: Optional[str] = None
    segment_val: Optional[str] = None
    period: Optional[str] = None      # "YYYY-MM" or month number as str for month-name columns
    period_col: Optional[str] = None
    direction: Optional[str] = None   # "lower" / "higher" / None
    derived_from: List[str] = field(default_factory=list)


def _stem(w: str) -> str:
    w = w.lower()
    if not w.isascii():
        return w                      # English suffix rules only make sense for English words
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)] + rep
    return w


_SPLIT_RE = re.compile(r"[\s\d.,;:!?¿¡\"“”‘’'()\[\]{}<>/\\|@#$%^&*+=~`_\-–—…·•।]+")


def _words(text: str) -> List[str]:
    """Words in any script: split on spaces/punctuation, so 'café', 'entregó' and Hindi survive."""
    return [w for w in _SPLIT_RE.split(str(text).lower()) if len(w) > 1]


def _name_tokens(col: str) -> List[str]:
    return [_stem(t) for t in re.findall(r"[a-z]+", humanize(col).lower()) if len(t) > 1]


def _find_derived_revenue(df: pd.DataFrame, roles: Dict[str, str]) -> Optional[Tuple[str, str, Optional[str]]]:
    nums = [c for c, r in roles.items() if r == "numeric"]
    qty = next((c for c in nums if re.search(r"(units|qty|quantity|volume)", c, re.I)), None)
    price = next((c for c in nums if re.search(r"price", c, re.I)), None)
    disc = next((c for c in nums if re.search(r"discount", c, re.I)), None)
    return (qty, price, disc) if qty and price else None


def _resolve_metric(df, roles, question) -> Optional[Tuple[str, str, List[str]]]:
    q_stems = {_stem(w) for w in _words(question)}
    nums = [c for c, r in roles.items() if r == "numeric"]
    best, best_score = None, 0
    for c in nums:
        toks = _name_tokens(c)
        score = sum(1 for t in toks if t in q_stems)
        for key, syns in SYNONYMS.items():
            if any(key in t for t in toks) and any(_stem(s) in q_stems for s in syns):
                score += 1
        if score > best_score or (score == best_score and score and len(c) < len(best)):
            best, best_score = c, score
    if REVENUE_WORDS.search(question) and not (best and MONEY_RE.search(best) and not re.search("price", best, re.I)):
        rev = next((c for c in nums if re.search(r"(revenue|sales|amount|turnover)", c, re.I)), None)
        if rev:
            return rev, humanize(rev), []
        parts = _find_derived_revenue(df, roles)
        if parts:
            return "__revenue", "Revenue", [p for p in parts if p]
    flag = _resolve_flag(df, roles, q_stems)
    if flag and (not best or flag[3] >= best_score):
        return flag[:3]
    if best:
        return best, humanize(best), []
    return None


POSITIVE_VALUES = {"yes", "y", "true", "t", "1", "churned", "left", "lost", "won", "converted", "returned",
                   "cancelled", "canceled", "default", "fraud"}


def _resolve_flag(df, roles, q_stems):
    """
    A yes/no column named in the question ("Why is attrition higher in Sales?")
    becomes a rate: the share of rows that are 'Yes'. Returns
    (metric, label, [column, positive value], name score) or None.
    """
    best = None
    for c, r in roles.items():
        vals = df[c].dropna().astype(str).str.strip()
        uniq = vals.unique()
        if len(uniq) != 2:
            continue
        score = sum(1 for t in _name_tokens(c) if t in q_stems)
        if not score:
            continue
        lower = {u.lower(): u for u in uniq}
        pos = next((lower[k] for k in lower if k in POSITIVE_VALUES), None) or vals.value_counts().idxmin()
        if not best or score > best[3]:
            best = ("__flag", f"{humanize(c)} rate", [c, pos], score)
    return best


def _resolve_segment(df, roles, question, skip: set) -> Optional[Tuple[str, str]]:
    q = question.lower()
    hits = []
    for c, r in roles.items():
        if r != "categorical" or c in skip:
            continue
        for v in df[c].dropna().astype(str).unique()[:60]:
            vs = v.strip()
            if len(vs) < 2:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(vs.lower())}(?:s|es)?(?![a-z0-9])", q):
                named = any(t in {_stem(w) for w in _words(q)} for t in _name_tokens(c))
                hits.append((named, len(vs), c, vs))
    if not hits:
        return None
    hits.sort(reverse=True)
    return hits[0][2], hits[0][3]


def _resolve_period(df, roles, question) -> Optional[Tuple[str, str]]:
    q = question.lower()
    month_col = next((c for c, r in roles.items() if r == "month"), None)
    date_col = next((c for c, r in roles.items() if r == "date"), None)
    m = re.search(r"\b(20\d{2})-(0[1-9]|1[0-2])\b", q)
    month, year = (int(m.group(2)), int(m.group(1))) if m else (None, None)
    if month is None:
        for i, name in enumerate(FULL_MONTHS):
            if re.search(rf"\b({name}|{name[:3]})\b", q) and not (name == "may" and re.search(r"\bmay (be|have|not)\b", q)):
                month = i + 1
                y = re.search(r"\b(20\d{2})\b", q)
                year = int(y.group(1)) if y else None
                break
    if month is None:
        return None
    if month_col:
        return month_col, str(month)
    if date_col:
        d = pd.to_datetime(df[date_col], errors="coerce")
        if year is None:
            years = d[d.dt.month == month].dt.year.dropna()
            if years.empty:
                return None
            year = int(years.max())
        return date_col, f"{year}-{month:02d}"
    return None


def parse_question(df: pd.DataFrame, question: str, roles: Optional[Dict[str, str]] = None) -> Optional[Spec]:
    roles = roles or classify(df)
    metric = _resolve_metric(df, roles, question)
    if not metric:
        return None
    col, label, parts = metric
    direction = "lower" if LOWER_WORDS.search(question) else "higher" if HIGHER_WORDS.search(question) else None
    agg = "mean" if col == "__flag" or (col != "__revenue" and MEAN_HINT.search(col)) else "sum"
    spec = Spec(metric=col, metric_label=label, agg=agg, direction=direction, derived_from=parts)
    seg = _resolve_segment(df, roles, question, skip={col})
    if seg:
        spec.segment_col, spec.segment_val = seg
        return spec
    per = _resolve_period(df, roles, question)
    if per:
        spec.period_col, spec.period = per
        return spec
    return None


def parse_with_llm(df: pd.DataFrame, question: str, llm, roles: Optional[Dict[str, str]] = None) -> Optional[Spec]:
    """Fallback for unusual wording. The model only picks names; they are validated here."""
    roles = roles or classify(df)
    nums = [c for c, r in roles.items() if r == "numeric"]
    cats = {c: [str(v) for v in df[c].dropna().unique()[:15]] for c, r in roles.items() if r == "categorical"}
    dates = [c for c, r in roles.items() if r in ("date", "month")]
    prompt = (
        "Map the question onto the table. Reply with JSON only:\n"
        '{"metric": <numeric column or "revenue">, "segment_column": <column or null>, '
        '"segment_value": <exact value or null>, "month": <"YYYY-MM" or null>}\n\n'
        f"Numeric columns: {nums}\nCategory columns and values: {json.dumps(cats, ensure_ascii=False)}\n"
        f"Date columns: {dates}\nQuestion: {question}"
    )
    try:
        raw = llm.invoke(prompt).content
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception as e:
        logger.info(f"🔎 Driver analysis: LLM parse failed ({e})")
        return None
    metric = data.get("metric")
    parts: List[str] = []
    if metric == "revenue" and metric not in nums:
        found = _find_derived_revenue(df, roles)
        if not found:
            return None
        metric, parts = "__revenue", [p for p in found if p]
    if metric not in nums and metric != "__revenue":
        return None
    label = "Revenue" if metric == "__revenue" else humanize(metric)
    agg = "mean" if metric != "__revenue" and MEAN_HINT.search(metric) else "sum"
    direction = "lower" if LOWER_WORDS.search(question) else "higher" if HIGHER_WORDS.search(question) else None
    spec = Spec(metric=metric, metric_label=label, agg=agg, direction=direction, derived_from=parts)
    sc, sv = data.get("segment_column"), data.get("segment_value")
    if sc in cats and sv is not None:
        match = next((v for v in df[sc].dropna().astype(str).unique() if v.lower() == str(sv).lower()), None)
        if match:
            spec.segment_col, spec.segment_val = sc, match
            return spec
    month = data.get("month")
    if month and re.fullmatch(r"20\d{2}-\d{2}", str(month)) and dates:
        spec.period_col, spec.period = dates[0], str(month)
        if roles[dates[0]] == "month":
            spec.period = str(int(str(month)[5:]))
        return spec
    return None


# ── formatting ────────────────────────────────────────────────────────────────

def _is_fraction(df, col) -> bool:
    if col not in df or not FRACTION_NAME.search(col):
        return False
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return not v.empty and v.min() >= 0 and v.max() <= 1


def _fmt(v: float, col: str, df: pd.DataFrame, total: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if col == "__flag" or _is_fraction(df, col):
        return f"{v * 100:.1f}%"
    money = bool(MONEY_RE.search(col)) or col == "__revenue"
    a = abs(v)
    if total and a >= 1e6:
        s = f"{v / 1e6:.2f}M"
    elif total and a >= 1e4:
        s = f"{v / 1e3:.1f}K"
    elif a >= 100 or float(v).is_integer():
        s = f"{v:,.0f}"
    else:
        s = f"{v:,.2f}"
    if money:
        return f"−${s.lstrip('-')}" if v < 0 else f"${s}"
    return s.replace("-", "−", 1) if v < 0 else s


def _pct(p: float) -> str:
    return f"{p * 100:.0f}%"


# ── features ──────────────────────────────────────────────────────────────────

def _metric_series(df: pd.DataFrame, spec: Spec) -> pd.Series:
    if spec.metric == "__flag":
        col, pos = spec.derived_from
        return (df[col].astype(str).str.strip() == pos).astype(float).where(df[col].notna())
    if spec.metric == "__revenue":
        q, p = (pd.to_numeric(df[c], errors="coerce") for c in spec.derived_from[:2])
        d = pd.to_numeric(df[spec.derived_from[2]], errors="coerce").fillna(0) if len(spec.derived_from) > 2 else 0
        return q * p * (1 - d)
    return pd.to_numeric(df[spec.metric], errors="coerce")


def _extreme_rows(df: pd.DataFrame, spec: Spec, y: pd.Series) -> pd.Series:
    """
    Rows to leave out as data-entry extremes. For a derived metric the typos
    live in its inputs (9,999 units), so check those; flagging the product
    would also drop legitimately large orders.
    """
    if spec.metric == "__flag":
        return pd.Series(False, index=df.index)
    if spec.derived_from:
        masks = [outlier_mask(df[c]) for c in spec.derived_from[:2]]
        return np.logical_or.reduce(masks) if masks else pd.Series(False, index=df.index)
    return outlier_mask(y)


def _lc(label: str) -> str:
    """Lowercase for mid-sentence use, keeping acronyms: 'CSAT' stays, 'Customer Rating' → 'customer rating'."""
    return " ".join(w if (w.isupper() and len(w) > 1) else w.lower() for w in str(label).split())


def _unit_noun(df: pd.DataFrame) -> str:
    """
    What one row is, from a column that identifies rows: 'Employee ID' /
    'EmployeeNumber' → 'employees', 'Ticket ID' → 'tickets'. The column must be
    unique per row: 'Clothing ID' on a reviews table names the item, not the row.
    """
    skip = re.compile(r"(row|record|index|line|serial|phone|mobile|contact|account|card|zip|pin|postal)", re.I)
    for c in df.columns:
        m = re.match(r"^\s*([A-Za-z][A-Za-z ]*?)[\s_-]*(?:id|#|number|no\.?)\s*$", str(c), re.I)
        if not m or not m.group(1).strip() or skip.search(m.group(1)):
            continue
        col = df[c].dropna()
        if len(col) and col.nunique() >= 0.98 * len(df):
            return _plural(m.group(1).strip().split()[-1].lower())
    return "rows"


def _possessive(name: str) -> str:
    return f"{name}’" if name.endswith("s") else f"{name}’s"


def _text_terms(series: pd.Series) -> Tuple[List[set], Dict[str, Counter]]:
    """Per-row stem sets for a free-text column, plus the surface forms seen per stem."""
    rows, forms = [], {}
    for t in series.fillna("").astype(str):
        stems = set()
        for w in _words(t):
            if len(w) < 3 or w in STOPWORDS:
                continue
            s = _stem(w)
            stems.add(s)
            forms.setdefault(s, Counter())[w] += 1
        rows.append(stems)
    return rows, forms


def _sentence_like(series: pd.Series) -> bool:
    s = series.dropna().astype(str)
    return not s.empty and s.str.len().mean() > 20 and s.str.split().str.len().mean() >= 3


_VEC_CACHE: Dict[Tuple[int, str], np.ndarray] = {}


def _embed_cached(embed, words: List[str]) -> np.ndarray:
    """Word vectors are reused across questions: a vocabulary barely changes between them."""
    key = id(embed)
    missing = [w for w in words if (key, w) not in _VEC_CACHE]
    if missing:
        if len(_VEC_CACHE) > 50_000:
            _VEC_CACHE.clear()
            missing = list(words)
        for w, v in zip(missing, embed(missing)):
            _VEC_CACHE[(key, w)] = np.asarray(v, dtype=float)
    return np.vstack([_VEC_CACHE[(key, w)] for w in words])


def _word_themes(rows: List[set], forms: Dict[str, Counter], n: int, embed=None, max_vocab: int = 600):
    """
    Candidate text features: single words, plus groups of related words.

    Related words are found by embedding the vocabulary (not the rows, so the
    cost doesn't grow with table size) and clustering it hierarchically: every
    node of the tree ("damaged, broken, cracked", "late, delayed, slow", and
    the larger nodes that join them) becomes a candidate "rows that mention
    any of these words". No word list is built in; which theme matters is
    decided by the statistics, in whatever language the embedder handles.
    Yields (kind, label, member stems, mask).
    """
    counts = Counter(s for st in rows for s in st)
    vocab = [s for s, k in counts.most_common(max_vocab) if max(5, 0.005 * n) <= k <= 0.6 * n]
    if not vocab:
        return
    index = {s: i for i, s in enumerate(vocab)}
    word_rows = [np.zeros(n, dtype=bool) for _ in vocab]
    for r, st in enumerate(rows):
        for s in st:
            i = index.get(s)
            if i is not None:
                word_rows[i][r] = True
    surface = [forms[s].most_common(1)[0][0] for s in vocab]
    for i, s in enumerate(vocab):
        yield "text", surface[i], [s], word_rows[i]
    if embed is None or len(vocab) < 4:
        return
    try:
        from scipy.cluster.hierarchy import linkage
        vecs = _embed_cached(embed, surface)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        tree = linkage(vecs, method="average", metric="cosine")
    except Exception as e:
        logger.info(f"🔎 Word themes unavailable ({e}); using single words")
        return
    members: List[List[int]] = [[i] for i in range(len(vocab))]
    masks: List[np.ndarray] = list(word_rows)
    seen = set()
    for a, b, _dist, _size in tree:
        m = members[int(a)] + members[int(b)]
        mask = masks[int(a)] | masks[int(b)]
        members.append(m)
        masks.append(mask)
        share = mask.mean()
        if len(m) > 40 or not (0.03 <= share <= 0.7):
            continue
        key = mask.tobytes()
        if key in seen:
            continue
        seen.add(key)
        yield "theme", ", ".join(surface[i] for i in m[:4]), [vocab[i] for i in m], mask


def _binary_features(df: pd.DataFrame, roles: Dict[str, str], spec: Spec, exclude: set, embed=None):
    """Yield (kind, column, label, mask, stems) for category values, words and word themes."""
    n = len(df)
    for c, r in roles.items():
        if c in exclude:
            continue
        if r == "categorical" and _sentence_like(df[c]):
            r = "text"            # a notes column with a few repeated sentences is still prose
        if r == "categorical" and 2 <= df[c].nunique() <= 30:
            col = df[c].astype(str)
            for v in df[c].dropna().astype(str).unique():
                mask = (col == v).to_numpy()
                if 5 <= mask.sum() <= 0.9 * n:
                    yield "category", c, v, mask, None
        elif r == "text":
            rows, forms = _text_terms(df[c])
            for kind, label, stems, mask in _word_themes(rows, forms, n, embed):
                yield kind, c, label, mask, stems


@dataclass
class Driver:
    kind: str
    column: str
    label: str
    effect: float
    share_seg: float = 0.0
    share_rest: float = 0.0
    y_with: float = float("nan")
    y_without: float = float("nan")
    mean_seg: float = float("nan")
    mean_rest: float = float("nan")
    mask: Any = None
    example: Optional[str] = None
    stems: Optional[List[str]] = None


def _find_drivers(df, roles, spec, y, seg, rest, gap, exclude, embed=None) -> List[Driver]:
    ns, nr = seg.sum(), rest.sum()
    both = seg | rest
    found: List[Driver] = []
    for kind, col, label, mask, stems in _binary_features(df, roles, spec, exclude, embed):
        ps, pr = mask[seg].mean(), mask[rest].mean()
        # Never/always present in the segment is structure, not a cause:
        # "0% of Fiber customers have No internet service", "100% of Trend
        # items are class Trend", "0% of Sales are Manufacturing Directors".
        if ps == 0 or ps >= 0.995 or pr >= 0.995:
            continue
        if mask[seg].sum() < max(5, 0.03 * ns) and mask[rest].sum() < max(5, 0.03 * nr):
            continue
        p = mask[both].mean()
        se = math.sqrt(max(p * (1 - p), 1e-9) * (1 / ns + 1 / nr))
        # Text yields hundreds of candidates (words and word groups), so it
        # needs stronger evidence than a handful of category values.
        z_min = 3.0 if kind in ("text", "theme") else 2.0
        if abs(ps - pr) < 0.03 or abs(ps - pr) / se < z_min:
            continue
        with_f, without_f = y[both & mask], y[both & ~mask]
        if len(with_f) < 8 or len(without_f) < 8:
            continue
        d = with_f.mean() - without_f.mean()
        effect = (ps - pr) * d
        if gap and np.sign(effect) == np.sign(gap) and abs(effect) >= 0.1 * abs(gap):
            found.append(Driver(kind, col, label, effect, ps, pr, with_f.mean(), without_f.mean(),
                                mask=mask, stems=stems))
    for c, r in roles.items():
        if r != "numeric" or c in exclude:
            continue
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        ok = both & ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 20 or np.nanstd(x[ok]) == 0:
            continue
        # A column that moves almost in lockstep with the metric ("Recommended"
        # next to "Rating") is the same outcome measured twice, not a cause.
        if abs(np.corrcoef(x[ok], y[ok])[0, 1]) >= 0.7:
            continue
        slope = np.cov(x[ok], y[ok])[0, 1] / np.var(x[ok], ddof=1)
        xs, xr = np.nanmean(x[seg & ok]), np.nanmean(x[rest & ok])
        sd = np.nanstd(x[ok])
        if abs(xs - xr) < 0.2 * sd:
            continue
        effect = (xs - xr) * slope
        if gap and np.sign(effect) == np.sign(gap) and abs(effect) >= 0.1 * abs(gap):
            found.append(Driver("numeric", c, humanize(c), effect, mean_seg=xs, mean_rest=xr))
    found.sort(key=lambda dr: abs(dr.effect), reverse=True)

    # One theme per group of words that appear in the same rows
    # ("arrived" and "damaged" in "Box arrived damaged").
    picked: List[Driver] = []
    for dr in found:
        if dr.mask is not None and any(
                p.mask is not None and (dr.mask & p.mask).sum() >= 0.6 * min(dr.mask.sum(), p.mask.sum())
                for p in picked):
            continue
        picked.append(dr)
        if len(picked) == 4:
            break
    return picked


def _theme_words(df, dr: Driver, seg) -> str:
    """The theme's words most used in the segment, as written: 'damaged, late, refund'."""
    rows, forms = _text_terms(df.loc[seg & dr.mask, dr.column])
    members = set(dr.stems or [])
    counts = Counter(s for st in rows for s in st if s in members)
    return ", ".join(forms[s].most_common(1)[0][0] for s, _ in counts.most_common(4)) or dr.label


def _example(df, dr: Driver, seg, y=None, gap: float = 0.0) -> Optional[str]:
    """
    A short note that shows the theme: from the segment's rows that have it,
    on the side of the gap (a low-rated review for "why lower"), shortest first.
    """
    if dr.kind not in ("text", "theme"):
        return None
    pick = seg & dr.mask
    if y is not None and gap:
        side = pick & ((y < np.nanmean(y[~seg])) if gap < 0 else (y > np.nanmean(y[~seg])))
        if side.any():
            pick = side
    rows = df.loc[pick, dr.column].dropna().astype(str).str.strip()
    rows = rows[rows.str.len() >= 20] if (rows.str.len() >= 20).any() else rows
    if rows.empty:
        return None
    counts = rows.value_counts()
    top = counts[counts == counts.max()].index
    text = min(top, key=len)
    return text if len(text) <= 140 else text[:137] + "…"


# ── analyses ──────────────────────────────────────────────────────────────────

def _segment_gap(df: pd.DataFrame, roles, spec: Spec, embed=None) -> Dict[str, Any]:
    y_all = _metric_series(df, spec)
    extreme = _extreme_rows(df, spec, y_all)
    keep = (~extreme & y_all.notna() & df[spec.segment_col].notna()).to_numpy()
    data = df[keep].reset_index(drop=True)
    y = y_all[keep].reset_index(drop=True).to_numpy(dtype=float)
    segcol = data[spec.segment_col].astype(str)
    seg = (segcol == spec.segment_val).to_numpy()
    rest = ~seg
    if seg.sum() < 5 or rest.sum() < 5:
        return {}

    per_group = pd.Series(y).groupby(segcol.values).agg(["mean", "sum", "count"])
    mean_seg, mean_rest = y[seg].mean(), y[rest].mean()
    peers = per_group.drop(index=spec.segment_val, errors="ignore")
    if spec.agg == "mean":
        headline_seg, headline_rest = mean_seg, mean_rest
    else:
        headline_seg, headline_rest = per_group.loc[spec.segment_val, "sum"], peers["sum"].mean()
    rank = (per_group[spec.agg].rank(ascending=False).loc[spec.segment_val], len(per_group))
    gap_row = mean_seg - mean_rest
    se = math.sqrt(np.var(y[seg], ddof=1) / seg.sum() + np.var(y[rest], ddof=1) / rest.sum())
    row_gap_real = bool(se and abs(gap_row) / se >= 2 and abs(gap_row) >= 0.02 * abs(mean_rest or 1))
    count_gap_real = spec.agg == "sum" and abs(seg.sum() / max(peers["count"].mean(), 1) - 1) >= 0.1

    exclude = {spec.segment_col, spec.metric, *spec.derived_from}
    drivers = _find_drivers(data, roles, spec, y, seg, rest, gap_row, exclude, embed)
    low = gap_row < 0

    # Counterfactual on the yes/no drivers: drop their rows from both sides
    cf = None
    masks = [d.mask for d in drivers if d.mask is not None][:3]
    if masks and gap_row:
        m = np.logical_or.reduce(masks)
        if (seg & ~m).sum() >= 5 and (rest & ~m).sum() >= 5:
            remaining = y[seg & ~m].mean() - y[rest & ~m].mean()
            cf = {"seg": y[seg & ~m].mean(), "rest": y[rest & ~m].mean(),
                  "explained": max(0.0, min(1.0, 1 - remaining / gap_row))}
    for d in drivers:
        d.example = _example(data, d, seg, y, gap_row)
        if d.kind == "theme":
            d.label = _theme_words(data, d, seg)

    return {"kind": "segment", "data": data, "y": y, "seg": seg, "rest": rest, "per_group": per_group,
            "headline": (headline_seg, headline_rest), "means": (mean_seg, mean_rest), "rank": rank,
            "counts": (int(seg.sum()), int(rest.sum()), float(peers["count"].mean())),
            "drivers": drivers if row_gap_real else [], "counterfactual": cf if row_gap_real else None,
            "excluded": int(extreme.sum()), "row_gap_real": row_gap_real, "count_gap_real": count_gap_real}


def _period_key(df, spec: Spec, roles) -> pd.Series:
    if roles.get(spec.period_col) == "month":
        return df[spec.period_col].astype(str).str.strip().str.lower().str[:3].map(MONTH_NAMES).astype("Int64").astype(str)
    d = pd.to_datetime(df[spec.period_col], errors="coerce")
    return d.dt.to_period("M").astype(str).where(d.notna())


def _period_change(df: pd.DataFrame, roles, spec: Spec, embed=None) -> Dict[str, Any]:
    y_all = _metric_series(df, spec)
    extreme = _extreme_rows(df, spec, y_all)
    key = _period_key(df, spec, roles)
    keep = (~extreme & y_all.notna() & key.notna() & (key != "<NA>")).to_numpy()
    data = df[keep].reset_index(drop=True)
    y = y_all[keep].reset_index(drop=True).to_numpy(dtype=float)
    k = key[keep].reset_index(drop=True)
    order = sorted(k.unique(), key=lambda s: (len(s), s)) if roles.get(spec.period_col) == "month" else sorted(k.unique())
    if spec.period not in order or order.index(spec.period) == 0:
        return {}
    prev = order[order.index(spec.period) - 1]
    cur_m, prev_m = (k == spec.period).to_numpy(), (k == prev).to_numpy()
    agg = spec.agg
    val = lambda m: (y[m].mean() if agg == "mean" else y[m].sum())  # noqa: E731
    v_cur, v_prev = val(cur_m), val(prev_m)
    series = pd.Series(y).groupby(k.values).agg(agg).reindex(order)

    contributions = []
    if agg == "sum":
        for c, r in roles.items():
            if r != "categorical" or c in (spec.period_col,) or not (2 <= data[c].nunique() <= 30):
                continue
            g_cur = pd.Series(y[cur_m]).groupby(data.loc[cur_m, c].astype(str).values).sum()
            g_prev = pd.Series(y[prev_m]).groupby(data.loc[prev_m, c].astype(str).values).sum()
            delta = g_cur.sub(g_prev, fill_value=0).sort_values()
            total = v_cur - v_prev
            if total and len(delta):
                top = delta.iloc[0] if total < 0 else delta.iloc[-1]
                contributions.append((abs(top / total), c, delta.index[0] if total < 0 else delta.index[-1], top))
        contributions.sort(reverse=True)

    co_moves = []
    for c, r in roles.items():
        if r != "numeric" or c in (spec.metric, *spec.derived_from):
            continue
        x = pd.to_numeric(data[c], errors="coerce")
        xa = x[cur_m].mean() if MEAN_HINT.search(c) else x[cur_m].sum()
        xb = x[prev_m].mean() if MEAN_HINT.search(c) else x[prev_m].sum()
        if xb and not math.isnan(xa) and not math.isnan(xb):
            co_moves.append((abs(xa / xb - 1), c, xb, xa))
    co_moves.sort(reverse=True)

    drivers: List[Driver] = []
    if agg == "mean" and cur_m.sum() >= 10 and prev_m.sum() >= 10:
        drivers = _find_drivers(data, roles, spec, y, cur_m, prev_m, v_cur - v_prev,
                                {spec.period_col, spec.metric, *spec.derived_from}, embed)

    return {"kind": "period", "data": data, "series": series, "cur": spec.period, "prev": prev,
            "values": (v_cur, v_prev), "counts": (int(cur_m.sum()), int(prev_m.sum())),
            "contributions": contributions[:2], "co_moves": co_moves[:2], "drivers": drivers,
            "excluded": int(extreme.sum())}


# ── wording ───────────────────────────────────────────────────────────────────

def _period_label(p: str, roles, spec) -> str:
    if roles.get(spec.period_col) == "month":
        return FULL_MONTHS[int(p) - 1].title()
    y, m = p.split("-")
    return f"{FULL_MONTHS[int(m) - 1].title()} {y}"


def _driver_line(dr: Driver, spec: Spec, df, seg_name: str, rest_name: str, unit: str) -> str:
    metric_col = spec.metric
    one = unit[:-1] if unit.endswith("s") else unit
    per = "" if spec.agg == "mean" else f" per {one}"
    metric = _lc(spec.metric_label)

    def outcome(prefix: str) -> str:
        return (f"{prefix} average {_fmt(dr.y_with, metric_col, df)} {metric}{per}, against "
                f"{_fmt(dr.y_without, metric_col, df)} for the rest.")

    if dr.kind in ("text", "theme"):
        words = f"“{dr.label}”" if dr.kind == "text" else dr.label
        line = (f"**{_pct(dr.share_seg)} of {seg_name} {unit} mention {words}** in "
                f"{humanize(dr.column).lower()}, vs {_pct(dr.share_rest)} {rest_name}. " + outcome(f"Those {unit}"))
        if dr.example:
            line += f" Typical note: “{dr.example}”"
        return line
    if dr.kind == "category":
        what = f"{humanize(dr.column).lower()} “{dr.label}”"
        if dr.share_rest == 0:
            # A group that only exists inside the segment (Sales Representatives within Sales)
            return (f"**{what}** ({_pct(dr.share_seg)} of {seg_name} {unit}, a group found only in "
                    f"{seg_name}): " + outcome("they"))
        return (f"**{_pct(dr.share_seg)} of {seg_name} {unit} have {what}**, vs {_pct(dr.share_rest)} "
                f"{rest_name}. " + outcome(f"Those {unit}"))
    direction = "lower" if (dr.effect < 0) == (dr.mean_seg > dr.mean_rest) else "higher"
    return (f"**{dr.label} averages {_fmt(dr.mean_seg, dr.column, df)}** for {seg_name} vs "
            f"{_fmt(dr.mean_rest, dr.column, df)} {rest_name}; higher {_lc(dr.label)} is linked to "
            f"{direction} {metric}.")


def _plural(word: str) -> str:
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word if word.endswith("s") else word + "s"


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _segment_answer(res, spec: Spec, df) -> str:
    seg_name = spec.segment_val
    group_word = humanize(spec.segment_col).lower()
    rest_name = "elsewhere"
    unit = _unit_noun(df)
    hs, hr = res["headline"]
    ms, mr = res["means"]
    total = spec.agg == "sum"
    what = (_lc(spec.metric_label) if spec.metric == "__flag" else f"average {_lc(spec.metric_label)}" if spec.agg == "mean" else f"total {_lc(spec.metric_label)}")
    r, n = res["rank"]
    pos = ("the lowest" if r == n else "the highest" if r == 1
           else f"the {_ordinal(int(r))}-highest" if r <= n / 2 else f"the {_ordinal(int(n - r + 1))}-lowest")
    others = f"{'on average ' if total else ''}for the other {_plural(group_word)}"
    ns, nr, avg_n = res["counts"]
    if not res["row_gap_real"] and not res["count_gap_real"]:
        return (f"**{_possessive(seg_name)} {what} ({_fmt(hs, spec.metric, df, total)}) is about the same as the "
                f"other {_plural(group_word)} ({_fmt(hr, spec.metric, df, total)}).** The difference is within "
                f"normal variation for {ns:,} {unit}, so there’s no real gap to explain."
                + (f"\n\n_{res['excluded']} extreme values left out._" if res["excluded"] else ""))
    lines = [f"**{seg_name} has {pos} {what}: {_fmt(hs, spec.metric, df, total)}** vs "
             f"{_fmt(hr, spec.metric, df, total)} {others}."]
    if total:
        count_note = (f"{seg_name} has {ns:,} {unit} vs {avg_n:,.0f} on average elsewhere, and "
                      f"{_fmt(ms, spec.metric, df)} per {unit[:-1]} vs {_fmt(mr, spec.metric, df)}.")
        lines.append(count_note)
    drivers = res["drivers"]
    if total and res["count_gap_real"] and not res["row_gap_real"]:
        lines.append(f"\nThe difference is volume, not order size: {seg_name} has "
                     f"{'fewer' if ns < avg_n else 'more'} {unit}, while the {unit[:-1]} value is about the same.")
    elif not drivers:
        lines.append(f"\nNothing else recorded in the data lines up with the difference: {seg_name} "
                     f"{unit} look like the rest on every other column, so the cause isn't in this file.")
    else:
        lines.append(f"\n**What’s different about {seg_name} {unit}:**")
        for dr in drivers[:3]:
            lines.append(f"- {_driver_line(dr, spec, df, seg_name, rest_name, unit)}")
        cf = res["counterfactual"]
        if cf and cf["explained"] >= 0.3:
            per = "" if spec.agg == "mean" else f" per {unit[:-1]}"
            lines.append(
                f"\nLeaving those {unit} out, {seg_name} averages {_fmt(cf['seg'], spec.metric, df)}{per} vs "
                f"{_fmt(cf['rest'], spec.metric, df)} {rest_name}, so they account for about "
                f"**{_pct(cf['explained'])} of the gap**.")
    foot = f"\n_Based on {ns:,} {seg_name} {unit} and {nr:,} others"
    if res["excluded"]:
        foot += f"; {res['excluded']} extreme values left out"
    lines.append(foot + "._")
    return "\n".join(lines)


def _period_answer(res, spec: Spec, df, roles) -> str:
    cur, prev = _period_label(res["cur"], roles, spec), _period_label(res["prev"], roles, spec)
    v_cur, v_prev = res["values"]
    total = spec.agg == "sum"
    change = (v_cur / v_prev - 1) if v_prev else float("nan")
    verb = "fell" if v_cur < v_prev else "rose" if v_cur > v_prev else "held flat"
    what = spec.metric_label if spec.metric == "__flag" else f"{'Total' if total else 'Average'} {_lc(spec.metric_label)}"
    lines = [f"**{what} {verb} {abs(change) * 100:.1f}% in {cur}: {_fmt(v_cur, spec.metric, df, total)}** "
             f"vs {_fmt(v_prev, spec.metric, df, total)} in {prev}."]
    nc, np_ = res["counts"]
    if total and nc != np_ and nc > 1:
        unit = _unit_noun(df)
        lines.append(f"There were {nc:,} {unit} in {cur} vs {np_:,} in {prev}.")
    bullets = []
    for share, col, val, delta in res["contributions"]:
        if share >= 0.25:
            amount = f"{'−' if delta < 0 else '+'}{_fmt(abs(delta), spec.metric, df, True)}"
            how = ("more than the whole change (other groups moved the other way)" if share > 1.05
                   else f"{_pct(share)} of the change")
            bullets.append(f"**{humanize(col)} “{val}”**: {amount}, {how}.")
    for _, col, before, after in res["co_moves"]:
        if before and abs(after / before - 1) >= 0.05:
            bullets.append(f"**{humanize(col)}** moved from {_fmt(before, col, df)} to {_fmt(after, col, df)} "
                           f"({(after / before - 1) * 100:+.0f}%) over the same months.")
    for dr in res["drivers"][:2]:
        bullets.append(_driver_line(dr, spec, df, cur, f"in {prev}", _unit_noun(df)))
    if bullets:
        lines.append("\n**What changed alongside it:**")
        lines += [f"- {b}" for b in bullets[:4]]
    else:
        lines.append("\nNo other column in the data moved with it, so the cause isn't recorded in this file.")
    if res["excluded"]:
        lines.append(f"\n_{res['excluded']} extreme values left out._")
    return "\n".join(lines)


def _plot(res, spec: Spec, df, roles) -> Optional[Dict[str, Any]]:
    if res["kind"] == "period":
        s = res["series"].dropna()
        labels = [_period_label(p, roles, spec) if roles.get(spec.period_col) == "month" else p for p in s.index]
        label = f"{'Total' if spec.agg == 'sum' else 'Average'} {_lc(spec.metric_label)}"
        return {"type": "line" if len(s) > 8 else "bar", "x": labels, "y": [float(v) for v in s.values],
                "datasets": [{"label": label, "data": [float(v) for v in s.values]}],
                "x_label": "Month", "y_label": label, "title": f"{spec.metric_label} by month"}
    data, groups = res["data"], res["data"][spec.segment_col].astype(str)
    top = next((d for d in res["drivers"] if d.mask is not None), None)
    if top:
        share = pd.Series(top.mask).groupby(groups.values).mean().sort_values(ascending=False) * 100
        what = (f"mention {top.label}" if top.kind == "theme" else f"mention “{top.label}”" if top.kind == "text"
                else f"are {humanize(top.column).lower()} “{top.label}”")
        label = f"% of {_unit_noun(data)} that {what}"
        return {"type": "bar", "x": list(share.index), "y": [round(float(v), 1) for v in share.values],
                "datasets": [{"label": label, "data": [round(float(v), 1) for v in share.values]}],
                "x_label": humanize(spec.segment_col), "y_label": label,
                "title": f"Share of {_unit_noun(data)} that {what}, by {humanize(spec.segment_col).lower()}"}
    g = res["per_group"][spec.agg].sort_values()
    label = f"{'Average' if spec.agg == 'mean' else 'Total'} {_lc(spec.metric_label)}"
    return {"type": "bar", "x": list(g.index), "y": [float(v) for v in g.values],
            "datasets": [{"label": label, "data": [float(v) for v in g.values]}],
            "x_label": humanize(spec.segment_col), "y_label": label,
            "title": f"{spec.metric_label} by {humanize(spec.segment_col).lower()}"}


def _driver_rows(res, spec, df) -> List[Dict[str, Any]]:
    out = []
    for d in res.get("drivers", []):
        row = {"factor": f"{humanize(d.column)}: {d.label}" if d.kind != "numeric" else d.label, "kind": d.kind,
               "effect": round(float(d.effect), 4)}
        if d.kind == "numeric":
            row.update({"segment_avg": round(float(d.mean_seg), 4), "others_avg": round(float(d.mean_rest), 4)})
        else:
            row.update({"segment_share": round(float(d.share_seg), 4), "others_share": round(float(d.share_rest), 4),
                        "metric_with": round(float(d.y_with), 4), "metric_without": round(float(d.y_without), 4)})
        out.append(row)
    return out


# ── entry point ───────────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame, question: str, llm=None, embed=None) -> Optional[Dict[str, Any]]:
    """
    Driver analysis for a "why" question on one table. Returns None when the
    question can't be tied to a metric and a segment/period in this table,
    so the caller can fall back to its usual path.
    """
    if df is None or len(df) < 3:
        return None
    roles = classify(df)
    spec = parse_question(df, question, roles) or (parse_with_llm(df, question, llm, roles) if llm else None)
    if not spec:
        return None
    logger.info(f"🔎 Driver analysis: {spec.metric_label} ({spec.agg}) · "
                f"{spec.segment_col}={spec.segment_val!r} period={spec.period}")
    res = _segment_gap(df, roles, spec, embed) if spec.segment_col else _period_change(df, roles, spec, embed)
    if not res:
        return None
    answer = _segment_answer(res, spec, df) if res["kind"] == "segment" else _period_answer(res, spec, df, roles)
    return {
        "success": True,
        "answer": answer,
        "explanation": answer,
        "plot": _plot(res, spec, df, roles),
        "data": _driver_rows(res, spec, df),
        "row_count": len(res.get("drivers", [])),
        "analysis": {"metric": spec.metric_label, "aggregation": spec.agg, "segment_column": spec.segment_col,
                     "segment_value": spec.segment_val, "period": spec.period, "kind": res["kind"],
                     "explained": round((res.get("counterfactual") or {}).get("explained", 0.0), 3)},
    }
