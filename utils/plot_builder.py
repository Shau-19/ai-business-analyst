# utils/plot_builder.py
"""
FIXED Plot Builder - Respects user's chart type requests
"""

import re
from typing import Dict, Any, List, Optional
from utils.logger import logger


class PlotBuilder:
    """Build chart specifications from query results"""

    MAX_POINTS = 100   # matches SQLExecutor's `data` cap
    MAX_SERIES = 4

    # Keywords that indicate need for plotting
    PLOT_KEYWORDS = [
        "chart", "plot", "graph", "visualize", "show",
        "trend", "over time", "breakdown", "distribution",
        "compare", "comparison"
    ]
    
    # CHART TYPE KEYWORDS - Priority order
    CHART_TYPE_KEYWORDS = {
        "bar": ["bar chart", "bar graph", "bar"],
        "line": ["line chart", "line graph", "line", "trend"],
        "pie": ["pie chart", "pie graph", "pie", "donut"],
    }
    
    def __init__(self):
        logger.info("📊 Plot Builder initialized")
    
    def needs_plot(self, question: str) -> bool:
        """Check if question requests a chart"""
        q = question.lower()
        return any(keyword in q for keyword in self.PLOT_KEYWORDS)
    
    def build_plot_spec(
        self, 
        sql_result: Dict[str, Any], 
        question: str
    ) -> Optional[Dict[str, Any]]:
        """Build chart specification from SQL result"""
        
        if not sql_result.get("success"):
            logger.warning("⚠️ Cannot plot failed query")
            return None
        
        # Use the full result (up to 100 rows), not the 10-row preview — the
        # preview silently cut e.g. "revenue by product" to 10 of 25 bars.
        rows = sql_result.get("data") or sql_result.get("data_preview", [])
        rows = rows[: self.MAX_POINTS]

        if not rows or len(rows) < 2:
            logger.warning("⚠️ Insufficient data for plotting (need 2+ rows)")
            return None

        keys = list(rows[0].keys())
        if not keys:
            logger.warning("⚠️ No columns found")
            return None

        def is_num_col(col):
            vals = [r.get(col) for r in rows if r.get(col) is not None]
            return bool(vals) and all(
                (isinstance(v, (int, float)) and not isinstance(v, bool)) or self._is_numeric_string(v)
                for v in vals
            )

        numeric_cols = [k for k in keys if is_num_col(k)]
        label_cols   = [k for k in keys if k not in numeric_cols]

        if label_cols:
            x_col = label_cols[0]
        elif len(numeric_cols) >= 2:
            # e.g. (year, revenue): first numeric column becomes the axis
            x_col = numeric_cols.pop(0)
        else:
            logger.warning("⚠️ Need a label column and a numeric column to plot")
            return None

        # ID-like numeric columns are not measures
        y_cols = [c for c in numeric_cols if not re.search(r'(^|_)id$', c, re.IGNORECASE)] or numeric_cols
        if not y_cols:
            logger.warning("⚠️ Y-axis values are not numeric")
            return None

        def num(v):
            if v is None:
                return None          # gap in the chart, not a fake zero
            return float(v) if isinstance(v, str) else v

        x_values = [str(r.get(x_col)) if r.get(x_col) is not None else "" for r in rows]
        datasets = [{"label": c, "data": [num(r.get(c)) for r in rows]} for c in y_cols[: self.MAX_SERIES]]

        chart_type = self._infer_chart_type(question, x_col, y_cols[0])
        if chart_type == "pie" and len(datasets) > 1:
            datasets = datasets[:1]   # a pie can only show one measure

        plot_spec = {
            "type":     chart_type,
            "x":        x_values,
            "y":        datasets[0]["data"],      # first series (back-compat)
            "datasets": datasets,                 # all series
            "x_label":  x_col,
            "y_label":  datasets[0]["label"] if len(datasets) == 1 else "value",
            "title":    self._generate_title(question, x_col, ", ".join(d["label"] for d in datasets)),
            "truncated": bool(sql_result.get("row_count", 0) > len(rows)),
        }

        logger.info(
            f"✅ Plot spec created: {chart_type} chart, {len(x_values)} points, "
            f"{len(datasets)} series"
        )
        return plot_spec
    
    def _infer_chart_type(self, question: str, x_col: str, y_col: str) -> str:
        """
        FIXED: Infer chart type with PRIORITY for user's explicit request
        
        Priority:
        1. User explicitly says "bar chart" → bar
        2. User explicitly says "line chart" → line  
        3. User explicitly says "pie chart" → pie
        4. Time-based data → line (default for trends)
        5. Categorical data → bar (default)
        """
        q = question.lower()
        x_lower = x_col.lower()
        
        # PRIORITY 1: Explicit user request (HIGHEST PRIORITY)
        # Check in order: bar → line → pie. Whole words only — substring
        # matching turned "online" into a line chart and "pieces" into a pie.
        for chart_type, keywords in self.CHART_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if re.search(rf'\b{re.escape(keyword)}\b', q):
                    logger.info(f"📊 Chart type: {chart_type} (explicit user request: '{keyword}')")
                    return chart_type
        
        # PRIORITY 2: Distribution/breakdown → pie
        if any(kw in q for kw in ["distribution", "breakdown", "share", "proportion"]):
            logger.info("📊 Chart type: pie (distribution query)")
            return "pie"
        
        # PRIORITY 3: Time-based data → line
        if any(time_kw in x_lower for time_kw in ["month", "year", "quarter", "date", "time", "week", "day"]):
            logger.info("📊 Chart type: line (time-based data)")
            return "line"
        
        # PRIORITY 4: Default to bar
        logger.info("📊 Chart type: bar (default)")
        return "bar"
    
    def _generate_title(self, question: str, x_col: str, y_col: str) -> str:
        """Generate a readable chart title: 'new_customers' → 'New customers'."""
        def human(raw: str) -> str:
            raw = str(raw)
            s = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw).replace("_", " ")
            s = re.sub(r"\s+", " ", s).strip()
            # Column-style names (snake_case, CamelCase, ALLCAPS) → sentence case
            if "_" in raw or raw.isupper() or re.search(r"[a-z][A-Z]", raw):
                s = s.lower()
            return s[:1].upper() + s[1:]
        return f"{human(y_col)} by {human(x_col).lower()}"
    
    def _is_numeric_string(self, value: Any) -> bool:
        """Check if string can be converted to number"""
        if not isinstance(value, str):
            return False
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False


# Convenience function
def create_plot_spec(sql_result: Dict, question: str) -> Optional[Dict]:
    """Create plot specification from SQL result and question"""
    builder = PlotBuilder()
    return builder.build_plot_spec(sql_result, question)