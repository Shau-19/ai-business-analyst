# agents/forecast_agent.py
"""
Prophet Forecast Agent
======================
Time series forecasting with anomaly detection and LLM narrative generation.

Never called directly — always dispatched via the A2A registry:
  orchestrator → A2AMessage(to="forecast_agent")
  → registry.route_message()
  → _wrap_forecast_agent handler in orchestrator.py
  → ForecastAgent.forecast()

PROPHET_AVAILABLE is exported so the orchestrator can gate the FORECAST
routing path at classification time without trying to import Prophet itself.
"""

from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

from protocols.a2a import A2AAgent, AgentCapability, MessageType, A2AMessage
from database.db_manager import DatabaseManager
from config import settings
from utils.logger import logger

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("⚠️  Prophet not installed. Run: pip install prophet --break-system-packages")

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class ForecastAgent:
    """
    Prophet-based forecasting agent.
    Instantiated by OrchestratorAgent and wrapped in A2A via _wrap_forecast_agent().
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

        self.llm = None
        if GROQ_AVAILABLE:
            try:
                self.llm = ChatGroq(
                    api_key=settings.GROQ_API_KEY,
                    model=settings.LLM_MODEL,
                    temperature=0.3,
                )
            except Exception as e:
                logger.warning(f"⚠️  ForecastAgent LLM init failed: {e}")

        logger.info("📈  ForecastAgent initialized")

    # ── Main entry point ──────────────────────────────────────────────────────

    async def forecast(
        self,
        question: str,
        conversation_id: str,
        periods: int = 6,
    ) -> Dict[str, Any]:
        """
        Fit Prophet on the session CSV, generate a forecast, detect anomalies,
        and return a structured result with LLM narrative and chart-ready data.
        """
        if not PROPHET_AVAILABLE:
            return {
                "success": False,
                "error":   "Prophet not installed. Run: pip install prophet --break-system-packages",
                "routing": "forecast",
            }

        try:
            df = self._load_session_csv(conversation_id)
            if df is None or df.empty:
                return {
                    "success": False,
                    "error":   "No CSV data found. Upload a CSV file first.",
                    "routing": "forecast",
                }

            date_col  = self._detect_date_column(df)
            value_col = self._detect_value_column(df, question)

            if not date_col:
                return {
                    "success": False,
                    "error":   "No date column found. Forecasting requires a date/time column.",
                    "routing": "forecast",
                }
            if not value_col:
                return {
                    "success": False,
                    "error":   "No numeric column found to forecast.",
                    "routing": "forecast",
                }

            logger.info(f"📈  date_col={date_col}, value_col={value_col}, periods={periods}")

            # Prepare Prophet dataframe
            pdf = df[[date_col, value_col]].copy()
            pdf.columns = ["ds", "y"]
            pdf["ds"]   = pd.to_datetime(pdf["ds"], errors="coerce")
            pdf["y"]    = pd.to_numeric(pdf["y"], errors="coerce")
            pdf         = pdf.dropna().sort_values("ds").reset_index(drop=True)

            if len(pdf) < 3:
                return {
                    "success": False,
                    "error":   f"Only {len(pdf)} valid rows — need at least 3 to forecast.",
                    "routing": "forecast",
                }

            # Fit Prophet
            freq  = self._infer_frequency(pdf["ds"])
            model = Prophet(
                yearly_seasonality="auto",
                weekly_seasonality="auto",
                daily_seasonality=False,
                interval_width=0.95,
            )
            model.fit(pdf)

            future   = model.make_future_dataframe(periods=periods, freq=freq)
            forecast = model.predict(future)

            anomalies  = self._detect_anomalies(pdf, forecast)
            chart_data = self._build_chart_data(pdf, forecast)
            explanation = self._generate_narrative(
                question, value_col, pdf, forecast, anomalies, periods
            )

            fc_rows  = forecast[forecast["ds"] > pdf["ds"].max()]
            next_fc  = round(float(fc_rows["yhat"].iloc[0]), 2) if len(fc_rows) else None

            plot_spec = {
                "type":    "line",
                "title":   f"{value_col} — Forecast ({periods} periods)",
                "x_key":   "date",
                "x_label": date_col,
                "y_label": value_col,
                "data":    chart_data,
                "series":  [
                    {"key": "actual",   "label": "Actual",   "color": "#6366f1"},
                    {"key": "forecast", "label": "Forecast", "color": "#f59e0b"},
                ],
            }

            return {
                "success":       True,
                "routing":       "forecast",
                "explanation":   explanation,
                "answer":        explanation,
                "data":          chart_data,
                "anomalies":     anomalies,
                "plot":          plot_spec,
                "value_column":  value_col,
                "date_column":   date_col,
                "periods":       periods,
                "data_points":   len(pdf),
                "last_actual":   round(float(pdf["y"].iloc[-1]), 2),
                "next_forecast": next_fc,
            }

        except Exception as e:
            logger.error(f"❌  ForecastAgent.forecast() error: {e}")
            import traceback; traceback.print_exc()
            return {"success": False, "error": str(e), "routing": "forecast"}

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_session_csv(self, conversation_id: str) -> Optional[pd.DataFrame]:
        """Load the first CSV table available for this session."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_manager.db_path)

            # Session-scoped tables first
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
                (f"%{conversation_id[:8]}%",),
            )
            tables = [r[0] for r in cursor.fetchall()]

            # Fallback: any csv_ prefixed table
            if not tables:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'csv_%'"
                )
                tables = [r[0] for r in cursor.fetchall()]

            if not tables:
                conn.close()
                return None

            df = pd.read_sql_query(f'SELECT * FROM "{tables[0]}"', conn)
            conn.close()
            logger.info(f"📊  Loaded {tables[0]}: {len(df)} rows × {len(df.columns)} cols")
            return df

        except Exception as e:
            logger.error(f"❌  _load_session_csv error: {e}")
            return None

    # ── Column detection ──────────────────────────────────────────────────────

    def _detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Return the first parseable date column, preferring name-hint matches."""
        DATE_HINTS = ["date", "month", "week", "year", "time", "day", "period", "quarter"]

        for col in df.columns:
            if any(h in col.lower() for h in DATE_HINTS):
                try:
                    pd.to_datetime(df[col], errors="raise")
                    return col
                except Exception:
                    pass

        for col in df.columns:
            if df[col].dtype == object:
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().sum() >= len(df) * 0.8:
                        return col
                except Exception:
                    pass

        return None

    def _detect_value_column(self, df: pd.DataFrame, question: str) -> Optional[str]:
        """Return the numeric column most relevant to the question text."""
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        if not num_cols:
            for col in df.columns:
                test = pd.to_numeric(df[col], errors="coerce")
                if test.notna().sum() >= len(df) * 0.5:
                    num_cols.append(col)

        if not num_cols:
            return None

        q_words   = set(question.lower().split())
        best_col  = num_cols[0]
        best_score = 0
        for col in num_cols:
            score = len(q_words & set(col.lower().replace("_", " ").split()))
            if score > best_score:
                best_score, best_col = score, col

        return best_col

    # ── Frequency inference ───────────────────────────────────────────────────

    def _infer_frequency(self, dates: pd.Series) -> str:
        if len(dates) < 2:
            return "MS"
        median_days = dates.diff().dropna().dt.days.median()
        if median_days <= 1:   return "D"
        if median_days <= 8:   return "W"
        if median_days <= 32:  return "MS"
        if median_days <= 95:  return "QS"
        return "YS"

    # ── Anomaly detection ─────────────────────────────────────────────────────

    def _detect_anomalies(
        self, pdf: pd.DataFrame, forecast: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Flag historical data points outside the 95% Prophet confidence interval."""
        merged = pdf.merge(
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left"
        )
        anomalies = merged[
            (merged["y"] < merged["yhat_lower"]) |
            (merged["y"] > merged["yhat_upper"])
        ]
        return [
            {
                "ds":    str(r["ds"])[:10],
                "y":     round(float(r["y"]), 2),
                "yhat":  round(float(r["yhat"]), 2),
                "lower": round(float(r["yhat_lower"]), 2),
                "upper": round(float(r["yhat_upper"]), 2),
            }
            for _, r in anomalies.iterrows()
        ]

    # ── Chart data ────────────────────────────────────────────────────────────

    def _build_chart_data(
        self, pdf: pd.DataFrame, forecast: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Build chart-ready rows with actual + forecast + confidence bands."""
        hist_set = set(pdf["ds"].astype(str).str[:10])
        rows     = []
        for _, row in forecast.iterrows():
            d       = str(row["ds"])[:10]
            is_hist = d in hist_set
            actual  = None
            if is_hist:
                m = pdf[pdf["ds"].astype(str).str[:10] == d]
                if not m.empty:
                    actual = round(float(m["y"].iloc[0]), 2)
            rows.append({
                "date":     d,
                "actual":   actual,
                "forecast": round(float(row["yhat"]), 2)       if not is_hist else None,
                "lower":    round(float(row["yhat_lower"]), 2)  if not is_hist else None,
                "upper":    round(float(row["yhat_upper"]), 2)  if not is_hist else None,
            })
        return rows

    # ── LLM narrative ─────────────────────────────────────────────────────────

    def _generate_narrative(
        self,
        question:    str,
        value_col:   str,
        pdf:         pd.DataFrame,
        forecast:    pd.DataFrame,
        anomalies:   List[Dict],
        periods:     int,
    ) -> str:
        """
        Generate a business-focused 3-4 sentence narrative using the LLM.
        Falls back to a template-based summary if the LLM is unavailable.
        """
        if not self.llm:
            return self._fallback_narrative(value_col, pdf, forecast, anomalies, periods)

        try:
            last_actual = float(pdf["y"].iloc[-1])
            fc_rows     = forecast[forecast["ds"] > pdf["ds"].max()]
            next_fc     = float(fc_rows["yhat"].iloc[0])  if len(fc_rows) else None
            last_fc     = float(fc_rows["yhat"].iloc[-1]) if len(fc_rows) else None
            pct         = ((last_fc - last_actual) / last_actual * 100) if last_fc and last_actual else 0
            trend       = "upward" if pct > 2 else "downward" if pct < -2 else "stable"
            anom_dates  = [a["ds"] for a in anomalies[:3]]

            prompt = (
                f"You are a business data analyst. Write a concise 3-4 sentence insight.\n\n"
                f"Forecast Summary:\n"
                f"- Metric: {value_col}\n"
                f"- Historical data points: {len(pdf)}\n"
                f"- Last actual value: {last_actual:.2f}\n"
                f"- Next period forecast: {f'{next_fc:.2f}' if next_fc is not None else 'N/A'}\n"
                f"- End of horizon forecast: {f'{last_fc:.2f}' if last_fc is not None else 'N/A'}\n"
                f"- Projected change over {periods} periods: {pct:+.1f}%\n"
                f"- Trend: {trend}\n"
                f"- Anomalies detected: {len(anomalies)}\n"
                f"- Anomaly dates: {anom_dates if anom_dates else 'none'}\n\n"
                f"User question: {question}\n\n"
                f"Write a business-focused insight in plain English. "
                f"Mention the trend direction, next-period forecast value, and whether "
                f"anomalies need attention. No bullet points. No headers."
            )

            return self.llm.invoke(prompt).content.strip()

        except Exception as e:
            logger.warning(f"⚠️  LLM narrative failed: {e} — using fallback")
            return self._fallback_narrative(value_col, pdf, forecast, anomalies, periods)

    def _fallback_narrative(
        self,
        value_col: str,
        pdf:       pd.DataFrame,
        forecast:  pd.DataFrame,
        anomalies: List[Dict],
        periods:   int,
    ) -> str:
        """Template-based narrative used when the LLM is unavailable."""
        last_actual = float(pdf["y"].iloc[-1])
        fc_rows     = forecast[forecast["ds"] > pdf["ds"].max()]
        next_fc     = float(fc_rows["yhat"].iloc[0]) if len(fc_rows) else None

        txt = (
            f"Forecast for **{value_col}** over the next {periods} periods. "
            f"Last actual value: **{last_actual:.2f}**."
        )
        if next_fc is not None:
            txt += f" Next period forecast: **{next_fc:.2f}**."
        txt += (
            f" **{len(anomalies)} anomaly/anomalies** detected outside the 95% confidence interval."
            if anomalies
            else " No anomalies detected — data is within expected bounds."
        )
        return txt