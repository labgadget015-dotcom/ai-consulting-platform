"""Streamlit e-commerce intelligence dashboard.

Demonstrates platform capabilities with three tabs:
  - Sales Forecast (Prophet-based projection)
  - AI Insights   (ai-analyze-think-act-core pipeline)
  - Data Explorer (basic EDA)

Gracefully degrades when optional deps (prophet, core) are not installed.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Soft imports — platform works without these installed
# ---------------------------------------------------------------------------
try:
    from app.core.ai_pipeline import (
        AIInsightsError,
        get_pipeline_status,
        run_ecommerce_analysis,
    )
    _PIPELINE_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PIPELINE_AVAILABLE = False
    AIInsightsError = Exception  # type: ignore[assignment,misc]

try:
    from app.models.forecasting import SalesForecaster
    _FORECASTER_AVAILABLE = True
except Exception:  # noqa: BLE001
    _FORECASTER_AVAILABLE = False
    SalesForecaster = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Consulting Platform",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("AI Consulting Platform")
    st.markdown("---")

    business_goal = st.selectbox(
        "Business Goal",
        options=[
            "increase_revenue",
            "reduce_churn",
            "optimize_inventory",
            "improve_margins",
        ],
        index=0,
    )

    uploaded_file = st.file_uploader(
        "Upload your e-commerce data",
        type=["csv"],
        help="CSV with at minimum a date column and a numeric sales/revenue column.",
    )

    st.markdown("---")
    st.caption("© AI Consulting Platform")

# ---------------------------------------------------------------------------
# Helper: load uploaded CSV
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_csv(raw_bytes: bytes) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(raw_bytes))


df: Optional[pd.DataFrame] = None
if uploaded_file is not None:
    try:
        df = _load_csv(uploaded_file.getvalue())
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")


# ---------------------------------------------------------------------------
# Helper: build dummy sales DataFrame
# ---------------------------------------------------------------------------
def _dummy_sales(n_days: int = 90) -> pd.DataFrame:
    import numpy as np

    rng = pd.date_range(end=datetime.today(), periods=n_days, freq="D")
    np.random.seed(42)
    trend = range(n_days)
    noise = np.random.normal(0, 200, n_days)
    revenue = [1000 + 5 * t + n for t, n in zip(trend, noise)]
    return pd.DataFrame({"ds": rng, "y": revenue})


# ---------------------------------------------------------------------------
# Detect date + value columns from an arbitrary DataFrame
# ---------------------------------------------------------------------------
def _detect_columns(frame: pd.DataFrame):
    date_col = None
    for col in frame.columns:
        try:
            pd.to_datetime(frame[col].iloc[:5])
            date_col = col
            break
        except Exception:
            pass

    numeric_cols = frame.select_dtypes(include="number").columns.tolist()
    value_col = numeric_cols[0] if numeric_cols else None
    return date_col, value_col


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_forecast, tab_insights, tab_explorer = st.tabs(
    ["📊 Sales Forecast", "🤖 AI Insights", "📈 Data Explorer"]
)

# ── Tab 1: Sales Forecast ────────────────────────────────────────────────────
with tab_forecast:
    st.header("Sales Forecast")
    st.markdown("30-day revenue projection powered by Facebook Prophet.")

    forecast_df: Optional[pd.DataFrame] = None
    history_df: Optional[pd.DataFrame] = None

    if df is not None:
        date_col, value_col = _detect_columns(df)
        if date_col is None or value_col is None:
            st.warning(
                "Could not auto-detect a date column and a numeric column in your CSV. "
                "Showing sample data instead."
            )
            history_df = _dummy_sales()
        else:
            history_df = pd.DataFrame(
                {"ds": pd.to_datetime(df[date_col]), "y": df[value_col]}
            ).dropna()
    else:
        st.info("No CSV uploaded — showing sample data.")
        history_df = _dummy_sales()

    if _FORECASTER_AVAILABLE and history_df is not None and len(history_df) >= 10:
        try:
            forecaster = SalesForecaster()
            forecaster.fit(history_df)
            future = forecaster.predict(periods=30)
            forecast_df = future[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30)
        except Exception as exc:
            st.warning(f"Prophet forecast failed ({exc}). Showing historical data only.")
    elif not _FORECASTER_AVAILABLE:
        st.info(
            "📦 `prophet` is not installed — install it to enable forecasting. "
            "Showing historical data only."
        )

    # Build chart
    if history_df is not None:
        fig = px.line(
            history_df,
            x="ds",
            y="y",
            labels={"ds": "Date", "y": "Revenue ($)"},
            title="Historical Revenue",
        )
        fig.update_traces(name="Historical", showlegend=True)

        if forecast_df is not None:
            fig.add_scatter(
                x=forecast_df["ds"],
                y=forecast_df["yhat"],
                mode="lines",
                name="Forecast",
                line={"dash": "dash", "color": "orange"},
            )
            fig.add_scatter(
                x=forecast_df["ds"],
                y=forecast_df["yhat_upper"],
                mode="lines",
                name="Upper bound",
                line={"width": 0},
                showlegend=False,
            )
            fig.add_scatter(
                x=forecast_df["ds"],
                y=forecast_df["yhat_lower"],
                mode="lines",
                name="Confidence interval",
                fill="tonexty",
                fillcolor="rgba(255,165,0,0.15)",
                line={"width": 0},
            )

        st.plotly_chart(fig, use_container_width=True)

        # Key metrics
        col1, col2 = st.columns(2)
        with col1:
            if forecast_df is not None:
                projected = forecast_df["yhat"].sum()
                st.metric("Projected Revenue (30 days)", f"${projected:,.0f}")
            else:
                recent = history_df["y"].tail(30).sum()
                st.metric("Recent Revenue (last 30 days)", f"${recent:,.0f}")
        with col2:
            last_vals = history_df["y"].tail(14)
            trend = "📈 Upward" if last_vals.iloc[-1] > last_vals.iloc[0] else "📉 Downward"
            st.metric("Trend Direction", trend)

# ── Tab 2: AI Insights ───────────────────────────────────────────────────────
with tab_insights:
    st.header("AI Insights")
    st.markdown("Powered by the `ai-analyze-think-act-core` analysis pipeline.")

    _has_api_key = bool(os.environ.get("OPENAI_API_KEY"))

    if not _PIPELINE_AVAILABLE:
        st.info(
            "🔌 The AI pipeline core is not installed in this environment.\n\n"
            "**Sample output:**"
        )
        st.success(
            "**Analysis Summary**\n\n"
            "Based on the uploaded e-commerce data, revenue is trending upward with "
            "a 12% MoM growth rate. Cart abandonment is the primary drag on conversion."
        )
        st.markdown("**Recommendations:**")
        for i, rec in enumerate(
            [
                "Launch a re-engagement email campaign targeting users who abandoned carts in the last 7 days.",
                "Introduce bundle discounts on top-3 SKUs to increase average order value.",
                "Reduce fulfillment time by pre-positioning inventory in the top two regions.",
            ],
            start=1,
        ):
            st.markdown(f"{i}. {rec}")

    elif df is None:
        st.warning("⬆️ Upload a CSV in the sidebar to run live AI analysis.")

    elif not _has_api_key:
        st.warning(
            "🔑 `OPENAI_API_KEY` is not set. "
            "Set it as an environment variable to enable live AI analysis."
        )

    else:
        with st.spinner("Running AI analysis…"):
            try:
                result = run_ecommerce_analysis(df, goal=business_goal)

                # Analysis summary card
                st.subheader("Analysis Summary")
                analysis_text = result.get("analysis", "No analysis returned.")
                if hasattr(analysis_text, "summary"):
                    analysis_text = analysis_text.summary
                st.info(str(analysis_text))

                # Recommendations
                recommendations = result.get("recommendations", [])
                if recommendations:
                    st.subheader("Recommendations")
                    for idx, rec in enumerate(recommendations, start=1):
                        rec_text = rec if isinstance(rec, str) else getattr(rec, "text", str(rec))
                        st.markdown(f"{idx}. {rec_text}")

                # Metadata
                metadata = result.get("metadata", {})
                if metadata:
                    with st.expander("Pipeline metadata"):
                        st.json(metadata)

            except AIInsightsError as exc:
                st.error(f"AI pipeline error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error during analysis: {exc}")

# ── Tab 3: Data Explorer ─────────────────────────────────────────────────────
with tab_explorer:
    st.header("Data Explorer")

    if df is None:
        st.info("⬆️ Upload a CSV in the sidebar to explore your data.")
    else:
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(include="all"), use_container_width=True)

        st.subheader("Column Info")
        col_info = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "non_null": df.count(),
                "null_count": df.isnull().sum(),
                "null_%": (df.isnull().mean() * 100).round(2),
                "unique": df.nunique(),
            }
        )
        st.dataframe(col_info, use_container_width=True)

        # Missing values heatmap (only if there are nulls)
        if df.isnull().any().any():
            st.subheader("Missing Values Heatmap")
            try:
                null_matrix = df.isnull().astype(int)
                fig_nulls = px.imshow(
                    null_matrix.T,
                    color_continuous_scale=["white", "crimson"],
                    labels={"color": "Missing"},
                    title="Missing Value Locations (red = missing)",
                    aspect="auto",
                )
                fig_nulls.update_coloraxes(showscale=False)
                st.plotly_chart(fig_nulls, use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not render heatmap: {exc}")
        else:
            st.success("✅ No missing values found in the dataset.")

        # Distribution charts for numeric columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            st.subheader("Numeric Distributions")
            chosen_col = st.selectbox("Select column", numeric_cols)
            fig_hist = px.histogram(
                df,
                x=chosen_col,
                nbins=40,
                title=f"Distribution of {chosen_col}",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
