"""AI analysis pipeline bridge — connects ai-analyze-think-act-core to the platform.

Wraps the core ingest→analyze→recommend pipeline with e-commerce-specific context
and provides a clean async-friendly interface for the API layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft-import core library so the platform can start even if the package
# is not yet installed (graceful degradation with clear error messaging).
# ---------------------------------------------------------------------------
try:
    from core import analyze, recommend
    from core.models import AnalysisResult

    _CORE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CORE_AVAILABLE = False
    logger.warning(
        "ai-analyze-think-act-core not installed. "
        "AI analysis endpoints will return 503. "
        "Install with: pip install git+https://github.com/labgadget015-dotcom/ai-analyze-think-act-core.git"
    )


class AIInsightsError(Exception):
    """Raised when the AI pipeline cannot produce insights."""


def _check_core_available() -> None:
    if not _CORE_AVAILABLE:
        raise AIInsightsError(
            "ai-analyze-think-act-core is not installed. "
            "See installation instructions in README."
        )


def run_ecommerce_analysis(
    df: pd.DataFrame,
    goal: str = "increase_revenue",
    extra_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full ingest→analyze→recommend pipeline on e-commerce data.

    Args:
        df: DataFrame with e-commerce metrics (orders, revenue, etc.)
        goal: Business goal string passed to the LLM prompt
        extra_context: Optional dict merged into the analysis context

    Returns:
        Dict with keys: analysis, recommendations, metadata

    Raises:
        AIInsightsError: If core library unavailable or pipeline fails
    """
    _check_core_available()

    ctx: Dict[str, Any] = {"goal": goal, "platform": "ecommerce"}
    if extra_context:
        ctx.update(extra_context)

    try:
        result: AnalysisResult = analyze(df, context=ctx)
        recs = recommend(result, context=ctx)
        return {
            "analysis": result.model_dump() if hasattr(result, "model_dump") else vars(result),
            "recommendations": recs if isinstance(recs, list) else [recs],
            "metadata": {
                "goal": goal,
                "rows_analyzed": len(df),
                "columns": list(df.columns),
            },
        }
    except Exception as exc:
        logger.exception("AI pipeline failed: %s", exc)
        raise AIInsightsError(f"Pipeline execution failed: {exc}") from exc


def get_pipeline_status() -> Dict[str, Any]:
    """Return installation and health status of the core library."""
    return {
        "core_available": _CORE_AVAILABLE,
        "package": "ai-analyze-think-act-core",
        "status": "ok" if _CORE_AVAILABLE else "not_installed",
    }
