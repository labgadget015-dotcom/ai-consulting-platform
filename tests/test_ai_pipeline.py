"""Tests for app/core/ai_pipeline.py — fully offline (no real OpenAI calls)."""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import app.core.ai_pipeline as pipeline_module
from app.core.ai_pipeline import AIInsightsError, get_pipeline_status, run_ecommerce_analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "revenue": range(100, 100 + n * 10, 10),
            "orders": range(10, 10 + n),
        }
    )


def _make_analysis_result() -> MagicMock:
    result = MagicMock()
    result.model_dump.return_value = {
        "summary": "Revenue trending up",
        "insights": ["Strong week-over-week growth"],
    }
    return result


# ---------------------------------------------------------------------------
# get_pipeline_status
# ---------------------------------------------------------------------------

class TestGetPipelineStatus:
    def test_returns_dict(self):
        status = get_pipeline_status()
        assert isinstance(status, dict)
        assert "core_available" in status
        assert "status" in status
        assert status["package"] == "ai-analyze-think-act-core"

    def test_status_reflects_core_available_true(self, monkeypatch):
        monkeypatch.setattr(pipeline_module, "_CORE_AVAILABLE", True)
        status = get_pipeline_status()
        assert status["status"] == "ok"
        assert status["core_available"] is True

    def test_status_reflects_core_available_false(self, monkeypatch):
        monkeypatch.setattr(pipeline_module, "_CORE_AVAILABLE", False)
        status = get_pipeline_status()
        assert status["status"] == "not_installed"
        assert status["core_available"] is False


# ---------------------------------------------------------------------------
# run_ecommerce_analysis — core unavailable
# ---------------------------------------------------------------------------

class TestRunEcommerceAnalysisUnavailable:
    def test_raises_when_core_unavailable(self, monkeypatch):
        monkeypatch.setattr(pipeline_module, "_CORE_AVAILABLE", False)
        with pytest.raises(AIInsightsError, match="not installed"):
            run_ecommerce_analysis(_make_df())


# ---------------------------------------------------------------------------
# run_ecommerce_analysis — happy path (mocked core)
# ---------------------------------------------------------------------------

class TestRunEcommerceAnalysisHappyPath:
    def _patch_core(self, monkeypatch):
        mock_result = _make_analysis_result()
        mock_analyze = MagicMock(return_value=mock_result)
        mock_recommend = MagicMock(return_value=["Increase ad spend", "Bundle slow sellers"])
        monkeypatch.setattr(pipeline_module, "_CORE_AVAILABLE", True)
        monkeypatch.setattr(pipeline_module, "analyze", mock_analyze, raising=False)
        monkeypatch.setattr(pipeline_module, "recommend", mock_recommend, raising=False)
        return mock_analyze, mock_recommend

    def test_returns_expected_keys(self, monkeypatch):
        self._patch_core(monkeypatch)
        result = run_ecommerce_analysis(_make_df())
        assert set(result.keys()) == {"analysis", "recommendations", "metadata"}

    def test_metadata_has_goal_and_row_count(self, monkeypatch):
        self._patch_core(monkeypatch)
        df = _make_df(8)
        result = run_ecommerce_analysis(df, goal="reduce_churn")
        assert result["metadata"]["goal"] == "reduce_churn"
        assert result["metadata"]["rows_analyzed"] == 8
        assert "date" in result["metadata"]["columns"]

    def test_extra_context_merged(self, monkeypatch):
        mock_analyze, _ = self._patch_core(monkeypatch)
        extra = {"shop": "my-store.myshopify.com"}
        run_ecommerce_analysis(_make_df(), extra_context=extra)
        call_kwargs = mock_analyze.call_args[1]
        assert call_kwargs["context"]["shop"] == "my-store.myshopify.com"
        assert call_kwargs["context"]["goal"] == "increase_revenue"

    def test_recommendations_always_list(self, monkeypatch):
        mock_result = _make_analysis_result()
        monkeypatch.setattr(pipeline_module, "_CORE_AVAILABLE", True)
        monkeypatch.setattr(pipeline_module, "analyze", MagicMock(return_value=mock_result), raising=False)
        # recommend returns a single string (not a list)
        monkeypatch.setattr(pipeline_module, "recommend", MagicMock(return_value="Do X"), raising=False)
        result = run_ecommerce_analysis(_make_df())
        assert isinstance(result["recommendations"], list)

    def test_pipeline_exception_wrapped(self, monkeypatch):
        monkeypatch.setattr(pipeline_module, "_CORE_AVAILABLE", True)
        monkeypatch.setattr(
            pipeline_module,
            "analyze",
            MagicMock(side_effect=RuntimeError("LLM timeout")),
            raising=False,
        )
        with pytest.raises(AIInsightsError, match="Pipeline execution failed"):
            run_ecommerce_analysis(_make_df())
