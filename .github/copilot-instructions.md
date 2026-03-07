# Copilot Instructions — ai-consulting-platform

## What this repo is
A FastAPI + Streamlit platform for e-commerce consulting analytics. Features:
- Prophet-based sales forecasting
- Shopify data connectors
- AI-powered insights via `ai-analyze-think-act-core`
- 3-tab Streamlit dashboard

## Project layout
```
app/
  api/main.py          # FastAPI routes (REST API)
  core/
    ai_pipeline.py     # Bridge to ai-analyze-think-act-core (soft-import)
    config.py          # Pydantic v2 Settings
    shopify_client.py  # Shopify API wrapper
  dashboard.py         # Streamlit 3-tab app
  models/              # SQLAlchemy models
tests/
  test_ai_pipeline.py  # 9 tests for the AI bridge
```

## Key conventions
- **Pydantic v2**: `model_config = ConfigDict(env_prefix="...", case_sensitive=False)`. Never use `Field(env="VAR")` — env var mapping is automatic from field name.
- **Soft imports**: `app/core/ai_pipeline.py` wraps core import in try/except. Always check `_CORE_AVAILABLE` before calling `run_ecommerce_analysis()`.
- **API responses**: return 503 with `{"detail": "Core pipeline not available"}` when core is missing.
- Heavy deps (prophet, xgboost, shopify) are excluded from CI. Wrap their imports in try/except too.

## Running locally
```bash
pip install -r requirements.txt
uvicorn app.api.main:app --reload          # API on :8000
streamlit run app/dashboard.py             # Dashboard on :8501
```

## AI endpoints
- `GET /api/v1/ai/status` — check if core pipeline is available
- `POST /api/v1/ai/insights` — body: `{"data": [...], "analysis_type": "ecommerce"}`

## Testing
```bash
pytest tests/ -v
```

## Deployment
Railway: `railway up` (uses `railway.json` + `Procfile`).
Set env vars: `OPENAI_API_KEY`, `DATABASE_URL`, `REDIS_URL`, `SHOPIFY_API_KEY`.

## Ecosystem
- Core dependency: [ai-analyze-think-act-core](https://github.com/labgadget015-dotcom/ai-analyze-think-act-core)
- Dashboard connects to same FastAPI backend
