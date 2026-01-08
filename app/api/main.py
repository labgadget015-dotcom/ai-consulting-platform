"""FastAPI main application for AI Consulting Platform."""

import io
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.reliability import (
    DataValidator,
    PerformanceMonitor,
    ValidationResult,
    retry_with_backoff,
    safe_execute,
)
from app.models.forecasting import SalesForecaster

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered consulting platform for e-commerce businesses",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance monitor
perf_monitor = PerformanceMonitor()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    version: str
    environment: str


class ForecastRequest(BaseModel):
    """Forecast generation request."""

    periods: int = Field(default=30, ge=1, le=365, description="Days to forecast")
    frequency: str = Field(default="D", description="D=daily, W=weekly, M=monthly")
    seasonality_mode: str = Field(
        default="multiplicative", description="additive or multiplicative"
    )


class ForecastPoint(BaseModel):
    """Single forecast data point."""

    date: str
    forecast: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    """Forecast generation response."""

    success: bool
    forecast: List[ForecastPoint]
    summary: Dict
    validation: ValidationResult
    processing_time_ms: float


class DataValidationResponse(BaseModel):
    """Data validation response."""

    success: bool
    validation: ValidationResult
    recommendations: List[str]


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    timestamp: datetime


# ============================================================================
# API ENDPOINTS
# ============================================================================


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "message": "AI Consulting Platform API",
        "version": settings.app_version,
        "docs": "/api/docs",
        "health": "/api/v1/health",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.app_version,
        environment=settings.environment,
    )


@app.post(
    "/api/v1/data/validate",
    response_model=DataValidationResponse,
    tags=["validation"],
)
async def validate_data(file: UploadFile = File(...)):
    """Validate uploaded CSV data for forecasting.

    Args:
        file: CSV file with sales data (date, value columns)

    Returns:
        Validation results with errors, warnings, and recommendations
    """
    start_time = time.time()

    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Validate structure
        result = DataValidator.validate_dataframe(
            df, required_columns=list(df.columns[:2]), min_rows=30
        )

        # Validate as time series if we have date column
        if len(df.columns) >= 2:
            ts_result = DataValidator.validate_time_series(
                df, date_column=df.columns[0], value_column=df.columns[1]
            )
            result.errors.extend(ts_result.errors)
            result.warnings.extend(ts_result.warnings)
            if not ts_result.is_valid:
                result.is_valid = False

        # Generate recommendations
        recommendations = []
        if not result.is_valid:
            recommendations.append(
                "Fix critical errors before generating forecast"
            )
        if result.warnings:
            recommendations.append(
                "Consider addressing warnings for better forecast accuracy"
            )
        if result.metadata.get("rows", 0) < 90:
            recommendations.append(
                "More historical data (90+ days) will improve forecast quality"
            )
        if not recommendations:
            recommendations.append("Data looks good! Ready for forecasting.")

        processing_time = (time.time() - start_time) * 1000
        perf_monitor.record("validation_time_ms", processing_time)

        return DataValidationResponse(
            success=result.is_valid,
            validation=result,
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to validate data: {str(e)}",
        )


@app.post(
    "/api/v1/forecast", response_model=ForecastResponse, tags=["forecasting"]
)
@retry_with_backoff(max_retries=2)
async def generate_forecast(
    file: UploadFile = File(...),
    periods: int = 30,
    frequency: str = "D",
    seasonality_mode: str = "multiplicative",
):
    """Generate sales forecast from uploaded data.

    Args:
        file: CSV file with historical sales data (date, value columns)
        periods: Number of periods to forecast (default: 30 days)
        frequency: Forecast frequency - D=daily, W=weekly, M=monthly
        seasonality_mode: additive or multiplicative (default: multiplicative)

    Returns:
        Forecast with predictions, confidence intervals, and summary statistics
    """
    start_time = time.time()

    try:
        # Read and validate data
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        logger.info(
            f"Generating forecast: {len(df)} rows, {periods} periods, freq={frequency}"
        )

        # Validate data first
        validation_result = DataValidator.validate_time_series(
            df, date_column=df.columns[0], value_column=df.columns[1]
        )

        if not validation_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Data validation failed: {validation_result.errors}",
            )

        # Initialize and fit forecaster
        forecaster = SalesForecaster(seasonality_mode=seasonality_mode)
        forecaster.fit(df)

        # Generate forecast
        forecast_df = forecaster.predict(periods=periods, freq=frequency)

        # Get summary statistics
        summary = forecaster.get_forecast_summary(forecast_df, periods=periods)

        # Convert to response format
        forecast_points = []
        for _, row in forecast_df.tail(periods).iterrows():
            forecast_points.append(
                ForecastPoint(
                    date=row["ds"].strftime("%Y-%m-%d"),
                    forecast=round(float(row["yhat"]), 2),
                    lower_bound=round(float(row["yhat_lower"]), 2),
                    upper_bound=round(float(row["yhat_upper"]), 2),
                )
            )

        processing_time = (time.time() - start_time) * 1000
        perf_monitor.record("forecast_time_ms", processing_time)

        logger.info(
            f"Forecast generated successfully in {processing_time:.2f}ms"
        )

        return ForecastResponse(
            success=True,
            forecast=forecast_points,
            summary=summary,
            validation=validation_result,
            processing_time_ms=round(processing_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate forecast: {str(e)}",
        )


@app.get("/api/v1/metrics", tags=["monitoring"])
async def get_metrics():
    """Get performance metrics."""
    return {
        "validation": perf_monitor.get_statistics("validation_time_ms"),
        "forecast": perf_monitor.get_statistics("forecast_time_ms"),
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc),
            timestamp=datetime.now(),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.debug else None,
            timestamp=datetime.now(),
        ).dict(),
    )


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down application")
