"""Reliability, safety, and adaptability framework for production systems."""

import functools
import logging
import time
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorSeverity(str, Enum):
    """Error severity levels for monitoring and alerting."""

    CRITICAL = "critical"  # System failure, immediate action required
    HIGH = "high"  # Major functionality impaired
    MEDIUM = "medium"  # Minor issues, degraded performance
    LOW = "low"  # Warnings, no immediate impact
    INFO = "info"  # Informational messages


class HealthStatus(str, Enum):
    """System health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ValidationResult(BaseModel):
    """Result of data validation."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_error(self, message: str) -> "ValidationResult":
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
        return self

    def add_warning(self, message: str) -> "ValidationResult":
        """Add a warning message."""
        self.warnings.append(message)
        return self


class CircuitBreaker:
    """Circuit breaker pattern for failing gracefully."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = "half-open"

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Reset circuit breaker on successful call."""
        self.failure_count = 0
        self.state = "closed"
        logger.info("Circuit breaker reset to CLOSED state")

    def _on_failure(self):
        """Handle failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(
                f"Circuit breaker OPEN after {self.failure_count} failures"
            )


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to prevent thundering herd
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries"
                        )
                        raise e

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base**attempt), max_delay)

                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + np.random.random() * 0.5)

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}. "
                        f"Retrying in {delay:.2f}s. Error: {str(e)}"
                    )
                    time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


def safe_execute(
    func: Callable[..., T],
    *args,
    default_value: Optional[T] = None,
    log_errors: bool = True,
    **kwargs,
) -> Tuple[Optional[T], Optional[Exception]]:
    """Safely execute a function and return result or default value.

    Args:
        func: Function to execute
        *args: Positional arguments
        default_value: Value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments

    Returns:
        Tuple of (result, exception)
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        if log_errors:
            logger.error(
                f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
            )
        return default_value, e


class DataValidator:
    """Comprehensive data validation for ML inputs."""

    @staticmethod
    def validate_dataframe(
        df: Any, required_columns: List[str], min_rows: int = 1
    ) -> ValidationResult:
        """Validate pandas DataFrame structure and content.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(is_valid=True)

        # Check if it's a DataFrame
        if not hasattr(df, "shape") or not hasattr(df, "columns"):
            return result.add_error("Input is not a valid DataFrame")

        # Check for required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            result.add_error(f"Missing required columns: {missing_columns}")

        # Check minimum rows
        if len(df) < min_rows:
            result.add_error(
                f"Insufficient data: {len(df)} rows (minimum {min_rows} required)"
            )

        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                pct = (count / len(df)) * 100
                if pct > 20:
                    result.add_error(
                        f"Column '{col}' has {pct:.1f}% null values (threshold: 20%)"
                    )
                elif pct > 5:
                    result.add_warning(
                        f"Column '{col}' has {pct:.1f}% null values"
                    )

        # Check for data quality issues
        for col in required_columns:
            if col in df.columns:
                # Check for constant values
                if df[col].nunique() == 1:
                    result.add_warning(f"Column '{col}' has constant values")

                # Check for outliers in numeric columns
                if df[col].dtype in [np.float64, np.int64]:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr

                    outliers = df[
                        (df[col] < lower_bound) | (df[col] > upper_bound)
                    ]
                    if len(outliers) > 0:
                        pct = (len(outliers) / len(df)) * 100
                        if pct > 10:
                            result.add_warning(
                                f"Column '{col}' has {pct:.1f}% potential outliers"
                            )

        result.metadata = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
        }

        return result

    @staticmethod
    def validate_time_series(
        df: Any, date_column: str, value_column: str
    ) -> ValidationResult:
        """Validate time series data for forecasting.

        Args:
            df: DataFrame with time series data
            date_column: Name of date column
            value_column: Name of value column

        Returns:
            ValidationResult
        """
        result = DataValidator.validate_dataframe(
            df, required_columns=[date_column, value_column], min_rows=30
        )

        if not result.is_valid:
            return result

        # Check date column is datetime
        if df[date_column].dtype != "datetime64[ns]":
            result.add_warning(f"Column '{date_column}' is not datetime type")

        # Check for gaps in time series
        df_sorted = df.sort_values(date_column)
        date_diffs = df_sorted[date_column].diff()
        median_diff = date_diffs.median()

        large_gaps = date_diffs[date_diffs > median_diff * 3]
        if len(large_gaps) > 0:
            result.add_warning(
                f"Found {len(large_gaps)} large gaps in time series (>3x median frequency)"
            )

        # Check for sufficient historical data
        date_range = (df_sorted[date_column].max() - df_sorted[date_column].min()).days
        if date_range < 90:
            result.add_warning(
                f"Limited historical data: {date_range} days (recommended: 90+ days)"
            )

        # Check value distribution
        if df[value_column].min() < 0:
            result.add_warning(f"Column '{value_column}' contains negative values")

        zeros_pct = (df[value_column] == 0).sum() / len(df) * 100
        if zeros_pct > 30:
            result.add_warning(
                f"Column '{value_column}' has {zeros_pct:.1f}% zero values"
            )

        return result


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.timestamps: Dict[str, List[datetime]] = {}

    def record(self, metric_name: str, value: float):
        """Record a performance metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
            self.timestamps[metric_name] = []

        self.metrics[metric_name].append(value)
        self.timestamps[metric_name].append(datetime.now())

    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}

        values = np.array(self.metrics[metric_name])
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "count": len(values),
        }

    def check_health(self, metric_name: str, threshold: float) -> HealthStatus:
        """Check if metric is within healthy range."""
        stats = self.get_statistics(metric_name)
        if not stats:
            return HealthStatus.UNKNOWN

        if stats["p95"] > threshold * 2:
            return HealthStatus.UNHEALTHY
        elif stats["p95"] > threshold:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
