"""Sales forecasting module using Prophet algorithm."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

logger = logging.getLogger(__name__)


class SalesForecaster:
    """Sales forecasting using Facebook Prophet."""

    def __init__(
        self,
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
    ):
        """Initialize forecaster with Prophet parameters.

        Args:
            seasonality_mode: Either 'additive' or 'multiplicative'
            yearly_seasonality: Include yearly seasonal component
            weekly_seasonality: Include weekly seasonal component
            daily_seasonality: Include daily seasonal component
            changepoint_prior_scale: Flexibility of trend (0.001-0.5)
        """
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model: Optional[Prophet] = None
        self.fitted = False

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet (ds, y columns).

        Args:
            data: DataFrame with date and value columns

        Returns:
            Prepared DataFrame with 'ds' and 'y' columns

        Raises:
            ValueError: If data is invalid
        """
        if data.empty:
            raise ValueError("Data cannot be empty")

        # Ensure we have date and value columns
        if len(data.columns) < 2:
            raise ValueError("Data must have at least 2 columns (date, value)")

        df = data.copy()

        # Rename columns to Prophet format
        df.columns = ["ds", "y"] + list(df.columns[2:])

        # Convert date column to datetime
        df["ds"] = pd.to_datetime(df["ds"])

        # Remove any NaN values
        df = df.dropna(subset=["ds", "y"])

        # Sort by date
        df = df.sort_values("ds").reset_index(drop=True)

        logger.info(f"Prepared {len(df)} data points for forecasting")
        return df

    def fit(self, data: pd.DataFrame) -> "SalesForecaster":
        """Fit the Prophet model on historical data.

        Args:
            data: Historical sales data

        Returns:
            self
        """
        df = self.prepare_data(data)

        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )

        logger.info("Fitting Prophet model...")
        self.model.fit(df)
        self.fitted = True
        logger.info("Model fitted successfully")

        return self

    def predict(self, periods: int = 30, freq: str = "D") -> pd.DataFrame:
        """Generate forecast for future periods.

        Args:
            periods: Number of periods to forecast
            freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)

        Returns:
            DataFrame with forecast including confidence intervals

        Raises:
            ValueError: If model hasn't been fitted
        """
        if not self.fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        # Make predictions
        logger.info(f"Generating forecast for {periods} {freq} periods")
        forecast = self.model.predict(future)

        # Return relevant columns
        return forecast[
            [
                "ds",
                "yhat",
                "yhat_lower",
                "yhat_upper",
                "trend",
                "yearly",
                "weekly",
            ]
        ]

    def get_forecast_summary(self, forecast: pd.DataFrame, periods: int = 7) -> Dict:
        """Get summary statistics for the forecast.

        Args:
            forecast: Forecast DataFrame from predict()
            periods: Number of future periods to summarize

        Returns:
            Dictionary with forecast summary
        """
        future_forecast = forecast.tail(periods)

        return {
            "forecast_period": periods,
            "average_forecast": float(future_forecast["yhat"].mean()),
            "total_forecast": float(future_forecast["yhat"].sum()),
            "min_forecast": float(future_forecast["yhat"].min()),
            "max_forecast": float(future_forecast["yhat"].max()),
            "confidence_interval_width": float(
                (future_forecast["yhat_upper"] - future_forecast["yhat_lower"]).mean()
            ),
            "trend_direction": "increasing"
            if future_forecast["trend"].iloc[-1] > future_forecast["trend"].iloc[0]
            else "decreasing",
        }

    def evaluate_model(self, horizon: str = "30 days") -> pd.DataFrame:
        """Evaluate model performance using cross-validation.

        Args:
            horizon: Forecast horizon for evaluation

        Returns:
            DataFrame with performance metrics (MAE, RMSE, MAPE)

        Raises:
            ValueError: If model hasn't been fitted
        """
        if not self.fitted or self.model is None:
            raise ValueError("Model must be fitted before evaluation")

        logger.info(f"Evaluating model with {horizon} horizon")

        # Perform cross-validation
        df_cv = cross_validation(
            self.model, initial="180 days", period="30 days", horizon=horizon
        )

        # Calculate performance metrics
        df_performance = performance_metrics(df_cv)

        logger.info("Model evaluation complete")
        return df_performance

    def detect_anomalies(
        self, data: pd.DataFrame, threshold: float = 0.95
    ) -> pd.DataFrame:
        """Detect anomalies in the data using fitted model.

        Args:
            data: Historical data to check for anomalies
            threshold: Confidence threshold (default 0.95 = 95%)

        Returns:
            DataFrame with anomaly flags

        Raises:
            ValueError: If model hasn't been fitted
        """
        if not self.fitted or self.model is None:
            raise ValueError("Model must be fitted before anomaly detection")

        df = self.prepare_data(data)
        forecast = self.model.predict(df)

        # Merge actual values with forecast
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
            df[["ds", "y"]], on="ds"
        )

        # Flag anomalies where actual value is outside confidence interval
        result["is_anomaly"] = (result["y"] < result["yhat_lower"]) | (
            result["y"] > result["yhat_upper"]
        )

        result["anomaly_type"] = result.apply(
            lambda row: (
                "spike"
                if row["y"] > row["yhat_upper"]
                else "drop" if row["y"] < row["yhat_lower"] else "normal"
            ),
            axis=1,
        )

        logger.info(
            f"Detected {result['is_anomaly'].sum()} anomalies in {len(result)} data points"
        )

        return result

    def get_component_importance(self) -> Dict[str, float]:
        """Get the importance of different seasonal components.

        Returns:
            Dictionary with component importance scores

        Raises:
            ValueError: If model hasn't been fitted
        """
        if not self.fitted or self.model is None:
            raise ValueError("Model must be fitted first")

        components = {}

        # Get component scales from the model
        if self.yearly_seasonality:
            components["yearly"] = float(
                np.std(self.model.seasonalities["yearly"]["data"])
            )

        if self.weekly_seasonality:
            components["weekly"] = float(
                np.std(self.model.seasonalities["weekly"]["data"])
            )

        components["trend"] = float(np.std(self.model.history["trend"]))

        return components
