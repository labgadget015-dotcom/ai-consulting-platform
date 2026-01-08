"""Application configuration using Pydantic settings."""

import os
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "AI Consulting Platform"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # API Configuration
    api_base_url: str = Field(default="http://localhost:8000", env="API_BASE_URL")
    api_rate_limit: str = Field(default="100/hour", env="API_RATE_LIMIT")

    # Shopify Integration
    shopify_api_key: Optional[str] = Field(default=None, env="SHOPIFY_API_KEY")
    shopify_api_secret: Optional[str] = Field(default=None, env="SHOPIFY_API_SECRET")
    shopify_access_token: Optional[str] = Field(default=None, env="SHOPIFY_ACCESS_TOKEN")
    shopify_shop_url: Optional[str] = Field(default=None, env="SHOPIFY_SHOP_URL")

    # Google Cloud
    google_cloud_project_id: Optional[str] = Field(
        default=None, env="GOOGLE_CLOUD_PROJECT_ID"
    )
    google_application_credentials: Optional[str] = Field(
        default=None, env="GOOGLE_APPLICATION_CREDENTIALS"
    )

    # BigQuery
    bigquery_dataset_id: str = Field(
        default="ai_consulting_dataset", env="BIGQUERY_DATASET_ID"
    )
    bigquery_table_prefix: str = Field(default="prod_", env="BIGQUERY_TABLE_PREFIX")

    # Database
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/ai_consulting",
        env="DATABASE_URL",
    )
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")

    # Security
    secret_key: str = Field(default="your_secret_key_here_min_32_chars", env="SECRET_KEY")
    jwt_secret_key: str = Field(default="your_jwt_secret_key_here", env="JWT_SECRET_KEY")
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")

    # Email/SMTP
    smtp_host: str = Field(default="smtp.gmail.com", env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_user: Optional[str] = Field(default=None, env="SMTP_USER")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    smtp_from_email: str = Field(
        default="noreply@aiconsulting.com", env="SMTP_FROM_EMAIL"
    )

    # Analytics
    ga_tracking_id: Optional[str] = Field(default=None, env="GA_TRACKING_ID")
    ga_measurement_id: Optional[str] = Field(default=None, env="GA_MEASUREMENT_ID")

    # Stripe
    stripe_api_key: Optional[str] = Field(default=None, env="STRIPE_API_KEY")
    stripe_publishable_key: Optional[str] = Field(
        default=None, env="STRIPE_PUBLISHABLE_KEY"
    )
    stripe_webhook_secret: Optional[str] = Field(
        default=None, env="STRIPE_WEBHOOK_SECRET"
    )

    # AWS
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(
        default=None, env="AWS_SECRET_ACCESS_KEY"
    )
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_s3_bucket: Optional[str] = Field(default=None, env="AWS_S3_BUCKET")

    # Feature Flags
    enable_forecasting: bool = Field(default=True, env="ENABLE_FORECASTING")
    enable_churn_prediction: bool = Field(default=True, env="ENABLE_CHURN_PREDICTION")
    enable_inventory_optimization: bool = Field(
        default=True, env="ENABLE_INVENTORY_OPTIMIZATION"
    )

    # ML Models
    model_cache_dir: str = Field(default="./models/cache", env="MODEL_CACHE_DIR")
    model_version: str = Field(default="v1.0.0", env="MODEL_VERSION")
    model_update_interval: int = Field(
        default=86400, env="MODEL_UPDATE_INTERVAL"
    )  # seconds

    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    apm_service_name: str = Field(
        default="ai-consulting-platform", env="APM_SERVICE_NAME"
    )
    apm_environment: str = Field(default="development", env="APM_ENVIRONMENT")

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


# Global settings instance
settings = Settings()
