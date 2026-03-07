"""Application configuration using Pydantic settings."""

from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # Application
    app_name: str = "AI Consulting Platform"
    app_version: str = "1.0.0"
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # API Configuration
    api_base_url: str = Field(default="http://localhost:8000")
    api_rate_limit: str = Field(default="100/hour")

    # Shopify Integration
    shopify_api_key: Optional[str] = Field(default=None)
    shopify_api_secret: Optional[str] = Field(default=None)
    shopify_access_token: Optional[str] = Field(default=None)
    shopify_shop_url: Optional[str] = Field(default=None)

    # Google Cloud
    google_cloud_project_id: Optional[str] = Field(default=None)
    google_application_credentials: Optional[str] = Field(default=None)

    # BigQuery
    bigquery_dataset_id: str = Field(default="ai_consulting_dataset")
    bigquery_table_prefix: str = Field(default="prod_")

    # Database
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/ai_consulting"
    )
    database_pool_size: int = Field(default=20)
    database_max_overflow: int = Field(default=10)

    # Security
    secret_key: str = Field(default="your_secret_key_here_min_32_chars")
    jwt_secret_key: str = Field(default="your_jwt_secret_key_here")
    encryption_key: Optional[str] = Field(default=None)

    # Email/SMTP
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_user: Optional[str] = Field(default=None)
    smtp_password: Optional[str] = Field(default=None)
    smtp_from_email: str = Field(default="noreply@aiconsulting.com")

    # Analytics
    ga_tracking_id: Optional[str] = Field(default=None)
    ga_measurement_id: Optional[str] = Field(default=None)

    # Stripe
    stripe_api_key: Optional[str] = Field(default=None)
    stripe_publishable_key: Optional[str] = Field(default=None)
    stripe_webhook_secret: Optional[str] = Field(default=None)

    # AWS
    aws_access_key_id: Optional[str] = Field(default=None)
    aws_secret_access_key: Optional[str] = Field(default=None)
    aws_region: str = Field(default="us-east-1")
    aws_s3_bucket: Optional[str] = Field(default=None)

    # Feature Flags
    enable_forecasting: bool = Field(default=True)
    enable_churn_prediction: bool = Field(default=True)
    enable_inventory_optimization: bool = Field(default=True)

    # ML Models
    model_cache_dir: str = Field(default="./models/cache")
    model_version: str = Field(default="v1.0.0")
    model_update_interval: int = Field(default=86400)  # seconds

    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None)
    apm_service_name: str = Field(default="ai-consulting-platform")
    apm_environment: str = Field(default="development")

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


# Global settings instance
settings = Settings()
