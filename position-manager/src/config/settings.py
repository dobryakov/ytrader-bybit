"""
Configuration management using pydantic-settings.

Loads configuration from environment variables with validation and type conversion.
"""

from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Service Configuration
    position_manager_port: int = Field(default=4800, alias="POSITION_MANAGER_PORT")
    position_manager_api_key: str = Field(..., alias="POSITION_MANAGER_API_KEY")
    position_manager_log_level: str = Field(default="INFO", alias="POSITION_MANAGER_LOG_LEVEL")
    position_manager_service_name: str = Field(default="position-manager", alias="POSITION_MANAGER_SERVICE_NAME")

    # Database Configuration
    postgres_host: str = Field(default="postgres", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(..., alias="POSTGRES_DB")
    postgres_user: str = Field(..., alias="POSTGRES_USER")
    postgres_password: str = Field(..., alias="POSTGRES_PASSWORD")

    # RabbitMQ Configuration
    rabbitmq_host: str = Field(default="rabbitmq", alias="RABBITMQ_HOST")
    rabbitmq_port: int = Field(default=5672, alias="RABBITMQ_PORT")
    rabbitmq_user: str = Field(default="guest", alias="RABBITMQ_USER")
    rabbitmq_password: str = Field(default="guest", alias="RABBITMQ_PASSWORD")

    # WebSocket Gateway Configuration
    ws_gateway_url: str = Field(default="http://ws-gateway:4400", alias="WS_GATEWAY_URL")
    ws_gateway_api_key: str = Field(..., alias="WS_GATEWAY_API_KEY")

    # Bybit API Configuration (for position synchronization)
    bybit_api_key: Optional[str] = Field(default=None, alias="BYBIT_API_KEY")
    bybit_api_secret: Optional[str] = Field(default=None, alias="BYBIT_API_SECRET")
    bybit_environment: str = Field(default="testnet", alias="BYBIT_ENVIRONMENT")

    # Position Management Configuration
    position_manager_snapshot_interval: int = Field(default=3600, alias="POSITION_MANAGER_SNAPSHOT_INTERVAL")
    position_manager_snapshot_retention_days: int = Field(default=365, alias="POSITION_MANAGER_SNAPSHOT_RETENTION_DAYS")
    position_manager_validation_interval: int = Field(default=1800, alias="POSITION_MANAGER_VALIDATION_INTERVAL")
    position_manager_bybit_sync_interval: int = Field(default=3600, alias="POSITION_MANAGER_BYBIT_SYNC_INTERVAL")
    position_manager_metrics_cache_ttl: int = Field(default=10, alias="POSITION_MANAGER_METRICS_CACHE_TTL")

    # Position Update Strategy
    position_manager_use_ws_avg_price: bool = Field(default=True, alias="POSITION_MANAGER_USE_WS_AVG_PRICE")
    position_manager_avg_price_diff_threshold: float = Field(default=0.001, alias="POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD")
    position_manager_size_validation_threshold: float = Field(default=0.0001, alias="POSITION_MANAGER_SIZE_VALIDATION_THRESHOLD")
    position_manager_price_staleness_threshold: int = Field(default=300, alias="POSITION_MANAGER_PRICE_STALENESS_THRESHOLD")
    position_manager_price_api_timeout: int = Field(default=5, alias="POSITION_MANAGER_PRICE_API_TIMEOUT")
    position_manager_price_api_retries: int = Field(default=3, alias="POSITION_MANAGER_PRICE_API_RETRIES")
    position_manager_optimistic_lock_retries: int = Field(default=3, alias="POSITION_MANAGER_OPTIMISTIC_LOCK_RETRIES")
    position_manager_optimistic_lock_backoff_base: int = Field(default=100, alias="POSITION_MANAGER_OPTIMISTIC_LOCK_BACKOFF_BASE")

    # Timestamp-based conflict resolution (Phase 9)
    position_manager_enable_timestamp_resolution: bool = Field(
        default=True,
        alias="POSITION_MANAGER_ENABLE_TIMESTAMP_RESOLUTION",
        description="Enable timestamp-based size conflict resolution between WebSocket and Order Manager updates.",
    )
    position_manager_timestamp_tolerance_seconds: int = Field(
        default=0,
        alias="POSITION_MANAGER_TIMESTAMP_TOLERANCE_SECONDS",
        description="Optional tolerance window in seconds when comparing WebSocket and Order Manager timestamps.",
    )

    # Rate Limiting
    position_manager_rate_limit_enabled: bool = Field(default=True, alias="POSITION_MANAGER_RATE_LIMIT_ENABLED")
    position_manager_rate_limit_default: int = Field(default=100, alias="POSITION_MANAGER_RATE_LIMIT_DEFAULT")
    position_manager_rate_limit_overrides: Optional[str] = Field(default=None, alias="POSITION_MANAGER_RATE_LIMIT_OVERRIDES")

    # Portfolio limit indicators (optional, used for T073)
    position_manager_portfolio_max_exposure_usdt: Optional[float] = Field(
        default=None,
        alias="POSITION_MANAGER_PORTFOLIO_MAX_EXPOSURE_USDT",
        description="Optional soft limit for total_exposure_usdt; when exceeded, limit_exceeded flag is set in portfolio metrics.",
    )

    @field_validator("position_manager_log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the standard levels."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    @field_validator("position_manager_snapshot_interval")
    @classmethod
    def validate_snapshot_interval(cls, v: int) -> int:
        """Validate snapshot interval is positive."""
        if v <= 0:
            raise ValueError("Snapshot interval must be positive")
        return v

    @field_validator("position_manager_validation_interval")
    @classmethod
    def validate_validation_interval(cls, v: int) -> int:
        """Validate validation interval is positive."""
        if v <= 0:
            raise ValueError("Validation interval must be positive")
        return v

    @field_validator("position_manager_bybit_sync_interval")
    @classmethod
    def validate_bybit_sync_interval(cls, v: int) -> int:
        """Validate Bybit sync interval is positive."""
        if v <= 0:
            raise ValueError("Bybit sync interval must be positive")
        return v

    @field_validator("position_manager_metrics_cache_ttl")
    @classmethod
    def validate_cache_ttl(cls, v: int) -> int:
        """Validate cache TTL is positive."""
        if v <= 0:
            raise ValueError("Cache TTL must be positive")
        return v

    @field_validator("position_manager_optimistic_lock_retries")
    @classmethod
    def validate_optimistic_lock_retries(cls, v: int) -> int:
        """Validate optimistic lock retries is positive."""
        if v <= 0:
            raise ValueError("Optimistic lock retries must be positive")
        return v

    @field_validator("position_manager_timestamp_tolerance_seconds")
    @classmethod
    def validate_timestamp_tolerance(cls, v: int) -> int:
        """Validate timestamp tolerance is non-negative."""
        if v < 0:
            raise ValueError("Timestamp tolerance seconds must be non-negative")
        return v

    @property
    def database_url_async(self) -> str:
        """Get async PostgreSQL connection URL for asyncpg."""
        # asyncpg uses postgresql:// URL format
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def rabbitmq_url(self) -> str:
        """Get RabbitMQ connection URL."""
        return (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}/"
        )


# Global settings instance
settings = Settings()

