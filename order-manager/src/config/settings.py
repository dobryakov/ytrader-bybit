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
    order_manager_port: int = Field(default=4600, alias="ORDERMANAGER_PORT")
    order_manager_api_key: str = Field(..., alias="ORDERMANAGER_API_KEY")
    order_manager_log_level: str = Field(default="INFO", alias="ORDERMANAGER_LOG_LEVEL")
    order_manager_service_name: str = Field(default="order-manager", alias="ORDERMANAGER_SERVICE_NAME")

    # Bybit API Configuration
    bybit_api_key: str = Field(..., alias="BYBIT_API_KEY")
    bybit_api_secret: str = Field(..., alias="BYBIT_API_SECRET")
    bybit_environment: str = Field(default="testnet", alias="BYBIT_ENVIRONMENT")

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
    order_manager_ws_gateway_host: str = Field(default="ws-gateway", alias="ORDERMANAGER_WS_GATEWAY_HOST")
    order_manager_ws_gateway_port: int = Field(default=4400, alias="ORDERMANAGER_WS_GATEWAY_PORT")
    order_manager_ws_gateway_api_key: str = Field(..., alias="ORDERMANAGER_WS_GATEWAY_API_KEY")

    # Order Execution Configuration
    order_manager_enable_dry_run: bool = Field(default=False, alias="ORDERMANAGER_ENABLE_DRY_RUN")
    order_manager_max_single_order_size: float = Field(default=10000.0, alias="ORDERMANAGER_MAX_SINGLE_ORDER_SIZE")
    order_manager_enable_order_splitting: bool = Field(default=False, alias="ORDERMANAGER_ENABLE_ORDER_SPLITTING")
    order_manager_order_execution_timeout: int = Field(default=30, alias="ORDERMANAGER_ORDER_EXECUTION_TIMEOUT")

    # Risk Limits Configuration
    order_manager_max_position_size: float = Field(default=1.0, alias="ORDERMANAGER_MAX_POSITION_SIZE")
    order_manager_max_exposure: float = Field(default=50000.0, alias="ORDERMANAGER_MAX_EXPOSURE")
    order_manager_max_order_size_ratio: float = Field(default=0.1, alias="ORDERMANAGER_MAX_ORDER_SIZE_RATIO")

    # Bybit API Retry Configuration
    order_manager_bybit_api_retry_max_attempts: int = Field(default=3, alias="ORDERMANAGER_BYBIT_API_RETRY_MAX_ATTEMPTS")
    order_manager_bybit_api_retry_base_delay: float = Field(default=1.0, alias="ORDERMANAGER_BYBIT_API_RETRY_BASE_DELAY")
    order_manager_bybit_api_retry_max_delay: float = Field(default=30.0, alias="ORDERMANAGER_BYBIT_API_RETRY_MAX_DELAY")
    order_manager_bybit_api_retry_multiplier: float = Field(default=2.0, alias="ORDERMANAGER_BYBIT_API_RETRY_MULTIPLIER")

    # Order Type Selection Configuration
    order_manager_market_order_confidence_threshold: float = Field(
        default=0.9, alias="ORDERMANAGER_MARKET_ORDER_CONFIDENCE_THRESHOLD"
    )
    order_manager_market_order_spread_threshold: float = Field(
        default=0.1, alias="ORDERMANAGER_MARKET_ORDER_SPREAD_THRESHOLD"
    )
    order_manager_limit_order_price_offset_ratio: float = Field(
        default=0.5, alias="ORDERMANAGER_LIMIT_ORDER_PRICE_OFFSET_RATIO"
    )

    # Position Management Configuration
    order_manager_position_snapshot_interval: int = Field(
        default=300, alias="ORDERMANAGER_POSITION_SNAPSHOT_INTERVAL"
    )
    order_manager_position_validation_interval: int = Field(
        default=3600, alias="ORDERMANAGER_POSITION_VALIDATION_INTERVAL"
    )

    # Order Cancellation Configuration
    order_manager_cancel_opposite_orders_only: bool = Field(
        default=False, alias="ORDERMANAGER_CANCEL_OPPOSITE_ORDERS_ONLY"
    )
    order_manager_cancel_stale_order_timeout: int = Field(
        default=3600, alias="ORDERMANAGER_CANCEL_STALE_ORDER_TIMEOUT"
    )

    # Risk Management Configuration
    order_manager_unrealized_loss_warning_threshold: float = Field(
        default=10.0, alias="ORDERMANAGER_UNREALIZED_LOSS_WARNING_THRESHOLD"
    )

    @field_validator("order_manager_log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the standard levels."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    @field_validator("bybit_environment")
    @classmethod
    def validate_bybit_environment(cls, v: str) -> str:
        """Validate Bybit environment is testnet or mainnet."""
        valid_environments = {"testnet", "mainnet"}
        v_lower = v.lower()
        if v_lower not in valid_environments:
            raise ValueError(f"Bybit environment must be one of {valid_environments}")
        return v_lower

    @field_validator("order_manager_max_order_size_ratio")
    @classmethod
    def validate_max_order_size_ratio(cls, v: float) -> float:
        """Validate max order size ratio is between 0 and 1."""
        if not 0.0 < v <= 1.0:
            raise ValueError("Max order size ratio must be between 0.0 and 1.0")
        return v

    @field_validator("order_manager_market_order_confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        """Validate confidence threshold is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v

    @field_validator("order_manager_limit_order_price_offset_ratio")
    @classmethod
    def validate_price_offset_ratio(cls, v: float) -> float:
        """Validate price offset ratio is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Price offset ratio must be between 0.0 and 1.0")
        return v

    @field_validator("order_manager_bybit_api_retry_max_attempts")
    @classmethod
    def validate_retry_max_attempts(cls, v: int) -> int:
        """Validate retry max attempts is positive."""
        if v <= 0:
            raise ValueError("Retry max attempts must be positive")
        return v

    @field_validator("order_manager_position_snapshot_interval")
    @classmethod
    def validate_snapshot_interval(cls, v: int) -> int:
        """Validate snapshot interval is positive."""
        if v <= 0:
            raise ValueError("Snapshot interval must be positive")
        return v

    @field_validator("order_manager_position_validation_interval")
    @classmethod
    def validate_validation_interval(cls, v: int) -> int:
        """Validate validation interval is positive."""
        if v <= 0:
            raise ValueError("Validation interval must be positive")
        return v

    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

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

    @property
    def ws_gateway_url(self) -> str:
        """Get WebSocket Gateway base URL."""
        return f"http://{self.order_manager_ws_gateway_host}:{self.order_manager_ws_gateway_port}"

    @property
    def bybit_api_base_url(self) -> str:
        """Get Bybit API base URL based on environment."""
        if self.bybit_environment == "testnet":
            return "https://api-testnet.bybit.com"
        return "https://api.bybit.com"


# Global settings instance
settings = Settings()

