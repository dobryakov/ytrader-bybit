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

    # Redis Configuration
    redis_host: str = Field(default="redis", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")

    # WebSocket Gateway Configuration
    ws_gateway_host: str = Field(default="ws-gateway", alias="WS_GATEWAY_HOST")
    ws_gateway_port: int = Field(default=4400, alias="WS_GATEWAY_PORT")
    ws_gateway_api_key: str = Field(..., alias="WS_GATEWAY_API_KEY")

    # Position Manager Configuration
    position_manager_host: str = Field(default="position-manager", alias="POSITION_MANAGER_HOST")
    position_manager_port: int = Field(default=4800, alias="POSITION_MANAGER_PORT")
    position_manager_api_key: str = Field(..., alias="POSITION_MANAGER_API_KEY")

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

    # Instruments-info refresh configuration
    order_manager_instrument_info_refresh_interval: int = Field(
        default=3600, alias="ORDERMANAGER_INSTRUMENT_INFO_REFRESH_INTERVAL"
    )

    # Fee rates cache configuration
    order_manager_fee_data_ttl_seconds: int = Field(
        default=3600,
        alias="ORDERMANAGER_FEE_DATA_TTL_SECONDS",
        description="TTL in seconds for cached Bybit fee rates in bybit_fee_rates table",
    )
    order_manager_max_fallback_fee_rate: float = Field(
        default=0.001,
        alias="ORDERMANAGER_MAX_FALLBACK_FEE_RATE",
        description="Conservative max fee rate (e.g., 0.001 = 0.1%) used when live fee data is unavailable",
    )
    order_manager_enable_min_notional_fee_check: bool = Field(
        default=True,
        alias="ORDERMANAGER_ENABLE_MIN_NOTIONAL_FEE_CHECK",
        description="Enable check that rejects orders whose notional value is less than or equal to expected fee",
    )

    # Pending Order Cancellation Configuration
    order_manager_pending_order_timeout_minutes: int = Field(
        default=5, alias="ORDERMANAGER_PENDING_ORDER_TIMEOUT_MINUTES",
        description="Timeout in minutes after which pending orders will be automatically cancelled"
    )
    order_manager_pending_order_check_interval: int = Field(
        default=60, alias="ORDERMANAGER_PENDING_ORDER_CHECK_INTERVAL",
        description="Interval in seconds for checking pending orders that exceed timeout"
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
    order_manager_enable_balance_check: bool = Field(
        default=True, alias="ORDERMANAGER_ENABLE_BALANCE_CHECK",
        description="Enable balance check before order creation. If False, balance validation is skipped (Bybit API will reject orders with insufficient balance)."
    )
    order_manager_enable_order_size_reduction: bool = Field(
        default=True, alias="ORDERMANAGER_ENABLE_ORDER_SIZE_REDUCTION",
        description="Enable automatic order size reduction when insufficient balance error (110007) occurs. If True, system will try to reduce order size and retry."
    )

    # Take Profit / Stop Loss Configuration
    order_manager_tp_sl_enabled: bool = Field(
        default=True, alias="ORDERMANAGER_TP_SL_ENABLED",
        description="Enable TP/SL order creation on Bybit. If True, take profit and stop loss orders will be created together with main order."
    )
    order_manager_tp_enabled: bool = Field(
        default=True, alias="ORDERMANAGER_TP_ENABLED",
        description="Enable take profit orders. If True, TP orders will be created when main order is placed."
    )
    order_manager_tp_threshold_pct: float = Field(
        default=3.0, alias="ORDERMANAGER_TP_THRESHOLD_PCT",
        description="Take profit threshold as percentage. For buy orders: TP = entry_price * (1 + threshold/100). For sell orders: TP = entry_price * (1 - threshold/100). Default: 3.0%"
    )
    order_manager_sl_enabled: bool = Field(
        default=True, alias="ORDERMANAGER_SL_ENABLED",
        description="Enable stop loss orders. If True, SL orders will be created when main order is placed."
    )
    order_manager_sl_threshold_pct: float = Field(
        default=-2.0, alias="ORDERMANAGER_SL_THRESHOLD_PCT",
        description="Stop loss threshold as percentage (negative value). For buy orders: SL = entry_price * (1 - abs(threshold)/100). For sell orders: SL = entry_price * (1 + abs(threshold)/100). Default: -2.0%"
    )
    order_manager_tp_sl_priority: str = Field(
        default="metadata", alias="ORDERMANAGER_TP_SL_PRIORITY",
        description="Priority for TP/SL calculation: 'metadata' (use signal.metadata if available), 'settings' (use .env values), 'both' (try metadata first, fallback to settings). Default: 'metadata'"
    )
    order_manager_tp_sl_trigger_by: str = Field(
        default="LastPrice", alias="ORDERMANAGER_TP_SL_TRIGGER_BY",
        description="TP/SL trigger method: 'LastPrice', 'IndexPrice', or 'MarkPrice'. Default: 'LastPrice'"
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

    @field_validator("order_manager_instrument_info_refresh_interval")
    @classmethod
    def validate_instrument_info_refresh_interval(cls, v: int) -> int:
        """Validate instruments-info refresh interval is positive."""
        if v <= 0:
            raise ValueError("Instruments-info refresh interval must be positive")
        return v

    @field_validator("order_manager_fee_data_ttl_seconds")
    @classmethod
    def validate_fee_data_ttl_seconds(cls, v: int) -> int:
        """Validate fee data TTL is positive."""
        if v <= 0:
            raise ValueError("Fee data TTL must be positive")
        return v

    @field_validator("order_manager_max_fallback_fee_rate")
    @classmethod
    def validate_max_fallback_fee_rate(cls, v: float) -> float:
        """Validate fallback fee rate is non-negative."""
        if v < 0.0:
            raise ValueError("Max fallback fee rate must be non-negative")
        return v

    @field_validator("order_manager_tp_threshold_pct")
    @classmethod
    def validate_tp_threshold_pct(cls, v: float) -> float:
        """Validate TP threshold is positive."""
        if v <= 0:
            raise ValueError("TP threshold percentage must be positive")
        return v

    @field_validator("order_manager_sl_threshold_pct")
    @classmethod
    def validate_sl_threshold_pct(cls, v: float) -> float:
        """Validate SL threshold is negative."""
        if v >= 0:
            raise ValueError("SL threshold percentage must be negative")
        return v

    @field_validator("order_manager_tp_sl_priority")
    @classmethod
    def validate_tp_sl_priority(cls, v: str) -> str:
        """Validate TP/SL priority is one of valid options."""
        valid_priorities = {"metadata", "settings", "both"}
        v_lower = v.lower()
        if v_lower not in valid_priorities:
            raise ValueError(f"TP/SL priority must be one of {valid_priorities}")
        return v_lower

    @field_validator("order_manager_tp_sl_trigger_by")
    @classmethod
    def validate_tp_sl_trigger_by(cls, v: str) -> str:
        """Validate TP/SL trigger method is one of valid options."""
        valid_triggers = {"LastPrice", "IndexPrice", "MarkPrice"}
        if v not in valid_triggers:
            raise ValueError(f"TP/SL trigger method must be one of {valid_triggers}")
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
        return f"http://{self.ws_gateway_host}:{self.ws_gateway_port}"

    @property
    def bybit_api_base_url(self) -> str:
        """Get Bybit API base URL based on environment."""
        if self.bybit_environment == "testnet":
            return "https://api-testnet.bybit.com"
        return "https://api.bybit.com"


# Global settings instance
settings = Settings()

