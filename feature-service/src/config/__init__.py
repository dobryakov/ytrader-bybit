"""
Configuration management using pydantic-settings.
"""
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""
    
    # PostgreSQL Configuration
    postgres_host: str = Field(..., env="POSTGRES_HOST", description="PostgreSQL hostname")
    postgres_port: int = Field(..., env="POSTGRES_PORT", description="PostgreSQL port")
    postgres_db: str = Field(..., env="POSTGRES_DB", description="PostgreSQL database name")
    postgres_user: str = Field(..., env="POSTGRES_USER", description="PostgreSQL username")
    postgres_password: str = Field(..., env="POSTGRES_PASSWORD", description="PostgreSQL password")
    
    # RabbitMQ Configuration
    rabbitmq_host: str = Field(..., env="RABBITMQ_HOST", description="RabbitMQ hostname")
    rabbitmq_port: int = Field(..., env="RABBITMQ_PORT", description="RabbitMQ port")
    rabbitmq_user: str = Field(..., env="RABBITMQ_USER", description="RabbitMQ username")
    rabbitmq_password: str = Field(..., env="RABBITMQ_PASSWORD", description="RabbitMQ password")
    
    # Feature Service Configuration
    feature_service_port: int = Field(..., env="FEATURE_SERVICE_PORT", description="Feature Service REST API port")
    feature_service_api_key: str = Field(..., env="FEATURE_SERVICE_API_KEY", description="Feature Service API key")
    feature_service_log_level: str = Field(default="INFO", env="FEATURE_SERVICE_LOG_LEVEL", description="Logging level")
    feature_service_data_dir: str = Field(default="/data", env="FEATURE_SERVICE_DATA_DIR", description="Data directory")
    feature_service_raw_data_path: str = Field(default="/data/raw", env="FEATURE_SERVICE_RAW_DATA_PATH", description="Raw data storage path")
    feature_service_dataset_storage_path: str = Field(default="/data/datasets", env="FEATURE_SERVICE_DATASET_STORAGE_PATH", description="Dataset storage path")
    feature_service_retention_days: int = Field(default=90, env="FEATURE_SERVICE_RETENTION_DAYS", description="Data retention days")
    feature_service_service_name: str = Field(default="feature-service", env="FEATURE_SERVICE_SERVICE_NAME", description="Service identifier")
    feature_service_symbols: str = Field(default="", env="FEATURE_SERVICE_SYMBOLS", description="Comma-separated list of symbols")
    
    # Feature Registry Configuration
    feature_registry_path: str = Field(default="/app/config/feature_registry.yaml", env="FEATURE_REGISTRY_CONFIG_PATH", description="Feature Registry config path")
    feature_registry_versions_dir: str = Field(default="/app/config/versions", env="FEATURE_REGISTRY_VERSIONS_DIR", description="Directory for Feature Registry version files")
    feature_registry_use_db: bool = Field(default=True, env="FEATURE_REGISTRY_USE_DB", description="Use database-driven version management (default: True)")
    feature_registry_auto_migrate: bool = Field(default=True, env="FEATURE_REGISTRY_AUTO_MIGRATE", description="Automatically migrate legacy feature_registry.yaml to versioned storage (default: True)")
    
    # WS Gateway API Configuration (for subscription management)
    ws_gateway_host: str = Field(..., env="WS_GATEWAY_HOST", description="WS Gateway hostname")
    ws_gateway_port: int = Field(..., env="WS_GATEWAY_PORT", description="WS Gateway port")
    ws_gateway_api_key: Optional[str] = Field(default=None, env="WS_GATEWAY_API_KEY", description="WS Gateway API key")
    
    # Bybit REST API Configuration (for historical data backfilling)
    bybit_api_key: Optional[str] = Field(default=None, env="BYBIT_API_KEY", description="Bybit API key (optional for public market data endpoints)")
    bybit_api_secret: Optional[str] = Field(default=None, env="BYBIT_API_SECRET", description="Bybit API secret (optional for public market data endpoints)")
    bybit_environment: str = Field(default="testnet", env="BYBIT_ENVIRONMENT", description="Bybit environment: mainnet or testnet (default: testnet for development)")
    feature_service_backfill_rate_limit_delay_ms: int = Field(default=100, env="FEATURE_SERVICE_BACKFILL_RATE_LIMIT_DELAY_MS", description="Delay between API requests in milliseconds to respect rate limits (for backfilling)")
    
    # Backfilling Configuration
    feature_service_backfill_enabled: bool = Field(default=True, env="FEATURE_SERVICE_BACKFILL_ENABLED", description="Enable/disable backfilling feature")
    feature_service_backfill_auto: bool = Field(default=True, env="FEATURE_SERVICE_BACKFILL_AUTO", description="Enable/disable automatic backfilling when data insufficient")
    feature_service_backfill_max_days: int = Field(default=90, env="FEATURE_SERVICE_BACKFILL_MAX_DAYS", description="Maximum days to backfill in one operation")
    feature_service_backfill_default_interval: int = Field(default=1, env="FEATURE_SERVICE_BACKFILL_DEFAULT_INTERVAL", description="Default kline interval in minutes (1 = 1 minute)")
    
    # Model Training Configuration
    model_classification_threshold: float = Field(default=0.005, env="MODEL_CLASSIFICATION_THRESHOLD", description="Classification threshold for target computation (default: 0.005 = 0.5%)")
    
    @property
    def bybit_rest_base_url(self) -> str:
        """Get Bybit REST API base URL based on environment."""
        if self.bybit_environment == "testnet":
            return "https://api-testnet.bybit.com"
        return "https://api.bybit.com"
    
    @property
    def ws_gateway_api_url(self) -> str:
        """Get ws-gateway API URL."""
        return f"http://{self.ws_gateway_host}:{self.ws_gateway_port}"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator("postgres_port", "rabbitmq_port", "feature_service_port", "ws_gateway_port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        """Validate port range."""
        if not (1 <= value <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {value}")
        return value
    
    @field_validator("feature_service_retention_days")
    @classmethod
    def validate_retention_days(cls, value: int) -> int:
        """Validate retention days."""
        if value < 0:
            raise ValueError(f"Retention days must be non-negative, got {value}")
        return value


# Global configuration instance
config = Config()

