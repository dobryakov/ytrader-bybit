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
    
    # Dataset Quality Control Configuration
    dataset_max_feature_nan_ratio: float = Field(default=0.5, env="DATASET_MAX_FEATURE_NAN_RATIO", description="Maximum ratio of NaN values per feature column (0.0-1.0, default: 0.5 = 50%)")
    dataset_max_row_nan_ratio: float = Field(default=0.8, env="DATASET_MAX_ROW_NAN_RATIO", description="Maximum ratio of NaN values per row across all features (0.0-1.0, default: 0.8 = 80%). Rows exceeding this will be dropped.")
    dataset_min_valid_features_ratio: float = Field(default=0.3, env="DATASET_MIN_VALID_FEATURES_RATIO", description="Minimum ratio of valid (non-NaN) features per row (0.0-1.0, default: 0.3 = 30%). Rows below this will be dropped.")
    dataset_fail_on_high_nan_ratio: bool = Field(default=False, env="DATASET_FAIL_ON_HIGH_NAN_RATIO", description="Fail dataset build if any feature has NaN ratio above threshold (default: False, only logs warning)")
    
    # Redis Configuration (PRIMARY cache)
    redis_host: str = Field(default="redis", env="REDIS_HOST", description="Redis hostname (default: 'redis' for Docker service name)")
    redis_port: int = Field(default=6379, env="REDIS_PORT", description="Redis port")
    redis_db: int = Field(default=0, env="REDIS_DB", description="Redis database number")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD", description="Redis password (optional)")
    redis_max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS", description="Redis connection pool size")
    redis_socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT", description="Redis socket timeout in seconds")
    redis_socket_connect_timeout: int = Field(default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT", description="Redis socket connect timeout in seconds")
    cache_redis_enabled: bool = Field(default=True, env="CACHE_REDIS_ENABLED", description="Enable Redis as primary cache (default: true). If false, will use memory cache only.")
    
    # Cache Configuration
    cache_ttl_historical_data_seconds: int = Field(default=86400, env="CACHE_TTL_HISTORICAL_DATA_SECONDS", description="TTL for historical data cache entries in seconds (default: 86400 = 24 hours). Applies to both Redis and memory fallback cache.")
    cache_ttl_features_seconds: int = Field(default=604800, env="CACHE_TTL_FEATURES_SECONDS", description="TTL for computed features cache entries in seconds (default: 604800 = 7 days). Applies to both Redis and memory fallback cache.")
    cache_max_size_mb: int = Field(default=1024, env="CACHE_MAX_SIZE_MB", description="Maximum cache size in MB for memory fallback cache only (default: 1024 = 1GB). Redis uses maxmemory setting from Redis configuration.")
    cache_max_entries: int = Field(default=10000, env="CACHE_MAX_ENTRIES", description="Maximum number of cache entries for memory fallback cache only (default: 10000). Redis uses maxmemory-policy from Redis configuration.")
    dataset_builder_cache_enabled: bool = Field(default=True, env="DATASET_BUILDER_CACHE_ENABLED", description="Enable caching for dataset building (default: true). Uses Redis if available, falls back to memory cache.")
    dataset_builder_cache_historical_data_enabled: bool = Field(default=True, env="DATASET_BUILDER_CACHE_HISTORICAL_DATA_ENABLED", description="Enable historical data caching (default: true). Uses Redis if available, falls back to memory cache.")
    dataset_builder_cache_features_enabled: bool = Field(default=True, env="DATASET_BUILDER_CACHE_FEATURES_ENABLED", description="Enable computed features caching (default: true). Uses Redis if available, falls back to memory cache.")
    cache_invalidation_on_registry_change: bool = Field(default=True, env="CACHE_INVALIDATION_ON_REGISTRY_CHANGE", description="Automatically invalidate cache when Feature Registry version changes (default: true). Works for both Redis and memory cache.")
    cache_invalidation_on_data_change: bool = Field(default=True, env="CACHE_INVALIDATION_ON_DATA_CHANGE", description="Automatically invalidate cache when historical data files are modified (default: true). Works for both Redis and memory cache.")
    
    # Dataset Builder Configuration
    dataset_builder_batch_size: int = Field(default=1000, env="DATASET_BUILDER_BATCH_SIZE", description="Batch size for processing timestamps in dataset builder (default: 1000)")
    
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
    
    @field_validator("postgres_port", "rabbitmq_port", "feature_service_port", "ws_gateway_port", "redis_port")
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

