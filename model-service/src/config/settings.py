"""
Configuration management using pydantic-settings.

Loads configuration from environment variables with validation and type conversion.
"""

from typing import Optional, List
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
    model_service_port: int = Field(default=4500, alias="MODEL_SERVICE_PORT")
    model_service_api_key: str = Field(..., alias="MODEL_SERVICE_API_KEY")
    model_service_log_level: str = Field(default="INFO", alias="MODEL_SERVICE_LOG_LEVEL")
    model_service_service_name: str = Field(default="model-service", alias="MODEL_SERVICE_SERVICE_NAME")

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
    ws_gateway_host: str = Field(default="ws-gateway", alias="WS_GATEWAY_HOST")
    ws_gateway_port: int = Field(default=4400, alias="WS_GATEWAY_PORT")
    ws_gateway_api_key: str = Field(..., alias="WS_GATEWAY_API_KEY")

    # Model Storage Configuration
    model_storage_path: str = Field(default="/models", alias="MODEL_STORAGE_PATH")

    # Model Training Configuration
    model_training_min_dataset_size: int = Field(default=1000, alias="MODEL_TRAINING_MIN_DATASET_SIZE")
    model_training_max_duration_seconds: int = Field(default=1800, alias="MODEL_TRAINING_MAX_DURATION_SECONDS")
    model_quality_threshold_accuracy: float = Field(default=0.75, alias="MODEL_QUALITY_THRESHOLD_ACCURACY")
    model_retraining_schedule: Optional[str] = Field(default=None, alias="MODEL_RETRAINING_SCHEDULE")

    # Signal Generation Configuration
    signal_generation_rate_limit: int = Field(default=60, alias="SIGNAL_GENERATION_RATE_LIMIT")
    signal_generation_burst_allowance: int = Field(default=10, alias="SIGNAL_GENERATION_BURST_ALLOWANCE")

    # Warm-up Mode Configuration
    warmup_mode_enabled: bool = Field(default=True, alias="WARMUP_MODE_ENABLED")
    warmup_signal_frequency: int = Field(default=60, alias="WARMUP_SIGNAL_FREQUENCY")

    # Trading Strategy Configuration
    trading_strategies: Optional[str] = Field(default=None, alias="TRADING_STRATEGIES")

    @field_validator("model_service_log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the standard levels."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    @field_validator("model_quality_threshold_accuracy")
    @classmethod
    def validate_accuracy_threshold(cls, v: float) -> float:
        """Validate accuracy threshold is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Accuracy threshold must be between 0.0 and 1.0")
        return v

    @field_validator("model_training_min_dataset_size")
    @classmethod
    def validate_min_dataset_size(cls, v: int) -> int:
        """Validate minimum dataset size is positive."""
        if v <= 0:
            raise ValueError("Minimum dataset size must be positive")
        return v

    @field_validator("signal_generation_rate_limit")
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        """Validate rate limit is positive."""
        if v <= 0:
            raise ValueError("Rate limit must be positive")
        return v

    @field_validator("warmup_signal_frequency")
    @classmethod
    def validate_warmup_frequency(cls, v: int) -> int:
        """Validate warm-up signal frequency is positive."""
        if v <= 0:
            raise ValueError("Warm-up signal frequency must be positive")
        return v

    @field_validator("warmup_randomness_level")
    @classmethod
    def validate_randomness_level(cls, v: float) -> float:
        """Validate randomness level is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Randomness level must be between 0.0 and 1.0")
        return v

    @field_validator("warmup_randomness_level")
    @classmethod
    def validate_randomness_level(cls, v: float) -> float:
        """Validate randomness level is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Randomness level must be between 0.0 and 1.0")
        return v

    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL."""
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
    def trading_strategy_list(self) -> List[str]:
        """Parse trading strategies from comma-separated string."""
        if not self.trading_strategies:
            return []
        return [s.strip() for s in self.trading_strategies.split(",") if s.strip()]

    def validate_on_startup(self) -> None:
        """
        Validate configuration on startup.

        Checks file paths, database connectivity, and other critical settings.
        """
        import os
        from pathlib import Path

        # Validate model storage path exists and is writable
        storage_path = Path(self.model_storage_path)
        if not storage_path.exists():
            try:
                storage_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Cannot create model storage directory: {e}")

        if not os.access(storage_path, os.W_OK):
            raise ValueError(f"Model storage path is not writable: {self.model_storage_path}")


# Global settings instance
settings = Settings()

