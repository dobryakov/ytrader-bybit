"""Configuration management using pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Bybit API Configuration
    bybit_api_key: str = Field(..., description="Bybit API key for WebSocket authentication")
    bybit_api_secret: str = Field(..., description="Bybit API secret")
    bybit_environment: str = Field(
        default="testnet", description="Bybit environment: 'testnet' or 'mainnet'"
    )

    # Database Configuration
    postgres_host: str = Field(default="postgres", description="PostgreSQL hostname")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="ytrader", description="PostgreSQL database name")
    postgres_user: str = Field(default="ytrader", description="PostgreSQL username")
    postgres_password: str = Field(..., description="PostgreSQL password")

    # RabbitMQ Configuration
    rabbitmq_host: str = Field(default="rabbitmq", description="RabbitMQ hostname")
    rabbitmq_port: int = Field(default=5672, description="RabbitMQ port")
    rabbitmq_user: str = Field(default="guest", description="RabbitMQ username")
    rabbitmq_password: str = Field(default="guest", description="RabbitMQ password")

    # WebSocket Gateway Service Configuration
    ws_gateway_port: int = Field(default=4400, description="REST API port")
    ws_gateway_api_key: str = Field(..., description="API key for REST API authentication")
    ws_gateway_log_level: str = Field(
        default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
    ws_gateway_service_name: str = Field(
        default="ws-gateway", description="Service identifier"
    )

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_async(self) -> str:
        """Construct async PostgreSQL connection URL for asyncpg."""
        # asyncpg uses postgresql:// URL format but requires separate connection parameters
        # This property returns the connection string format, but asyncpg.create_pool
        # actually takes individual parameters or a connection string
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def rabbitmq_url(self) -> str:
        """Construct RabbitMQ connection URL."""
        return (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}/"
        )

    @property
    def bybit_ws_url(self) -> str:
        """Get Bybit WebSocket URL based on environment."""
        # Use private endpoint for authenticated connections (supports both public and private subscriptions)
        if self.bybit_environment == "mainnet":
            return "wss://stream.bybit.com/v5/private"
        else:
            return "wss://stream-testnet.bybit.com/v5/private"


# Global settings instance
settings = Settings()

