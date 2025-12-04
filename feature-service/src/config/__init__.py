"""
Configuration management using pydantic-settings.
"""
from pydantic import Field
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
    feature_service_data_dir: str = Field(default="/data/feature-service", env="FEATURE_SERVICE_DATA_DIR", description="Data directory")
    feature_service_retention_days: int = Field(default=90, env="FEATURE_SERVICE_RETENTION_DAYS", description="Data retention days")
    feature_service_service_name: str = Field(default="feature-service", env="FEATURE_SERVICE_SERVICE_NAME", description="Service identifier")
    feature_service_symbols: str = Field(default="", env="FEATURE_SERVICE_SYMBOLS", description="Comma-separated list of symbols")
    
    # Feature Registry Configuration
    feature_registry_path: str = Field(default="/app/config/feature_registry.yaml", env="FEATURE_REGISTRY_CONFIG_PATH", description="Feature Registry config path")
    
    # WS Gateway API Configuration (for subscription management)
    ws_gateway_host: str = Field(..., env="WS_GATEWAY_HOST", description="WS Gateway hostname")
    ws_gateway_port: int = Field(..., env="WS_GATEWAY_PORT", description="WS Gateway port")
    ws_gateway_api_key: Optional[str] = Field(default=None, env="WS_GATEWAY_API_KEY", description="WS Gateway API key")
    
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
    
    def validate_port(self, value: int) -> int:
        """Validate port range."""
        if not (1 <= value <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {value}")
        return value
    
    def validate_retention_days(self, value: int) -> int:
        """Validate retention days."""
        if value < 0:
            raise ValueError(f"Retention days must be non-negative, got {value}")
        return value


# Global configuration instance
config = Config()

