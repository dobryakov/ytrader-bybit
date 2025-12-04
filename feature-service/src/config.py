"""
Configuration management using pydantic-settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""
    
    # PostgreSQL Configuration
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "ytrader"
    postgres_user: str = "ytrader"
    postgres_password: str
    
    # RabbitMQ Configuration
    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"
    
    # Feature Service Configuration
    feature_service_port: int = 4500
    feature_service_api_key: str
    feature_service_log_level: str = "INFO"
    feature_service_data_dir: str = "/data/feature-service"
    feature_service_retention_days: int = 90
    
    # Feature Registry Configuration
    feature_registry_path: str = "/app/config/feature_registry.yaml"
    
    # WS Gateway API Configuration (for subscription management)
    ws_gateway_api_url: str = "http://ws-gateway:8080"
    ws_gateway_api_key: Optional[str] = None
    
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

