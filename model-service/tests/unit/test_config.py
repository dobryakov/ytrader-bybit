"""
Unit tests for configuration management.
"""

import pytest
from src.config.settings import settings
from src.config.exceptions import ConfigurationError


def test_settings_loaded():
    """Test that settings are loaded correctly."""
    assert settings.model_service_port == 4500
    assert settings.model_service_service_name == "model-service"
    assert settings.model_storage_path == "/models"


def test_database_url_format():
    """Test that database URL is formatted correctly."""
    db_url = settings.database_url
    assert db_url.startswith("postgresql://")
    assert settings.postgres_host in db_url
    assert settings.postgres_db in db_url


def test_rabbitmq_url_format():
    """Test that RabbitMQ URL is formatted correctly."""
    rmq_url = settings.rabbitmq_url
    assert rmq_url.startswith("amqp://")
    assert settings.rabbitmq_host in rmq_url


def test_ws_gateway_url_format():
    """Test that WebSocket Gateway URL is formatted correctly."""
    ws_url = settings.ws_gateway_url
    assert ws_url.startswith("http://")
    assert settings.ws_gateway_host in ws_url


def test_trading_strategy_list():
    """Test that trading strategies are parsed correctly."""
    strategies = settings.trading_strategy_list
    assert isinstance(strategies, list)


def test_configuration_validation():
    """Test that configuration validation works."""
    # This should not raise an error if config is valid
    try:
        settings.validate_on_startup()
    except ConfigurationError:
        pytest.fail("Configuration validation should pass with valid settings")

