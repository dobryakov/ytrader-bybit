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


def test_threshold_optimization_metric_validation():
    """Test that MODEL_TRAINING_THRESHOLD_OPTIMIZATION_METRIC validation works correctly."""
    from src.config.settings import Settings
    
    # Test valid classification metrics
    valid_classification_metrics = ["f1", "pr_auc", "roc_auc", "balanced_accuracy", "recall", "accuracy"]
    for metric in valid_classification_metrics:
        settings = Settings.model_validate({"MODEL_TRAINING_THRESHOLD_OPTIMIZATION_METRIC": metric})
        assert settings.model_training_threshold_optimization_metric == metric.lower()
    
    # Test valid regression metrics
    valid_regression_metrics = ["r2_score", "directional_accuracy", "sharpe_ratio", "information_coefficient", "ic", "rmse"]
    for metric in valid_regression_metrics:
        settings = Settings.model_validate({"MODEL_TRAINING_THRESHOLD_OPTIMIZATION_METRIC": metric})
        assert settings.model_training_threshold_optimization_metric == metric.lower()
    
    # Test that roc_auc is specifically accepted
    settings = Settings.model_validate({"MODEL_TRAINING_THRESHOLD_OPTIMIZATION_METRIC": "roc_auc"})
    assert settings.model_training_threshold_optimization_metric == "roc_auc"
    
    # Test that invalid metric raises error
    with pytest.raises(ValueError, match="MODEL_TRAINING_THRESHOLD_OPTIMIZATION_METRIC must be one of"):
        Settings.model_validate({"MODEL_TRAINING_THRESHOLD_OPTIMIZATION_METRIC": "invalid_metric"})
    
    # Test case insensitivity
    settings = Settings.model_validate({"MODEL_TRAINING_THRESHOLD_OPTIMIZATION_METRIC": "ROC_AUC"})
    assert settings.model_training_threshold_optimization_metric == "roc_auc"

