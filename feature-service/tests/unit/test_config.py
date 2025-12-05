"""
Unit tests for configuration management.
"""
import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError


@pytest.fixture
def sample_env_vars():
    """Sample environment variables for testing."""
    return {
        "POSTGRES_HOST": "postgres",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "ytrader",
        "POSTGRES_USER": "ytrader",
        "POSTGRES_PASSWORD": "test_password",
        "RABBITMQ_HOST": "rabbitmq",
        "RABBITMQ_PORT": "5672",
        "RABBITMQ_USER": "guest",
        "RABBITMQ_PASSWORD": "guest",
        "FEATURE_SERVICE_PORT": "4500",
        "FEATURE_SERVICE_API_KEY": "test-api-key",
        "FEATURE_SERVICE_LOG_LEVEL": "INFO",
        "FEATURE_SERVICE_DATA_DIR": "/data/feature-service",
        "FEATURE_SERVICE_RETENTION_DAYS": "90",
        "FEATURE_REGISTRY_PATH": "/app/config/feature_registry.yaml",
        "WS_GATEWAY_HOST": "ws-gateway",
        "WS_GATEWAY_PORT": "4400",
    }


class TestConfig:
    """Tests for configuration management."""
    
    def test_config_loads_from_environment(self, sample_env_vars):
        """Test that configuration loads from environment variables."""
        with patch.dict("os.environ", sample_env_vars):
            from src.config import Config
            
            config = Config()
            
            assert config.postgres_host == "postgres"
            assert config.postgres_port == 5432
            assert config.postgres_db == "ytrader"
            assert config.postgres_user == "ytrader"
            assert config.postgres_password == "test_password"
            assert config.rabbitmq_host == "rabbitmq"
            assert config.rabbitmq_port == 5672
            assert config.feature_service_port == 4500
            assert config.feature_service_api_key == "test-api-key"
            assert config.feature_service_log_level == "INFO"
            assert config.feature_service_data_dir == "/data/feature-service"
            assert config.feature_service_retention_days == 90
            assert config.feature_registry_path == "/app/config/feature_registry.yaml"
    
    def test_config_uses_defaults_when_missing(self, sample_env_vars):
        """Test that configuration uses defaults when environment variables are missing."""
        # Add only required fields, check that defaults are used for optional ones
        minimal_env_vars = {
            "POSTGRES_HOST": sample_env_vars["POSTGRES_HOST"],
            "POSTGRES_PORT": sample_env_vars["POSTGRES_PORT"],
            "POSTGRES_DB": sample_env_vars["POSTGRES_DB"],
            "POSTGRES_USER": sample_env_vars["POSTGRES_USER"],
            "POSTGRES_PASSWORD": sample_env_vars["POSTGRES_PASSWORD"],
            "RABBITMQ_HOST": sample_env_vars["RABBITMQ_HOST"],
            "RABBITMQ_PORT": sample_env_vars["RABBITMQ_PORT"],
            "RABBITMQ_USER": sample_env_vars["RABBITMQ_USER"],
            "RABBITMQ_PASSWORD": sample_env_vars["RABBITMQ_PASSWORD"],
            "FEATURE_SERVICE_PORT": sample_env_vars["FEATURE_SERVICE_PORT"],
            "FEATURE_SERVICE_API_KEY": sample_env_vars["FEATURE_SERVICE_API_KEY"],
            "WS_GATEWAY_HOST": "ws-gateway",
            "WS_GATEWAY_PORT": "4400",
            # Explicitly set optional fields to empty to test defaults
            "FEATURE_SERVICE_SYMBOLS": "",
        }
        
        with patch.dict("os.environ", minimal_env_vars, clear=False):
            # Remove any existing FEATURE_SERVICE_SYMBOLS if present
            import os
            if "FEATURE_SERVICE_SYMBOLS" in os.environ:
                del os.environ["FEATURE_SERVICE_SYMBOLS"]
            
            from src.config import Config
            
            config = Config()
            
            # Check that defaults are used for optional fields
            assert config.feature_service_log_level == "INFO"
            assert config.feature_service_data_dir == "/data/feature-service"
            assert config.feature_service_retention_days == 90
            assert config.feature_service_service_name == "feature-service"
            assert config.feature_service_symbols == ""
            assert config.feature_registry_path == "/app/config/feature_registry.yaml"
    
    def test_config_validates_required_fields(self):
        """Test that configuration validates required fields."""
        with patch.dict("os.environ", {}, clear=True):
            from src.config import Config
            
            # This should raise ValidationError if required fields are missing
            # Adjust based on actual Config implementation
            with pytest.raises((ValidationError, ValueError)):
                config = Config()
    
    def test_config_validates_port_range(self, sample_env_vars):
        """Test that configuration validates port ranges."""
        sample_env_vars["FEATURE_SERVICE_PORT"] = "99999"  # Invalid port
        
        with patch.dict("os.environ", sample_env_vars):
            from src.config import Config
            
            with pytest.raises(ValidationError):
                config = Config()
    
    def test_config_validates_retention_days(self, sample_env_vars):
        """Test that configuration validates retention days."""
        sample_env_vars["FEATURE_SERVICE_RETENTION_DAYS"] = "-1"  # Invalid
        
        with patch.dict("os.environ", sample_env_vars):
            from src.config import Config
            
            with pytest.raises(ValidationError):
                config = Config()

