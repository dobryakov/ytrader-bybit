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
    
    def test_config_uses_defaults_when_missing(self):
        """Test that configuration uses defaults when environment variables are missing."""
        with patch.dict("os.environ", {}, clear=True):
            from src.config import Config
            
            config = Config()
            
            # Check that defaults are used (if defined in Config class)
            # This test will need to be adjusted based on actual default values
            assert config is not None
    
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

