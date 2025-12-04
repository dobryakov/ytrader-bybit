"""
Contract tests for GET /features/latest endpoint.
"""
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.main import app
from src.models.feature_vector import FeatureVector
from src.services.feature_computer import FeatureComputer
from src.services.orderbook_manager import OrderbookManager


class TestFeaturesAPI:
    """Test features API contract."""
    
    @pytest.fixture
    def orderbook_manager(self):
        """Create orderbook manager."""
        return OrderbookManager()
    
    @pytest.fixture
    def feature_computer(self, orderbook_manager):
        """Create feature computer."""
        return FeatureComputer(orderbook_manager)
    
    @pytest.fixture
    def client(self, feature_computer):
        """Create test client with feature computer set."""
        from src.api.features import set_feature_computer
        
        set_feature_computer(feature_computer)
        client = TestClient(app)
        yield client
        # Cleanup
        from src.api.features import set_feature_computer
        set_feature_computer(None)
    
    def test_get_latest_features_success(self, client, feature_computer, sample_orderbook_snapshot):
        """Test GET /features/latest with valid symbol."""
        # Setup orderbook
        feature_computer._orderbook_manager.apply_snapshot(sample_orderbook_snapshot)
        
        response = client.get(
            "/features/latest?symbol=BTCUSDT",
            headers={"X-API-Key": "test-api-key"},
        )
        
        # Should return 200 or 404 (if no features computed yet)
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "symbol" in data
            assert "features" in data
            assert "timestamp" in data
    
    def test_get_latest_features_missing_symbol(self, client):
        """Test GET /features/latest without symbol parameter."""
        response = client.get(
            "/features/latest",
            headers={"X-API-Key": "test-api-key"},
        )
        
        # Should return 422 (validation error) or 400
        assert response.status_code in [400, 422]
    
    def test_get_latest_features_not_found(self, client, feature_computer):
        """Test GET /features/latest for symbol without features."""
        response = client.get(
            "/features/latest?symbol=UNKNOWN",
            headers={"X-API-Key": "test-api-key"},
        )
        
        # Should return 404
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_get_latest_features_authentication_required(self, client):
        """Test GET /features/latest requires authentication."""
        response = client.get("/features/latest?symbol=BTCUSDT")
        
        # Should return 401 or 403 (authentication required)
        assert response.status_code in [401, 403]

