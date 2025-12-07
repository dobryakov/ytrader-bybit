"""
Contract tests for backfilling API endpoints.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from datetime import date

from src.api.backfill import router, set_backfilling_service
from src.services.backfilling_service import BackfillingService
from fastapi import FastAPI


class TestBackfillAPI:
    """Contract tests for backfilling API."""
    
    @pytest.fixture
    def mock_backfilling_service(self):
        """Create mock backfilling service."""
        service = MagicMock(spec=BackfillingService)
        service.backfill_historical = AsyncMock(return_value="test-job-id")
        service.get_job_status = MagicMock(return_value={
            "job_id": "test-job-id",
            "symbol": "BTCUSDT",
            "status": "completed",
            "progress": {"dates_completed": 1, "dates_total": 1},
            "start_date": "2025-01-01",
            "end_date": "2025-01-02",
            "data_types": ["klines"],
        })
        return service
    
    @pytest.fixture
    def app(self, mock_backfilling_service):
        """Create FastAPI app with backfilling router."""
        app = FastAPI()
        app.include_router(router)
        set_backfilling_service(mock_backfilling_service)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    def test_post_backfill_historical_contract(self, client, mock_backfilling_service):
        """Test POST /backfill/historical endpoint contract."""
        with patch("src.api.middleware.auth.config") as mock_config:
            mock_config.feature_service_api_key = "test-api-key"
            
            response = client.post(
                "/backfill/historical",
                json={
                    "symbol": "BTCUSDT",
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-02",
                    "data_types": ["klines"],
                },
                headers={"X-API-Key": "test-api-key"},
            )
            
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert "status" in data
            assert "message" in data
            assert data["status"] == "pending"
            
            # Verify service was called
            mock_backfilling_service.backfill_historical.assert_called_once()
    
    def test_post_backfill_historical_without_data_types(self, client, mock_backfilling_service):
        """Test POST /backfill/historical without data_types (uses Feature Registry)."""
        with patch("src.api.middleware.auth.config") as mock_config:
            mock_config.feature_service_api_key = "test-api-key"
            
            response = client.post(
                "/backfill/historical",
                json={
                    "symbol": "BTCUSDT",
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-02",
                },
                headers={"X-API-Key": "test-api-key"},
            )
            
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            
            # Verify service was called with data_types=None
            call_args = mock_backfilling_service.backfill_historical.call_args
            assert call_args[1]["data_types"] is None
    
    def test_post_backfill_auto_contract(self, client, mock_backfilling_service):
        """Test POST /backfill/auto endpoint contract."""
        with patch("src.api.middleware.auth.config") as mock_config:
            mock_config.feature_service_api_key = "test-api-key"
            
            response = client.post(
                "/backfill/auto",
                json={
                    "symbol": "BTCUSDT",
                    "max_days": 30,
                },
                headers={"X-API-Key": "test-api-key"},
            )
            
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert "status" in data
            assert "message" in data
            assert "date_range" in data
            assert "start_date" in data["date_range"]
            assert "end_date" in data["date_range"]
            
            # Verify service was called
            mock_backfilling_service.backfill_historical.assert_called_once()
    
    def test_get_backfill_status_contract(self, client, mock_backfilling_service):
        """Test GET /backfill/status/{job_id} endpoint contract."""
        with patch("src.api.middleware.auth.config") as mock_config:
            mock_config.feature_service_api_key = "test-api-key"
            
            response = client.get(
                "/backfill/status/test-job-id",
                headers={"X-API-Key": "test-api-key"},
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "job_id" in data
            assert "symbol" in data
            assert "status" in data
            assert "progress" in data
            assert "start_date" in data
            assert "end_date" in data
            assert "data_types" in data
            
            # Verify service was called
            mock_backfilling_service.get_job_status.assert_called_once_with("test-job-id")
    
    def test_get_backfill_status_not_found(self, client, mock_backfilling_service):
        """Test GET /backfill/status/{job_id} returns 404 for non-existent job."""
        with patch("src.api.middleware.auth.config") as mock_config:
            mock_config.feature_service_api_key = "test-api-key"
            
            mock_backfilling_service.get_job_status = MagicMock(return_value=None)
            
            response = client.get(
                "/backfill/status/nonexistent-job-id",
                headers={"X-API-Key": "test-api-key"},
            )
            
            assert response.status_code == 404
            data = response.json()
            assert "detail" in data
    
    def test_backfill_historical_authentication_required(self, client):
        """Test that authentication is required for backfilling endpoints."""
        response = client.post(
            "/backfill/historical",
            json={
                "symbol": "BTCUSDT",
                "start_date": "2025-01-01",
                "end_date": "2025-01-02",
            },
        )
        
        # Should require authentication (401 or 403)
        assert response.status_code in [401, 403]
    
    def test_backfill_auto_authentication_required(self, client):
        """Test that authentication is required for auto backfilling endpoint."""
        response = client.post(
            "/backfill/auto",
            json={
                "symbol": "BTCUSDT",
            },
        )
        
        # Should require authentication (401 or 403)
        assert response.status_code in [401, 403]
    
    def test_backfill_status_authentication_required(self, client):
        """Test that authentication is required for status endpoint."""
        response = client.get("/backfill/status/test-job-id")
        
        # Should require authentication (401 or 403)
        assert response.status_code in [401, 403]
    
    def test_backfill_historical_error_response(self, client, mock_backfilling_service):
        """Test error response for backfilling endpoint."""
        with patch("src.api.middleware.auth.config") as mock_config:
            mock_config.feature_service_api_key = "test-api-key"
            
            mock_backfilling_service.backfill_historical = AsyncMock(side_effect=ValueError("Invalid date range"))
            
            response = client.post(
                "/backfill/historical",
                json={
                    "symbol": "BTCUSDT",
                    "start_date": "2025-01-01",
                    "end_date": "2025-01-02",
                },
                headers={"X-API-Key": "test-api-key"},
            )
            
            assert response.status_code == 400
            data = response.json()
            assert "detail" in data
    
    def test_backfill_auto_uses_feature_registry(self, client, mock_backfilling_service):
        """Test that auto backfilling uses Feature Registry to determine data types."""
        with patch("src.api.middleware.auth.config") as mock_config:
            mock_config.feature_service_api_key = "test-api-key"
            
            response = client.post(
                "/backfill/auto",
                json={
                    "symbol": "BTCUSDT",
                },
                headers={"X-API-Key": "test-api-key"},
            )
            
            assert response.status_code == 202
            
            # Verify service was called with data_types=None (should use Feature Registry)
            call_args = mock_backfilling_service.backfill_historical.call_args
            assert call_args[1]["data_types"] is None

