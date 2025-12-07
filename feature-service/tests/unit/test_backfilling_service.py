"""
Unit tests for backfilling service.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
from datetime import date, datetime, timezone

from src.services.backfilling_service import BackfillingService, BackfillingJob
from src.utils.bybit_client import BybitAPIError
from src.storage.parquet_storage import ParquetStorage
from src.services.feature_registry import FeatureRegistryLoader


class TestBackfillingService:
    """Tests for backfilling service."""
    
    @pytest.fixture
    def mock_parquet_storage(self):
        """Create mock Parquet storage."""
        storage = MagicMock(spec=ParquetStorage)
        storage.write_klines = AsyncMock()
        storage.write_trades = AsyncMock()
        storage.write_orderbook_snapshots = AsyncMock()
        storage.write_orderbook_deltas = AsyncMock()
        storage.write_ticker = AsyncMock()
        storage.write_funding = AsyncMock()
        storage.read_klines = AsyncMock(side_effect=FileNotFoundError())
        storage.read_trades = AsyncMock(side_effect=FileNotFoundError())
        storage.read_orderbook_snapshots = AsyncMock(side_effect=FileNotFoundError())
        storage.read_orderbook_deltas = AsyncMock(side_effect=FileNotFoundError())
        storage.read_ticker = AsyncMock(side_effect=FileNotFoundError())
        storage.read_funding = AsyncMock(side_effect=FileNotFoundError())
        storage._get_klines_path = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=False)))
        storage._get_trades_path = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=False)))
        storage._get_orderbook_snapshots_path = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=False)))
        storage._get_orderbook_deltas_path = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=False)))
        storage._get_ticker_path = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=False)))
        storage._get_funding_path = MagicMock(return_value=MagicMock(exists=MagicMock(return_value=False)))
        return storage
    
    @pytest.fixture
    def mock_bybit_client(self):
        """Create mock Bybit client."""
        client = MagicMock()
        client.get = AsyncMock()
        client.close = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_feature_registry_loader(self):
        """Create mock Feature Registry loader."""
        loader = MagicMock(spec=FeatureRegistryLoader)
        loader.get_required_data_types = MagicMock(return_value={"kline"})
        loader.get_data_type_mapping = MagicMock(return_value={"kline": ["klines"]})
        return loader
    
    @pytest.fixture
    def backfilling_service(self, mock_parquet_storage, mock_feature_registry_loader, mock_bybit_client):
        """Create backfilling service for testing."""
        service = BackfillingService(
            parquet_storage=mock_parquet_storage,
            feature_registry_loader=mock_feature_registry_loader,
            bybit_client=mock_bybit_client,
        )
        return service
    
    @pytest.mark.asyncio
    async def test_backfill_klines_pagination(self, backfilling_service, mock_bybit_client):
        """Test kline backfilling with pagination."""
        # Use real timestamps for 2025-01-01
        base_timestamp = int(datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        interval_ms = 60 * 1000  # 1 minute in milliseconds
        
        # Mock API response with 200 candles (max per request)
        mock_response_1 = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    [str(base_timestamp + i * interval_ms), "50000", "50100", "49900", "50050", "100.5", "5025000"]
                    for i in range(200)
                ],
            },
        }
        
        # Second response starts after first batch
        second_batch_start = base_timestamp + 200 * interval_ms
        mock_response_2 = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    [str(second_batch_start + i * interval_ms), "50000", "50100", "49900", "50050", "100.5", "5025000"]
                    for i in range(100)
                ],
            },
        }
        
        # Return empty list after second call to stop pagination
        mock_response_empty = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {"list": []},
        }
        
        mock_bybit_client.get = AsyncMock(side_effect=[mock_response_1, mock_response_2, mock_response_empty])
        
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 2)
        
        klines = await backfilling_service.backfill_klines("BTCUSDT", start_date, end_date, interval=1)
        
        assert len(klines) == 300  # 200 + 100
        assert mock_bybit_client.get.call_count >= 2  # At least 2 API calls for pagination
    
    @pytest.mark.asyncio
    async def test_backfill_klines_data_format_conversion(self, backfilling_service, mock_bybit_client):
        """Test data format conversion from Bybit API to internal format."""
        # Use timestamp that matches the date range to avoid pagination issues
        timestamp_ms = int(datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        mock_response = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    [str(timestamp_ms), "50000.5", "50100.0", "49900.0", "50050.0", "100.5", "5025000.0"],
                ],
            },
        }
        
        # Return empty list on second call to stop pagination
        mock_response_empty = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {"list": []},
        }
        
        mock_bybit_client.get = AsyncMock(side_effect=[mock_response, mock_response_empty])
        
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 1)
        
        klines = await backfilling_service.backfill_klines("BTCUSDT", start_date, end_date, interval=1)
        
        assert len(klines) == 1
        kline = klines[0]
        assert kline["symbol"] == "BTCUSDT"
        assert kline["open"] == 50000.5
        assert kline["high"] == 50100.0
        assert kline["low"] == 49900.0
        assert kline["close"] == 50050.0
        assert kline["volume"] == 100.5
        assert isinstance(kline["timestamp"], datetime)
    
    @pytest.mark.asyncio
    async def test_data_availability_check(self, backfilling_service, mock_parquet_storage):
        """Test data availability check logic."""
        # Mock: first date has data, second doesn't
        async def mock_read_klines(symbol, date_str):
            if date_str == "2025-01-01":
                return pd.DataFrame({"timestamp": [datetime.now(timezone.utc)]})
            else:
                raise FileNotFoundError()
        
        mock_parquet_storage.read_klines = AsyncMock(side_effect=mock_read_klines)
        
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 2)
        
        missing_data = await backfilling_service._check_data_availability("BTCUSDT", start_date, end_date, ["klines"])
        
        # Only 2025-01-02 should be missing
        assert date(2025, 1, 2) in missing_data
        assert date(2025, 1, 1) not in missing_data
    
    @pytest.mark.asyncio
    async def test_data_validation_passes(self, backfilling_service, mock_parquet_storage):
        """Test data validation passes for correct data."""
        # Mock reading back correct data
        read_data = pd.DataFrame({
            "timestamp": pd.to_datetime([datetime.now(timezone.utc)]),
            "open": [50000.0],
            "high": [50100.0],
            "low": [49900.0],
            "close": [50050.0],
            "volume": [100.5],
            "symbol": ["BTCUSDT"],
        })
        
        mock_parquet_storage.read_klines = AsyncMock(return_value=read_data)
        
        validation_passed = await backfilling_service._validate_saved_data(
            "BTCUSDT",
            "2025-01-01",
            expected_count=1,
            data_type="klines",
        )
        
        assert validation_passed is True
    
    @pytest.mark.asyncio
    async def test_data_validation_fails_count_mismatch(self, backfilling_service, mock_parquet_storage):
        """Test data validation fails for count mismatch."""
        # Mock reading back wrong count
        read_data = pd.DataFrame({
            "timestamp": pd.to_datetime([datetime.now(timezone.utc)]),
            "open": [50000.0],
            "high": [50100.0],
            "low": [49900.0],
            "close": [50050.0],
            "volume": [100.5],
            "symbol": ["BTCUSDT"],
        })
        
        mock_parquet_storage.read_klines = AsyncMock(return_value=read_data)
        
        validation_passed = await backfilling_service._validate_saved_data(
            "BTCUSDT",
            "2025-01-01",
            expected_count=5,  # Expected 5, got 1
            data_type="klines",
        )
        
        assert validation_passed is False
    
    @pytest.mark.asyncio
    async def test_data_validation_fails_missing_fields(self, backfilling_service, mock_parquet_storage):
        """Test data validation fails for missing fields."""
        # Mock reading back data with missing fields
        read_data = pd.DataFrame({
            "timestamp": pd.to_datetime([datetime.now(timezone.utc)]),
            "open": [50000.0],
            # Missing: high, low, close, volume, symbol
        })
        
        mock_parquet_storage.read_klines = AsyncMock(return_value=read_data)
        
        validation_passed = await backfilling_service._validate_saved_data(
            "BTCUSDT",
            "2025-01-01",
            expected_count=1,
            data_type="klines",
        )
        
        assert validation_passed is False
    
    @pytest.mark.asyncio
    async def test_backfill_historical_with_registry(self, backfilling_service, mock_bybit_client, mock_parquet_storage):
        """Test backfill_historical uses Feature Registry to determine data types."""
        mock_bybit_client.get = AsyncMock(return_value={
            "retCode": 0,
            "retMsg": "OK",
            "result": {"list": []},
        })
        
        mock_parquet_storage.read_klines = AsyncMock(side_effect=FileNotFoundError())
        
        job_id = await backfilling_service.backfill_historical(
            "BTCUSDT",
            date(2025, 1, 1),
            date(2025, 1, 2),
            data_types=None,  # Should use Feature Registry
        )
        
        assert job_id is not None
        # Verify Feature Registry was used
        backfilling_service._feature_registry_loader.get_required_data_types.assert_called()
    
    @pytest.mark.asyncio
    async def test_backfill_historical_error_handling(self, backfilling_service, mock_bybit_client):
        """Test error handling and retry logic."""
        # Mock API error
        mock_bybit_client.get = AsyncMock(side_effect=BybitAPIError("API error"))
        
        with pytest.raises(BybitAPIError):
            await backfilling_service.backfill_klines(
                "BTCUSDT",
                date(2025, 1, 1),
                date(2025, 1, 1),
                interval=1,
            )
    
    @pytest.mark.asyncio
    async def test_job_tracking(self, backfilling_service, mock_bybit_client, mock_parquet_storage):
        """Test backfilling job tracking."""
        mock_bybit_client.get = AsyncMock(return_value={
            "retCode": 0,
            "retMsg": "OK",
            "result": {"list": []},
        })
        
        mock_parquet_storage.read_klines = AsyncMock(side_effect=FileNotFoundError())
        
        job_id = await backfilling_service.backfill_historical(
            "BTCUSDT",
            date(2025, 1, 1),
            date(2025, 1, 2),
        )
        
        # Get job status
        status = backfilling_service.get_job_status(job_id)
        
        assert status is not None
        assert status["job_id"] == job_id
        assert status["symbol"] == "BTCUSDT"
        assert "status" in status
        assert "progress" in status
    
    @pytest.mark.asyncio
    async def test_backfill_trades(self, backfilling_service, mock_bybit_client):
        """Test trades backfilling."""
        timestamp_ms = int(datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        mock_response = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "execId": "test-exec-id-1",
                        "symbol": "BTCUSDT",
                        "price": "50000.5",
                        "size": "0.1",
                        "side": "Buy",
                        "time": str(timestamp_ms),
                        "isBlockTrade": False,
                    },
                    {
                        "execId": "test-exec-id-2",
                        "symbol": "BTCUSDT",
                        "price": "50001.0",
                        "size": "0.2",
                        "side": "Sell",
                        "time": str(timestamp_ms + 1000),
                        "isBlockTrade": False,
                    },
                ],
            },
        }
        
        mock_bybit_client.get = AsyncMock(return_value=mock_response)
        
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 1)
        
        trades = await backfilling_service.backfill_trades("BTCUSDT", start_date, end_date)
        
        assert len(trades) == 2
        assert trades[0]["symbol"] == "BTCUSDT"
        assert trades[0]["price"] == 50000.5
        assert trades[0]["quantity"] == 0.1
        assert trades[0]["side"] == "Buy"
        assert isinstance(trades[0]["timestamp"], datetime)
    
    @pytest.mark.asyncio
    async def test_backfill_funding(self, backfilling_service, mock_bybit_client):
        """Test funding rate backfilling."""
        timestamp_ms = int(datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        mock_response = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "fundingRate": "0.0001",
                        "fundingRateTimestamp": str(timestamp_ms),
                    },
                    {
                        "symbol": "BTCUSDT",
                        "fundingRate": "0.0002",
                        "fundingRateTimestamp": str(timestamp_ms + 8 * 60 * 60 * 1000),  # 8 hours later
                    },
                ],
            },
        }
        
        mock_response_empty = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {"list": []},
        }
        
        mock_bybit_client.get = AsyncMock(side_effect=[mock_response, mock_response_empty])
        
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 2)
        
        funding = await backfilling_service.backfill_funding("BTCUSDT", start_date, end_date)
        
        assert len(funding) == 2
        assert funding[0]["symbol"] == "BTCUSDT"
        assert funding[0]["funding_rate"] == 0.0001
        assert isinstance(funding[0]["timestamp"], datetime)
    
    @pytest.mark.asyncio
    async def test_backfill_orderbook_snapshots_not_available(self, backfilling_service):
        """Test that orderbook snapshots return empty list (not available via REST API)."""
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 2)
        
        snapshots = await backfilling_service.backfill_orderbook_snapshots("BTCUSDT", start_date, end_date)
        
        assert snapshots == []
    
    @pytest.mark.asyncio
    async def test_backfill_orderbook_deltas_not_available(self, backfilling_service):
        """Test that orderbook deltas return empty list (not available via REST API)."""
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 2)
        
        deltas = await backfilling_service.backfill_orderbook_deltas("BTCUSDT", start_date, end_date)
        
        assert deltas == []
    
    @pytest.mark.asyncio
    async def test_backfill_ticker(self, backfilling_service, mock_bybit_client):
        """Test ticker backfilling (returns current ticker only)."""
        timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        mock_response = {
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "list": [
                    {
                        "symbol": "BTCUSDT",
                        "lastPrice": "50000.5",
                        "bid1Price": "49999.0",
                        "ask1Price": "50001.0",
                        "volume24h": "1000.5",
                        "time": str(timestamp_ms),
                    },
                ],
            },
        }
        
        mock_bybit_client.get = AsyncMock(return_value=mock_response)
        
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 2)
        
        tickers = await backfilling_service.backfill_ticker("BTCUSDT", start_date, end_date)
        
        assert len(tickers) == 1
        assert tickers[0]["symbol"] == "BTCUSDT"
        assert tickers[0]["last_price"] == 50000.5
        assert tickers[0]["bid_price"] == 49999.0
        assert tickers[0]["ask_price"] == 50001.0
        assert tickers[0]["volume_24h"] == 1000.5
        assert isinstance(tickers[0]["timestamp"], datetime)
    
    @pytest.mark.asyncio
    async def test_data_availability_check_all_types(self, backfilling_service, mock_parquet_storage):
        """Test data availability check for all data types."""
        async def mock_read_klines(symbol, date_str):
            if date_str == "2025-01-01":
                return pd.DataFrame({"timestamp": [datetime.now(timezone.utc)]})
            else:
                raise FileNotFoundError()
        
        async def mock_read_trades(symbol, date_str):
            if date_str == "2025-01-01":
                return pd.DataFrame({"timestamp": [datetime.now(timezone.utc)]})
            else:
                raise FileNotFoundError()
        
        async def mock_read_orderbook_snapshots(symbol, date_str):
            if date_str == "2025-01-01":
                return pd.DataFrame({"timestamp": [datetime.now(timezone.utc)]})
            else:
                raise FileNotFoundError()
        
        async def mock_read_orderbook_deltas(symbol, date_str):
            if date_str == "2025-01-01":
                return pd.DataFrame({"timestamp": [datetime.now(timezone.utc)]})
            else:
                raise FileNotFoundError()
        
        async def mock_read_ticker(symbol, date_str):
            if date_str == "2025-01-01":
                return pd.DataFrame({"timestamp": [datetime.now(timezone.utc)]})
            else:
                raise FileNotFoundError()
        
        async def mock_read_funding(symbol, date_str):
            if date_str == "2025-01-01":
                return pd.DataFrame({"timestamp": [datetime.now(timezone.utc)]})
            else:
                raise FileNotFoundError()
        
        mock_parquet_storage.read_klines = AsyncMock(side_effect=mock_read_klines)
        mock_parquet_storage.read_trades = AsyncMock(side_effect=mock_read_trades)
        mock_parquet_storage.read_orderbook_snapshots = AsyncMock(side_effect=mock_read_orderbook_snapshots)
        mock_parquet_storage.read_orderbook_deltas = AsyncMock(side_effect=mock_read_orderbook_deltas)
        mock_parquet_storage.read_ticker = AsyncMock(side_effect=mock_read_ticker)
        mock_parquet_storage.read_funding = AsyncMock(side_effect=mock_read_funding)
        
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 2)
        
        data_types = ["klines", "trades", "orderbook_snapshots", "orderbook_deltas", "ticker", "funding"]
        missing_data = await backfilling_service._check_data_availability("BTCUSDT", start_date, end_date, data_types)
        
        # Only 2025-01-02 should be missing for all types
        assert date(2025, 1, 2) in missing_data
        assert len(missing_data[date(2025, 1, 2)]) == len(data_types)
        assert date(2025, 1, 1) not in missing_data
    
    @pytest.mark.asyncio
    async def test_validate_saved_data_trades(self, backfilling_service, mock_parquet_storage):
        """Test data validation for trades."""
        read_data = pd.DataFrame({
            "timestamp": pd.to_datetime([datetime.now(timezone.utc)]),
            "price": [50000.0],
            "quantity": [0.1],
            "side": ["Buy"],
            "symbol": ["BTCUSDT"],
        })
        
        mock_parquet_storage.read_trades = AsyncMock(return_value=read_data)
        
        validation_passed = await backfilling_service._validate_saved_data(
            "BTCUSDT",
            "2025-01-01",
            expected_count=1,
            data_type="trades",
        )
        
        assert validation_passed is True
    
    @pytest.mark.asyncio
    async def test_validate_saved_data_funding(self, backfilling_service, mock_parquet_storage):
        """Test data validation for funding rates."""
        read_data = pd.DataFrame({
            "timestamp": pd.to_datetime([datetime.now(timezone.utc)]),
            "funding_rate": [0.0001],
            "symbol": ["BTCUSDT"],
        })
        
        mock_parquet_storage.read_funding = AsyncMock(return_value=read_data)
        
        validation_passed = await backfilling_service._validate_saved_data(
            "BTCUSDT",
            "2025-01-01",
            expected_count=1,
            data_type="funding",
        )
        
        assert validation_passed is True

