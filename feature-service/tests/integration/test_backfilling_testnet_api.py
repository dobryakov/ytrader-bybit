"""
Integration test for backfilling using real Bybit testnet API.

This test verifies that:
1. Backfilling service can fetch data from Bybit testnet API
2. Data is correctly saved to Parquet storage
3. Saved data has correct structure and values (non-zero prices, correct types, etc.)
"""
import pytest
import pytest_asyncio
import pandas as pd
from datetime import date, datetime, timezone
import tempfile
import shutil
from pathlib import Path

from src.services.backfilling_service import BackfillingService
from src.storage.parquet_storage import ParquetStorage
from src.services.feature_registry import FeatureRegistryLoader
from src.utils.bybit_client import BybitClient
from src.config import Config


@pytest.mark.integration
class TestBackfillingTestnetAPI:
    """Integration tests for backfilling with real Bybit testnet API."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def parquet_storage(self, temp_data_dir):
        """Create Parquet storage for testing."""
        return ParquetStorage(base_path=temp_data_dir)
    
    @pytest_asyncio.fixture
    async def testnet_bybit_client(self):
        """Create Bybit client configured for testnet."""
        # Use testnet API (no authentication required for public endpoints)
        client = BybitClient(
            api_key=None,
            api_secret=None,
            base_url="https://api-testnet.bybit.com",
            rate_limit_delay_ms=100,  # Respect rate limits
        )
        yield client
        # Cleanup
        try:
            await client.close()
        except Exception:
            pass
    
    @pytest.fixture
    def feature_registry_loader(self):
        """Create Feature Registry loader for testing."""
        from src.config import config
        loader = FeatureRegistryLoader(config_path=config.feature_registry_path)
        # Load config to initialize required data types
        loader.load()
        return loader
    
    @pytest.fixture
    def backfilling_service(self, parquet_storage, feature_registry_loader, testnet_bybit_client):
        """Create backfilling service for testing."""
        return BackfillingService(
            parquet_storage=parquet_storage,
            feature_registry_loader=feature_registry_loader,
            bybit_client=testnet_bybit_client,
        )
    
    @pytest.mark.asyncio
    async def test_backfill_klines_from_testnet_api(
        self,
        backfilling_service,
        parquet_storage,
        testnet_bybit_client,
    ):
        """
        Test backfilling klines from real Bybit testnet API.
        
        This test:
        1. Fetches klines for January 1, 2025 from testnet API
        2. Verifies data is received
        3. Saves data to Parquet storage
        4. Verifies data is correctly written
        5. Validates data structure and values
        """
        symbol = "BTCUSDT"
        test_date = date(2025, 1, 1)
        date_str = test_date.strftime("%Y-%m-%d")
        
        # Step 1: Fetch klines from testnet API
        klines = await backfilling_service.backfill_klines(
            symbol=symbol,
            start_date=test_date,
            end_date=test_date,
            interval=1,  # 1-minute intervals
        )
        
        # Step 2: Verify data is received
        assert len(klines) > 0, f"Expected to receive klines for {date_str}, got 0"
        
        # Verify kline structure
        first_kline = klines[0]
        required_fields = ["timestamp", "symbol", "interval", "open", "high", "low", "close", "volume"]
        for field in required_fields:
            assert field in first_kline, f"Missing required field: {field}"
        
        # Verify timestamp is datetime
        assert isinstance(first_kline["timestamp"], datetime), "Timestamp should be datetime object"
        assert first_kline["timestamp"].tzinfo == timezone.utc, "Timestamp should be UTC"
        
        # Verify prices are positive
        assert first_kline["open"] > 0, "Open price should be positive"
        assert first_kline["high"] > 0, "High price should be positive"
        assert first_kline["low"] > 0, "Low price should be positive"
        assert first_kline["close"] > 0, "Close price should be positive"
        assert first_kline["volume"] >= 0, "Volume should be non-negative"
        
        # Verify high >= low and high >= close >= low
        assert first_kline["high"] >= first_kline["low"], "High should be >= Low"
        assert first_kline["high"] >= first_kline["close"] >= first_kline["low"], "Close should be between Low and High"
        
        # Verify symbol matches
        assert first_kline["symbol"] == symbol, f"Symbol should be {symbol}"
        
        # Step 3: Save to Parquet storage
        await backfilling_service._save_klines(symbol, date_str, klines)
        
        # Step 4: Verify file exists
        file_path = parquet_storage._get_klines_path(symbol, date_str)
        assert file_path.exists(), f"Parquet file should exist at {file_path}"
        assert file_path.stat().st_size > 0, "Parquet file should not be empty"
        
        # Step 5: Read back and validate data
        read_data = await parquet_storage.read_klines(symbol, date_str)
        
        assert not read_data.empty, "Read data should not be empty"
        assert len(read_data) == len(klines), f"Read data count ({len(read_data)}) should match saved count ({len(klines)})"
        
        # Verify DataFrame structure
        required_columns = ["timestamp", "symbol", "interval", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            assert col in read_data.columns, f"Missing required column: {col}"
        
        # Verify data types
        assert pd.api.types.is_datetime64_any_dtype(read_data["timestamp"]), "Timestamp should be datetime type"
        assert pd.api.types.is_numeric_dtype(read_data["open"]), "Open should be numeric"
        assert pd.api.types.is_numeric_dtype(read_data["close"]), "Close should be numeric"
        assert pd.api.types.is_numeric_dtype(read_data["volume"]), "Volume should be numeric"
        
        # Verify no zero prices
        zero_prices = (read_data["close"] == 0).sum()
        assert zero_prices == 0, f"Found {zero_prices} records with zero close prices"
        
        # Verify price relationships
        invalid_high_low = (read_data["high"] < read_data["low"]).sum()
        assert invalid_high_low == 0, f"Found {invalid_high_low} records where high < low"
        
        invalid_close_range = (
            (read_data["close"] < read_data["low"]) | (read_data["close"] > read_data["high"])
        ).sum()
        assert invalid_close_range == 0, f"Found {invalid_close_range} records where close is outside [low, high]"
        
        # Verify timestamps are in correct date range
        read_data["date"] = read_data["timestamp"].dt.date
        unique_dates = read_data["date"].unique()
        assert test_date in unique_dates, f"Data should contain records for {test_date}"
        
        # Verify timestamps are sorted
        assert read_data["timestamp"].is_monotonic_increasing, "Timestamps should be sorted in ascending order"
        
        # Verify no duplicate timestamps
        duplicate_timestamps = read_data["timestamp"].duplicated().sum()
        assert duplicate_timestamps == 0, f"Found {duplicate_timestamps} duplicate timestamps"
        
        # Step 6: Validate saved data using backfilling service validation
        validation_passed = await backfilling_service._validate_saved_data(
            symbol=symbol,
            date_str=date_str,
            expected_count=len(klines),
            data_type="klines",
        )
        assert validation_passed is True, "Data validation should pass"
        
        # Step 7: Verify data format matches expected schema (with optional fields)
        optional_fields = ["internal_timestamp", "exchange_timestamp"]
        for field in optional_fields:
            if field in read_data.columns:
                # If present and has non-null values, should be datetime
                # For backfilling, these fields may be None (object dtype), which is acceptable
                non_null_values = read_data[field].dropna()
                if len(non_null_values) > 0:
                    # If there are non-null values, they should be datetime type
                    assert pd.api.types.is_datetime64_any_dtype(read_data[field]) or \
                           all(isinstance(v, (datetime, pd.Timestamp)) for v in non_null_values.head(10)), \
                           f"{field} should be datetime type when non-null"
    
    @pytest.mark.asyncio
    async def test_backfill_historical_job_from_testnet_api(
        self,
        backfilling_service,
        parquet_storage,
    ):
        """
        Test backfilling historical job from real Bybit testnet API.
        
        This test verifies the full backfilling job workflow:
        1. Creates a backfilling job for a specific date
        2. Waits for job completion
        3. Verifies data is saved correctly
        """
        symbol = "BTCUSDT"
        test_date = date(2025, 1, 1)
        date_str = test_date.strftime("%Y-%m-%d")
        
        # Step 1: Create backfilling job
        job_id = await backfilling_service.backfill_historical(
            symbol=symbol,
            start_date=test_date,
            end_date=test_date,
            data_types=["klines"],
        )
        
        assert job_id is not None, "Job ID should not be None"
        
        # Step 2: Wait for job completion (with timeout)
        import asyncio
        max_wait_seconds = 60
        waited = 0
        while waited < max_wait_seconds:
            status = backfilling_service.get_job_status(job_id)
            if status:
                job_status = status.get("status")
                if job_status in ["completed", "failed"]:
                    break
            await asyncio.sleep(1)
            waited += 1
        
        # Step 3: Verify job status
        final_status = backfilling_service.get_job_status(job_id)
        assert final_status is not None, "Job status should be available"
        
        job_status = final_status.get("status")
        assert job_status == "completed", f"Job should be completed, got {job_status}"
        
        # Step 4: Verify completed dates
        completed_dates = final_status.get("completed_dates", [])
        assert len(completed_dates) > 0, "Should have at least one completed date"
        assert test_date in completed_dates or date_str in completed_dates, f"Date {test_date} should be in completed dates"
        
        # Step 5: Verify data is saved
        file_path = parquet_storage._get_klines_path(symbol, date_str)
        assert file_path.exists(), f"Parquet file should exist at {file_path}"
        
        # Step 6: Read and validate data
        read_data = await parquet_storage.read_klines(symbol, date_str)
        assert not read_data.empty, "Read data should not be empty"
        assert len(read_data) > 0, "Should have at least one record"
        
        # Verify data quality
        assert (read_data["close"] > 0).all(), "All close prices should be positive"
        assert (read_data["high"] >= read_data["low"]).all(), "All high prices should be >= low prices"

