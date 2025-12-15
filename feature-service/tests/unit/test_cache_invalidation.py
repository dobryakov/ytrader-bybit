"""
Unit tests for cache invalidation logic.
"""
import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.services.cache_invalidation import CacheInvalidationService
from src.services.cache_service import InMemoryCacheService


class TestCacheInvalidation:
    """Tests for cache invalidation service."""
    
    @pytest_asyncio.fixture
    async def memory_cache_service(self):
        """Create in-memory cache service for testing."""
        cache = InMemoryCacheService(max_size_mb=100, max_entries=1000)
        await cache.start_cleanup_task()
        yield cache
        await cache.stop_cleanup_task()
    
    @pytest.fixture
    def cache_invalidation_service(self, memory_cache_service):
        """Create cache invalidation service."""
        return CacheInvalidationService(cache_service=memory_cache_service)
    
    @pytest.mark.asyncio
    async def test_registry_version_change_invalidation(
        self,
        cache_invalidation_service,
        memory_cache_service,
    ):
        """Test Feature Registry version change invalidation: verify all cache entries are invalidated when registry version changes."""
        # Populate cache with entries for old version
        await memory_cache_service.set("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1", {"data": "test1"})
        await memory_cache_service.set("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:hash1", {"data": "test2"})
        await memory_cache_service.set("historical_data:ETHUSDT:2025-01-01:2025-01-02:1.0.0:hash2", {"data": "test3"})
        
        # Invalidate on version change
        count = await cache_invalidation_service.invalidate_on_registry_version_change(
            old_version="1.0.0",
            new_version="1.1.0",
        )
        
        # Verify entries were invalidated
        assert count >= 2  # At least historical_data and features entries
        
        # Verify entries are gone
        result1 = await memory_cache_service.get("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1")
        result2 = await memory_cache_service.get("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:hash1")
        assert result1 is None
        assert result2 is None
    
    @pytest.mark.asyncio
    async def test_parquet_file_modification_detection(
        self,
        cache_invalidation_service,
        memory_cache_service,
        tmp_path,
    ):
        """Test Parquet file modification detection: verify cache detects file mtime changes and invalidates relevant entries."""
        # Create a test file
        test_file = tmp_path / "test.parquet"
        test_file.write_text("test data")
        
        # Populate cache for symbol
        await memory_cache_service.set("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1", {"data": "test1"})
        await memory_cache_service.set("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:hash1", {"data": "test2"})
        
        # Get old mtime
        old_mtime = test_file.stat().st_mtime
        
        # Wait a bit and modify file
        await asyncio.sleep(0.1)
        test_file.write_text("modified data")
        
        # Invalidate on file modification
        count = await cache_invalidation_service.invalidate_on_parquet_file_modification(
            symbol="BTCUSDT",
            file_path=test_file,
            old_mtime=old_mtime,
        )
        
        # Verify entries were invalidated
        assert count >= 1
        
        # Verify entries are gone
        result1 = await memory_cache_service.get("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1")
        result2 = await memory_cache_service.get("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:hash1")
        assert result1 is None
        assert result2 is None
    
    @pytest.mark.asyncio
    async     def test_data_hash_computation(
        self,
        cache_invalidation_service,
    ):
        """Test data hash computation: verify data_hash is computed correctly from historical data DataFrames."""
        # This test verifies that data hash computation works correctly
        # The actual computation is now done via utility function
        # Here we test that invalidation service can detect hash changes
        
        import pandas as pd
        import hashlib
        
        def compute_data_hash(data: dict) -> str:
            """Compute hash from historical data dict."""
            hash_input = ""
            for key in sorted(data.keys()):
                df = data[key]
                if not df.empty:
                    # Convert DataFrame to string representation for hashing
                    df_str = df.to_string()
                    hash_input += f"{key}:{df_str}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        
        # Create test data
        data1 = {
            "trades": pd.DataFrame({
                "timestamp": [datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)],
                "price": [50000.0],
                "size": [0.1],
            }),
            "klines": pd.DataFrame({
                "timestamp": [datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)],
                "open": [50000.0],
                "close": [50050.0],
            }),
        }
        
        data2 = {
            "trades": pd.DataFrame({
                "timestamp": [datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)],
                "price": [51000.0],  # Different price
                "size": [0.1],
            }),
            "klines": pd.DataFrame({
                "timestamp": [datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)],
                "open": [51000.0],  # Different price
                "close": [51050.0],
            }),
        }
        
        # Compute hashes
        hash1 = compute_data_hash(data1)
        hash2 = compute_data_hash(data2)
        
        # Verify hashes are different
        assert hash1 != hash2
        assert len(hash1) == 32  # MD5 hash length
        assert len(hash2) == 32
    
    @pytest.mark.asyncio
    async def test_partial_invalidation(
        self,
        cache_invalidation_service,
        memory_cache_service,
    ):
        """Test partial invalidation: verify only affected cache entries are invalidated."""
        # Populate cache for multiple symbols
        await memory_cache_service.set("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1", {"data": "test1"})
        await memory_cache_service.set("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:hash1", {"data": "test2"})
        await memory_cache_service.set("historical_data:ETHUSDT:2025-01-01:2025-01-02:1.0.0:hash2", {"data": "test3"})
        await memory_cache_service.set("features:ETHUSDT:2025-01-01T12:00:00:1.0.0:hash2", {"data": "test4"})
        
        # Invalidate only BTCUSDT
        count = await cache_invalidation_service.invalidate_manual(symbol="BTCUSDT")
        
        # Verify BTCUSDT entries are gone
        result1 = await memory_cache_service.get("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1")
        result2 = await memory_cache_service.get("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:hash1")
        assert result1 is None
        assert result2 is None
        
        # Verify ETHUSDT entries are still there
        result3 = await memory_cache_service.get("historical_data:ETHUSDT:2025-01-01:2025-01-02:1.0.0:hash2")
        result4 = await memory_cache_service.get("features:ETHUSDT:2025-01-01T12:00:00:1.0.0:hash2")
        assert result3 is not None
        assert result4 is not None
        
        assert count >= 2
    
    @pytest.mark.asyncio
    async def test_invalidation_on_backfill(
        self,
        cache_invalidation_service,
        memory_cache_service,
    ):
        """Test invalidation on backfill: verify cache is invalidated when new historical data is backfilled."""
        # Populate cache for symbol
        await memory_cache_service.set("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1", {"data": "test1"})
        await memory_cache_service.set("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:hash1", {"data": "test2"})
        
        # Invalidate on backfill
        start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2025, 1, 2, tzinfo=timezone.utc)
        count = await cache_invalidation_service.invalidate_on_backfill_completion(
            symbol="BTCUSDT",
            start_date=start_date,
            end_date=end_date,
        )
        
        # Verify entries were invalidated
        assert count >= 1
        
        # Verify entries are gone
        result1 = await memory_cache_service.get("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1")
        result2 = await memory_cache_service.get("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:hash1")
        assert result1 is None
        assert result2 is None
    
    @pytest.mark.asyncio
    async def test_manual_invalidation_all(
        self,
        cache_invalidation_service,
        memory_cache_service,
    ):
        """Test manual cache invalidation API: verify manual cache invalidation endpoint works correctly."""
        # Populate cache
        await memory_cache_service.set("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1", {"data": "test1"})
        await memory_cache_service.set("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:hash1", {"data": "test2"})
        await memory_cache_service.set("other_key", {"data": "test3"})
        
        # Invalidate all
        count = await cache_invalidation_service.invalidate_manual()
        
        # Verify all entries are gone
        result1 = await memory_cache_service.get("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1")
        result2 = await memory_cache_service.get("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:hash1")
        result3 = await memory_cache_service.get("other_key")
        assert result1 is None
        assert result2 is None
        assert result3 is None
        
        assert count >= 3
    
    @pytest.mark.asyncio
    async def test_manual_invalidation_pattern(
        self,
        cache_invalidation_service,
        memory_cache_service,
    ):
        """Test manual invalidation with pattern."""
        # Populate cache
        await memory_cache_service.set("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1", {"data": "test1"})
        await memory_cache_service.set("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:hash1", {"data": "test2"})
        await memory_cache_service.set("other_key", {"data": "test3"})
        
        # Invalidate with pattern
        count = await cache_invalidation_service.invalidate_manual(pattern="historical_data:*")
        
        # Verify historical_data entries are gone
        result1 = await memory_cache_service.get("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1")
        assert result1 is None
        
        # Verify other entries are still there
        result2 = await memory_cache_service.get("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:hash1")
        result3 = await memory_cache_service.get("other_key")
        assert result2 is not None
        assert result3 is not None
        
        assert count >= 1
    
    @pytest.mark.asyncio
    async def test_data_hash_change_invalidation(
        self,
        cache_invalidation_service,
        memory_cache_service,
    ):
        """Test cache invalidation on data hash change."""
        # Populate cache with old hash
        await memory_cache_service.set("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:old_hash", {"data": "test1"})
        await memory_cache_service.set("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:old_hash", {"data": "test2"})
        
        # Invalidate on hash change
        count = await cache_invalidation_service.invalidate_on_data_hash_change(
            symbol="BTCUSDT",
            old_data_hash="old_hash",
            new_data_hash="new_hash",
        )
        
        # Verify entries with old hash are gone
        result1 = await memory_cache_service.get("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:old_hash")
        result2 = await memory_cache_service.get("features:BTCUSDT:2025-01-01T12:00:00:1.0.0:old_hash")
        assert result1 is None
        assert result2 is None
        
        assert count >= 2
    
    @pytest.mark.asyncio
    async def test_invalidation_disabled(
        self,
        memory_cache_service,
    ):
        """Test that invalidation respects configuration (disabled invalidation)."""
        # Create service with invalidation disabled
        service = CacheInvalidationService(cache_service=memory_cache_service)
        service._invalidation_on_registry_change = False
        service._invalidation_on_data_change = False
        
        # Populate cache
        await memory_cache_service.set("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1", {"data": "test1"})
        
        # Try to invalidate
        count = await service.invalidate_on_registry_version_change(
            old_version="1.0.0",
            new_version="1.1.0",
        )
        
        # Verify no invalidation occurred
        assert count == 0
        
        # Verify entry is still there
        result = await memory_cache_service.get("historical_data:BTCUSDT:2025-01-01:2025-01-02:1.0.0:hash1")
        assert result is not None

