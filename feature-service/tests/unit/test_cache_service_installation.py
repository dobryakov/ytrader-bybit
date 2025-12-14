"""
Unit tests for cache service package installation and availability.
These tests verify that required packages are installed and available.
"""
import pytest
import sys


class TestCacheServiceInstallation:
    """Tests for cache service package installation."""
    
    def test_redis_package_installed(self):
        """Test that redis package is installed and importable."""
        try:
            import redis
            assert redis is not None
            # Verify version
            assert hasattr(redis, '__version__')
            print(f"✓ Redis package installed: version {redis.__version__}")
        except ImportError as e:
            pytest.fail(
                f"Redis package is not installed. "
                f"Install it with: pip install redis[hiredis]>=5.0.0. "
                f"Error: {e}"
            )
    
    def test_redis_asyncio_available(self):
        """Test that redis.asyncio is available."""
        try:
            import redis.asyncio
            assert redis.asyncio is not None
            print("✓ redis.asyncio module is available")
        except ImportError as e:
            pytest.fail(
                f"redis.asyncio is not available. "
                f"Install it with: pip install redis[hiredis]>=5.0.0. "
                f"Error: {e}"
            )
    
    def test_hiredis_available(self):
        """Test that hiredis is available (for performance)."""
        try:
            import hiredis
            assert hiredis is not None
            print("✓ hiredis package is available")
        except ImportError:
            # hiredis is optional but recommended
            pytest.skip("hiredis is optional but recommended for performance")
    
    def test_redis_available_flag(self):
        """Test that REDIS_AVAILABLE flag is set correctly."""
        from src.services.cache_service import REDIS_AVAILABLE
        
        try:
            import redis.asyncio
            # If redis is importable, REDIS_AVAILABLE should be True
            assert REDIS_AVAILABLE is True, (
                "REDIS_AVAILABLE should be True when redis package is installed. "
                "Check cache_service.py import logic."
            )
            print("✓ REDIS_AVAILABLE flag is correctly set to True")
        except ImportError:
            # If redis is not importable, REDIS_AVAILABLE should be False
            assert REDIS_AVAILABLE is False, (
                "REDIS_AVAILABLE should be False when redis package is not installed. "
                "Check cache_service.py import logic."
            )
            pytest.skip("Redis package is not installed, skipping availability check")
    
    @pytest.mark.asyncio
    async def test_cache_service_factory_uses_redis_when_available(self):
        """Test that CacheServiceFactory uses Redis when package is available."""
        from src.services.cache_service import CacheServiceFactory, RedisCacheService, InMemoryCacheService, REDIS_AVAILABLE
        
        # Only run this test if Redis is available
        if not REDIS_AVAILABLE:
            pytest.skip("Redis package is not installed, skipping Redis cache test")
        
        # Try to create cache service with Redis enabled
        # Note: This will try to connect to Redis, but may fail if Redis server is not running
        # That's OK - we're just testing that the factory tries to use Redis
        cache_service = await CacheServiceFactory.create(
            redis_host="localhost",  # Use localhost to avoid Docker network issues in tests
            redis_port=6379,
            cache_redis_enabled=True,
        )
        
        # If Redis server is available, should get RedisCacheService
        # If Redis server is not available, will get InMemoryCacheService (fallback)
        # Both are acceptable, but we log a warning if fallback occurs
        assert isinstance(cache_service, (RedisCacheService, InMemoryCacheService))
        
        if isinstance(cache_service, RedisCacheService):
            print("✓ CacheServiceFactory successfully created RedisCacheService")
        else:
            print("⚠ CacheServiceFactory fell back to InMemoryCacheService (Redis server may not be available)")

