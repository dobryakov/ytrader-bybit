"""
Cache invalidation service for managing cache invalidation triggers and logic.
"""
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
import structlog

from src.services.cache_service import CacheService
from src.config import config

logger = structlog.get_logger(__name__)


class CacheInvalidationService:
    """Service for managing cache invalidation."""
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        """
        Initialize cache invalidation service.
        
        Args:
            cache_service: Cache service instance (optional, will use from config if not provided)
        """
        self._cache_service = cache_service
        self._invalidation_on_registry_change = config.cache_invalidation_on_registry_change
        self._invalidation_on_data_change = config.cache_invalidation_on_data_change
    
    def set_cache_service(self, cache_service: CacheService) -> None:
        """Set cache service instance."""
        self._cache_service = cache_service
    
    async def invalidate_on_registry_version_change(
        self,
        old_version: str,
        new_version: str,
    ) -> int:
        """
        Invalidate all cache entries with old registry version.
        
        Args:
            old_version: Old Feature Registry version
            new_version: New Feature Registry version
            
        Returns:
            Number of cache entries invalidated
        """
        if not self._cache_service:
            logger.warning("cache_service_not_available_for_invalidation")
            return 0
        
        if not self._invalidation_on_registry_change:
            logger.debug("cache_invalidation_on_registry_change_disabled")
            return 0
        
        # Invalidate all entries with old registry version
        # Pattern: "*:{old_version}:*"
        pattern_historical = f"historical_data:*:{old_version}:*"
        pattern_features = f"features:*:{old_version}:*"
        
        invalidated_count = 0
        
        try:
            # Clear historical data cache entries
            count1 = await self._cache_service.clear(pattern_historical)
            if count1 > 0:
                invalidated_count += count1
                logger.info(
                    "cache_invalidated_historical_data",
                    old_version=old_version,
                    new_version=new_version,
                    count=count1,
                )
            
            # Clear features cache entries
            count2 = await self._cache_service.clear(pattern_features)
            if count2 > 0:
                invalidated_count += count2
                logger.info(
                    "cache_invalidated_features",
                    old_version=old_version,
                    new_version=new_version,
                    count=count2,
                )
        except Exception as e:
            logger.error(
                "cache_invalidation_error",
                error=str(e),
                old_version=old_version,
                new_version=new_version,
            )
        
        return invalidated_count
    
    async def invalidate_on_parquet_file_modification(
        self,
        symbol: str,
        file_path: Path,
        old_mtime: Optional[float] = None,
    ) -> int:
        """
        Invalidate cache entries for affected symbol and date range when Parquet file is modified.
        
        Args:
            symbol: Symbol for which data was modified
            file_path: Path to modified Parquet file
            old_mtime: Previous modification time (optional)
            
        Returns:
            Number of cache entries invalidated
        """
        if not self._cache_service:
            logger.warning("cache_service_not_available_for_invalidation")
            return 0
        
        if not self._invalidation_on_data_change:
            logger.debug("cache_invalidation_on_data_change_disabled")
            return 0
        
        # Check if file was actually modified
        if old_mtime is not None:
            try:
                current_mtime = file_path.stat().st_mtime
                if current_mtime == old_mtime:
                    # File not modified
                    return 0
            except Exception:
                # File might not exist, invalidate anyway
                pass
        
        # Invalidate all cache entries for this symbol
        # Pattern: "historical_data:{symbol}:*" and "features:{symbol}:*"
        pattern_historical = f"historical_data:{symbol}:*"
        pattern_features = f"features:{symbol}:*"
        
        invalidated_count = 0
        
        try:
            # Clear historical data cache entries
            count1 = await self._cache_service.clear(pattern_historical)
            if count1 > 0:
                invalidated_count += count1
                logger.info(
                    "cache_invalidated_historical_data_file_modification",
                    symbol=symbol,
                    file_path=str(file_path),
                    count=count1,
                )
            
            # Clear features cache entries (features depend on historical data)
            count2 = await self._cache_service.clear(pattern_features)
            if count2 > 0:
                invalidated_count += count2
                logger.info(
                    "cache_invalidated_features_file_modification",
                    symbol=symbol,
                    file_path=str(file_path),
                    count=count2,
                )
        except Exception as e:
            logger.error(
                "cache_invalidation_error",
                error=str(e),
                symbol=symbol,
                file_path=str(file_path),
            )
        
        return invalidated_count
    
    async def invalidate_on_data_hash_change(
        self,
        symbol: str,
        old_data_hash: str,
        new_data_hash: str,
    ) -> int:
        """
        Invalidate cache entries when data hash changes (indicates data was updated).
        
        Args:
            symbol: Symbol for which data was modified
            old_data_hash: Previous data hash
            new_data_hash: New data hash
            
        Returns:
            Number of cache entries invalidated
        """
        if not self._cache_service:
            logger.warning("cache_service_not_available_for_invalidation")
            return 0
        
        if not self._invalidation_on_data_change:
            logger.debug("cache_invalidation_on_data_change_disabled")
            return 0
        
        if old_data_hash == new_data_hash:
            # Data hash unchanged, no invalidation needed
            return 0
        
        # Invalidate all cache entries for this symbol
        # Note: Cache keys include data_hash, so entries with old hash won't match new requests
        # But we should still clear them to free up space
        pattern_historical = f"historical_data:{symbol}:*:{old_data_hash}"
        pattern_features = f"features:{symbol}:*:*:{old_data_hash}"
        
        invalidated_count = 0
        
        try:
            # Clear historical data cache entries with old hash
            count1 = await self._cache_service.clear(pattern_historical)
            if count1 > 0:
                invalidated_count += count1
                logger.info(
                    "cache_invalidated_historical_data_hash_change",
                    symbol=symbol,
                    old_hash=old_data_hash[:8],
                    new_hash=new_data_hash[:8],
                    count=count1,
                )
            
            # Clear features cache entries with old hash
            count2 = await self._cache_service.clear(pattern_features)
            if count2 > 0:
                invalidated_count += count2
                logger.info(
                    "cache_invalidated_features_hash_change",
                    symbol=symbol,
                    old_hash=old_data_hash[:8],
                    new_hash=new_data_hash[:8],
                    count=count2,
                )
        except Exception as e:
            logger.error(
                "cache_invalidation_error",
                error=str(e),
                symbol=symbol,
                old_hash=old_data_hash[:8],
                new_hash=new_data_hash[:8],
            )
        
        return invalidated_count
    
    async def invalidate_on_backfill_completion(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> int:
        """
        Invalidate cache for backfilled period and symbol.
        
        Args:
            symbol: Symbol for which data was backfilled
            start_date: Start date of backfilled period
            end_date: End date of backfilled period
            
        Returns:
            Number of cache entries invalidated
        """
        if not self._cache_service:
            logger.warning("cache_service_not_available_for_invalidation")
            return 0
        
        if not self._invalidation_on_data_change:
            logger.debug("cache_invalidation_on_data_change_disabled")
            return 0
        
        # Invalidate cache entries for this symbol and date range
        # Pattern matching for date ranges is complex, so we'll invalidate all entries for symbol
        pattern_historical = f"historical_data:{symbol}:*"
        pattern_features = f"features:{symbol}:*"
        
        invalidated_count = 0
        
        try:
            # Clear historical data cache entries
            count1 = await self._cache_service.clear(pattern_historical)
            if count1 > 0:
                invalidated_count += count1
                logger.info(
                    "cache_invalidated_backfill_historical_data",
                    symbol=symbol,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    count=count1,
                )
            
            # Clear features cache entries
            count2 = await self._cache_service.clear(pattern_features)
            if count2 > 0:
                invalidated_count += count2
                logger.info(
                    "cache_invalidated_backfill_features",
                    symbol=symbol,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    count=count2,
                )
        except Exception as e:
            logger.error(
                "cache_invalidation_error",
                error=str(e),
                symbol=symbol,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
        
        return invalidated_count
    
    async def invalidate_manual(
        self,
        pattern: Optional[str] = None,
        symbol: Optional[str] = None,
        date_range: Optional[tuple[datetime, datetime]] = None,
    ) -> int:
        """
        Manually invalidate cache entries.
        
        Args:
            pattern: Cache key pattern to match (e.g., "historical_data:*" or "features:*")
            symbol: Symbol to invalidate (if provided, invalidates all entries for this symbol)
            date_range: Date range to invalidate (if provided, invalidates entries in this range)
            
        Returns:
            Number of cache entries invalidated
        """
        if not self._cache_service:
            logger.warning("cache_service_not_available_for_invalidation")
            return 0
        
        invalidated_count = 0
        
        try:
            if pattern:
                # Use provided pattern
                count = await self._cache_service.clear(pattern)
                invalidated_count += count
                logger.info(
                    "cache_invalidated_manual_pattern",
                    pattern=pattern,
                    count=count,
                )
            elif symbol:
                # Invalidate all entries for symbol
                pattern_historical = f"historical_data:{symbol}:*"
                pattern_features = f"features:{symbol}:*"
                
                count1 = await self._cache_service.clear(pattern_historical)
                count2 = await self._cache_service.clear(pattern_features)
                invalidated_count += count1 + count2
                
                logger.info(
                    "cache_invalidated_manual_symbol",
                    symbol=symbol,
                    count=count1 + count2,
                )
            elif date_range:
                # Invalidate entries in date range
                # This is complex with pattern matching, so we'll invalidate all
                # More precise invalidation would require scanning all keys
                count = await self._cache_service.clear("*")
                invalidated_count += count
                logger.warning(
                    "cache_invalidated_manual_date_range_all",
                    start_date=date_range[0].isoformat(),
                    end_date=date_range[1].isoformat(),
                    count=count,
                    note="Date range invalidation clears all cache entries (pattern matching limitation)",
                )
            else:
                # Invalidate all
                count = await self._cache_service.clear()
                invalidated_count += count
                logger.info(
                    "cache_invalidated_manual_all",
                    count=count,
                )
        except Exception as e:
            logger.error(
                "cache_invalidation_error",
                error=str(e),
                pattern=pattern,
                symbol=symbol,
                date_range=date_range,
            )
        
        return invalidated_count

