"""
Adaptive Prefetcher for optimized dataset building.

Prefetches data ahead of processing based on processing speed to maintain
a buffer of ready-to-use data.
"""
from datetime import datetime, timedelta, timezone, date
from typing import Optional
import structlog

from .daily_cache import OptimizedDailyDataCache

logger = structlog.get_logger(__name__)


class AdaptivePrefetcher:
    """
    Adaptively prefetches data based on processing speed.
    
    Measures processing speed and maintains a buffer of prefetched data
    to avoid waiting for data loading during dataset building.
    """
    
    def __init__(
        self,
        cache: OptimizedDailyDataCache,
        target_buffer_hours: float = 1.0,
        min_buffer_hours: float = 0.5,
        max_buffer_hours: float = 4.0,
    ):
        """
        Initialize adaptive prefetcher.
        
        Args:
            cache: OptimizedDailyDataCache instance
            target_buffer_hours: Target buffer size in hours (default: 1.0)
            min_buffer_hours: Minimum buffer size in hours (default: 0.5)
            max_buffer_hours: Maximum buffer size in hours (default: 4.0)
        """
        self.cache = cache
        self.target_buffer_hours = target_buffer_hours
        self.min_buffer_hours = min_buffer_hours
        self.max_buffer_hours = max_buffer_hours
        
        # Processing speed tracking
        self.processing_speed_hours_per_second: Optional[float] = None
        self.last_measurement_time: Optional[datetime] = None
        self.last_processed_timestamp: Optional[datetime] = None
        
        # Prefetch tracking
        self.prefetched_dates: set = set()
        self.last_prefetch_time: Optional[datetime] = None
        
        logger.info(
            "adaptive_prefetcher_initialized",
            target_buffer_hours=target_buffer_hours,
            min_buffer_hours=min_buffer_hours,
            max_buffer_hours=max_buffer_hours,
        )
    
    def update_processing_speed(
        self, current_timestamp: datetime, processed_records: int = 1
    ) -> None:
        """
        Update processing speed measurement.
        
        Args:
            current_timestamp: Current processing timestamp
            processed_records: Number of records processed since last update
        """
        now = datetime.now(timezone.utc)
        
        if self.last_measurement_time is None:
            # First measurement
            self.last_measurement_time = now
            self.last_processed_timestamp = current_timestamp
            return
        
        # Calculate time elapsed
        time_elapsed_seconds = (now - self.last_measurement_time).total_seconds()
        
        if time_elapsed_seconds <= 0:
            return
        
        # Calculate timestamp progress (in hours)
        if self.last_processed_timestamp:
            timestamp_progress_hours = (
                (current_timestamp - self.last_processed_timestamp).total_seconds() / 3600
            )
        else:
            timestamp_progress_hours = 0.0
        
        # Calculate processing speed (hours per second)
        if timestamp_progress_hours > 0:
            speed = timestamp_progress_hours / time_elapsed_seconds
            
            # Update speed with exponential moving average (smoothing)
            if self.processing_speed_hours_per_second is None:
                self.processing_speed_hours_per_second = speed
            else:
                # EMA with alpha = 0.3 (more weight to recent measurements)
                alpha = 0.3
                self.processing_speed_hours_per_second = (
                    alpha * speed + (1 - alpha) * self.processing_speed_hours_per_second
                )
        
        # Update tracking
        self.last_measurement_time = now
        self.last_processed_timestamp = current_timestamp
        
        logger.debug(
            "processing_speed_updated",
            speed_hours_per_second=self.processing_speed_hours_per_second,
            timestamp_progress_hours=timestamp_progress_hours,
            time_elapsed_seconds=time_elapsed_seconds,
        )
    
    async def prefetch_if_needed(
        self,
        current_timestamp: datetime,
        period_end: datetime,
    ) -> None:
        """
        Prefetch data if buffer is below target.
        
        Args:
            current_timestamp: Current processing timestamp
            period_end: End of processing period
        """
        # Calculate how much data we need to prefetch
        hours_ahead = self._calculate_hours_ahead(current_timestamp, period_end)
        
        if hours_ahead <= 0:
            # Already at or past period end
            return
        
        # Determine which dates to prefetch
        dates_to_prefetch = self._get_dates_to_prefetch(
            current_timestamp, hours_ahead
        )
        
        # Prefetch dates that aren't already prefetched
        for date_obj in dates_to_prefetch:
            if date_obj not in self.prefetched_dates:
                try:
                    await self.cache.prefetch_day(date_obj)
                    self.prefetched_dates.add(date_obj)
                    logger.debug(
                        "prefetch_completed",
                        date=date_obj.isoformat(),
                        symbol=self.cache.symbol,
                    )
                except Exception as e:
                    logger.warning(
                        "prefetch_failed",
                        date=date_obj.isoformat(),
                        symbol=self.cache.symbol,
                        error=str(e),
                    )
        
        self.last_prefetch_time = datetime.now(timezone.utc)
    
    async def prefetch_day(self, date_obj: date) -> None:
        """
        Backwards-compatible wrapper to prefetch a full day.

        Delegates to the underlying cache's prefetch_day method.
        This is used by streaming builders that want simple day-level prefetching.
        """
        try:
            await self.cache.prefetch_day(date_obj)
        except Exception as e:
            logger.warning(
                "prefetch_day_failed",
                date=date_obj.isoformat(),
                symbol=getattr(self.cache, "symbol", None),
                error=str(e),
            )
    
    def _calculate_hours_ahead(
        self, current_timestamp: datetime, period_end: datetime
    ) -> float:
        """
        Calculate how many hours ahead we should prefetch.
        
        Args:
            current_timestamp: Current processing timestamp
            period_end: End of processing period
            
        Returns:
            Hours ahead to prefetch
        """
        # Calculate remaining hours
        remaining_hours = (period_end - current_timestamp).total_seconds() / 3600
        
        if remaining_hours <= 0:
            return 0.0
        
        # If we have processing speed, use it to calculate buffer
        if self.processing_speed_hours_per_second is not None and self.processing_speed_hours_per_second > 0:
            # Calculate how long current buffer will last
            # Buffer should last at least target_buffer_hours of processing time
            # Processing time = buffer_hours / processing_speed
            # We want: buffer_hours / processing_speed >= target_buffer_hours
            # So: buffer_hours >= target_buffer_hours * processing_speed
            
            # Estimate buffer needed based on processing speed
            # If processing speed is high, we need larger buffer
            estimated_buffer_hours = self.target_buffer_hours * (
                1.0 / max(self.processing_speed_hours_per_second, 0.001)
            )
            
            # Clamp to min/max
            buffer_hours = max(
                self.min_buffer_hours,
                min(estimated_buffer_hours, self.max_buffer_hours),
            )
        else:
            # No speed data yet - use target buffer
            buffer_hours = self.target_buffer_hours
        
        # Don't prefetch more than remaining hours
        return min(buffer_hours, remaining_hours)
    
    def _get_dates_to_prefetch(
        self, current_timestamp: datetime, hours_ahead: float
    ) -> list[date]:
        """
        Get list of dates to prefetch based on hours ahead.
        
        Args:
            current_timestamp: Current timestamp
            hours_ahead: Hours ahead to prefetch
            
        Returns:
            List of date objects to prefetch
        """
        dates = []
        current_date = current_timestamp.date()
        
        # Calculate end timestamp
        end_timestamp = current_timestamp + timedelta(hours=hours_ahead)
        end_date = end_timestamp.date()
        
        # Get all dates between current_date and end_date (inclusive)
        date_obj = current_date
        while date_obj <= end_date:
            # Skip current date if we're still processing it
            if date_obj == current_date:
                # Check if we're past midnight
                if current_timestamp.time().hour >= 23:
                    # Prefetch next day
                    dates.append(date_obj + timedelta(days=1))
            else:
                dates.append(date_obj)
            
            date_obj += timedelta(days=1)
        
        return dates
    
    def get_statistics(self) -> dict:
        """
        Get prefetcher statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "processing_speed_hours_per_second": self.processing_speed_hours_per_second,
            "target_buffer_hours": self.target_buffer_hours,
            "prefetched_dates_count": len(self.prefetched_dates),
            "prefetched_dates": sorted([d.isoformat() for d in self.prefetched_dates]),
            "last_prefetch_time": (
                self.last_prefetch_time.isoformat()
                if self.last_prefetch_time
                else None
            ),
        }
    
    def reset(self) -> None:
        """Reset prefetcher state."""
        self.processing_speed_hours_per_second = None
        self.last_measurement_time = None
        self.last_processed_timestamp = None
        self.prefetched_dates.clear()
        self.last_prefetch_time = None
        logger.debug("prefetcher_reset")

