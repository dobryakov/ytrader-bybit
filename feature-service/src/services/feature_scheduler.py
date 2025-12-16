"""
Feature computation scheduler for computing features at intervals.
"""
import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, Set
import structlog

from src.services.feature_computer import FeatureComputer
from src.publishers.feature_publisher import FeaturePublisher

logger = structlog.get_logger(__name__)


class FeatureScheduler:
    """Schedules feature computation at intervals (1s, 3s, 15s, 1m)."""
    
    def __init__(
        self,
        feature_computer: FeatureComputer,
        feature_publisher: FeaturePublisher,
        symbols: Optional[list] = None,
    ):
        """Initialize feature scheduler."""
        self._feature_computer = feature_computer
        self._feature_publisher = feature_publisher
        self._symbols = set(symbols or [])
        self._intervals = {
            "1s": 1,
            "3s": 3,
            "15s": 15,
            "1m": 60,
        }
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False
    
    async def start(self) -> None:
        """Start scheduling feature computations."""
        logger.info("Starting feature scheduler", symbols=list(self._symbols))
        
        self._running = True
        
        # Start scheduler tasks for each interval
        for interval_name, interval_seconds in self._intervals.items():
            task = asyncio.create_task(
                self._schedule_interval(interval_name, interval_seconds)
            )
            self._tasks[interval_name] = task
        
        logger.info("Feature scheduler started")
    
    async def stop(self) -> None:
        """Stop scheduling feature computations."""
        logger.info("Stopping feature scheduler")
        
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        
        self._tasks.clear()
        logger.info("Feature scheduler stopped")
    
    def add_symbol(self, symbol: str) -> None:
        """Add symbol to schedule."""
        self._symbols.add(symbol)
        logger.debug("symbol_added_to_scheduler", symbol=symbol)
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove symbol from schedule."""
        self._symbols.discard(symbol)
        logger.debug("symbol_removed_from_scheduler", symbol=symbol)
    
    async def _schedule_interval(self, interval_name: str, interval_seconds: int) -> None:
        """Schedule feature computation for a specific interval."""
        try:
            while self._running:
                # Compute features for all symbols at this interval
                await self._compute_features_for_interval(interval_name, interval_seconds)
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
        
        except asyncio.CancelledError:
            logger.info("scheduler_interval_cancelled", interval=interval_name)
        except Exception as e:
            logger.error(
                "scheduler_interval_error",
                interval=interval_name,
                error=str(e),
                exc_info=True,
            )
    
    async def _compute_features_for_interval(
        self,
        interval_name: str,
        interval_seconds: int,
    ) -> None:
        """Compute features for all symbols at specified interval."""
        timestamp = datetime.now(timezone.utc)
        
        for symbol in list(self._symbols):  # Copy list to avoid modification during iteration
            try:
                # Compute features
                feature_vector = self._feature_computer.compute_features(
                    symbol=symbol,
                    timestamp=timestamp,
                )
                
                if feature_vector is None:
                    logger.warning(
                        "feature_computation_failed",
                        symbol=symbol,
                        interval=interval_name,
                    )
                    continue
                
                # Publish features
                await self._feature_publisher.publish(feature_vector)
                
                logger.info(
                    "features_computed_and_published",
                    symbol=symbol,
                    interval=interval_name,
                    features_count=len(feature_vector.features),
                )
            
            except Exception as e:
                logger.error(
                    "feature_computation_error",
                    symbol=symbol,
                    interval=interval_name,
                    error=str(e),
                    exc_info=True,
                )

