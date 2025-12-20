"""
Feature computation scheduler for computing features at intervals.
"""
import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, Set, TYPE_CHECKING
import structlog

from src.services.feature_computer import FeatureComputer
from src.publishers.feature_publisher import FeaturePublisher
from src.services.feature_requirements import FeatureRequirementsAnalyzer

if TYPE_CHECKING:
    from src.services.feature_registry import FeatureRegistryLoader
    from src.services.target_registry_version_manager import TargetRegistryVersionManager

logger = structlog.get_logger(__name__)


class FeatureScheduler:
    """Schedules feature computation at dynamically computed intervals based on Feature Registry and Target Registry."""
    
    def __init__(
        self,
        feature_computer: FeatureComputer,
        feature_publisher: FeaturePublisher,
        symbols: Optional[list] = None,
        feature_registry_loader: Optional["FeatureRegistryLoader"] = None,
        target_registry_version_manager: Optional["TargetRegistryVersionManager"] = None,
    ):
        """
        Initialize feature scheduler.
        
        Args:
            feature_computer: FeatureComputer instance
            feature_publisher: FeaturePublisher instance
            symbols: List of symbols to schedule
            feature_registry_loader: Optional FeatureRegistryLoader for dynamic interval computation
            target_registry_version_manager: Optional TargetRegistryVersionManager for dynamic interval computation
        """
        self._feature_computer = feature_computer
        self._feature_publisher = feature_publisher
        self._symbols = set(symbols or [])
        self._feature_registry_loader = feature_registry_loader
        self._target_registry_version_manager = target_registry_version_manager
        # Default intervals (will be updated dynamically if loaders are provided)
        self._intervals = {
            "1s": 1,
            "3s": 3,
            "15s": 15,
            "1m": 60,
        }
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._update_lock = asyncio.Lock()
    
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
    
    @staticmethod
    def _parse_interval_to_seconds(interval_name: str) -> Optional[int]:
        """
        Parse interval name to seconds.
        
        Args:
            interval_name: Interval name (e.g., "1s", "3s", "15s", "1m")
            
        Returns:
            Seconds or None if invalid
        """
        if not interval_name:
            return None
        
        try:
            unit = interval_name[-1]
            value = int(interval_name[:-1])
            
            if unit == "s":
                return value
            elif unit == "m":
                return value * 60
            elif unit == "h":
                return value * 3600
            else:
                return None
        except (ValueError, IndexError):
            return None
    
    async def _compute_required_intervals(
        self,
    ) -> Dict[str, int]:
        """
        Compute required intervals based on Feature Registry and Target Registry.
        
        Returns:
            Dictionary mapping interval names to seconds
        """
        # Default intervals if no loaders available
        default_intervals = {
            "1s": 1,
            "3s": 3,
            "15s": 15,
            "1m": 60,
        }
        
        if not self._feature_registry_loader:
            logger.info(
                "feature_registry_loader_not_available",
                message="Using default intervals",
                intervals=default_intervals,
            )
            return default_intervals
        
        try:
            # 1. Get intervals from Feature Registry
            feature_requirements_analyzer = FeatureRequirementsAnalyzer(self._feature_registry_loader)
            window_requirements = feature_requirements_analyzer.compute_requirements()
            trade_intervals = window_requirements.trade_intervals
            
            # 2. Get horizon from Target Registry
            horizon_seconds: Optional[int] = None
            if self._target_registry_version_manager:
                try:
                    target_config = await self._target_registry_version_manager.load_active_version()
                    horizon_seconds = target_config.get("horizon")
                    if horizon_seconds:
                        logger.debug(
                            "target_registry_horizon_loaded",
                            horizon_seconds=horizon_seconds,
                        )
                except Exception as e:
                    logger.warning(
                        "failed_to_load_target_registry_horizon",
                        error=str(e),
                        message="Continuing without target registry constraint",
                    )
            
            # 3. Compute required intervals
            required_intervals: Dict[str, int] = {}
            
            # Sort intervals by duration (shortest first)
            sorted_intervals = sorted(
                trade_intervals,
                key=lambda x: self._parse_interval_to_seconds(x) or float('inf')
            )
            
            for interval_name in sorted_intervals:
                seconds = self._parse_interval_to_seconds(interval_name)
                if seconds is None:
                    continue
                
                # If horizon is set, only include intervals <= horizon
                if horizon_seconds is not None and seconds > horizon_seconds:
                    logger.debug(
                        "interval_exceeds_horizon",
                        interval=interval_name,
                        interval_seconds=seconds,
                        horizon_seconds=horizon_seconds,
                        message="Skipping interval that exceeds prediction horizon",
                    )
                    continue
                
                required_intervals[interval_name] = seconds
            
            # If no intervals found, use default
            if not required_intervals:
                logger.warning(
                    "no_intervals_computed",
                    message="Using default intervals",
                    default_intervals=default_intervals,
                )
                return default_intervals
            
            # Always ensure at least 1m interval is present if horizon allows
            if horizon_seconds is None or horizon_seconds >= 60:
                if "1m" not in required_intervals:
                    required_intervals["1m"] = 60
            
            logger.info(
                "intervals_computed_from_registries",
                feature_registry_intervals=list(trade_intervals),
                target_registry_horizon=horizon_seconds,
                computed_intervals=required_intervals,
            )
            
            return required_intervals
            
        except Exception as e:
            logger.error(
                "failed_to_compute_intervals",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
                message="Using default intervals",
            )
            return default_intervals
    
    async def update_intervals(self) -> None:
        """
        Update intervals based on current Feature Registry and Target Registry.
        Stops old tasks and starts new ones with updated intervals.
        """
        async with self._update_lock:
            try:
                # Compute new intervals
                new_intervals = await self._compute_required_intervals()
                
                # If intervals haven't changed, skip update
                if new_intervals == self._intervals:
                    logger.debug(
                        "intervals_unchanged",
                        intervals=self._intervals,
                        message="Skipping scheduler update",
                    )
                    return
                
                logger.info(
                    "updating_scheduler_intervals",
                    old_intervals=self._intervals,
                    new_intervals=new_intervals,
                )
                
                # Stop old tasks if running
                was_running = self._running
                if was_running:
                    # Temporarily set _running to False to stop gracefully
                    self._running = False
                    # Cancel all tasks
                    for task in self._tasks.values():
                        task.cancel()
                    # Wait for tasks to complete
                    await asyncio.gather(*self._tasks.values(), return_exceptions=True)
                    self._tasks.clear()
                
                # Update intervals
                self._intervals = new_intervals
                
                # Restart if was running
                if was_running:
                    self._running = True
                    # Start scheduler tasks for each interval
                    for interval_name, interval_seconds in self._intervals.items():
                        task = asyncio.create_task(
                            self._schedule_interval(interval_name, interval_seconds)
                        )
                        self._tasks[interval_name] = task
                    
                logger.info(
                    "scheduler_intervals_updated",
                    new_intervals=new_intervals,
                )
                
            except Exception as e:
                logger.error(
                    "failed_to_update_intervals",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                    message="Scheduler may be in inconsistent state",
                )
                raise
    
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

