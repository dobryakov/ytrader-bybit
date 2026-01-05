"""
Feature Computer Manager for managing multiple FeatureComputer instances by version.

Manages a pool of FeatureComputer instances, one per Feature Registry version,
all sharing common rolling windows for efficiency.
"""
import asyncio
from typing import Dict, Optional, TYPE_CHECKING
import structlog

from src.models.rolling_windows import RollingWindows
from src.services.feature_requirements import FeatureRequirementsAnalyzer, WindowRequirements

if TYPE_CHECKING:
    from src.services.feature_computer import FeatureComputer
    from src.services.orderbook_manager import OrderbookManager
    from src.services.feature_registry_version_manager import FeatureRegistryVersionManager
    from src.services.feature_registry import FeatureRegistryLoader
    from src.models.feature_registry import FeatureRegistry

logger = structlog.get_logger(__name__)


class FeatureComputerManager:
    """Manages FeatureComputer instances for different Feature Registry versions."""
    
    def __init__(
        self,
        orderbook_manager: "OrderbookManager",
        feature_registry_version_manager: "FeatureRegistryVersionManager",
        shared_rolling_windows: Dict[str, RollingWindows],
    ):
        """
        Initialize FeatureComputer manager.
        
        Args:
            orderbook_manager: OrderbookManager instance
            feature_registry_version_manager: FeatureRegistryVersionManager for loading versions
            shared_rolling_windows: Dictionary of symbol -> RollingWindows (shared across all versions)
        """
        self._orderbook_manager = orderbook_manager
        self._version_manager = feature_registry_version_manager
        self._shared_rolling_windows = shared_rolling_windows
        self._computers: Dict[str, "FeatureComputer"] = {}  # version -> FeatureComputer
        self._active_version: Optional[str] = None
        self._lock = asyncio.Lock()
        self._version_requirements: Dict[str, WindowRequirements] = {}  # version -> requirements
    
    async def get_or_create_computer(
        self,
        feature_registry_version: Optional[str] = None
    ) -> "FeatureComputer":
        """
        Get or create FeatureComputer for specified version.
        
        Args:
            feature_registry_version: Feature Registry version (None = active version)
        
        Returns:
            FeatureComputer instance for the version
        """
        # If version not specified, use active version
        if feature_registry_version is None:
            version_record = await self._version_manager.load_active_version()
            feature_registry_version = version_record["version"]
            self._active_version = feature_registry_version
        
        # Check cache
        if feature_registry_version in self._computers:
            return self._computers[feature_registry_version]
        
        async with self._lock:
            # Double-check after acquiring lock
            if feature_registry_version in self._computers:
                return self._computers[feature_registry_version]
            
            # Load version config
            try:
                config_data = await self._version_manager.load_version(feature_registry_version)
            except FileNotFoundError as e:
                logger.error(
                    "feature_registry_version_not_found",
                    version=feature_registry_version,
                    error=str(e),
                )
                raise ValueError(f"Feature Registry version {feature_registry_version} not found") from e
            
            # Create FeatureRegistryLoader for this version
            from src.models.feature_registry import FeatureRegistry
            from src.services.feature_registry import FeatureRegistryLoader
            
            loader = FeatureRegistryLoader(
                use_db=True,
                version_manager=self._version_manager,
            )
            # Set config directly (bypass async load)
            loader._validate_and_store_config(config_data)
            
            # Compute window requirements for this version
            analyzer = FeatureRequirementsAnalyzer(loader)
            requirements = analyzer.compute_requirements()
            self._version_requirements[feature_registry_version] = requirements
            
            # Update shared rolling windows structure if needed
            await self._update_shared_rolling_windows_requirements()
            
            # Create FeatureComputer with shared rolling windows
            from src.services.feature_computer import FeatureComputer
            
            computer = FeatureComputer(
                orderbook_manager=self._orderbook_manager,
                feature_registry_version=feature_registry_version,
                feature_registry_loader=loader,
                shared_rolling_windows=self._shared_rolling_windows,
            )
            
            self._computers[feature_registry_version] = computer
            
            logger.info(
                "feature_computer_created",
                version=feature_registry_version,
                cached_computers=len(self._computers),
            )
            
            return computer
    
    async def _update_shared_rolling_windows_requirements(self) -> None:
        """
        Update shared rolling windows structure based on union of all version requirements.
        
        Merges trade_intervals and max_lookback_minutes_1m from all loaded versions.
        """
        if not self._version_requirements:
            return
        
        # Compute union of all requirements
        all_trade_intervals: set = set()
        max_lookback = 0
        
        for version, requirements in self._version_requirements.items():
            all_trade_intervals.update(requirements.trade_intervals)
            max_lookback = max(max_lookback, requirements.max_lookback_minutes_1m)
        
        # Ensure 1m interval is always present (for klines)
        all_trade_intervals.add("1m")
        
        # Update existing rolling windows structure if needed
        # Note: We can't change structure of existing windows, but we ensure
        # new windows are created with the merged structure
        logger.debug(
            "shared_rolling_windows_requirements_updated",
            all_trade_intervals=sorted(all_trade_intervals),
            max_lookback_minutes=max_lookback,
            versions_count=len(self._version_requirements),
        )
    
    def get_shared_rolling_windows(self, symbol: str) -> Optional[RollingWindows]:
        """
        Get shared rolling windows for symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            RollingWindows instance or None if not initialized
        """
        return self._shared_rolling_windows.get(symbol)
    
    async def initialize_shared_rolling_windows(
        self,
        symbol: str,
        trade_intervals: Optional[set] = None,
        max_lookback_minutes_1m: Optional[int] = None,
    ) -> RollingWindows:
        """
        Initialize shared rolling windows for symbol with merged requirements.
        
        Args:
            symbol: Trading pair symbol
            trade_intervals: Optional set of trade intervals (default: from merged requirements)
            max_lookback_minutes_1m: Optional max lookback (default: from merged requirements)
        
        Returns:
            RollingWindows instance
        """
        if symbol in self._shared_rolling_windows:
            return self._shared_rolling_windows[symbol]
        
        # Compute merged requirements if not provided
        if trade_intervals is None or max_lookback_minutes_1m is None:
            all_trade_intervals: set = set()
            max_lookback = 0
            
            for requirements in self._version_requirements.values():
                all_trade_intervals.update(requirements.trade_intervals)
                max_lookback = max(max_lookback, requirements.max_lookback_minutes_1m)
            
            all_trade_intervals.add("1m")
            if max_lookback <= 0:
                max_lookback = 30  # Default
            
            trade_intervals = trade_intervals or all_trade_intervals
            max_lookback_minutes_1m = max_lookback_minutes_1m or max_lookback
        
        # Create rolling windows with merged structure
        from datetime import datetime, timezone
        import pandas as pd
        
        windows: Dict[str, pd.DataFrame] = {}
        for interval in trade_intervals:
            if interval == "1m":
                # Klines window
                windows[interval] = pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
            else:
                # Trade windows
                windows[interval] = pd.DataFrame(
                    columns=["timestamp", "price", "volume", "side"]
                )
        
        rolling_windows = RollingWindows(
            symbol=symbol,
            windows=windows,
            last_update=datetime.now(timezone.utc),
            window_intervals=trade_intervals,
            max_lookback_minutes_1m=max_lookback_minutes_1m,
        )
        
        self._shared_rolling_windows[symbol] = rolling_windows
        
        logger.info(
            "shared_rolling_windows_initialized",
            symbol=symbol,
            trade_intervals=sorted(trade_intervals),
            max_lookback_minutes_1m=max_lookback_minutes_1m,
        )
        
        return rolling_windows

