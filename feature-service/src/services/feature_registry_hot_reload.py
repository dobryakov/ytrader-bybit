"""
Hot reload mechanism for Feature Registry.
"""
from typing import Dict, Any, Optional
from threading import Lock
from src.logging import get_logger
from src.services.feature_computer import FeatureComputer
from src.services.orderbook_manager import OrderbookManager
from src.services.dataset_builder import DatasetBuilder
from src.services.feature_registry import FeatureRegistryLoader
from src.api.features import set_feature_computer

logger = get_logger(__name__)

# Lock for hot reload to prevent concurrent reloads
_reload_lock = Lock()


async def reload_registry_in_memory(
    new_config: Dict[str, Any],
    feature_computer: FeatureComputer,
    feature_registry_loader: FeatureRegistryLoader,
    dataset_builder: Optional[DatasetBuilder],
    orderbook_manager: OrderbookManager,
) -> Dict[str, Any]:
    """
    Reload Feature Registry in memory without service restart (hot reload).
    
    Updates:
    - FeatureComputer with new version
    - DatasetBuilder version (for new datasets only)
    - FeatureRegistryLoader config
    - Global variables atomically
    
    Args:
        new_config: New Feature Registry configuration dict
        feature_computer: Current FeatureComputer instance
        feature_registry_loader: Current FeatureRegistryLoader instance
        dataset_builder: Current DatasetBuilder instance (optional)
        orderbook_manager: OrderbookManager instance
        
    Returns:
        Reload result with status
        
    Raises:
        RuntimeError: If reload fails (rollback attempted)
    """
    # Prevent concurrent reloads
    if not _reload_lock.acquire(blocking=False):
        raise RuntimeError("Another reload operation is in progress")
    
    old_config = None
    old_version = None
    old_feature_computer = None
    
    try:
        # Save current state for rollback
        if feature_registry_loader:
            old_config = feature_registry_loader.get_config()
            if old_config:
                old_version = old_config.get("version")
        
        old_feature_computer = feature_computer
        
        # Extract new version
        new_version = new_config.get("version", "1.0.0")
        
        logger.info(
            "Starting hot reload of Feature Registry",
            old_version=old_version,
            new_version=new_version,
        )
        
        # Update FeatureRegistryLoader config
        if feature_registry_loader:
            feature_registry_loader.set_config(new_config)
        
        # Recreate FeatureComputer with new version
        new_feature_computer = FeatureComputer(
            orderbook_manager=orderbook_manager,
            feature_registry_version=new_version,
            feature_registry_loader=feature_registry_loader,
        )
        
        # Update API reference
        set_feature_computer(new_feature_computer)
        
        # Update DatasetBuilder version (for new datasets only)
        # Note: Existing builds continue with old version, new builds use new version
        if dataset_builder:
            dataset_builder._feature_registry_version = new_version
            # Recreate OfflineEngine with new version
            from src.services.offline_engine import OfflineEngine
            dataset_builder._offline_engine = OfflineEngine(new_version)
        
        logger.info(
            "Feature Registry hot reload completed successfully",
            old_version=old_version,
            new_version=new_version,
        )
        
        return {
            "status": "success",
            "old_version": old_version,
            "new_version": new_version,
            "hot_reload": True,
            "new_feature_computer": new_feature_computer,  # Return new instance for caller to update
        }
    
    except Exception as e:
        logger.error(
            "Feature Registry hot reload failed, attempting rollback",
            error=str(e),
            old_version=old_version,
        )
        
        # Rollback: restore previous state
        try:
            if old_config and feature_registry_loader:
                feature_registry_loader.set_config(old_config)
            
            if old_feature_computer:
                set_feature_computer(old_feature_computer)
            
            if old_version and dataset_builder:
                dataset_builder._feature_registry_version = old_version
                from src.services.offline_engine import OfflineEngine
                dataset_builder._offline_engine = OfflineEngine(
                    feature_registry_version=old_version,
                    feature_registry_loader=feature_registry_loader,
                )
            
            logger.info("Rollback completed", restored_version=old_version)
        except Exception as rollback_error:
            logger.error(
                "Rollback failed - service may be in inconsistent state",
                error=str(rollback_error),
            )
        
        raise RuntimeError(f"Hot reload failed: {str(e)}") from e
    
    finally:
        _reload_lock.release()

