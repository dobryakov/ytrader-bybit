"""
Hot reload mechanism for Feature Registry.
"""
from typing import Dict, Any, Optional, TYPE_CHECKING
from threading import Lock
from src.logging import get_logger
from src.services.feature_computer import FeatureComputer
from src.services.orderbook_manager import OrderbookManager
from src.services.optimized_dataset.optimized_builder import OptimizedDatasetBuilder
from src.services.feature_registry import FeatureRegistryLoader
from src.api.features import set_feature_computer

if TYPE_CHECKING:
    from src.services.feature_scheduler import FeatureScheduler
    from src.services.target_registry_version_manager import TargetRegistryVersionManager

logger = get_logger(__name__)

# Lock for hot reload to prevent concurrent reloads
_reload_lock = Lock()


async def reload_registry_in_memory(
    new_config: Dict[str, Any],
    feature_computer: FeatureComputer,
    feature_registry_loader: FeatureRegistryLoader,
    dataset_builder: Optional[OptimizedDatasetBuilder],
    orderbook_manager: OrderbookManager,
    feature_scheduler: Optional["FeatureScheduler"] = None,
    target_registry_version_manager: Optional["TargetRegistryVersionManager"] = None,
) -> Dict[str, Any]:
    """
    Reload Feature Registry in memory without service restart (hot reload).
    
    Updates:
    - FeatureComputer with new version
    - DatasetBuilder version (for new datasets only)
    - FeatureRegistryLoader config
    - FeatureScheduler intervals (if provided)
    - Global variables atomically
    
    Args:
        new_config: New Feature Registry configuration dict
        feature_computer: Current FeatureComputer instance
        feature_registry_loader: Current FeatureRegistryLoader instance
        dataset_builder: Current DatasetBuilder instance (optional)
        orderbook_manager: OrderbookManager instance
        feature_scheduler: Optional FeatureScheduler instance to update intervals
        target_registry_version_manager: Optional TargetRegistryVersionManager for interval computation
        
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
        
        # Update OptimizedDatasetBuilder - version is handled via feature_registry_loader
        # Note: Existing builds continue with old version, new builds use new version
        # OptimizedDatasetBuilder uses feature_registry_loader which is already updated above
        if dataset_builder:
            # Version is managed through feature_registry_loader, no direct update needed
            pass
        
        # Update FeatureScheduler intervals if provided
        scheduler_updated = False
        if feature_scheduler:
            try:
                await feature_scheduler.update_intervals()
                scheduler_updated = True
                logger.info(
                    "FeatureScheduler intervals updated after hot reload",
                    new_intervals=feature_scheduler._intervals,
                )
            except Exception as scheduler_error:
                logger.warning(
                    "failed_to_update_scheduler_after_hot_reload",
                    error=str(scheduler_error),
                    message="Hot reload succeeded but scheduler intervals not updated",
                )
        
        logger.info(
            "Feature Registry hot reload completed successfully",
            old_version=old_version,
            new_version=new_version,
            scheduler_updated=scheduler_updated,
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
            
            # OptimizedDatasetBuilder version is managed through feature_registry_loader
            # which is already rolled back above
            if dataset_builder:
                pass
            
            logger.info("Rollback completed", restored_version=old_version)
        except Exception as rollback_error:
            logger.error(
                "Rollback failed - service may be in inconsistent state",
                error=str(rollback_error),
            )
        
        raise RuntimeError(f"Hot reload failed: {str(e)}") from e
    
    finally:
        _reload_lock.release()

