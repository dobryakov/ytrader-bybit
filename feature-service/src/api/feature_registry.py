"""
Feature Registry API endpoints.
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel, Field
import structlog

from src.services.feature_registry import FeatureRegistryLoader
from src.services.feature_registry_version_manager import FeatureRegistryVersionManager
from src.services.feature_registry_hot_reload import reload_registry_in_memory
from src.storage.metadata_storage import MetadataStorage
from src.models.feature_registry import FeatureRegistry
from src.api.middleware.auth import verify_api_key

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/feature-registry", tags=["feature-registry"])

# Global instances (will be set during app startup)
_feature_registry_loader: Optional[FeatureRegistryLoader] = None
_feature_registry_version_manager: Optional[FeatureRegistryVersionManager] = None
_metadata_storage: Optional[MetadataStorage] = None
_feature_computer: Optional[Any] = None  # FeatureComputer instance
_dataset_builder: Optional[Any] = None  # OptimizedDatasetBuilder instance
_orderbook_manager: Optional[Any] = None  # OrderbookManager instance
_feature_scheduler: Optional[Any] = None  # FeatureScheduler instance
_target_registry_version_manager: Optional[Any] = None  # TargetRegistryVersionManager instance


def set_feature_registry_loader(loader: FeatureRegistryLoader) -> None:
    """Set global feature registry loader instance."""
    global _feature_registry_loader
    _feature_registry_loader = loader


def set_feature_registry_version_manager(manager: FeatureRegistryVersionManager) -> None:
    """Set global feature registry version manager instance."""
    global _feature_registry_version_manager
    _feature_registry_version_manager = manager


def set_metadata_storage_for_registry(storage: MetadataStorage) -> None:
    """Set global metadata storage instance."""
    global _metadata_storage
    _metadata_storage = storage


def set_feature_computer_for_registry(computer: Any) -> None:
    """Set global feature computer instance for hot reload."""
    global _feature_computer
    _feature_computer = computer


def set_dataset_builder_for_registry(builder: Any) -> None:
    """Set global dataset builder instance for hot reload."""
    global _dataset_builder
    _dataset_builder = builder


def set_orderbook_manager_for_registry(manager: Any) -> None:
    """Set global orderbook manager instance for hot reload."""
    global _orderbook_manager
    _orderbook_manager = manager


def set_feature_scheduler_for_registry(scheduler: Any) -> None:
    """Set global feature scheduler instance for hot reload."""
    global _feature_scheduler
    _feature_scheduler = scheduler


def set_target_registry_version_manager_for_registry(manager: Any) -> None:
    """Set global target registry version manager instance for hot reload."""
    global _target_registry_version_manager
    _target_registry_version_manager = manager


# Request/Response models
class FeatureRegistryVersionCreateRequest(BaseModel):
    """Request model for creating a new Feature Registry version."""
    version: str = Field(description="Version identifier (e.g., '1.0.0')")
    config: Dict[str, Any] = Field(description="Feature Registry configuration")


class FeatureRegistryVersionActivateRequest(BaseModel):
    """Request model for activating a Feature Registry version."""
    acknowledge_breaking_changes: bool = Field(
        default=False,
        description="Acknowledge breaking changes (required if breaking changes detected)"
    )
    activation_reason: Optional[str] = Field(
        default=None,
        description="Reason for activation"
    )
    activated_by: Optional[str] = Field(
        default=None,
        description="User who activates this version"
    )


class FeatureRegistrySyncFileResponse(BaseModel):
    """Response model for sync-file operation."""
    version: str
    file_path: str
    validation_status: str
    validation_errors: Optional[List[str]] = None


# Basic endpoints (T175-T177)
@router.get("")
async def get_feature_registry(
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get current Feature Registry configuration.
    
    Returns:
        Current Feature Registry configuration
    """
    if _feature_registry_loader is None:
        raise HTTPException(status_code=503, detail="Feature Registry loader not available")
    
    try:
        config = _feature_registry_loader.get_config()
        if config is None:
            raise HTTPException(
                status_code=404,
                detail="Feature Registry not loaded"
            )
        return config
    except Exception as e:
        logger.error("Failed to get Feature Registry", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get Feature Registry: {str(e)}")


@router.post("/reload")
async def reload_feature_registry(
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Reload Feature Registry configuration from file or database.
    
    Returns:
        Reloaded Feature Registry configuration
    """
    if _feature_registry_loader is None:
        raise HTTPException(status_code=503, detail="Feature Registry loader not available")
    
    try:
        if _feature_registry_loader._use_db and _feature_registry_loader._version_manager:
            # Database-driven mode - reload from DB
            config = await _feature_registry_loader.load_async()
        else:
            # File mode - reload from file
            config = _feature_registry_loader.reload()
        
        logger.info("Feature Registry reloaded", version=config.get("version"))
        return config
    except Exception as e:
        logger.error("Failed to reload Feature Registry", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to reload Feature Registry: {str(e)}")


@router.get("/validate")
async def validate_feature_registry(
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Validate current Feature Registry configuration.
    
    Returns:
        Validation result with status and errors (if any)
    """
    if _feature_registry_loader is None:
        raise HTTPException(status_code=503, detail="Feature Registry loader not available")
    
    try:
        config = _feature_registry_loader.get_config()
        if config is None:
            raise HTTPException(
                status_code=404,
                detail="Feature Registry not loaded"
            )
        
        # Validate using FeatureRegistry model
        from src.models.feature_registry import FeatureRegistry
        try:
            FeatureRegistry(**config)
            return {
                "status": "valid",
                "version": config.get("version"),
                "errors": []
            }
        except Exception as validation_error:
            return {
                "status": "invalid",
                "version": config.get("version"),
                "errors": [str(validation_error)]
            }
    except Exception as e:
        logger.error("Failed to validate Feature Registry", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to validate Feature Registry: {str(e)}")


# Version management endpoints (T208-T213, T317-T319)
@router.get("/versions")
async def list_feature_registry_versions(
    _: None = Depends(verify_api_key),
) -> List[Dict[str, Any]]:
    """
    List all Feature Registry versions with metadata.
    
    Returns:
        List of version records with metadata
    """
    if _metadata_storage is None:
        raise HTTPException(status_code=503, detail="Metadata storage not available")
    
    try:
        versions = await _metadata_storage.list_feature_registry_versions()
        return versions
    except Exception as e:
        logger.error("Failed to list Feature Registry versions", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list versions: {str(e)}"
        )


@router.get("/versions/{version}")
async def get_feature_registry_version(
    version: str,
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get specific Feature Registry version metadata.
    
    Args:
        version: Version identifier
        
    Returns:
        Version record with metadata
    """
    if _metadata_storage is None:
        raise HTTPException(status_code=503, detail="Metadata storage not available")
    
    try:
        version_record = await _metadata_storage.get_feature_registry_version(version)
        if version_record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Feature Registry version not found: {version}"
            )
        return version_record
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get Feature Registry version", version=version, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get version: {str(e)}"
        )


@router.post("/versions")
async def create_feature_registry_version(
    request: FeatureRegistryVersionCreateRequest,
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Create a new Feature Registry version.
    
    Saves config to file and creates DB record with metadata.
    
    Args:
        request: Version creation request with config and version string
        
    Returns:
        Created version record with metadata
    """
    if _feature_registry_version_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Feature Registry version manager not available"
        )
    
    try:
        version_record = await _feature_registry_version_manager.create_version(
            version=request.version,
            config_data=request.config,
            created_by=None,  # TODO: Extract from auth context
        )
        logger.info("Feature Registry version created", version=request.version)
        return version_record
    except ValueError as e:
        logger.warning("Failed to create version", version=request.version, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create Feature Registry version", version=request.version, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create version: {str(e)}"
        )


@router.post("/versions/{version}/activate")
async def activate_feature_registry_version(
    version: str,
    request: FeatureRegistryVersionActivateRequest = Body(...),
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Activate a Feature Registry version with automatic rollback on failure.
    
    Args:
        version: Version identifier to activate
        request: Activation request with breaking changes acknowledgment
        
    Returns:
        Activated version record with metadata
    """
    global _feature_computer, _feature_registry_loader, _dataset_builder, _orderbook_manager
    
    if _feature_registry_version_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Feature Registry version manager not available"
        )
    
    try:
        activated_record = await _feature_registry_version_manager.activate_version(
            version=version,
            activated_by=request.activated_by,
            activation_reason=request.activation_reason,
            acknowledge_breaking_changes=request.acknowledge_breaking_changes,
        )
        
        # Hot reload: reload registry in memory without restart
        try:
            # Load config from activated version file
            config_data = await _feature_registry_version_manager.load_active_version()
            
            if not config_data:
                raise ValueError("Failed to load config from activated version")
            
            # Get global instances for hot reload
            if not _feature_computer or not _feature_registry_loader or not _orderbook_manager:
                raise RuntimeError("Required services not initialized for hot reload")
            
            # Reload in memory
            reload_result = await reload_registry_in_memory(
                new_config=config_data,
                feature_computer=_feature_computer,
                feature_registry_loader=_feature_registry_loader,
                dataset_builder=_dataset_builder,
                orderbook_manager=_orderbook_manager,
                feature_scheduler=_feature_scheduler,
                target_registry_version_manager=_target_registry_version_manager,
            )
            
            # Update global feature_computer reference
            # Note: reload_registry_in_memory already updates API reference via set_feature_computer
            # Update our local reference
            _feature_computer = reload_result.get("new_feature_computer", _feature_computer)
            
            logger.info(
                "Feature Registry version activated with hot reload",
                version=version,
                hot_reload=True,
            )
            
            return {
                **activated_record,
                "hot_reload": True,
                "reload_result": reload_result,
            }
        except Exception as reload_error:
            logger.error(
                "Hot reload failed after activation, attempting rollback",
                version=version,
                error=str(reload_error),
            )
            
            # Rollback activation
            try:
                await _metadata_storage.rollback_feature_registry_version()
                logger.info("Activation rolled back due to hot reload failure", version=version)
            except Exception as rollback_error:
                logger.error(
                    "Failed to rollback activation after hot reload failure",
                    version=version,
                    error=str(rollback_error),
                )
            
            raise HTTPException(
                status_code=500,
                detail=f"Activation succeeded but hot reload failed: {str(reload_error)}. Activation has been rolled back."
            )
    except ValueError as e:
        logger.warning("Failed to activate version", version=version, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.warning("Version file not found", version=version, error=str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to activate Feature Registry version", version=version, error=str(e))
        # TODO: Implement automatic rollback on failure
        raise HTTPException(
            status_code=500,
            detail=f"Failed to activate version: {str(e)}"
        )


@router.post("/rollback")
async def rollback_feature_registry_version(
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Rollback to previous Feature Registry version.
    
    Returns:
        Activated version record (previous version)
    """
    if _metadata_storage is None:
        raise HTTPException(status_code=503, detail="Metadata storage not available")
    
    try:
        rolled_back_version = await _metadata_storage.rollback_feature_registry_version()
        if rolled_back_version is None:
            raise HTTPException(
                status_code=404,
                detail="No previous version available for rollback"
            )
        
        logger.info("Feature Registry rolled back", version=rolled_back_version["version"])
        return rolled_back_version
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to rollback Feature Registry", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rollback: {str(e)}"
        )


@router.get("/versions/{version}/usage")
async def get_feature_registry_version_usage(
    version: str,
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Check Feature Registry version usage (how many datasets use this version).
    
    Args:
        version: Version identifier
        
    Returns:
        Usage information with dataset count
    """
    if _metadata_storage is None:
        raise HTTPException(status_code=503, detail="Metadata storage not available")
    
    try:
        usage_count = await _metadata_storage.check_version_usage(version)
        return {
            "version": version,
            "usage_count": usage_count,
            "in_use": usage_count > 0,
        }
    except Exception as e:
        logger.error("Failed to check version usage", version=version, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check usage: {str(e)}"
        )


@router.delete("/versions/{version}")
async def delete_feature_registry_version(
    version: str,
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Delete Feature Registry version (only if not in use).
    
    Args:
        version: Version identifier to delete
        
    Returns:
        Deletion result
    """
    if _feature_registry_version_manager is None or _metadata_storage is None:
        raise HTTPException(
            status_code=503,
            detail="Feature Registry version manager not available"
        )
    
    try:
        # Check if version can be deleted
        can_delete = await _feature_registry_version_manager.can_delete_version(version)
        if not can_delete:
            usage_count = await _metadata_storage.check_version_usage(version)
            raise HTTPException(
                status_code=409,
                detail=f"Version {version} is in use by {usage_count} dataset(s) and cannot be deleted"
            )
        
        # Delete version record from DB
        # Note: File deletion should be handled separately (manual or via cleanup script)
        deleted = await _metadata_storage.delete_feature_registry_version(version)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Feature Registry version not found: {version}"
            )
        
        logger.info("Feature Registry version deleted", version=version)
        return {
            "version": version,
            "deleted": True,
            "message": "Version record deleted (file deletion should be handled separately)"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete Feature Registry version", version=version, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete version: {str(e)}"
        )


@router.post("/versions/{version}/sync-file")
async def sync_feature_registry_file(
    version: str,
    _: None = Depends(verify_api_key),
) -> FeatureRegistrySyncFileResponse:
    """
    Sync Feature Registry file to database metadata.
    
    Use case: File was edited manually, need to sync metadata in DB.
    
    Args:
        version: Version identifier
        
    Returns:
        Sync result with validation status
    """
    if _feature_registry_version_manager is None or _metadata_storage is None:
        raise HTTPException(
            status_code=503,
            detail="Feature Registry version manager not available"
        )
    
    try:
        # Get version record
        version_record = await _metadata_storage.get_feature_registry_version(version)
        if version_record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Feature Registry version not found: {version}"
            )
        
        file_path = version_record["file_path"]
        
        # Load and validate config from file
        from pathlib import Path
        import yaml
        
        path = Path(file_path)
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Feature Registry file not found: {file_path}"
            )
        
        with open(path, "r") as f:
            config_data = yaml.safe_load(f)
        
        # Validate config
        validation_errors = []
        try:
            FeatureRegistry(**config_data)
            validation_status = "valid"
        except Exception as e:
            validation_status = "invalid"
            validation_errors = [str(e)]
        
        # Update DB metadata (validated_at, validation_errors)
        # TODO: Implement update_feature_registry_version_metadata method in MetadataStorage
        # For now, just return sync result
        
        logger.info(
            "Feature Registry file synced",
            version=version,
            file_path=file_path,
            validation_status=validation_status,
        )
        
        return FeatureRegistrySyncFileResponse(
            version=version,
            file_path=file_path,
            validation_status=validation_status,
            validation_errors=validation_errors if validation_errors else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to sync Feature Registry file", version=version, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync file: {str(e)}"
        )

