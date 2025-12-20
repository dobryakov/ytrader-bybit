"""
Target Registry API endpoints.
"""
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import structlog

from src.services.target_registry_version_manager import TargetRegistryVersionManager
from src.storage.metadata_storage import MetadataStorage
from src.models.dataset import TargetConfig
from src.api.middleware.auth import verify_api_key

if TYPE_CHECKING:
    from src.services.feature_scheduler import FeatureScheduler

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/target-registry", tags=["target-registry"])

# Global instances (will be set during app startup)
_target_registry_version_manager: Optional[TargetRegistryVersionManager] = None
_metadata_storage: Optional[MetadataStorage] = None
_feature_scheduler: Optional[Any] = None  # FeatureScheduler instance


def set_target_registry_version_manager(manager: TargetRegistryVersionManager) -> None:
    """Set global target registry version manager instance."""
    global _target_registry_version_manager
    _target_registry_version_manager = manager


def set_metadata_storage_for_target_registry(storage: MetadataStorage) -> None:
    """Set global metadata storage instance."""
    global _metadata_storage
    _metadata_storage = storage


def set_feature_scheduler_for_target_registry(scheduler: Any) -> None:
    """Set global feature scheduler instance for updating intervals."""
    global _feature_scheduler
    _feature_scheduler = scheduler


# Request/Response models
class TargetRegistryVersionCreateRequest(BaseModel):
    """Request model for creating a new Target Registry version."""
    version: str = Field(description="Version identifier (e.g., '1.4.0')")
    config: TargetConfig = Field(description="Target configuration")
    description: Optional[str] = Field(default=None, description="Description of this version")


class TargetRegistryVersionActivateRequest(BaseModel):
    """Request model for activating a Target Registry version."""
    activation_reason: Optional[str] = Field(
        default=None,
        description="Reason for activation"
    )
    activated_by: Optional[str] = Field(
        default=None,
        description="User who activates this version"
    )


# Basic endpoints
@router.get("")
async def get_target_registry(
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get active Target Registry version.
    
    Returns:
        Active Target Registry configuration
    """
    if _target_registry_version_manager is None:
        raise HTTPException(status_code=503, detail="Target Registry not initialized")
    
    try:
        config = await _target_registry_version_manager.load_active_version()
        return {
            "active": True,
            "config": config,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No active Target Registry version found")
    except Exception as e:
        logger.error("Failed to load active Target Registry", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to load Target Registry: {str(e)}")


@router.get("/versions")
async def list_target_registry_versions(
    _: None = Depends(verify_api_key),
) -> List[Dict[str, Any]]:
    """
    List all Target Registry versions.
    
    Returns:
        List of all Target Registry versions
    """
    if _target_registry_version_manager is None:
        raise HTTPException(status_code=503, detail="Target Registry not initialized")
    
    try:
        versions = await _target_registry_version_manager.list_versions()
        return versions
    except Exception as e:
        logger.error("Failed to list Target Registry versions", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list versions: {str(e)}")


@router.get("/versions/{version}")
async def get_target_registry_version(
    version: str,
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Get a specific Target Registry version.
    
    Args:
        version: Version identifier
        
    Returns:
        Target Registry version configuration
    """
    if _target_registry_version_manager is None:
        raise HTTPException(status_code=503, detail="Target Registry not initialized")
    
    try:
        config = await _target_registry_version_manager.get_version(version)
        if config is None:
            raise HTTPException(status_code=404, detail=f"Target Registry version not found: {version}")
        
        version_record = await _metadata_storage.get_target_registry_version(version)
        return {
            "version": version,
            "config": config,
            "is_active": version_record["is_active"] if version_record else False,
            "created_at": version_record["created_at"] if version_record else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get Target Registry version", version=version, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get version: {str(e)}")


@router.post("/versions", status_code=201)
async def create_target_registry_version(
    request: TargetRegistryVersionCreateRequest,
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Create a new Target Registry version.
    
    Args:
        request: Version creation request
        
    Returns:
        Created version record
    """
    if _target_registry_version_manager is None:
        raise HTTPException(status_code=503, detail="Target Registry not initialized")
    
    try:
        version_record = await _target_registry_version_manager.create_version(
            version=request.version,
            config_data=request.config.model_dump(),
            description=request.description,
        )
        return version_record
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create Target Registry version", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create version: {str(e)}")


@router.post("/versions/{version}/activate", status_code=200)
async def activate_target_registry_version(
    version: str,
    request: TargetRegistryVersionActivateRequest,
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """
    Activate a Target Registry version.
    
    Args:
        version: Version identifier to activate
        request: Activation request
        
    Returns:
        Activated version record
    """
    if _target_registry_version_manager is None:
        raise HTTPException(status_code=503, detail="Target Registry not initialized")
    
    try:
        activated_record = await _target_registry_version_manager.activate_version(
            version=version,
            activated_by=request.activated_by,
            activation_reason=request.activation_reason,
        )
        
        # Update FeatureScheduler intervals if scheduler is available
        if _feature_scheduler:
            try:
                await _feature_scheduler.update_intervals()
                logger.info(
                    "FeatureScheduler intervals updated after Target Registry activation",
                    version=version,
                    new_intervals=_feature_scheduler._intervals,
                )
            except Exception as scheduler_error:
                logger.warning(
                    "failed_to_update_scheduler_after_target_registry_activation",
                    version=version,
                    error=str(scheduler_error),
                    message="Target Registry activated but scheduler intervals not updated",
                )
        
        return activated_record
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to activate Target Registry version", version=version, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to activate version: {str(e)}")

