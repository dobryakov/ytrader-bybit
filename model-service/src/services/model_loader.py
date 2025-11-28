"""
Model loader service.

Loads trained models from file system, validates model files, and caches active models.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import joblib
from xgboost import XGBClassifier, XGBRegressor

from ..database.repositories.model_version_repo import ModelVersionRepository
from ..config.settings import settings
from ..config.exceptions import ModelNotFoundError, ModelLoadError
from ..config.logging import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Loads and caches trained models for inference."""

    def __init__(self):
        """Initialize model loader."""
        self.model_version_repo = ModelVersionRepository()
        self._model_cache: Dict[str, Any] = {}  # Cache: {version: model}
        self._model_metadata_cache: Dict[str, Dict[str, Any]] = {}  # Cache: {version: metadata}

    async def load_active_model(
        self,
        strategy_id: Optional[str] = None,
        force_reload: bool = False,
    ) -> Optional[Any]:
        """
        Load the active model for a strategy.

        Args:
            strategy_id: Trading strategy identifier (None for default strategy)
            force_reload: Force reload even if model is cached

        Returns:
            Loaded model object or None if no active model exists
        """
        # Get active model version from database
        model_version = await self.model_version_repo.get_active_by_strategy(strategy_id)
        if not model_version:
            logger.debug("No active model found", strategy_id=strategy_id)
            return None

        version = model_version["version"]
        model_type = model_version["model_type"]
        file_path = model_version["file_path"]

        # Check cache
        cache_key = f"{strategy_id or 'default'}:{version}"
        if not force_reload and cache_key in self._model_cache:
            logger.debug("Using cached model", version=version, strategy_id=strategy_id)
            return self._model_cache[cache_key]

        # Load model from file system
        try:
            model = self._load_model_from_file(file_path, model_type, version)
            if model:
                # Cache the model
                self._model_cache[cache_key] = model
                self._model_metadata_cache[cache_key] = {
                    "version": version,
                    "model_type": model_type,
                    "strategy_id": strategy_id,
                    "file_path": file_path,
                }
                logger.info("Model loaded and cached", version=version, strategy_id=strategy_id, model_type=model_type)
            return model
        except Exception as e:
            logger.error("Failed to load active model", version=version, strategy_id=strategy_id, error=str(e), exc_info=True)
            raise ModelLoadError(f"Failed to load model {version}: {e}") from e

    def _load_model_from_file(
        self,
        file_path: str,
        model_type: str,
        version: str,
    ) -> Any:
        """
        Load a model from file system.

        Args:
            file_path: Path to model file
            model_type: Type of model ('xgboost', 'random_forest', etc.)
            version: Model version identifier

        Returns:
            Loaded model object

        Raises:
            ModelNotFoundError: If model file does not exist
            ModelLoadError: If loading fails
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.error("Model file not found", file_path=file_path, version=version)
            raise ModelNotFoundError(f"Model file not found: {file_path}")

        try:
            if model_type == "xgboost":
                # Determine task type from file extension or default to classification
                # XGBoost models saved as .json
                if file_path.endswith(".json"):
                    # Try classification first (most common for trading signals)
                    try:
                        model = XGBClassifier()
                        model.load_model(file_path)
                        logger.debug("Loaded XGBoost classifier", file_path=file_path, version=version)
                        return model
                    except Exception:
                        # Try regression if classification fails
                        model = XGBRegressor()
                        model.load_model(file_path)
                        logger.debug("Loaded XGBoost regressor", file_path=file_path, version=version)
                        return model
                else:
                    # Fallback: try classification
                    model = XGBClassifier()
                    model.load_model(file_path)
                    logger.debug("Loaded XGBoost model", file_path=file_path, version=version)
                    return model
            else:
                # Load scikit-learn models using joblib
                model = joblib.load(file_path)
                logger.debug("Loaded scikit-learn model", file_path=file_path, version=version, model_type=model_type)
                return model

        except Exception as e:
            logger.error("Failed to load model from file", file_path=file_path, model_type=model_type, error=str(e), exc_info=True)
            raise ModelLoadError(f"Failed to load model from {file_path}: {e}") from e

    async def load_model_by_version(
        self,
        version: str,
        force_reload: bool = False,
    ) -> Optional[Any]:
        """
        Load a model by version identifier.

        Args:
            version: Model version identifier (e.g., 'v1', 'v2.1')
            force_reload: Force reload even if model is cached

        Returns:
            Loaded model object or None if version not found
        """
        # Check cache
        if not force_reload and version in self._model_cache:
            logger.debug("Using cached model", version=version)
            return self._model_cache[version]

        # Get model version from database
        model_version = await self.model_version_repo.get_by_version(version)
        if not model_version:
            logger.warning("Model version not found", version=version)
            return None

        model_type = model_version["model_type"]
        file_path = model_version["file_path"]

        # Load model from file system
        try:
            model = self._load_model_from_file(file_path, model_type, version)
            if model:
                # Cache the model
                self._model_cache[version] = model
                self._model_metadata_cache[version] = {
                    "version": version,
                    "model_type": model_type,
                    "strategy_id": model_version.get("strategy_id"),
                    "file_path": file_path,
                }
                logger.info("Model loaded and cached", version=version, model_type=model_type)
            return model
        except Exception as e:
            logger.error("Failed to load model by version", version=version, error=str(e), exc_info=True)
            raise ModelLoadError(f"Failed to load model {version}: {e}") from e

    def clear_cache(self, version: Optional[str] = None) -> None:
        """
        Clear model cache.

        Args:
            version: Specific version to clear (None to clear all)
        """
        if version:
            if version in self._model_cache:
                del self._model_cache[version]
            if version in self._model_metadata_cache:
                del self._model_metadata_cache[version]
            logger.info("Cleared model cache", version=version)
        else:
            self._model_cache.clear()
            self._model_metadata_cache.clear()
            logger.info("Cleared all model cache")

    def get_cached_model(self, version: str) -> Optional[Any]:
        """
        Get a cached model without loading.

        Args:
            version: Model version identifier

        Returns:
            Cached model object or None if not cached
        """
        return self._model_cache.get(version)

    def get_cached_metadata(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Get cached model metadata.

        Args:
            version: Model version identifier

        Returns:
            Model metadata dictionary or None if not cached
        """
        return self._model_metadata_cache.get(version)

    def validate_model_file(self, file_path: str, model_type: str) -> bool:
        """
        Validate that a model file exists and can be loaded.

        Args:
            file_path: Path to model file
            model_type: Type of model

        Returns:
            True if model file is valid, False otherwise
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            logger.warning("Model file does not exist", file_path=file_path)
            return False

        try:
            # Try to load the model to validate it
            self._load_model_from_file(file_path, model_type, "validation")
            logger.debug("Model file validated", file_path=file_path, model_type=model_type)
            return True
        except Exception as e:
            logger.warning("Model file validation failed", file_path=file_path, model_type=model_type, error=str(e))
            return False


# Global model loader instance
model_loader = ModelLoader()

