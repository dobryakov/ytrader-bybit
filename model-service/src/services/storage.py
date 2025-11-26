"""
Model storage utilities for file system operations.

Provides file system operations for model files including save, load, and health checks.
"""

import os
import shutil
from pathlib import Path
from typing import Optional
import joblib

from ..config.settings import settings
from ..config.exceptions import ModelStorageError, ModelNotFoundError, ModelLoadError, ModelSaveError
from ..config.logging import get_logger

logger = get_logger(__name__)


class ModelStorage:
    """Manages model file storage on the file system."""

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize model storage.

        Args:
            base_path: Base path for model storage (defaults to MODEL_STORAGE_PATH)
        """
        self.base_path = Path(base_path or settings.model_storage_path)
        self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Ensure the base storage directory exists."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info("Model storage directory ready", path=str(self.base_path))
        except OSError as e:
            logger.error("Failed to create model storage directory", path=str(self.base_path), error=str(e))
            raise ModelStorageError(f"Cannot create model storage directory: {e}") from e

    def get_model_path(self, version: str, filename: str = "model.pkl") -> Path:
        """
        Get the file path for a model version.

        Args:
            version: Model version identifier (e.g., 'v1', 'v2.1')
            filename: Model filename (default: 'model.pkl')

        Returns:
            Full path to model file
        """
        version_dir = self.base_path / f"v{version.lstrip('v')}"
        return version_dir / filename

    def get_version_directory(self, version: str) -> Path:
        """
        Get the directory path for a model version.

        Args:
            version: Model version identifier (e.g., 'v1', 'v2.1')

        Returns:
            Directory path for the version
        """
        return self.base_path / f"v{version.lstrip('v')}"

    def save_model(self, model: any, version: str, filename: str = "model.pkl") -> str:
        """
        Save a model to the file system.

        Args:
            model: Model object (scikit-learn or XGBoost model)
            version: Model version identifier
            filename: Model filename (default: 'model.pkl')

        Returns:
            Full path to saved model file

        Raises:
            ModelSaveError: If saving fails
        """
        try:
            version_dir = self.get_version_directory(version)
            version_dir.mkdir(parents=True, exist_ok=True)

            model_path = version_dir / filename

            # Save using joblib for scikit-learn models
            # For XGBoost, we'll use native JSON format (handled separately)
            joblib.dump(model, model_path)

            logger.info("Model saved successfully", version=version, path=str(model_path))
            return str(model_path)
        except Exception as e:
            logger.error("Failed to save model", version=version, error=str(e), exc_info=True)
            raise ModelSaveError(f"Failed to save model: {e}") from e

    def load_model(self, version: str, filename: str = "model.pkl") -> any:
        """
        Load a model from the file system.

        Args:
            version: Model version identifier
            filename: Model filename (default: 'model.pkl')

        Returns:
            Loaded model object

        Raises:
            ModelNotFoundError: If model file does not exist
            ModelLoadError: If loading fails
        """
        model_path = self.get_model_path(version, filename)

        if not model_path.exists():
            logger.error("Model file not found", version=version, path=str(model_path))
            raise ModelNotFoundError(f"Model file not found: {model_path}")

        try:
            model = joblib.load(model_path)
            logger.info("Model loaded successfully", version=version, path=str(model_path))
            return model
        except Exception as e:
            logger.error("Failed to load model", version=version, path=str(model_path), error=str(e), exc_info=True)
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def model_exists(self, version: str, filename: str = "model.pkl") -> bool:
        """
        Check if a model file exists.

        Args:
            version: Model version identifier
            filename: Model filename (default: 'model.pkl')

        Returns:
            True if model exists, False otherwise
        """
        model_path = self.get_model_path(version, filename)
        return model_path.exists()

    def delete_model(self, version: str, filename: str = "model.pkl") -> None:
        """
        Delete a model file.

        Args:
            version: Model version identifier
            filename: Model filename (default: 'model.pkl')

        Raises:
            ModelNotFoundError: If model file does not exist
            ModelStorageError: If deletion fails
        """
        model_path = self.get_model_path(version, filename)

        if not model_path.exists():
            raise ModelNotFoundError(f"Model file not found: {model_path}")

        try:
            model_path.unlink()
            logger.info("Model deleted successfully", version=version, path=str(model_path))

            # Clean up version directory if empty
            version_dir = self.get_version_directory(version)
            try:
                if version_dir.exists() and not any(version_dir.iterdir()):
                    version_dir.rmdir()
                    logger.info("Empty version directory removed", version=version)
            except OSError:
                # Directory not empty or other error, ignore
                pass
        except Exception as e:
            logger.error("Failed to delete model", version=version, path=str(model_path), error=str(e), exc_info=True)
            raise ModelStorageError(f"Failed to delete model: {e}") from e

    def delete_version_directory(self, version: str) -> None:
        """
        Delete an entire version directory and all its contents.

        Args:
            version: Model version identifier

        Raises:
            ModelStorageError: If deletion fails
        """
        version_dir = self.get_version_directory(version)

        if not version_dir.exists():
            logger.warning("Version directory does not exist", version=version, path=str(version_dir))
            return

        try:
            shutil.rmtree(version_dir)
            logger.info("Version directory deleted successfully", version=version, path=str(version_dir))
        except Exception as e:
            logger.error("Failed to delete version directory", version=version, path=str(version_dir), error=str(e), exc_info=True)
            raise ModelStorageError(f"Failed to delete version directory: {e}") from e

    def get_disk_space_info(self) -> dict:
        """
        Get disk space information for the model storage directory.

        Returns:
            Dictionary with disk space information (total, used, free, available)
        """
        try:
            stat = shutil.disk_usage(self.base_path)
            return {
                "total": stat.total,
                "used": stat.used,
                "free": stat.free,
                "available": stat.free,  # Same as free on most systems
            }
        except Exception as e:
            logger.error("Failed to get disk space info", error=str(e), exc_info=True)
            raise ModelStorageError(f"Failed to get disk space info: {e}") from e

    def health_check(self) -> dict:
        """
        Perform health check on model storage.

        Returns:
            Dictionary with health check results (healthy, path, writable, disk_space)
        """
        try:
            # Check directory exists and is writable
            if not self.base_path.exists():
                return {
                    "healthy": False,
                    "path": str(self.base_path),
                    "writable": False,
                    "error": "Directory does not exist",
                }

            if not os.access(self.base_path, os.W_OK):
                return {
                    "healthy": False,
                    "path": str(self.base_path),
                    "writable": False,
                    "error": "Directory is not writable",
                }

            # Get disk space
            disk_info = self.get_disk_space_info()

            return {
                "healthy": True,
                "path": str(self.base_path),
                "writable": True,
                "disk_space": disk_info,
            }
        except Exception as e:
            logger.error("Model storage health check failed", error=str(e), exc_info=True)
            return {
                "healthy": False,
                "path": str(self.base_path),
                "writable": False,
                "error": str(e),
            }


# Global model storage instance
model_storage = ModelStorage()

