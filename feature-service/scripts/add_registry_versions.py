#!/usr/bin/env python3
"""
Скрипт для добавления новых версий регистров в БД.

Использование:
    python scripts/add_registry_versions.py
"""
import asyncio
import yaml
from pathlib import Path
import sys

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.logging import get_logger
from src.storage.metadata_storage import MetadataStorage
from src.services.feature_registry_version_manager import FeatureRegistryVersionManager
from src.services.target_registry_version_manager import TargetRegistryVersionManager

logger = get_logger(__name__)


async def add_feature_registry_version(version: str, file_path: Path) -> None:
    """Добавить версию feature registry в БД."""
    logger.info("Adding feature registry version", version=version, file_path=str(file_path))
    
    # Загружаем конфигурацию из файла
    with open(file_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    if not config_data:
        raise ValueError(f"Feature Registry file is empty: {file_path}")
    
    # Создаем менеджер версий
    metadata_storage = MetadataStorage()
    await metadata_storage.initialize()
    
    version_manager = FeatureRegistryVersionManager(
        metadata_storage=metadata_storage,
        versions_dir=config.feature_registry_versions_dir,
    )
    
    # Добавляем версию в БД
    try:
        version_record = await version_manager.create_version(
            version=version,
            config_data=config_data,
            created_by="script",
        )
        logger.info("Feature registry version added successfully", version=version, record=version_record)
    except ValueError as e:
        if "already exists" in str(e):
            logger.warning("Feature registry version already exists", version=version)
        else:
            raise
    finally:
        await metadata_storage.close()


async def add_target_registry_version(version: str, file_path: Path) -> None:
    """Добавить версию target registry в БД."""
    logger.info("Adding target registry version", version=version, file_path=str(file_path))
    
    # Загружаем конфигурацию из файла
    with open(file_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    
    if not yaml_data:
        raise ValueError(f"Target Registry file is empty: {file_path}")
    
    config_data = yaml_data.get("config", yaml_data)
    description = yaml_data.get("description")
    
    # Создаем менеджер версий
    metadata_storage = MetadataStorage()
    await metadata_storage.initialize()
    
    version_manager = TargetRegistryVersionManager(
        metadata_storage=metadata_storage,
        versions_dir=config.feature_registry_versions_dir,
    )
    
    # Добавляем версию в БД
    try:
        version_record = await version_manager.create_version(
            version=version,
            config_data=config_data,
            created_by="script",
            description=description,
        )
        logger.info("Target registry version added successfully", version=version, record=version_record)
    except ValueError as e:
        if "already exists" in str(e):
            logger.warning("Target registry version already exists", version=version)
        else:
            raise
    finally:
        await metadata_storage.close()


async def main():
    """Основная функция."""
    versions_dir = Path(config.feature_registry_versions_dir)
    
    # Добавляем feature registry v1.7.3
    feature_registry_file = versions_dir / "feature_registry_v1.7.3.yaml"
    if feature_registry_file.exists():
        await add_feature_registry_version("1.7.3", feature_registry_file)
    else:
        logger.error("Feature registry file not found", file_path=str(feature_registry_file))
        return
    
    # Добавляем target registry v1.7.3
    target_registry_file = versions_dir / "target_registry_v1.7.3.yaml"
    if target_registry_file.exists():
        await add_target_registry_version("1.7.3", target_registry_file)
    else:
        logger.error("Target registry file not found", file_path=str(target_registry_file))
        return
    
    logger.info("All registry versions added successfully")


if __name__ == "__main__":
    asyncio.run(main())

