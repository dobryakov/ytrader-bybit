#!/usr/bin/env python3
"""Check active Feature Registry version and features."""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.services.feature_registry_version_manager import FeatureRegistryVersionManager
from src.services.metadata_storage import MetadataStorage
from src.config import config


async def main():
    """Check active Feature Registry."""
    storage = MetadataStorage()
    await storage.initialize()
    
    manager = FeatureRegistryVersionManager(
        metadata_storage=storage,
        versions_dir=config.feature_registry_versions_dir,
    )
    
    cfg = await manager.load_active_version()
    features = [f['name'] for f in cfg.get('features', [])]
    
    print(f"Version: {cfg.get('version')}")
    print(f"Total features: {len(features)}")
    print(f"\nChecking specific features:")
    print(f"  returns_3m: {'returns_3m' in features}")
    print(f"  returns_5m: {'returns_5m' in features}")
    print(f"  volatility_10m: {'volatility_10m' in features}")
    print(f"  volatility_15m: {'volatility_15m' in features}")
    
    print(f"\nAll features (first 20):")
    for i, name in enumerate(sorted(features)[:20], 1):
        print(f"  {i}. {name}")
    
    # Check if these features are in the registry
    target_features = ['returns_3m', 'returns_5m', 'volatility_10m', 'volatility_15m']
    missing = [f for f in target_features if f not in features]
    if missing:
        print(f"\n⚠️  Missing features: {missing}")
    else:
        print(f"\n✅ All target features are in the registry")


if __name__ == "__main__":
    asyncio.run(main())

