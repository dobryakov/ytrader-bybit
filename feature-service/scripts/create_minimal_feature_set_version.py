#!/usr/bin/env python3
"""
Script to create and activate Feature Registry version 1.2.0 for minimal feature set.

This script:
1. Loads current active version config
2. Prepares updated config with minimal feature set (7 features)
3. Creates new version 1.2.0 via API
4. Activates version 1.2.0 via API
"""
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from src.config import config


def load_current_config() -> Dict[str, Any]:
    """Load current active version config from file."""
    config_path = Path(config.feature_registry_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Feature Registry config not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_minimal_config(current_config: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare updated config with minimal feature set."""
    # Features to keep (minimal set)
    minimal_features = [
        {
            "name": "returns_5m",
            "input_sources": ["kline"],
            "lookback_window": "5m",
            "lookahead_forbidden": True,
            "max_lookback_days": 1,
            "data_sources": [
                {
                    "source": "kline",
                    "timestamp_required": True,
                }
            ],
        },
        {
            "name": "volatility_5m",
            "input_sources": ["kline"],
            "lookback_window": "5m",
            "lookahead_forbidden": True,
            "max_lookback_days": 1,
            "data_sources": [
                {
                    "source": "kline",
                    "timestamp_required": True,
                }
            ],
        },
        {
            "name": "rsi_14",
            "input_sources": ["kline"],
            "lookback_window": "14m",
            "lookahead_forbidden": True,
            "max_lookback_days": 1,
            "data_sources": [
                {
                    "source": "kline",
                    "timestamp_required": True,
                }
            ],
        },
        {
            "name": "ema_21",
            "input_sources": ["kline"],
            "lookback_window": "21m",
            "lookahead_forbidden": True,
            "max_lookback_days": 1,
            "data_sources": [
                {
                    "source": "kline",
                    "timestamp_required": True,
                }
            ],
        },
        {
            "name": "price_ema21_ratio",
            "input_sources": ["kline"],
            "lookback_window": "21m",
            "lookahead_forbidden": True,
            "max_lookback_days": 1,
            "data_sources": [
                {
                    "source": "kline",
                    "timestamp_required": True,
                }
            ],
        },
        {
            "name": "volume_ratio_20",
            "input_sources": ["kline"],
            "lookback_window": "20m",
            "lookahead_forbidden": True,
            "max_lookback_days": 1,
            "data_sources": [
                {
                    "source": "kline",
                    "timestamp_required": True,
                }
            ],
        },
        {
            "name": "funding_rate",
            "input_sources": ["funding"],
            "lookback_window": "0s",
            "lookahead_forbidden": True,
            "max_lookback_days": 0,
            "data_sources": [
                {
                    "source": "funding",
                    "timestamp_required": True,
                }
            ],
        },
    ]
    
    # Create new config
    new_config = {
        "version": "1.2.0",
        "features": minimal_features,
        "_comment": "Minimal feature set optimized for 5-15 minute prediction horizon, threshold 0.001-0.002, all features from backfillable sources (klines + funding). Removed features requiring historical trades/orderbook/ticker data unavailable via REST API backfilling",
    }
    
    return new_config


def create_version_via_api(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create new version via API."""
    port = getattr(config, 'feature_service_port', 4900)
    api_url = f"http://localhost:{port}/feature-registry/versions"
    api_key = getattr(config, 'ws_gateway_api_key', None) or "test-key"
    
    request_data = {
        "version": "1.2.0",
        "config": config_data,
    }
    
    with httpx.Client() as client:
        response = client.post(
            api_url,
            json=request_data,
            headers={"X-API-Key": api_key},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


def activate_version_via_api(version: str) -> Dict[str, Any]:
    """Activate version via API."""
    port = getattr(config, 'feature_service_port', 4900)
    api_url = f"http://localhost:{port}/feature-registry/versions/{version}/activate"
    api_key = getattr(config, 'ws_gateway_api_key', None) or "test-key"
    
    request_data = {
        "acknowledge_breaking_changes": True,
        "activation_reason": "Minimal feature set: removed features requiring unavailable backfill data, added 7 features optimized for 5-15 minute prediction horizon",
        "activated_by": "system",
    }
    
    with httpx.Client() as client:
        response = client.post(
            api_url,
            json=request_data,
            headers={"X-API-Key": api_key},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()


def main():
    """Main function."""
    print("Loading current config...")
    current_config = load_current_config()
    print(f"Current version: {current_config.get('version', 'unknown')}")
    
    print("Preparing minimal feature set config...")
    new_config = prepare_minimal_config(current_config)
    print(f"New version: {new_config['version']}")
    print(f"Features count: {len(new_config['features'])}")
    
    print("Creating version 1.2.0 via API...")
    try:
        version_record = create_version_via_api(new_config)
        print(f"Version created: {version_record.get('version')}")
        print(f"File path: {version_record.get('file_path')}")
    except Exception as e:
        print(f"Error creating version: {e}")
        return 1
    
    print("Activating version 1.2.0 via API...")
    try:
        activated_record = activate_version_via_api("1.2.0")
        print(f"Version activated: {activated_record.get('version')}")
        print(f"Is active: {activated_record.get('is_active')}")
        if "breaking_changes" in activated_record:
            print(f"Breaking changes: {activated_record['breaking_changes']}")
    except Exception as e:
        print(f"Error activating version: {e}")
        return 1
    
    print("Success!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

