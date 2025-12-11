"""
Test fixtures for Feature Registry configurations.
"""
from typing import Dict, Any


def get_valid_feature_registry_config() -> Dict[str, Any]:
    """
    Get a valid Feature Registry configuration.
    
    Returns:
        Valid Feature Registry configuration dict
    """
    return {
        "version": "1.0.0",
        "features": [
            {
                "name": "mid_price",
                "input_sources": ["orderbook"],
                "lookback_window": "0s",
                "lookahead_forbidden": True,
                "max_lookback_days": 0,
                "data_sources": [
                    {
                        "source": "orderbook",
                        "timestamp_required": True
                    }
                ]
            },
            {
                "name": "returns_1s",
                "input_sources": ["kline"],
                "lookback_window": "1s",
                "lookahead_forbidden": True,
                "max_lookback_days": 1,
                "data_sources": [
                    {
                        "source": "kline",
                        "timestamp_required": True
                    }
                ]
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
                        "timestamp_required": True
                    }
                ]
            }
        ]
    }


def get_invalid_feature_registry_config_missing_version() -> Dict[str, Any]:
    """
    Get an invalid Feature Registry configuration (missing version).
    
    Returns:
        Invalid Feature Registry configuration dict
    """
    return {
        "features": [
            {
                "name": "mid_price",
                "input_sources": ["orderbook"],
                "lookback_window": "0s",
                "lookahead_forbidden": True,
                "max_lookback_days": 0
            }
        ]
    }


def get_invalid_feature_registry_config_missing_features() -> Dict[str, Any]:
    """
    Get an invalid Feature Registry configuration (missing features).
    
    Returns:
        Invalid Feature Registry configuration dict
    """
    return {
        "version": "1.0.0"
    }


def get_invalid_feature_registry_config_invalid_lookback() -> Dict[str, Any]:
    """
    Get an invalid Feature Registry configuration (invalid lookback_window format).
    
    Returns:
        Invalid Feature Registry configuration dict
    """
    return {
        "version": "1.0.0",
        "features": [
            {
                "name": "mid_price",
                "input_sources": ["orderbook"],
                "lookback_window": "invalid",
                "lookahead_forbidden": True,
                "max_lookback_days": 0
            }
        ]
    }


def get_data_leakage_feature_registry_config() -> Dict[str, Any]:
    """
    Get a Feature Registry configuration with data leakage (lookahead_forbidden=False).
    
    Returns:
        Feature Registry configuration dict with data leakage
    """
    return {
        "version": "1.0.0",
        "features": [
            {
                "name": "future_price",
                "input_sources": ["kline"],
                "lookback_window": "1m",
                "lookahead_forbidden": False,  # Data leakage: allows future data
                "max_lookback_days": 1,
                "data_sources": [
                    {
                        "source": "kline",
                        "timestamp_required": True
                    }
                ]
            }
        ]
    }


def get_data_leakage_feature_registry_config_negative_lookback() -> Dict[str, Any]:
    """
    Get a Feature Registry configuration with data leakage (negative lookback_window).
    
    Returns:
        Feature Registry configuration dict with data leakage
    """
    return {
        "version": "1.0.0",
        "features": [
            {
                "name": "future_returns",
                "input_sources": ["kline"],
                "lookback_window": "-1m",  # Data leakage: negative lookback
                "lookahead_forbidden": True,
                "max_lookback_days": 1,
                "data_sources": [
                    {
                        "source": "kline",
                        "timestamp_required": True
                    }
                ]
            }
        ]
    }


def get_data_leakage_feature_registry_config_excessive_lookback() -> Dict[str, Any]:
    """
    Get a Feature Registry configuration with data leakage (excessive max_lookback_days).
    
    Returns:
        Feature Registry configuration dict with data leakage
    """
    return {
        "version": "1.0.0",
        "features": [
            {
                "name": "long_term_returns",
                "input_sources": ["kline"],
                "lookback_window": "1m",
                "lookahead_forbidden": True,
                "max_lookback_days": 365,  # Data leakage: excessive lookback
                "data_sources": [
                    {
                        "source": "kline",
                        "timestamp_required": True
                    }
                ]
            }
        ]
    }


def get_feature_registry_config_without_temporal_boundaries() -> Dict[str, Any]:
    """
    Get a Feature Registry configuration without temporal boundaries validation.
    
    Returns:
        Feature Registry configuration dict without temporal boundaries
    """
    return {
        "version": "1.0.0",
        "features": [
            {
                "name": "mid_price",
                "input_sources": ["orderbook"],
                # Missing lookback_window, lookahead_forbidden, max_lookback_days
                "data_sources": [
                    {
                        "source": "orderbook",
                        "timestamp_required": True
                    }
                ]
            }
        ]
    }

