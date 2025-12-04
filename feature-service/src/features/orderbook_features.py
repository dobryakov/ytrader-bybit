"""
Orderbook features computation module.
"""
from typing import Dict, Optional

from src.models.orderbook_state import OrderbookState


def compute_depth_bid_top5(orderbook: Optional[OrderbookState]) -> float:
    """Compute bid depth for top 5 levels."""
    if orderbook is None:
        return 0.0
    
    return orderbook.get_depth_bid_top5()


def compute_depth_bid_top10(orderbook: Optional[OrderbookState]) -> float:
    """Compute bid depth for top 10 levels."""
    if orderbook is None:
        return 0.0
    
    return orderbook.get_depth_bid_top10()


def compute_depth_ask_top5(orderbook: Optional[OrderbookState]) -> float:
    """Compute ask depth for top 5 levels."""
    if orderbook is None:
        return 0.0
    
    return orderbook.get_depth_ask_top5()


def compute_depth_ask_top10(orderbook: Optional[OrderbookState]) -> float:
    """Compute ask depth for top 10 levels."""
    if orderbook is None:
        return 0.0
    
    return orderbook.get_depth_ask_top10()


def compute_depth_imbalance_top5(orderbook: Optional[OrderbookState]) -> float:
    """Compute orderbook imbalance for top 5 levels."""
    if orderbook is None:
        return 0.0
    
    return orderbook.get_imbalance_top5()


def compute_all_orderbook_features(
    orderbook: Optional[OrderbookState],
) -> Dict[str, float]:
    """Compute all orderbook features."""
    features = {}
    
    features["depth_bid_top5"] = compute_depth_bid_top5(orderbook)
    features["depth_bid_top10"] = compute_depth_bid_top10(orderbook)
    features["depth_ask_top5"] = compute_depth_ask_top5(orderbook)
    features["depth_ask_top10"] = compute_depth_ask_top10(orderbook)
    features["depth_imbalance_top5"] = compute_depth_imbalance_top5(orderbook)
    
    return features

