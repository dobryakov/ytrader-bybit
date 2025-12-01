"""Portfolio manager service for aggregate portfolio metrics and caching."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from ..config.logging import get_logger
from ..config.settings import settings
from ..models import PortfolioByAsset, PortfolioExposure, PortfolioMetrics, PortfolioPnL, Position
from .position_manager import PositionManager

logger = get_logger(__name__)


class PortfolioMetricsCacheEntry:
    """Simple in-memory cache entry for portfolio metrics."""

    def __init__(self, metrics: PortfolioMetrics, expires_at: datetime) -> None:
        self.metrics = metrics
        self.expires_at = expires_at

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at


class PortfolioManager:
    """Service for calculating portfolio metrics with in-memory caching."""

    def __init__(self, position_manager: Optional[PositionManager] = None) -> None:
        self._position_manager = position_manager or PositionManager()
        self._cache: Dict[str, PortfolioMetricsCacheEntry] = {}

    # === Cache management ==================================================

    def _cache_key(self, asset_filter: Optional[str], include_positions: bool) -> str:
        asset_part = asset_filter.upper() if asset_filter else "ALL"
        include_part = "WITH_POS" if include_positions else "NO_POS"
        return f"{asset_part}:{include_part}"

    def invalidate_cache(self) -> None:
        """Invalidate all cached portfolio metrics (called on position updates)."""
        self._cache.clear()
        logger.debug("portfolio_cache_invalidated")

    # === Public API ========================================================

    async def get_portfolio_metrics(
        self,
        include_positions: bool = False,
        asset_filter: Optional[str] = None,
    ) -> PortfolioMetrics:
        """Get portfolio metrics, using cache when valid."""
        cache_key = self._cache_key(asset_filter, include_positions)
        ttl_seconds = settings.position_manager_metrics_cache_ttl

        entry = self._cache.get(cache_key)
        if entry and not entry.is_expired:
            logger.debug("portfolio_metrics_cache_hit", cache_key=cache_key)
            return entry.metrics

        logger.debug("portfolio_metrics_cache_miss", cache_key=cache_key)
        positions = await self._position_manager.get_all_positions()

        if asset_filter:
            positions = self._position_manager.filter_by_asset(positions, asset_filter)

        metrics = self.calculate_metrics_from_positions(positions, include_positions=include_positions)

        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        self._cache[cache_key] = PortfolioMetricsCacheEntry(metrics, expires_at)

        return metrics

    async def get_total_exposure(self) -> PortfolioExposure:
        """Return total exposure-only view."""
        metrics = await self.get_portfolio_metrics(include_positions=False)
        return PortfolioExposure(
            total_exposure_usdt=metrics.total_exposure_usdt,
            calculated_at=metrics.calculated_at,
        )

    async def get_portfolio_pnl(self) -> PortfolioPnL:
        """Return PnL-only view."""
        metrics = await self.get_portfolio_metrics(include_positions=False)
        total_pnl = metrics.total_unrealized_pnl_usdt + metrics.total_realized_pnl_usdt
        return PortfolioPnL(
            total_unrealized_pnl_usdt=metrics.total_unrealized_pnl_usdt,
            total_realized_pnl_usdt=metrics.total_realized_pnl_usdt,
            total_pnl_usdt=total_pnl,
            calculated_at=metrics.calculated_at,
        )

    # === Metrics calculation ===============================================

    def calculate_metrics_from_positions(
        self,
        positions: List[Position],
        include_positions: bool = False,
    ) -> PortfolioMetrics:
        """Calculate aggregate metrics from a list of positions."""
        if not positions:
            # Empty portfolio handling
            zero = Decimal("0")
            return PortfolioMetrics(
                total_exposure_usdt=zero,
                total_unrealized_pnl_usdt=zero,
                total_realized_pnl_usdt=zero,
                portfolio_value_usdt=zero,
                open_positions_count=0,
                long_positions_count=0,
                short_positions_count=0,
                net_exposure_usdt=zero,
                by_asset={},
                calculated_at=datetime.utcnow(),
            )

        total_exposure = Decimal("0")
        total_unrealized = Decimal("0")
        total_realized = Decimal("0")
        portfolio_value = Decimal("0")
        open_positions_count = 0
        long_positions_count = 0
        short_positions_count = 0
        asset_breakdown: Dict[str, PortfolioByAsset] = {}

        for p in positions:
            if p.size != 0:
                open_positions_count += 1
                if p.size > 0:
                    long_positions_count += 1
                elif p.size < 0:
                    short_positions_count += 1

            unrealized = p.unrealized_pnl or Decimal("0")
            realized = p.realized_pnl or Decimal("0")
            total_unrealized += unrealized
            total_realized += realized

            if p.current_price is not None:
                exposure = abs(p.size * p.current_price)
                total_exposure += exposure
                portfolio_value += p.size * p.current_price
            else:
                exposure = Decimal("0")

            asset_key = p.asset
            if asset_key not in asset_breakdown:
                asset_breakdown[asset_key] = PortfolioByAsset(
                    exposure_usdt=Decimal("0"),
                    unrealized_pnl_usdt=Decimal("0"),
                    realized_pnl_usdt=Decimal("0"),
                    size=Decimal("0"),
                )

            asset_entry = asset_breakdown[asset_key]
            asset_entry.exposure_usdt += exposure
            asset_entry.unrealized_pnl_usdt += unrealized
            asset_entry.realized_pnl_usdt += realized
            asset_entry.size += p.size

        long_exposure = sum(
            (entry.exposure_usdt for entry in asset_breakdown.values() if entry.size > 0),
            Decimal("0"),
        )
        short_exposure = sum(
            (entry.exposure_usdt for entry in asset_breakdown.values() if entry.size < 0),
            Decimal("0"),
        )
        net_exposure = long_exposure - short_exposure

        metrics = PortfolioMetrics(
            total_exposure_usdt=total_exposure,
            total_unrealized_pnl_usdt=total_unrealized,
            total_realized_pnl_usdt=total_realized,
            portfolio_value_usdt=portfolio_value,
            open_positions_count=open_positions_count,
            long_positions_count=long_positions_count,
            short_positions_count=short_positions_count,
            net_exposure_usdt=net_exposure,
            by_asset=asset_breakdown,
            calculated_at=datetime.utcnow(),
        )

        # Optionally attach positions list in API layer; core metrics stay lean.
        if include_positions:
            # Attach as attribute for later serialization (not part of schema)
            setattr(metrics, "_positions", positions)

        return metrics
        

# Shared default instance used across the service (for cache invalidation, etc.)
default_portfolio_manager = PortfolioManager()

