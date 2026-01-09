"""Position-related API routes for Position Manager."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from ...config.logging import get_logger
from ...models import Position, PositionSnapshot
from ...services.position_manager import PositionManager
from ...utils.tracing import get_or_create_trace_id
from ..middleware.auth import api_key_auth


logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["positions"])


def get_position_manager() -> PositionManager:
    return PositionManager()


@router.get(
    "/positions",
    dependencies=[Depends(api_key_auth)],
)
async def list_positions(
    asset: Optional[str] = Query(None, description="Filter by trading pair"),
    mode: Optional[str] = Query(None, description="Filter by trading mode (one-way, hedge)"),
    size_min: Optional[Decimal] = Query(None, description="Minimum position size filter"),
    size_max: Optional[Decimal] = Query(None, description="Maximum position size filter"),
    position_manager: PositionManager = Depends(get_position_manager),
):
    """List positions with optional filtering."""
    trace_id = get_or_create_trace_id()
    logger.info(
        "position_list_request",
        asset=asset,
        mode=mode,
        size_min=str(size_min) if size_min is not None else None,
        size_max=str(size_max) if size_max is not None else None,
        trace_id=trace_id,
    )

    # Mode validation
    if mode is not None:
        mode_lower = mode.lower()
        if mode_lower not in {"one-way", "hedge"}:
            raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")
    else:
        mode_lower = None

    try:
        # Get all active positions (by default, list_positions returns only active)
        positions = await position_manager.get_all_active_positions()
        positions = position_manager.filter_by_asset(positions, asset)
        positions = position_manager.filter_by_mode(positions, mode_lower)
        positions = position_manager.filter_by_size(positions, size_min, size_max)

        positions_data = [serialize_position_with_features(p, position_manager) for p in positions]

        logger.info("position_list_completed", count=len(positions_data), trace_id=trace_id)
        return JSONResponse(
            status_code=200,
            content={
                "positions": positions_data,
                "count": len(positions_data),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("position_list_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve positions") from e


@router.get(
    "/positions/{asset}",
    dependencies=[Depends(api_key_auth)],
)
async def get_position_by_asset(
    asset: str,
    mode: str = Query("one-way", description="Trading mode (one-way, hedge)"),
    position_manager: PositionManager = Depends(get_position_manager),
):
    """Get position for a specific asset."""
    trace_id = get_or_create_trace_id()
    logger.info("position_get_request", asset=asset, mode=mode, trace_id=trace_id)

    mode_lower = mode.lower()
    if mode_lower not in {"one-way", "hedge"}:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")

    try:
        position = await position_manager.get_active_position(asset, mode_lower)
        if position is None:
            logger.warning("position_not_found", asset=asset, mode=mode, trace_id=trace_id)
            raise HTTPException(
                status_code=404,
                detail=f"Position not found for asset: {asset}, mode: {mode}",
            )

        data = serialize_position_with_features(position, position_manager)
        logger.info(
            "position_get_completed",
            asset=asset,
            mode=mode,
            created_at=data.get("created_at"),
            position_created_at=str(position.created_at) if position.created_at else None,
            trace_id=trace_id,
        )
        return JSONResponse(status_code=200, content=data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("position_get_failed", asset=asset, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve position") from e


@router.post(
    "/positions/{asset}/validate",
    dependencies=[Depends(api_key_auth)],
)
async def validate_position(
    asset: str,
    mode: str = Query("one-way", description="Trading mode (one-way, hedge)"),
    fix_discrepancies: bool = Query(
        True,
        description="Whether to automatically fix detected discrepancies",
    ),
    position_manager: PositionManager = Depends(get_position_manager),
):
    """Trigger position validation and optional correction."""
    trace_id = get_or_create_trace_id()
    logger.info(
        "position_validate_request",
        asset=asset,
        mode=mode,
        fix_discrepancies=fix_discrepancies,
        trace_id=trace_id,
    )

    mode_lower = mode.lower()
    if mode_lower not in {"one-way", "hedge"}:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")

    try:
        is_valid, error_message, updated_position = await position_manager.validate_position(
            asset, mode_lower, fix_discrepancies=fix_discrepancies, trace_id=trace_id
        )
        response = {
            "is_valid": is_valid,
            "error_message": error_message,
            "updated_position": serialize_position_with_features(updated_position, position_manager)
            if updated_position
            else None,
        }
        logger.info(
            "position_validate_completed",
            asset=asset,
            mode=mode,
            is_valid=is_valid,
            trace_id=trace_id,
        )
        return JSONResponse(status_code=200, content=response)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("position_validate_failed", asset=asset, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to validate position") from e


@router.post(
    "/positions/{asset}/snapshot",
    dependencies=[Depends(api_key_auth)],
)
async def create_snapshot(
    asset: str,
    mode: str = Query("one-way", description="Trading mode (one-way, hedge)"),
    position_manager: PositionManager = Depends(get_position_manager),
):
    """Manually trigger snapshot creation for a position."""
    trace_id = get_or_create_trace_id()
    logger.info("position_snapshot_request", asset=asset, mode=mode, trace_id=trace_id)

    mode_lower = mode.lower()
    if mode_lower not in {"one-way", "hedge"}:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")

    try:
        position = await position_manager.get_active_position(asset, mode_lower)
        if position is None:
            logger.warning("position_not_found_for_snapshot", asset=asset, mode=mode, trace_id=trace_id)
            raise HTTPException(
                status_code=404,
                detail=f"Position not found for asset: {asset}, mode: {mode}",
            )

        snapshot = await position_manager.create_position_snapshot(position)
        logger.info(
            "position_snapshot_completed",
            asset=asset,
            mode=mode,
            snapshot_id=str(snapshot.id),
            trace_id=trace_id,
        )
        return JSONResponse(status_code=201, content=serialize_snapshot(snapshot))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("position_snapshot_failed", asset=asset, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create position snapshot") from e


@router.get(
    "/positions/{asset}/snapshots",
    dependencies=[Depends(api_key_auth)],
)
async def list_snapshots(
    asset: str,
    mode: str = Query("one-way", description="Trading mode (one-way, hedge)"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    position_manager: PositionManager = Depends(get_position_manager),
):
    """Return historical snapshots for a given asset/mode, paginated."""
    trace_id = get_or_create_trace_id()
    logger.info(
        "position_snapshots_list_request",
        asset=asset,
        mode=mode,
        limit=limit,
        offset=offset,
        trace_id=trace_id,
    )

    mode_lower = mode.lower()
    if mode_lower not in {"one-way", "hedge"}:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")

    try:
        position = await position_manager.get_active_position(asset, mode_lower)
        if position is None:
            logger.warning(
                "position_not_found_for_snapshot_history",
                asset=asset,
                mode=mode,
                trace_id=trace_id,
            )
            raise HTTPException(
                status_code=404,
                detail=f"Position not found for asset: {asset}, mode: {mode}",
            )

        snapshots = await position_manager.get_position_snapshots(
            position_id=position.id,
            limit=limit,
            offset=offset,
        )
        payload = [serialize_snapshot(s) for s in snapshots]

        logger.info(
            "position_snapshots_list_completed",
            asset=asset,
            mode=mode,
            count=len(payload),
            trace_id=trace_id,
        )
        return JSONResponse(
            status_code=200,
            content={
                "snapshots": payload,
                "count": len(payload),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "position_snapshots_list_failed",
            asset=asset,
            mode=mode,
            error=str(e),
            trace_id=trace_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve position snapshots") from e


@router.get(
    "/positions/{asset}/history",
    dependencies=[Depends(api_key_auth)],
)
async def get_position_history(
    asset: str,
    mode: str = Query("one-way", description="Trading mode (one-way, hedge)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    position_manager: PositionManager = Depends(get_position_manager),
):
    """Get history of closed positions for an asset/mode pair."""
    trace_id = get_or_create_trace_id()
    logger.info(
        "position_history_request",
        asset=asset,
        mode=mode,
        limit=limit,
        offset=offset,
        trace_id=trace_id,
    )

    mode_lower = mode.lower()
    if mode_lower not in {"one-way", "hedge"}:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")

    try:
        closed_positions = await position_manager.get_position_history(
            asset=asset,
            mode=mode_lower,
            limit=limit,
            offset=offset,
        )

        closed_positions_data = [serialize_position_with_features(cp, position_manager) for cp in closed_positions]

        logger.info(
            "position_history_completed",
            count=len(closed_positions_data),
            trace_id=trace_id,
        )
        return JSONResponse(
            status_code=200,
            content={
                "positions": closed_positions_data,
                "count": len(closed_positions_data),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "position_history_failed",
            error=str(e),
            trace_id=trace_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve position history") from e


@router.get(
    "/positions/closed",
    dependencies=[Depends(api_key_auth)],
)
async def list_closed_positions(
    asset: Optional[str] = Query(None, description="Filter by trading pair"),
    mode: Optional[str] = Query(None, description="Filter by trading mode (one-way, hedge)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    position_manager: PositionManager = Depends(get_position_manager),
):
    """List all closed positions (across all assets)."""
    trace_id = get_or_create_trace_id()
    logger.info(
        "closed_positions_list_request",
        asset=asset,
        mode=mode,
        limit=limit,
        offset=offset,
        trace_id=trace_id,
    )

    mode_lower = mode.lower() if mode else None
    if mode_lower is not None and mode_lower not in {"one-way", "hedge"}:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'one-way' or 'hedge'")

    try:
        # Get all closed positions by querying each asset/mode combination
        # For simplicity, if asset is provided, use get_position_history
        if asset:
            if mode_lower is None:
                mode_lower = "one-way"
            closed_positions = await position_manager.get_position_history(
            asset=asset,
                mode=mode_lower,
            limit=limit,
            offset=offset,
        )
        else:
            # For all assets, we'd need a new method or query all assets
            # For now, return empty if no asset specified
            closed_positions = []

        closed_positions_data = [serialize_position_with_features(cp, position_manager) for cp in closed_positions]

        logger.info(
            "closed_positions_list_completed",
            count=len(closed_positions_data),
            trace_id=trace_id,
        )
        return JSONResponse(
            status_code=200,
            content={
                "positions": closed_positions_data,
                "count": len(closed_positions_data),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "closed_positions_list_failed",
            error=str(e),
            trace_id=trace_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve closed positions") from e


@router.get(
    "/positions/{position_id}",
    dependencies=[Depends(api_key_auth)],
)
async def get_position_by_id(
    position_id: str,
    position_manager: PositionManager = Depends(get_position_manager),
):
    """Get a position by ID (active or closed)."""
    trace_id = get_or_create_trace_id()
    logger.info(
        "position_get_by_id_request",
        position_id=position_id,
        trace_id=trace_id,
    )

    try:
        from uuid import UUID
        pos_id = UUID(position_id)
        
        # Try to get as closed position first
        closed_position = await position_manager.get_closed_position_by_id(pos_id)
        if closed_position:
            position_data = serialize_position_with_features(closed_position, position_manager)
            logger.info(
                "position_get_by_id_completed",
                position_id=position_id,
                is_closed=True,
            trace_id=trace_id,
        )
            return JSONResponse(status_code=200, content=position_data)
        
        # If not found as closed, try to find any position by ID (active or closed)
        position = await position_manager.get_position_by_id(pos_id)
        if position:
            position_data = serialize_position_with_features(position, position_manager)
            logger.info(
                "position_get_by_id_completed",
                position_id=position_id,
                is_closed=position.is_closed,
                trace_id=trace_id,
            )
            return JSONResponse(status_code=200, content=position_data)
        
        logger.warning(
            "position_not_found_by_id",
            position_id=position_id,
            trace_id=trace_id,
        )
        raise HTTPException(status_code=404, detail="Position not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid position ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "position_get_by_id_failed",
            position_id=position_id,
            error=str(e),
            trace_id=trace_id,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve position") from e


def serialize_position_with_features(
    position: Position,
    manager: PositionManager,
) -> dict:
    """Serialize position including ML features and ISO timestamps."""
    if position is None:
        return None

    unrealized_pct = manager.calculate_unrealized_pnl_pct(position)
    time_held = manager.calculate_time_held_minutes(position)

    data = {
        "id": str(position.id),
        "asset": position.asset,
        "mode": position.mode,
        "size": str(position.size),
        "average_entry_price": str(position.average_entry_price)
        if position.average_entry_price is not None
        else None,
        "current_price": str(position.current_price) if position.current_price is not None else None,
        "unrealized_pnl": str(position.unrealized_pnl),
        "realized_pnl": str(position.realized_pnl),
        "long_size": str(position.long_size) if position.long_size is not None else None,
        "short_size": str(position.short_size) if position.short_size is not None else None,
        "long_avg_price": str(position.long_avg_price) if position.long_avg_price is not None else None,
        "short_avg_price": str(position.short_avg_price) if position.short_avg_price is not None else None,
        "leverage": str(position.leverage) if position.leverage is not None else None,
        "position_value": str(position.position_value) if position.position_value is not None else None,
        "liq_price": str(position.liq_price) if position.liq_price is not None else None,
        "bust_price": str(position.bust_price) if position.bust_price is not None else None,
        "take_profit": str(position.take_profit) if position.take_profit is not None else None,
        "stop_loss": str(position.stop_loss) if position.stop_loss is not None else None,
        "cum_realised_pnl": str(position.cum_realised_pnl) if position.cum_realised_pnl is not None else None,
        "cum_unrealised_pnl": str(position.cum_unrealised_pnl) if position.cum_unrealised_pnl is not None else None,
        "total_fees": str(position.total_fees),
        "opening_fees": str(position.opening_fees) if position.opening_fees is not None else None,
        "closing_fees": str(position.closing_fees) if position.closing_fees is not None else None,
        "margin_used": str(position.margin_used) if position.margin_used is not None else None,
        "available_margin": str(position.available_margin) if position.available_margin is not None else None,
        "maintenance_margin": str(position.maintenance_margin) if position.maintenance_margin is not None else None,
        "max_size": str(position.max_size) if position.max_size is not None else None,
        "min_size": str(position.min_size) if position.min_size is not None else None,
        "total_volume_traded": str(position.total_volume_traded),
        "first_entry_price": str(position.first_entry_price) if position.first_entry_price is not None else None,
        "last_entry_price": str(position.last_entry_price) if position.last_entry_price is not None else None,
        "exit_price": str(position.exit_price) if position.exit_price is not None else None,
        "peak_unrealized_pnl": str(position.peak_unrealized_pnl) if position.peak_unrealized_pnl is not None else None,
        "peak_unrealized_pnl_at": position.peak_unrealized_pnl_at.isoformat() + "Z" if position.peak_unrealized_pnl_at else None,
        "worst_unrealized_pnl": str(position.worst_unrealized_pnl) if position.worst_unrealized_pnl is not None else None,
        "worst_unrealized_pnl_at": position.worst_unrealized_pnl_at.isoformat() + "Z" if position.worst_unrealized_pnl_at else None,
        "version": position.version,
        "last_updated": position.last_updated.isoformat() + "Z",
        "closed_at": position.closed_at.isoformat() + "Z" if position.closed_at else None,
        "created_at": position.created_at.isoformat() + "Z" if position.created_at else None,
        "created_at": position.created_at.isoformat() + "Z",
        "source": position.source,
        "last_sync_with_bybit": position.last_sync_with_bybit.isoformat() + "Z" if position.last_sync_with_bybit else None,
        "bybit_position_data": position.bybit_position_data,
        # Computed fields
        "is_active": position.is_active,
        "is_closed": position.is_closed,
        "total_pnl": str(position.total_pnl),
        "holding_time_minutes": position.holding_time_minutes,
        # ML features
        "unrealized_pnl_pct": str(unrealized_pct) if unrealized_pct is not None else None,
        "time_held_minutes": time_held,
    }
    return data




def _normalize_snapshot_value(value: Any) -> Any:
    """Normalize snapshot payload values for JSON serialization."""
    if isinstance(value, datetime):
        return value.isoformat() + "Z"
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _normalize_snapshot_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_snapshot_value(v) for v in value]
    return value


def serialize_snapshot(snapshot: PositionSnapshot) -> dict:
    return {
        "id": str(snapshot.id),
        "position_id": str(snapshot.position_id),
        "asset": snapshot.asset,
        "mode": snapshot.mode,
        "snapshot_data": _normalize_snapshot_value(snapshot.snapshot_data),
        "created_at": snapshot.created_at.isoformat() + "Z",
    }


@router.post(
    "/positions/sync-bybit",
    dependencies=[Depends(api_key_auth)],
)
async def sync_positions_with_bybit(
    force: bool = Query(False, description="Force sync: update local positions to match Bybit"),
    asset: Optional[str] = Query(None, description="Optional asset filter: sync only this asset (e.g., BTCUSDT)"),
    position_manager: PositionManager = Depends(get_position_manager),
):
    """Synchronize local positions with Bybit API and return comparison report.

    Args:
        force: If True, force update local positions to match Bybit data
        asset: Optional asset filter to sync only specific asset

    Returns:
        JSON response with sync report including:
        - bybit_positions_count: Number of positions on Bybit
        - local_positions_count: Number of positions in local DB
        - comparisons: Detailed comparison for each asset
        - updated: List of positions that were updated (if force=True)
        - created: List of positions that were created (if force=True)
        - errors: List of errors encountered during sync
    """
    trace_id = get_or_create_trace_id()
    logger.info("position_sync_bybit_request", force=force, asset=asset, trace_id=trace_id)

    try:
        report = await position_manager.sync_positions_with_bybit(force=force, asset=asset, trace_id=trace_id)

        logger.info(
            "position_sync_bybit_completed",
            bybit_count=report["bybit_positions_count"],
            local_count=report["local_positions_count"],
            updated_count=len(report["updated"]),
            created_count=len(report["created"]),
            errors_count=len(report["errors"]),
            force=force,
            trace_id=trace_id,
        )

        return JSONResponse(status_code=200, content=report)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("position_sync_bybit_failed", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to sync positions with Bybit: {e}") from e


