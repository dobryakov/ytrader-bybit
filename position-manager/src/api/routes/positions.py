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
        positions = await position_manager.get_all_positions()
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
        position = await position_manager.get_position(asset, mode_lower)
        if position is None:
            logger.warning("position_not_found", asset=asset, mode=mode, trace_id=trace_id)
            raise HTTPException(
                status_code=404,
                detail=f"Position not found for asset: {asset}, mode: {mode}",
            )

        data = serialize_position_with_features(position, position_manager)
        logger.info("position_get_completed", asset=asset, mode=mode, trace_id=trace_id)
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
        position = await position_manager.get_position(asset, mode_lower)
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
        position = await position_manager.get_position(asset, mode_lower)
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
        "version": position.version,
        "last_updated": position.last_updated.isoformat() + "Z",
        "closed_at": position.closed_at.isoformat() + "Z" if position.closed_at else None,
        "created_at": position.created_at.isoformat() + "Z",
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



