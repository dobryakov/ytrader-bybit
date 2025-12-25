"""Subscription management REST API (v1)."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ...config.logging import get_logger
from ...exceptions import SubscriptionError, ValidationError
from ...services.subscription.subscription_service import SubscriptionService
from ...utils.tracing import get_or_create_trace_id
from .schemas import (
    CreateSubscriptionRequest,
    ErrorResponse,
    SubscriptionListResponse,
    SubscriptionResponse,
)

router = APIRouter(prefix="/api/v1/subscriptions", tags=["subscriptions"])
logger = get_logger(__name__)


def _map_exception_to_error_response(exc: Exception) -> ErrorResponse:
    message = str(exc)
    code: Optional[str] = None

    if isinstance(exc, ValidationError):
        if "Invalid channel_type" in message:
            code = "INVALID_CHANNEL_TYPE"
        elif "Symbol is required" in message:
            code = "MISSING_SYMBOL"
        else:
            code = "VALIDATION_ERROR"
    elif isinstance(exc, SubscriptionError):
        code = "SUBSCRIPTION_ERROR"

    return ErrorResponse(error=message, code=code)


def _to_response_model(subscription) -> SubscriptionResponse:
    """Convert internal Subscription dataclass to API response model."""
    return SubscriptionResponse(
        id=subscription.id,
        channel_type=subscription.channel_type,
        symbol=subscription.symbol,
        topic=subscription.topic,
        requesting_service=subscription.requesting_service,
        is_active=subscription.is_active,
        created_at=subscription.created_at,
        updated_at=subscription.updated_at,
        last_event_at=subscription.last_event_at,
    )


@router.post(
    "",
    response_model=SubscriptionResponse,
    responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Create a new subscription",
)
async def create_subscription(
    request: CreateSubscriptionRequest,
) -> SubscriptionResponse:
    """Create a new subscription."""
    try:
        # Use subscribe() method which creates subscription and sends it to WebSocket
        subscription = await SubscriptionService.subscribe(
            channel_type=request.channel_type,
            requesting_service=request.requesting_service,
            symbol=request.symbol,
        )
        return _to_response_model(subscription)
    except (ValidationError, SubscriptionError) as exc:
        trace_id = get_or_create_trace_id()
        error = _map_exception_to_error_response(exc)
        status_code = 400 if isinstance(exc, ValidationError) else 409
        logger.error(
            "subscription_create_error",
            error=str(exc),
            error_type=type(exc).__name__,
            request_data=request.model_dump(),
            status_code=status_code,
            trace_id=trace_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status_code,
            detail=error.model_dump(),
        )


@router.get(
    "",
    response_model=SubscriptionListResponse,
    responses={401: {"model": ErrorResponse}},
    summary="List subscriptions",
)
async def list_subscriptions(
    requesting_service: Optional[str] = Query(
        default=None, description="Filter by requesting service name"
    ),
    is_active: Optional[bool] = Query(
        default=None, description="Filter by active status"
    ),
    channel_type: Optional[str] = Query(
        default=None, description="Filter by channel type"
    ),
) -> SubscriptionListResponse:
    """List subscriptions with optional filters."""
    subscriptions = await SubscriptionService.list_subscriptions(
        requesting_service=requesting_service,
        is_active=is_active,
        channel_type=channel_type,
    )
    items = [_to_response_model(sub) for sub in subscriptions]
    return SubscriptionListResponse(subscriptions=items, total=len(items))


@router.get(
    "/{subscription_id}",
    response_model=SubscriptionResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get subscription by ID",
)
async def get_subscription(subscription_id: UUID) -> SubscriptionResponse:
    """Retrieve a subscription by ID."""
    subscription = await SubscriptionService.get_subscription_by_id(subscription_id)
    if not subscription:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="Subscription not found", code="SUBSCRIPTION_NOT_FOUND"
            ).model_dump(),
        )
    return _to_response_model(subscription)


@router.delete(
    "/{subscription_id}",
    responses={200: {"model": dict}, 404: {"model": ErrorResponse}},
    summary="Cancel a subscription",
)
async def cancel_subscription(subscription_id: UUID):
    """Cancel (deactivate) a subscription."""
    existed = await SubscriptionService.deactivate_subscription_if_exists(
        subscription_id
    )
    if not existed:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                error="Subscription not found", code="SUBSCRIPTION_NOT_FOUND"
            ).model_dump(),
        )
    return {
        "message": "Subscription cancelled successfully",
        "subscription_id": str(subscription_id),
    }


@router.delete(
    "/by-service/{service_name}",
    responses={200: {"model": dict}},
    summary="Cancel all subscriptions for a service",
)
async def cancel_service_subscriptions(service_name: str):
    """Cancel all subscriptions for a given service."""
    cancelled_count = await SubscriptionService.deactivate_subscriptions_by_service(
        service_name
    )
    return {
        "message": f"Cancelled {cancelled_count} subscriptions for service '{service_name}'",
        "cancelled_count": cancelled_count,
    }


@router.post(
    "/refresh/{channel_type}",
    responses={200: {"model": dict}, 404: {"model": ErrorResponse}},
    summary="Refresh subscription last_event_at for a channel type",
)
async def refresh_subscription(
    channel_type: str,
    requesting_service: Optional[str] = Query(
        default=None, description="Service name (optional, defaults to all services for this channel)"
    ),
):
    """Refresh (update last_event_at) for active subscriptions of a given channel type.
    
    This is used by services to indicate that they are actively processing events
    even when no new events are received from Bybit (e.g., during validation or sync).
    """
    from datetime import datetime, timezone
    
    try:
        subscriptions = await SubscriptionService.list_subscriptions(
            requesting_service=requesting_service,
            is_active=True,
            channel_type=channel_type,
        )
        
        if not subscriptions:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error=f"No active subscriptions found for channel_type '{channel_type}'" + 
                          (f" and service '{requesting_service}'" if requesting_service else ""),
                    code="SUBSCRIPTION_NOT_FOUND"
                ).model_dump(),
            )
        
        # Use UTC timezone-aware datetime
        now = datetime.now(timezone.utc)
        updated_count = 0
        
        for subscription in subscriptions:
            # Ensure timezone-aware datetime for database
            await SubscriptionService.update_last_event_at(subscription.id, now)
            updated_count += 1
        
        logger.info(
            "subscription_refreshed",
            channel_type=channel_type,
            requesting_service=requesting_service,
            updated_count=updated_count,
        )
        
        return {
            "message": f"Refreshed {updated_count} subscription(s)",
            "channel_type": channel_type,
            "requesting_service": requesting_service,
            "updated_count": updated_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        trace_id = get_or_create_trace_id()
        logger.error(
            "subscription_refresh_error",
            channel_type=channel_type,
            requesting_service=requesting_service,
            error=str(e),
            error_type=type(e).__name__,
            trace_id=trace_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=f"Failed to refresh subscription: {str(e)}",
                code="INTERNAL_ERROR"
            ).model_dump(),
        )


