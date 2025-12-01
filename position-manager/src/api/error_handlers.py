"""Global error handlers for Position Manager FastAPI application.

These handlers provide a consistent error response envelope across the API
and ensure that internal exceptions (database, queue, validation) are
translated into appropriate HTTP status codes with structured logging
context.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Type

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..config.logging import get_logger
from ..exceptions import (
    DatabaseError,
    NotFoundError,
    PositionManagerError,
    QueueError,
    ValidationError,
)
from ..utils.tracing import get_or_create_trace_id


logger = get_logger(__name__)


def _build_error_payload(
    *,
    code: str,
    message: str,
    trace_id: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "error_code": code,
        "message": message,
        "trace_id": trace_id,
    }
    if extra:
        payload["details"] = extra
    return payload


def register_error_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    exception_mapping: Tuple[Tuple[Type[Exception], int, str], ...] = (
        (ValidationError, 400, "VALIDATION_ERROR"),
        (NotFoundError, 404, "NOT_FOUND"),
        (DatabaseError, 503, "DATABASE_ERROR"),
        (QueueError, 503, "QUEUE_ERROR"),
        (PositionManagerError, 500, "POSITION_MANAGER_ERROR"),
    )

    for exc_type, status_code, code in exception_mapping:

        @app.exception_handler(exc_type)  # type: ignore[misc]
        async def _handler(  # pragma: no cover - wiring; behaviour covered via tests
            request: Request, exc: Exception, *, _status=status_code, _code=code
        ) -> JSONResponse:
            trace_id = get_or_create_trace_id()
            logger.error(
                "unhandled_position_manager_exception",
                error_type=exc.__class__.__name__,
                error=str(exc),
                path=request.url.path,
                method=request.method,
                status_code=_status,
                trace_id=trace_id,
                exc_info=True,
            )
            payload = _build_error_payload(
                code=_code,
                message=str(exc),
                trace_id=trace_id,
            )
            return JSONResponse(status_code=_status, content=payload)

    @app.exception_handler(Exception)
    async def _generic_handler(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover - defensive
        trace_id = get_or_create_trace_id()
        logger.error(
            "unhandled_generic_exception",
            error_type=exc.__class__.__name__,
            error=str(exc),
            path=request.url.path,
            method=request.method,
            trace_id=trace_id,
            exc_info=True,
        )
        payload = _build_error_payload(
            code="INTERNAL_SERVER_ERROR",
            message="Internal server error",
            trace_id=trace_id,
        )
        return JSONResponse(status_code=500, content=payload)


