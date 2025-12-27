"""Registry management endpoints (proxying to Feature Service)."""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Body, Path
from fastapi.responses import JSONResponse
import httpx

from ...config.logging import get_logger
from ...config.settings import settings
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)
router = APIRouter()


def get_feature_service_url(path: str) -> str:
    """Get full URL for Feature Service endpoint."""
    return f"http://{settings.feature_service_host}:{settings.feature_service_port}{path}"


def get_feature_service_headers() -> Dict[str, str]:
    """Get headers for Feature Service requests."""
    headers: Dict[str, str] = {}
    api_key = settings.feature_service_api_key
    if api_key and api_key.strip():
        headers["X-API-Key"] = str(api_key).strip()
    return headers


# Feature Registry endpoints
@router.get("/feature-registry/versions")
async def list_feature_registry_versions():
    """List all Feature Registry versions."""
    trace_id = get_or_create_trace_id()
    logger.info("feature_registry_versions_list_request", trace_id=trace_id)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                get_feature_service_url("/feature-registry/versions"),
                headers=get_feature_service_headers(),
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            "feature_registry_versions_list_failed",
            status_code=e.response.status_code,
            response_text=e.response.text,
            trace_id=trace_id,
        )
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("feature_registry_versions_list_error", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-registry/versions/{version}")
async def get_feature_registry_version(version: str):
    """Get specific Feature Registry version."""
    trace_id = get_or_create_trace_id()
    logger.info("feature_registry_version_get_request", version=version, trace_id=trace_id)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                get_feature_service_url(f"/feature-registry/versions/{version}"),
                headers=get_feature_service_headers(),
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            "feature_registry_version_get_failed",
            version=version,
            status_code=e.response.status_code,
            response_text=e.response.text,
            trace_id=trace_id,
        )
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("feature_registry_version_get_error", version=version, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feature-registry/versions/{version}/activate")
async def activate_feature_registry_version(
    version: str,
    request: Dict[str, Any] = Body(...),
):
    """Activate a Feature Registry version."""
    trace_id = get_or_create_trace_id()
    logger.info("feature_registry_version_activate_request", version=version, trace_id=trace_id)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                get_feature_service_url(f"/feature-registry/versions/{version}/activate"),
                headers=get_feature_service_headers(),
                json=request,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            "feature_registry_version_activate_failed",
            version=version,
            status_code=e.response.status_code,
            response_text=e.response.text,
            trace_id=trace_id,
        )
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("feature_registry_version_activate_error", version=version, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Target Registry endpoints
@router.get("/target-registry/versions")
async def list_target_registry_versions():
    """List all Target Registry versions."""
    trace_id = get_or_create_trace_id()
    logger.info("target_registry_versions_list_request", trace_id=trace_id)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                get_feature_service_url("/target-registry/versions"),
                headers=get_feature_service_headers(),
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            "target_registry_versions_list_failed",
            status_code=e.response.status_code,
            response_text=e.response.text,
            trace_id=trace_id,
        )
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("target_registry_versions_list_error", error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/target-registry/versions/{version}")
async def get_target_registry_version(version: str):
    """Get specific Target Registry version."""
    trace_id = get_or_create_trace_id()
    logger.info("target_registry_version_get_request", version=version, trace_id=trace_id)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                get_feature_service_url(f"/target-registry/versions/{version}"),
                headers=get_feature_service_headers(),
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            "target_registry_version_get_failed",
            version=version,
            status_code=e.response.status_code,
            response_text=e.response.text,
            trace_id=trace_id,
        )
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("target_registry_version_get_error", version=version, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/target-registry/versions/{version}/activate")
async def activate_target_registry_version(
    version: str,
    request: Dict[str, Any] = Body(...),
):
    """Activate a Target Registry version."""
    trace_id = get_or_create_trace_id()
    logger.info("target_registry_version_activate_request", version=version, trace_id=trace_id)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                get_feature_service_url(f"/target-registry/versions/{version}/activate"),
                headers=get_feature_service_headers(),
                json=request,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            "target_registry_version_activate_failed",
            version=version,
            status_code=e.response.status_code,
            response_text=e.response.text,
            trace_id=trace_id,
        )
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("target_registry_version_activate_error", version=version, error=str(e), trace_id=trace_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

