"""
Inference API endpoints for manual intelligent signal generation.

Позволяет вручную дергать инференс (ту же логику, что запускает таймер
IntelligentOrchestrator), для диагностики и отладки.
"""

from typing import Optional, List

from fastapi import APIRouter, Header, HTTPException, status, Query

from ..config.settings import settings
from ..config.logging import get_logger, bind_context
from ..models.signal import TradingSignal
from ..services.intelligent_orchestrator import intelligent_orchestrator

logger = get_logger(__name__)

router = APIRouter(tags=["inference"])


def _verify_api_key(api_key: Optional[str]) -> None:
    """Проверка API-ключа для ручного инференса."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )
    if api_key != settings.model_service_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )


@router.post(
    "/inference/manual",
    response_model=TradingSignal | None,
    status_code=status.HTTP_200_OK,
)
async def trigger_manual_inference(
    asset: str = Query(..., description="Торговая пара (например, BTCUSDT)"),
    strategy_id: str = Query(..., description="Идентификатор стратегии (например, test-strategy)"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_trace_id: Optional[str] = Header(None, alias="X-Trace-ID"),
) -> Optional[TradingSignal]:
    """
    Ручной запуск инференса для одной пары (asset, strategy_id).

    Использует тот же пайплайн, что и IntelligentOrchestrator:
    - IntelligentSignalGenerator.generate_signal(...)
    - валидация сигнала
    - публикация в RabbitMQ (если сигнал валидный)
    """
    _verify_api_key(x_api_key)

    trace_id = x_trace_id
    bind_context(strategy_id=strategy_id, asset=asset, trace_id=trace_id)

    logger.info(
        "manual_inference_triggered",
        asset=asset,
        strategy_id=strategy_id,
        trace_id=trace_id,
    )

    signal = await intelligent_orchestrator.generate_single_signal(
        asset=asset,
        strategy_id=strategy_id,
        trace_id=trace_id,
    )

    if not signal:
        # 200 с null в теле, чтобы можно было видеть логи/причину валидации
        logger.info(
            "manual_inference_completed_no_signal",
            asset=asset,
            strategy_id=strategy_id,
            trace_id=trace_id,
        )
        return None

    logger.info(
        "manual_inference_completed",
        signal_id=signal.signal_id,
        asset=asset,
        strategy_id=strategy_id,
        trace_id=trace_id,
    )

    return signal


