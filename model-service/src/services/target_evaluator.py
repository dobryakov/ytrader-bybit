"""
Target evaluation service.

Computes actual target values for prediction_targets based on configured presets.

NOTE: На первом этапе реализуется безопасный каркас:
- корректно обрабатывает pending-записи
- пишет ошибки вычисления в actual_values_computation_error
- не падает и не блокирует основной сервис.

Детализированные методы `_compute_*_actual` можно дорабатывать итеративно.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config.logging import get_logger
from ..database.repositories.prediction_target_repo import PredictionTargetRepository
from ..database.repositories.trading_signal_repo import TradingSignalRepository
from ..services.feature_service_client import feature_service_client

# Импорт общего модуля для публикации событий
try:
    from common.trading_events import trading_events_publisher
except ImportError:
    # Fallback для случаев, когда модуль не доступен
    trading_events_publisher = None

logger = get_logger(__name__)


class TargetEvaluator:
    """Service for computing actual target values for prediction targets."""

    def __init__(self) -> None:
        self.prediction_repo = PredictionTargetRepository()
        self.trading_signal_repo = TradingSignalRepository()

    async def evaluate_pending_targets(self, limit: int = 100) -> int:
        """
        Evaluate all pending prediction targets whose target_timestamp has passed.

        Returns:
            Number of successfully evaluated targets.
        """
        pending: List[Dict[str, Any]] = await self.prediction_repo.get_pending_computations(
            limit=limit
        )

        if not pending:
            return 0

        evaluated_count = 0

        for target in pending:
            try:
                actual_values = await self._compute_actual_values(target)
                if actual_values is None:
                    # Ничего не вычислили (например, preset ещё не реализован) –
                    # пропускаем, не помечая запись как computed.
                    logger.debug(
                        "Target evaluation skipped (no actual_values)",
                        prediction_target_id=str(target.get("id")),
                    )
                    continue

                await self.prediction_repo.update_actual_values(
                    prediction_target_id=target["id"],
                    actual_values=actual_values,
                    computation_error=None,
                )
                evaluated_count += 1

                # Публикуем событие о вычислении фактического таргета
                await self._publish_target_evaluated_event(target, actual_values)

            except Exception as e:
                logger.error(
                    "Failed to evaluate prediction target",
                    prediction_target_id=str(target.get("id")),
                    error=str(e),
                    exc_info=True,
                )
                # Пишем ошибку, но не падаем
                try:
                    await self.prediction_repo.update_actual_values(
                        prediction_target_id=target["id"],
                        actual_values=target.get("actual_values") or {},
                        computation_error=str(e),
                    )
                except Exception:
                    # Вторая ошибка – только лог
                    logger.warning(
                        "Failed to store target evaluation error",
                        prediction_target_id=str(target.get("id")),
                    )

        logger.info(
            "Target evaluation completed",
            pending=len(pending),
            evaluated=evaluated_count,
        )
        return evaluated_count

    async def check_and_evaluate_immediate(self, prediction_target_id: str) -> bool:
        """
        Check single prediction_target by id and evaluate immediately if target_timestamp passed.

        Returns:
            True if evaluated, False otherwise.
        """
        from uuid import UUID

        target = await self.prediction_repo.get_by_id(UUID(prediction_target_id))
        if not target:
            logger.debug(
                "Prediction target not found for immediate evaluation",
                prediction_target_id=prediction_target_id,
            )
            return False

        target_ts: datetime = target["target_timestamp"]
        # Приводим к UTC-наивному для корректного сравнения
        if target_ts.tzinfo is not None:
            target_ts = target_ts.astimezone(timezone.utc).replace(tzinfo=None)

        now = datetime.utcnow()
        if target_ts > now:
            logger.debug(
                "Target timestamp not reached yet, skipping immediate evaluation",
                prediction_target_id=prediction_target_id,
                target_timestamp=target_ts.isoformat(),
            )
            return False

        try:
            actual_values = await self._compute_actual_values(target)
            if actual_values is None:
                return False

            await self.prediction_repo.update_actual_values(
                prediction_target_id=target["id"],
                actual_values=actual_values,
                computation_error=None,
            )
            logger.info(
                "Immediate target evaluation completed",
                prediction_target_id=prediction_target_id,
            )

            # Публикуем событие о вычислении фактического таргета
            await self._publish_target_evaluated_event(target, actual_values)

            return True
        except Exception as e:
            logger.error(
                "Immediate target evaluation failed",
                prediction_target_id=prediction_target_id,
                error=str(e),
                exc_info=True,
            )
            try:
                await self.prediction_repo.update_actual_values(
                    prediction_target_id=target["id"],
                    actual_values=target.get("actual_values") or {},
                    computation_error=str(e),
                )
            except Exception:
                logger.warning(
                    "Failed to store error for immediate evaluation",
                    prediction_target_id=prediction_target_id,
                )
            return False

    async def _compute_actual_values(self, target: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Compute actual values for a single prediction target.

        На данном этапе реализуем минимальный каркас:
        - Определяем preset
        - Вызываем соответствующий helper (пока заглушки)
        """
        import json
        # target_config может быть строкой (JSON) или словарем
        target_config_raw = target.get("target_config")
        if target_config_raw is None:
            raise ValueError("target_config is missing in prediction target")
        
        if isinstance(target_config_raw, str):
            try:
                target_config: Dict[str, Any] = json.loads(target_config_raw)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(
                    "Failed to parse target_config JSON",
                    prediction_target_id=str(target.get("id")),
                    target_config_type=type(target_config_raw).__name__,
                    target_config_preview=str(target_config_raw)[:200],
                    error=str(e),
                )
                raise ValueError(f"Invalid target_config JSON: {e}") from e
        elif isinstance(target_config_raw, dict):
            target_config: Dict[str, Any] = target_config_raw
        else:
            logger.error(
                "Unexpected target_config type",
                prediction_target_id=str(target.get("id")),
                target_config_type=type(target_config_raw).__name__,
                target_config_value=str(target_config_raw)[:200],
            )
            raise ValueError(f"target_config must be dict or JSON string, got {type(target_config_raw).__name__}")
        
        target_type = target_config.get("type", "regression")
        computation = target_config.get("computation")
        if computation is None:
            computation = {}
        elif isinstance(computation, str):
            try:
                computation = json.loads(computation)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to parse computation JSON, using empty dict",
                    prediction_target_id=str(target.get("id")),
                )
                computation = {}
        preset = computation.get("preset", "returns") if isinstance(computation, dict) else "returns"

        try:
            if target_type == "regression" and preset == "returns":
                return await self._compute_returns_actual(target)
            if target_type == "classification" and preset == "next_candle_direction":
                return await self._compute_candle_direction_actual(target)
            if target_type == "risk_adjusted" and preset == "sharpe_ratio":
                return await self._compute_sharpe_actual(target)

            # Для неизвестных комбинаций не делаем вычисление
            logger.warning(
                "Unsupported target_type/preset combination for actual computation",
                target_type=target_type,
                preset=preset,
                prediction_target_id=str(target.get("id")),
            )
            return None
        except Exception as e:
            logger.error(
                "Error computing actual values",
                prediction_target_id=str(target.get("id")),
                target_type=target_type,
                preset=preset,
                error=str(e),
                exc_info=True,
            )
            raise

    async def _compute_returns_actual(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute actual values for 'returns' preset.

        Использует новый endpoint /api/v1/targets/compute в feature-service.
        """
        from ..config.settings import settings
        
        signal_id = str(target["signal_id"])

        # Получаем сигнал, чтобы знать asset и точный timestamp предсказания
        signal = await self.trading_signal_repo.get_by_signal_id(signal_id)
        if not signal:
            raise ValueError(f"Trading signal not found for signal_id={signal_id}")

        asset: str = signal["asset"]
        prediction_ts: datetime = target["prediction_timestamp"]
        target_ts: datetime = target["target_timestamp"]

        # Нормализуем к UTC timezone-aware
        if prediction_ts.tzinfo is None:
            prediction_ts = prediction_ts.replace(tzinfo=timezone.utc)
        else:
            prediction_ts = prediction_ts.astimezone(timezone.utc)
        
        if target_ts.tzinfo is None:
            target_ts = target_ts.replace(tzinfo=timezone.utc)
        else:
            target_ts = target_ts.astimezone(timezone.utc)

        # Получаем target_registry_version из prediction_targets
        target_registry_version = target["target_registry_version"]
        
        # Вычисляем horizon из timestamps
        horizon_seconds = int((target_ts - prediction_ts).total_seconds())
        
        # Запрашиваем готовый таргет через новый endpoint
        result = await feature_service_client.compute_target(
            symbol=asset,
            prediction_timestamp=prediction_ts,
            target_timestamp=target_ts,
            target_registry_version=target_registry_version,
            horizon_seconds=horizon_seconds,
            max_lookback_seconds=settings.feature_service_target_computation_max_lookback_seconds,
        )
        
        if result is None:
            # Log additional context before raising error
            logger.warning(
                "Target computation returned None for returns",
                asset=asset,
                signal_id=signal_id,
                prediction_target_id=str(target.get("id")),
                prediction_timestamp=prediction_ts.isoformat(),
                target_timestamp=target_ts.isoformat(),
                target_registry_version=target_registry_version,
                horizon_seconds=horizon_seconds,
            )
            raise ValueError(
                f"Target computation failed or data unavailable "
                f"(asset={asset}, signal_id={signal_id}, "
                f"prediction_timestamp={prediction_ts.isoformat()}, "
                f"target_timestamp={target_ts.isoformat()}, "
                f"horizon_seconds={horizon_seconds})"
            )
        
        # Форматируем ответ в зависимости от типа таргета
        target_type = result.get("target_type", "regression")
        preset = result.get("preset", "returns")
        
        if target_type == "regression" and preset == "returns":
            return {
                "value": result["target_value"],
                "price_at_prediction": result["price_at_prediction"],
                "price_at_target": result["price_at_target"],
            }
        else:
            # Fallback: возвращаем базовую структуру
            return {
                "value": result.get("target_value"),
                "target_type": target_type,
                "preset": preset,
            }

    async def _compute_candle_direction_actual(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute actual values for 'next_candle_direction' preset.

        Использует новый endpoint /api/v1/targets/compute в feature-service.
        """
        from ..config.settings import settings
        
        signal_id = str(target["signal_id"])

        # Получаем сигнал, чтобы знать asset и точный timestamp предсказания
        signal = await self.trading_signal_repo.get_by_signal_id(signal_id)
        if not signal:
            raise ValueError(f"Trading signal not found for signal_id={signal_id}")

        asset: str = signal["asset"]
        prediction_ts: datetime = target["prediction_timestamp"]
        target_ts: datetime = target["target_timestamp"]

        # Нормализуем к UTC timezone-aware
        if prediction_ts.tzinfo is None:
            prediction_ts = prediction_ts.replace(tzinfo=timezone.utc)
        else:
            prediction_ts = prediction_ts.astimezone(timezone.utc)
        
        if target_ts.tzinfo is None:
            target_ts = target_ts.replace(tzinfo=timezone.utc)
        else:
            target_ts = target_ts.astimezone(timezone.utc)

        # Получаем target_registry_version из prediction_targets
        target_registry_version = target["target_registry_version"]
        
        # Вычисляем horizon из timestamps
        horizon_seconds = int((target_ts - prediction_ts).total_seconds())
        
        # Запрашиваем готовый таргет через новый endpoint
        result = await feature_service_client.compute_target(
            symbol=asset,
            prediction_timestamp=prediction_ts,
            target_timestamp=target_ts,
            target_registry_version=target_registry_version,
            horizon_seconds=horizon_seconds,
            max_lookback_seconds=settings.feature_service_target_computation_max_lookback_seconds,
        )
        
        if result is None:
            # Log additional context before raising error
            logger.warning(
                "Target computation returned None for candle direction",
                asset=asset,
                signal_id=signal_id,
                prediction_target_id=str(target.get("id")),
                prediction_timestamp=prediction_ts.isoformat(),
                target_timestamp=target_ts.isoformat(),
                target_registry_version=target_registry_version,
                horizon_seconds=horizon_seconds,
            )
            raise ValueError(
                f"Target computation failed or data unavailable "
                f"(asset={asset}, signal_id={signal_id}, "
                f"prediction_timestamp={prediction_ts.isoformat()}, "
                f"target_timestamp={target_ts.isoformat()}, "
                f"horizon_seconds={horizon_seconds})"
            )
        
        # Форматируем ответ для classification (next_candle_direction)
        return {
            "direction": result["direction"],
            "candle_open": result["candle_open"],
            "candle_close": result["candle_close"],
            "return_value": result.get("return_value"),  # Для совместимости
        }

    async def _compute_sharpe_actual(self, target: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute actual values for 'sharpe_ratio' preset.

        Использует новый endpoint /api/v1/targets/compute в feature-service.
        """
        from ..config.settings import settings
        
        signal_id = str(target["signal_id"])

        # Получаем сигнал, чтобы знать asset и точный timestamp предсказания
        signal = await self.trading_signal_repo.get_by_signal_id(signal_id)
        if not signal:
            raise ValueError(f"Trading signal not found for signal_id={signal_id}")

        asset: str = signal["asset"]
        prediction_ts: datetime = target["prediction_timestamp"]
        target_ts: datetime = target["target_timestamp"]

        # Нормализуем к UTC timezone-aware
        if prediction_ts.tzinfo is None:
            prediction_ts = prediction_ts.replace(tzinfo=timezone.utc)
        else:
            prediction_ts = prediction_ts.astimezone(timezone.utc)
        
        if target_ts.tzinfo is None:
            target_ts = target_ts.replace(tzinfo=timezone.utc)
        else:
            target_ts = target_ts.astimezone(timezone.utc)

        # Получаем target_registry_version из prediction_targets
        target_registry_version = target["target_registry_version"]
        
        # Вычисляем horizon из timestamps
        horizon_seconds = int((target_ts - prediction_ts).total_seconds())
        
        # Запрашиваем готовый таргет через новый endpoint
        result = await feature_service_client.compute_target(
            symbol=asset,
            prediction_timestamp=prediction_ts,
            target_timestamp=target_ts,
            target_registry_version=target_registry_version,
            horizon_seconds=horizon_seconds,
            max_lookback_seconds=settings.feature_service_target_computation_max_lookback_seconds,
        )
        
        if result is None:
            # Log additional context before raising error
            logger.warning(
                "Target computation returned None for sharpe ratio",
                asset=asset,
                signal_id=signal_id,
                prediction_target_id=str(target.get("id")),
                prediction_timestamp=prediction_ts.isoformat(),
                target_timestamp=target_ts.isoformat(),
                target_registry_version=target_registry_version,
                horizon_seconds=horizon_seconds,
            )
            raise ValueError(
                f"Target computation failed or data unavailable "
                f"(asset={asset}, signal_id={signal_id}, "
                f"prediction_timestamp={prediction_ts.isoformat()}, "
                f"target_timestamp={target_ts.isoformat()}, "
                f"horizon_seconds={horizon_seconds})"
            )
        
        # Форматируем ответ для risk_adjusted (sharpe_ratio)
        return {
            "sharpe": result["sharpe_value"],
            "returns_series": result.get("returns_series", []),
            "volatility": result.get("volatility"),
        }

    async def _publish_target_evaluated_event(
        self,
        target: Dict[str, Any],
        actual_values: Dict[str, Any],
    ) -> None:
        """
        Публикует событие о вычислении фактического таргета для связи с исходным торговым сигналом.

        Args:
            target: Словарь с данными prediction_target из БД
            actual_values: Вычисленные фактические значения таргета
        """
        if trading_events_publisher is None:
            logger.debug(
                "trading_events_publisher not available, skipping event publication",
                prediction_target_id=str(target.get("id")),
            )
            return

        try:
            signal_id = str(target["signal_id"])
            prediction_target_id = str(target["id"])

            # Получаем сигнал для извлечения asset и strategy_id
            signal = await self.trading_signal_repo.get_by_signal_id(signal_id)
            if not signal:
                logger.warning(
                    "Cannot publish target_evaluated event: signal not found",
                    signal_id=signal_id,
                    prediction_target_id=prediction_target_id,
                )
                return

            asset = signal.get("asset")
            strategy_id = signal.get("strategy_id")

            # Формируем payload события
            payload: Dict[str, Any] = {
                "signal_id": signal_id,
                "prediction_target_id": prediction_target_id,
                "asset": asset,
                "strategy_id": strategy_id,
                "prediction_timestamp": target.get("prediction_timestamp"),
                "target_timestamp": target.get("target_timestamp"),
                "predicted_values": target.get("predicted_values"),
                "actual_values": actual_values,
                "model_version": target.get("model_version"),
                "target_registry_version": target.get("target_registry_version"),
                "feature_registry_version": target.get("feature_registry_version"),
            }

            # Публикуем событие
            await trading_events_publisher.publish_event(
                {
                    "event_type": "prediction_target_evaluated",
                    "service": "model-service",
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "level": "info",
                    "env": trading_events_publisher._environment,
                    "payload": payload,
                }
            )

            logger.debug(
                "Published prediction_target_evaluated event",
                signal_id=signal_id,
                prediction_target_id=prediction_target_id,
                asset=asset,
            )

        except Exception as e:
            # Не блокируем основной процесс при ошибке публикации события
            logger.warning(
                "Failed to publish prediction_target_evaluated event",
                prediction_target_id=str(target.get("id")),
                error=str(e),
                exc_info=True,
            )


# Глобальный инстанс по аналогии с другими сервисами
target_evaluator = TargetEvaluator()


