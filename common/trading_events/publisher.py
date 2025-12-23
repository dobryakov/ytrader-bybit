import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import aio_pika


class TradingEventsPublisher:
    """
    Publisher for trading/business events to RabbitMQ exchange `trading_events`.

    Сервисы публикуют события в exchange, а микросервис trading-events-forwarder
    читает их из своей очереди и отправляет в Graylog (GELF).
    """

    def __init__(
        self,
        exchange_name: Optional[str] = None,
    ) -> None:
        self._exchange_name = exchange_name or os.getenv("RABBITMQ_TRADING_EVENTS_EXCHANGE", "trading_events")

        self._host = os.getenv("RABBITMQ_HOST", "rabbitmq")
        self._port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self._user = os.getenv("RABBITMQ_USER", "guest")
        self._password = os.getenv("RABBITMQ_PASSWORD", "guest")
        self._environment = os.getenv("ENVIRONMENT", "production")

        self._connection: Optional[aio_pika.RobustConnection] = None
        self._exchange: Optional[aio_pika.Exchange] = None

    async def _get_exchange(self) -> aio_pika.Exchange:
        # RobustExchange в aio-pika не имеет атрибута is_closed, поэтому
        # просто переиспользуем сохранённый объект; RobustConnection сам
        # позаботится о восстановлении при сбоях.
        if self._exchange is not None:
            return self._exchange

        if not self._connection or self._connection.is_closed:
            self._connection = await aio_pika.connect_robust(
                host=self._host,
                port=self._port,
                login=self._user,
                password=self._password,
            )

        channel = await self._connection.channel()
        self._exchange = await channel.declare_exchange(
            self._exchange_name,
            aio_pika.ExchangeType.FANOUT,
            durable=True,
        )
        return self._exchange

    async def publish_event(self, event: Dict[str, Any]) -> None:
        """
        Публикация произвольного события в exchange trading_events.

        Ожидается, что event содержит хотя бы:
          - event_type
          - service
          - ts (ISO или unix timestamp)
          - payload (dict)
        """
        exchange = await self._get_exchange()

        # Добавим env по умолчанию, если не задан
        event.setdefault("env", self._environment)

        body = json.dumps(event, default=self._default_serializer).encode("utf-8")

        await exchange.publish(
            aio_pika.Message(
                body,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key="",
        )

    async def publish_trading_signal_event(
        self,
        *,
        event_type: str,
        service: str,
        signal_payload: Dict[str, Any],
        trace_id: Optional[str] = None,
        level: str = "info",
        ts: Optional[datetime] = None,
    ) -> None:
        """
        Публикация события о торговом сигнале.

        signal_payload должен содержать те же поля, которые пишутся в БД trading_signals:
          - signal_id, strategy_id, asset, side, price, confidence, timestamp,
            model_version, is_warmup, market_data_snapshot, metadata, trace_id,
            prediction_horizon_seconds, target_timestamp.
        """
        timestamp = ts or datetime.utcnow()
        event: Dict[str, Any] = {
            "event_type": event_type,
            "service": service,
            "ts": timestamp.isoformat() + "Z",
            "level": level,
            "env": self._environment,
            "payload": signal_payload,
        }
        if trace_id:
            event["trace_id"] = trace_id

        await self.publish_event(event)

    @staticmethod
    def _default_serializer(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat() + "Z"
        return str(obj)


# Глобальный экземпляр, который можно переиспользовать во всех сервисах
trading_events_publisher = TradingEventsPublisher()


