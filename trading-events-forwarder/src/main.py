import asyncio
import json
import os
import socket
import time
from typing import Any, Dict

import aio_pika
from aio_pika.exceptions import ChannelClosed
import logging
import structlog


logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")
RABBITMQ_TRADING_EVENTS_EXCHANGE = os.getenv("RABBITMQ_TRADING_EVENTS_EXCHANGE", "trading_events")
# Очередь принадлежит самому forwarder'у (создаётся им), поэтому passive=True не используется
RABBITMQ_TRADING_EVENTS_QUEUE = os.getenv("RABBITMQ_TRADING_EVENTS_QUEUE", "trading_events_forwarder")

GRAYLOG_HOST = os.getenv("GRAYLOG_HOST", "graylog")
GRAYLOG_CUSTOM_EVENTS_GELF_PORT = int(os.getenv("GRAYLOG_CUSTOM_EVENTS_GELF_PORT", "4712"))

SERVICE_NAME = os.getenv("TRADING_EVENTS_FORWARDER_SERVICE_NAME", "trading-events-forwarder")
LOG_LEVEL = os.getenv("TRADING_EVENTS_FORWARDER_LOG_LEVEL", "INFO")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")


def setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def build_gelf(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Преобразование события из очереди trading_events в GELF-сообщение.

    Ожидаемый формат event:
    {
      "event_type": "position_closed",
      "service": "position-manager",
      "ts": "2025-12-23T14:20:13.837Z" | unix_ts (float/int) | None,
      "level": "info" | "warning" | "error" | ...,
      "env": "production",
      "payload": {...},
      "trace_id": "abc123"
    }
    """
    event_type = event.get("event_type", "unknown_event")
    service = event.get("service") or event.get("service_name") or "unknown_service"
    level_str = str(event.get("level", "info")).lower()

    level_map = {
        "debug": 7,
        "info": 6,
        "warning": 4,
        "warn": 4,
        "error": 3,
        "critical": 2,
    }
    level = level_map.get(level_str, 6)

    ts = event.get("ts")
    if isinstance(ts, (int, float)):
        timestamp = float(ts)
    else:
        # если ts строка в ISO — можно парсить, но пока достаточно текущего времени
        timestamp = time.time()

    payload = event.get("payload", {})

    gelf: Dict[str, Any] = {
        "version": "1.1",
        "host": service,
        "short_message": event_type,
        "timestamp": timestamp,
        "level": level,
        "_event_type": event_type,
        "_service_name": service,
        "_env": event.get("env", ENVIRONMENT),
        "_source": SERVICE_NAME,
    }

    trace_id = event.get("trace_id")
    if trace_id:
        gelf["_trace_id"] = str(trace_id)

    # Универсальная распаковка payload:
    #  - все вложенные поля превращаются в плоские поля с префиксом _payload_
    #  - структура не зашита в код (работает для любых новых полей)
    def _flatten_payload(obj: Any, prefix: str) -> None:
        """
        Рекурсивно разворачивает словари/списки в плоские ключи.

        Пример:
          {"predicted_values": {"direction": "green"}}
        ->
          _payload_predicted_values_direction = "green"
        """
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_prefix = f"{prefix}_{k}" if prefix else k
                _flatten_payload(v, new_prefix)
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                new_prefix = f"{prefix}_{idx}" if prefix else str(idx)
                _flatten_payload(v, new_prefix)
        else:
            # Листовой узел — сохраняем как отдельное поле
            field_name = f"_payload_{prefix}" if prefix else "_payload_value"
            # Простые типы пишем как есть, сложные всё равно пройдут через json.dumps выше по стеку
            gelf[field_name] = obj

    if payload:
        # Сохраним "сырой" payload для отладки
        try:
            gelf["_payload_raw"] = (
                json.dumps(payload, default=str)
                if not isinstance(payload, str)
                else payload
            )
        except Exception:  # noqa: BLE001
            # В худшем случае просто проигнорируем raw-представление
            pass

        if isinstance(payload, dict):
            _flatten_payload(payload, "")
        elif isinstance(payload, str):
            # Если payload строка JSON - пробуем распарсить и развернуть
            try:
                parsed_payload = json.loads(payload)
                if isinstance(parsed_payload, dict):
                    _flatten_payload(parsed_payload, "")
                else:
                    # Нестандартный формат - сохраняем как одно поле
                    gelf["_payload_value"] = parsed_payload
            except (json.JSONDecodeError, TypeError):
                # Некорректный JSON - просто сохраняем как строку
                gelf["_payload_value"] = payload
        else:
            # Для других типов сохраняем как одно поле
            gelf["_payload_value"] = payload

    # Добавим все дополнительные поля с префиксом _
    for key, value in event.items():
        if key in {"event_type", "service", "service_name", "ts", "level", "env", "payload", "trace_id"}:
            continue
        field_name = f"_{key}" if not key.startswith("_") else key
        gelf[field_name] = value

    return gelf


def send_gelf_udp(message: Dict[str, Any]) -> None:
    data = json.dumps(message).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(data, (GRAYLOG_HOST, GRAYLOG_CUSTOM_EVENTS_GELF_PORT))
    finally:
        sock.close()


async def handle_message(message: aio_pika.IncomingMessage) -> None:
    async with message.process():
        try:
            body = message.body.decode("utf-8")
            event = json.loads(body)
        except Exception as exc:
            logger.error("failed_to_decode_event", error=str(exc))
            return

        gelf = build_gelf(event)
        try:
            send_gelf_udp(gelf)
            logger.info(
                "event_forwarded",
                event_type=gelf.get("_event_type"),
                service_name=gelf.get("_service_name"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("failed_to_send_gelf", error=str(exc))
            # сообщение уже будет ack'нуто из-за message.process(); при желании можно поднять исключение для nack


async def setup_topology(connection: aio_pika.RobustConnection) -> aio_pika.RobustQueue:
    """
    Настройка exchange и очереди с попыткой самовосстановления в случае неправильных атрибутов.
    """
    while True:
        channel = await connection.channel()
        try:
            # Объявляем exchange для торговых событий (fanout: роутинг по ключам не используется)
            exchange = await channel.declare_exchange(
                RABBITMQ_TRADING_EVENTS_EXCHANGE,
                aio_pika.ExchangeType.FANOUT,
                durable=True,
            )

            # Объявляем собственную очередь forwarder'а и биндим к exchange
            queue = await channel.declare_queue(
                RABBITMQ_TRADING_EVENTS_QUEUE,
                durable=True,
            )
            await queue.bind(exchange)
            return queue
        except ChannelClosed as exc:
            # Возможен случай, когда exchange/queue уже существуют, но с другими атрибутами.
            # Пытаемся удалить и пересоздать их, т.к. это наши сущности (owned-by forwarder).
            logger.warning(
                "channel_closed_during_topology_setup",
                error=str(exc),
            )
            try:
                mgmt_channel = await connection.channel()
                # Пытаемся удалить очередь, если существует
                try:
                    q = await mgmt_channel.declare_queue(
                        RABBITMQ_TRADING_EVENTS_QUEUE,
                        passive=True,
                    )
                    await q.delete(if_unused=False, if_empty=False)
                    logger.warning(
                        "deleted_conflicting_queue",
                        queue=RABBITMQ_TRADING_EVENTS_QUEUE,
                    )
                except Exception:  # noqa: BLE001
                    pass

                # Пытаемся удалить exchange, если существует
                try:
                    ex = await mgmt_channel.declare_exchange(
                        RABBITMQ_TRADING_EVENTS_EXCHANGE,
                        aio_pika.ExchangeType.FANOUT,
                        passive=True,
                    )
                    await ex.delete(if_unused=False)
                    logger.warning(
                        "deleted_conflicting_exchange",
                        exchange=RABBITMQ_TRADING_EVENTS_EXCHANGE,
                    )
                except Exception:  # noqa: BLE001
                    pass

                await mgmt_channel.close()
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.error("topology_cleanup_failed", error=str(cleanup_exc))

            await asyncio.sleep(5)
        except Exception as exc:  # noqa: BLE001
            logger.error("topology_setup_failed", error=str(exc))
            await asyncio.sleep(5)


async def main() -> None:
    setup_logging()
    logger.info(
        "trading_events_forwarder_bootstrap",
        rabbitmq_host=RABBITMQ_HOST,
        rabbitmq_port=RABBITMQ_PORT,
        trading_events_exchange=RABBITMQ_TRADING_EVENTS_EXCHANGE,
        trading_events_queue=RABBITMQ_TRADING_EVENTS_QUEUE,
        graylog_host=GRAYLOG_HOST,
        graylog_port=GRAYLOG_CUSTOM_EVENTS_GELF_PORT,
    )

    while True:
        try:
            connection: aio_pika.RobustConnection = await aio_pika.connect_robust(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                login=RABBITMQ_USER,
                password=RABBITMQ_PASSWORD,
            )

            queue = await setup_topology(connection)
            await queue.consume(handle_message, no_ack=False)

            logger.info("trading_events_forwarder_started")

            # Держим процесс живым; RobustConnection сам попытается восстановиться при отвале RabbitMQ.
            while True:
                await asyncio.sleep(60)
        except Exception as exc:  # noqa: BLE001
            logger.error("forwarder_main_loop_error", error=str(exc))
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())


