import json
import os
import socket
from typing import Tuple

import aio_pika
import asyncio


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")

GRAYLOG_HOST = os.getenv("GRAYLOG_HOST", "graylog")
GRAYLOG_CUSTOM_EVENTS_GELF_PORT = int(os.getenv("GRAYLOG_CUSTOM_EVENTS_GELF_PORT", "4712"))


async def check_rabbitmq() -> Tuple[bool, str]:
    try:
        connection = await aio_pika.connect_robust(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            login=RABBITMQ_USER,
            password=RABBITMQ_PASSWORD,
        )
        await connection.close()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"rabbitmq_error:{exc}"


def check_graylog_udp() -> Tuple[bool, str]:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Отправим маленький ping-пакет, не ожидая ответа
        payload = json.dumps({"_healthcheck": "trading-events-forwarder"}).encode("utf-8")
        sock.sendto(payload, (GRAYLOG_HOST, GRAYLOG_CUSTOM_EVENTS_GELF_PORT))
        sock.close()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"graylog_udp_error:{exc}"


async def main() -> None:
    ok_rabbit, msg_rabbit = await check_rabbitmq()
    ok_graylog, msg_graylog = check_graylog_udp()

    if ok_rabbit and ok_graylog:
        print("OK")
        raise SystemExit(0)

    print(json.dumps({"rabbitmq": msg_rabbit, "graylog": msg_graylog}))
    raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())


