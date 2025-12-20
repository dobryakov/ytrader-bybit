#!/bin/bash
#
# Скрипт для отправки сообщений в RabbitMQ
#
# Использование:
#   ./send_rabbitmq_message.sh <queue_name> <message_content>
#   ./send_rabbitmq_message.sh order-manager.order_events '{"event_type":"filled","order_id":"test"}'
#
# Параметры:
#   queue_name      - имя очереди/топика в RabbitMQ
#   message_content - содержимое сообщения (JSON строка)
#

set -euo pipefail

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Проверка аргументов
if [ $# -lt 2 ]; then
    echo -e "${RED}Ошибка: недостаточно аргументов${NC}"
    echo ""
    echo "Использование: $0 <queue_name> <message_content>"
    echo ""
    echo "Примеры:"
    echo "  $0 order-manager.order_events '{\"event_type\":\"filled\",\"order_id\":\"test-123\"}'"
    echo "  $0 model-service.trading_signals '{\"signal_id\":\"test\",\"asset\":\"BTCUSDT\"}'"
    echo ""
    exit 1
fi

QUEUE_NAME="$1"
MESSAGE_CONTENT="$2"

# Получаем credentials из переменных окружения или используем значения по умолчанию
# Используем те же credentials, что и в контейнерах (читаем из docker-compose или .env)
RABBITMQ_HOST="${RABBITMQ_HOST:-rabbitmq}"
RABBITMQ_PORT="${RABBITMQ_PORT:-5672}"

# Пытаемся получить credentials из переменных окружения или из order-manager контейнера
if [ -z "${RABBITMQ_USER:-}" ] || [ -z "${RABBITMQ_PASSWORD:-}" ]; then
    # Пытаемся получить из order-manager контейнера (там точно правильные credentials)
    RABBITMQ_CREDS=$(docker compose exec -T order-manager env 2>/dev/null | grep -E "^RABBITMQ_(USER|PASSWORD)=" | head -2 || true)
    if [ -n "${RABBITMQ_CREDS}" ]; then
        eval $(echo "${RABBITMQ_CREDS}" | sed 's/^/export /')
    fi
fi

# Используем дефолтные значения если не найдено
RABBITMQ_USER="${RABBITMQ_USER:-guest}"
RABBITMQ_PASSWORD="${RABBITMQ_PASSWORD:-guest}"

echo -e "${YELLOW}Отправка сообщения в RabbitMQ...${NC}"
echo "  Очередь: ${QUEUE_NAME}"
echo "  Host: ${RABBITMQ_HOST}:${RABBITMQ_PORT}"
echo "  User: ${RABBITMQ_USER}"
echo "  Сообщение: ${MESSAGE_CONTENT:0:100}$([ ${#MESSAGE_CONTENT} -gt 100 ] && echo '...' || echo '')"
echo ""

# Проверяем, что docker compose доступен
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Ошибка: docker не найден${NC}"
    exit 1
fi

# Кодируем сообщение в base64 для безопасной передачи через переменную окружения
MESSAGE_B64=$(echo -n "${MESSAGE_CONTENT}" | base64 -w 0)

# Используем order-manager контейнер, так как там есть aio_pika
# Передаем параметры через переменные окружения для надежности
docker compose exec -T \
    -e QUEUE_NAME="${QUEUE_NAME}" \
    -e MESSAGE_B64="${MESSAGE_B64}" \
    -e RABBITMQ_HOST="${RABBITMQ_HOST}" \
    -e RABBITMQ_PORT="${RABBITMQ_PORT}" \
    -e RABBITMQ_USER="${RABBITMQ_USER}" \
    -e RABBITMQ_PASSWORD="${RABBITMQ_PASSWORD}" \
    order-manager python3 << 'PYTHON_EOF'
import asyncio
import json
import sys
import os
import base64
from aio_pika import connect_robust, Message
from aio_pika.abc import DeliveryMode

async def send_message():
    try:
        queue_name = os.environ['QUEUE_NAME']
        message_b64 = os.environ['MESSAGE_B64']
        rabbitmq_host = os.environ['RABBITMQ_HOST']
        rabbitmq_port = os.environ['RABBITMQ_PORT']
        rabbitmq_user = os.environ['RABBITMQ_USER']
        rabbitmq_password = os.environ['RABBITMQ_PASSWORD']
        
        # Декодируем сообщение из base64
        try:
            message_content = base64.b64decode(message_b64).decode('utf-8')
        except Exception as e:
            print(f"Ошибка: Не удалось декодировать сообщение из base64: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Проверяем валидность JSON
        try:
            json.loads(message_content)
        except json.JSONDecodeError as e:
            print(f"Ошибка: Некорректный JSON: {e}", file=sys.stderr)
            print(f"Содержимое: {message_content[:200]}", file=sys.stderr)
            sys.exit(1)
        
        # Подключаемся к RabbitMQ
        connection = await connect_robust(
            f"amqp://{rabbitmq_user}:{rabbitmq_password}@{rabbitmq_host}:{rabbitmq_port}/",
        )
        channel = await connection.channel()
        
        # Публикуем сообщение
        await channel.default_exchange.publish(
            Message(
                message_content.encode('utf-8'),
                delivery_mode=DeliveryMode.PERSISTENT,
                content_type="application/json"
            ),
            routing_key=queue_name,
        )
        
        print(f"✅ Сообщение успешно отправлено в очередь {queue_name}")
        
        await connection.close()
        
    except KeyError as e:
        print(f"❌ Ошибка: отсутствует переменная окружения: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка при отправке сообщения: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

asyncio.run(send_message())
PYTHON_EOF

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Сообщение отправлено успешно${NC}"
else
    echo -e "${RED}✗ Ошибка при отправке сообщения${NC}"
    exit $EXIT_CODE
fi

