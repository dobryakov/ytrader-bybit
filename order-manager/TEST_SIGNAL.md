# Тестирование отправки торгового сигнала

## ✅ Что уже реализовано

1. **Signal Consumer** - слушает очередь `model-service.trading_signals`
2. **Signal Processor** - обрабатывает сигналы и принимает решения
3. **Order Executor** - создает ордера на Bybit (с поддержкой dry-run)
4. **Risk Manager** - проверяет баланс и лимиты
5. **Position Manager** - управляет позициями

## 🚀 Быстрый тест

### 1. Запустите сервисы

```bash
# Запустите зависимости (PostgreSQL, RabbitMQ)
docker compose up -d postgres rabbitmq

# Дождитесь готовности (проверьте логи)
docker compose logs -f postgres rabbitmq
# Нажмите Ctrl+C когда сервисы готовы

# Запустите order-manager
docker compose up -d order-manager

# Проверьте логи order-manager
docker compose logs -f order-manager
```

Вы должны увидеть в логах:
```
signal_consumer_started queue_name=model-service.trading_signals
application_started port=4600
```

### 2. Включите dry-run режим (рекомендуется для тестирования)

Отредактируйте `.env` файл:
```bash
ORDERMANAGER_ENABLE_DRY_RUN=true
```

Перезапустите сервис:
```bash
docker compose restart order-manager
```

### 3. Отправьте тестовый сигнал

**Вариант 1: Используя Python скрипт**

```bash
# Установите pika если нужно
pip install pika

# Запустите скрипт
python3 order-manager/test_send_signal.py
```

**Вариант 2: Используя rabbitmqadmin (внутри контейнера)**

```bash
docker compose exec rabbitmq rabbitmqadmin publish routing_key=model-service.trading_signals payload='{
  "signal_id": "test-signal-001",
  "signal_type": "buy",
  "asset": "BTCUSDT",
  "amount": "1000.0",
  "confidence": "0.85",
  "timestamp": "2025-01-27T10:00:00Z",
  "strategy_id": "test-strategy",
  "model_version": null,
  "is_warmup": true,
  "market_data_snapshot": {
    "price": "50000.0",
    "spread": "0.0015",
    "volume_24h": "1000000.0",
    "volatility": "0.02",
    "orderbook_depth": {
      "bid_depth": "100.0",
      "ask_depth": "120.0"
    },
    "technical_indicators": null
  },
  "metadata": {
    "reasoning": "Test signal",
    "risk_score": "0.3"
  },
  "trace_id": "test-trace-001"
}'
```

**Вариант 3: Используя Python напрямую**

```python
import pika
import json
from datetime import datetime, timezone
import uuid

connection = pika.BlockingConnection(
    pika.ConnectionParameters('localhost', 5672, credentials=pika.PlainCredentials('guest', 'guest'))
)
channel = connection.channel()
channel.queue_declare(queue='model-service.trading_signals', durable=True)

signal = {
    "signal_id": str(uuid.uuid4()),
    "signal_type": "buy",
    "asset": "BTCUSDT",
    "amount": "1000.0",
    "confidence": "0.85",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "strategy_id": "test-strategy",
    "model_version": None,
    "is_warmup": True,
    "market_data_snapshot": {
        "price": "50000.0",
        "spread": "0.0015",
        "volume_24h": "1000000.0",
        "volatility": "0.02",
        "orderbook_depth": {"bid_depth": "100.0", "ask_depth": "120.0"},
        "technical_indicators": None
    },
    "metadata": {"reasoning": "Test", "risk_score": "0.3"},
    "trace_id": f"test-{uuid.uuid4()}"
}

channel.basic_publish(
    exchange='',
    routing_key='model-service.trading_signals',
    body=json.dumps(signal),
    properties=pika.BasicProperties(delivery_mode=2)
)
connection.close()
print("Signal sent!")
```

### 4. Проверьте логи order-manager

```bash
docker compose logs -f order-manager
```

Вы должны увидеть:
- `signal_message_received` - сигнал получен
- `signal_processed_successfully` - сигнал обработан
- `order_created` или `order_simulated` (в dry-run режиме)

### 5. Проверьте результат

**В dry-run режиме:**
- Ордер будет создан в базе данных со статусом `dry_run`
- Реальный ордер на Bybit НЕ будет создан
- Все операции будут залогированы

**В live режиме (ORDERMANAGER_ENABLE_DRY_RUN=false):**
- Ордер будет создан на Bybit testnet
- Требуются валидные Bybit API ключи в `.env`

## 📋 Формат сигнала

Обязательные поля:
- `signal_id` (UUID)
- `signal_type` ("buy" или "sell")
- `asset` (например, "BTCUSDT")
- `amount` (десятичное число в USDT)
- `confidence` (0.0-1.0)
- `timestamp` (ISO 8601)
- `strategy_id` (строка)
- `market_data_snapshot` (объект с полем `price`)

Опциональные поля:
- `model_version` (null для warm-up)
- `is_warmup` (boolean, по умолчанию false)
- `metadata` (любой объект)
- `trace_id` (строка для трейсинга)

## ⚠️ Важные замечания

1. **Dry-run режим** - используйте для безопасного тестирования без реальных ордеров
2. **Bybit API ключи** - нужны только для live режима, используйте testnet для разработки
3. **База данных** - убедитесь, что миграции применены (через ws-gateway)
4. **RabbitMQ** - очередь должна существовать (создается автоматически при первом подключении)

## 🔍 Отладка

Если сигнал не обрабатывается:

1. Проверьте, что order-manager запущен:
   ```bash
   docker compose ps order-manager
   ```

2. Проверьте логи на ошибки:
   ```bash
   docker compose logs order-manager | grep -i error
   ```

3. Проверьте, что очередь существует:
   ```bash
   docker compose exec rabbitmq rabbitmqadmin list queues
   ```

4. Проверьте подключение к RabbitMQ:
   ```bash
   docker compose logs order-manager | grep rabbitmq
   ```

5. Проверьте формат сигнала - все обязательные поля должны быть заполнены

