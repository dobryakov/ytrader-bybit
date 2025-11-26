# Архитектура поддержки публичных и приватных эндпоинтов Bybit WebSocket

## Текущее состояние

Сейчас `ws-gateway` использует **приватный эндпоинт** Bybit (`/v5/private`), который:

- Требует аутентификации через API ключи
- Поддерживает подписку как на приватные (wallet, position, order), так и на публичные (tickers, trade, orderbook, kline, liquidation) каналы
- Все подписки идут через одно WebSocket соединение

## Проблема

В будущем может потребоваться:

- Использование **публичного эндпоинта** (`/v5/public`) для публичных данных
- Разделение нагрузки между публичными и приватными соединениями
- Упрощение архитектуры (не нужны API ключи для публичных данных)
- Масштабирование (несколько соединений для разных типов данных)

## Предлагаемое решение

### Вариант 1: Двойное соединение (рекомендуемый)

Поддерживать два независимых WebSocket соединения:

1. **Приватное соединение** (`/v5/private`) — для приватных каналов (wallet, position, order)
2. **Публичное соединение** (`/v5/public`) — для публичных каналов (tickers, trade, orderbook, kline, liquidation)

#### Преимущества

- ✅ Разделение нагрузки
- ✅ Публичные данные не требуют API ключей
- ✅ Независимое переподключение для каждого типа
- ✅ Масштабируемость
- ✅ Более чистая архитектура

#### Недостатки

- ⚠️ Два соединения вместо одного
- ⚠️ Необходимость управления двумя соединениями

### Вариант 2: Единое соединение с выбором эндпоинта

Использовать одно соединение, но автоматически выбирать эндпоинт в зависимости от типа подписки:

- Если есть хотя бы одна приватная подписка → использовать `/v5/private`
- Если только публичные подписки → использовать `/v5/public`

#### Преимущества варианта 2

- ✅ Простота (одно соединение)
- ✅ Автоматический выбор эндпоинта

#### Недостатки варианта 2

- ⚠️ При добавлении приватной подписки нужно переподключаться
- ⚠️ Менее гибко для масштабирования

## Рекомендуемая реализация (Вариант 1)

### Архитектурные изменения

#### 1. Классификация каналов

Определить, какие каналы являются публичными, а какие приватными:

```python
PUBLIC_CHANNELS = {
    "trades",      # trade.BTCUSDT
    "ticker",       # tickers.BTCUSDT
    "orderbook",   # orderbook.1.BTCUSDT
    "kline",       # kline.1.BTCUSDT
    "liquidation", # liquidation
}

PRIVATE_CHANNELS = {
    "balance",     # wallet
    "order",       # order
    "position",    # position (если будет добавлен)
}
```

#### 2. Менеджер соединений

Создать `ConnectionManager`, который управляет двумя соединениями:

```python
class ConnectionManager:
    def __init__(self):
        self._private_connection: Optional[WebSocketConnection] = None
        self._public_connection: Optional[WebSocketConnection] = None
    
    async def get_connection_for_subscription(self, subscription: Subscription) -> WebSocketConnection:
        """Возвращает подходящее соединение для подписки."""
        if subscription.channel_type in PUBLIC_CHANNELS:
            return await self.get_public_connection()
        else:
            return await self.get_private_connection()
    
    async def get_public_connection(self) -> WebSocketConnection:
        """Получить или создать публичное соединение."""
        if not self._public_connection or not self._public_connection.is_connected:
            self._public_connection = WebSocketConnection(endpoint_type="public")
            await self._public_connection.connect()
        return self._public_connection
    
    async def get_private_connection(self) -> WebSocketConnection:
        """Получить или создать приватное соединение."""
        if not self._private_connection or not self._private_connection.is_connected:
            self._private_connection = WebSocketConnection(endpoint_type="private")
            await self._private_connection.connect()
        return self._private_connection
```

#### 3. Изменения в `WebSocketConnection`

Добавить поддержку типа эндпоинта:

```python
class WebSocketConnection:
    def __init__(self, endpoint_type: Literal["public", "private"] = "private"):
        self._endpoint_type = endpoint_type
        # ...
    
    async def connect(self) -> None:
        """Подключение с учетом типа эндпоинта."""
        url = self._get_ws_url()
        # ...
    
    def _get_ws_url(self) -> str:
        """Получить URL в зависимости от типа эндпоинта."""
        if self._endpoint_type == "public":
            if settings.bybit_environment == "mainnet":
                return "wss://stream.bybit.com/v5/public"
            else:
                return "wss://stream-testnet.bybit.com/v5/public"
        else:  # private
            if settings.bybit_environment == "mainnet":
                return "wss://stream.bybit.com/v5/private"
            else:
                return "wss://stream-testnet.bybit.com/v5/private"
    
    async def _authenticate(self) -> None:
        """Аутентификация только для приватного эндпоинта."""
        if self._endpoint_type == "public":
            # Публичный эндпоинт не требует аутентификации
            logger.info("websocket_public_endpoint_no_auth_required")
            return
        # ... существующая логика аутентификации для приватного эндпоинта
```

#### 4. Изменения в `SubscriptionService`

Модифицировать метод подписки для использования правильного соединения:

```python
async def subscribe(
    self,
    channel_type: str,
    requesting_service: str,
    symbol: Optional[str] = None,
):
    """Создать подписку и отправить сообщение подписки на правильный эндпоинт."""
    subscription = await SubscriptionService.create_subscription(
        channel_type=channel_type,
        requesting_service=requesting_service,
        symbol=symbol,
    )
    
    # Получить правильное соединение для этого типа канала
    connection_manager = get_connection_manager()
    connection = await connection_manager.get_connection_for_subscription(subscription)
    
    from .subscription import build_subscribe_message
    msg = build_subscribe_message([subscription])
    await connection.send(msg)
    # ...
```

### Конфигурация

Добавить настройку для выбора стратегии (опционально):

```python
# В settings.py
bybit_ws_strategy: str = Field(
    default="dual",  # "dual" (два соединения) или "single" (одно соединение)
    description="WebSocket connection strategy: 'dual' or 'single'"
)
```

### Миграция

1. **Обратная совместимость**: По умолчанию использовать приватный эндпоинт (как сейчас)
2. **Постепенный переход**: Добавить флаг для включения двойного соединения
3. **Тестирование**: Протестировать оба эндпоинта на testnet

### Примеры использования

#### Подписка на публичные данные (тикеры)

```bash
curl -X POST http://localhost:4400/api/v1/subscriptions \
  -H "Authorization: Bearer $WS_GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_type": "ticker",
    "symbol": "BTCUSDT",
    "requesting_service": "model-service"
  }'
```

Автоматически будет использовано публичное соединение.

#### Подписка на приватные данные (баланс)

```bash
curl -X POST http://localhost:4400/api/v1/subscriptions \
  -H "Authorization: Bearer $WS_GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_type": "balance",
    "requesting_service": "order-manager"
  }'
```

Автоматически будет использовано приватное соединение.

## План реализации

### Этап 1: Подготовка (обратная совместимость)

1. Добавить классификацию каналов (публичные/приватные)
2. Добавить поддержку типа эндпоинта в `WebSocketConnection`
3. Добавить метод `_get_ws_url()` с выбором эндпоинта
4. Модифицировать `_authenticate()` для пропуска аутентификации на публичном эндпоинте

### Этап 2: Менеджер соединений

1. Создать `ConnectionManager` для управления двумя соединениями
2. Реализовать логику выбора соединения по типу канала
3. Обновить `SubscriptionService` для использования менеджера

### Этап 3: Тестирование

1. Протестировать публичное соединение на testnet
2. Протестировать одновременную работу обоих соединений
3. Проверить переподключение для каждого типа соединения

### Этап 4: Документация

1. Обновить README с описанием новой архитектуры
2. Добавить примеры использования
3. Обновить спецификацию ws-service.md

## Альтернативный подход (если не нужна немедленная реализация)

Можно оставить текущую архитектуру (одно приватное соединение) и добавить возможность переключения в будущем через конфигурацию. Это позволит:

- Сохранить простоту текущей реализации
- Добавить гибкость для будущих изменений
- Минимизировать изменения в коде

## Заключение

Рекомендуется реализовать **Вариант 1 (двойное соединение)** для:

- Лучшей масштабируемости
- Разделения ответственности
- Независимого управления публичными и приватными данными
- Готовности к будущему росту системы

Текущая архитектура (одно приватное соединение) продолжит работать, но двойное соединение даст больше гибкости и возможностей для оптимизации.
