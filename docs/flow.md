# Цепочка работы Order Manager

## Прямой поток (создание ордеров)

```
1. Model Service
   └─> Публикует торговый сигнал
       └─> RabbitMQ очередь: "model-service.trading_signals"
           └─> Order Manager (SignalConsumer)
               └─> SignalProcessor обрабатывает сигнал
                   ├─> Валидация сигнала
                   ├─> Проверка рисков (RiskManager)
                   ├─> Выбор типа ордера (OrderTypeSelector)
                   ├─> Расчет количества (QuantityCalculator)
                   └─> OrderExecutor создает ордер
                       └─> Bybit REST API
                           └─> Ордер создан на бирже
                               └─> Сохранение в PostgreSQL (таблица orders)
```

## Обратный поток (обновление статуса ордеров)

```
2. Bybit Exchange
   └─> WebSocket события (order execution)
       └─> WS-Gateway (подписан на "order" канал)
           └─> Парсинг и обогащение события
               └─> RabbitMQ очередь: "ws-gateway.order"
                   └─> Order Manager (EventSubscriber)
                       └─> Обновление статуса ордера в PostgreSQL
                           └─> Обновление позиции (PositionManager)
```

## Публикация событий (обратная связь)

```
3. Order Manager
   └─> При изменении статуса ордера
       └─> OrderEventPublisher
           └─> RabbitMQ очередь: "order-manager.order_events"
               └─> Model Service (и другие сервисы)
                   └─> Обогащенные события с деталями выполнения
```

## Ручные операции (REST API)

```
4. Клиент/Администратор
   └─> REST API запросы (с X-API-Key)
       └─> Order Manager API endpoints
           ├─> GET /api/v1/orders - список ордеров
           ├─> GET /api/v1/positions - позиции
           └─> POST /api/v1/sync - ручная синхронизация с Bybit
```

## Схема взаимодействия

```
┌─────────────┐         ┌──────────┐         ┌──────────────┐
│ Model       │────────>│ RabbitMQ │────────>│ Order        │
│ Service     │ сигналы │          │         │ Manager      │
└─────────────┘         └──────────┘         └──────┬───────┘
                                                     │
                                                     │ REST API
                                                     ▼
                                              ┌──────────────┐
                                              │ Bybit        │
                                              │ Exchange     │
                                              └──────┬───────┘
                                                     │
                                                     │ WebSocket
                                                     ▼
                                              ┌──────────────┐
                                              │ WS-Gateway   │
                                              └──────┬───────┘
                                                     │
                                                     │ события
                                                     ▼
                                              ┌──────────┐
                                              │ RabbitMQ │
                                              └────┬─────┘
                                                   │
                                                   │ события
                                                   ▼
                                              ┌──────────────┐
                                              │ Order        │
                                              │ Manager      │
                                              └──────┬───────┘
                                                     │
                                                     │ обогащенные
                                                     │ события
                                                     ▼
                                              ┌──────────┐
                                              │ RabbitMQ │
                                              └────┬─────┘
                                                   │
                                                   │ события
                                                   ▼
                                              ┌─────────────┐
                                              │ Model       │
                                              │ Service     │
                                              └─────────────┘
```

## Очереди RabbitMQ

- **`model-service.trading_signals`** — сигналы от Model Service к Order Manager
- **`ws-gateway.order`** — события выполнения ордеров от WS-Gateway к Order Manager
- **`order-manager.order_events`** — обогащенные события от Order Manager к другим сервисам

## База данных

- **`orders`** — все созданные ордера
- **`positions`** — текущие позиции
- **`signal_order_relationships`** — связь сигналов и ордеров
- **`position_snapshots`** — исторические снимки позиций

## Итог

Event-driven архитектура с асинхронной обработкой через RabbitMQ и синхронизацией состояния через WebSocket события.

