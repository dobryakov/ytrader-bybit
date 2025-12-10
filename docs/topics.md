# Потоки данных: Bybit → Очереди → Сервисы

Документ описывает архитектуру потоков данных: какие потоки читаются из Bybit, в какие очереди RabbitMQ они раскладываются, и какими сервисами потребляются.

---

## Потоки данных из Bybit

Все потоки данных из Bybit проходят через **ws-gateway** сервис, который подключается к Bybit WebSocket API и публикует события в RabbitMQ очереди.

### Публичные потоки (Public Channels)

Не требуют аутентификации, доступны через `/v5/public` endpoint:

| Канал | Bybit Topic | Описание |
|-------|-------------|----------|
| **trades** | `publicTrade.{symbol}` | Сделки (trades) по символу |
| **ticker** | `tickers.{symbol}` | Данные тикера (цена, объем, изменения за 24ч) |
| **orderbook** | `orderbook.1.{symbol}` | Стакан заявок уровня 1 (snapshot + delta) |
| **kline** | `kline.1.{symbol}` | Свечные данные (1-минутные свечи) |
| **funding** | `fundingRate.{symbol}` | Funding rate для perpetual контрактов |
| **liquidation** | `liquidation` | События ликвидации позиций |

### Приватные потоки (Private Channels)

Требуют аутентификации, доступны через `/v5/private` endpoint:

| Канал | Bybit Topic | Описание |
|-------|-------------|----------|
| **balance** | `wallet` | Обновления баланса кошелька |
| **order** | `order` | Статусы собственных ордеров |
| **position** | `position` | Обновления позиций |

---

## Очереди RabbitMQ

ws-gateway публикует события в очереди по формату: `ws-gateway.{event_type}`

### Очереди от ws-gateway

| Очередь | Event Type | Источник | Описание |
|---------|------------|----------|----------|
| `ws-gateway.trade` | `trade` | Bybit trades | События сделок |
| `ws-gateway.ticker` | `ticker` | Bybit tickers | Данные тикера |
| `ws-gateway.orderbook` | `orderbook` | Bybit orderbook | События стакана (snapshot + delta) |
| `ws-gateway.kline` | `kline` | Bybit kline | Свечные данные |
| `ws-gateway.funding` | `funding` | Bybit funding | Funding rate |
| `ws-gateway.order` | `order` | Bybit order | Статусы ордеров |
| `ws-gateway.balance` | `balance` | Bybit wallet | Обновления баланса |
| `ws-gateway.position` | `position` | Bybit position | Обновления позиций |
| `ws-gateway.liquidation` | `liquidation` | Bybit liquidation | События ликвидации |

**Примечание:** Event type может отличаться от channel type (например, `channel_type="trades"` → `event_type="trade"`).

### Другие очереди

Очереди, публикуемые другими сервисами:

| Очередь | Публикует | Потребляет | Описание |
|---------|-----------|------------|----------|
| `features.live` | feature-service | model-service | Вычисленные признаки в реальном времени |
| `features.dataset.ready` | feature-service | model-service | Уведомления о завершении сборки датасета |
| `model-service.trading_signals` | model-service | order-manager | Торговые сигналы для исполнения |
| `order-manager.order_events` | order-manager | - | Обогащенные события исполнения ордеров |
| `order-manager.order_executed` | order-manager | - | События успешного исполнения ордеров |
| `position-manager.position_updated` | position-manager | model-service | Обновления позиций |
| `position-manager.portfolio_updated` | position-manager | - | Обновления портфеля |
| `position-manager.position_snapshot_created` | position-manager | - | Снимки позиций |

---

## Потребители очередей

### feature-service

Потребляет публичные потоки для вычисления признаков и построения датасетов:

| Очередь | Назначение |
|---------|------------|
| `ws-gateway.orderbook` | Вычисление orderbook features, сохранение snapshots/deltas |
| `ws-gateway.trade` | Вычисление trade features, сохранение trades |
| `ws-gateway.ticker` | Вычисление ticker features, сохранение ticker данных |
| `ws-gateway.kline` | Вычисление kline features, сохранение klines |
| `ws-gateway.funding` | Вычисление funding features, сохранение funding данных |

**Публикует:**
- `features.live` — вычисленные признаки в реальном времени (потребляет model-service)
- `features.dataset.ready` — уведомления о завершении сборки датасета (потребляет model-service)

### order-manager

Потребляет события ордеров для отслеживания исполнения:

| Очередь | Назначение |
|---------|------------|
| `ws-gateway.order` | Отслеживание статусов собственных ордеров |

**Публикует:**
- `order-manager.order_events` — обогащенные события исполнения ордеров
- `order-manager.order_executed` — события успешного исполнения

**Потребляет:**
- `model-service.trading_signals` — торговые сигналы от model-service

### position-manager

Потребляет события позиций для синхронизации состояния:

| Очередь | Назначение |
|---------|------------|
| `ws-gateway.position` | Синхронизация позиций из WebSocket |

**Публикует:**
- `position-manager.position_updated` — обновления позиций
- `position-manager.portfolio_updated` — обновления портфеля
- `position-manager.position_snapshot_created` — снимки позиций

### model-service

Потребляет вычисленные признаки и публикует торговые сигналы:

| Очередь | Назначение |
|---------|------------|
| `features.live` | Вычисленные признаки от feature-service (опционально, может использовать REST API) |
| `features.dataset.ready` | Уведомления о завершении сборки датасета для обучения |

**Публикует:**
- `model-service.trading_signals` — торговые сигналы (потребляет order-manager)

**Потребляет (опционально):**
- `position-manager.position_updated` — обновления позиций для risk management

---

## Схема потоков данных

```
┌─────────────────────────────────────────────────────────────────┐
│                         BYBIT WEBSOCKET                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         WS-GATEWAY                              │
│  Подписки: trades, ticker, orderbook, kline, funding,          │
│            order, balance, position                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────────┐                    ┌───────────────────┐
│  PUBLIC QUEUES    │                    │  PRIVATE QUEUES   │
├───────────────────┤                    ├───────────────────┤
│ ws-gateway.trade  │                    │ ws-gateway.order  │
│ ws-gateway.ticker │                    │ ws-gateway.balance│
│ ws-gateway.order- │                    │ ws-gateway.posi-  │
│   book            │                    │   tion            │
│ ws-gateway.kline  │                    └───────────────────┘
│ ws-gateway.funding│                              │
└───────────────────┘                              │
        │                                          │
        ▼                                          ▼
┌───────────────────┐                    ┌───────────────────┐
│ FEATURE-SERVICE   │                    │ ORDER-MANAGER    │
│                   │                    │                  │
│ Потребляет:       │                    │ Потребляет:      │
│ - orderbook       │                    │ - order          │
│ - trade           │                    │ - trading_signals│
│ - ticker          │                    │                  │
│ - kline           │                    │ Публикует:       │
│ - funding         │                    │ - order_events   │
│                   │                    └───────────────────┘
│ Публикует:        │
│ - features.live   │
│ - dataset.ready   │
└───────────────────┘
        │
        ▼
┌───────────────────┐                    ┌───────────────────┐
│ MODEL-SERVICE     │                    │ POSITION-MANAGER │
│                   │                    │                  │
│ Потребляет:       │                    │ Потребляет:      │
│ - features.live   │                    │ - position       │
│ - dataset.ready   │                    │                  │
│ - position_up-    │                    │ Публикует:       │
│   dated (опц.)    │                    │ - position_up-   │
│                   │                    │   dated          │
│ Публикует:        │                    │ - portfolio_up-  │
│ - trading_signals │                    │   dated          │
└───────────────────┘                    └───────────────────┘
        │
        ▼
┌───────────────────┐
│ ORDER-MANAGER    │
│ (см. выше)       │
└───────────────────┘
```

---

## Детализация по сервисам

### ws-gateway

**Роль:** Агрегатор и роутер данных от Bybit

**Подписки:**
- Создаются через REST API `/api/v1/subscriptions`
- Сохраняются в PostgreSQL для автоматического восстановления после переподключения
- Поддерживают публичные и приватные каналы

**Публикация:**
- Все события публикуются в соответствующие очереди `ws-gateway.{event_type}`
- Очереди создаются автоматически при первой публикации
- Настройки очередей: TTL 24 часа, максимум 100K сообщений

### feature-service

**Роль:** Вычисление признаков и построение датасетов

**Подписки на WebSocket:**
- Создает подписки через ws-gateway REST API при старте
- Подписывается на: orderbook, trades, ticker, kline, funding

**Потребление:**
- `ws-gateway.orderbook` — для вычисления orderbook features и сохранения snapshots/deltas
- `ws-gateway.trade` — для вычисления trade features и сохранения trades
- `ws-gateway.ticker` — для вычисления ticker features
- `ws-gateway.kline` — для вычисления kline features и сохранения klines
- `ws-gateway.funding` — для вычисления funding features

**Публикация:**
- `features.live` — вычисленные признаки в реальном времени (опционально, если `FEATURE_SERVICE_USE_QUEUE=true`)
- `features.dataset.ready` — уведомления о завершении сборки датасета (потребляет model-service)

**Хранение:**
- Сохраняет сырые данные в Parquet формат: `/data/raw/{type}/{date}/{symbol}.parquet`
- Типы данных: klines, trades, orderbook_snapshots, orderbook_deltas, ticker, funding

### order-manager

**Роль:** Управление исполнением ордеров

**Подписки на WebSocket:**
- Создает подписку на `order` через ws-gateway REST API

**Потребление:**
- `ws-gateway.order` — статусы собственных ордеров от Bybit
- `model-service.trading_signals` — торговые сигналы для исполнения

**Публикация:**
- `order-manager.order_events` — обогащенные события исполнения ордеров
- `order-manager.order_executed` — события успешного исполнения

### position-manager

**Роль:** Управление позициями и портфелем

**Потребление:**
- `ws-gateway.position` — обновления позиций из WebSocket

**Публикация:**
- `position-manager.position_updated` — обновления позиций
- `position-manager.portfolio_updated` — обновления портфеля
- `position-manager.position_snapshot_created` — снимки позиций

### model-service

**Роль:** Генерация торговых сигналов на основе ML моделей

**Потребление:**
- `features.live` — вычисленные признаки от feature-service (если `FEATURE_SERVICE_USE_QUEUE=true`)
- `features.dataset.ready` — уведомления о завершении сборки датасета для обучения
- Альтернатива: REST API `/features/latest?symbol=...` (если очередь отключена)

**Публикация:**
- `model-service.trading_signals` — торговые сигналы для исполнения order-manager

**Потребление (опционально):**
- `position-manager.position_updated` — для risk management и exit strategies

---

## Особенности

### Определение типа orderbook событий

- Bybit отправляет поле `type` на верхнем уровне сообщения (`"snapshot"` или `"delta"`)
- ws-gateway копирует это поле в payload для правильной обработки в feature-service
- feature-service использует поле `type` для определения, сохранять ли событие как snapshot или delta

### Исторические данные

- **klines**: Доступны через REST API для backfilling исторических данных
- **trades, orderbook**: Доступны только в реальном времени через WebSocket (исторические данные недоступны через REST API)
- Для построения исторических датасетов (>2 дней) используются только klines данные

### Резервные механизмы

- Если очередь `features.live` недоступна, model-service использует REST API feature-service
- Если подписки на WebSocket не созданы, feature-service логирует предупреждения и продолжает работу
- Очереди создаются автоматически при первой публикации (lazy creation)

---

## Связанные документы

- [WebSocket Gateway Specification](../specs/001-websocket-gateway/spec.md)
- [Feature Service Specification](../specs/005-feature-service/spec.md)
- [Model Service Specification](../specs/001-model-service/spec.md)
- [Order Manager Specification](../specs/004-order-manager/spec.md)
- [Position Manager Specification](../specs/001-position-management/spec.md)

