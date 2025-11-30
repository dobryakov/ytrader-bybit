# Техническое задание: Микросервис управления портфолио позиций (Position Manager Service)

**Дата создания**: 2025-01-XX  
**Статус**: Draft  
**Приоритет**: P2  
**Зависимости**: Order Manager, WebSocket Gateway, Model Service

## 1. Обзор

### 1.1. Цель проекта

Создать отдельный микросервис **Position Manager Service** для централизованного управления портфолио позиций. Сервис будет отвечать за:

- Агрегацию всех позиций в портфолио
- Расчет портфолио-метрик (общий exposure, общий PnL, стоимость портфолио)
- Предоставление портфолио-данных через REST API
- Интеграцию с Risk Manager для проверки лимитов на уровне портфолио
- Управление жизненным циклом позиций (создание, обновление, валидация, снимки)

### 1.2. Проблема текущей архитектуры

В текущей архитектуре функциональность управления позициями разбросана по нескольким сервисам:

1. **Order Manager** (`order-manager/src/services/position_manager.py`):
   - Управление отдельными позициями (CRUD операции)
   - Обновление позиций при исполнении ордеров
   - Валидация позиций
   - Создание снимков позиций
   - **Отсутствует**: агрегация портфолио, расчет общих метрик

2. **Risk Manager** (`order-manager/src/services/risk_manager.py`):
   - Метод `check_max_exposure()` существует, но **не используется**
   - Не вычисляет `total_exposure` самостоятельно
   - Проверяет лимиты только на уровне отдельных позиций

3. **Model Service** (`model-service/src/models/position_state.py`):
   - Модель `OrderPositionState` с методами `get_total_exposure()`, `get_unrealized_pnl()`
   - Используется только для чтения состояния при генерации сигналов
   - Не управляет портфолио, не сохраняет агрегированные метрики

4. **REST API** (`order-manager/src/api/routes/positions.py`):
   - Возвращает список позиций без агрегации
   - Нет endpoints для портфолио-метрик

### 1.3. Преимущества нового сервиса

- **Централизация**: единая точка управления всеми позициями
- **Масштабируемость**: независимое масштабирование сервиса управления позициями
- **Разделение ответственности**: Order Manager фокусируется на ордерах, Position Manager — на позициях
- **Расширяемость**: легче добавлять новую функциональность (аналитика, отчеты, алерты)
- **Производительность**: оптимизированные запросы для портфолио-метрик
- **Консистентность**: единый источник истины для портфолио-данных

## 2. Архитектура

### 2.1. Общая архитектура

```
┌─────────────────┐
│  Order Manager  │ ──┐
└─────────────────┘   │
                      │ Обновления позиций
┌─────────────────┐   │ (через RabbitMQ)
│  WS Gateway     │ ──┤
└─────────────────┘   │
                      ▼
         ┌─────────────────────────┐
         │  Position Manager       │
         │  (новый сервис)         │
         │                         │
         │  - Управление позициями │
         │  - Агрегация портфолио  │
         │  - Расчет метрик         │
         │  - REST API              │
         └─────────────────────────┘
                      │
                      │ Запросы портфолио
                      ▼
         ┌─────────────────────────┐
         │  Model Service          │
         │  Risk Manager           │
         │  UI / Monitoring        │
         └─────────────────────────┘
```

### 2.2. Технологический стек

- **Язык**: Python 3.11+
- **Фреймворк**: FastAPI
- **База данных**: PostgreSQL (общая БД, таблицы `positions`, `position_snapshots`)
- **Очереди**: RabbitMQ (потребление событий обновления позиций)
- **Логирование**: structlog с trace IDs
- **Конфигурация**: pydantic-settings
- **Контейнеризация**: Docker, docker-compose

### 2.3. Порт и конфигурация

- **Порт REST API**: 4800 (по аналогии с ws-gateway: 4400, order-manager: 4600)
- **Имя сервиса**: `position-manager`
- **Префикс переменных окружения**: `POSITION_MANAGER_*`

## 3. Функциональные требования

### 3.1. Управление отдельными позициями

#### 3.1.1. CRUD операции для позиций

- **Получение позиции по активу**: `get_position(asset, mode)`
- **Получение всех позиций**: `get_all_positions()`
- **Обновление позиции из исполнения ордера**: `update_position_from_order_fill()`
- **Обновление позиции из WebSocket события**: `update_position_from_websocket()`
  - Обновление `unrealized_pnl`, `realized_pnl` (всегда)
  - Обновление `average_entry_price` из `avgPrice` (если присутствует в событии и отличается от сохраненного)
  - Валидация `size` (проверка расхождений без прямого обновления)
  - Пересчет готовых features с учетом обновленных данных
- **Валидация позиции**: `validate_position(asset, mode)`
- **Создание снимка позиции**: `create_position_snapshot(position)`
- **Вычисление готовых features для ML моделей**:
  - `unrealized_pnl_pct` — процент нереализованного PnL (относительно entry price или размера позиции)
  - `time_held_minutes` — время удержания позиции в минутах (от момента создания/последнего обновления)
  - `position_size_norm` — нормализованный размер позиции (относительно total_exposure или баланса)

**Источник**: извлечь из `order-manager/src/services/position_manager.py`

#### 3.1.2. Обновление позиций из разных источников

Сервис должен обрабатывать обновления позиций из двух источников:

1. **Исполнение ордеров** (через RabbitMQ очередь `order-manager.order_executed`):
   - Обновление `size` и `average_entry_price` на основе исполненных ордеров
   - Источник истины для размера позиции и средней цены входа

2. **WebSocket события position** (через RabbitMQ очередь `ws-gateway.position`):
   - Обновление `unrealized_pnl`, `realized_pnl` из событий Bybit
   - Обновление `average_entry_price` из `avgPrice` в событии (если присутствует)
   - Источник истины для PnL метрик и средней цены входа (если предоставляется биржей)

**Стратегия разрешения конфликтов** (гибридный подход):

**Приоритеты источников данных**:
- **WebSocket события** — источник истины для:
  * `unrealized_pnl`, `realized_pnl` (приоритет 1 — всегда обновлять)
  * `average_entry_price` (из `avgPrice`) — приоритет 1, если присутствует в событии
  * `size` — для валидации (не обновлять напрямую, только проверять расхождения)
- **Исполнение ордеров** — источник истины для:
  * `size` (приоритет 1 — всегда обновлять при исполнении)
  * `average_entry_price` — вычисленное из ордеров (приоритет 2, если нет в WebSocket событии)

**Логика обновления**:
- При обновлении из WebSocket:
  * Всегда обновлять `unrealized_pnl`, `realized_pnl`
  * Если в событии присутствует `avgPrice`:
    - Сравнить с сохраненным `average_entry_price`
    - Если расхождение > порога (`POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD`), обновить `average_entry_price`
    - Логировать все обновления с trace_id
  * Не обновлять `size` напрямую, но использовать для валидации расхождений
- При обновлении из ордера:
  * Обновлять `size` и вычислять `average_entry_price` (weighted average)
  * Если `average_entry_price` уже обновлен из WebSocket и отличается, залогировать расхождение
  * Пересчитывать PnL если нужно (опционально, так как WebSocket — источник истины для PnL)

**Валидация расхождений**:
- При критических расхождениях между источниками запускать валидацию позиции
- Логировать все расхождения для мониторинга и анализа

### 3.2. Управление портфолио

#### 3.2.1. Агрегация позиций

- Получение всех позиций из БД
- Группировка по активам, режимам торговли
- Фильтрация по различным критериям (актив, режим, размер позиции)

#### 3.2.2. Расчет портфолио-метрик

**Общий exposure (в USDT)**:
- Сумма абсолютных значений позиций, конвертированных в USDT
- Формула: `SUM(ABS(position.size) * current_price)` для всех позиций
- Учет текущих рыночных цен для конвертации

**Общий нереализованный PnL (в USDT)**:
- Сумма `unrealized_pnl` всех позиций
- Формула: `SUM(position.unrealized_pnl)` для всех позиций

**Общий реализованный PnL (в USDT)**:
- Сумма `realized_pnl` всех позиций
- Формула: `SUM(position.realized_pnl)` для всех позиций

**Стоимость портфолио (в USDT)**:
- Сумма текущей стоимости всех позиций
- Формула: `SUM(position.size * current_price)` для всех позиций

**Количество открытых позиций**:
- Количество позиций с `size != 0`

**Распределение по активам**:
- Группировка позиций по активам с метриками для каждого

**Распределение по направлениям**:
- Количество long позиций
- Количество short позиций
- Net exposure (long - short)

#### 3.2.3. Кэширование метрик

- Кэширование агрегированных метрик в памяти (TTL: 5-10 секунд)
- Инвалидация кэша при обновлении позиций
- Опциональное сохранение метрик в БД для исторического анализа

### 3.3. Валидация и синхронизация

#### 3.3.1. Периодическая валидация позиций

- Вычисление позиций из истории ордеров
- Сравнение с сохраненным состоянием
- Автоматическое исправление расхождений
- Логирование всех расхождений

**Источник**: извлечь из `order-manager/src/services/position_manager.py` метод `validate_position()`

#### 3.3.2. Создание снимков позиций

- Периодическое создание снимков всех позиций
- Сохранение в таблицу `position_snapshots`
- Использование для исторического анализа и аудита

**Источник**: извлечь из `order-manager/src/services/position_manager.py` метод `create_position_snapshot()`

### 3.4. Интеграция с Risk Manager

#### 3.4.1. Предоставление данных для проверки лимитов

- Метод `get_total_exposure()` для получения общего exposure
- Метод `get_portfolio_metrics()` для получения всех метрик
- REST API endpoint для запроса метрик

#### 3.4.2. Предоставление готовых features для ML моделей

Position Manager вычисляет и предоставляет готовые features для использования в ML моделях:

- **`unrealized_pnl_pct`**: Процент нереализованного PnL
  - Формула: `(unrealized_pnl / (abs(size) * average_entry_price)) * 100` (если есть entry price)
  - Альтернатива: `(unrealized_pnl / total_exposure) * 100` (относительно exposure портфолио)
  - Использование: процентная доходность позиции для оценки эффективности

- **`time_held_minutes`**: Время удержания позиции в минутах
  - Формула: `(current_timestamp - last_updated) / 60` (или время с момента создания позиции)
  - Использование: временные паттерны, оценка длительности позиций

- **`position_size_norm`**: Нормализованный размер позиции
  - Формула: `abs(size * current_price) / total_exposure` (относительно exposure портфолио)
  - Альтернатива: `abs(size * current_price) / balance` (относительно баланса)
  - Использование: относительный размер для сравнения между активами, оценка концентрации портфолио

Эти features включаются в:
- REST API ответы при запросе позиций (`GET /api/v1/positions`, `GET /api/v1/positions/{asset}`)
- События обновления позиций (`position-manager.position_updated`)
- События создания снапшотов (`position-manager.position_snapshot_created`)
- Портфолио-метрики (если `include_positions=true`)

**Преимущества**:
- Единый источник вычисления features (консистентность)
- Оптимизация производительности (вычисление один раз, использование многими)
- Упрощение feature engineering в Model Service (не нужно вычислять локально)

#### 3.4.3. Поддержка правил риск-менеджмента для Model Service

Position Manager предоставляет данные для реализации правил риск-менеджмента в Model Service:

**1. Take Profit (принудительный SELL при достижении прибыли)**:
- **Данные**: `unrealized_pnl_pct` из Position Manager (через REST API или события)
- **Правило**: Если `unrealized_pnl_pct > MODEL_SERVICE_TAKE_PROFIT_PCT`, Model Service должен принудительно генерировать SELL сигнал для закрытия позиции
- **Реализация**: В `IntelligentSignalGenerator.generate_signal()` перед генерацией сигнала модели проверять `unrealized_pnl_pct` из Position Manager
- **Преимущества**: Автоматическая фиксация прибыли, защита от откатов, дисциплинированное управление рисками

**2. Position Size Limit (пропуск BUY при большом размере позиции)**:
- **Данные**: `position_size_norm` из Position Manager (через REST API или события)
- **Правило**: Если `position_size_norm > MODEL_SERVICE_MAX_POSITION_SIZE_RATIO`, Model Service должен пропускать генерацию BUY сигналов
- **Реализация**: В `IntelligentSignalGenerator.generate_signal()` перед генерацией BUY сигнала проверять `position_size_norm` из Position Manager
- **Преимущества**: Защита от переэкспозиции, диверсификация портфолио, управление концентрацией риска

**Интеграция**:
- Model Service при генерации сигнала использует данные из кэша, обновляемого из событий `position-manager.position_updated` или непосредственно получает через REST API (`GET /api/v1/positions/{asset}`)
- Конфигурация порогов: `MODEL_SERVICE_TAKE_PROFIT_PCT` (по умолчанию 3.0), `MODEL_SERVICE_MAX_POSITION_SIZE_RATIO` (по умолчанию 0.8)

**Примечание**: Эти правила дополняют существующие проверки Risk Manager в Order Manager, но применяются на этапе генерации сигналов, что позволяет избежать создания ненужных сигналов и снизить нагрузку на систему.

#### 3.4.4. Проверка лимитов на уровне портфолио

- Интеграция с Risk Manager для проверки `max_exposure`
- Предоставление данных для проверки других портфолио-лимитов
- Публикация событий при превышении лимитов

## 4. REST API

### 4.1. Endpoints для отдельных позиций

#### GET /api/v1/positions
Получить список всех позиций с опциональной фильтрацией.

**Query параметры**:
- `asset` (optional): Фильтр по торговой паре (например, BTCUSDT)
- `mode` (optional): Фильтр по режиму торговли (one-way, hedge)
- `size_min` (optional): Минимальный размер позиции
- `size_max` (optional): Максимальный размер позиции

**Ответ**:
```json
{
  "positions": [
    {
      "id": "uuid",
      "asset": "BTCUSDT",
      "size": "1.5",
      "average_entry_price": "50000.00",
      "unrealized_pnl": "150.00",
      "realized_pnl": "50.00",
      "mode": "one-way",
      "long_size": null,
      "short_size": null,
      "unrealized_pnl_pct": "0.30",
      "time_held_minutes": 120,
      "position_size_norm": "0.15",
      "last_updated": "2025-01-XXT...",
      "last_snapshot_at": "2025-01-XXT..."
    }
  ],
  "count": 1
}
```

**Примечание**: Поля `unrealized_pnl_pct`, `time_held_minutes`, `position_size_norm` являются вычисляемыми features, готовыми для использования в ML моделях:
- `unrealized_pnl_pct`: Процент нереализованного PnL относительно entry price или размера позиции (формула: `(unrealized_pnl / (size * average_entry_price)) * 100` или `(unrealized_pnl / total_exposure) * 100`)
- `time_held_minutes`: Время удержания позиции в минутах (разница между текущим временем и `last_updated` или временем создания позиции)
- `position_size_norm`: Нормализованный размер позиции (относительно total_exposure портфолио или баланса, формула: `abs(size * current_price) / total_exposure` или `abs(size * current_price) / balance`)
```

**Источник**: извлечь из `order-manager/src/api/routes/positions.py`

#### GET /api/v1/positions/{asset}
Получить позицию для конкретного актива.

**Path параметры**:
- `asset`: Торговая пара (например, BTCUSDT)

**Query параметры**:
- `mode` (optional, default: "one-way"): Режим торговли

**Ответ**: Объект позиции (аналогично элементу в списке выше)

**Источник**: извлечить из `order-manager/src/api/routes/positions.py`

### 4.2. Endpoints для портфолио

#### GET /api/v1/portfolio
Получить агрегированные метрики портфолио.

**Query параметры**:
- `include_positions` (optional, default: false): Включить список позиций в ответ
- `asset` (optional): Рассчитать метрики только для указанного актива

**Ответ**:
```json
{
  "total_exposure_usdt": "10000.00",
  "total_unrealized_pnl_usdt": "150.00",
  "total_realized_pnl_usdt": "50.00",
  "portfolio_value_usdt": "10200.00",
  "open_positions_count": 3,
  "long_positions_count": 2,
  "short_positions_count": 1,
  "net_exposure_usdt": "5000.00",
  "by_asset": {
    "BTCUSDT": {
      "exposure_usdt": "7500.00",
      "unrealized_pnl_usdt": "100.00",
      "size": "1.5"
    },
    "ETHUSDT": {
      "exposure_usdt": "2500.00",
      "unrealized_pnl_usdt": "50.00",
      "size": "10.0"
    }
  },
  "positions": [...],  // если include_positions=true
  "calculated_at": "2025-01-XXT..."
}
```

#### GET /api/v1/portfolio/exposure
Получить только общий exposure портфолио.

**Ответ**:
```json
{
  "total_exposure_usdt": "10000.00",
  "calculated_at": "2025-01-XXT..."
}
```

#### GET /api/v1/portfolio/pnl
Получить только PnL метрики портфолио.

**Ответ**:
```json
{
  "total_unrealized_pnl_usdt": "150.00",
  "total_realized_pnl_usdt": "50.00",
  "total_pnl_usdt": "200.00",
  "calculated_at": "2025-01-XXT..."
}
```

### 4.3. Endpoints для управления

#### POST /api/v1/positions/{asset}/validate
Запустить валидацию позиции вручную.

**Path параметры**:
- `asset`: Торговая пара

**Query параметры**:
- `mode` (optional, default: "one-way"): Режим торговли
- `fix_discrepancies` (optional, default: true): Автоматически исправлять расхождения

**Ответ**:
```json
{
  "is_valid": true,
  "error_message": null,
  "updated_position": null
}
```

#### POST /api/v1/positions/{asset}/snapshot
Создать снимок позиции вручную.

**Path параметры**:
- `asset`: Торговая пара

**Query параметры**:
- `mode` (optional, default: "one-way"): Режим торговли

**Ответ**: Объект снимка позиции

#### GET /api/v1/positions/{asset}/snapshots
Получить историю снимков позиции.

**Query параметры**:
- `limit` (optional, default: 100): Максимальное количество снимков
- `offset` (optional, default: 0): Смещение для пагинации

**Ответ**: Список снимков позиции

### 4.4. Аутентификация

Все endpoints требуют аутентификации через API Key:
- Header: `X-API-Key: <api_key>`
- Конфигурация: `POSITION_MANAGER_API_KEY`

### 4.5. Health Check

#### GET /health
Проверка здоровья сервиса.

**Ответ**:
```json
{
  "status": "healthy",
  "service": "position-manager",
  "database_connected": true,
  "queue_connected": true,
  "positions_count": 5,
  "timestamp": "2025-01-XXT..."
}
```

## 5. Потребление событий из RabbitMQ

### 5.1. События обновления позиций из Order Manager

**Очередь**: `order-manager.order_executed` (или новая очередь `order-manager.position_updated`)

**Формат события**:
```json
{
  "event_type": "position_updated_from_order",
  "order_id": "uuid",
  "asset": "BTCUSDT",
  "side": "Buy",
  "filled_quantity": "0.1",
  "execution_price": "50000.00",
  "mode": "one-way",
  "trace_id": "uuid"
}
```

**Обработка**:
- Вызвать `update_position_from_order_fill()`
- Обновить позицию в БД
- Инвалидировать кэш портфолио-метрик
- Опубликовать событие обновления позиции (опционально)

### 5.2. События обновления позиций из WebSocket Gateway

**Очередь**: `ws-gateway.position`

**Формат события**: Стандартный формат события от WS Gateway

**Формат payload WebSocket position события от Bybit**:
```json
{
  "symbol": "BTCUSDT",
  "size": "1.5",
  "side": "Buy",
  "avgPrice": "50000.00",  // Средняя цена входа (важно: использовать если присутствует)
  "unrealisedPnl": "150.00",
  "realisedPnl": "50.00",
  "mode": "one-way",
  "leverage": "10",
  "markPrice": "50100.00"
}
```

**Обработка**:
- Парсить данные позиции из payload
- Извлечь поля: `unrealisedPnl`, `realisedPnl`, `avgPrice` (если присутствует), `size` (для валидации)
- Вызвать `update_position_from_websocket()` с полными данными
- Обновить позицию в БД:
  * Всегда обновлять `unrealized_pnl`, `realized_pnl`
  * Если `avgPrice` присутствует в событии:
    - Сравнить с сохраненным `average_entry_price`
    - Если расхождение > порога (`POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD`), обновить `average_entry_price`
    - Логировать обновление с trace_id
  * Использовать `size` из события для валидации (не обновлять напрямую, только проверять расхождения)
- Инвалидировать кэш портфолио-метрик
- Пересчитать готовые features (`unrealized_pnl_pct`, `position_size_norm`) с учетом обновленного `average_entry_price`
- Опубликовать событие обновления позиции (опционально)

**Обработка ошибок**:
- Если `avgPrice` отсутствует в событии, использовать сохраненное значение `average_entry_price`
- Если расхождение `size` критическое, запустить валидацию позиции
- Логировать все расхождения для мониторинга

## 6. Публикация событий

### 6.1. События обновления позиций

**Очередь**: `position-manager.position_updated`

**Формат события**:
```json
{
  "event_type": "position_updated",
  "position_id": "uuid",
  "asset": "BTCUSDT",
  "size": "1.5",
  "unrealized_pnl": "150.00",
  "realized_pnl": "50.00",
  "mode": "one-way",
  "unrealized_pnl_pct": "0.30",
  "time_held_minutes": 120,
  "position_size_norm": "0.15",
  "update_source": "order_execution" | "websocket",
  "trace_id": "uuid",
  "timestamp": "2025-01-XXT..."
}
```

**Подписчики**:
- Model Service (для обновления состояния при генерации сигналов)
- Risk Manager (для проверки лимитов)
- UI / Monitoring (для отображения в реальном времени)

### 6.2. События обновления портфолио

**Очередь**: `position-manager.portfolio_updated`

**Формат события**:
```json
{
  "event_type": "portfolio_updated",
  "total_exposure_usdt": "10000.00",
  "total_unrealized_pnl_usdt": "150.00",
  "total_realized_pnl_usdt": "50.00",
  "open_positions_count": 3,
  "trace_id": "uuid",
  "timestamp": "2025-01-XXT..."
}
```

**Подписчики**:
- Risk Manager (для проверки лимитов на уровне портфолио)
- UI / Monitoring (для отображения метрик)
- Alerting Service (для алертов при превышении лимитов)

### 6.3. События создания снапшотов позиций

**Очередь**: `position-manager.position_snapshot_created`

**Формат события**:
```json
{
  "event_type": "position_snapshot_created",
  "snapshot_id": "uuid",
  "position_id": "uuid",
  "asset": "BTCUSDT",
  "size": "1.5",
  "average_entry_price": "50000.00",
  "unrealized_pnl": "150.00",
  "realized_pnl": "50.00",
  "mode": "one-way",
  "long_size": null,
  "short_size": null,
  "unrealized_pnl_pct": "0.30",
  "time_held_minutes": 120,
  "position_size_norm": "0.15",
  "snapshot_timestamp": "2025-01-XXT...",
  "trace_id": "uuid",
  "timestamp": "2025-01-XXT..."
}
```

**Подписчики**:
- **Model Service** (для исторической реконструкции состояния позиций при обучении моделей):
  - Использование снапшотов для восстановления состояния позиций на момент каждого execution event
  - Снапшоты позволяют точно определить состояние портфолио в любой момент времени для feature engineering при обучении
  - Потребление снапшотов через очередь обеспечивает асинхронную обработку и не блокирует обучение моделей
- UI / Monitoring (для исторического анализа и визуализации)
- Analytics Service (для построения отчетов и аналитики)

**Примечание**: Снапшоты создаются периодически (настраивается через `POSITION_MANAGER_SNAPSHOT_INTERVAL`). Каждый снапшот содержит полное состояние позиции на момент создания, что критично для точной исторической реконструкции при обучении моделей.

## 7. План миграции функциональности

### 7.1. Этап 1: Создание нового сервиса

1. Создать структуру проекта `position-manager/`
2. Настроить Docker, docker-compose
3. Настроить базовую инфраструктуру (логирование, БД, RabbitMQ)
4. Создать модели данных (Position, PositionSnapshot, PortfolioMetrics)
5. **Валидация задач WebSocket Gateway**: Проверить и при необходимости обновить невыполненные задачи в `specs/001-websocket-gateway/tasks.md` (Phase 7.5: Position Channel Support, задачи T125-T136), чтобы они коррелировали с архитектурой нового Position Manager сервиса. Убедиться, что:
   - Задачи по сохранению позиций в БД (T129-T135) учитывают, что Position Manager будет основным сервисом для управления позициями
   - Задачи по маршрутизации событий (T133) учитывают, что события должны доставляться в очередь `ws-gateway.position` для потребления Position Manager
   - Документация задач отражает интеграцию с Position Manager вместо прямого использования в Order Manager

### 7.2. Этап 2: Извлечение функциональности из Order Manager

#### 7.2.1. Извлечь PositionManager

**Источник**: `order-manager/src/services/position_manager.py`

**Действия**:
1. Скопировать класс `PositionManager` в новый сервис
2. Адаптировать под новую структуру (изменить импорты, пути)
3. Добавить новые методы для портфолио-метрик
4. **Обновить метод `update_position_from_websocket()`**:
   - Добавить обработку `avgPrice` из WebSocket события
   - Реализовать логику сравнения с сохраненным `average_entry_price`
   - Добавить обновление `average_entry_price` при превышении порога расхождения (`POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD`)
   - Добавить валидацию `size` из WebSocket события (без прямого обновления, только проверка расхождений)
   - Добавить пересчет готовых features (`unrealized_pnl_pct`, `position_size_norm`) после обновления
   - Добавить логирование всех обновлений с trace_id
   - Добавить обработку случая, когда `avgPrice` отсутствует в событии (использовать сохраненное значение)
5. Обновить Order Manager для использования Position Manager через поток событий (RabbitMQ)

**Варианты интеграции**:
- **Вариант A (рекомендуемый)**: Order Manager публикует события в RabbitMQ, Position Manager обрабатывает их через consumers. Преимущества: асинхронность, отказоустойчивость, масштабируемость, слабая связанность сервисов.
- **Вариант B**: Order Manager вызывает Position Manager через REST API. Использовать только для синхронных операций, требующих немедленного ответа (например, проверка позиции перед созданием ордера).

#### 7.2.2. Извлечь REST API endpoints

**Источник**: `order-manager/src/api/routes/positions.py`

**Действия**:
1. Скопировать endpoints в новый сервис
2. Добавить новые endpoints для портфолио
3. Удалить endpoints из Order Manager
4. Обновить документацию API

#### 7.2.3. Извлечь фоновые задачи

**Источник**: `order-manager/src/main.py` (классы `PositionSnapshotTask`, `PositionValidationTask`)

**Действия**:
1. Скопировать задачи в новый сервис
2. Адаптировать под новую структуру
3. Удалить из Order Manager
4. Настроить запуск в новом сервисе

#### 7.2.4. Очистка tasks.md Order Manager

**Источник**: `specs/004-order-manager/tasks.md`

**Действия**:
1. Удалить невыполненные задачи, относящиеся к позициям, которые будут реализованы в Position Manager:
   - **Phase 4.5: Position Updates via WebSocket** (задачи T075-T084) — функциональность обновления позиций из WebSocket будет реализована в Position Manager
   - Обновить описание задач, которые частично связаны с позициями, чтобы отразить использование Position Manager через API
2. Обновить счетчики задач в summary секции
3. Добавить примечания о том, что функциональность управления позициями перенесена в Position Manager сервис
4. Обновить зависимости между фазами, если они изменились

### 7.3. Этап 3: Интеграция с Risk Manager

**Текущее состояние Risk Manager**:
- **Расположение**: `order-manager/src/services/risk_manager.py`
- **Использование**: Вызывается из `SignalProcessor` перед созданием ордеров для проверки рисков
- **Текущие функции**:
  - `check_balance()` — проверка достаточности баланса для ордера (запрос к Bybit API)
  - `check_order_size()` — проверка размера ордера относительно `max_exposure * max_order_size_ratio`
  - `check_position_size()` — проверка размера позиции для конкретного актива (использует `PositionManager.get_position()`)
  - `check_max_exposure()` — проверка общего exposure портфолио (метод существует, но **не используется**)
- **Зависимости**: Использует `PositionManager` для получения текущей позиции по активу

**После реализации Position Manager**:
- **Расположение**: Risk Manager **остается в Order Manager** (проверки рисков выполняются синхронно перед созданием ордера)
- **Новые функции**:
  - Получение `total_exposure` из Position Manager через REST API (`GET /api/v1/portfolio/exposure`)
  - Использование `check_max_exposure()` с реальными данными портфолио
  - Получение портфолио-метрик для расширенных проверок рисков
- **Интеграция**: REST API вызовы к Position Manager для синхронных проверок (требуется немедленный ответ перед созданием ордера)

**Действия**:
1. Добавить REST API endpoint в Position Manager для получения `total_exposure` (`GET /api/v1/portfolio/exposure`)
2. Добавить метод в Risk Manager для получения `total_exposure` из Position Manager через REST API
3. Обновить `check_max_exposure()` в Risk Manager для использования данных из Position Manager
4. Реализовать проверку `max_exposure` на уровне портфолио перед созданием ордеров
5. Добавить обработку ошибок при недоступности Position Manager (fallback логика)
6. Добавить логирование и алерты при превышении лимитов
7. Оптимизировать производительность (кэширование метрик, batch запросы при проверке нескольких сигналов)

### 7.4. Этап 4: Интеграция с Model Service

**Текущее состояние Model Service**:
- **Расположение**: `model-service/src/services/intelligent_signal_generator.py`, `model-service/src/database/repositories/position_state_repo.py`
- **Использование**: Читает позиции напрямую из БД (таблица `positions`) при генерации сигналов
- **Текущие функции**:
  - `PositionStateRepository.get_order_position_state()` — синхронное чтение позиций из БД
  - `OrderPositionState.get_total_exposure()` — локальное вычисление метрик из списка позиций
  - Используется для feature engineering и расчета размера ордера при генерации сигналов

**После реализации Position Manager**:
- **Основной способ интеграции**: **REST API** (синхронные запросы при генерации сигналов)
  - Model Service нужен немедленный ответ с актуальными данными при генерации сигнала
  - Position Manager предоставляет уже агрегированные метрики (total_exposure, total_pnl), не нужно вычислять локально
  - Использовать `GET /api/v1/portfolio` для получения портфолио-метрик
  - Использовать `GET /api/v1/positions` для получения списка позиций (если нужны детали)
- **Опционально: RabbitMQ события для кэширования**:
  - Подписаться на события обновления позиций (`position-manager.position_updated`) для обновления локального кэша
  - Кэшировать портфолио-метрики в памяти Model Service
  - Уменьшить количество REST запросов при частой генерации сигналов
  - **Важно**: Кэш используется для оптимизации, но REST API остается основным источником актуальных данных

**Действия**:
1. **Валидация tasks.md Model Service**: Проверить и обновить незакрытые задачи в `specs/001-model-service/tasks.md`, связанные с позициями и портфолио:
   - Удалить неактуальные задачи, которые будут реализованы в Position Manager (например, задачи по управлению позициями, если такие есть)
   - Обновить задачи, связанные с чтением позиций, чтобы отразить использование Position Manager через REST API
   - Добавить новые задачи для интеграции с Position Manager (REST API клиент, кэширование, обработка событий)
   - Обновить зависимости между фазами, если они изменились
   - Обновить счетчики задач в summary секции
2. Добавить REST API клиент в Model Service для вызова Position Manager
3. Заменить `PositionStateRepository._get_open_positions()` на REST API запрос `GET /api/v1/positions`
4. Заменить локальные вычисления метрик (`get_total_exposure()`, `get_unrealized_pnl()`) на REST API запрос `GET /api/v1/portfolio`
5. Реализовать кэширование портфолио-метрик в памяти Model Service (опционально)
6. Подписаться на события обновления позиций из RabbitMQ для инвалидации кэша (опционально)
7. **Подписаться на события создания снапшотов позиций** (`position-manager.position_snapshot_created`) из RabbitMQ:
   - Создать consumer для очереди `position-manager.position_snapshot_created`
   - Сохранять снапшоты в локальное хранилище (БД или файловая система) для исторической реконструкции
   - Реализовать индексацию снапшотов по timestamp для быстрого поиска
8. **Интегрировать снапшоты в процесс обучения моделей**:
   - Обновить `DatasetBuilder` для использования снапшотов при построении training dataset
   - Реализовать метод восстановления `OrderPositionState` из снапшотов для конкретного timestamp
   - Передавать исторический `OrderPositionState` в `FeatureEngineer` для точной feature engineering
9. **Реализовать правила риск-менеджмента** (см. раздел 3.4.3):
   - Take Profit: добавить проверку `unrealized_pnl_pct` перед генерацией сигнала модели, принудительно генерировать SELL при превышении порога
   - Position Size Limit: добавить проверку `position_size_norm` перед генерацией BUY сигнала, пропускать BUY при превышении порога
   - Использовать данные из Position Manager через REST API (`GET /api/v1/positions/{asset}`)
   - Добавить конфигурацию `MODEL_SERVICE_TAKE_PROFIT_PCT` и `MODEL_SERVICE_MAX_POSITION_SIZE_RATIO`
   - Интегрировать в `IntelligentSignalGenerator.generate_signal()` перед вызовом модели
10. Добавить обработку ошибок при недоступности Position Manager (fallback к чтению из БД или использованию кэша)
11. Оптимизировать производительность (TTL кэша, batch запросы при генерации нескольких сигналов)

### 7.5. Этап 5: Тестирование и валидация

1. Unit тесты для всех методов Position Manager
2. Integration тесты для REST API
3. E2E тесты для полного потока обновления позиций
4. Тесты производительности для портфолио-метрик
5. Валидация миграции (сравнение результатов до/после)

### 7.6. Этап 6: Переработка дашбордов Grafana

После развертывания Position Manager необходимо обновить дашборды Grafana, чтобы они использовали данные из нового сервиса вместо прямых запросов к таблице `positions` в БД.

#### 7.6.1. Дашборд "Trading Performance" (`trading-performance.json`)

**Текущее состояние**: Дашборд использует прямые SQL запросы к таблице `positions` для получения метрик PnL.

**Задачи**:

1. **Панель "Total PnL"** (ID: 1):
   - **Текущий запрос**: `SELECT COALESCE(SUM((e.performance->>'realized_pnl')::DECIMAL), 0) + COALESCE((SELECT SUM(unrealized_pnl) FROM positions WHERE size != 0), 0) as total_pnl FROM execution_events e`
   - **Новый подход**: 
     - Заменить `(SELECT SUM(unrealized_pnl) FROM positions WHERE size != 0)` на REST API запрос к Position Manager: `GET /api/v1/portfolio/pnl`
     - Использовать Infinity datasource для HTTP запросов к Position Manager API
     - Или создать PostgreSQL функцию/представление, которое читает из Position Manager через HTTP (если поддерживается)
   - **Альтернатива**: Использовать PostgreSQL datasource с запросом к представлению `portfolio_metrics_view` (если создано), которое синхронизируется с Position Manager

2. **Панель "Unrealized PnL"** (ID: 3):
   - **Текущий запрос**: `SELECT COALESCE(SUM(unrealized_pnl), 0) as unrealized_pnl FROM positions WHERE size != 0`
   - **Новый подход**: 
     - Заменить на REST API запрос: `GET /api/v1/portfolio/pnl` → использовать поле `total_unrealized_pnl`
     - Использовать Infinity datasource для HTTP запросов
   - **Альтернатива**: Использовать PostgreSQL представление, синхронизированное с Position Manager

3. **Панель "Cumulative PnL Over Time"** (ID: 9):
   - **Текущий запрос**: Включает `(SELECT SUM(unrealized_pnl) FROM positions WHERE size != 0) as current_unrealized_pnl`
   - **Новый подход**: 
     - Заменить подзапрос на REST API запрос к Position Manager
     - Или использовать исторические данные из снапшотов позиций (`position_snapshots`) для построения временного ряда unrealized PnL
   - **Рекомендация**: Использовать снапшоты позиций для исторического графика, так как они содержат полную историю изменений

4. **Панель "PnL by Asset"** (ID: 13):
   - **Текущий запрос**: Агрегирует данные из `execution_events` по `asset`
   - **Новый подход**: 
     - Добавить панель с данными из Position Manager: `GET /api/v1/portfolio?include_positions=true`
     - Использовать поле `by_asset` из ответа API для отображения PnL по активам
     - Сохранить существующую панель для realized PnL из execution_events, добавить новую для unrealized PnL из Position Manager

**Файл для обновления**: `grafana/dashboards/trading-performance.json`

---

#### 7.6.2. Дашборд "Trading System Monitoring" (`trading-system-monitoring.json`)

**Текущее состояние**: Панель "Order Execution" использует JOIN с таблицей `positions` для получения данных о позициях.

**Задачи**:

1. **Панель "Order Execution"** (ID: 2):
   - **Текущий запрос**: Использует `LEFT JOIN LATERAL (SELECT unrealized_pnl, size, average_entry_price FROM positions WHERE asset = o.asset) p`
   - **Новый подход**: 
     - **Вариант 1**: Использовать PostgreSQL представление `positions_view`, которое синхронизируется с Position Manager через триггеры или периодическое обновление
     - **Вариант 2**: Разделить панель на две части:
       - Основная таблица с данными ордеров (из `orders` и `execution_events`)
       - Дополнительная таблица с данными позиций из Position Manager через REST API (Infinity datasource)
     - **Вариант 3**: Использовать PostgreSQL функцию, которая делает HTTP запрос к Position Manager API (если PostgreSQL поддерживает HTTP расширения)
   - **Рекомендация**: Использовать вариант 1 (представление) для сохранения производительности и совместимости с существующими запросами

**Файл для обновления**: `grafana/dashboards/trading-system-monitoring.json`

---

#### 7.6.3. Дашборд "Order Execution Panel" (`order-execution-panel.json`)

**Текущее состояние**: Использует JOIN с таблицей `positions` для расчета PnL.

**Задачи**:

1. **Панель "Order Execution Panel"**:
   - **Текущий запрос**: Использует `LEFT JOIN LATERAL (SELECT unrealized_pnl, size, average_entry_price, current_price FROM positions WHERE asset = o.asset) p`
   - **Новый подход**: 
     - Аналогично панели в "Trading System Monitoring", использовать представление `positions_view`
     - Или разделить на две панели: ордера + позиции из Position Manager API

**Файл для обновления**: `grafana/dashboards/order-execution-panel.json`

---

#### 7.6.4. Новые дашборды и панели

**Задачи**:

1. **Создать новый дашборд "Portfolio Management"**:
   - Панель "Total Exposure" с данными из `GET /api/v1/portfolio/exposure`
   - Панель "Portfolio PnL Breakdown" с детализацией по активам из `GET /api/v1/portfolio?include_positions=true`
   - Панель "Position Size Distribution" с использованием `position_size_norm` из Position Manager
   - Панель "Unrealized PnL by Asset" с использованием `unrealized_pnl_pct` из Position Manager
   - Панель "Time Held by Position" с использованием `time_held_minutes` из Position Manager
   - Панель "Position Snapshots History" с данными из `GET /api/v1/positions/{asset}/snapshots`

2. **Добавить панель "Position Manager Health"** в дашборд "System Health":
   - Health check статус Position Manager: `GET /health`
   - Метрики производительности: время ответа API, количество обновлений позиций
   - Статистика валидаций: количество расхождений, успешных синхронизаций

3. **Добавить панель "Risk Management Metrics"**:
   - Отображение `total_exposure` для проверки лимитов Risk Manager
   - Визуализация `position_size_norm` для каждого актива (для правила Position Size Limit)
   - Визуализация `unrealized_pnl_pct` для каждого актива (для правила Take Profit)

**Файлы для создания**:
- `grafana/dashboards/portfolio-management.json` (новый дашборд)

---

#### 7.6.5. Технические детали реализации

**Подход 1: PostgreSQL представления (рекомендуется)**

Создать представления в БД, которые синхронизируются с Position Manager:

```sql
-- Представление для метрик портфолио
CREATE VIEW portfolio_metrics_view AS
SELECT 
    'total_exposure' as metric_name,
    (SELECT total_exposure::text FROM http_get('http://position-manager:4800/api/v1/portfolio/exposure')::json) as metric_value
UNION ALL
SELECT 
    'total_unrealized_pnl' as metric_name,
    (SELECT total_unrealized_pnl::text FROM http_get('http://position-manager:4800/api/v1/portfolio/pnl')::json) as metric_value;

-- Представление для позиций (синхронизируется через триггеры или периодическое обновление)
CREATE VIEW positions_view AS
SELECT * FROM positions;  -- Если positions таблица синхронизируется с Position Manager
```

**Подход 2: Infinity datasource для REST API**

Использовать Grafana Infinity datasource для прямых HTTP запросов к Position Manager API:

- Настроить Infinity datasource с URL `http://position-manager:4800`
- Использовать JSON parser для парсинга ответов API
- Создать переменные для фильтрации по активам, стратегиям и т.д.

**Подход 3: Гибридный подход**

- Для реального времени: использовать Infinity datasource для прямых запросов к Position Manager API
- Для исторических данных: использовать PostgreSQL с данными из снапшотов позиций (`position_snapshots`)

---

#### 7.6.6. Порядок выполнения задач

1. **Подготовка**:
   - Создать PostgreSQL представления или настроить Infinity datasource для Position Manager
   - Протестировать доступность Position Manager API из Grafana контейнера
   - Создать backup существующих дашбордов

2. **Обновление существующих дашбордов**:
   - Обновить `trading-performance.json` (панели Total PnL, Unrealized PnL, Cumulative PnL)
   - Обновить `trading-system-monitoring.json` (панель Order Execution)
   - Обновить `order-execution-panel.json`

3. **Создание новых дашбордов**:
   - Создать `portfolio-management.json` с новыми панелями
   - Добавить панели в существующие дашборды (System Health, Risk Management)

4. **Тестирование**:
   - Проверить корректность отображения данных
   - Сравнить результаты с данными из старой таблицы `positions` (до миграции)
   - Проверить производительность запросов

5. **Документация**:
   - Обновить документацию дашбордов
   - Добавить описание новых панелей и источников данных

---

#### 7.6.7. Зависимости

- Position Manager должен быть развернут и доступен
- REST API Position Manager должен быть протестирован
- Grafana должна иметь доступ к Position Manager (сеть, порт 4800)
- Если используется Infinity datasource, плагин должен быть установлен в Grafana

### 7.7. Этап 7: Документация и развертывание

1. Обновить README с инструкциями по развертыванию
2. Обновить документацию API
3. Обновить docker-compose.yml
4. Обновить env.example
5. Развернуть в тестовой среде
6. Мониторинг и отладка
7. Обновить документацию дашбордов Grafana (после выполнения этапа 7.6)

## 8. Технические детали

### 8.1. Структура проекта

```
position-manager/
├── Dockerfile
├── docker-compose.yml (добавить в основной)
├── requirements.txt
├── README.md
├── env.example
├── src/
│   ├── main.py
│   ├── config/
│   │   ├── settings.py
│   │   ├── database.py
│   │   ├── rabbitmq.py
│   │   └── logging.py
│   ├── models/
│   │   ├── position.py (извлечь из order-manager)
│   │   ├── portfolio.py (новый)
│   │   └── __init__.py
│   ├── services/
│   │   ├── position_manager.py (извлечь и расширить)
│   │   ├── portfolio_manager.py (новый)
│   │   ├── position_event_consumer.py (новый)
│   │   └── position_sync.py (новый)
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── positions.py (извлечь и расширить)
│   │   │   ├── portfolio.py (новый)
│   │   │   └── health.py
│   │   └── middleware/
│   │       ├── auth.py
│   │       └── logging.py
│   ├── consumers/
│   │   ├── order_position_consumer.py (новый)
│   │   └── websocket_position_consumer.py (новый)
│   ├── publishers/
│   │   └── position_event_publisher.py (новый)
│   ├── tasks/
│   │   ├── position_snapshot_task.py (извлечь)
│   │   └── position_validation_task.py (извлечь)
│   └── utils/
│       └── tracing.py
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

### 8.2. Конфигурация

**Переменные окружения** (добавить в `env.example`):

```bash
# Position Manager Service Configuration
POSITION_MANAGER_PORT=4800
POSITION_MANAGER_API_KEY=<api_key>
POSITION_MANAGER_LOG_LEVEL=INFO
POSITION_MANAGER_SERVICE_NAME=position-manager

# Database (использует общую БД)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ytrader
POSTGRES_USER=ytrader
POSTGRES_PASSWORD=<password>

# RabbitMQ (использует общий RabbitMQ)
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest

# Position Management
POSITION_MANAGER_SNAPSHOT_INTERVAL=3600  # секунды
POSITION_MANAGER_VALIDATION_INTERVAL=1800  # секунды
POSITION_MANAGER_METRICS_CACHE_TTL=10  # секунды

# Position Update Strategy
POSITION_MANAGER_USE_WS_AVG_PRICE=true  # Использовать avgPrice из WebSocket событий для обновления average_entry_price
POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD=0.001  # Порог расхождения для обновления average_entry_price (0.1% = 0.001)
POSITION_MANAGER_SIZE_VALIDATION_THRESHOLD=0.0001  # Порог расхождения size для запуска валидации позиции

# Integration
ORDER_MANAGER_URL=http://order-manager:4600
WS_GATEWAY_URL=http://ws-gateway:4400
```

### 8.3. База данных

**Используемые таблицы** (уже существуют в общей БД):
- `positions` - текущие позиции
- `position_snapshots` - снимки позиций

**Новая таблица** (опционально, для кэширования метрик):
```sql
CREATE TABLE portfolio_metrics_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    total_exposure_usdt DECIMAL(20, 8) NOT NULL,
    total_unrealized_pnl_usdt DECIMAL(20, 8) NOT NULL,
    total_realized_pnl_usdt DECIMAL(20, 8) NOT NULL,
    portfolio_value_usdt DECIMAL(20, 8) NOT NULL,
    open_positions_count INTEGER NOT NULL,
    calculated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL
);

CREATE INDEX idx_portfolio_metrics_expires_at ON portfolio_metrics_cache(expires_at);
```

### 8.4. Производительность

#### 8.4.1. Оптимизация запросов

- Использовать индексы на `positions.asset`, `positions.mode`
- Кэшировать агрегированные метрики в памяти (Redis опционально)
- Batch обработка обновлений позиций
- Асинхронные запросы к БД

#### 8.4.2. Масштабирование

- Горизонтальное масштабирование через несколько инстансов
- Shared state через БД и RabbitMQ
- Stateless REST API endpoints
- Кэширование метрик для снижения нагрузки на БД

## 9. Интеграции

### 9.1. Order Manager

**Интеграция**: REST API или RabbitMQ события

**Изменения в Order Manager**:
- Удалить `PositionManager` класс
- Удалить REST API endpoints для позиций
- Удалить фоновые задачи для позиций
- Заменить вызовы `PositionManager` на REST API вызовы к Position Manager
- Публиковать события обновления позиций в RabbitMQ

### 9.2. Risk Manager

**Интеграция**: REST API

**Изменения в Risk Manager**:
- Добавить вызов `GET /api/v1/portfolio/exposure` для получения `total_exposure`
- Использовать полученное значение в `check_max_exposure()`
- Добавить проверку других портфолио-лимитов

### 9.3. Model Service

**Интеграция**: REST API (основной способ) + RabbitMQ события (опционально для кэширования)

**Изменения в Model Service**:
- **Основной способ**: REST API запросы к Position Manager при генерации сигналов
  - Заменить `PositionStateRepository._get_open_positions()` на REST API запрос `GET /api/v1/positions`
  - Заменить локальные вычисления метрик (`get_total_exposure()`, `get_unrealized_pnl()`) на REST API запрос `GET /api/v1/portfolio`
  - Использовать синхронные REST запросы для получения актуальных данных при генерации сигнала
  - **Использовать готовые features из Position Manager**: `unrealized_pnl_pct`, `time_held_minutes`, `position_size_norm` — эти features уже вычислены в Position Manager и включены в ответы REST API, не нужно вычислять их локально в Model Service
  - **Реализовать правила риск-менеджмента** (см. раздел 3.4.3):
    - Take Profit: принудительный SELL при `unrealized_pnl_pct > MODEL_SERVICE_TAKE_PROFIT_PCT` (по умолчанию 3.0%)
    - Position Size Limit: пропуск BUY при `position_size_norm > MODEL_SERVICE_MAX_POSITION_SIZE_RATIO` (по умолчанию 0.8)
    - Использовать данные из Position Manager через REST API для проверки этих правил перед генерацией сигнала модели
- **RabbitMQ события для кэширования** (опционально):
  - Подписаться на события обновления позиций (`position-manager.position_updated`) для инвалидации локального кэша
  - Реализовать кэширование портфолио-метрик в памяти для уменьшения количества REST запросов
  - Кэш используется для оптимизации, но REST API остается основным источником актуальных данных
- **RabbitMQ события для исторической реконструкции** (обязательно для обучения моделей):
  - Подписаться на события создания снапшотов позиций (`position-manager.position_snapshot_created`)
  - Сохранять снапшоты в локальном хранилище (БД или файловая система) для последующего использования при обучении
  - Использовать снапшоты для восстановления состояния позиций на момент каждого execution event при построении training dataset
  - Реализовать поиск снапшотов по timestamp для точной реконструкции исторического состояния позиций
  - Интегрировать снапшоты в `DatasetBuilder` для передачи исторического `OrderPositionState` в feature engineering
- Добавить обработку ошибок при недоступности Position Manager (fallback к чтению из БД или использованию кэша)

### 9.4. WebSocket Gateway

**Интеграция**: RabbitMQ события

**Изменения**: Нет (WebSocket Gateway уже публикует события в очередь)

## 10. Безопасность

### 10.1. Аутентификация

- API Key аутентификация для всех REST API endpoints
- Валидация API Key на каждом запросе
- Логирование всех запросов с trace IDs

### 10.2. Валидация данных

- Валидация всех входных данных (Pydantic models)
- Проверка типов и диапазонов значений
- Санитизация строковых параметров

### 10.3. Обработка ошибок

- Graceful degradation при недоступности зависимостей
- Retry логика для внешних вызовов
- Детальное логирование ошибок

## 11. Мониторинг и логирование

### 11.1. Метрики

- Количество позиций
- Количество обновлений позиций в секунду
- Время ответа REST API endpoints
- Размер кэша метрик
- Количество ошибок валидации

### 11.2. Логирование

- Структурированное логирование с trace IDs
- Логирование всех операций с позициями
- Логирование расхождений при валидации
- Логирование превышения лимитов

### 11.3. Health Checks

- Проверка подключения к БД
- Проверка подключения к RabbitMQ
- Проверка доступности зависимых сервисов

## 12. Тестирование

### 12.1. Unit тесты

- Тесты для всех методов PositionManager
- Тесты для расчета портфолио-метрик
- Тесты для валидации позиций
- Тесты для обработки событий

### 12.2. Integration тесты

- Тесты REST API endpoints
- Тесты интеграции с БД
- Тесты интеграции с RabbitMQ
- Тесты обработки событий

### 12.3. E2E тесты

- Полный поток обновления позиции из ордера
- Полный поток обновления позиции из WebSocket
- Проверка портфолио-метрик после обновлений
- Проверка валидации и исправления расхождений

### 12.4. Критически важные тест-кейсы для автотестов

Ниже приведены детальные тест-кейсы, которые должны быть реализованы как автотесты. Тесты имитируют реальные сценарии через RabbitMQ сообщения или REST API запросы.

#### 12.4.1. Обновление позиций из WebSocket событий

**TC-001: Создание новой позиции из WebSocket события**

**Предусловия**: 
- Позиция для BTCUSDT не существует в БД
- Position Manager запущен и подключен к RabbitMQ

**Шаги**:
1. Отправить RabbitMQ сообщение в очередь `ws-gateway.position`:
```json
{
  "event_type": "position",
  "channel": "position",
  "data": {
    "symbol": "BTCUSDT",
    "side": "Buy",
    "size": "1.5",
    "avgPrice": "50000.00",
    "unrealisedPnl": "150.00",
    "positionValue": "75000.00",
    "mode": "one-way",
    "timestamp": "2025-01-15T10:00:00Z"
  },
  "trace_id": "test-trace-001"
}
```

**Ожидаемый результат**:
- Позиция создана в БД с `asset="BTCUSDT"`, `size=1.5`, `average_entry_price=50000.00`
- `unrealized_pnl=150.00`
- Событие `position-manager.position_updated` опубликовано в RabbitMQ
- REST API `GET /api/v1/positions/BTCUSDT` возвращает созданную позицию
- Портфолио-метрики обновлены (`total_exposure` включает новую позицию)

**Проверки**:
- HTTP 200 на `GET /api/v1/positions/BTCUSDT`
- `position.size == "1.5"`
- `position.average_entry_price == "50000.00"`
- `position.unrealized_pnl == "150.00"`
- `position.unrealized_pnl_pct` вычислен корректно
- `position.position_size_norm` вычислен корректно

---

**TC-002: Обновление существующей позиции из WebSocket события**

**Предусловия**: 
- Позиция для BTCUSDT существует: `size=1.0`, `average_entry_price=48000.00`
- Position Manager запущен

**Шаги**:
1. Отправить RabbitMQ сообщение в очередь `ws-gateway.position`:
```json
{
  "event_type": "position",
  "channel": "position",
  "data": {
    "symbol": "BTCUSDT",
    "side": "Buy",
    "size": "2.0",
    "avgPrice": "49000.00",
    "unrealisedPnl": "200.00",
    "positionValue": "98000.00",
    "mode": "one-way",
    "timestamp": "2025-01-15T10:05:00Z"
  },
  "trace_id": "test-trace-002"
}
```

**Ожидаемый результат**:
- Позиция обновлена: `size=2.0`, `average_entry_price=49000.00` (используется `avgPrice` из WebSocket)
- `unrealized_pnl=200.00`
- Событие `position-manager.position_updated` опубликовано
- Портфолио-метрики пересчитаны

**Проверки**:
- `position.size == "2.0"`
- `position.average_entry_price == "49000.00"`
- `position.last_updated` обновлен
- `position.time_held_minutes` вычислен корректно

---

**TC-003: Обновление позиции с валидацией расхождения размера**

**Предусловия**: 
- Позиция для BTCUSDT: `size=1.5`, `average_entry_price=50000.00`
- `POSITION_MANAGER_SIZE_VALIDATION_THRESHOLD=0.0001`

**Шаги**:
1. Отправить WebSocket событие с `size=1.8` (расхождение > 0.0001):
```json
{
  "event_type": "position",
  "data": {
    "symbol": "BTCUSDT",
    "size": "1.8",
    "avgPrice": "50000.00",
    "unrealisedPnl": "180.00"
  },
  "trace_id": "test-trace-003"
}
```

**Ожидаемый результат**:
- Позиция обновлена с новым размером
- Лог предупреждения о расхождении размера
- Событие валидации может быть инициировано (если настроено)

**Проверки**:
- `position.size == "1.8"`
- В логах присутствует предупреждение о расхождении

---

#### 12.4.2. Обновление позиций из Order Manager событий

**TC-004: Обновление позиции при исполнении ордера (BUY)**

**Предусловия**: 
- Позиция для BTCUSDT: `size=1.0`, `average_entry_price=48000.00`
- Order Manager публикует событие исполнения ордера

**Шаги**:
1. Отправить RabbitMQ сообщение в очередь `order-manager.order_executed`:
```json
{
  "event_type": "order_executed",
  "order_id": "order-123",
  "asset": "BTCUSDT",
  "side": "BUY",
  "execution_price": "50000.00",
  "execution_quantity": "0.5",
  "execution_fees": "25.00",
  "executed_at": "2025-01-15T10:10:00Z",
  "trace_id": "test-trace-004"
}
```

**Ожидаемый результат**:
- Позиция обновлена: `size=1.5` (1.0 + 0.5)
- `average_entry_price` пересчитан: `(1.0 * 48000.00 + 0.5 * 50000.00) / 1.5 = 48666.67`
- `realized_pnl` не изменился (ордер открывает позицию)
- Событие `position-manager.position_updated` опубликовано

**Проверки**:
- `position.size == "1.5"`
- `position.average_entry_price == "48666.67"` (с точностью до 2 знаков)
- `position.realized_pnl` без изменений

---

**TC-005: Частичное закрытие позиции при исполнении SELL ордера**

**Предусловия**: 
- Позиция для BTCUSDT: `size=2.0`, `average_entry_price=50000.00`, `unrealized_pnl=200.00`
- Текущая цена: 51000.00

**Шаги**:
1. Отправить событие исполнения SELL ордера:
```json
{
  "event_type": "order_executed",
  "order_id": "order-124",
  "asset": "BTCUSDT",
  "side": "SELL",
  "execution_price": "51000.00",
  "execution_quantity": "0.5",
  "execution_fees": "25.50",
  "executed_at": "2025-01-15T10:15:00Z",
  "trace_id": "test-trace-005"
}
```

**Ожидаемый результат**:
- Позиция обновлена: `size=1.5` (2.0 - 0.5)
- `realized_pnl` увеличен: `(51000.00 - 50000.00) * 0.5 - 25.50 = 474.50`
- `unrealized_pnl` пересчитан для оставшейся позиции
- Событие обновления опубликовано

**Проверки**:
- `position.size == "1.5"`
- `position.realized_pnl` увеличен на ~474.50
- `position.unrealized_pnl` пересчитан

---

**TC-006: Полное закрытие позиции при исполнении SELL ордера**

**Предусловия**: 
- Позиция для BTCUSDT: `size=1.0`, `average_entry_price=50000.00`

**Шаги**:
1. Отправить событие исполнения SELL ордера с `execution_quantity=1.0`:
```json
{
  "event_type": "order_executed",
  "asset": "BTCUSDT",
  "side": "SELL",
  "execution_price": "51000.00",
  "execution_quantity": "1.0",
  "execution_fees": "51.00",
  "trace_id": "test-trace-006"
}
```

**Ожидаемый результат**:
- Позиция закрыта: `size=0.0` или позиция удалена/помечена как закрытая
- `realized_pnl` финализирован
- Событие `position-manager.position_closed` опубликовано (если реализовано)
- Портфолио-метрики обновлены (позиция исключена из exposure)

**Проверки**:
- `GET /api/v1/positions/BTCUSDT` возвращает `size=0.0` или 404
- Портфолио не включает закрытую позицию

---

#### 12.4.3. Разрешение конфликтов между источниками

**TC-007: Конфликт размера позиции (WebSocket vs Order Manager)**

**Предусловия**: 
- Позиция для BTCUSDT: `size=1.0`, `average_entry_price=50000.00`
- `POSITION_MANAGER_USE_WS_AVG_PRICE=true`
- `POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD=0.001`

**Шаги**:
1. Отправить WebSocket событие: `size=1.2`, `avgPrice=50000.00`
2. Сразу отправить Order Manager событие: `execution_quantity=0.3` (ожидаемый размер 1.3)

**Ожидаемый результат**:
- Позиция обновлена с учетом обоих источников
- `average_entry_price` использует `avgPrice` из WebSocket (если расхождение < 0.1%)
- Размер синхронизирован (приоритет Order Manager для размера)
- Логи содержат информацию о разрешении конфликта

**Проверки**:
- Финальный размер позиции корректен
- `average_entry_price` соответствует стратегии разрешения конфликтов
- В логах присутствует информация о конфликте

---

**TC-008: Валидация average_entry_price с порогом расхождения**

**Предусловия**: 
- Позиция: `size=1.0`, `average_entry_price=50000.00` (из Order Manager)
- `POSITION_MANAGER_AVG_PRICE_DIFF_THRESHOLD=0.001` (0.1%)

**Шаги**:
1. Отправить WebSocket событие с `avgPrice=50050.00` (расхождение 0.1% = 0.001):
```json
{
  "event_type": "position",
  "data": {
    "symbol": "BTCUSDT",
    "avgPrice": "50050.00",
    "size": "1.0"
  },
  "trace_id": "test-trace-008"
}
```

**Ожидаемый результат**:
- Если расхождение <= 0.001: `average_entry_price` обновлен на 50050.00
- Если расхождение > 0.001: `average_entry_price` не обновлен, логируется предупреждение

**Проверки**:
- `position.average_entry_price` соответствует ожидаемому поведению
- Логи содержат информацию о валидации

---

#### 12.4.4. Расчет портфолио-метрик

**TC-009: Расчет total_exposure для множественных позиций**

**Предусловия**: 
- Позиция BTCUSDT: `size=1.0`, текущая цена 50000.00
- Позиция ETHUSDT: `size=10.0`, текущая цена 3000.00

**Шаги**:
1. Выполнить REST API запрос: `GET /api/v1/portfolio/exposure`

**Ожидаемый результат**:
- `total_exposure = 1.0 * 50000.00 + 10.0 * 3000.00 = 80000.00`
- Ответ содержит детализацию по активам

**Проверки**:
- HTTP 200
- `response.total_exposure == "80000.00"`
- `response.by_asset` содержит BTCUSDT и ETHUSDT

---

**TC-010: Расчет total_unrealized_pnl для портфолио**

**Предусловия**: 
- Позиция BTCUSDT: `unrealized_pnl=150.00`
- Позиция ETHUSDT: `unrealized_pnl=-50.00`

**Шаги**:
1. Выполнить REST API запрос: `GET /api/v1/portfolio/pnl`

**Ожидаемый результат**:
- `total_unrealized_pnl = 150.00 + (-50.00) = 100.00`
- `total_realized_pnl` суммирован из всех позиций

**Проверки**:
- HTTP 200
- `response.total_unrealized_pnl == "100.00"`
- `response.by_asset` содержит детализацию

---

**TC-011: Кэширование портфолио-метрик**

**Предусловия**: 
- `POSITION_MANAGER_METRICS_CACHE_TTL=10` секунд
- Позиция BTCUSDT существует

**Шаги**:
1. Выполнить `GET /api/v1/portfolio` (t=0s)
2. Обновить позицию через WebSocket событие (t=5s)
3. Выполнить `GET /api/v1/portfolio` (t=5s)
4. Выполнить `GET /api/v1/portfolio` (t=15s)

**Ожидаемый результат**:
- Первый запрос (t=0s): метрики из БД
- Второй запрос (t=5s): метрики из кэша (если TTL не истек) или обновленные
- Третий запрос (t=15s): метрики пересчитаны из БД (кэш истек)

**Проверки**:
- Время ответа второго запроса меньше (если кэш работает)
- Третий запрос возвращает обновленные метрики

---

#### 12.4.5. REST API endpoints

**TC-012: Получение списка позиций с фильтрацией**

**Предусловия**: 
- Позиции: BTCUSDT (size=1.0), ETHUSDT (size=10.0), SOLUSDT (size=100.0)

**Шаги**:
1. `GET /api/v1/positions`
2. `GET /api/v1/positions?asset=BTCUSDT`
3. `GET /api/v1/positions?size_min=5.0`

**Ожидаемый результат**:
- Запрос 1: все 3 позиции
- Запрос 2: только BTCUSDT
- Запрос 3: ETHUSDT и SOLUSDT (size >= 5.0)

**Проверки**:
- HTTP 200 для всех запросов
- Корректная фильтрация результатов
- Все позиции содержат вычисленные features (`unrealized_pnl_pct`, `time_held_minutes`, `position_size_norm`)

---

**TC-013: Получение портфолио с детализацией позиций**

**Предусловия**: 
- Множественные позиции существуют

**Шаги**:
1. `GET /api/v1/portfolio?include_positions=true`

**Ожидаемый результат**:
- Ответ содержит `total_exposure`, `total_unrealized_pnl`, `total_realized_pnl`
- Ответ содержит массив `positions` с детализацией всех позиций
- Каждая позиция содержит вычисленные features

**Проверки**:
- HTTP 200
- `response.positions` не пустой
- Все метрики вычислены корректно

---

**TC-014: Валидация позиции через API**

**Предусловия**: 
- Позиция BTCUSDT существует

**Шаги**:
1. `POST /api/v1/positions/BTCUSDT/validate`

**Ожидаемый результат**:
- HTTP 200 или 202 (зависит от реализации)
- Валидация выполнена (синхронизация с внешним источником, если необходимо)
- Логи содержат результаты валидации

**Проверки**:
- Позиция валидирована
- При расхождениях они исправлены или залогированы

---

**TC-015: Создание снапшота позиции через API**

**Предусловия**: 
- Позиция BTCUSDT: `size=1.0`, `unrealized_pnl=150.00`

**Шаги**:
1. `POST /api/v1/positions/BTCUSDT/snapshot`

**Ожидаемый результат**:
- HTTP 201 или 200
- Снапшот создан в БД
- Событие `position-manager.position_snapshot_created` опубликовано в RabbitMQ
- `GET /api/v1/positions/BTCUSDT/snapshots` возвращает созданный снапшот

**Проверки**:
- Снапшот сохранен в БД
- Событие опубликовано в RabbitMQ
- Снапшот содержит все поля позиции на момент создания

---

**TC-016: Аутентификация через API Key**

**Предусловия**: 
- Position Manager настроен с API Key

**Шаги**:
1. `GET /api/v1/positions` без API Key
2. `GET /api/v1/positions` с неверным API Key
3. `GET /api/v1/positions` с верным API Key

**Ожидаемый результат**:
- Запрос 1: HTTP 401 Unauthorized
- Запрос 2: HTTP 401 Unauthorized
- Запрос 3: HTTP 200 OK

**Проверки**:
- Корректная обработка отсутствующего/неверного API Key
- Логи содержат информацию о неудачных попытках аутентификации

---

#### 12.4.6. Правила риск-менеджмента (интеграция с Model Service)

**TC-017: Take Profit - принудительный SELL при достижении прибыли**

**Предусловия**: 
- Позиция BTCUSDT: `size=1.0`, `average_entry_price=50000.00`, текущая цена 51500.00
- `unrealized_pnl_pct = (51500.00 - 50000.00) / 50000.00 * 100 = 3.0%`
- `MODEL_SERVICE_TAKE_PROFIT_PCT=3.0`

**Шаги**:
1. Model Service запрашивает позицию: `GET /api/v1/positions/BTCUSDT`
2. Model Service проверяет `unrealized_pnl_pct >= 3.0%`
3. Model Service генерирует принудительный SELL сигнал

**Ожидаемый результат**:
- REST API возвращает `unrealized_pnl_pct=3.0` (или выше)
- Model Service получает данные для принятия решения
- Model Service генерирует SELL сигнал с `confidence=1.0`, `reason="take_profit_triggered"`

**Проверки**:
- `position.unrealized_pnl_pct` вычислен корректно
- Model Service может использовать это значение для правила Take Profit

---

**TC-018: Position Size Limit - пропуск BUY при большом размере позиции**

**Предусловия**: 
- Позиция BTCUSDT: `size=1.0`, текущая цена 50000.00
- `total_exposure = 100000.00`
- `position_size_norm = (1.0 * 50000.00) / 100000.00 = 0.5`
- `MODEL_SERVICE_MAX_POSITION_SIZE_RATIO=0.8`

**Шаги**:
1. Model Service запрашивает позицию: `GET /api/v1/positions/BTCUSDT`
2. Model Service проверяет `position_size_norm < 0.8`
3. Model Service генерирует BUY сигнал (размер позиции в пределах лимита)

**Ожидаемый результат**:
- REST API возвращает `position_size_norm=0.5`
- Model Service генерирует BUY сигнал (0.5 < 0.8)

**Проверки**:
- `position.position_size_norm` вычислен корректно
- Model Service может использовать это значение для правила Position Size Limit

---

**TC-019: Position Size Limit - пропуск BUY при превышении лимита**

**Предусловия**: 
- Позиция BTCUSDT: `size=2.0`, текущая цена 50000.00
- `total_exposure = 100000.00`
- `position_size_norm = (2.0 * 50000.00) / 100000.00 = 1.0`
- `MODEL_SERVICE_MAX_POSITION_SIZE_RATIO=0.8`

**Шаги**:
1. Model Service запрашивает позицию: `GET /api/v1/positions/BTCUSDT`
2. Model Service проверяет `position_size_norm > 0.8`
3. Model Service пропускает генерацию BUY сигнала

**Ожидаемый результат**:
- REST API возвращает `position_size_norm=1.0`
- Model Service пропускает BUY сигнал, логирует причину `reason="position_size_limit"`

**Проверки**:
- `position.position_size_norm=1.0` (превышает лимит)
- Model Service корректно обрабатывает превышение лимита

---

#### 12.4.7. Создание снапшотов позиций

**TC-020: Автоматическое создание снапшотов по расписанию**

**Предусловия**: 
- `POSITION_MANAGER_SNAPSHOT_INTERVAL=3600` секунд (1 час)
- Позиция BTCUSDT существует

**Шаги**:
1. Дождаться истечения интервала (или симулировать через API)
2. Проверить создание снапшота

**Ожидаемый результат**:
- Снапшот создан автоматически
- Событие `position-manager.position_snapshot_created` опубликовано в RabbitMQ
- Снапшот содержит состояние позиции на момент создания

**Проверки**:
- Снапшот в БД
- Событие в RabbitMQ
- `snapshot.position_id` соответствует позиции
- `snapshot.snapshot_data` содержит все поля позиции

---

**TC-021: Получение истории снапшотов позиции**

**Предусловия**: 
- Создано несколько снапшотов для BTCUSDT

**Шаги**:
1. `GET /api/v1/positions/BTCUSDT/snapshots`

**Ожидаемый результат**:
- HTTP 200
- Ответ содержит массив снапшотов, отсортированных по `created_at` (DESC)
- Каждый снапшот содержит полные данные позиции на момент создания

**Проверки**:
- Все снапшоты возвращены
- Снапшоты отсортированы по дате
- Данные снапшотов корректны

---

#### 12.4.8. Интеграция с RabbitMQ

**TC-022: Публикация события обновления позиции**

**Предусловия**: 
- Позиция BTCUSDT обновлена через WebSocket событие

**Шаги**:
1. Отправить WebSocket событие обновления позиции
2. Проверить сообщение в очереди `position-manager.position_updated`

**Ожидаемый результат**:
- Событие опубликовано в RabbitMQ
- Формат события соответствует спецификации (раздел 6.1)
- Событие содержит все необходимые поля, включая вычисленные features

**Проверки**:
- Сообщение в очереди RabbitMQ
- `event.position.unrealized_pnl_pct` присутствует
- `event.position.time_held_minutes` присутствует
- `event.position.position_size_norm` присутствует

---

**TC-023: Публикация события обновления портфолио**

**Предусловия**: 
- Позиция обновлена, портфолио-метрики пересчитаны

**Шаги**:
1. Обновить позицию
2. Проверить сообщение в очереди `position-manager.portfolio_updated`

**Ожидаемый результат**:
- Событие опубликовано в RabbitMQ
- Формат события соответствует спецификации (раздел 6.2)
- Событие содержит `total_exposure`, `total_unrealized_pnl`, `total_realized_pnl`

**Проверки**:
- Сообщение в очереди RabbitMQ
- Все метрики портфолио присутствуют
- Метрики вычислены корректно

---

**TC-024: Публикация события создания снапшота**

**Предусловия**: 
- Снапшот позиции создан (автоматически или через API)

**Шаги**:
1. Создать снапшот через API или дождаться автоматического создания
2. Проверить сообщение в очереди `position-manager.position_snapshot_created`

**Ожидаемый результат**:
- Событие опубликовано в RabbitMQ
- Формат события соответствует спецификации (раздел 6.3)
- Событие содержит полные данные снапшота

**Проверки**:
- Сообщение в очереди RabbitMQ
- `event.snapshot.snapshot_data` содержит все поля позиции
- `event.snapshot.created_at` присутствует

---

#### 12.4.9. Валидация позиций

**TC-025: Периодическая валидация позиций**

**Предусловия**: 
- `POSITION_MANAGER_VALIDATION_INTERVAL=1800` секунд (30 минут)
- Позиции существуют в БД

**Шаги**:
1. Дождаться истечения интервала (или симулировать через API)
2. Проверить выполнение валидации

**Ожидаемый результат**:
- Валидация выполнена для всех позиций
- Расхождения обнаружены и исправлены (если есть)
- Логи содержат результаты валидации

**Проверки**:
- Валидация выполнена
- При расхождениях они исправлены или залогированы
- Позиции синхронизированы с внешними источниками (если настроено)

---

**TC-026: Валидация позиции с обнаружением расхождений**

**Предусловия**: 
- Позиция BTCUSDT в БД: `size=1.0`, `average_entry_price=50000.00`
- Внешний источник (WebSocket/API) сообщает: `size=1.2`, `avgPrice=51000.00`

**Шаги**:
1. Выполнить валидацию: `POST /api/v1/positions/BTCUSDT/validate`
2. Проверить синхронизацию с внешним источником

**Ожидаемый результат**:
- Расхождения обнаружены
- Позиция синхронизирована с внешним источником (если настроено)
- Логи содержат информацию о расхождениях и исправлениях

**Проверки**:
- Позиция обновлена с учетом внешнего источника
- Логи содержат детали расхождений

---

#### 12.4.10. Обработка ошибок и граничные случаи

**TC-027: Обработка недоступности БД**

**Предусловия**: 
- Position Manager запущен
- БД недоступна (симулировать отключение)

**Шаги**:
1. Выполнить REST API запрос: `GET /api/v1/positions`

**Ожидаемый результат**:
- HTTP 503 Service Unavailable или HTTP 500
- Health check `/health` возвращает `database_connected=false`
- Логи содержат информацию об ошибке подключения к БД

**Проверки**:
- Корректная обработка ошибки БД
- Health check отражает состояние

---

**TC-028: Обработка недоступности RabbitMQ**

**Предусловия**: 
- Position Manager запущен
- RabbitMQ недоступен (симулировать отключение)

**Шаги**:
1. Попытаться обновить позицию через WebSocket событие
2. Проверить health check

**Ожидаемый результат**:
- Позиция обновлена в БД (если возможно)
- Публикация события в RabbitMQ пропущена или отложена
- Health check возвращает `queue_connected=false`
- Логи содержат информацию об ошибке RabbitMQ

**Проверки**:
- Позиция обновлена в БД
- Health check отражает состояние RabbitMQ

---

**TC-029: Обработка некорректных данных в событиях**

**Предусловия**: 
- Position Manager запущен

**Шаги**:
1. Отправить RabbitMQ сообщение с некорректными данными:
```json
{
  "event_type": "position",
  "data": {
    "symbol": "INVALID",
    "size": "not-a-number",
    "avgPrice": null
  }
}
```

**Ожидаемый результат**:
- Сообщение отклонено (валидация Pydantic)
- Ошибка залогирована
- Позиция не обновлена
- Событие не опубликовано

**Проверки**:
- Валидация данных работает
- Некорректные данные не обрабатываются
- Логи содержат информацию об ошибке валидации

---

**TC-030: Обработка дублирующихся событий**

**Предусловия**: 
- Позиция BTCUSDT существует
- `trace_id` используется для дедупликации (если реализовано)

**Шаги**:
1. Отправить WebSocket событие с `trace_id="dup-001"`
2. Отправить то же событие повторно с тем же `trace_id`

**Ожидаемый результат**:
- Первое событие обработано
- Второе событие пропущено (дедупликация) или обработано (если дедупликация не реализована)
- Логи содержат информацию о дубликате (если применимо)

**Проверки**:
- Дедупликация работает (если реализована)
- Позиция обновлена только один раз (если дедупликация есть)

---

#### 12.4.11. Производительность и нагрузочное тестирование

**TC-031: Обработка множественных одновременных обновлений**

**Предусловия**: 
- Позиции для 10 различных активов существуют

**Шаги**:
1. Отправить 100 WebSocket событий одновременно (по 10 для каждого актива)
2. Измерить время обработки
3. Проверить корректность финального состояния позиций

**Ожидаемый результат**:
- Все события обработаны
- Время обработки приемлемо (< 5 секунд для 100 событий)
- Финальное состояние позиций корректно
- Нет потери данных

**Проверки**:
- Все события обработаны
- Производительность в пределах ожиданий
- Данные консистентны

---

**TC-032: Нагрузка на REST API endpoints**

**Предусловия**: 
- Множественные позиции существуют

**Шаги**:
1. Выполнить 1000 последовательных запросов `GET /api/v1/portfolio`
2. Измерить время ответа и использование кэша

**Ожидаемый результат**:
- Все запросы обработаны
- Среднее время ответа < 100ms (с кэшем)
- Кэш эффективно используется

**Проверки**:
- Производительность REST API в пределах ожиданий
- Кэш работает эффективно

---

### 12.5. Инструменты и инфраструктура для автотестов

**Рекомендуемые инструменты**:
- **pytest** для Python тестов
- **pytest-asyncio** для асинхронных тестов
- **pytest-rabbitmq** или **aio-pika** для тестирования RabbitMQ
- **httpx** или **requests** для HTTP запросов
- **testcontainers** для изоляции БД и RabbitMQ в тестах
- **faker** для генерации тестовых данных

**Структура тестов**:
```
position-manager/tests/
├── unit/           # Unit тесты для отдельных компонентов
├── integration/     # Integration тесты с БД и RabbitMQ
│   ├── test_rabbitmq_events.py
│   ├── test_rest_api.py
│   └── test_position_updates.py
└── e2e/            # End-to-end тесты полных сценариев
    ├── test_websocket_flow.py
    ├── test_order_manager_flow.py
    └── test_portfolio_metrics.py
```

**Запуск тестов**:
- Unit тесты: `pytest position-manager/tests/unit/`
- Integration тесты: `pytest position-manager/tests/integration/`
- E2E тесты: `pytest position-manager/tests/e2e/`
- Все тесты: `pytest position-manager/tests/`

**CI/CD интеграция**:
- Автоматический запуск тестов при каждом коммите
- Запуск полного набора тестов перед деплоем
- Отчеты о покрытии кода тестами

## 13. Этапы реализации

### Этап 1: Подготовка (1-2 дня)
- Создание структуры проекта
- Настройка инфраструктуры
- Создание базовых моделей

### Этап 2: Извлечение функциональности (3-5 дней)
- Извлечение PositionManager
- Извлечение REST API endpoints
- Извлечение фоновых задач
- Адаптация под новую структуру

### Этап 3: Новая функциональность (3-5 дней)
- Реализация PortfolioManager
- Реализация расчета портфолио-метрик
- Реализация новых REST API endpoints
- Реализация кэширования метрик

### Этап 4: Интеграции (2-3 дня)
- Интеграция с Order Manager
- Интеграция с Risk Manager
- Интеграция с Model Service
- Настройка RabbitMQ consumers

### Этап 5: Тестирование (2-3 дня)
- Unit тесты
- Integration тесты
- E2E тесты
- Исправление багов

### Этап 6: Документация и развертывание (1-2 дня)
- Обновление документации
- Развертывание в тестовой среде
- Валидация работы
- Развертывание в production

**Общая оценка**: 12-20 дней разработки

## 14. Риски и митигация

### 14.1. Риски

1. **Разрыв функциональности при миграции**
   - Митигация: Постепенная миграция, параллельная работа старого и нового сервиса

2. **Производительность при расчете метрик**
   - Митигация: Кэширование, оптимизация запросов, индексы в БД

3. **Консистентность данных между сервисами**
   - Митигация: Единая БД, транзакции, валидация

4. **Сложность интеграции с существующими сервисами**
   - Митигация: Постепенная миграция, обратная совместимость API

### 14.2. Откат

- Возможность отката к старой архитектуре
- Сохранение старого кода до полной валидации
- Постепенное переключение трафика

## 15. Дальнейшее развитие

### 15.1. Потенциальные улучшения

- Исторический анализ портфолио
- Аналитика и отчеты
- Алерты при превышении лимитов
- Оптимизация портфолио (ребалансировка)
- Интеграция с внешними аналитическими инструментами

### 15.2. Масштабирование

- Горизонтальное масштабирование
- Использование Redis для кэширования
- Оптимизация запросов к БД
- Асинхронная обработка событий

## 16. Заключение

Создание отдельного микросервиса Position Manager обеспечит:

- Централизованное управление позициями и портфолио
- Улучшенную масштабируемость и производительность
- Четкое разделение ответственности между сервисами
- Расширяемую архитектуру для будущих улучшений

Миграция должна быть выполнена постепенно с тщательным тестированием на каждом этапе.

