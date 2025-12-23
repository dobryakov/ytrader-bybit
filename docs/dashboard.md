# Техническое задание: Dashboard для визуализации данных торговой системы

**Дата создания**: 2025-01-XX  
**Статус**: Draft  
**Приоритет**: P2  
**Зависимости**: PostgreSQL, Order Manager, Position Manager, Model Service

## 1. Обзор

### 1.1. Цель проекта

Создать веб-дашборд для визуализации данных торговой системы с возможностью просмотра позиций, ордеров, торговых сигналов, моделей и аналитических метрик в режиме реального времени.

### 1.2. Архитектура решения

Решение состоит из двух компонентов:

1. **Dashboard API** (Backend) - FastAPI сервис для получения данных из БД
2. **Dashboard Frontend** (Frontend) - React приложение для визуализации данных

### 1.3. Преимущества решения

- **Прямой доступ к БД**: быстрые запросы без промежуточных API-слоев
- **Минимальная зависимость**: независимость от других сервисов (кроме БД)
- **Гибкость запросов**: возможность сложных SQL-запросов для аналитики
- **Исторические данные**: доступ ко всем историческим данным (snapshots, events)
- **Real-time обновления**: polling для критичных данных

## 2. Архитектура

### 2.1. Backend (Dashboard API)

#### 2.1.1. Технологии

- **Framework**: FastAPI (Python)
- **База данных**: PostgreSQL (asyncpg для асинхронных запросов)
- **Порт**: 4050
- **Контейнер**: `dashboard-api`

#### 2.1.2. Подключение к БД

Прямое подключение к общей PostgreSQL БД проекта. Чтение данных из таблиц:

- `positions` - текущие и исторические позиции
- `orders` - ордера и их статусы
- `trading_signals` - торговые сигналы
- `model_versions` + `model_quality_metrics` - модели и метрики качества
- `position_snapshots` - история позиций
- `execution_events` - события исполнения
- `account_balances` - балансы аккаунта
- `position_orders` - связь позиций и ордеров

#### 2.1.3. API Endpoints

##### Позиции

- `GET /api/v1/positions` - список позиций с фильтрацией
  - Query параметры: `asset`, `mode`, `size_min`, `size_max`
  - Возвращает: массив позиций с полями (id, asset, size, average_entry_price, current_price, unrealized_pnl, realized_pnl, mode, etc.)

- `GET /api/v1/positions/{asset}` - детали позиции по asset
  - Возвращает: полная информация о позиции + связанные ордера

- `GET /api/v1/positions/{asset}/history` - история позиции
  - Query параметры: `date_from`, `date_to`
  - Возвращает: данные из `position_snapshots` для построения графика

##### Ордера

- `GET /api/v1/orders` - список ордеров с пагинацией
  - Query параметры: `asset`, `status`, `signal_id`, `side`, `date_from`, `date_to`, `page`, `page_size`, `sort_by`, `sort_order`
  - Возвращает: массив ордеров с пагинацией

- `GET /api/v1/orders/{order_id}` - детали ордера
  - Возвращает: полная информация об ордере + связанные execution_events

##### Сигналы

- `GET /api/v1/signals` - история торговых сигналов
  - Query параметры: `signal_type`, `asset`, `strategy_id`, `date_from`, `date_to`, `page`, `page_size`
  - Возвращает: массив сигналов

- `GET /api/v1/signals/{signal_id}` - детали сигнала
  - Возвращает: полная информация о сигнале + связанные ордера

##### Модели

- `GET /api/v1/models` - список моделей
  - Query параметры: `symbol`, `strategy_id`, `is_active`
  - Возвращает: массив моделей с метриками качества

- `GET /api/v1/models/{model_id}` - детали модели
  - Возвращает: полная информация о модели + метрики качества

##### Метрики

- `GET /api/v1/metrics/overview` - агрегированные метрики
  - Возвращает: баланс, unrealized PnL, realized PnL, количество открытых позиций, total exposure

- `GET /api/v1/metrics/portfolio` - метрики портфолио
  - Возвращает: распределение по активам, общий PnL, exposure по активам

##### Графики

- `GET /api/v1/charts/pnl` - данные для графика PnL
  - Query параметры: `date_from`, `date_to`, `interval` (1h, 4h, 1d)
  - Возвращает: временной ряд PnL (realized + unrealized)

- `GET /api/v1/charts/positions-history` - история позиций
  - Query параметры: `asset`, `date_from`, `date_to`
  - Возвращает: временной ряд размеров позиций

- `GET /api/v1/charts/signals-confidence` - график confidence сигналов
  - Query параметры: `asset`, `strategy_id`, `date_from`, `date_to`
  - Возвращает: временной ряд confidence по времени

#### 2.1.4. Аутентификация

- **Метод**: API Key через заголовок `X-API-Key`
- **Переменная окружения**: `DASHBOARD_API_KEY`
- **Middleware**: проверка ключа для всех `/api/*` endpoints
- Исключения: `/health`, `/live`, `/ready`

#### 2.1.5. Кэширование (опционально)

- Redis для тяжелых запросов (агрегированные метрики, графики)
- TTL: 10-30 секунд для real-time данных
- TTL: 60-300 секунд для исторических данных

### 2.2. Frontend (React Dashboard)

#### 2.2.1. Технологии

- **Framework**: React 18 + TypeScript
- **Build tool**: Vite
- **UI библиотека**: shadcn/ui
- **Графики**: Recharts
- **State management**: React Query (TanStack Query)
- **HTTP client**: Axios
- **Порт**: 4051 (dev), проксирование на API (4050)
- **Контейнер**: `dashboard-frontend`

#### 2.2.2. shadcn/ui компоненты

Основные компоненты для использования:

- `chart` - графики PnL, цен, истории сделок
- `table` - таблицы ордеров, позиций, сигналов
- `card` - карточки метрик (баланс, открытые позиции, прибыль)
- `tabs` - разделы навигации (Обзор, Позиции, Ордера, Сигналы, Модели)
- `badge` - отображение статусов (active, filled, cancelled, etc.)
- `skeleton` - индикаторы загрузки
- `button`, `input`, `select` - формы и фильтры
- `dialog` - модальные окна для деталей
- `dropdown-menu` - меню действий

#### 2.2.3. Структура страниц

##### Главная страница (Overview) - `/`

**Ключевые метрики (карточки):**
- Баланс (из `account_balances`)
- Unrealized PnL (сумма из `positions`)
- Realized PnL (сумма из `positions`)
- Количество открытых позиций
- Total exposure

**График PnL:**
- Временной ряд realized + unrealized PnL
- Интервалы: 1h, 4h, 1d, 7d, 30d
- Использует endpoint `/api/v1/charts/pnl`

**Последние сигналы:**
- Топ-10 последних сигналов
- Отображение: asset, signal_type, confidence, timestamp
- Ссылка на детали сигнала

**Активные модели:**
- Список активных моделей (`is_active = true`)
- Отображение: version, symbol, strategy_id, последние метрики

##### Позиции - `/positions`

**Таблица позиций:**
- Колонки: Asset, Size, Entry Price, Current Price, Unrealized PnL, Realized PnL, Mode, Last Updated
- Фильтры:
  - Asset (select)
  - Mode (one-way/hedge)
  - Size range (min/max)
- Сортировка по всем колонкам
- Клик на строку → детали позиции

**Детали позиции:**
- Полная информация о позиции
- История изменений (график из `position_snapshots`)
- Связанные ордера

##### Ордера - `/orders`

**Таблица ордеров:**
- Колонки: Order ID, Asset, Side, Type, Quantity, Price, Status, Filled, Avg Price, Fees, Created At
- Фильтры:
  - Asset (select)
  - Status (select: pending, filled, cancelled, etc.)
  - Side (Buy/Sell)
  - Date range (date_from, date_to)
  - Signal ID
- Пагинация (page, page_size)
- Сортировка (created_at, updated_at, executed_at)
- Клик на строку → детали ордера

**Детали ордера:**
- Полная информация об ордере
- Связанные execution_events (история исполнения)

##### Сигналы - `/signals`

**Таблица сигналов:**
- Колонки: Signal ID, Type, Asset, Amount, Confidence, Strategy, Model Version, Timestamp, Status
- Фильтры:
  - Signal type (buy/sell)
  - Asset (select)
  - Strategy ID
  - Date range
- График confidence по времени (для выбранного asset/strategy)
- Клик на строку → детали сигнала

**Детали сигнала:**
- Полная информация о сигнале
- Связанные ордера
- Market data snapshot

##### Модели - `/models`

**Таблица моделей:**
- Колонки: Version, Symbol, Strategy, Type, Trained At, Is Active, Metrics (f1, precision, recall)
- Фильтры:
  - Symbol (select)
  - Strategy ID
  - Is Active
- Сортировка по trained_at
- Клик на строку → детали модели

**Детали модели:**
- Полная информация о модели
- Метрики качества (график метрик по версиям)
- Training config
- История тренировок

#### 2.2.4. Real-time обновления

- **Polling**: обновление критичных данных каждые 5-10 секунд
  - Позиции (Overview, Positions)
  - Баланс (Overview)
  - Активные ордера (Orders)
- **Опционально**: WebSocket для real-time обновлений (будущее улучшение)

#### 2.2.5. Управление состоянием

- **React Query**: кэширование данных, автоматический refetch, синхронизация
- **Конфигурация**:
  - `staleTime`: 5-10 секунд для real-time данных
  - `cacheTime`: 5 минут
  - `refetchInterval`: 10 секунд для критичных данных

### 2.3. Docker конфигурация

#### 2.3.1. dashboard-api контейнер

```yaml
dashboard-api:
  build:
    context: ./dashboard-api
    dockerfile: Dockerfile
  container_name: dashboard-api
  ports:
    - "127.0.0.1:${DASHBOARD_API_PORT:-4050}:4050"
  volumes:
    - ./dashboard-api/src:/app/src
    - ./dashboard-api/migrations:/app/migrations
    - ./dashboard-api/tests:/app/tests
  environment:
    - POSTGRES_HOST=${POSTGRES_HOST:-postgres}
    - POSTGRES_PORT=${POSTGRES_PORT:-5432}
    - POSTGRES_DB=${POSTGRES_DB:-ytrader}
    - POSTGRES_USER=${POSTGRES_USER:-ytrader}
    - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    - DASHBOARD_API_PORT=${DASHBOARD_API_PORT:-4050}
    - DASHBOARD_API_KEY=${DASHBOARD_API_KEY}
    - REDIS_HOST=${REDIS_HOST:-redis}
    - REDIS_PORT=${REDIS_PORT:-6379}
    - REDIS_PASSWORD=${REDIS_PASSWORD:-}
  depends_on:
    postgres:
      condition: service_healthy
    redis:
      condition: service_healthy
  networks:
    - ytrader-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:4050/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 40s
```

#### 2.3.2. dashboard-frontend контейнер

```yaml
dashboard-frontend:
  build:
    context: ./dashboard-frontend
    dockerfile: Dockerfile
  container_name: dashboard-frontend
  ports:
    - "127.0.0.1:${DASHBOARD_FRONTEND_PORT:-4051}:4051"
  volumes:
    - ./dashboard-frontend/src:/app/src
    - ./dashboard-frontend/public:/app/public
  environment:
    # VITE_API_URL не устанавливается - используется относительный путь /api
    # который проксируется через Vite на dashboard-api:4050
    - VITE_API_KEY=${DASHBOARD_API_KEY}
  depends_on:
    - dashboard-api
  networks:
    - ytrader-network
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:4051"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### 2.4. Конфигурация в .env

```env
# Dashboard API
DASHBOARD_API_PORT=4050
DASHBOARD_API_KEY=XXXX

# Dashboard Frontend
DASHBOARD_FRONTEND_PORT=4051
```

**Примечание**: API ключи для доступа к другим сервисам не требуются, так как используется прямой доступ к БД.

## 3. Структура проекта

### 3.1. dashboard-api

```
dashboard-api/
├── src/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── positions.py       # Endpoints для позиций
│   │   │   ├── orders.py          # Endpoints для ордеров
│   │   │   ├── signals.py         # Endpoints для сигналов
│   │   │   ├── models.py          # Endpoints для моделей
│   │   │   ├── metrics.py         # Endpoints для метрик
│   │   │   └── charts.py          # Endpoints для графиков
│   │   ├── middleware/
│   │   │   └── auth.py            # API Key middleware
│   │   └── main.py                # FastAPI app
│   ├── db/
│   │   ├── connection.py          # Подключение к PostgreSQL
│   │   └── queries.py             # SQL запросы
│   ├── models/
│   │   ├── position.py            # Pydantic модели
│   │   ├── order.py
│   │   ├── signal.py
│   │   └── model.py
│   ├── services/
│   │   ├── position_service.py    # Бизнес-логика для позиций
│   │   ├── order_service.py
│   │   └── metrics_service.py     # Расчет метрик
│   └── config/
│       └── settings.py            # Настройки из env
├── migrations/                    # SQL миграции (если нужны views/индексы)
├── tests/
│   ├── test_api/
│   └── test_db/
├── Dockerfile
└── requirements.txt
```

### 3.2. dashboard-frontend

```
dashboard-frontend/
├── src/
│   ├── components/
│   │   ├── ui/                    # shadcn компоненты
│   │   ├── charts/
│   │   │   ├── PnLChart.tsx
│   │   │   ├── PositionsHistoryChart.tsx
│   │   │   └── SignalsConfidenceChart.tsx
│   │   ├── tables/
│   │   │   ├── PositionsTable.tsx
│   │   │   ├── OrdersTable.tsx
│   │   │   ├── SignalsTable.tsx
│   │   │   └── ModelsTable.tsx
│   │   └── metrics/
│   │       ├── MetricCard.tsx
│   │       └── OverviewMetrics.tsx
│   ├── pages/
│   │   ├── Overview.tsx
│   │   ├── Positions.tsx
│   │   ├── Orders.tsx
│   │   ├── Signals.tsx
│   │   └── Models.tsx
│   ├── hooks/
│   │   ├── usePositions.ts
│   │   ├── useOrders.ts
│   │   ├── useSignals.ts
│   │   └── useModels.ts
│   ├── lib/
│   │   ├── api.ts                 # Axios клиент
│   │   ├── queryClient.ts         # React Query конфигурация
│   │   └── utils.ts
│   ├── App.tsx
│   └── main.tsx
├── public/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
└── Dockerfile
```

## 4. SQL запросы (примеры)

### 4.1. Получение позиций с фильтрацией

```sql
SELECT 
    id, asset, size, average_entry_price, current_price,
    unrealized_pnl, realized_pnl, mode, long_size, short_size,
    last_updated, created_at, closed_at
FROM positions
WHERE 
    ($1::varchar IS NULL OR asset = $1)
    AND ($2::varchar IS NULL OR mode = $2)
    AND ($3::decimal IS NULL OR ABS(size) >= $3)
    AND ($4::decimal IS NULL OR ABS(size) <= $4)
ORDER BY last_updated DESC;
```

### 4.2. Агрегированные метрики

```sql
SELECT 
    COALESCE(SUM(unrealized_pnl), 0) as total_unrealized_pnl,
    COALESCE(SUM(realized_pnl), 0) as total_realized_pnl,
    COUNT(*) FILTER (WHERE size != 0) as open_positions_count,
    COUNT(*) as total_positions_count
FROM positions;
```

### 4.3. PnL временной ряд

```sql
SELECT 
    DATE_TRUNC($1, timestamp) as time_bucket,
    SUM(unrealized_pnl) as unrealized_pnl,
    SUM(realized_pnl) as realized_pnl
FROM position_snapshots
WHERE timestamp BETWEEN $2 AND $3
GROUP BY time_bucket
ORDER BY time_bucket;
```

### 4.4. Ордера с пагинацией

```sql
SELECT 
    id, order_id, signal_id, asset, side, order_type,
    quantity, price, status, filled_quantity, average_price,
    fees, created_at, updated_at
FROM orders
WHERE 
    ($1::varchar IS NULL OR asset = $1)
    AND ($2::varchar IS NULL OR status = $2)
    AND ($3::timestamp IS NULL OR created_at >= $3)
    AND ($4::timestamp IS NULL OR created_at <= $4)
ORDER BY created_at DESC
LIMIT $5 OFFSET $6;
```

## 5. Преимущества прямого доступа к БД

1. **Производительность**: прямой SQL без промежуточных API-слоев
2. **Гибкость**: возможность сложных JOIN-запросов и агрегаций
3. **Независимость**: минимальная зависимость от других сервисов
4. **Исторические данные**: полный доступ ко всем таблицам (snapshots, events)
5. **Оптимизация**: возможность создания индексов и views специально для дашборда

## 6. Миграции БД

Если нужны оптимизации для дашборда (views, индексы), можно создать миграции:

- Вариант 1: миграции в `dashboard-api/migrations/`
- Вариант 2: добавить в `ws-gateway/migrations/` (если там централизованы все миграции)

**Примеры возможных оптимизаций:**

- View для агрегированных метрик
- Индексы на часто фильтруемые поля (`orders.created_at`, `positions.asset`)
- Materialized views для тяжелых агрегаций (опционально)

## 7. Безопасность

1. **API Key**: все endpoints требуют валидный `X-API-Key`
2. **Read-only доступ**: дашборд API должен иметь только права на чтение (SELECT)
3. **Валидация входных данных**: проверка всех query параметров (SQL injection prevention)
4. **Rate limiting**: ограничение количества запросов (опционально)

## 8. Тестирование

### 8.1. Backend тесты

- Unit тесты для SQL запросов
- Integration тесты для API endpoints
- Тесты аутентификации

### 8.2. Frontend тесты

- Unit тесты для компонентов
- E2E тесты для основных сценариев (Playwright)

## 9. Мониторинг

- Health check endpoints: `/health`, `/live`, `/ready`
- Логирование запросов (структурированные логи)
- Метрики производительности (опционально)

## 10. Будущие улучшения

1. **WebSocket**: real-time обновления через WebSocket вместо polling
2. **Экспорт данных**: экспорт таблиц в CSV/Excel
3. **Алерты**: настройка алертов на метрики (email, webhook)
4. **Дашборды**: пользовательские дашборды с настраиваемыми виджетами
5. **Аналитика**: расширенная аналитика (коэффициенты, распределения, корреляции)

