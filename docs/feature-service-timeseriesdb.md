# План реализации: Time Series DB для Feature Service

**Дата**: 2025-12-25  
**Цель**: Решить проблему высокого потребления памяти и блокировок CPU при билде датасетов через использование Time Series Database

## Проблема

### Текущая ситуация

1. **Высокое потребление памяти**: Весь датасет (17k+ записей) удерживается в памяти через `pd.concat()`
2. **Блокировки CPU**: Операции `concat`, `merge`, `sort` блокируют event loop
3. **Медленная обработка запросов**: Во время билда Feature Service не отвечает на запросы скачивания датасетов
4. **Ограниченная масштабируемость**: Невозможно обрабатывать несколько датасетов параллельно

### Корневая причина

Текущая архитектура: **собрать всё → обработать → разделить → записать**

```python
# Текущий подход
all_features = []
for day in days:
    all_features.append(process_day(day))
features_df = pd.concat(all_features)  # Вся память!
merged = features_df.merge(targets_df)  # Еще больше памяти!
splits = split_by_periods(merged)  # Вся память!
write_splits(splits)
```

## Текущая архитектура хранения данных

### Текущее состояние

**Raw data (исходные данные)**:
- Хранятся в **Parquet файлах** на диске
- Типы данных: klines, trades, orderbook snapshots/deltas, ticker, funding
- Структура: `/data/raw/{symbol}/{date}/{data_type}.parquet`
- Retention: 90 дней (настраивается)

**Features и targets (вычисленные данные)**:
- Вычисляются из raw data при билде датасета
- Хранятся временно в памяти, затем экспортируются в **Parquet файлы** (датасеты)

### Проблемы текущего подхода

**Текущий пайплайн**:
```
Raw data (Parquet) 
  → Чтение из Parquet (в память)
  → Вычисление features (в память)
  → pd.concat() - вся память!
  → Split (в память)
  → Экспорт в Parquet (датасеты)
```

**Проблемы**:
1. **Высокое потребление памяти**: Весь датасет в RAM через `pd.concat()`
2. **Двойное чтение**: Raw data читаются из Parquet при каждом билде
3. **Неэффективность**: Parquet → память → Parquet (лишние шаги)
4. **Масштабируемость**: При частых пересозданиях датасетов многократное чтение и обработка

## Решение: Time Series DB

### Выбор технологии

**Рекомендация: TimescaleDB (PostgreSQL extension)**

**Обоснование**:
- ✅ Уже используем PostgreSQL (shared database)
- ✅ Минимальные изменения в архитектуре
- ✅ SQL интерфейс (знакомый стек)
- ✅ Hypertables для автоматической партиционизации по времени
- ✅ Эффективные time-based запросы
- ✅ Compression для временных рядов
- ✅ Continuous aggregates для предвычисленных метрик

**Почему не ClickHouse?**
- ❌ Требует отдельный сервис (нарушает простоту архитектуры)
- ❌ Избыточен для наших объемов (17k записей, не петабайты)
- ❌ Другой SQL диалект (дополнительная сложность)
- ❌ Оптимизирован для аналитики, не для билда датасетов
- ❌ Высокие операционные затраты без достаточной выгоды
- ✅ TimescaleDB использует существующий PostgreSQL, минимальные изменения

**Другие альтернативы**:
- **InfluxDB**: Специализированная TSDB, но требует отдельного сервиса
- **QuestDB**: Быстрая, но менее зрелая экосистема

### Архитектура решения: Полная миграция на TimescaleDB

```
┌─────────────────────────────────────────────────────────┐
│                    Feature Service                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Real-time Processing (события из RabbitMQ)            │
│  └─> TimescaleDB hypertables (raw data)                │
│      ├─> Streaming write klines, trades, orderbook      │
│      └─> Streaming write ticker, funding                │
│                                                          │
│  Backfilling (исторические данные)                     │
│  └─> TimescaleDB hypertables (raw data)                │
│      └─> Streaming write из Bybit API                   │
│                                                          │
│  Dataset Building                                        │
│  └─> TimescaleDB hypertables                            │
│      ├─> SQL queries для raw data (streaming)           │
│      ├─> Вычисление features (streaming, не в память!)  │
│      ├─> SQL queries для split (не в память!)          │
│      ├─> SQL queries для targets (не в память!)       │
│      └─> Streaming export в Parquet (chunked)          │
│                                                          │
│  API Endpoints (запросы от model-service)              │
│  └─> TimescaleDB hypertables                            │
│      ├─> GET /features/latest - чтение из БД            │
│      ├─> GET /dataset/{id}/download - чтение из БД     │
│      └─> GET /targets - чтение из БД                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   TimescaleDB         │
              │   (PostgreSQL ext)    │
              │   Shared Database     │
              │                       │
              │  Raw Data:            │
              │  - klines_hypertable  │
              │  - trades_hypertable  │
              │  - orderbook_hypertable│
              │  - ticker_hypertable  │
              │  - funding_hypertable│
              │                       │
              │  Computed:            │
              │  - features_hypertable│
              │  - targets_hypertable │
              │                       │
              │  - Compression        │
              │  - Continuous aggs   │
              └───────────────────────┘
```

**Принципы полной миграции**:
1. **Все данные в TimescaleDB**: Raw data, features и targets хранятся в hypertables
2. **Real-time processing**: События из RabbitMQ пишутся напрямую в TimescaleDB (не в Parquet)
3. **Backfilling**: Исторические данные из Bybit API пишутся напрямую в TimescaleDB (не в Parquet)
4. **API endpoints**: Все запросы от model-service (features, targets, datasets) читают из TimescaleDB (не из Parquet)
5. **Streaming обработка**: Никогда не загружаем весь датасет в память
6. **SQL-first подход**: Все операции через SQL запросы (read, split, merge)
7. **Единое хранилище**: Все данные в одном месте для эффективных запросов
8. **Compression**: Автоматическое сжатие старых chunks
9. **Retention policy**: Автоматическое удаление данных старше retention period
10. **Старые данные**: Не мигрируются из Parquet, проще перебакфиллить при необходимости
11. **Parquet только для экспорта**: Parquet файлы используются ТОЛЬКО для экспорта датасетов при скачивании через `/dataset/{id}/download`. Файлы создаются на лету при запросе и не хранятся постоянно. Все остальные операции (raw data, features, targets, билд датасетов) работают исключительно с TimescaleDB.

### Структура данных в TimescaleDB

#### Hypertables для raw data

**Klines (свечи)**

```sql
CREATE TABLE klines_hypertable (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume FLOAT NOT NULL,
    turnover FLOAT,
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable(
    'klines_hypertable',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_klines_symbol_timestamp 
    ON klines_hypertable (symbol, timestamp DESC);
```

**Trades (сделки)**

```sql
CREATE TABLE trades_hypertable (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    price FLOAT NOT NULL,
    size FLOAT NOT NULL,
    side VARCHAR(10) NOT NULL,  -- 'Buy' or 'Sell'
    trade_id VARCHAR(100),
    PRIMARY KEY (timestamp, symbol, trade_id)
);

SELECT create_hypertable(
    'trades_hypertable',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_trades_symbol_timestamp 
    ON trades_hypertable (symbol, timestamp DESC);
```

**Orderbook snapshots**

```sql
CREATE TABLE orderbook_snapshots_hypertable (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    bids JSONB NOT NULL,  -- Array of [price, size]
    asks JSONB NOT NULL,  -- Array of [price, size]
    PRIMARY KEY (timestamp, symbol)
);

SELECT create_hypertable(
    'orderbook_snapshots_hypertable',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_orderbook_snapshots_symbol_timestamp 
    ON orderbook_snapshots_hypertable (symbol, timestamp DESC);
CREATE INDEX idx_orderbook_snapshots_jsonb_gin 
    ON orderbook_snapshots_hypertable USING GIN (bids, asks);
```

**Ticker и Funding** (аналогично, с соответствующими полями)

#### Hypertable для features

```sql
CREATE TABLE features_hypertable (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    feature_registry_version VARCHAR(50) NOT NULL,
    dataset_id UUID,  -- NULL для вычисленных features, UUID для датасета
    features JSONB NOT NULL,  -- Все features в JSONB
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, feature_registry_version, dataset_id)
);

SELECT create_hypertable(
    'features_hypertable',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Индексы для быстрого split
CREATE INDEX idx_features_dataset_symbol_timestamp 
    ON features_hypertable (dataset_id, symbol, timestamp DESC)
    WHERE dataset_id IS NOT NULL;
CREATE INDEX idx_features_symbol_date_version 
    ON features_hypertable (symbol, timestamp, feature_registry_version);
CREATE INDEX idx_features_jsonb_gin 
    ON features_hypertable USING GIN (features);
```

#### Hypertable для targets

```sql
CREATE TABLE targets_hypertable (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    dataset_id UUID NOT NULL,
    target FLOAT,  -- для regression
    target_class INTEGER,  -- для classification
    target_raw FLOAT,  -- исходное значение до классификации
    PRIMARY KEY (timestamp, symbol, dataset_id)
);

SELECT create_hypertable(
    'targets_hypertable',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_targets_dataset_symbol_timestamp 
    ON targets_hypertable (dataset_id, symbol, timestamp DESC);
```

#### Compression policy

```sql
-- Автоматическое сжатие chunks старше 7 дней для raw data
SELECT add_compression_policy('klines_hypertable', INTERVAL '7 days');
SELECT add_compression_policy('trades_hypertable', INTERVAL '7 days');
SELECT add_compression_policy('orderbook_snapshots_hypertable', INTERVAL '7 days');
SELECT add_compression_policy('ticker_hypertable', INTERVAL '7 days');
SELECT add_compression_policy('funding_hypertable', INTERVAL '7 days');

-- Автоматическое сжатие chunks старше 7 дней для computed data
SELECT add_compression_policy('features_hypertable', INTERVAL '7 days');
SELECT add_compression_policy('targets_hypertable', INTERVAL '7 days');
```

### Новый пайплайн билда датасетов

#### Источники raw data в TimescaleDB

**1. Real-time processing (события из RabbitMQ)**

```python
async def process_rabbitmq_events():
    # Обработка событий из очередей RabbitMQ
    # (заменяет запись в Parquet файлы)
    async for event in rabbitmq_consumer:
        if event.type == 'klines':
            await write_klines_to_hypertable(symbol, event.data)
        elif event.type == 'trades':
            await write_trades_to_hypertable(symbol, event.data)
        elif event.type == 'orderbook':
            await write_orderbook_to_hypertable(symbol, event.data)
        # ... и т.д.
```

**2. Backfilling (исторические данные из Bybit API)**

```python
async def backfill_to_timescaledb():
    # Получение исторических данных из Bybit API
    # (заменяет запись в Parquet файлы)
    for date in date_range:
        klines = await bybit_client.get_klines(symbol, date)
        await write_klines_to_hypertable(symbol, klines)
        
        trades = await bybit_client.get_trades(symbol, date)
        await write_trades_to_hypertable(symbol, trades)
        # ... и т.д.
```

**Примечание**: Старые данные из Parquet файлов не мигрируются. При необходимости их можно перебакфиллить через обновленный backfilling сервис, который будет писать напрямую в TimescaleDB.

#### Этап 1: Вычисление features из TimescaleDB (streaming)

#### Этап 2: Compute targets через SQL (streaming)

```python
async def compute_features_from_db():
    # Читаем raw data из TimescaleDB по дням (streaming)
    for day in days_to_process:
        # Читаем raw data из БД (не из Parquet!)
        raw_data = await read_raw_data_from_db(
            symbol=symbol,
            date=day,
            data_types=['klines', 'trades', 'orderbook']
        )
        
        # Вычисляем features (streaming, не в память!)
        day_features = await compute_features_streaming(raw_data)
        
        # Пишем features в БД
        await write_features_to_hypertable(
            dataset_id=dataset_id,
            symbol=symbol,
            features_df=day_features,
            feature_registry_version=feature_registry_version
        )
        
        # Высвобождаем память
        del raw_data, day_features
```

#### Этап 3: Split через SQL запросы (streaming)

```python
# Targets вычисляются через SQL запросы с использованием features из БД
# Не нужно держать весь features_df в памяти!
# Вариант A: Вычисление через SQL функции (если возможно)
# Вариант B: Streaming read features, compute targets, streaming write

async def compute_targets_streaming():
    # Читаем features по частям из БД
    async for features_chunk in read_features_streaming(dataset_id):
        # Вычисляем targets для этого chunk
        targets_chunk = compute_targets_for_chunk(features_chunk, target_config)
        
        # Сразу пишем в БД (streaming write)
        await write_targets_to_hypertable(
            dataset_id=dataset_id,
            targets_df=targets_chunk
        )
        
        # Высвобождаем память
        del features_chunk, targets_chunk
```

#### Этап 4: Streaming export в Parquet

```python
# Split выполняется через SQL, без загрузки всего датасета
# Каждый split читается streaming (chunked)

async def split_via_sql_streaming(dataset_id, periods):
    splits = {}
    
    for split_name, (start_time, end_time) in periods.items():
        # Создаем async iterator для streaming read
        split_iterator = read_merged_split_streaming(
            dataset_id=dataset_id,
            start_time=start_time,
            end_time=end_time
        )
        splits[split_name] = split_iterator
    
    return splits

# Использование:
splits = await split_via_sql_streaming(
    dataset_id=dataset_id,
    periods={
        "train": (train_start, train_end),
        "validation": (val_start, val_end),
        "test": (test_start, test_end)
    }
)

# splits = {
#     "train": AsyncIterator[pd.DataFrame],
#     "validation": AsyncIterator[pd.DataFrame],
#     "test": AsyncIterator[pd.DataFrame]
# }
```


```python
# Экспортируем сплиты в Parquet по частям (chunked write)
async def export_splits_to_parquet_streaming(splits, dataset_id, output_format):
    for split_name, split_iterator in splits.items():
        file_path = dataset_dir / f"{split_name}.{output_format}"
        
        # Используем ParquetWriter для streaming write
        writer = None
        async for chunk_df in split_iterator:
            if writer is None:
                # Инициализируем writer с первой chunk (получаем schema)
                writer = pq.ParquetWriter(
                    file_path,
                    schema=pa.Schema.from_pandas(chunk_df),
                    compression='snappy'
                )
            
            # Пишем chunk
            table = pa.Table.from_pandas(chunk_df)
            writer.write_table(table)
            
            # Высвобождаем память
            del chunk_df
        
        if writer:
            writer.close()
        
        logger.info(f"Exported {split_name} split", dataset_id=dataset_id, file_path=file_path)
```

## Компоненты, требующие обновления

### Полный список файлов для обновления

**Storage и Repository**:
- ✅ `feature-service/src/storage/timescaledb_repository.py` (новый файл)
- ⚠️ `feature-service/src/storage/parquet_storage.py` (можно оставить для обратной совместимости или удалить после миграции)

**Services**:
- ✅ `feature-service/src/services/data_storage.py` (полная переработка для TimescaleDB)
- ✅ `feature-service/src/services/backfilling_service.py` (обновить для TimescaleDB)
- ✅ `feature-service/src/services/optimized_dataset/streaming_builder.py` (полная переработка)
- ✅ `feature-service/src/services/optimized_dataset/optimized_builder.py` (полная переработка)
- ⚠️ `feature-service/src/services/optimized_dataset/daily_cache.py` (обновить Level 3 fallback `_read_from_parquet()`)
- ⚠️ `feature-service/src/services/target_computation_data.py` (обновить `find_available_data_range()` и `load_historical_data_for_target_computation()`)

**API Endpoints**:
- ✅ `feature-service/src/api/features.py` (обновить для TimescaleDB)
- ✅ `feature-service/src/api/dataset.py` (обновить для TimescaleDB, включая `/download` и `/resplit`)
- ✅ `feature-service/src/api/targets.py` (обновить `/api/v1/targets/compute` для TimescaleDB)
- ⚠️ `feature-service/src/api/raw_data.py` (новый файл для проверки доступности данных)

**Main и Configuration**:
- ⚠️ `feature-service/src/main.py` (обновить инициализацию, убрать ParquetStorage для raw data или оставить только для обратной совместимости)
- ⚠️ `feature-service/src/config/__init__.py` (возможно, убрать `FEATURE_SERVICE_RAW_DATA_PATH` или оставить для обратной совместимости)

**Примечание**: ✅ = уже учтено в плане, ⚠️ = требует дополнительного внимания

### Ключевые изменения в каждом компоненте

**DailyCache (Level 3 fallback)**:
- Текущий код: `_read_from_parquet()` читает из Parquet файлов
- Новый код: `_read_from_timescaledb()` читает из TimescaleDB hypertables
- Влияние: Fallback механизм для кэша, используется при отсутствии данных в Redis

**TargetComputationData**:
- Текущий код: `parquet_storage.read_klines()` для поиска доступных данных
- Новый код: SQL запросы к `klines_hypertable`
- Влияние: Target computation API должен работать с TimescaleDB

**Dataset Download**:
- Текущий код: Читает Parquet файлы с диска
- Новый код: Читает из TimescaleDB, экспортирует в Parquet на лету или streaming response
- Влияние: Model-service получает датасеты из БД, а не из файлов

**Dataset Resplit**:
- Текущий код: Загружает все сплиты из файлов, мерджит, пересоздает
- Новый код: Читает из TimescaleDB через SQL запросы, пересоздает сплиты
- Влияние: Быстрое пересоздание сплитов без загрузки файлов

**Raw Data Availability**:
- Текущий код: Проверка наличия Parquet файлов на диске (через `check_data_quality.py`)
- Новый код: SQL запросы к TimescaleDB для проверки наличия данных по дням
- Влияние: Быстрая проверка доступности данных перед билдом датасета, диагностика пропусков данных

## План реализации

### Этап 1: Подготовка инфраструктуры

**Задачи**:
1. Установить TimescaleDB extension в PostgreSQL (обновить Docker образ)
2. Создать миграции для всех hypertables (raw data + features + targets)
3. Настроить connection pooling (используем существующий asyncpg pool)
4. Добавить конфигурацию в `.env` и docker-compose.yml
5. Настроить TimescaleDB compression и retention policies

**Файлы**:
- `docker-compose.yml` (обновить postgres image на timescale/timescaledb)
- `ws-gateway/migrations/046_enable_timescaledb.sql`
- `ws-gateway/migrations/047_create_raw_data_hypertables.sql` (klines, trades, orderbook, ticker, funding)
- `ws-gateway/migrations/048_create_features_hypertable.sql`
- `ws-gateway/migrations/049_create_targets_hypertable.sql`
- `ws-gateway/migrations/050_add_timescaledb_compression.sql`
- `ws-gateway/migrations/051_add_timescaledb_retention_policy.sql`
- `feature-service/src/config/__init__.py` (добавить TimescaleDB настройки)

**Детали**:
- Использовать `timescale/timescaledb:latest-pg15` вместо `postgres:15-alpine`
- Hypertables для всех типов raw data (klines, trades, orderbook snapshots/deltas, ticker, funding)
- JSONB для features (гибкость)
- Compression policy: сжимать chunks старше 7 дней
- Retention policy: удалять данные старше 90 дней

**Оценка**: 4-5 часов

### Этап 2: Репозиторий для TimescaleDB

**Задачи**:
1. Создать `TimescaleDBRepository` для работы с hypertables
2. Методы для raw data:
   - `write_klines_streaming()` - batch insert klines
   - `write_trades_streaming()` - batch insert trades
   - `write_orderbook_snapshots_streaming()` - batch insert orderbook
   - `write_ticker_streaming()` - batch insert ticker
   - `write_funding_streaming()` - batch insert funding
   - `read_raw_data_streaming()` - streaming read raw data по дням
3. Методы для features и targets:
   - `write_features_streaming()` - batch insert по дням
   - `write_targets_streaming()` - batch insert по дням
   - `read_merged_split_streaming()` - streaming read с JOIN
   - `delete_dataset_data()` - удаление данных датасета
   - `get_dataset_stats()` - статистика по датасету
4. Использование asyncpg COPY для быстрой вставки
5. Использование asyncpg cursor для streaming read

**Файлы**:
- `feature-service/src/storage/timescaledb_repository.py`
- `feature-service/src/storage/__init__.py`

**Детали реализации**:
- Использовать существующий `MetadataStorage` connection pool
- COPY для batch insert (1000-5000 записей за раз)
- Cursor с prefetch для streaming read (10000 записей за chunk)
- Транзакции для атомарности операций

**Оценка**: 8-10 часов

### Этап 3: Обновление real-time processing и backfilling

**Задачи**:
1. Обновить `DataStorageService` для записи в TimescaleDB вместо Parquet
2. Обновить обработку событий из RabbitMQ для записи в TimescaleDB
3. Обновить `BackfillingService` для записи в TimescaleDB вместо Parquet
4. Убрать зависимость от Parquet файлов для raw data
5. Обновить retention policy: удаление старых данных из TimescaleDB

**Файлы**:
- `feature-service/src/services/data_storage.py` (полная переработка)
- `feature-service/src/services/backfilling_service.py` (обновить для TimescaleDB)
- `feature-service/src/services/queue_consumer.py` (обновить для TimescaleDB, если есть)

**Детали реализации**:
- **Real-time processing**: События из RabbitMQ пишутся напрямую в TimescaleDB через `TimescaleDBRepository`
- **Backfilling**: Данные из Bybit API пишутся напрямую в TimescaleDB через `TimescaleDBRepository`
- Убрать весь код записи в Parquet файлы для raw data
- Старые Parquet файлы остаются на диске, но не используются (можно удалить позже)
- При необходимости старые данные можно перебакфиллить через обновленный backfilling

**Оценка**: 6-8 часов

### Этап 4: Обновление API endpoints для работы с TimescaleDB

**Задачи**:
1. Обновить `/features/latest` endpoint для чтения из TimescaleDB
2. Обновить `/dataset/{id}/download` endpoint для чтения из TimescaleDB (streaming)
3. Обновить `/api/v1/targets/compute` endpoint для чтения из TimescaleDB
4. Обновить `/dataset/{id}/resplit` для работы с TimescaleDB
5. Создать новый endpoint `/raw-data/availability` для проверки доступности исходных данных по дням
6. Обновить все endpoints, которые читают features/targets/datasets
7. Убрать зависимость от Parquet файлов в API endpoints
8. Реализовать streaming read для больших датасетов

**Файлы**:
- `feature-service/src/api/features.py` (обновить для TimescaleDB)
- `feature-service/src/api/dataset.py` (обновить для TimescaleDB)
- `feature-service/src/api/targets.py` (обновить для TimescaleDB)
- `feature-service/src/api/raw_data.py` (новый файл для проверки доступности данных)
- `feature-service/src/services/target_computation_data.py` (обновить для TimescaleDB)

**Детали реализации**:
- **GET /features/latest**: Читать из `features_hypertable` через SQL запрос
- **GET /dataset/{id}/download**: Читать из `features_hypertable` и `targets_hypertable` через SQL запросы (streaming), экспортировать в Parquet на лету или отдавать streaming response
- **GET /raw-data/availability**: Проверка доступности исходных данных по дням (klines, trades, orderbook, ticker, funding)
- **POST /api/v1/targets/compute**: Читать klines из `klines_hypertable` вместо Parquet для вычисления targets
- **POST /dataset/{id}/resplit**: Читать из TimescaleDB, пересоздавать сплиты через SQL запросы
- Все запросы от model-service должны работать с TimescaleDB, а не с Parquet файлами
- Streaming read для больших датасетов (chunked response)
- Target computation: `find_available_data_range()` и `load_historical_data_for_target_computation()` должны читать из TimescaleDB
- **GET /raw-data/availability**: Новый endpoint для проверки доступности исходных данных по дням (klines, trades, orderbook, ticker, funding)

**Оценка**: 8-10 часов (включая новый endpoint для проверки доступности данных)

### Этап 5: Интеграция в Dataset Builder

**Задачи**:
1. Полностью переписать `StreamingDatasetBuilder` для работы с TimescaleDB
2. Удалить старый код с `pd.concat()` и in-memory обработкой
3. Изменить чтение raw data: из TimescaleDB вместо Parquet
4. Реализовать `compute_targets_streaming()` - streaming computation через БД
5. Реализовать `TimescaleDBSplitter` для time-based и walk-forward split
6. Реализовать streaming export в Parquet (chunked write)

**Файлы**:
- `feature-service/src/services/optimized_dataset/optimized_builder.py` (полная переработка)
- `feature-service/src/services/optimized_dataset/streaming_builder.py` (полная переработка)
- `feature-service/src/services/optimized_dataset/daily_cache.py` (обновить Level 3 fallback)
- `feature-service/src/services/optimized_dataset/timescaledb_splitter.py` (новый)
- `feature-service/src/services/optimized_dataset/timescaledb_target_computer.py` (новый)

**Детали реализации**:
- **Удалить** весь код с `all_features.append()` и `pd.concat()`
- **Удалить** in-memory merge и split операции
- **Удалить** чтение raw data из Parquet
- Заменить на чтение raw data из TimescaleDB (streaming)
- Заменить на `write_features_streaming()` - прямой запись в БД
- **Daily cache**: Обновить Level 3 fallback (`_read_from_parquet()`) для чтения из TimescaleDB вместо Parquet
- Targets вычисляются по chunks из БД, сразу пишутся обратно
- Split выполняется через SQL запросы с streaming read
- Export использует ParquetWriter для chunked write (только для скачивания, основное хранение в БД)
- **Dataset storage**: Датасеты хранятся в TimescaleDB, Parquet файлы создаются только на лету при запросе скачивания (опционально, можно отдавать streaming response напрямую из БД)
- **Важно**: Parquet используется ТОЛЬКО для экспорта датасетов при скачивании. Все остальные операции (чтение raw data, features, targets, билд датасетов) работают исключительно с TimescaleDB
- **Жёсткий переход**: старый код полностью удаляется, без возможности отката

**Оценка**: 14-18 часов

### Этап 6: Очистка и управление данными

**Задачи**:
1. Реализовать автоматическую очистку данных после экспорта (опционально)
2. Retention policy для старых датасетов
3. Мониторинг размера БД
4. Скрипт для миграции существующих Parquet датасетов (опционально, для тестирования)

**Файлы**:
- `feature-service/src/services/optimized_dataset/timescaledb_cleanup.py` (новый)
- `ws-gateway/migrations/050_add_timescaledb_retention_policy.sql`
- `feature-service/scripts/migrate_datasets_to_timescaledb.py` (опционально)

**Стратегия очистки**:
- **Вариант A**: Удалять данные сразу после экспорта в Parquet (экономия места)
- **Вариант B**: Хранить в БД для быстрого пересоздания сплитов (экономия времени)
- **Рекомендация**: Вариант A для production (по умолчанию), Вариант B можно включить через конфигурацию при необходимости

**Оценка**: 3-4 часа

### Этап 7: Тестирование и оптимизация

**Задачи**:
1. Unit тесты для TimescaleDB repository
2. Integration тесты для нового пайплайна билда
3. Performance тесты (память, CPU, время билда)
4. Оптимизация SQL запросов и индексов
5. Нагрузочное тестирование (параллельные билды датасетов)

**Файлы**:
- `feature-service/tests/unit/test_timescaledb_repository.py`
- `feature-service/tests/integration/test_timescaledb_build.py`
- `feature-service/tests/performance/test_build_performance.py`

**Оценка**: 6-8 часов

### Этап 8: Документация и деплой

**Задачи**:
1. Обновить документацию по билду датасетов
2. Добавить примеры SQL запросов
3. Обновить quickstart guide
4. Деплой в production

**Файлы**:
- `docs/feature-service.md` (обновить)
- `docs/feature-service-timeseriesdb.md` (этот файл)
- `specs/005-feature-service/quickstart.md` (обновить)

**Оценка**: 2-3 часа

## Детали реализации

### SQL запросы для чтения raw data

#### Чтение klines для вычисления features

```sql
-- Чтение klines за день для вычисления features
SELECT 
    timestamp,
    symbol,
    open,
    high,
    low,
    close,
    volume,
    turnover
FROM klines_hypertable
WHERE symbol = $1
    AND timestamp >= $2  -- start of day
    AND timestamp < $3   -- end of day
ORDER BY timestamp;
```

#### Чтение trades для вычисления features

```sql
-- Чтение trades за день
SELECT 
    timestamp,
    symbol,
    price,
    size,
    side
FROM trades_hypertable
WHERE symbol = $1
    AND timestamp >= $2
    AND timestamp < $3
ORDER BY timestamp;
```

#### Чтение orderbook snapshots

```sql
-- Чтение orderbook snapshots за день
SELECT 
    timestamp,
    symbol,
    bids,  -- JSONB
    asks   -- JSONB
FROM orderbook_snapshots_hypertable
WHERE symbol = $1
    AND timestamp >= $2
    AND timestamp < $3
ORDER BY timestamp;
```

### SQL запросы для API endpoints (запросы от model-service)

#### GET /features/latest

```sql
-- Получение последних features для символа
SELECT 
    timestamp,
    symbol,
    features,  -- JSONB
    feature_registry_version
FROM features_hypertable
WHERE symbol = $1
    AND dataset_id IS NULL  -- Только вычисленные features, не из датасета
ORDER BY timestamp DESC
LIMIT 1;
```

#### GET /dataset/{id}/download

```sql
-- Получение датасета (features + targets) для скачивания
-- Streaming read по chunks
SELECT 
    f.timestamp,
    f.symbol,
    f.features,  -- JSONB
    t.target,
    t.target_class,
    t.target_raw
FROM features_hypertable f
INNER JOIN targets_hypertable t 
    ON f.timestamp = t.timestamp 
    AND f.symbol = t.symbol 
    AND f.dataset_id = t.dataset_id
WHERE f.dataset_id = $1  -- dataset_id
ORDER BY f.timestamp
LIMIT $2 OFFSET $3;  -- Для pagination/streaming
```

#### POST /api/v1/targets/compute

```sql
-- Получение klines для вычисления targets (используется в target computation)
SELECT 
    timestamp,
    symbol,
    open,
    high,
    low,
    close,
    volume
FROM klines_hypertable
WHERE symbol = $1
    AND timestamp >= $2  -- target_timestamp - lookback
    AND timestamp <= $3   -- target_timestamp + tolerance
ORDER BY timestamp;
```

**Примечание**: Функции `find_available_data_range()` и `load_historical_data_for_target_computation()` в `target_computation_data.py` должны использовать TimescaleDB вместо ParquetStorage.

#### GET /raw-data/availability

```sql
-- Проверка доступности данных по дням для символа
-- Возвращает список дней и типов данных, доступных для каждого дня

-- Для каждого типа данных проверяем наличие записей по дням
WITH date_range AS (
    SELECT generate_series(
        $2::date,  -- start_date
        $3::date,  -- end_date
        '1 day'::interval
    )::date AS check_date
),
klines_availability AS (
    SELECT 
        DATE(timestamp) AS date,
        COUNT(*) AS record_count,
        MIN(timestamp) AS first_timestamp,
        MAX(timestamp) AS last_timestamp
    FROM klines_hypertable
    WHERE symbol = $1
        AND DATE(timestamp) >= $2
        AND DATE(timestamp) <= $3
    GROUP BY DATE(timestamp)
),
trades_availability AS (
    SELECT 
        DATE(timestamp) AS date,
        COUNT(*) AS record_count,
        MIN(timestamp) AS first_timestamp,
        MAX(timestamp) AS last_timestamp
    FROM trades_hypertable
    WHERE symbol = $1
        AND DATE(timestamp) >= $2
        AND DATE(timestamp) <= $3
    GROUP BY DATE(timestamp)
),
orderbook_availability AS (
    SELECT 
        DATE(timestamp) AS date,
        COUNT(*) AS record_count,
        MIN(timestamp) AS first_timestamp,
        MAX(timestamp) AS last_timestamp
    FROM orderbook_snapshots_hypertable
    WHERE symbol = $1
        AND DATE(timestamp) >= $2
        AND DATE(timestamp) <= $3
    GROUP BY DATE(timestamp)
),
ticker_availability AS (
    SELECT 
        DATE(timestamp) AS date,
        COUNT(*) AS record_count,
        MIN(timestamp) AS first_timestamp,
        MAX(timestamp) AS last_timestamp
    FROM ticker_hypertable
    WHERE symbol = $1
        AND DATE(timestamp) >= $2
        AND DATE(timestamp) <= $3
    GROUP BY DATE(timestamp)
),
funding_availability AS (
    SELECT 
        DATE(timestamp) AS date,
        COUNT(*) AS record_count,
        MIN(timestamp) AS first_timestamp,
        MAX(timestamp) AS last_timestamp
    FROM funding_hypertable
    WHERE symbol = $1
        AND DATE(timestamp) >= $2
        AND DATE(timestamp) <= $3
    GROUP BY DATE(timestamp)
)
SELECT 
    dr.check_date AS date,
    COALESCE(k.record_count, 0) AS klines_count,
    COALESCE(t.record_count, 0) AS trades_count,
    COALESCE(o.record_count, 0) AS orderbook_count,
    COALESCE(tk.record_count, 0) AS ticker_count,
    COALESCE(f.record_count, 0) AS funding_count,
    CASE 
        WHEN k.record_count > 0 THEN true 
        ELSE false 
    END AS has_klines,
    CASE 
        WHEN t.record_count > 0 THEN true 
        ELSE false 
    END AS has_trades,
    CASE 
        WHEN o.record_count > 0 THEN true 
        ELSE false 
    END AS has_orderbook,
    CASE 
        WHEN tk.record_count > 0 THEN true 
        ELSE false 
    END AS has_ticker,
    CASE 
        WHEN f.record_count > 0 THEN true 
        ELSE false 
    END AS has_funding,
    k.first_timestamp AS klines_first,
    k.last_timestamp AS klines_last,
    t.first_timestamp AS trades_first,
    t.last_timestamp AS trades_last
FROM date_range dr
LEFT JOIN klines_availability k ON dr.check_date = k.date
LEFT JOIN trades_availability t ON dr.check_date = t.date
LEFT JOIN orderbook_availability o ON dr.check_date = o.date
LEFT JOIN ticker_availability tk ON dr.check_date = tk.date
LEFT JOIN funding_availability f ON dr.check_date = f.date
ORDER BY dr.check_date;
```

**Пример ответа**:
```json
{
  "symbol": "BTCUSDT",
  "start_date": "2025-12-20",
  "end_date": "2025-12-25",
  "availability": [
    {
      "date": "2025-12-20",
      "klines_count": 1440,
      "trades_count": 125000,
      "orderbook_count": 86400,
      "ticker_count": 86400,
      "funding_count": 8,
      "has_klines": true,
      "has_trades": true,
      "has_orderbook": true,
      "has_ticker": true,
      "has_funding": true,
      "klines_first": "2025-12-20T00:00:00Z",
      "klines_last": "2025-12-20T23:59:00Z"
    },
    {
      "date": "2025-12-21",
      "klines_count": 0,
      "trades_count": 0,
      "orderbook_count": 0,
      "ticker_count": 0,
      "funding_count": 0,
      "has_klines": false,
      "has_trades": false,
      "has_orderbook": false,
      "has_ticker": false,
      "has_funding": false
    }
  ],
  "summary": {
    "total_days": 6,
    "days_with_klines": 5,
    "days_with_trades": 5,
    "days_with_orderbook": 5,
    "days_with_ticker": 5,
    "days_with_funding": 5,
    "days_with_all_data": 5,
    "days_with_missing_data": 1
  }
}
```

### SQL запросы для split

#### Time-based split

```sql
-- Train split
SELECT * FROM features_hypertable f
INNER JOIN targets_hypertable t 
    ON f.timestamp = t.timestamp 
    AND f.symbol = t.symbol 
    AND f.dataset_id = t.dataset_id
WHERE f.dataset_id = $1
    AND f.timestamp >= $2  -- train_start
    AND f.timestamp <= $3  -- train_end
ORDER BY f.timestamp;

-- Validation split
SELECT * FROM features_hypertable f
INNER JOIN targets_hypertable t 
    ON f.timestamp = t.timestamp 
    AND f.symbol = t.symbol 
    AND f.dataset_id = t.dataset_id
WHERE f.dataset_id = $1
    AND f.timestamp >= $4  -- val_start
    AND f.timestamp <= $5  -- val_end
ORDER BY f.timestamp;

-- Test split
SELECT * FROM features_hypertable f
INNER JOIN targets_hypertable t 
    ON f.timestamp = t.timestamp 
    AND f.symbol = t.symbol 
    AND f.dataset_id = t.dataset_id
WHERE f.dataset_id = $1
    AND f.timestamp >= $6  -- test_start
    AND f.timestamp <= $7  -- test_end
ORDER BY f.timestamp;
```

#### Walk-forward split

```sql
-- Для каждого окна выполняем запрос
WITH window_bounds AS (
    SELECT 
        $2 + (step * INTERVAL '1 day') AS train_start,
        $2 + (step * INTERVAL '1 day') + INTERVAL '30 days' AS train_end,
        $2 + (step * INTERVAL '1 day') + INTERVAL '30 days' AS val_start,
        $2 + (step * INTERVAL '1 day') + INTERVAL '37 days' AS val_end,
        $2 + (step * INTERVAL '1 day') + INTERVAL '37 days' AS test_start,
        $2 + (step * INTERVAL '1 day') + INTERVAL '44 days' AS test_end
    FROM generate_series(0, $3) AS step
)
SELECT * FROM features_hypertable f
INNER JOIN targets_hypertable t 
    ON f.timestamp = t.timestamp 
    AND f.symbol = t.symbol 
    AND f.dataset_id = t.dataset_id
INNER JOIN window_bounds wb ON 
    f.timestamp >= wb.train_start 
    AND f.timestamp < wb.train_end
WHERE f.dataset_id = $1
ORDER BY f.timestamp;
```

### Streaming write (batch insert)

```python
async def write_features_to_hypertable(
    self,
    dataset_id: str,
    symbol: str,
    features_df: pd.DataFrame
) -> None:
    """Write features to TimescaleDB hypertable in batches."""
    if features_df.empty:
        return
    
    # Подготовка данных: конвертируем features в JSONB
    records = []
    for _, row in features_df.iterrows():
        # Извлекаем features (все колонки кроме timestamp, symbol)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['timestamp', 'symbol']]
        features_dict = {col: float(row[col]) for col in feature_cols}
        
        records.append({
            'timestamp': row['timestamp'],
            'symbol': symbol,
            'dataset_id': dataset_id,
            'features': json.dumps(features_dict)
        })
    
    # Batch insert через COPY (самый быстрый способ)
    async with self._pool.acquire() as conn:
        await conn.copy_records_to_table(
            'features_hypertable',
            records=records,
            columns=['timestamp', 'symbol', 'dataset_id', 'features']
        )
    
    logger.debug(
        f"Written {len(records)} features to hypertable",
        dataset_id=dataset_id,
        symbol=symbol
    )
```

### Streaming read (chunked queries)

```python
async def read_merged_split_streaming(
    self,
    dataset_id: str,
    start_time: datetime,
    end_time: datetime
) -> AsyncIterator[pd.DataFrame]:
    """Read merged features+targets split in chunks (streaming)."""
    chunk_size = 10000
    
    async with self._pool.acquire() as conn:
        async with conn.transaction():
            # Используем cursor для streaming read
            async with conn.cursor(
                """
                SELECT 
                    f.timestamp,
                    f.symbol,
                    f.features,  -- JSONB
                    t.target,
                    t.target_class,
                    t.target_raw
                FROM features_hypertable f
                INNER JOIN targets_hypertable t 
                    ON f.timestamp = t.timestamp 
                    AND f.symbol = t.symbol 
                    AND f.dataset_id = t.dataset_id
                WHERE f.dataset_id = $1
                    AND f.timestamp >= $2
                    AND f.timestamp <= $3
                ORDER BY f.timestamp
                """,
                dataset_id, start_time, end_time
            ) as cursor:
                # Читаем по chunks
                while True:
                    chunk_records = await cursor.fetch(chunk_size)
                    if not chunk_records:
                        break
                    
                    # Конвертируем в DataFrame
                    chunk_df = pd.DataFrame(chunk_records)
                    
                    # Распаковываем JSONB features в колонки
                    if 'features' in chunk_df.columns:
                        features_df = pd.json_normalize(chunk_df['features'])
                        chunk_df = pd.concat([chunk_df.drop('features', axis=1), features_df], axis=1)
                    
                    yield chunk_df
                    
                    # Высвобождаем память
                    del chunk_records, chunk_df
```

## Риски и митигации

### Риск 1: Увеличение нагрузки на PostgreSQL

**Митигация**:
- Использовать connection pooling
- Настроить TimescaleDB chunking для эффективной работы
- Мониторить производительность БД
- Рассмотреть read replicas для read-heavy операций

### Риск 2: Сложность миграции существующего кода

**Митигация**:
- Полная переработка Dataset Builder с удалением старого кода
- Тщательное тестирование перед деплоем (unit + integration + performance тесты)
- Подробное логирование для отладки
- Документация изменений в архитектуре

### Риск 3: Увеличение размера БД

**Митигация**:
- Использовать TimescaleDB compression (автоматическое сжатие старых chunks)
- Retention policy: удалять данные старше 90 дней (raw data и features)
- Мониторинг размера БД и очистка
- Compression для raw data: сжимать chunks старше 7 дней

### Риск 4: Производительность SQL запросов

**Митигация**:
- Правильные индексы на (dataset_id, timestamp)
- Использовать EXPLAIN ANALYZE для оптимизации
- Continuous aggregates для часто используемых запросов

### Риск 5: Обновление всех зависимостей от ParquetStorage

**Найденные компоненты, использующие ParquetStorage**:
1. ✅ `DataStorageService` - запись raw data (уже учтено в Этапе 3)
2. ✅ `BackfillingService` - запись/чтение raw data (уже учтено в Этапе 3)
3. ✅ `StreamingDatasetBuilder` - чтение raw data (уже учтено в Этапе 5)
4. ✅ `OptimizedBuilder` - чтение raw data (уже учтено в Этапе 5)
5. ⚠️ `OptimizedDailyDataCache` - Level 3 fallback (нужно обновить в Этапе 5)
6. ⚠️ `TargetComputationData` - чтение klines для target computation (нужно обновить в Этапе 4)
7. ✅ API endpoints - чтение данных (уже учтено в Этапе 4)

**Митигация**:
- Создать полный список всех файлов, использующих `ParquetStorage` перед началом работы
- Обновить каждый компонент систематически
- Тестировать каждый компонент отдельно после обновления
- Убедиться, что все fallback механизмы (например, Daily cache Level 3) обновлены

## Метрики успеха

### Производительность

- **Память**: Снижение с ~4-8 GB до < 1 GB при билде
- **CPU**: Снижение пиковой нагрузки на 60-80%
- **Время билда**: Сохранить или улучшить текущее время
- **Параллелизм**: Возможность обрабатывать 2-3 датасета одновременно

### Надежность

- **Uptime**: Feature Service остается доступным во время билда
- **Ошибки**: < 1% ошибок при билде датасетов
- **Data integrity**: 100% соответствие данных (валидация перед удалением старого кода)

## Преимущества полной миграции

### Производительность
- **Память**: Снижение с 4-8 GB до < 500 MB (данные в БД, не в RAM)
- **CPU**: Снижение пиковой нагрузки на 70-80% (нет concat/merge в памяти)
- **Параллелизм**: Возможность обрабатывать 3-5 датасетов одновременно
- **Скорость**: SQL запросы оптимизированы TimescaleDB для временных рядов

### Надежность
- **Uptime**: Feature Service остается полностью доступным во время билда
- **Масштабируемость**: Легко обрабатывать большие датасеты (миллионы записей)
- **Восстановление**: Можно пересоздать сплиты из БД без пересчета features

### Гибкость
- **Быстрый пересоздание сплитов**: Изменение train/val/test периодов без пересчета
- **Аналитика**: SQL запросы для анализа данных датасетов
- **Валидация**: Легко проверить распределение данных по периодам

## Риски и митигации

### Риск 1: Увеличение нагрузки на PostgreSQL

**Митигация**:
- Использовать существующий connection pool (не создавать новые соединения)
- Настроить TimescaleDB chunking для эффективной работы (1 day chunks)
- Мониторить производительность БД (slow queries, connection pool usage)
- Рассмотреть read replicas только если нагрузка критична
- Использовать compression для экономии места и ускорения запросов

### Риск 2: Сложность миграции существующего кода

**Митигация**:
- Полная переработка Dataset Builder с удалением старого кода
- Тщательное тестирование перед деплоем (unit + integration + performance тесты)
- Подробное логирование для отладки
- Документация изменений в архитектуре
- Валидация данных: сравнение результатов нового и старого подходов на тестовых датасетах перед удалением старого кода

### Риск 3: Увеличение размера БД

**Митигация**:
- Использовать TimescaleDB compression (автоматическое сжатие старых chunks)
- Автоматическое удаление данных после экспорта в Parquet (опционально)
- Retention policy: удалять данные датасетов старше 90 дней
- Мониторинг размера БД и очистка

### Риск 4: Производительность SQL запросов

**Митигация**:
- Правильные индексы: `(dataset_id, timestamp)` для быстрого split
- Использовать EXPLAIN ANALYZE для оптимизации запросов
- Continuous aggregates для часто используемых запросов (если нужно)
- Chunk exclusion: TimescaleDB автоматически исключает ненужные chunks

### Риск 5: Динамическая схема features

**Митигация**:
- Использовать JSONB для features (гибко, но медленнее)
- Или ALTER TABLE для добавления колонок (быстрее, но сложнее)
- Валидация схемы перед записью
- Версионирование схемы через metadata

## Стратегия перехода

### Жёсткий переход без feature flag

**Принцип**: Полная замена старого подхода на новый, без возможности отката через feature flag.

**Порядок действий**:
1. **Разработка нового кода** параллельно со старым (в отдельных ветках/файлах)
2. **Валидация**: Сравнение результатов нового и старого подходов на тестовых датасетах
3. **Тестирование**: Unit, integration и performance тесты нового подхода
4. **Деплой**: Замена старого кода новым в одной транзакции
5. **Мониторинг**: Отслеживание метрик после деплоя

**Критерии готовности к деплою**:
- ✅ Все тесты проходят
- ✅ Валидация показала 100% соответствие данных
- ✅ Performance тесты показали улучшение (память, CPU)
- ✅ Документация обновлена
- ✅ Мониторинг настроен

**Откат**: В случае критических проблем - откат через git revert (не через feature flag).

## Порядок реализации

### Фаза 1: Инфраструктура (Этап 1)
- Установка TimescaleDB extension
- Создание всех hypertables (raw data + features + targets)
- Настройка compression и retention policies

### Фаза 2: Базовый функционал (Этап 2)
- Репозиторий для TimescaleDB
- Streaming write/read методы для raw data и features
- Тестирование на малых датасетах

### Фаза 3: Обновление real-time processing и backfilling (Этап 3)
- Обновление обработки событий из RabbitMQ для записи в TimescaleDB
- Обновление backfilling для записи в TimescaleDB
- Удаление зависимости от Parquet файлов

### Фаза 4: Обновление API endpoints (Этап 4)
- Обновление endpoints для чтения из TimescaleDB
- Обновление запросов от model-service
- Streaming read для больших датасетов

### Фаза 5: Интеграция (Этап 5)
- Полная переработка Dataset Builder
- Чтение raw data из TimescaleDB вместо Parquet
- Валидация: сравнение результатов со старым подходом
- Тестирование на реальных датасетах
- **После валидации**: удаление старого кода

### Фаза 6: Оптимизация (Этап 7)
- Performance тесты
- Оптимизация SQL запросов
- Настройка индексов

### Фаза 7: Production (Этап 8)
- Документация
- Мониторинг
- Деплой с полным переходом на TimescaleDB

## Следующие шаги

1. ✅ Создать план (этот документ)
2. ⏳ Обсудить подход с командой
3. ⏳ Утвердить архитектуру (JSONB vs динамические колонки)
4. ⏳ Начать реализацию (Этап 1: Инфраструктура)

## Ссылки

- [TimescaleDB Documentation](https://docs.timescale.com/)
- [TimescaleDB Hypertables](https://docs.timescale.com/use-timescale/latest/hypertables/)
- [TimescaleDB Compression](https://docs.timescale.com/use-timescale/latest/compression/)
- [PostgreSQL COPY](https://www.postgresql.org/docs/current/sql-copy.html)

