## Пересмотренная архитектура: единая таблица и модель для активных и исторических позиций

### 1. Концепция

**Одна таблица `positions`** для всех позиций:
- Активная позиция: `closed_at IS NULL` и `size != 0`
- Закрытая позиция: `closed_at IS NOT NULL` и `size = 0`
- При переоткрытии создается новая запись с новым `id`

**Одна модель `Position`** для всех состояний:
- Определение состояния через `closed_at`
- Методы для работы с активными и историческими позициями

### 2. Жизненный цикл позиции

```
[Создание] → [Открыта (size != 0, closed_at = NULL)] → [Обновления] 
                                                              ↓
[Закрыта (size = 0, closed_at = NOW())] ← [Частичное закрытие]
     ↓
[История (запись зафиксирована, не обновляется)]
     ↓
[Переоткрытие] → [Новая запись с новым id] → [Открыта] → ...
```

### 3. Структура таблицы `positions`

```sql
CREATE TABLE positions (
    -- Идентификация
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset VARCHAR(20) NOT NULL,
    mode VARCHAR(20) NOT NULL,
    
    -- Состояние позиции
    size DECIMAL(20, 8) NOT NULL,
    average_entry_price DECIMAL(20, 8),
    current_price DECIMAL(20, 8),
    
    -- PnL
    unrealized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
    realized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
    
    -- Hedge mode
    long_size DECIMAL(20, 8),
    short_size DECIMAL(20, 8),
    long_avg_price DECIMAL(20, 8),
    short_avg_price DECIMAL(20, 8),
    
    -- Финансовые показатели из Bybit
    leverage DECIMAL(10, 2),
    position_value DECIMAL(20, 8),
    liq_price DECIMAL(20, 8),
    bust_price DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    stop_loss DECIMAL(20, 8),
    cum_realised_pnl DECIMAL(20, 8),
    cum_unrealised_pnl DECIMAL(20, 8),
    
    -- Комиссии и маржи
    total_fees DECIMAL(20, 8) NOT NULL DEFAULT 0,
    opening_fees DECIMAL(20, 8),
    closing_fees DECIMAL(20, 8),
    margin_used DECIMAL(20, 8),
    available_margin DECIMAL(20, 8),
    maintenance_margin DECIMAL(20, 8),
    
    -- Объём
    max_size DECIMAL(20, 8),  -- Максимальный размер за историю этой позиции
    min_size DECIMAL(20, 8),  -- Минимальный размер за историю этой позиции
    total_volume_traded DECIMAL(20, 8) NOT NULL DEFAULT 0,
    
    -- Цены
    first_entry_price DECIMAL(20, 8),  -- Цена первого входа
    last_entry_price DECIMAL(20, 8),   -- Цена последнего входа
    exit_price DECIMAL(20, 8),         -- Цена выхода (при закрытии)
    
    -- PnL метрики
    peak_unrealized_pnl DECIMAL(20, 8),
    peak_unrealized_pnl_at TIMESTAMP,
    worst_unrealized_pnl DECIMAL(20, 8),
    worst_unrealized_pnl_at TIMESTAMP,
    
    -- Временные метки
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),  -- Создание этой записи (также представляет время открытия, т.к. каждая запись создается один раз)
    last_updated TIMESTAMP NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMP,                          -- Закрытие (NULL = активна, NOT NULL = закрыта)
    
    -- Метаданные
    version INTEGER NOT NULL DEFAULT 1,           -- Optimistic locking
    source VARCHAR(50),                           -- Источник обновления
    last_sync_with_bybit TIMESTAMP,
    bybit_position_data JSONB,                   -- Полные данные от Bybit
    
    -- Индексы
    CONSTRAINT chk_mode CHECK (mode IN ('one-way', 'hedge')),
    CONSTRAINT chk_size_closed CHECK (
        (closed_at IS NULL) OR (closed_at IS NOT NULL AND size = 0)
    )
);

-- Индексы
CREATE INDEX idx_positions_asset ON positions(asset);
CREATE INDEX idx_positions_mode ON positions(mode);
CREATE INDEX idx_positions_asset_mode ON positions(asset, mode);
CREATE INDEX idx_positions_closed_at ON positions(closed_at DESC);
CREATE INDEX idx_positions_active ON positions(asset, mode) WHERE closed_at IS NULL;  -- Для быстрого поиска активных
CREATE INDEX idx_positions_historical ON positions(asset, mode, closed_at DESC) WHERE closed_at IS NOT NULL;  -- Для истории
CREATE UNIQUE INDEX idx_positions_active_unique ON positions(asset, mode) WHERE closed_at IS NULL;  -- Критично: предотвращение дубликатов активных позиций
```

### 4. Логика работы

#### 4.1. Создание новой позиции (первое открытие)
```python
# Когда size меняется с 0 на non-zero для нового актива
# КРИТИЧНО: Использовать транзакцию и SELECT FOR UPDATE для проверки отсутствия активной позиции
# См. раздел 7 "Защита от race conditions" для деталей

BEGIN TRANSACTION;
  -- 1. Проверяем отсутствие активной позиции (с блокировкой)
  SELECT id FROM positions 
  WHERE asset = X AND mode = Y AND closed_at IS NULL
  FOR UPDATE;
  
  -- 2. Если активной позиции нет, создаем новую
  INSERT INTO positions (
      asset, mode, size, average_entry_price, created_at, ...
  ) VALUES (...)
  -- closed_at = NULL (активная)
  -- created_at = NOW() (также представляет время открытия)
  -- version = 1
COMMIT;
```

#### 4.2. Обновление активной позиции
```python
# Когда позиция уже открыта (closed_at IS NULL)
# КРИТИЧНО: Использовать транзакцию, SELECT FOR UPDATE и optimistic locking
# См. раздел 7 "Защита от race conditions" для деталей

BEGIN TRANSACTION;
  -- 1. Читаем позицию с блокировкой и проверкой версии
  SELECT id, version FROM positions
  WHERE asset = X AND mode = Y AND closed_at IS NULL
  FOR UPDATE;
  
  -- 2. Обновляем с проверкой версии (optimistic locking)
  UPDATE positions
  SET size = ..., average_entry_price = ..., last_updated = NOW(), version = version + 1
  WHERE asset = X AND mode = Y AND closed_at IS NULL AND version = <read_version>
  -- closed_at остается NULL
  -- Если UPDATE вернул 0 строк - версия изменилась, нужен retry
COMMIT;
```

#### 4.3. Закрытие позиции
```python
# Когда size становится 0
# КРИТИЧНО: Использовать транзакцию, SELECT FOR UPDATE и optimistic locking
# КРИТИЧНО: Валидация ДОЛЖНА быть выполнена ПЕРЕД закрытием
# См. раздел 7 "Защита от race conditions" для деталей

BEGIN TRANSACTION;
  -- 1. Читаем позицию с блокировкой и проверкой версии
  SELECT id, size, version FROM positions
  WHERE asset = X AND mode = Y AND closed_at IS NULL
  FOR UPDATE;
  
  -- 2. ВАЛИДАЦИЯ: Вычисляем позицию из истории ордеров и сравниваем
  --    Все расхождения должны быть исправлены ДО закрытия
  --    После закрытия позиция "заморожена" и не валидируется
  
  -- 3. Проверяем, что позиция не закрыта (size != 0)
  -- 4. Обновляем с проверкой версии (optimistic locking)
  UPDATE positions
  SET 
      size = 0,
      closed_at = NOW(),
      exit_price = current_price,  -- Фиксируем цену выхода
      last_updated = NOW(),
      version = version + 1
  WHERE asset = X AND mode = Y AND closed_at IS NULL AND version = <read_version>
  -- Если UPDATE вернул 0 строк - версия изменилась, нужен retry
  -- После этого запись больше НЕ обновляется (только чтение)
  -- Примечание: unrealized_pnl остается в поле unrealized_pnl для исторической записи
COMMIT;
```

#### 4.4. Переоткрытие позиции
```python
# Когда позиция закрыта (closed_at IS NOT NULL), но снова открывается (size != 0)
# Создаем НОВУЮ запись
# КРИТИЧНО: Использовать транзакцию и SELECT FOR UPDATE для проверки отсутствия активной позиции
# См. раздел 7 "Защита от race conditions" для деталей

BEGIN TRANSACTION;
  -- 1. Проверяем отсутствие активной позиции (с блокировкой)
  SELECT id FROM positions 
  WHERE asset = X AND mode = Y AND closed_at IS NULL
  FOR UPDATE;
  
  -- 2. Если активной позиции нет, создаем новую
  INSERT INTO positions (
      id,  -- НОВЫЙ UUID
      asset, mode, size, average_entry_price, 
      created_at = NOW()  -- НОВОЕ время создания (также представляет время открытия)
      created_at = NOW(),  -- НОВОЕ время создания
      closed_at = NULL,  -- Активная
      version = 1,
      ...
  ) VALUES (...)
  -- Старая запись остается в истории с closed_at
COMMIT;
```

### 5. Модель Position (объединенная)

```python
class Position(BaseModel):
    """Единая модель для активных и исторических позиций."""
    
    # Идентификация
    id: UUID
    asset: str
    mode: str
    
    # Состояние
    size: Decimal
    average_entry_price: Optional[Decimal]
    current_price: Optional[Decimal]
    
    # PnL
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    
    # Hedge mode
    long_size: Optional[Decimal] = None
    short_size: Optional[Decimal] = None
    long_avg_price: Optional[Decimal] = None
    short_avg_price: Optional[Decimal] = None
    
    # Финансовые показатели из Bybit
    leverage: Optional[Decimal] = None
    position_value: Optional[Decimal] = None
    liq_price: Optional[Decimal] = None
    bust_price: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    cum_realised_pnl: Optional[Decimal] = None
    cum_unrealised_pnl: Optional[Decimal] = None
    
    # Комиссии и маржи
    total_fees: Decimal = Decimal("0")
    opening_fees: Optional[Decimal] = None
    closing_fees: Optional[Decimal] = None
    margin_used: Optional[Decimal] = None
    available_margin: Optional[Decimal] = None
    maintenance_margin: Optional[Decimal] = None
    
    # Объём
    max_size: Optional[Decimal] = None
    min_size: Optional[Decimal] = None
    total_volume_traded: Decimal = Decimal("0")
    
    # Цены
    first_entry_price: Optional[Decimal] = None
    last_entry_price: Optional[Decimal] = None
    exit_price: Optional[Decimal] = None  # Только для закрытых
    
    # PnL метрики
    peak_unrealized_pnl: Optional[Decimal] = None
    peak_unrealized_pnl_at: Optional[datetime] = None
    worst_unrealized_pnl: Optional[Decimal] = None
    worst_unrealized_pnl_at: Optional[datetime] = None
    
    # Временные метки
    created_at: datetime
    last_updated: datetime
    closed_at: Optional[datetime] = None  # NULL = активна, NOT NULL = закрыта
    
    # Метаданные
    version: int = 1
    source: Optional[str] = None
    last_sync_with_bybit: Optional[datetime] = None
    bybit_position_data: Optional[Dict[str, Any]] = None
    
    # Computed fields
    @computed_field
    @property
    def is_active(self) -> bool:
        """Проверка, является ли позиция активной."""
        return self.closed_at is None
    
    @computed_field
    @property
    def is_closed(self) -> bool:
        """Проверка, является ли позиция закрытой."""
        return self.closed_at is not None
    
    @computed_field
    @property
    def total_pnl(self) -> Decimal:
        """Общий PnL (unrealized + realized)."""
        return self.unrealized_pnl + self.realized_pnl
    
    @computed_field
    @property
    def holding_time_minutes(self) -> Optional[int]:
        """Время удержания позиции в минутах."""
        if self.is_closed and self.created_at:
            delta = self.closed_at - self.created_at
            return int(delta.total_seconds() // 60)
        elif self.created_at:
            delta = datetime.utcnow() - self.created_at
            return int(delta.total_seconds() // 60)
        return None
    
    # Методы для получения связанных ордеров
    async def get_opening_order(self) -> Optional[Order]:
        """Получить ордер, открывший позицию."""
        # Запрос к position_orders WHERE position_id = self.id AND relationship_type = 'opened'
        ...
    
    async def get_increasing_orders(self) -> List[Order]:
        """Получить ордера, увеличившие позицию."""
        # Запрос к position_orders WHERE position_id = self.id AND relationship_type = 'increased'
        ...
    
    async def get_decreasing_orders(self) -> List[Order]:
        """Получить ордера, уменьшившие позицию."""
        # Запрос к position_orders WHERE position_id = self.id AND relationship_type = 'decreased'
        ...
    
    async def get_reversing_orders(self) -> List[Order]:
        """Получить ордера, развернувшие позицию."""
        # Запрос к position_orders WHERE position_id = self.id AND relationship_type = 'reversed'
        ...
    
    async def get_closing_order(self) -> Optional[Order]:
        """Получить ордер, закрывший позицию."""
        # Запрос к position_orders WHERE position_id = self.id AND relationship_type = 'closed'
        ...
    
    async def get_all_orders(self) -> List[Order]:
        """Получить все ордера, связанные с позицией."""
        # Запрос к position_orders WHERE position_id = self.id
        ...
```

### 6. Методы PositionManager

```python
class PositionManager:
    # Получение текущей активной позиции
    async def get_active_position(
        self, asset: str, mode: str = "one-way"
    ) -> Optional[Position]:
        """Получить текущую активную позицию."""
        # SELECT * FROM positions 
        # WHERE asset = $1 AND mode = $2 AND closed_at IS NULL
    
    # Получение всех активных позиций
    async def get_all_active_positions(self) -> List[Position]:
        """Получить все активные позиции."""
        # SELECT * FROM positions WHERE closed_at IS NULL AND size != 0
    
    # Получение истории позиций для актива
    async def get_position_history(
        self, asset: str, mode: str = "one-way", limit: int = 100
    ) -> List[Position]:
        """Получить историю закрытых позиций для актива."""
        # SELECT * FROM positions 
        # WHERE asset = $1 AND mode = $2 AND closed_at IS NOT NULL
        # ORDER BY closed_at DESC LIMIT $3
    
    # Получение конкретной исторической позиции
    async def get_closed_position_by_id(
        self, position_id: UUID
    ) -> Optional[Position]:
        """Получить закрытую позицию по ID."""
        # SELECT * FROM positions WHERE id = $1 AND closed_at IS NOT NULL
    
    # Обновление активной позиции
    async def update_position_from_websocket(...):
        """Обновить активную позицию из WebSocket события."""
        # UPDATE positions WHERE asset = X AND mode = Y AND closed_at IS NULL
    
    # Закрытие позиции
    async def close_position(
        self, asset: str, mode: str = "one-way"
    ) -> Position:
        """Закрыть активную позицию."""
        # UPDATE positions SET closed_at = NOW(), size = 0, ... 
        # WHERE asset = X AND mode = Y AND closed_at IS NULL
    
    # Создание новой позиции при переоткрытии
    async def create_position_on_reopen(
        self, asset: str, mode: str = "one-way", size: Decimal, ...
    ) -> Position:
        """Создать новую позицию при переоткрытии."""
        # INSERT INTO positions (новый id, новый created_at, closed_at = NULL)
```

### 7. Защита от race conditions

**Критически важно:** При закрытии и переоткрытии позиций могут возникать race conditions, когда несколько событий обрабатываются одновременно. Необходимо обеспечить атомарность операций и защиту от конфликтов.

#### 7.1. Optimistic Locking через поле `version`

Все операции обновления позиций должны использовать optimistic locking:

```python
# Пример: Закрытие позиции с проверкой версии
async def close_position(self, asset: str, mode: str = "one-way") -> Position:
    """Закрыть активную позицию с защитой от race conditions."""
    pool = await DatabaseConnection.get_pool()
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1. Читаем текущую позицию с блокировкой строки
            position = await conn.fetchrow(
                """
                SELECT id, size, version, closed_at
                FROM positions
                WHERE asset = $1 AND mode = $2 AND closed_at IS NULL
                FOR UPDATE  -- Блокируем строку для обновления
                """,
                asset, mode
            )
            
            if not position:
                raise PositionNotFoundError(f"No active position for {asset}")
            
            if position["size"] == 0:
                # Позиция уже закрыта (race condition)
                raise PositionAlreadyClosedError(f"Position for {asset} already closed")
            
            current_version = position["version"]
            
            # 2. Обновляем с проверкой версии
            result = await conn.execute(
                """
                UPDATE positions
                SET 
                    size = 0,
                    closed_at = NOW(),
                    exit_price = current_price,
                    last_updated = NOW(),
                    version = version + 1
                WHERE asset = $1 
                  AND mode = $2 
                  AND closed_at IS NULL
                  AND version = $3  -- Проверка версии для optimistic locking
                """,
                asset, mode, current_version
            )
            
            if result == "UPDATE 0":
                # Версия изменилась - другой процесс обновил позицию
                raise VersionConflictError("Position version conflict, retry needed")
            
            # 3. Возвращаем обновленную позицию
            return await self.get_closed_position_by_id(position["id"])
```

#### 7.2. Защита при переоткрытии позиции

При переоткрытии необходимо проверить, что активной позиции нет:

```python
# Пример: Создание новой позиции при переоткрытии с защитой от дубликатов
async def create_position_on_reopen(
    self, asset: str, mode: str = "one-way", size: Decimal, ...
) -> Position:
    """Создать новую позицию при переоткрытии с защитой от race conditions."""
    pool = await DatabaseConnection.get_pool()
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1. Проверяем, нет ли уже активной позиции (с блокировкой)
            existing = await conn.fetchrow(
                """
                SELECT id, closed_at
                FROM positions
                WHERE asset = $1 AND mode = $2 AND closed_at IS NULL
                FOR UPDATE  -- Блокируем для проверки
                """,
                asset, mode
            )
            
            if existing:
                # Активная позиция уже существует - обновляем её вместо создания новой
                logger.warning(
                    "active_position_exists_on_reopen",
                    asset=asset,
                    mode=mode,
                    existing_id=str(existing["id"]),
                    reason="Race condition: active position found during reopen"
                )
                # Обновляем существующую позицию
                return await self.update_position_from_websocket(...)
            
            # 2. Создаем новую позицию
            new_id = uuid4()
            await conn.execute(
                """
                INSERT INTO positions (
                    id, asset, mode, size, average_entry_price,
                    created_at, closed_at, version, ...
                )
                VALUES ($1, $2, $3, $4, $5, NOW(), NOW(), NOW(), NULL, 1, ...)
                """,
                new_id, asset, mode, size, ...
            )
            
            return await self.get_active_position(asset, mode)
```

#### 7.3. Использование транзакций

Все критические операции должны выполняться в транзакциях:

```python
# Пример: Атомарное обновление позиции из WebSocket события
async def update_position_from_websocket(...):
    """Обновить активную позицию с защитой от race conditions."""
    pool = await DatabaseConnection.get_pool()
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1. Блокируем строку для обновления
            position = await conn.fetchrow(
                """
                SELECT id, size, version, closed_at
                FROM positions
                WHERE asset = $1 AND mode = $2 AND closed_at IS NULL
                FOR UPDATE
                """,
                asset, mode
            )
            
            if not position:
                # Позиции нет - создаем новую
                return await self._create_new_position(...)
            
            # 2. Проверяем, не закрыта ли позиция
            if position["closed_at"] is not None:
                # Позиция закрыта - создаем новую
                return await self.create_position_on_reopen(...)
            
            # 3. Обновляем с проверкой версии
            current_version = position["version"]
            result = await conn.execute(
                """
                UPDATE positions
                SET size = $1, average_entry_price = $2, ..., version = version + 1
                WHERE asset = $3 AND mode = $4 AND closed_at IS NULL AND version = $5
                """,
                new_size, new_avg_price, asset, mode, current_version
            )
            
            if result == "UPDATE 0":
                # Конфликт версии - повторяем попытку
                return await self.update_position_from_websocket(...)  # Retry
```

#### 7.4. Retry логика при конфликтах версий

При обнаружении конфликта версий необходимо повторить операцию:

```python
async def update_position_with_retry(
    self, asset: str, mode: str, update_func: Callable, max_retries: int = 3
) -> Position:
    """Обновить позицию с retry логикой при конфликтах версий."""
    for attempt in range(max_retries):
        try:
            return await update_func(asset, mode)
        except VersionConflictError:
            if attempt == max_retries - 1:
                raise
            # Экспоненциальная задержка перед retry
            await asyncio.sleep(0.1 * (2 ** attempt))
            continue
    raise MaxRetriesExceededError("Failed to update position after retries")
```

#### 7.5. Уникальный индекс для активных позиций

Для предотвращения создания дубликатов активных позиций:

```sql
-- Уникальный индекс для активных позиций (только для closed_at IS NULL)
CREATE UNIQUE INDEX idx_positions_active_unique 
ON positions(asset, mode) 
WHERE closed_at IS NULL;

-- Примечание: Этот индекс гарантирует, что для каждого (asset, mode) 
-- может существовать только одна активная позиция (closed_at IS NULL)
-- Закрытые позиции (closed_at IS NOT NULL) не попадают под этот индекс,
-- поэтому может быть множество исторических записей
```

#### 7.6. Обработка одновременного закрытия и переоткрытия

Сценарий race condition:
1. Процесс A: закрывает позицию (size → 0, closed_at → NOW())
2. Процесс B: одновременно получает событие об открытии (size != 0)

**Решение:**
- Использовать `SELECT FOR UPDATE` для блокировки строки
- Проверять `closed_at IS NULL` перед созданием новой позиции
- Использовать транзакции для атомарности
- При обнаружении закрытой позиции создавать новую запись

```python
# Пример: Обработка одновременного закрытия и переоткрытия
async def handle_position_update_with_race_protection(...):
    """Обработать обновление позиции с защитой от race conditions."""
    pool = await DatabaseConnection.get_pool()
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Блокируем строку для чтения/обновления
            position = await conn.fetchrow(
                """
                SELECT * FROM positions
                WHERE asset = $1 AND mode = $2
                FOR UPDATE  -- Критично: блокируем строку
                """,
                asset, mode
            )
            
            if not position:
                # Позиции нет - создаем новую
                return await self._create_new_position(...)
            
            if position["closed_at"] is not None:
                # Позиция закрыта - создаем новую при переоткрытии
                if new_size != 0:
                    return await self.create_position_on_reopen(...)
                else:
                    # Позиция закрыта и size = 0 - ничего не делаем
                    return None
            
            # Позиция активна - обновляем
            if new_size == 0:
                # Закрываем позицию
                return await self.close_position(asset, mode)
            else:
                # Обновляем активную позицию
                return await self.update_active_position(...)
```

### 8. Важные детали реализации

#### 8.1. Связь с ордерами через `position_orders`

При переоткрытии позиции создается новая запись с новым `id`. Связи с ордерами работают следующим образом:

- **Старые ордера** остаются привязанными к старой закрытой позиции через `position_orders.position_id = <старый_id>`
- **Новые ордера** привязываются к новой активной позиции через `position_orders.position_id = <новый_id>`
- Методы `get_opening_order()`, `get_closing_order()` и т.д. работают в контексте конкретной позиции (по её `id`)
- Для получения всех ордеров по активу (включая исторические) нужен запрос через `position_orders` с JOIN к `positions` по `asset` и `mode`

#### 8.2. Снапшоты позиций (`position_snapshots`)

- **Создание снапшотов**: Снапшоты создаются только для активных позиций (`closed_at IS NULL`)
- **При закрытии позиции**: Создается финальный снапшот с полным состоянием позиции на момент закрытия
- **При переоткрытии**: Снапшоты старой позиции остаются привязанными к старой записи (`position_id = <старый_id>`)
- **Новая позиция**: Начинает создавать свои снапшоты с момента открытия
- **Историческая реконструкция**: Model Service может использовать снапшоты для восстановления состояния позиций на любой момент времени

#### 8.3. Частичное закрытие vs полное закрытие

**Частичное закрытие:**
- `size` уменьшается, но остается `!= 0`
- `closed_at` остается `NULL` (позиция остается активной)
- Позиция продолжает обновляться из WebSocket событий
- Снапшоты продолжают создаваться

**Полное закрытие:**
- `size` становится `0`
- `closed_at` устанавливается в `NOW()`
- Позиция больше не обновляется (только чтение)
- Создается финальный снапшот
- Публикуется событие закрытия позиции

**Логика определения:**
```python
if new_size == 0 and current_size != 0:
    # Полное закрытие
    closed_at = NOW()
    create_final_snapshot()
    publish_position_closed_event()
elif new_size != 0 and current_size == 0:
    # Переоткрытие (создание новой записи)
    create_new_position()
    publish_position_reopened_event()
else:
    # Частичное изменение или обновление активной позиции
    update_position()
```

#### 8.4. Обработка ошибок при переоткрытии

При создании новой позиции при переоткрытии может возникнуть `UniqueViolationError` из-за уникального индекса `idx_positions_active_unique`:

```python
try:
    # Создаем новую позицию
    await create_position_on_reopen(...)
except UniqueViolationError:
    # Активная позиция уже существует (race condition)
    # Обновляем существующую позицию вместо создания новой
    logger.warning("active_position_exists_on_reopen", asset=asset, reason="Race condition")
    return await update_position_from_websocket(...)
```

#### 8.5. Граничный случай: size = 0, но closed_at = NULL

Если обнаружена позиция с `size = 0` и `closed_at = NULL` (некорректное состояние):

```python
# Автоматическое исправление при чтении позиции
if position.size == 0 and position.closed_at is None:
    # Устанавливаем closed_at для корректности
    await conn.execute(
        "UPDATE positions SET closed_at = NOW() WHERE id = $1",
        position.id
    )
```

#### 8.6. Связь с сигналами и стратегиями

Для отслеживания, какие сигналы/стратегии открыли/закрыли позицию:

- **Цепочка связей**: `positions` ← `position_orders` ← `orders` ← `signal_order_relationships` ← `trading_signals`
- Метод `get_all_orders()` возвращает все ордера позиции
- Через ордера можно получить связанные сигналы
- Для агрегированной статистики по стратегиям нужны JOIN запросы через эту цепочку

---

## Решения по архитектурным вопросам

### События RabbitMQ при закрытии и переоткрытии

**Решение:** Оставляем общее событие `position_updated` для всех изменений позиций (закрытие, переоткрытие, обновление). Подписчиков не меняем.

**Обоснование:** Упрощает обработку событий, подписчики могут определить тип изменения через поля `closed_at` и `size` в payload события.

---

### Агрегированная аналитика по активам

**Решение:** Агрегированную аналитику пока не делаем. При необходимости можно добавить позже через SQL запросы в dashboard-api.

---

### Производительность и архивация исторических позиций

**Решение:** Архивация исторических позиций пока не требуется. Достаточно индексов для быстрого поиска. При необходимости (когда таблица вырастет до миллионов записей) можно добавить архивацию позже.

---

### Валидация исторических позиций

**Решение:** **КРИТИЧНО:** Валидация позиции **ДОЛЖНА** выполняться при закрытии позиции. После установки `closed_at` позиция считается "замороженной" и **не должна** валидироваться или изменяться. Все вычисления и проверки должны быть выполнены до закрытия.

**Обоснование:** Закрытая позиция — это историческая запись, которая не должна изменяться. Любые расхождения должны быть обнаружены и исправлены до закрытия.

---

### Синхронизация с Bybit при переоткрытии

**Решение:** По умолчанию полагаемся на то, что данные по WebSocket приходят корректно и своевременно. Синхронизацию с Bybit API через `position_bybit_sync_task` выполнять **ТОЛЬКО** в случае:
- Обнаружения проблем (расхождения между WebSocket и вычисленными позициями)
- Ручного запроса через API endpoint
- Периодической проверки (опционально, с большим интервалом)

**Обоснование:** WebSocket события должны быть основным источником данных. Синхронизация через API — это fallback механизм для исправления проблем.

---

### Мониторинг и алерты

**Решение:** Мониторинг и алерты выходят за рамки текущей задачи. Можно добавить позже при необходимости.

---

### REST API для агрегированной статистики

**Решение:** Агрегированную статистику пока не делаем. При необходимости можно добавить позже через SQL запросы в dashboard-api.

---

### 9. Преимущества

1. Проще структура: одна таблица и одна модель
2. Проще запросы: не нужны JOIN между таблицами
3. История сохраняется автоматически: при закрытии фиксируется состояние
4. Каждая сессия позиции — отдельная запись: удобно анализировать
5. Легко получить текущую позицию: `WHERE closed_at IS NULL`
6. Легко получить историю: `WHERE closed_at IS NOT NULL ORDER BY closed_at DESC`
7. Защита от race conditions: optimistic locking, транзакции, уникальный индекс

### 10. Миграция данных

Если есть таблица `closed_positions`, нужно:
1. Перенести данные из `closed_positions` в `positions`
2. Установить `closed_at` для перенесенных записей
3. Удалить таблицу `closed_positions`

### 11. Изменения в коде

1. Убрать модель `ClosedPosition`
2. Расширить модель `Position` всеми полями
3. Обновить `PositionManager` для работы с единой таблицей
4. Обновить методы получения позиций (активные vs исторические)
5. Обновить логику закрытия (установка `closed_at`)
6. Обновить логику переоткрытия (создание новой записи)

---

## 12. Доработки всех затронутых сервисов

### 12.1. ws-gateway (Миграции БД)

**Задачи:**

1. **Создать миграцию для расширения таблицы `positions`**:
   - Добавить все новые поля из раздела 3
   - Добавить индексы для активных и исторических позиций
   - Добавить constraint `chk_size_closed`
   - **Критично:** Добавить уникальный индекс `idx_positions_active_unique` на `(asset, mode) WHERE closed_at IS NULL` для предотвращения дубликатов активных позиций

2. **Создать миграцию для переноса данных из `closed_positions`**:
   - Перенести все записи из `closed_positions` в `positions`
   - Установить `closed_at` для перенесенных записей

3. **Удалить таблицу `closed_positions`**:
   - После проверки корректности переноса данных

**Файлы для изменения:**
- `ws-gateway/migrations/XXX_extend_positions_table.sql` (новая миграция)
- `ws-gateway/migrations/XXX_migrate_closed_positions.sql` (новая миграция)
- `ws-gateway/migrations/XXX_drop_closed_positions_table.sql` (выполнить после проверки корректности миграции)

---

### 12.2. position-manager (Основной сервис)

**Задачи:**

1. **Обновить модель `Position`** (`position-manager/src/models/position.py`):
   - Удалить класс `ClosedPosition`
   - Расширить класс `Position` всеми полями из раздела 5
   - Добавить computed fields: `is_active`, `is_closed`, `total_pnl`, `holding_time_minutes`
   - Добавить методы для получения связанных ордеров (если нужно)

2. **Обновить `PositionManager`** (`position-manager/src/services/position_manager.py`):
   - Изменить `get_position()` → `get_active_position()` (добавить `WHERE closed_at IS NULL`) - внутренний метод
   - Добавить `get_all_active_positions()` (только активные) - внутренний метод
   - Добавить `get_position_history()` (история закрытых) - внутренний метод
   - Добавить `get_closed_position_by_id()` (закрытая по ID) - внутренний метод
   - Обновить `update_position_from_websocket()` (только для активных: `WHERE closed_at IS NULL`)
     - **Критично:** Использовать транзакции и `SELECT FOR UPDATE` для защиты от race conditions
     - Использовать optimistic locking через поле `version`
     - Реализовать retry логику при конфликтах версий
   - Обновить `close_position()` (установить `closed_at = NOW()`, `exit_price = current_price`)
     - **Критично:** Использовать транзакции и `SELECT FOR UPDATE`
     - Проверять версию перед обновлением
     - Обрабатывать случаи, когда позиция уже закрыта
     - **КРИТИЧНО:** Валидация позиции ДОЛЖНА быть выполнена ПЕРЕД закрытием
     - После установки `closed_at` позиция "заморожена" и не должна изменяться или валидироваться
   - Добавить `create_position_on_reopen()` (создание новой записи при переоткрытии)
     - **Критично:** Проверять отсутствие активной позиции с `SELECT FOR UPDATE`
     - Использовать уникальный индекс для предотвращения дубликатов
     - Обрабатывать race condition: если активная позиция найдена, обновлять её вместо создания новой
   - Обновить `get_all_positions()` → разделить на активные и исторические (или переименовать в `get_all_active_positions()`)
   - Обновить `validate_position()` (работа только с активными)
   - Обновить `create_position_snapshot()` (только для активных)
   - Удалить методы работы с `closed_positions` таблицей
   - **Примечание:** REST API endpoints остаются прежними (`GET /api/v1/positions/{asset}`), но теперь возвращают только активные позиции через внутренний метод `get_active_position()`

3. **Обновить REST API** (`position-manager/src/api/routes/positions.py`):
   - `GET /api/v1/positions` → возвращать только активные (по умолчанию) или все (с параметром `include_closed=true`)
   - `GET /api/v1/positions/{asset}` → возвращать только активную позицию
   - Добавить `GET /api/v1/positions/{asset}/history` → история закрытых позиций для актива
   - Добавить `GET /api/v1/positions/closed` → все закрытые позиции
   - Добавить `GET /api/v1/positions/{position_id}` → получить позицию по ID (активную или закрытую)
   - Добавить `GET /api/v1/positions/{asset}/orders` → все ордера для актива (активные + исторические через position_orders)
   - Обновить `serialize_position_with_features()` для работы с единой моделью
   - Удалить `serialize_closed_position()` (не нужна)
   - **Примечание:** Агрегированная статистика (total PnL, win rate и т.д.) пока не реализуется

4. **Обновить consumers**:
   - `websocket_position_consumer.py`: 
     * При обнаружении переоткрытия (size != 0, но позиция закрыта) создавать новую запись
     * Обрабатывать `UniqueViolationError` при создании новой позиции (fallback на обновление существующей)
     * При полном закрытии (size = 0) создавать финальный снапшот
   - `position_order_linker_consumer.py`: 
     * Обновить для работы с новой логикой
     * При переоткрытии новые ордера связываются с новой позицией (новый `position_id`)
     * Старые ордера остаются привязанными к старой закрытой позиции

5. **Обновить tasks**:
   - `position_validation_task.py`: 
     * Валидировать только активные позиции (`closed_at IS NULL`)
     * **КРИТИЧНО:** При закрытии позиции валидация ДОЛЖНА быть выполнена ПЕРЕД установкой `closed_at`
     * После закрытия позиция не валидируется (историческая запись)
   - `position_bybit_sync_task.py`: 
     * Синхронизировать только активные позиции
     * **По умолчанию:** Полагаться на WebSocket события (данные корректны и своевременны)
     * **Синхронизация через API:** ТОЛЬКО в случае проблем (расхождения) или ручного запроса
     * Не выполнять автоматическую синхронизацию при переоткрытии (полагаться на WebSocket)
   - При обнаружении позиций с `size = 0` и `closed_at = NULL` автоматически устанавливать `closed_at`

**Файлы для изменения:**
- `position-manager/src/models/position.py`
- `position-manager/src/services/position_manager.py`
- `position-manager/src/api/routes/positions.py`
- `position-manager/src/consumers/websocket_position_consumer.py`
- `position-manager/src/consumers/position_order_linker_consumer.py`
- `position-manager/src/tasks/position_validation_task.py`
- `position-manager/src/tasks/position_bybit_sync_task.py`

---

### 12.3. order-manager

**Важно:** order-manager **НЕ должен** работать с позициями напрямую через БД. Все операции с позициями должны выполняться **только через HTTP API** position-manager через `PositionManagerClient`.

**Задачи:**

1. **Обновить `PositionManagerClient`** (`order-manager/src/services/position_manager_client.py`):
   - Убедиться, что `get_position()` вызывает `GET /api/v1/positions/{asset}` (возвращает только активную позицию)
   - Убедиться, что `get_all_positions()` вызывает `GET /api/v1/positions` (возвращает только активные по умолчанию)
   - Метод `get_portfolio_exposure()` уже использует API (`GET /api/v1/portfolio/exposure`)
   - **Удалить** любые прямые запросы к БД (если есть)
   - **Удалить** метод `get_position_from_bybit()` или переделать его на вызов API position-manager для синхронизации
   - Все методы должны работать **только через HTTP API**, никаких прямых SQL-запросов

2. **Обновить `RiskManager`** (`order-manager/src/services/risk_manager.py`):
   - Все методы должны использовать `PositionManagerClient` для получения позиций
   - `check_position_size()`: использовать `position_manager_client.get_position()` (API возвращает только активные)
   - `check_max_exposure_from_position_manager()`: уже использует API через `get_portfolio_exposure()`
   - **Удалить** любые прямые SQL-запросы к таблице `positions`
   - **Удалить** любые прямые обращения к БД для получения позиций

3. **Обновить `SignalProcessor`** (`order-manager/src/services/signal_processor.py`):
   - Все методы работы с позициями должны использовать `position_manager_client.get_position()`
   - `_check_and_refresh_position_if_stale()`: использовать API position-manager
   - `_find_order_that_opened_position()`: использовать API для получения ордеров позиции (если есть endpoint)
   - `_close_position_before_new_order()`: использовать API position-manager для закрытия (если есть endpoint) или оставить логику в position-manager
   - **Удалить** любые прямые SQL-запросы к таблице `positions`
   - Метод `get_position_from_bybit()` должен вызывать API position-manager для синхронизации, а не работать напрямую

4. **Обновить `OrderExecutor`** (`order-manager/src/services/order_executor.py`):
   - Все проверки позиций должны использовать `position_manager_client.get_position()`
   - **Удалить** любые прямые SQL-запросы к таблице `positions`
   - **Удалить** любые прямые обращения к БД для получения позиций

5. **Обновить `TargetHorizonCloseTask`** (`order-manager/src/services/target_horizon_close_task.py`):
   - Использовать `position_manager_client.get_position()` для получения позиций
   - **Удалить** любые прямые SQL-запросы к таблице `positions`

6. **Удалить модель `Position` из order-manager** (`order-manager/src/models/position.py`):
   - Модель может остаться только для десериализации ответов от API position-manager
   - **Удалить** любые методы модели, которые работают напрямую с БД
   - Модель должна быть только DTO (Data Transfer Object) для API responses

7. **Проверить отсутствие прямых запросов к БД**:
   - Найти все места, где используется `SELECT * FROM positions` или `INSERT/UPDATE positions`
   - Заменить на вызовы через `PositionManagerClient`
   - Проверить импорты: не должно быть прямых импортов `DatabaseConnection` для работы с позициями

**Файлы для изменения:**
- `order-manager/src/services/position_manager_client.py` - убедиться, что все методы используют только HTTP API
- `order-manager/src/services/risk_manager.py` - заменить прямые запросы на API вызовы
- `order-manager/src/services/signal_processor.py` - заменить прямые запросы на API вызовы
- `order-manager/src/services/order_executor.py` - заменить прямые запросы на API вызовы
- `order-manager/src/services/target_horizon_close_task.py` - заменить прямые запросы на API вызовы
- `order-manager/src/models/position.py` - оставить только DTO для API responses, удалить методы работы с БД

**Проверка:**
- Выполнить поиск по кодовой базе: `grep -r "SELECT.*FROM positions" order-manager/`
- Выполнить поиск: `grep -r "INSERT.*INTO positions" order-manager/`
- Выполнить поиск: `grep -r "UPDATE positions" order-manager/`
- Все найденные места должны быть заменены на вызовы через `PositionManagerClient`

---

### 12.4. model-service

**Важно:** model-service **НЕ должен** работать с позициями напрямую через БД. Все операции с позициями должны выполняться **только через HTTP API** position-manager через `PositionManagerClient`.

**Задачи:**

1. **Обновить `PositionStateRepository`** (`model-service/src/database/repositories/position_state_repo.py`):
   - **Удалить** метод `_get_open_positions()` (прямой SQL-запрос к таблице `positions`)
   - **Удалить** метод `get_position_for_asset()` (прямой SQL-запрос)
   - Заменить на использование `PositionManagerClient` для получения позиций через HTTP API
   - Метод `get_order_position_state()` должен использовать `PositionManagerClient.get_position()` или `get_all_positions()` вместо прямых SQL-запросов
   - **Удалить** все SQL-запросы вида `SELECT * FROM positions`
   - Метод `_get_open_orders()` может остаться (работает с таблицей `orders`, не `positions`)

2. **Обновить `PositionManagerClient`** (`model-service/src/services/position_manager_client.py`):
   - Убедиться, что `get_position()` вызывает `GET /api/v1/positions/{asset}` (возвращает только активную позицию)
   - Добавить метод `get_all_positions()` для получения всех активных позиций через `GET /api/v1/positions`
   - Обновить кэширование для учета `closed_at` (кэшировать только активные позиции)
   - Все методы должны работать **только через HTTP API**, никаких прямых SQL-запросов

3. **Обновить `PositionStateTracker`** (`model-service/src/services/position_state_tracker.py`):
   - **Удалить** метод `_load_from_database()` (прямой SQL-запрос к таблице `positions`)
   - Заменить на использование `PositionManagerClient` для получения позиций через HTTP API
   - Метод `get_or_create_state()` должен использовать `PositionManagerClient.get_position()` вместо прямых запросов к БД
   - **Удалить** любые прямые SQL-запросы к таблице `positions`
   - Методы `_save_to_database()` и `_remove_from_database()` могут остаться (работают с локальной таблицей `position_states`, не `positions`)

4. **Обновить `IntelligentSignalGenerator`** (`model-service/src/services/intelligent_signal_generator.py`):
   - Использовать `PositionManagerClient` для получения позиций при генерации сигналов
   - **Удалить** любые прямые SQL-запросы к таблице `positions`
   - **Удалить** использование `PositionStateRepository` для получения позиций (использовать только для ордеров)

5. **Обновить `BalanceCalculator`** (`model-service/src/services/balance_calculator.py`):
   - Метод `get_available_position_for_sell()` уже использует `position_manager_client.get_position()` - проверить корректность
   - Убедиться, что нет прямых SQL-запросов к таблице `positions`

6. **Проверить другие сервисы**:
   - Найти все места, где используется `PositionStateRepository.get_position_for_asset()` или `_get_open_positions()`
   - Заменить на использование `PositionManagerClient`
   - Проверить `model-service/src/api/prediction_chain.py` - убедиться, что позиции получаются через API

**Файлы для изменения:**
- `model-service/src/database/repositories/position_state_repo.py` - удалить прямые SQL-запросы к positions, использовать PositionManagerClient
- `model-service/src/services/position_manager_client.py` - добавить метод `get_all_positions()`, обновить кэширование
- `model-service/src/services/position_state_tracker.py` - заменить прямые запросы на API вызовы
- `model-service/src/services/intelligent_signal_generator.py` - использовать PositionManagerClient вместо PositionStateRepository для позиций
- `model-service/src/services/balance_calculator.py` - проверить использование API (уже используется)
- `model-service/src/api/prediction_chain.py` - проверить получение позиций через API

**Проверка:**
- Выполнить поиск по кодовой базе: `grep -r "SELECT.*FROM positions" model-service/`
- Выполнить поиск: `grep -r "INSERT.*INTO positions" model-service/`
- Выполнить поиск: `grep -r "UPDATE positions" model-service/`
- Выполнить поиск: `grep -r "_get_open_positions\|get_position_for_asset" model-service/`
- Все найденные места должны быть заменены на вызовы через `PositionManagerClient`

---

### 12.5. dashboard-api

**Примечание:** dashboard-api может работать с БД напрямую для чтения данных (это допустимо для read-only операций), но должен учитывать новую структуру с `closed_at`.

**Задачи:**

1. **Обновить `GET /api/positions`** (`dashboard-api/src/api/routes/positions.py`):
   - По умолчанию возвращать только активные позиции (`WHERE closed_at IS NULL`)
   - Добавить параметр `include_closed=true` для получения всех позиций
   - Добавить параметр `closed_only=true` для получения только закрытых

2. **Обновить `GET /api/positions/{asset}`**:
   - Возвращать только активную позицию (`WHERE closed_at IS NULL`)
   - Добавить `GET /api/positions/{asset}/history` для истории закрытых позиций (`WHERE closed_at IS NOT NULL`)

3. **Обновить `GET /api/positions/closed`**:
   - Изменить запрос на `WHERE closed_at IS NOT NULL` (работать с единой таблицей `positions`)
   - **Удалить** JOIN с таблицей `closed_positions` (если есть)

4. **Обновить `GET /api/positions/{position_id}`**:
   - Работать с единой таблицей `positions`
   - Поддерживать как активные (`closed_at IS NULL`), так и закрытые (`closed_at IS NOT NULL`) позиции

5. **Обновить `GET /api/positions/{asset}/orders`**:
   - Работать с активной позицией (`WHERE closed_at IS NULL`)
   - Использовать `position_id` из активной позиции для связи с ордерами

**Файлы для изменения:**
- `dashboard-api/src/api/routes/positions.py`

---

### 12.6. dashboard-frontend

**Задачи:**

1. **Обновить компоненты для работы с новой архитектурой**:
   - `PositionDetail.tsx`: поддержка отображения как активных, так и закрытых позиций
   - `Positions.tsx`: фильтрация активных/закрытых позиций
   - `usePositions.ts`: обновить хуки для работы с новой структурой

2. **Обновить UI для истории позиций**:
   - Cуществующие два списка "Текущие и исторические торговые позиции" и "История закрытых торговых позиций" привести к одному (так как теперь это один класс объектов)

**Файлы для изменения:**
- `dashboard-frontend/src/pages/PositionDetail.tsx`
- `dashboard-frontend/src/pages/Positions.tsx`
- `dashboard-frontend/src/hooks/usePositions.ts`

---

### 12.7. Тесты

**Задачи:**

1. **Обновить все unit-тесты**:
   - Убрать тесты для `ClosedPosition`
   - Обновить тесты для `Position` с учетом новых полей
   - Обновить моки для работы с `closed_at`

2. **Обновить integration-тесты**:
   - Тесты для получения активных позиций
   - Тесты для получения истории
   - Тесты для закрытия и переоткрытия позиций
   - **Критично:** Тесты для race conditions:
     * Одновременное закрытие и переоткрытие позиции
     * Конфликты версий при optimistic locking
     * Создание дубликатов активных позиций (должно быть предотвращено уникальным индексом)
     * Retry логика при конфликтах версий

3. **Обновить E2E-тесты**:
   - Полный цикл: открытие → обновление → закрытие → переоткрытие
   - **Критично:** E2E тесты для race conditions:
     * Параллельная обработка событий закрытия и открытия
     * Проверка атомарности транзакций
     * Проверка отсутствия дубликатов активных позиций

**Файлы для изменения:**
- Все тесты в `position-manager/tests/`
- Все тесты в `order-manager/tests/` (связанные с позициями)
- Все тесты в `model-service/tests/` (связанные с позициями)

---

### 12.8. Порядок выполнения миграции

1. **Этап 1: Подготовка БД** (ws-gateway):
   - Создать миграцию для расширения таблицы `positions`
   - **Критично:** Добавить уникальный индекс `idx_positions_active_unique` для защиты от дубликатов активных позиций
   - Применить миграцию

2. **Этап 2: Обновление position-manager**:
   - Обновить модель `Position`
   - Обновить `PositionManager`
   - Обновить REST API
   - Обновить consumers и tasks
   - Протестировать

3. **Этап 3: Миграция данных** (ws-gateway):
   - Создать миграцию для переноса данных из `closed_positions`
   - Применить миграцию
   - Проверить корректность переноса

4. **Этап 4: Обновление зависимых сервисов**:
   - order-manager
   - model-service
   - dashboard-api
   - Протестировать каждый сервис

5. **Этап 5: Обновление фронтенда** (опционально):
   - dashboard-frontend

6. **Этап 6: Очистка** (опционально):
   - Удалить таблицу `closed_positions` (после проверки)
   - Удалить неиспользуемый код

---

### 12.9. Обратная совместимость

**Важные моменты:**

1. **API endpoints**: Сохранить существующие endpoints, но изменить поведение:
   - `GET /api/v1/positions` по умолчанию возвращает только активные (как раньше)
   - Добавить параметры для получения закрытых

2. **Миграция данных**: Сохранить все данные из `closed_positions` в `positions`

3. **Постепенный переход**: После миграции данных можно временно оставить таблицу `closed_positions` (пометить как deprecated) для возможности отката, но удалить после полной проверки (см. раздел 10.1, пункт 3)

4. **Логирование**: Добавить логирование всех изменений для отладки

---

### 12.10. Проверочный список

- [x] Миграция БД создана и применена (миграции 052 и 053 применены успешно)
- [x] Уникальный индекс `idx_positions_active_unique` создан
- [x] Модель `Position` обновлена
- [x] Модель `ClosedPosition` удалена
- [x] `PositionManager` обновлен
- [x] Защита от race conditions реализована (транзакции, `SELECT FOR UPDATE`, optimistic locking)
- [x] Retry логика при конфликтах версий реализована
- [x] REST API обновлен
- [x] Consumers обновлены (с защитой от race conditions)
- [x] Tasks обновлены
- [x] order-manager обновлен
- [x] model-service обновлен (PositionStateRepository теперь использует Position Manager API вместо прямых SQL-запросов к таблице positions)
- [x] dashboard-api обновлен
- [x] Тесты обновлены (удален test_opened_at.py, обновлен test_position_manager.py для использования created_at)
- [x] Тесты для race conditions написаны (test_race_conditions.py создан с тестами для optimistic locking, retry логики, SELECT FOR UPDATE)
- [x] Данные из `closed_positions` перенесены (миграция 053 применена, 232 записи обработаны)
- [x] E2E тесты написаны (test_unified_positions_architecture.py создан с тестами для создания, закрытия, переоткрытия, race conditions, истории позиций)
- [x] Документация обновлена

