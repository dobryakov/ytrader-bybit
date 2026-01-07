# Отслеживание поля `opened_at` в Position

## Обзор

Поле `opened_at` отслеживает момент последнего открытия позиции (когда размер позиции изменился с 0 на non-zero). Это позволяет точно определить, когда позиция была открыта, в отличие от `created_at` (момент первого создания записи) и `last_updated` (момент последнего обновления, которое может быть просто изменением цены).

## Краткая сводка

| Кто обновляет | Когда | Источник данных |
|---------------|-------|-----------------|
| `get_position()` | При обнаружении переоткрытия (closed_at != NULL, но size != 0) | Чтение из БД |
| `update_position_from_websocket()` | При создании новой позиции с size != 0 | WebSocket события от Bybit |
| `update_position_from_websocket()` | При изменении размера с 0 → non-zero | WebSocket события от Bybit |
| `validate_position()` | При валидации, если вычисленный размер меняется с 0 → non-zero | Вычисление из истории ордеров |
| `validate_position()` | При создании позиции через валидацию с size != 0 | Вычисление из истории ордеров |

## Кто обновляет `opened_at`

### 1. `PositionManager.get_position()` - при обнаружении переоткрытия

**Когда:** При чтении позиции из БД, если обнаружено, что позиция была закрыта (`closed_at != NULL`), но сейчас имеет ненулевой размер (`size != 0`).

**Код:**
```python
# position-manager/src/services/position_manager.py:173-199
if closed_at is not None and position_size != 0:
    # Позиция переоткрыта после закрытия
    await pool.execute(
        """
        UPDATE positions
        SET closed_at = NULL,
            opened_at = NOW(),  # ← Устанавливается здесь
            total_fees = 0,
            version = version + 1,
            last_updated = NOW()
        WHERE asset = $1 AND mode = $2
        """
    )
```

**Вызывается:** Автоматически при каждом вызове `get_position()`, если обнаружено рассинхронизированное состояние.

---

### 2. `PositionManager.update_position_from_websocket()` - при создании новой позиции

**Когда:** При создании новой позиции через WebSocket событие, если размер позиции не равен нулю.

**Код:**
```python
# position-manager/src/services/position_manager.py:386-395
# Set opened_at if position size is non-zero (position is being opened)
opened_at_value = "NOW()" if new_size != 0 else "NULL"

insert_query = f"""
    INSERT INTO positions (
        asset, mode, size, average_entry_price,
        unrealized_pnl, realized_pnl, total_fees,
        current_price, version, last_updated, opened_at, created_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, 0, $7, 1, NOW(), {opened_at_value}, NOW())
    ...
"""
```

**Вызывается:** 
- `WebSocketPositionConsumer._handle_message()` → `update_position_from_websocket()`
- Источник: события из очереди `ws-gateway.position` (данные от Bybit WebSocket)

---

### 3. `PositionManager.update_position_from_websocket()` - при изменении размера с 0 → non-zero

**Когда:** При обновлении существующей позиции через WebSocket, если размер меняется с 0 на non-zero.

**Код:**
```python
# position-manager/src/services/position_manager.py:681-693
# Track position opening: if size changes from 0 to non-zero, set opened_at
opened_at_update = ""
if position.size == 0 and resolved_size != 0:
    opened_at_update = ", opened_at = NOW()"
    logger.info(
        "position_opening_tracked",
        asset=asset,
        mode=mode,
        old_size=str(position.size),
        new_size=str(resolved_size),
        trace_id=trace_id,
    )

update_query = f"""
    UPDATE positions
    SET ...
    {opened_at_update}  # ← Добавляется в запрос
    ...
"""
```

**Вызывается:**
- `WebSocketPositionConsumer._handle_message()` → `update_position_from_websocket()`
- Источник: события из очереди `ws-gateway.position` (данные от Bybit WebSocket)

---

### 4. `PositionManager.validate_position()` - при валидации и исправлении позиции

**Когда:** При валидации позиции через API или фоновую задачу, если вычисленный размер меняется с 0 на non-zero.

**Код:**
```python
# position-manager/src/services/position_manager.py:1478-1483
# Track position opening: if size changes from 0 to non-zero, set opened_at
existing_size = Decimal(str(existing["size"])) if existing.get("size") else Decimal("0")
opened_at_update = ""
if existing_size == 0 and computed_size != 0:
    opened_at_update = ", opened_at = NOW()"
```

**Вызывается:**
- `POST /api/v1/positions/{asset}/validate` (API endpoint)
- `PositionValidationTask` (фоновая задача валидации)
- Источник: вычисление размера позиции из истории ордеров

---

### 5. `PositionManager.validate_position()` - при создании позиции через валидацию

**Когда:** При создании новой позиции через валидацию, если вычисленный размер не равен нулю.

**Код:**
```python
# position-manager/src/services/position_manager.py:1520-1527
# Set opened_at if position size is non-zero (position is being opened)
opened_at_value = "NOW()" if computed_size != 0 else "NULL"

upsert_query = f"""
    INSERT INTO positions (
        asset, mode, size, average_entry_price, version, last_updated, opened_at, created_at
    )
    VALUES ($1, $2, $3, $4, 1, NOW(), {opened_at_value}, NOW())
    ...
"""
```

**Вызывается:**
- `POST /api/v1/positions/{asset}/validate` (API endpoint)
- `PositionValidationTask` (фоновая задача валидации)

---

## Когда `opened_at` НЕ обновляется

- При изменении размера позиции, если она уже была открыта (size != 0 до и после)
- При обновлении цены, PnL или других полей без изменения размера
- При закрытии позиции (size становится 0) - `opened_at` остается без изменений

## Миграция данных

При применении миграции `051_add_opened_at_to_positions.sql`:
- Для существующих открытых позиций (`closed_at IS NULL AND size != 0`) устанавливается `opened_at = last_updated` как лучшее приближение
- Для закрытых позиций `opened_at` остается `NULL`

## Использование `opened_at`

- **График движения цены:** Используется для определения времени начала графика (5 минут до открытия)
- **Время удержания позиции:** Свойство `time_held_minutes` использует `opened_at` если доступно, иначе fallback на `created_at`
- **Аналитика:** Точное время последнего открытия позиции для отчетов и аналитики

