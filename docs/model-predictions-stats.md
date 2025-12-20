## Структура таблицы `prediction_targets`

### Основные атрибуты:

| Атрибут | Тип | Описание |
|---------|-----|----------|
| **id** | UUID | Первичный ключ |
| **signal_id** | UUID | Ссылка на `trading_signals.signal_id` (FK) |
| **prediction_timestamp** | TIMESTAMP | Время создания предсказания |
| **target_timestamp** | TIMESTAMP | Время, на которое сделано предсказание (prediction_timestamp + horizon) |
| **model_version** | VARCHAR(50) | Версия модели, которая сделала предсказание |
| **feature_registry_version** | VARCHAR(50) | Версия реестра фич, использованных для предсказания |
| **target_registry_version** | VARCHAR(50) | Версия реестра таргетов, использованная для предсказания |
| **target_config** | JSONB | Полная конфигурация таргета (type, horizon, computation preset, options) |
| **predicted_values** | JSONB | Предсказанные значения (например, `{"direction": "green", "confidence": 0.99}`) |
| **actual_values** | JSONB | Фактические значения (заполняется после `target_timestamp`) |
| **actual_values_computed_at** | TIMESTAMP | Время вычисления фактических значений |
| **actual_values_computation_error** | TEXT | Ошибка при вычислении фактических значений (если была) |
| **created_at** | TIMESTAMP | Время создания записи |
| **updated_at** | TIMESTAMP | Время последнего обновления |

### Ограничения (Constraints):

1. `chk_target_config_structure` — проверяет структуру `target_config`:
   - Должен содержать `type` и `horizon`
   - `type` должен быть: `'regression'`, `'classification'` или `'risk_adjusted'`
   - `horizon` должен быть > 0

2. `chk_predicted_values_not_empty` — `predicted_values` не должен быть пустым объектом

3. `chk_target_timestamp_after_prediction` — `target_timestamp >= prediction_timestamp`

### Индексы:

- По `signal_id` (для связи с сигналами)
- По `prediction_timestamp` и `target_timestamp` (для временных запросов)
- По версиям (`model_version`, `feature_registry_version`, `target_registry_version`)
- По полям внутри `target_config` (type, horizon, preset)
- Частичный индекс для незавершенных вычислений (`actual_values_computed_at IS NULL`)

### Пример данных:

- `target_config`: содержит полную конфигурацию таргета (тип, горизонт, preset, опции)
- `predicted_values`: `{"direction": "green", "confidence": 0.99}` для классификации
- `actual_values`: заполняется позже, когда наступает `target_timestamp`

## Таблица результатов торговли: `prediction_trading_results`

### Структура таблицы:

| Атрибут | Тип | Описание |
|---------|-----|----------|
| **id** | UUID | Первичный ключ |
| **prediction_target_id** | UUID | Ссылка на `prediction_targets.id` (FK, ON DELETE CASCADE) |
| **signal_id** | UUID | Ссылка на `trading_signals.signal_id` (FK, ON DELETE CASCADE) |
| **realized_pnl** | DECIMAL(20,8) | Реализованная прибыль/убыток (по умолчанию 0) |
| **unrealized_pnl** | DECIMAL(20,8) | Нереализованная прибыль/убыток (по умолчанию 0) |
| **total_pnl** | DECIMAL(20,8) | Общий PnL = realized_pnl + unrealized_pnl (по умолчанию 0) |
| **fees** | DECIMAL(20,8) | Комиссии (по умолчанию 0) |
| **entry_price** | DECIMAL(20,8) | Цена входа в позицию |
| **exit_price** | DECIMAL(20,8) | Цена выхода из позиции |
| **entry_signal_id** | UUID | Ссылка на сигнал входа (FK на `trading_signals.signal_id`) |
| **exit_signal_id** | UUID | Ссылка на сигнал выхода (FK на `trading_signals.signal_id`) |
| **position_size_at_entry** | DECIMAL(20,8) | Размер позиции при входе |
| **position_size_at_exit** | DECIMAL(20,8) | Размер позиции при выходе |
| **entry_timestamp** | TIMESTAMP | Время входа в позицию |
| **exit_timestamp** | TIMESTAMP | Время выхода из позиции |
| **is_closed** | BOOLEAN | Закрыта ли позиция (по умолчанию false) |
| **is_partial_close** | BOOLEAN | Частичное закрытие (по умолчанию false) |
| **computed_at** | TIMESTAMP | Время создания записи (по умолчанию NOW()) |
| **updated_at** | TIMESTAMP | Время последнего обновления (по умолчанию NOW()) |

### Связи с другими таблицами:

1. `prediction_targets` (один к одному):
   - `prediction_target_id` → `prediction_targets.id`
   - ON DELETE CASCADE
   - Связывает результат торговли с предсказанием модели

2. `trading_signals` (многие к одному):
   - `signal_id` → `trading_signals.signal_id` (основной сигнал)
   - `entry_signal_id` → `trading_signals.signal_id` (сигнал входа)
   - `exit_signal_id` → `trading_signals.signal_id` (сигнал выхода)
   - ON DELETE CASCADE

### Ограничения (Constraints):

1. `chk_pnl_consistency`: `total_pnl = realized_pnl + unrealized_pnl`
2. `chk_entry_exit_prices`: `entry_price > 0` и `exit_price > 0` (если не NULL)

### Индексы:

- По `prediction_target_id` (для связи с предсказаниями)
- По `signal_id` (для связи с сигналами)
- По `entry_signal_id` и `exit_signal_id` (для поиска по сигналам входа/выхода)
- По `is_closed` (для фильтрации закрытых позиций)
- По `total_pnl DESC` (для сортировки по прибыльности)
- По `computed_at DESC` (для временных запросов)

### Текущая статистика:

- Всего `prediction_targets` за 24 часа: 108
- С результатами торговли: 8 (7.4%)
- Сигналов с результатами: 8

### Схема связей:

```
prediction_targets (предсказание модели)
    │
    └──> prediction_trading_results (результат торговли)
            │
            ├──> trading_signals (сигнал, который привел к торговле)
            ├──> trading_signals (entry_signal_id - сигнал входа)
            ├──> trading_signals (exit_signal_id - сигнал выхода)
            └──> positions (позиция, связанная с торговлей)
```

### Назначение:

Таблица связывает предсказания модели (`prediction_targets`) с фактическими результатами торговли, позволяя:
- Оценивать качество модели по реальным торговым результатам
- Сравнивать предсказания с фактическими PnL
- Анализировать эффективность торговых сигналов
- Отслеживать входы/выходы и связанные сигналы

## Кто и когда заполняет/обновляет `prediction_trading_results`

### Кто заполняет таблицу:

**Сервис**: `PredictionTradingLinker` (`model-service/src/services/prediction_trading_linker.py`)

**Триггер**: обработка `OrderExecutionEvent` в `model-service/src/main.py`

### Когда создаются записи:

1. При получении `OrderExecutionEvent` (строки 232-398 в `main.py`):
   - Событие приходит из RabbitMQ при исполнении ордера
   - Обрабатывается в `handle_execution_event`
   - Если для сигнала есть `prediction_target`, создается запись в `prediction_trading_results`

2. Условия создания:
   - Есть `prediction_target` для `signal_id`
   - Есть запись в `position_orders` для ордера
   - Запись еще не существует (проверка по `prediction_target_id`)

### Когда обновляются записи:

1. При каждом `OrderExecutionEvent` (строки 367-374 в `main.py`):
   - Вызывается `update_trading_result_on_order_fill`
   - Обновляются: `realized_pnl`, `total_pnl`, `exit_price`, `exit_timestamp`, `is_closed`

2. Логика обновления:
   - Для закрытых/уменьшенных позиций (`relationship_type` = "closed" или "decreased")
   - Вычисляется `realized_pnl_delta` на основе цен входа/выхода
   - Обновляется только если `is_closed = false`

### Поток данных:

```
OrderExecutionEvent (RabbitMQ)
    ↓
ExecutionEventConsumer (model-service)
    ↓
handle_execution_event (main.py)
    ↓
PredictionTradingLinker.link_prediction_to_trading()  ← СОЗДАНИЕ
    ↓
PredictionTradingLinker.update_trading_result_on_order_fill()  ← ОБНОВЛЕНИЕ
    ↓
PredictionTradingResultsRepository.create/update()
    ↓
prediction_trading_results (PostgreSQL)
```

### Методы сервиса:

1. `link_prediction_to_trading()` (строки 27-112):
   - Создает запись при первом исполнении ордера
   - Ищет `prediction_target` по `signal_id`
   - Проверяет, что запись еще не существует
   - Сохраняет данные входа: `entry_price`, `entry_timestamp`, `position_size_at_entry`

2. `update_trading_result_on_order_fill()` (строки 114-189):
   - Обновляет при каждом исполнении ордера
   - Обновляет `realized_pnl`, `total_pnl`
   - При закрытии позиции устанавливает `exit_price`, `exit_timestamp`, `is_closed = true`

### Важные детали:

1. Создание происходит только если:
   - Есть `prediction_target` для сигнала
   - Есть связь ордера с позицией (`position_orders`)

2. Обновление происходит:
   - При каждом `OrderExecutionEvent`
   - Только для открытых позиций (`is_closed = false`)
   - С пересчетом PnL на основе цен входа/выхода

3. PnL вычисляется:
   - Для long: `(exit_price - entry_price) * quantity - fees`
   - Для short: `(entry_price - exit_price) * quantity - fees`

### Статистика:

- Всего записей: 8
- Все с `prediction_target_id` и `signal_id`
- Все открыты (`is_closed = false`)
- PnL = 0 (позиции еще не закрыты)

**Вывод**: Таблица заполняется и обновляется в реальном времени при обработке событий исполнения ордеров через RabbitMQ.

## API статистики от сигналов до таргетов и результатов

Это API, которое возвращает статистику от торговых сигналов до таргетов и результатов торговли.

**`GET /api/v1/stats`** (position-manager)

**Путь**: `/home/ubuntu/ytrader/position-manager/src/api/routes/stats.py`

### Параметры запроса:

- `asset` (optional) — фильтр по торговой паре (например, BTCUSDT)
- `mode` (optional) — фильтр по режиму торговли (one-way, hedge)
- `status` (optional) — фильтр по статусу (open, closed, all)
- `from_date` (optional) — начальная дата
- `to_date` (optional) — конечная дата
- `group_by` (optional) — группировка (asset, status, none)
- `include_details` (bool, default: false) — включить детальную информацию по позициям

### Что возвращает:

#### Базовая статистика (всегда):
- `summary` — общая статистика (total_positions, open_positions, closed_positions, total_pnl)
- `duration_stats` — статистика по времени удержания позиций
- `pnl_stats` — статистика по PnL (win_rate, winning_positions, losing_positions)

#### Детальная информация (при `include_details=true`):
Для каждой позиции включает:
- `entry_signal_id` — ID сигнала входа
- `exit_signal_id` — ID сигнала выхода
- `entry_prediction_json` — предсказание из метаданных сигнала входа
- `entry_predicted_values` — предсказанные значения из `prediction_targets`
- `entry_actual_values` — фактические значения из `prediction_targets`
- `exit_reason` — причина закрытия позиции
- `exit_rule_triggered` — правило, которое сработало

### Источник данных:

API объединяет данные из:
- `positions` — позиции
- `prediction_trading_results` — результаты торговли
- `trading_signals` — торговые сигналы (entry и exit)
- `prediction_targets` — предсказания модели (predicted_values и actual_values)
- `position_orders` — связь позиций с ордерами
- `signal_order_relationships` — связь сигналов с ордерами

### Пример использования:

```bash
curl -H "X-API-Key: your-position-manager-api-key" \
  "http://localhost:4800/api/v1/stats?include_details=true&status=closed"
```

