# План реализации прослеживаемости цепочки от предсказания до факта

**Дата создания**: 2025-01-27  
**Статус**: План (не реализовано)

## Цели

1. Проследить полную цепочку: предсказание → сигнал → ордер → позиция → PnL → фактический таргет → вывод об успешности предсказания
2. Поддержать любые типы таргетов через гибкую JSONB-структуру
3. Обеспечить версионирование через snapshot конфигураций registry
4. Связать позиции с конкретными ордерами
5. **Связать предсказания модели с фактической торговой прибылью** - ответить на вопрос "модель предсказала buy/sell - я купил/продал - я фактически заработал (или не заработал)"

---

## Этап 1: Миграции БД

### 1.1. Миграция `028_create_position_orders_table.sql`

Создание таблицы связи позиций с ордерами.

**Файл**: `ws-gateway/migrations/028_create_position_orders_table.sql`

```sql
-- Migration: Create position_orders table
-- Reversible: Yes
-- Purpose: Link positions with orders that created/modified/closed them

CREATE TABLE IF NOT EXISTS position_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    position_id UUID NOT NULL REFERENCES positions(id) ON DELETE CASCADE,
    order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    relationship_type VARCHAR(20) NOT NULL CHECK (relationship_type IN ('opened', 'increased', 'decreased', 'closed', 'reversed')),
    size_delta DECIMAL(20, 8) NOT NULL,
    execution_price DECIMAL(20, 8) NOT NULL CHECK (execution_price > 0),
    executed_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT uq_position_order UNIQUE (position_id, order_id)
);

CREATE INDEX idx_position_orders_position_id ON position_orders(position_id);
CREATE INDEX idx_position_orders_order_id ON position_orders(order_id);
CREATE INDEX idx_position_orders_executed_at ON position_orders(executed_at DESC);
CREATE INDEX idx_position_orders_relationship_type ON position_orders(relationship_type);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_position_orders_relationship_type;
-- DROP INDEX IF EXISTS idx_position_orders_executed_at;
-- DROP INDEX IF EXISTS idx_position_orders_order_id;
-- DROP INDEX IF EXISTS idx_position_orders_position_id;
-- DROP TABLE IF EXISTS position_orders;
```

### 1.2. Миграция `029_create_prediction_targets_table.sql`

Создание гибкой таблицы предсказаний с JSONB.

**Файл**: `ws-gateway/migrations/029_create_prediction_targets_table.sql`

```sql
-- Migration: Create prediction_targets table
-- Reversible: Yes
-- Purpose: Store predictions and actual target values with flexible JSONB structure

CREATE TABLE IF NOT EXISTS prediction_targets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID NOT NULL REFERENCES trading_signals(signal_id) ON DELETE CASCADE,
    
    -- Timestamps
    prediction_timestamp TIMESTAMP NOT NULL,
    target_timestamp TIMESTAMP NOT NULL,
    
    -- Registry versions (for reproducibility)
    model_version VARCHAR(50) NOT NULL,
    feature_registry_version VARCHAR(50) NOT NULL,
    target_registry_version VARCHAR(50) NOT NULL,
    
    -- Full target configuration snapshot (JSONB)
    -- Stores complete config from target_registry_versions.config
    -- Allows understanding what config was used even if registry changes
    target_config JSONB NOT NULL,
    
    -- Predicted values (JSONB - flexible structure)
    -- Structure depends on target_config.type and target_config.computation.preset
    predicted_values JSONB NOT NULL,
    
    -- Actual values (JSONB - filled after target_timestamp)
    -- Structure matches predicted_values
    actual_values JSONB,
    
    -- Metadata for actual values computation
    actual_values_computed_at TIMESTAMP,
    actual_values_computation_error TEXT,
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_target_config_structure CHECK (
        target_config ? 'type' AND 
        target_config ? 'horizon' AND
        target_config->>'type' IN ('regression', 'classification', 'risk_adjusted') AND
        (target_config->>'horizon')::integer > 0
    ),
    CONSTRAINT chk_predicted_values_not_empty CHECK (
        jsonb_typeof(predicted_values) = 'object' AND
        predicted_values != '{}'::jsonb
    ),
    CONSTRAINT chk_target_timestamp_after_prediction CHECK (
        target_timestamp >= prediction_timestamp
    )
);

-- Indexes
CREATE INDEX idx_prediction_targets_signal_id ON prediction_targets(signal_id);
CREATE INDEX idx_prediction_targets_prediction_timestamp ON prediction_targets(prediction_timestamp DESC);
CREATE INDEX idx_prediction_targets_target_timestamp ON prediction_targets(target_timestamp DESC);
CREATE INDEX idx_prediction_targets_target_registry_version ON prediction_targets(target_registry_version);
CREATE INDEX idx_prediction_targets_model_version ON prediction_targets(model_version);
CREATE INDEX idx_prediction_targets_feature_registry_version ON prediction_targets(feature_registry_version);
CREATE INDEX idx_prediction_targets_pending_computation ON prediction_targets(target_timestamp) 
    WHERE actual_values_computed_at IS NULL AND target_timestamp <= NOW();
CREATE INDEX idx_prediction_targets_target_config_type ON prediction_targets((target_config->>'type'));
CREATE INDEX idx_prediction_targets_target_config_horizon ON prediction_targets(((target_config->>'horizon')::integer));
CREATE INDEX idx_prediction_targets_target_config_preset ON prediction_targets((target_config->'computation'->>'preset'))
    WHERE target_config->'computation' IS NOT NULL;

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_prediction_targets_target_config_preset;
-- DROP INDEX IF EXISTS idx_prediction_targets_target_config_horizon;
-- DROP INDEX IF EXISTS idx_prediction_targets_target_config_type;
-- DROP INDEX IF EXISTS idx_prediction_targets_pending_computation;
-- DROP INDEX IF EXISTS idx_prediction_targets_feature_registry_version;
-- DROP INDEX IF EXISTS idx_prediction_targets_model_version;
-- DROP INDEX IF EXISTS idx_prediction_targets_target_registry_version;
-- DROP INDEX IF EXISTS idx_prediction_targets_target_timestamp;
-- DROP INDEX IF EXISTS idx_prediction_targets_prediction_timestamp;
-- DROP INDEX IF EXISTS idx_prediction_targets_signal_id;
-- DROP TABLE IF EXISTS prediction_targets;
```

### 1.3. Миграция `030_add_horizon_to_trading_signals.sql`

Добавление полей для удобства запросов.

**Файл**: `ws-gateway/migrations/030_add_horizon_to_trading_signals.sql`

```sql
-- Migration: Add prediction horizon fields to trading_signals
-- Reversible: Yes
-- Purpose: Store horizon for easier queries (denormalization for performance)

ALTER TABLE trading_signals 
    ADD COLUMN IF NOT EXISTS prediction_horizon_seconds INTEGER,
    ADD COLUMN IF NOT EXISTS target_timestamp TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_trading_signals_target_timestamp ON trading_signals(target_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_horizon ON trading_signals(prediction_horizon_seconds);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_trading_signals_horizon;
-- DROP INDEX IF EXISTS idx_trading_signals_target_timestamp;
-- ALTER TABLE trading_signals DROP COLUMN IF EXISTS target_timestamp;
-- ALTER TABLE trading_signals DROP COLUMN IF EXISTS prediction_horizon_seconds;
```

### 1.4. Миграция `031_create_prediction_trading_results_table.sql`

Создание таблицы для связи предсказаний с торговой прибылью.

**Файл**: `ws-gateway/migrations/031_create_prediction_trading_results_table.sql`

```sql
-- Migration: Create prediction_trading_results table
-- Reversible: Yes
-- Purpose: Link predictions with actual trading PnL for model quality evaluation

CREATE TABLE IF NOT EXISTS prediction_trading_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_target_id UUID NOT NULL REFERENCES prediction_targets(id) ON DELETE CASCADE,
    signal_id UUID NOT NULL REFERENCES trading_signals(signal_id) ON DELETE CASCADE,
    
    -- Trading PnL metrics
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    fees DECIMAL(20, 8) DEFAULT 0,
    
    -- Entry/exit information
    entry_price DECIMAL(20, 8),
    exit_price DECIMAL(20, 8),
    entry_signal_id UUID REFERENCES trading_signals(signal_id),
    exit_signal_id UUID REFERENCES trading_signals(signal_id),
    
    -- Position information
    position_id UUID REFERENCES positions(id),
    position_size_at_entry DECIMAL(20, 8),
    position_size_at_exit DECIMAL(20, 8),
    
    -- Timestamps
    entry_timestamp TIMESTAMP,
    exit_timestamp TIMESTAMP,
    
    -- Status
    is_closed BOOLEAN NOT NULL DEFAULT false,
    is_partial_close BOOLEAN NOT NULL DEFAULT false,
    
    -- Metadata
    computed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_pnl_consistency CHECK (
        total_pnl = realized_pnl + unrealized_pnl
    ),
    CONSTRAINT chk_entry_exit_prices CHECK (
        (entry_price IS NULL OR entry_price > 0) AND
        (exit_price IS NULL OR exit_price > 0)
    )
);

-- Indexes
CREATE INDEX idx_prediction_trading_results_prediction_target_id ON prediction_trading_results(prediction_target_id);
CREATE INDEX idx_prediction_trading_results_signal_id ON prediction_trading_results(signal_id);
CREATE INDEX idx_prediction_trading_results_entry_signal_id ON prediction_trading_results(entry_signal_id);
CREATE INDEX idx_prediction_trading_results_exit_signal_id ON prediction_trading_results(exit_signal_id);
CREATE INDEX idx_prediction_trading_results_position_id ON prediction_trading_results(position_id);
CREATE INDEX idx_prediction_trading_results_is_closed ON prediction_trading_results(is_closed);
CREATE INDEX idx_prediction_trading_results_total_pnl ON prediction_trading_results(total_pnl DESC);
CREATE INDEX idx_prediction_trading_results_computed_at ON prediction_trading_results(computed_at DESC);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_prediction_trading_results_computed_at;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_total_pnl;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_is_closed;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_position_id;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_exit_signal_id;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_entry_signal_id;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_signal_id;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_prediction_target_id;
-- DROP TABLE IF EXISTS prediction_trading_results;
```

---

## Этап 2: Новые репозитории и модели

### 2.1. `model-service/src/database/repositories/prediction_target_repo.py`

Репозиторий для работы с `prediction_targets`.

**Основные методы**:
- `create()` - создание записи предсказания
- `update_actual_values()` - обновление фактических значений
- `get_pending_computations()` - получение предсказаний, ожидающих вычисления
- `get_by_signal_id()` - получение по signal_id

### 2.2. `order-manager/src/database/repositories/position_order_repo.py`

Репозиторий для связи позиций с ордерами.

**Основные методы**:
- `create()` - создание связи позиция-ордер
- `get_by_position_id()` - получение всех ордеров для позиции
- `get_by_order_id()` - получение всех позиций для ордера

### 2.3. `model-service/src/models/prediction_target.py`

Модель данных для предсказаний таргетов.

### 2.4. `model-service/src/database/repositories/prediction_trading_results_repo.py`

Репозиторий для работы с `prediction_trading_results`.

**Основные методы**:
- `create()` - создание записи торгового результата
- `update()` - обновление торгового результата (при закрытии позиции)
- `get_by_prediction_target_id()` - получение по prediction_target_id
- `get_by_signal_id()` - получение по signal_id
- `get_open_positions()` - получение открытых позиций для предсказаний
- `aggregate_pnl_by_signal()` - агрегация PnL по сигналу

---

## Этап 3: Получение конфигурации Target Registry

### 3.1. `model-service/src/services/target_registry_client.py`

Клиент для получения конфигурации Target Registry из БД.

**Основной метод**:
- `get_target_config(version: str) -> Optional[Dict[str, Any]]` - получение конфигурации из `target_registry_versions.config`

---

## Этап 4: Изменения в model-service

### 4.1. `model-service/src/services/intelligent_signal_generator.py`

**Изменения**:

1. После получения `prediction_result`:
   - Получить `target_registry_version` из модели/настроек
   - Получить конфигурацию Target Registry через `TargetRegistryClient`
   - Вычислить `target_timestamp = signal.timestamp + horizon_seconds`
   - Отформатировать `predicted_values` в зависимости от типа таргета
   - Сохранить в `prediction_targets`

2. Новый метод `_format_predicted_values()`:
   - Для regression: `{"value": float, "confidence": float}`
   - Для classification: `{"class": int, "probabilities": dict, "confidence": float}`
   - Для next_candle_direction: `{"direction": str, "confidence": float}`

3. Новый метод `_save_prediction_target()`:
   - Сохранение через `PredictionTargetRepository`

### 4.2. `model-service/src/publishers/signal_publisher.py`

**Изменения**:

В методе `_persist_signal()`:
- Извлечь `prediction_horizon_seconds` из `target_config`
- Вычислить `target_timestamp`
- Сохранить в `trading_signals`

---

## Этап 5: Изменения в order-manager

### 5.1. `order-manager/src/services/position_manager.py`

**Изменения**:

1. В методе `update_position()`:
   - Добавить параметр `order_id: Optional[UUID]`
   - После обновления позиции создать запись в `position_orders`
   - Определить `relationship_type` по изменению размера позиции

2. Новый метод `_determine_relationship_type()`:
   - Определяет тип связи: 'opened', 'increased', 'decreased', 'closed', 'reversed'

---

## Этап 6: Изменения в position-manager

### 6.1. `position-manager/src/services/position_manager.py`

Аналогичные изменения для записи связи позиций с ордерами.

---

## Этап 7: Сервис связи предсказаний с торговой прибылью

### 7.1. `model-service/src/services/prediction_trading_linker.py`

Сервис для связи предсказаний с торговой прибылью.

**Основные методы**:
- `link_prediction_to_trading()` - связать предсказание с торговым результатом
- `update_trading_result_on_order_fill()` - обновить результат при исполнении ордера
- `update_trading_result_on_position_close()` - обновить результат при закрытии позиции
- `compute_signal_pnl()` - вычислить PnL для сигнала (агрегация всех execution events)
- `link_entry_exit_signals()` - связать сигналы открытия и закрытия позиции

**Логика работы**:

1. **При создании сигнала** (если сигнал открывает позицию):
   - Создать запись в `prediction_trading_results` с `entry_signal_id`
   - Записать `entry_price` из execution event
   - Установить `is_closed = false`

2. **При исполнении ордера**:
   - Если ордер закрывает позицию (relationship_type = 'closed'):
     - Обновить `prediction_trading_results` с `exit_signal_id`
     - Вычислить `realized_pnl` из execution events
     - Установить `is_closed = true`

3. **При частичном закрытии**:
   - Создать новую запись для частичного закрытия
   - Обновить `realized_pnl` пропорционально закрытой части

4. **Периодическое обновление**:
   - Обновлять `unrealized_pnl` для открытых позиций
   - Вычислять `total_pnl = realized_pnl + unrealized_pnl`

### 7.2. Интеграция в order-manager

В `order-manager/src/services/order_executor.py` при обработке execution event:

```python
# После обновления позиции
if relationship_type == 'closed' or relationship_type == 'decreased':
    # Уведомить model-service о закрытии/уменьшении позиции
    await prediction_trading_linker.update_trading_result_on_order_fill(
        signal_id=signal.signal_id,
        order_id=order.id,
        execution_price=execution_price,
        execution_quantity=execution_quantity,
        realized_pnl=realized_pnl_delta,
        relationship_type=relationship_type,
    )
```

### 7.3. Периодическое обновление unrealized PnL

В `model-service/src/tasks/prediction_trading_update_task.py`:

```python
class PredictionTradingUpdateTask:
    """Периодически обновляет unrealized PnL для открытых позиций."""
    
    async def _update_unrealized_pnl(self):
        """Обновить unrealized PnL для всех открытых позиций."""
        # Получить все открытые prediction_trading_results
        # Для каждой позиции получить текущую цену
        # Вычислить unrealized_pnl = (current_price - entry_price) * size
        # Обновить запись
        pass
```

---

## Этап 8: Сервис вычисления фактических значений

### 8.1. `feature-service/src/api/targets.py` (НОВЫЙ)

**Endpoint**: `POST /api/v1/targets/compute`

Вычисление фактических значений таргетов делегировано в `feature-service` для:
- Консистентности с процессом обучения моделей (используется та же логика `TargetComputationEngine`)
- Устранения дублирования кода
- Поддержки разных типов таргетов (regression, classification, risk_adjusted) с динамической структурой ответа

**Параметры запроса**:
- `symbol`: Trading pair symbol
- `prediction_timestamp`: Timestamp when prediction was made
- `target_timestamp`: Timestamp for target computation (optional, computed from horizon if not provided)
- `target_registry_version`: Target Registry version
- `horizon_seconds`: Optional horizon override
- `max_lookback_seconds`: Maximum lookback for data availability fallback (default: 300)

**Особенности**:
- Двухуровневая архитектура fallback для обработки задержек данных
- Динамическая структура ответа в зависимости от типа таргета
- Переиспользование существующей логики из dataset builder

**Документация**: См. `docs/target-computation-api-plan.md` для полной спецификации.

### 8.2. `model-service/src/services/target_evaluator.py`

Сервис для вычисления фактических значений таргетов.

**Основные методы**:
- `evaluate_pending_targets()` - пакетная обработка ожидающих предсказаний
- `check_and_evaluate_immediate()` - немедленная проверка конкретного предсказания
- `_compute_actual_values()` - вычисление фактических значений на основе конфигурации таргета

**Методы вычисления** (теперь используют endpoint в feature-service):
- `_compute_returns_actual()` - для preset "returns" (вызывает `feature_service_client.compute_target()`)
- `_compute_candle_direction_actual()` - для preset "next_candle_direction" (вызывает `feature_service_client.compute_target()`)
- `_compute_sharpe_actual()` - для preset "sharpe_ratio" (вызывает `feature_service_client.compute_target()`)

### 8.3. `model-service/src/services/feature_service_client.py`

**Новый метод**: `compute_target()`

Клиент для вызова endpoint `/api/v1/targets/compute` в feature-service.

**Параметры**:
- `symbol`: Trading pair symbol
- `prediction_timestamp`: Timestamp when prediction was made
- `target_timestamp`: Timestamp for target computation
- `target_registry_version`: Target Registry version
- `horizon_seconds`: Optional horizon override
- `max_lookback_seconds`: Maximum lookback for data availability fallback

**Возвращает**: Dict с результатами вычисления или None при ошибке

### 8.4. `model-service/src/tasks/target_evaluation_task.py`

Периодическая задача с адаптивным интервалом.

**Особенности**:
- Адаптивный интервал на основе количества pending targets
- Минимальный интервал: 5 секунд (при большом количестве pending)
- Максимальный интервал: 60 секунд (когда нет pending)
- Базовый интервал: 10 секунд

**Методы**:
- `start()` - запуск задачи
- `stop()` - остановка задачи
- `_evaluation_loop()` - основной цикл
- `_calculate_adaptive_interval()` - расчет адаптивного интервала
- `trigger_immediate_check()` - немедленная проверка конкретного предсказания

### 8.5. Интеграция в `model-service/src/main.py`

**В startup**:
```python
from ..tasks.target_evaluation_task import target_evaluation_task
try:
    await target_evaluation_task.start()
    app.state.target_evaluation_task = target_evaluation_task
    logger.info("Target evaluation task started")
except Exception as e:
    logger.error("Failed to start target evaluation task", error=str(e), exc_info=True)
```

**В shutdown**:
```python
async def stop_target_evaluation_task():
    try:
        if hasattr(app.state, "target_evaluation_task"):
            await asyncio.wait_for(app.state.target_evaluation_task.stop(), timeout=5.0)
            logger.info("Target evaluation task stopped")
    except asyncio.TimeoutError:
        logger.warning("Target evaluation task stop timed out")
    except Exception as e:
        logger.error("Error stopping target evaluation task", error=str(e), exc_info=True)

shutdown_tasks.append(stop_target_evaluation_task())
```

### 8.6. Event-driven проверка

В `model-service/src/services/intelligent_signal_generator.py` после сохранения `prediction_target`:

```python
# Trigger immediate check if target timestamp has already passed
if target_timestamp <= datetime.utcnow():
    from ..tasks.target_evaluation_task import target_evaluation_task
    await target_evaluation_task.trigger_immediate_check(
        prediction_target_id=str(prediction_target["id"])
    )
```

### 8.7. Настройки в `model-service/src/config/settings.py`

**Новые переменные**:
- `FEATURE_SERVICE_TARGET_COMPUTATION_MAX_LOOKBACK_SECONDS` (default: 300) - максимальный lookback для поиска данных

**Настройки в `feature-service/src/config/__init__.py`**:
- `TARGET_COMPUTATION_MAX_EXPECTED_DELAY_SECONDS` (default: 30) - максимальная ожидаемая задержка данных
- `TARGET_COMPUTATION_MAX_LOOKBACK_SECONDS` (default: 300) - максимальный lookback для поиска данных
- `TARGET_COMPUTATION_DATA_BUFFER_SECONDS` (default: 60) - буфер для загрузки исторических данных

```python
# Target Evaluation Configuration
target_evaluation_base_interval_seconds: int = Field(
    default=10,
    alias="TARGET_EVALUATION_BASE_INTERVAL_SECONDS",
    description="Base interval for target evaluation task (seconds). Default: 10"
)
target_evaluation_min_interval_seconds: int = Field(
    default=5,
    alias="TARGET_EVALUATION_MIN_INTERVAL_SECONDS",
    description="Minimum interval when many pending targets (seconds). Default: 5"
)
target_evaluation_max_interval_seconds: int = Field(
    default=60,
    alias="TARGET_EVALUATION_MAX_INTERVAL_SECONDS",
    description="Maximum interval when no pending targets (seconds). Default: 60"
)
target_evaluation_pending_threshold_fast: int = Field(
    default=10,
    alias="TARGET_EVALUATION_PENDING_THRESHOLD_FAST",
    description="If pending targets >= this, use min_interval. Default: 10"
)
target_evaluation_pending_threshold_slow: int = Field(
    default=0,
    alias="TARGET_EVALUATION_PENDING_THRESHOLD_SLOW",
    description="If pending targets <= this, use max_interval. Default: 0"
)
```

### 8.6. Обновление `env.example`

```bash
# Target Evaluation Configuration
TARGET_EVALUATION_BASE_INTERVAL_SECONDS=10
TARGET_EVALUATION_MIN_INTERVAL_SECONDS=5
TARGET_EVALUATION_MAX_INTERVAL_SECONDS=60
TARGET_EVALUATION_PENDING_THRESHOLD_FAST=10
TARGET_EVALUATION_PENDING_THRESHOLD_SLOW=0
```

---

## Этап 9: API для прослеживания цепочки

### 9.1. `model-service/src/api/prediction_chain.py`

Новый endpoint для прослеживания цепочки.

**Endpoint**: `GET /api/v1/predictions/{signal_id}/chain`

**Возвращает**:
```json
{
  "prediction": {
    "signal_id": "uuid",
    "prediction_timestamp": "2025-01-27T10:00:00Z",
    "target_timestamp": "2025-01-27T10:03:00Z",
    "predicted_values": {...},
    "actual_values": {...}
  },
  "target": {
    "actual_price": 50000.0,
    "actual_return": 0.0012,
    "actual_value_computed_at": "2025-01-27T10:03:05Z"
  },
  "trading_result": {
    "realized_pnl": 100.5,
    "unrealized_pnl": 0.0,
    "total_pnl": 100.5,
    "fees": 2.0,
    "entry_price": 50000.0,
    "exit_price": 50100.0,
    "is_closed": true
  },
  "signal": {...},
  "orders": [...],
  "execution_events": [...],
  "positions": [...],
  "position_orders": [...]
}
```

### 9.2. `model-service/src/api/model_quality.py`

Новый endpoint для оценки качества модели на основе торговой прибыли.

**Endpoint**: `GET /api/v1/model-quality/trading-metrics`

**Параметры**:
- `model_version` - версия модели (опционально)
- `strategy_id` - стратегия (опционально)
- `start_date` - начальная дата (опционально)
- `end_date` - конечная дата (опционально)

**Возвращает**:
```json
{
  "model_version": "v1.0",
  "strategy_id": "default",
  "period": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-01-27T23:59:59Z"
  },
  "metrics": {
    "total_predictions": 1000,
    "closed_positions": 850,
    "open_positions": 150,
    "total_realized_pnl": 5000.5,
    "total_unrealized_pnl": 200.3,
    "total_pnl": 5200.8,
    "total_fees": 150.2,
    "net_pnl": 5050.6,
    "win_rate": 0.65,
    "average_win": 120.5,
    "average_loss": -80.3,
    "profit_factor": 1.5,
    "sharpe_ratio": 1.2
  },
  "by_prediction_type": {
    "buy": {
      "count": 500,
      "realized_pnl": 2500.0,
      "win_rate": 0.60
    },
    "sell": {
      "count": 500,
      "realized_pnl": 2500.5,
      "win_rate": 0.70
    }
  }
}
```

---

## Этап 10: API в feature-service для исторических цен

### 10.1. `feature-service/src/api/historical.py`

Новый endpoint для получения исторических цен.

**Endpoint**: `GET /api/v1/historical/price`

**Параметры**:
- `symbol` - торговый инструмент
- `timestamp` - целевой timestamp
- `lookback_seconds` - окно поиска данных

**Возвращает**: исторические данные цен для вычисления фактических таргетов

---

## Этап 11: Тесты

### 11.1. Unit-тесты

- `model-service/tests/unit/test_prediction_target_repo.py`
- `model-service/tests/unit/test_prediction_trading_results_repo.py`
- `model-service/tests/unit/test_prediction_trading_linker.py`
- `model-service/tests/unit/test_target_evaluator.py`
- `model-service/tests/unit/test_target_evaluation_task.py`
- `order-manager/tests/unit/test_position_order_repo.py`

### 11.2. Интеграционные тесты

- `model-service/tests/integration/test_prediction_chain.py`
- `model-service/tests/integration/test_prediction_trading_results.py`
- `tests/e2e/test_prediction_to_target_chain.py`
- `tests/e2e/test_prediction_to_trading_pnl.py`

---

## Порядок реализации

### Фаза 1: Подготовка БД (без breaking changes)

1. ✅ Создать миграцию `028_create_position_orders_table.sql`
2. ✅ Создать миграцию `029_create_prediction_targets_table.sql`
3. ✅ Создать миграцию `030_add_horizon_to_trading_signals.sql`
4. ✅ Создать миграцию `031_create_prediction_trading_results_table.sql`
5. Применить миграции: `docker compose exec ws-gateway python -m alembic upgrade head`

### Фаза 2: Репозитории и модели

6. Создать `model-service/src/database/repositories/prediction_target_repo.py`
7. Создать `model-service/src/database/repositories/prediction_trading_results_repo.py`
8. Создать `order-manager/src/database/repositories/position_order_repo.py`
9. Создать `model-service/src/models/prediction_target.py`
10. Создать `model-service/src/services/target_registry_client.py`

### Фаза 3: Запись данных

11. Изменить `model-service/src/services/intelligent_signal_generator.py`:
    - Добавить получение target_registry_version
    - Добавить сохранение в prediction_targets
12. Изменить `model-service/src/publishers/signal_publisher.py`:
    - Добавить сохранение horizon в trading_signals
13. Изменить `order-manager/src/services/position_manager.py`:
    - Добавить запись в position_orders
14. Изменить `position-manager/src/services/position_manager.py`:
    - Аналогично order-manager
15. Создать `model-service/src/services/prediction_trading_linker.py`
16. Интегрировать prediction_trading_linker в order-manager при обработке execution events
17. Создать `model-service/src/tasks/prediction_trading_update_task.py` для периодического обновления unrealized PnL

### Фаза 4: Вычисление фактических значений

18. Создать `model-service/src/services/target_evaluator.py`
19. Создать `model-service/src/tasks/target_evaluation_task.py`
20. Создать `feature-service/src/api/historical.py`
21. Интегрировать target_evaluation_task в `model-service/src/main.py`
22. Добавить event-driven проверку в `intelligent_signal_generator.py`
23. Добавить настройки в `model-service/src/config/settings.py`
24. Обновить `env.example`
25. Протестировать вычисление фактических значений
26. Протестировать связь предсказаний с торговой прибылью

### Фаза 5: API и аналитика

27. Создать `model-service/src/api/prediction_chain.py`
28. Создать `model-service/src/api/model_quality.py`
29. Добавить endpoints в роутер
30. Написать тесты
31. Обновить документацию

### Фаза 6: Обратная совместимость (опционально)

32. Создать скрипт для заполнения `prediction_targets` из существующих `trading_signals`
33. Создать скрипт для заполнения `position_orders` из истории ордеров
34. Создать скрипт для заполнения `prediction_trading_results` из существующих execution events

---

## Обратная совместимость

- ✅ Существующие таблицы не изменяются (кроме добавления полей)
- ✅ Старые сигналы продолжают работать
- ✅ `metadata` в `trading_signals` сохраняется для совместимости
- ✅ Новые таблицы заполняются только для новых событий
- ✅ Старые предсказания можно заполнить через backfill-скрипт

---

## Документация структур JSONB

### Схемы для `predicted_values` и `actual_values`

Документировать в `docs/prediction-targets-schemas.md`:

#### Regression (returns preset)
```json
predicted_values: {"value": float, "confidence": float}
actual_values: {"value": float, "price_at_prediction": float, "price_at_target": float}
```

#### Classification (default)
```json
predicted_values: {
  "class": int,
  "probabilities": {"-1": float, "0": float, "1": float},
  "confidence": float
}
actual_values: {
  "class": int,
  "price_at_prediction": float,
  "price_at_target": float
}
```

#### Classification (next_candle_direction preset)
```json
predicted_values: {"direction": "green"|"red", "confidence": float}
actual_values: {
  "direction": "green"|"red",
  "candle_open": float,
  "candle_close": float
}
```

#### Risk Adjusted (sharpe_ratio preset)
```json
predicted_values: {"sharpe": float, "confidence": float}
actual_values: {
  "sharpe": float,
  "returns_series": [float],
  "volatility": float
}
```

---

## Примеры запросов

### Найти все предсказания цвета свечи на 15 минут:
```sql
SELECT * FROM prediction_targets
WHERE target_config->>'type' = 'classification'
  AND (target_config->>'horizon')::integer = 900
  AND target_config->'computation'->>'preset' = 'next_candle_direction';
```

### Найти все предсказания return на 5 секунд:
```sql
SELECT * FROM prediction_targets
WHERE target_config->>'type' = 'regression'
  AND (target_config->>'horizon')::integer = 5
  AND target_config->'computation'->>'preset' = 'returns';
```

### Сравнить точность разных горизонтов:
```sql
SELECT 
    (target_config->>'horizon')::integer as horizon,
    target_config->>'type' as target_type,
    COUNT(*) as total_predictions,
    AVG((predicted_values->>'confidence')::float) as avg_confidence,
    AVG(ABS((predicted_values->>'value')::float - (actual_values->>'value')::float)) as mae
FROM prediction_targets
WHERE actual_values IS NOT NULL
GROUP BY horizon, target_type;
```

### Проследить цепочку для сигнала:
```sql
-- Получить полную цепочку
SELECT 
    ts.signal_id,
    ts.timestamp as signal_timestamp,
    pt.predicted_values,
    pt.actual_values,
    ptr.realized_pnl,
    ptr.unrealized_pnl,
    ptr.total_pnl,
    ptr.is_closed,
    o.id as order_id,
    o.status as order_status,
    po.relationship_type,
    p.realized_pnl as position_realized_pnl,
    p.unrealized_pnl as position_unrealized_pnl
FROM trading_signals ts
LEFT JOIN prediction_targets pt ON ts.signal_id = pt.signal_id
LEFT JOIN prediction_trading_results ptr ON pt.id = ptr.prediction_target_id
LEFT JOIN orders o ON ts.signal_id = o.signal_id
LEFT JOIN position_orders po ON o.id = po.order_id
LEFT JOIN positions p ON po.position_id = p.id
WHERE ts.signal_id = '...';
```

### Оценить качество модели по торговой прибыли:
```sql
-- Метрики качества модели на основе торговой прибыли
SELECT 
    pt.model_version,
    ts.strategy_id,
    COUNT(DISTINCT ptr.id) as total_predictions,
    COUNT(DISTINCT CASE WHEN ptr.is_closed THEN ptr.id END) as closed_positions,
    SUM(ptr.realized_pnl) as total_realized_pnl,
    SUM(ptr.unrealized_pnl) as total_unrealized_pnl,
    SUM(ptr.total_pnl) as total_pnl,
    SUM(ptr.fees) as total_fees,
    AVG(CASE WHEN ptr.is_closed AND ptr.total_pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
    AVG(CASE WHEN ptr.is_closed AND ptr.total_pnl > 0 THEN ptr.total_pnl END) as avg_win,
    AVG(CASE WHEN ptr.is_closed AND ptr.total_pnl < 0 THEN ptr.total_pnl END) as avg_loss
FROM prediction_targets pt
JOIN trading_signals ts ON pt.signal_id = ts.signal_id
LEFT JOIN prediction_trading_results ptr ON pt.id = ptr.prediction_target_id
WHERE pt.model_version = 'v1.0'
GROUP BY pt.model_version, ts.strategy_id;
```

### Ответить на вопрос "модель предсказала sell - я заработал?":
```sql
-- Для конкретного предсказания получить торговую прибыль
SELECT 
    pt.predicted_values->>'class' as predicted_class,
    ts.side as signal_side,
    ptr.realized_pnl,
    ptr.unrealized_pnl,
    ptr.total_pnl,
    ptr.fees,
    ptr.is_closed,
    ptr.entry_price,
    ptr.exit_price,
    CASE 
        WHEN ptr.total_pnl > 0 THEN 'PROFIT'
        WHEN ptr.total_pnl < 0 THEN 'LOSS'
        ELSE 'BREAKEVEN'
    END as result
FROM prediction_targets pt
JOIN trading_signals ts ON pt.signal_id = ts.signal_id
LEFT JOIN prediction_trading_results ptr ON pt.id = ptr.prediction_target_id
WHERE pt.signal_id = '...';
```

---

## Ожидаемый результат

После реализации можно будет:

1. ✅ Проследить цепочку от предсказания до фактического таргета через БД
2. ✅ Поддерживать любые типы таргетов через JSONB (regression, classification, risk_adjusted)
3. ✅ Сохранять snapshot конфигураций для воспроизводимости (даже при изменении registry)
4. ✅ Связывать позиции с конкретными ордерами
5. ✅ Анализировать качество предсказаний модели на исторических данных
6. ✅ Сравнивать разные конфигурации таргетов (разные горизонты, presets)
7. ✅ Вычислять фактические значения с адаптивной частотой проверки (5-60 секунд)
8. ✅ **Связать предсказания модели с фактической торговой прибылью** - ответить на вопрос "модель предсказала sell - я продал - я фактически заработал (или не заработал)"
9. ✅ Оценивать качество модели на основе торговых метрик (win rate, profit factor, Sharpe ratio)
10. ✅ Анализировать прибыльность по типам предсказаний (buy vs sell)
11. ✅ Отслеживать как открытые, так и закрытые позиции для предсказаний

---

## Риски и ограничения

1. **Производительность**: Дополнительные записи при каждом сигнале/ордере
   - **Решение**: Индексы, асинхронная запись где возможно

2. **Исторические данные**: Для старых сигналов фактические значения могут быть недоступны
   - **Решение**: Заполнение только для новых сигналов, опциональный backfill

3. **Точность таргетов**: Получение точной цены в момент `target_timestamp`
   - **Решение**: Использование данных из feature-service (Parquet), интерполяция при необходимости

4. **Горизонт предсказания**: Может отличаться для разных моделей
   - **Решение**: Хранение `horizon_seconds` в `prediction_targets` и `trading_signals`

5. **Частота проверки**: Для таргетов с коротким горизонтом (5 секунд) нужна частая проверка
   - **Решение**: Адаптивный интервал (5-60 секунд) + event-driven проверка при создании

6. **Связь предсказания с прибылью**: Прибыль может быть реализована только при закрытии позиции, которое может произойти от другого сигнала
   - **Решение**: Таблица `prediction_trading_results` связывает предсказание с entry/exit сигналами и вычисляет PnL на основе всех execution events

7. **Агрегация PnL**: Один сигнал может привести к нескольким ордерам и execution events
   - **Решение**: Агрегация всех execution events по `signal_id` в `prediction_trading_results`

---

## Примечания

- Все миграции должны быть обратимыми (rollback секция)
- Все новые сервисы должны иметь graceful shutdown
- Все ошибки должны логироваться, но не должны останавливать основной процесс
- Тесты должны покрывать основные сценарии использования
- Документация должна обновляться параллельно с реализацией

## Важные замечания по связи предсказаний с прибылью

### Сценарии использования

1. **Сигнал открывает позицию**:
   - Создается запись в `prediction_trading_results` с `entry_signal_id`
   - `is_closed = false`
   - `unrealized_pnl` обновляется периодически

2. **Сигнал закрывает позицию**:
   - Обновляется запись в `prediction_trading_results` с `exit_signal_id`
   - Вычисляется `realized_pnl` из всех execution events
   - `is_closed = true`

3. **Частичное закрытие**:
   - Создается новая запись для частичного закрытия
   - Оригинальная запись остается открытой с уменьшенным размером

4. **Несколько ордеров от одного сигнала**:
   - Все execution events агрегируются в одну запись `prediction_trading_results`
   - `realized_pnl` = сумма всех execution events для этого сигнала

### Метрики качества модели

Система позволяет вычислять:
- **Win Rate**: Процент прибыльных сделок
- **Average Win/Loss**: Средняя прибыль/убыток
- **Profit Factor**: Отношение общей прибыли к общему убытку
- **Sharpe Ratio**: Риск-скорректированная доходность
- **PnL по типам предсказаний**: Сравнение buy vs sell
- **PnL по горизонтам**: Сравнение разных горизонтов предсказаний

