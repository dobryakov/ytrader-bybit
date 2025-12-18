# План реализации API вычисления таргетов в feature-service

**Дата создания**: 2025-01-27  
**Дата реализации**: 2025-01-27  
**Статус**: ✅ Реализовано

## Цель

Реализовать endpoint в `feature-service` для вычисления фактических значений таргетов на текущий момент (например, на момент закрытия позиции для анализа качества торговли модели), чтобы убрать дублирование логики из `model-service` и обеспечить консистентность вычислений с процессом обучения моделей.

**Важно**: Endpoint должен поддерживать **разные типы таргетов** (regression, classification, risk_adjusted) и **разные presets** (returns, next_candle_direction, sharpe_ratio), так как модели могут предсказывать разные вещи в разных версиях. Структура ответа должна **динамически адаптироваться** под тип таргета из `target_config`.

---

## Проблема: задержки данных в feature-service

### Контекст

`feature-service` получает данные из RabbitMQ очереди с некоторой задержкой:
- Данные приходят из `ws-gateway` через RabbitMQ
- Данные буферизуются в `DataStorageService` (buffer_size=100)
- Данные записываются в Parquet файлы асинхронно
- Может быть задержка между реальным временем события и моментом, когда данные доступны в Parquet

**Важно**: Когда `model-service` запрашивает таргет для "текущего момента" (например, `target_timestamp = NOW()`), данные могут еще не быть доступны в ParquetStorage.

### Решение: адаптивный поиск с fallback в прошлое

Endpoint должен уметь:
1. Сначала пытаться найти данные для точного `target_timestamp`
2. Если данных нет, искать ближайшие доступные данные в прошлом (в пределах `max_lookback_seconds`)
3. Возвращать метаданные о том, какие данные были фактически использованы

---

## Этап 1: Новый endpoint `/api/v1/targets/compute`

Смысл эндпоинта - model service просит feature service посчитать таргет, фактически сложившийся на рынке на данный момент, чтобы затем сопоставить торговое предсказание и реальный результат при наступлении времени горизонта.

### 1.1. Спецификация endpoint

**Endpoint**: `POST /api/v1/targets/compute`

**Параметры запроса** (упрощенный вариант):
```json
{
  "symbol": "BTCUSDT",
  "prediction_timestamp": "2025-01-27T10:00:00Z",
  "target_timestamp": "2025-01-27T10:01:00Z",
  "target_registry_version": "1.0.0",
  "horizon_seconds": 60,  // Опционально, переопределяет horizon из registry
  "max_lookback_seconds": 300  // Опционально, по умолчанию 300 (5 минут)
}
```

**Альтернативный вариант (если target_timestamp вычисляется из horizon)**:
```json
{
  "symbol": "BTCUSDT",
  "prediction_timestamp": "2025-01-27T10:00:00Z",
  "horizon_seconds": 60,
  "target_registry_version": "1.0.0",
  "max_lookback_seconds": 300
}
```

**Логика работы**:
1. Feature-service загружает `target_config` из Target Registry по `target_registry_version`
2. Если указан `horizon_seconds`, он переопределяет `horizon` из конфига (для вычисления фактических значений может отличаться от обучения)
3. Используется `computation` конфигурация из registry (preset, price_source, lookup_method, tolerance_seconds)
4. Вычисляется таргет с использованием `TargetComputationEngine`

**Преимущества упрощенного API**:
- ✅ Меньше данных в запросе (только версия + горизонт вместо полного `target_config`)
- ✅ Конфигурация загружается из registry (консистентность с процессом обучения)
- ✅ Горизонт можно переопределить для вычисления фактических значений (может отличаться от обучения)
- ✅ Меньше дублирования (не передаем то, что уже есть в registry)
- ✅ `target_registry_version` уже хранится в `prediction_targets` таблице, легко извлечь

**Примечание**: 
- `target_config` хранится в `prediction_targets.target_config` как snapshot для воспроизводимости
- Но для вычисления фактических значений используем актуальную версию из registry (или snapshot, если версия совпадает)
- Горизонт может отличаться от обучения (например, если нужно вычислить фактическое значение для другого горизонта)

**Ответ (успех)** - структура зависит от типа таргета и preset из `target_config`:

**Для regression (returns preset)**:
```json
{
  "target_type": "regression",
  "preset": "returns",
  "target_value": 0.0012,  // Фактический return
  "price_at_prediction": 50000.0,
  "price_at_target": 50060.0,
  "prediction_timestamp_used": "2025-01-27T10:00:00Z",
  "target_timestamp_used": "2025-01-27T10:00:58Z",
  "data_available": true,
  "timestamp_adjusted": true,
  "lookback_seconds_used": 2,
  "computation_timestamp": "2025-01-27T10:01:05Z"
}
```

**Для classification (next_candle_direction preset)**:
```json
{
  "target_type": "classification",
  "preset": "next_candle_direction",
  "direction": "green",  // "green" или "red"
  "candle_open": 50000.0,
  "candle_close": 50060.0,
  "return_value": 0.0012,  // Для совместимости и анализа
  "prediction_timestamp_used": "2025-01-27T10:00:00Z",
  "target_timestamp_used": "2025-01-27T10:00:58Z",
  "data_available": true,
  "timestamp_adjusted": true,
  "lookback_seconds_used": 2,
  "computation_timestamp": "2025-01-27T10:01:05Z"
}
```

**Для risk_adjusted (sharpe_ratio preset)**:
```json
{
  "target_type": "risk_adjusted",
  "preset": "sharpe_ratio",
  "sharpe_value": 1.25,
  "returns_series": [0.001, 0.0005, -0.0003, ...],  // Серия returns для расчета
  "volatility": 0.0015,
  "prediction_timestamp_used": "2025-01-27T10:00:00Z",
  "target_timestamp_used": "2025-01-27T10:00:58Z",
  "data_available": true,
  "timestamp_adjusted": true,
  "lookback_seconds_used": 2,
  "computation_timestamp": "2025-01-27T10:01:05Z"
}
```

**Важно**: Структура ответа определяется динамически на основе `target_config.type` и `target_config.computation.preset` из Target Registry. Это обеспечивает гибкость для разных версий моделей и типов таргетов.

### 1.2. Практическое применение ответа

**Пример сценария 1 (regression)**: Модель предсказала `SELL` на горизонте 15 минут (900 секунд).

**Шаг 1**: Модель делает предсказание в `10:00:00`
- `predicted_values`: `{"value": -0.002, "confidence": 0.85}` (предсказанный return = -0.2%)
- Сигнал: `SELL` (отрицательный return)
- Сохраняется в `prediction_targets.predicted_values`

**Шаг 2**: Через 15 минут (`10:15:00`) вызывается endpoint `/api/v1/targets/compute`
- Запрос: `prediction_timestamp=10:00:00`, `target_timestamp=10:15:00`, `horizon_seconds=900`
- Ответ: `{"target_type": "regression", "preset": "returns", "target_value": -0.0015, "price_at_prediction": 50000.0, "price_at_target": 49925.0, ...}`
- Сохраняется в `prediction_targets.actual_values`

**Пример сценария 2 (classification, next_candle_direction)**: Модель предсказала цвет следующей свечи.

**Шаг 1**: Модель делает предсказание в `10:00:00`
- `predicted_values`: `{"direction": "red", "confidence": 0.75}` (предсказана красная свеча)
- Сигнал: `SELL` (на основе direction)
- Сохраняется в `prediction_targets.predicted_values`

**Шаг 2**: Через 15 минут (`10:15:00`) вызывается endpoint `/api/v1/targets/compute`
- Запрос: `prediction_timestamp=10:00:00`, `target_timestamp=10:15:00`, `horizon_seconds=900`
- Ответ: `{"target_type": "classification", "preset": "next_candle_direction", "direction": "red", "candle_open": 50000.0, "candle_close": 49925.0, "return_value": -0.0015, ...}`
- Сохраняется в `prediction_targets.actual_values`

**Шаг 3**: Model-service анализирует результаты

**Выводы, которые можно сделать** (зависят от типа таргета):

**Для regression (returns)**:
1. **Точность предсказания направления**:
   - Предсказано: `-0.002` (падение цены)
   - Фактически: `-0.0015` (падение цены)
   - ✅ **Направление угадано правильно** (оба отрицательные)
   - ❌ **Величина отличается** (предсказано -0.2%, фактически -0.15%)

2. **Ошибка предсказания (MAE)**:
   - `MAE = |predicted - actual| = |-0.002 - (-0.0015)| = 0.0005` (0.05%)

**Для classification (next_candle_direction)**:
1. **Точность предсказания направления свечи**:
   - Предсказано: `{"direction": "red"}` (красная свеча)
   - Фактически: `{"direction": "red"}` (красная свеча)
   - ✅ **Направление угадано правильно**

2. **Точность классификации**:
   - Можно рассчитать accuracy: `COUNT(correct) / COUNT(total)`
   - Для multi-class: confusion matrix

**Общие выводы** (для всех типов):

3. **Связь с торговым результатом**:
   - Если был открыт ордер по этому сигналу
   - `prediction_trading_results.total_pnl` покажет реальную прибыль/убыток
   - Можно сравнить: предсказание → факт → прибыль

4. **Анализ качества модели** (гибкие SQL-запросы):
   ```sql
   -- Для regression: точность по направлению и MAE
   SELECT 
       COUNT(*) as total,
       COUNT(CASE WHEN 
           (predicted_values->>'value')::float < 0 AND (actual_values->>'value')::float < 0 OR
           (predicted_values->>'value')::float > 0 AND (actual_values->>'value')::float > 0
       THEN 1 END) as correct_direction,
       AVG(ABS((predicted_values->>'value')::float - (actual_values->>'value')::float)) as mae
   FROM prediction_targets
   WHERE actual_values IS NOT NULL AND target_config->>'type' = 'regression';
   
   -- Для classification: accuracy по direction
   SELECT 
       COUNT(*) as total,
       COUNT(CASE WHEN 
           predicted_values->>'direction' = actual_values->>'direction'
       THEN 1 END) as correct_direction,
       COUNT(*) FILTER (WHERE predicted_values->>'direction' = actual_values->>'direction')::float / COUNT(*) as accuracy
   FROM prediction_targets
   WHERE actual_values IS NOT NULL 
     AND target_config->>'type' = 'classification'
     AND target_config->'computation'->>'preset' = 'next_candle_direction';
   ```

5. **Корреляция предсказания и торговли**:
   - Если предсказание правильное и `prediction_trading_results.total_pnl > 0`
   - ✅ **Модель работает корректно**: предсказание → факт → прибыль

6. **Использование для переобучения**:
   - Накопленные `predicted_values` и `actual_values` можно использовать как новый dataset
   - Сравнение с тестовыми метриками модели
   - Триггер переобучения при деградации точности

**Важно**: 
- `actual_values` дают **объективную оценку качества предсказаний** независимо от торговых результатов
- Структура ответа **динамически адаптируется** под тип таргета из `target_config`
- Это обеспечивает **гибкость для разных версий моделей** (regression → classification → risk_adjusted)

**Ответ (ошибка - данные недоступны)**:
```json
{
  "error": "data_unavailable",
  "message": "No data available within max_lookback_seconds",
  "requested_target_timestamp": "2025-01-27T10:01:00Z",
  "latest_available_timestamp": "2025-01-27T09:55:00Z",  // Последние доступные данные
  "max_lookback_seconds": 300,
  "data_gap_seconds": 360  // Разрыв между запрошенным временем и последними данными
}
```

---

## Этап 2: Логика адаптивного поиска данных

### 2.1. Переиспользование существующей логики fallback

**Важно**: В `TargetComputationEngine._compute_base_target()` уже реализована логика fallback для поиска ближайших данных:

1. **lookup_method** с вариантами:
   - `nearest_forward` - ищет ближайшее значение вперед (по умолчанию)
   - `nearest_backward` - ищет ближайшее значение назад
   - `nearest` - ищет ближайшее в любом направлении
   - `exact` - точное совпадение

2. **tolerance_seconds** - максимальное отклонение по времени при поиске

3. **merge_asof с direction** - механизм pandas для поиска ближайших значений

**Эта логика используется в dataset builder** (`optimized_builder._compute_targets()`) и уже работает корректно для исторических данных.

### 2.2. Дополнительный уровень fallback для задержек данных

Однако, существующая логика работает **внутри уже загруженных данных**. Нам нужен **дополнительный уровень** для определения доступности данных в ParquetStorage:

```python
async def find_available_data_range(
    symbol: str,
    target_timestamp: datetime,
    max_lookback_seconds: int = 300,
) -> Optional[Dict[str, Any]]:
    """
    Найти диапазон доступных данных и адаптировать target_timestamp.
    
    Это ДОПОЛНИТЕЛЬНЫЙ уровень fallback перед вызовом TargetComputationEngine.
    
    Алгоритм:
    1. Определить дату для target_timestamp
    2. Попытаться прочитать trades/klines за эту дату (и возможно предыдущую)
    3. Найти последний доступный timestamp в данных
    4. Если target_timestamp <= latest_available_timestamp:
       - Данные доступны, использовать target_timestamp как есть
    5. Если target_timestamp > latest_available_timestamp:
       - Вычислить разрыв: gap = target_timestamp - latest_available_timestamp
       - Если gap <= max_lookback_seconds:
         - Адаптировать target_timestamp = latest_available_timestamp
         - Вернуть метаданные об адаптации
       - Если gap > max_lookback_seconds:
         - Вернуть ошибку (данные слишком старые)
    
    Возвращает:
    {
        "adjusted_target_timestamp": datetime,  // Может отличаться от запрошенного
        "latest_available_timestamp": datetime,
        "timestamp_adjusted": bool,
        "lookback_seconds_used": int,
        "historical_data": pd.DataFrame  // Загруженные данные для TargetComputationEngine
    }
    """
```

### 2.3. Двухуровневая архитектура fallback

```
Уровень 1 (НОВЫЙ): Определение доступности данных в ParquetStorage
  ↓
  Найти последние доступные данные
  Адаптировать target_timestamp если нужно
  ↓
Уровень 2 (СУЩЕСТВУЮЩИЙ): TargetComputationEngine с lookup_method и tolerance
  ↓
  Использовать merge_asof для поиска ближайших цен
  Применить tolerance_seconds для фильтрации
  ↓
Результат: вычисленный таргет
```

**Преимущества**:
- Переиспользуем проверенную логику из dataset builder
- Добавляем только один дополнительный уровень для обработки задержек
- Консистентность с процессом обучения моделей

### 2.2. Определение "последних доступных данных"

**Проблема**: Как определить, что данные "последние доступные"?

**Варианты решения**:

1. **Простой подход**: Последний timestamp в Parquet файле за сегодняшнюю дату
   - Плюсы: Просто реализовать
   - Минусы: Может быть устаревшим, если данные не приходят

2. **С учетом задержки**: `latest_available = NOW() - expected_delay_seconds`
   - `expected_delay_seconds` = конфигурируемое значение (например, 30 секунд)
   - Плюсы: Учитывает реальную задержку
   - Минусы: Нужно настраивать

3. **Гибридный**: Использовать последний timestamp в Parquet, но с проверкой возраста
   - Если последний timestamp старше `NOW() - max_expected_delay`, считать его устаревшим
   - Плюсы: Баланс между простотой и надежностью

**Рекомендация**: Вариант 3 (гибридный)

### 2.3. Обработка edge cases

1. **Данные за сегодня отсутствуют**:
   - Попробовать данные за вчера
   - Если и там нет - вернуть ошибку

2. **Данные есть, но слишком старые**:
   - Если разрыв > `max_lookback_seconds` - вернуть ошибку
   - Логировать предупреждение

3. **Данные есть, но недостаточно для вычисления таргета**:
   - Например, для sharpe_ratio нужна серия данных
   - Вернуть ошибку с описанием

---

## Этап 3: Интеграция с TargetComputationEngine

### 3.1. Переиспользование существующего движка

`TargetComputationEngine.compute_target()` уже используется в dataset builder (`optimized_builder._compute_targets()`) и поддерживает:
- Работу с историческими данными через `historical_price_data`
- Различные presets (returns, sharpe_ratio, next_candle_direction, etc.)
- **Fallback логику через lookup_method и tolerance_seconds** (уже реализовано!)

**План использования** (аналогично dataset builder):

```python
# 1. Определить доступный диапазон данных (НОВЫЙ уровень)
data_range = await find_available_data_range(
    symbol=symbol,
    target_timestamp=target_timestamp,
    max_lookback_seconds=max_lookback_seconds,
)

if not data_range:
    return error_response("data_unavailable")

adjusted_target_timestamp = data_range["adjusted_target_timestamp"]
historical_data = data_range["historical_data"]

# 2. Создать DataFrame с prediction_timestamp (как в dataset builder)
prediction_df = pd.DataFrame({
    "timestamp": [prediction_timestamp],
    "price": [None]  # Будет заполнено из historical_data
})

# 3. Загрузить target_config из Target Registry по версии
# Используем тот же метод, что и в dataset builder (get_version)
target_config = await target_registry_version_manager.get_version(target_registry_version)
if not target_config:
    return error_response("target_registry_version_not_found")

# 4. Применить horizon override если указан
if horizon_seconds is not None:
    target_config["horizon"] = horizon_seconds
else:
    horizon_seconds = target_config.get("horizon", 60)

# 5. Вызвать TargetComputationEngine (СУЩЕСТВУЮЩАЯ логика)
computation_config = TargetComputationPresets.get_computation_config(
    target_config.get("computation")
)

# Используем ту же логику, что и в dataset builder
targets_df = TargetComputationEngine.compute_target(
    data=prediction_df,
    horizon=int(horizon_seconds),
    computation_config=computation_config,
    historical_price_data=historical_data,  # Как в optimized_builder._compute_targets()
)

# 6. Извлечь результат
if targets_df.empty:
    return error_response("computation_failed")

target_value = targets_df.iloc[0]["target"]

# 7. Извлечь цены для метаданных (если доступны в historical_data)
# Можно найти цены из historical_data по timestamps для возврата в ответе
# price_at_prediction = найти из historical_data по prediction_timestamp
# price_at_target = найти из historical_data по adjusted_target_timestamp
```

**Ключевое отличие от dataset builder**:
- Dataset builder работает с большим DataFrame (множество timestamp'ов)
- Наш endpoint работает с одним timestamp, но использует ту же логику вычисления

### 3.2. Загрузка исторических данных

**Переиспользование логики из dataset builder**:

В `optimized_builder._compute_targets()` используется:
```python
price_df = await self._parquet_storage.read_klines_range(
    symbol, start_date.date(), end_date.date()
)
```

**Наша реализация будет аналогичной**:

```python
async def load_historical_data_for_target_computation(
    symbol: str,
    prediction_timestamp: datetime,
    target_timestamp: datetime,
    buffer_seconds: int = 60,
) -> pd.DataFrame:
    """
    Загрузить исторические данные для вычисления таргета.
    
    Аналогично optimized_builder._compute_targets(), но для одного timestamp.
    
    Загружает данные в окне:
    - Начало: prediction_timestamp - buffer_seconds
    - Конец: target_timestamp + buffer_seconds
    
    Использует ParquetStorage.read_klines_range() (как в dataset builder)
    или read_trades_range() если нужны более точные данные.
    """
    start_date = (prediction_timestamp - timedelta(seconds=buffer_seconds)).date()
    end_date = (target_timestamp + timedelta(seconds=buffer_seconds)).date()
    
    # Используем ту же логику, что и dataset builder
    price_df = await parquet_storage.read_klines_range(
        symbol, start_date, end_date
    )
    
    return price_df
```

**Примечание**: Можно также использовать `read_trades_range()` для более точных данных, но klines обычно достаточно и быстрее.

---

## Этап 4: Обновление model-service

### 4.1. Изменения в `target_evaluator.py`

**Текущий код**:
```python
async def _compute_returns_actual(self, target: Dict[str, Any]):
    # Запрашивает две цены
    price_at_prediction = await feature_service_client.get_historical_price(...)
    price_at_target = await feature_service_client.get_historical_price(...)
    # Вычисляет return локально
    value = (price_at_target - price_at_prediction) / price_at_prediction
```

**Новый код**:
```python
async def _compute_returns_actual(self, target: Dict[str, Any]):
    # Получаем target_registry_version из prediction_targets
    target_registry_version = target["target_registry_version"]
    
    # Вычисляем horizon из timestamps
    horizon_seconds = int((target_ts - prediction_ts).total_seconds())
    
    # Запрашиваем готовый таргет (упрощенный API)
    result = await feature_service_client.compute_target(
        symbol=asset,
        prediction_timestamp=prediction_ts,
        target_timestamp=target_ts,
        target_registry_version=target_registry_version,
        horizon_seconds=horizon_seconds,  # Опционально, для переопределения
        max_lookback_seconds=300,  # Конфигурируемо
    )
    return {
        "value": result["target_value"],
        "price_at_prediction": result["price_at_prediction"],
        "price_at_target": result["price_at_target"],
    }
```

### 4.2. Новый метод в `feature_service_client.py`

```python
async def compute_target(
    self,
    symbol: str,
    prediction_timestamp: datetime,
    target_timestamp: datetime,
    target_registry_version: str,
    horizon_seconds: Optional[int] = None,  # Опционально, переопределяет horizon из registry
    max_lookback_seconds: int = 300,
    trace_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Вычислить фактическое значение таргета через Feature Service API.
    
    Args:
        symbol: Trading pair symbol
        prediction_timestamp: Timestamp when prediction was made
        target_timestamp: Timestamp for target computation
        target_registry_version: Target Registry version (config will be loaded from registry)
        horizon_seconds: Optional horizon override (if None, uses horizon from registry config)
        max_lookback_seconds: Maximum lookback for data availability fallback
        trace_id: Optional trace ID
    
    Returns:
        Dict с результатами вычисления или None при ошибке
    """
```

---

## Этап 5: Конфигурация

### 5.1. Новые переменные окружения

**feature-service**:
```bash
# Максимальная ожидаемая задержка данных (секунды)
TARGET_COMPUTATION_MAX_EXPECTED_DELAY_SECONDS=30

# Максимальный lookback для поиска данных (секунды)
TARGET_COMPUTATION_MAX_LOOKBACK_SECONDS=300

# Буфер для загрузки исторических данных (секунды)
TARGET_COMPUTATION_DATA_BUFFER_SECONDS=60
```

**model-service**:
```bash
# Максимальный lookback при запросе таргетов (секунды)
FEATURE_SERVICE_TARGET_COMPUTATION_MAX_LOOKBACK_SECONDS=300
```

---

## Этап 6: Обработка ошибок и логирование

### 6.1. Типы ошибок

1. **DATA_UNAVAILABLE**: Данные недоступны в пределах max_lookback_seconds
2. **INSUFFICIENT_DATA**: Данных недостаточно для вычисления (например, для sharpe_ratio)
3. **INVALID_CONFIG**: Неверная конфигурация target_config
4. **COMPUTATION_ERROR**: Ошибка при вычислении таргета

### 6.2. Логирование

Логировать:
- Когда используется адаптивный поиск (timestamp_adjusted = true)
- Размер lookback_seconds_used
- Разрыв между запрошенным и фактическим timestamp
- Предупреждения о старых данных

---

## Этап 7: Тестирование

### 7.1. Unit-тесты

- Тест адаптивного поиска данных (данные доступны)
- Тест адаптивного поиска данных (данные отстают, но в пределах max_lookback)
- Тест ошибки при слишком большом разрыве
- Тест вычисления различных presets

### 7.2. Интеграционные тесты

- Тест полного цикла: запрос → загрузка данных → вычисление → ответ
- Тест с реальными данными из Parquet

---

## Порядок реализации

1. ✅ Реализовать функцию `find_available_data_range()` - **НОВЫЙ уровень fallback**
2. ✅ Реализовать функцию `load_historical_data_for_target_computation()` - **переиспользование логики из dataset builder**
3. ✅ Реализовать endpoint `/api/v1/targets/compute` в `feature-service`
4. ✅ Интегрировать с `TargetComputationEngine` - **переиспользование существующей логики**
5. ✅ Добавить конфигурацию
6. ✅ Обновить `model-service` для использования нового endpoint'а
7. ✅ Написать тесты (включая тесты переиспользования логики)
8. ✅ Обновить документацию

---

## Переиспользование существующей логики

### Что переиспользуем:

1. **TargetComputationEngine.compute_target()** - полностью переиспользуем
   - Уже используется в `optimized_builder._compute_targets()` (строка 478, 487)
   - Поддерживает все presets и fallback логику через `lookup_method` и `tolerance_seconds`
   - Использует `merge_asof` с direction для поиска ближайших данных

2. **ParquetStorage.read_klines_range()** - переиспользуем
   - Уже используется в dataset builder (`optimized_builder._compute_targets()`, строка 455)
   - Проверенная логика загрузки исторических данных

3. **TargetComputationPresets.get_computation_config()** - переиспользуем
   - Уже используется в dataset builder (строка 472)
   - Консистентная обработка конфигурации таргетов

### Что добавляем (новый уровень):

1. **find_available_data_range()** - новый уровень для обработки задержек данных
   - Определяет последние доступные данные в ParquetStorage
   - Адаптирует `target_timestamp` если данные отстают
   - Работает **ДО** вызова `TargetComputationEngine`

2. **Endpoint wrapper** - обертка для вызова существующей логики
   - Принимает запрос от model-service
   - Вызывает `find_available_data_range()` → `load_historical_data()` → `TargetComputationEngine.compute_target()`
   - Возвращает результат с метаданными

### Двухуровневая архитектура fallback:

```
Уровень 1 (НОВЫЙ): Определение доступности данных в ParquetStorage
  ↓ find_available_data_range()
  Найти последние доступные данные
  Адаптировать target_timestamp если нужно
  ↓
Уровень 2 (СУЩЕСТВУЮЩИЙ): TargetComputationEngine с lookup_method и tolerance
  ↓ TargetComputationEngine.compute_target()
  Использовать merge_asof для поиска ближайших цен (lookup_method: nearest_forward/backward/nearest)
  Применить tolerance_seconds для фильтрации
  ↓
Результат: вычисленный таргет
```

**Преимущества переиспользования**:
- ✅ Консистентность с процессом обучения моделей (та же логика вычисления)
- ✅ Меньше кода для поддержки (переиспользуем проверенную логику)
- ✅ Автоматическая поддержка всех presets (returns, sharpe_ratio, next_candle_direction, etc.)
- ✅ Уже протестированная логика fallback внутри `TargetComputationEngine`

---

## Гибкость структуры для разных типов таргетов

### Поддержка разных версий моделей

Endpoint должен поддерживать **динамическую структуру ответа** в зависимости от типа таргета:

1. **Regression (returns preset)**:
   - Ответ: `{"target_type": "regression", "preset": "returns", "target_value": float, ...}`
   - Используется для моделей, предсказывающих return

2. **Classification (next_candle_direction preset)**:
   - Ответ: `{"target_type": "classification", "preset": "next_candle_direction", "direction": "green"|"red", ...}`
   - Используется для моделей, предсказывающих цвет следующей свечи

3. **Risk Adjusted (sharpe_ratio preset)**:
   - Ответ: `{"target_type": "risk_adjusted", "preset": "sharpe_ratio", "sharpe_value": float, ...}`
   - Используется для моделей, предсказывающих risk-adjusted метрики

### Преимущества гибкой структуры

- ✅ **Поддержка разных версий моделей**: Модель может предсказывать return в v1.0, цвет свечи в v2.0, sharpe в v3.0
- ✅ **Расширяемость**: Новые типы таргетов добавляются без изменения API контракта
- ✅ **Консистентность**: Структура ответа соответствует структуре `actual_values` в БД
- ✅ **Автоматическая адаптация**: Тип определяется из `target_config` в Target Registry

### Реализация гибкости

1. Endpoint загружает `target_config` из Target Registry по `target_registry_version`
2. Определяет `target_type` и `preset` из конфига
3. Форматирует ответ в зависимости от типа (см. раздел 3.1)
4. Model-service получает структурированный ответ и сохраняет в `actual_values`

---

## Важные замечания

1. **Консистентность**: Использовать ту же логику вычисления, что и при обучении моделей (через переиспользование `TargetComputationEngine`)
2. **Гибкость**: Структура ответа динамически адаптируется под тип таргета из `target_config` для поддержки разных версий моделей
3. **Производительность**: Кэшировать загруженные данные где возможно
4. **Надежность**: Graceful degradation при недоступности данных (двухуровневый fallback)
5. **Мониторинг**: Логировать случаи использования адаптивного поиска для анализа задержек данных
6. **Переиспользование**: Максимально использовать существующую логику из dataset builder для минимизации дублирования кода

