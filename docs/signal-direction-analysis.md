# Анализ цепочки преобразования торговых сигналов

**Дата**: 2025-01-27  
**Цель**: Проверка корректности преобразования предсказаний модели (buy/sell) в торговые ордера

## Резюме

✅ **Логика преобразования сигналов корректна** при использовании `label_mapping` (современный подход).  
⚠️ **Потенциальная проблема** в legacy fallback без `label_mapping` (зависит от того, как была обучена модель).

---

## Цепочка преобразования

### 1. Предсказание модели (`model-service/src/services/model_inference.py`)

#### 1.1. Классификация с label_mapping (рекомендуемый подход)

**Обучение модели** (`model_trainer.py`, строки 209-231):
- Для бинарной классификации с метками `{-1, +1}`:
  - `label_mapping = {-1: 0, 1: 1}`  # исходные метки → индексы модели
  - `reverse_mapping = {0: -1, 1: 1}`  # индексы модели → исходные метки
  - Сохраняется в `model._label_mapping_for_inference`

**Инференс** (`model_inference.py`, строки 424-431, 512-516):
```python
# Создание семантических вероятностей
if label_mapping:
    for class_idx, sem_label in label_mapping.items():
        sem_probs[sem_label] = probabilities[class_idx]
    # sem_probs[-1] = probabilities[0]  # класс 0 модели = семантический -1 (sell)
    # sem_probs[1] = probabilities[1]   # класс 1 модели = семантический +1 (buy)

# Вычисление buy/sell вероятностей
buy_probability = sem_probs.get(1, 0.0)   # вероятность класса +1 (buy)
sell_probability = sem_probs.get(-1, 0.0) # вероятность класса -1 (sell)
```

✅ **Корректно**: `+1` → `buy`, `-1` → `sell`

#### 1.2. Классификация без label_mapping (legacy fallback)

**Инференс** (`model_inference.py`, строки 522-527):
```python
# Legacy convention: [class_0_prob, class_1_prob]
# class_0 = buy, class_1 = sell (original behavior)
buy_probability = float(probabilities[0])
sell_probability = float(probabilities[1])
```

⚠️ **Потенциальная проблема**: Это работает только если модель была обучена с конвенцией `class_0 = buy, class_1 = sell`. Если модель была обучена с другими метками (например, `class_0 = sell, class_1 = buy`), то сигналы будут перепутаны.

#### 1.3. Регрессия

**Инференс** (`model_inference.py`, строки 541-557):
```python
predicted_return = float(prediction)
# Положительный return → buy, отрицательный → sell
```

✅ **Корректно**: Логика проста и понятна

---

### 2. Определение типа сигнала (`model-service/src/services/intelligent_signal_generator.py`)

**Метод `_determine_signal_type()`** (строки 610-651):

```python
# Для регрессии
if predicted_return > threshold:
    return "buy"
elif predicted_return < -threshold:
    return "sell"
else:
    return None  # HOLD

# Для классификации
if buy_probability > sell_probability:
    return "buy"
else:
    return "sell"
```

✅ **Корректно**: Логика правильная

---

### 3. Создание торгового сигнала (`model-service/src/services/intelligent_signal_generator.py`)

**Создание TradingSignal** (строки 464-488):
```python
signal = TradingSignal(
    signal_type=signal_type,  # "buy" или "sell"
    ...
)
```

✅ **Корректно**: `signal_type` правильно устанавливается

---

### 4. Обработка сигнала (`order-manager/src/services/signal_processor.py`)

**Преобразование в order_side** (строка 248):
```python
order_side = "Buy" if signal.signal_type.lower() == "buy" else "SELL"
```

✅ **Корректно**: `"buy"` → `"Buy"`, `"sell"` → `"SELL"`

---

### 5. Создание ордера (`order-manager/src/services/order_executor.py`)

**Преобразование для API и БД** (строки 1305-1307):
```python
# Side for Bybit API: "Buy" or "Sell"
side_api = "Buy" if signal.signal_type.lower() == "buy" else "Sell"

# Side for database: "Buy" or "SELL" (uppercase for SELL per constraint)
side_db = "Buy" if signal.signal_type.lower() == "buy" else "SELL"
```

✅ **Корректно**: Правильное преобразование для API и БД

---

## Проверка корректности

### Сценарий 1: Модель предсказывает BUY (с label_mapping)

1. Модель выдает: `probabilities = [0.2, 0.8]` (класс 0 = -1, класс 1 = +1)
2. `sem_probs = {-1: 0.2, 1: 0.8}`
3. `buy_probability = 0.8`, `sell_probability = 0.2`
4. `signal_type = "buy"` (т.к. 0.8 > 0.2)
5. `order_side = "Buy"`
6. Ордер создается как **BUY** ✅

### Сценарий 2: Модель предсказывает SELL (с label_mapping)

1. Модель выдает: `probabilities = [0.8, 0.2]` (класс 0 = -1, класс 1 = +1)
2. `sem_probs = {-1: 0.8, 1: 0.2}`
3. `buy_probability = 0.2`, `sell_probability = 0.8`
4. `signal_type = "sell"` (т.к. 0.2 < 0.8)
5. `order_side = "SELL"`
6. Ордер создается как **SELL** ✅

### Сценарий 3: Регрессия предсказывает положительный return

1. Модель выдает: `predicted_return = 0.002` (0.2%)
2. `threshold = 0.001` (0.1%)
3. `signal_type = "buy"` (т.к. 0.002 > 0.001)
4. Ордер создается как **BUY** ✅

### Сценарий 4: Регрессия предсказывает отрицательный return

1. Модель выдает: `predicted_return = -0.002` (-0.2%)
2. `threshold = 0.001` (0.1%)
3. `signal_type = "sell"` (т.к. -0.002 < -0.001)
4. Ордер создается как **SELL** ✅

---

## Потенциальные проблемы

### Проблема 1: Legacy fallback без label_mapping

**Описание**: Если модель была обучена без `label_mapping` и использует другую конвенцию (например, `class_0 = sell, class_1 = buy`), то сигналы будут перепутаны.

**Решение**: 
- ✅ Все новые модели должны использовать `label_mapping`
- ✅ Проверить существующие модели на наличие `label_mapping_for_inference` в `training_config`
- ⚠️ Если модель без `label_mapping`, проверить документацию/код обучения, чтобы понять конвенцию

**Проверка**:
```sql
-- Проверить, какие модели имеют label_mapping
SELECT 
    version,
    strategy_id,
    training_config->>'label_mapping_for_inference' as label_mapping
FROM model_versions
WHERE is_active = true
ORDER BY created_at DESC;
```

### Проблема 2: Несоответствие между обучением и инференсом

**Описание**: Если модель была обучена с одной конвенцией, но инференс использует другую.

**Решение**: 
- ✅ `label_mapping` сохраняется в `training_config` и загружается при инференсе
- ✅ Проверить, что `model_loader` правильно загружает `label_mapping`

---

## Рекомендации

### 1. Проверить активные модели

Выполнить SQL-запрос для проверки наличия `label_mapping`:

```sql
SELECT 
    version,
    strategy_id,
    is_active,
    CASE 
        WHEN training_config->>'label_mapping_for_inference' IS NOT NULL 
        THEN 'HAS_LABEL_MAPPING' 
        ELSE 'NO_LABEL_MAPPING' 
    END as mapping_status,
    training_config->>'label_mapping_for_inference' as label_mapping
FROM model_versions
WHERE is_active = true
ORDER BY created_at DESC;
```

### 2. Добавить логирование для отладки

Добавить логирование в `model_inference.py` для проверки преобразования:

```python
logger.debug(
    "Signal direction mapping",
    label_mapping=label_mapping,
    probabilities=probabilities.tolist(),
    sem_probs=sem_probs,
    buy_probability=buy_probability,
    sell_probability=sell_probability,
    semantic_prediction=semantic_prediction,
)
```

### 3. Добавить тесты

Создать unit-тесты для проверки преобразования сигналов:

```python
def test_buy_signal_with_label_mapping():
    # Модель предсказывает класс 1 (buy) с высокой вероятностью
    probabilities = np.array([0.2, 0.8])
    label_mapping = {0: -1, 1: 1}
    
    # Должно быть: buy_probability = 0.8, sell_probability = 0.2
    # signal_type = "buy"
    
def test_sell_signal_with_label_mapping():
    # Модель предсказывает класс 0 (sell) с высокой вероятностью
    probabilities = np.array([0.8, 0.2])
    label_mapping = {0: -1, 1: 1}
    
    # Должно быть: buy_probability = 0.2, sell_probability = 0.8
    # signal_type = "sell"
```

### 4. Документировать конвенции

Добавить в документацию явное описание конвенций:
- `+1` = `buy` (цена пойдет вверх)
- `-1` = `sell` (цена пойдет вниз)
- `class_0` модели = семантический `-1` (sell)
- `class_1` модели = семантический `+1` (buy)

---

## Выводы

1. ✅ **Логика преобразования корректна** при использовании `label_mapping` (современный подход)
2. ✅ **Регрессия работает правильно**: положительный return → buy, отрицательный → sell
3. ⚠️ **Legacy fallback требует проверки**: убедиться, что модели без `label_mapping` используют правильную конвенцию
4. ✅ **Преобразование в ордера корректно**: `"buy"` → `"Buy"`, `"sell"` → `"SELL"` / `"Sell"`

**Рекомендация**: Проверить активные модели на наличие `label_mapping` и при необходимости добавить его или задокументировать конвенцию для legacy моделей.

---

## Связь цвета свечи с торговыми сигналами

### Определение цвета свечи

В `feature-service/src/features/candle_patterns.py` (строки 86-87):

```python
is_green = 1.0 if close > open_price else 0.0  # Зеленая свеча = цена закрытия выше цены открытия
is_red = 1.0 if close < open_price else 0.0   # Красная свеча = цена закрытия ниже цены открытия
```

**Определение**:
- **Зеленая свеча** (`green`): `close > open` — цена закрытия выше цены открытия (цена выросла)
- **Красная свеча** (`red`): `close < open` — цена закрытия ниже цены открытия (цена упала)

### Таргет `next_candle_direction`

В `feature-service/src/services/target_computation.py` (строки 56-62):

```python
"next_candle_direction": {
    "formula": "returns",
    "price_source": "close",
    "future_price_source": "close",
    "lookup_method": "nearest_forward",
    "description": "Direction of next candle based on forward return sign",
}
```

**Вычисление**:
- Таргет вычисляется как `returns = (future_price - price) / price`
- **Положительный return** → зеленая свеча (цена выросла)
- **Отрицательный return** → красная свеча (цена упала)

### Преобразование предсказания цвета в торговый сигнал

В `model-service/src/services/intelligent_signal_generator.py`:

#### 1. Форматирование предсказания (строки 1202-1210):

```python
if preset == "next_candle_direction":
    buy_probability = prediction_result.get("buy_probability", 0.0)
    sell_probability = prediction_result.get("sell_probability", 0.0)
    direction = "green" if buy_probability > sell_probability else "red"
```

#### 2. Определение типа сигнала (строки 644-651):

```python
if buy_probability > sell_probability:
    return "buy"  # → Зеленая свеча → BUY
else:
    return "sell" # → Красная свеча → SELL
```

### Итоговое соответствие

| Предсказание модели | Цвет свечи | Торговый сигнал | Логика |
|---------------------|------------|-----------------|--------|
| `buy_probability > sell_probability` | **green** (зеленая) | **BUY** (покупка) | Модель предсказывает рост цены → покупаем |
| `sell_probability > buy_probability` | **red** (красная) | **SELL** (продажа) | Модель предсказывает падение цены → продаем |

### Проверка корректности

✅ **Логика корректна**:
- **Зеленая свеча** = цена выросла (`close > open`) → **BUY сигнал** ✅
- **Красная свеча** = цена упала (`close < open`) → **SELL сигнал** ✅

Это соответствует стандартной торговой логике:
- Если модель предсказывает рост цены (зеленая свеча), нужно покупать (BUY)
- Если модель предсказывает падение цены (красная свеча), нужно продавать (SELL)

### Примеры

#### Пример 1: Модель предсказывает зеленую свечу

1. Модель выдает: `buy_probability = 0.8`, `sell_probability = 0.2`
2. `direction = "green"` (т.к. 0.8 > 0.2)
3. `signal_type = "buy"` (т.к. 0.8 > 0.2)
4. Ордер создается как **BUY** ✅

**Интерпретация**: Модель предсказывает, что следующая свеча будет зеленой (цена вырастет), поэтому система генерирует сигнал на покупку.

#### Пример 2: Модель предсказывает красную свечу

1. Модель выдает: `buy_probability = 0.2`, `sell_probability = 0.8`
2. `direction = "red"` (т.к. 0.2 < 0.8)
3. `signal_type = "sell"` (т.к. 0.2 < 0.8)
4. Ордер создается как **SELL** ✅

**Интерпретация**: Модель предсказывает, что следующая свеча будет красной (цена упадет), поэтому система генерирует сигнал на продажу.

---

## Выводы по цвету свечи

1. ✅ **Зеленая свеча** (`green`) = рост цены (`close > open`) → **BUY сигнал** ✅
2. ✅ **Красная свеча** (`red`) = падение цены (`close < open`) → **SELL сигнал** ✅
3. ✅ **Логика преобразования корректна**: предсказание цвета свечи правильно преобразуется в торговые сигналы

---

## Гистерезис между BUY и SELL сигналами

### Текущая реализация

#### 1. Регрессия: есть гистерезис ✅

В `intelligent_signal_generator.py` (строки 632-643):

```python
if predicted_return > threshold:
    return "buy"
elif predicted_return < -threshold:
    return "sell"
else:
    # HOLD: predicted return is within threshold range
    return None
```

**Гистерезис**: Если `predicted_return` находится в диапазоне `[-threshold, threshold]`, сигнал не генерируется (HOLD).

**Пример**:
- `threshold = 0.001` (0.1%)
- `predicted_return = 0.0005` → **HOLD** (не генерируется сигнал)
- `predicted_return = 0.0015` → **BUY**
- `predicted_return = -0.0015` → **SELL**

#### 2. Классификация с calibrated thresholds: есть зона неопределенности ✅

В `model_inference.py` (строки 441-487):

```python
if probability_thresholds and isinstance(probability_thresholds, dict) and sem_probs:
    buy_threshold = thresholds_sem.get(1)
    sell_threshold = thresholds_sem.get(-1)
    p_buy = sem_probs.get(1, 0.0)
    p_sell = sem_probs.get(-1, 0.0)
    
    candidates = {}
    if buy_threshold is not None and p_buy >= buy_threshold:
        candidates[1] = p_buy
    if sell_threshold is not None and p_sell >= sell_threshold:
        candidates[-1] = p_sell
    
    if candidates:
        semantic_prediction = max(candidates.items(), key=lambda kv: kv[1])[0]
    else:
        # Ни один порог не выполнен — интерпретируем как отсутствие сигнала (hold)
        semantic_prediction = 0
```

**Гистерезис**: Если ни `p_buy >= buy_threshold`, ни `p_sell >= sell_threshold` не выполняется, то `semantic_prediction = 0` (HOLD).

**Пример**:
- `buy_threshold = 0.6`, `sell_threshold = 0.6`
- `p_buy = 0.5`, `p_sell = 0.5` → **HOLD** (ни один порог не пройден)
- `p_buy = 0.7`, `p_sell = 0.3` → **BUY** (порог пройден)
- `p_buy = 0.3`, `p_sell = 0.7` → **SELL** (порог пройден)

#### 3. Классификация без thresholds: нет гистерезиса ⚠️

В `intelligent_signal_generator.py` (строки 645-652):

```python
buy_probability = prediction_result.get("buy_probability", 0.0)
sell_probability = prediction_result.get("sell_probability", 0.0)

if buy_probability > sell_probability:
    return "buy"
else:
    return "sell"
```

**Проблема**: Если `buy_probability == sell_probability` (например, оба 0.5), всегда возвращается **"sell"** (так как условие `buy_probability > sell_probability` не выполняется).

**Примеры**:
- `buy_probability = 0.5`, `sell_probability = 0.5` → **SELL** ⚠️ (нет гистерезиса)
- `buy_probability = 0.5001`, `sell_probability = 0.4999` → **BUY**
- `buy_probability = 0.4999`, `sell_probability = 0.5001` → **SELL`

### Дополнительная защита: confidence threshold

В `intelligent_signal_generator.py` (строки 200-212):

```python
confidence = prediction_result.get("confidence", 0.0)
if confidence < self.min_confidence_threshold:
    return None  # Сигнал не генерируется
```

**Защита**: Если `confidence < min_confidence_threshold` (по умолчанию 0.6), сигнал не генерируется вообще.

**Пример**:
- `buy_probability = 0.5`, `sell_probability = 0.5` → `confidence = max(0.5, 0.5) = 0.5`
- Если `min_confidence_threshold = 0.6`, то сигнал **не генерируется** (возвращается `None`)

### Итоговая таблица гистерезиса

| Тип модели | Есть гистерезис? | Зона неопределенности |
|------------|------------------|----------------------|
| **Регрессия** | ✅ Да | `[-threshold, threshold]` → HOLD |
| **Классификация с calibrated thresholds** | ✅ Да | Если ни один порог не пройден → HOLD (0) |
| **Классификация без thresholds** | ⚠️ Нет | `buy_probability == sell_probability` → всегда SELL |
| **Confidence threshold** | ✅ Да | `confidence < threshold` → сигнал не генерируется |

### Проблема: равенство вероятностей

**Сценарий**: `buy_probability = 0.5`, `sell_probability = 0.5`

**Текущее поведение**:
1. `confidence = max(0.5, 0.5) = 0.5`
2. Если `min_confidence_threshold = 0.6` → сигнал **не генерируется** ✅ (защита работает)
3. Если `min_confidence_threshold <= 0.5` → генерируется сигнал **SELL** ⚠️ (нет гистерезиса)

**Риск**: При низком `min_confidence_threshold` может генерироваться SELL сигнал даже при полной неопределенности модели.

### Рекомендации

#### 1. Добавить минимальную разницу вероятностей

Модифицировать `_determine_signal_type()`:

```python
def _determine_signal_type(self, prediction_result: Dict[str, Any]) -> Optional[str]:
    buy_probability = prediction_result.get("buy_probability", 0.0)
    sell_probability = prediction_result.get("sell_probability", 0.0)
    
    # Минимальная разница для генерации сигнала (гистерезис)
    min_probability_diff = settings.model_min_probability_diff or 0.05  # 5%
    probability_diff = abs(buy_probability - sell_probability)
    
    if probability_diff < min_probability_diff:
        # Разница слишком мала - не генерируем сигнал (HOLD)
        return None
    
    if buy_probability > sell_probability:
        return "buy"
    else:
        return "sell"
```

#### 2. Использовать calibrated thresholds

Рекомендуется всегда использовать модели с `calibrated thresholds`, которые обеспечивают зону неопределенности.

#### 3. Увеличить min_confidence_threshold

Убедиться, что `min_confidence_threshold` достаточно высокий (например, 0.6), чтобы фильтровать неопределенные предсказания.

### Выводы

1. ✅ **Регрессия**: Есть гистерезис через threshold
2. ✅ **Классификация с thresholds**: Есть зона неопределенности
3. ⚠️ **Классификация без thresholds**: Нет гистерезиса при равенстве вероятностей
4. ✅ **Confidence threshold**: Дополнительная защита от неопределенных сигналов

**Рекомендация**: Добавить минимальную разницу вероятностей (`min_probability_diff`) для классификации без thresholds или всегда использовать модели с calibrated thresholds.

