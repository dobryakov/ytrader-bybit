# Инварианты торговой логики

**Дата создания**: 2025-01-27  
**Статус**: Активные инварианты

## Цель

Документ фиксирует критические инварианты торговой логики, которые должны соблюдаться во всех реализациях. Нарушение инвариантов может привести к неправильной генерации торговых сигналов и финансовым потерям.

---

## Инвариант 1: Гистерезис для классификации без calibrated thresholds

### Формулировка

**Для классификационных моделей без calibrated thresholds:**

```
Если |buy_probability - sell_probability| < min_probability_diff:
    → signal_type = None (HOLD)
    → Сигнал НЕ генерируется
```

**Этот проверка применяется ДО проверки confidence threshold.**

### Обоснование

1. **Предотвращение сигналов при неопределенности**: Если модель не уверена в направлении (разница вероятностей мала), не следует генерировать торговый сигнал.

2. **Гистерезис**: Минимальная разница вероятностей создает "мертвую зону", предотвращающую частые переключения между BUY и SELL при небольших изменениях предсказаний.

3. **Независимость от confidence**: Даже если `confidence >= min_confidence_threshold`, но разница вероятностей мала, сигнал не генерируется. Это предотвращает ситуации, когда модель уверена в целом (высокий confidence), но не уверена в направлении.

### Реализация

**Файл**: `model-service/src/services/intelligent_signal_generator.py`

**Метод**: `_determine_signal_type()`

```python
# INVARIANT: For classification without calibrated thresholds:
# - If |buy_probability - sell_probability| < min_probability_diff → HOLD (None)
# - This check applies BEFORE confidence threshold
# - If difference is too small, signal is HOLD regardless of confidence
if not has_calibrated_thresholds:
    min_probability_diff = settings.model_min_probability_diff
    probability_diff = abs(buy_probability - sell_probability)
    
    if probability_diff < min_probability_diff:
        return None  # HOLD
```

### Настройка

**Параметр**: `MODEL_MIN_PROBABILITY_DIFF`

**Значение по умолчанию**: `0.05` (5%)

**Описание**: Минимальная разница между `buy_probability` и `sell_probability` для генерации сигнала.

### Примеры

#### Пример 1: Разница достаточна → сигнал генерируется

- `buy_probability = 0.6`, `sell_probability = 0.4`
- `probability_diff = |0.6 - 0.4| = 0.2`
- `min_probability_diff = 0.05`
- `0.2 >= 0.05` → ✅ Сигнал генерируется: **BUY**

#### Пример 2: Разница недостаточна → HOLD

- `buy_probability = 0.52`, `sell_probability = 0.48`
- `probability_diff = |0.52 - 0.48| = 0.04`
- `min_probability_diff = 0.05`
- `0.04 < 0.05` → ❌ Сигнал **НЕ генерируется** (HOLD)

**Важно**: Это применяется даже если `confidence = max(0.52, 0.48) = 0.52 >= min_confidence_threshold`.

#### Пример 3: Равенство вероятностей → HOLD

- `buy_probability = 0.5`, `sell_probability = 0.5`
- `probability_diff = |0.5 - 0.5| = 0.0`
- `min_probability_diff = 0.05`
- `0.0 < 0.05` → ❌ Сигнал **НЕ генерируется** (HOLD)

### Исключения

**Модели с calibrated thresholds**: Если модель имеет `probability_thresholds`, гистерезис уже обеспечивается в `model_inference.py`:
- Если ни `p_buy >= buy_threshold`, ни `p_sell >= sell_threshold` не выполняется → `semantic_prediction = 0` (HOLD)
- В этом случае `min_probability_diff` не применяется, так как гистерезис уже обеспечен

---

## Инвариант 2: Confidence threshold как дополнительный фильтр

### Формулировка

**Confidence threshold является дополнительным фильтром, но НЕ заменяет гистерезис.**

```
Порядок проверок:
1. Гистерезис (min_probability_diff или calibrated thresholds)
2. Confidence threshold
3. Генерация сигнала
```

### Обоснование

1. **Разные цели**: 
   - Гистерезис предотвращает сигналы при неопределенности в направлении
   - Confidence threshold предотвращает сигналы при низкой общей уверенности модели

2. **Дополнительная защита**: Даже если разница вероятностей достаточна, но общая уверенность низка, сигнал не генерируется.

3. **Независимость**: Оба фильтра работают независимо и должны применяться последовательно.

### Реализация

**Файл**: `model-service/src/services/intelligent_signal_generator.py`

**Метод**: `generate_signal()`

```python
# Step 1: Check confidence threshold (applies to all models)
confidence = prediction_result.get("confidence", 0.0)
if confidence < self.min_confidence_threshold:
    return None  # Signal not generated

# Step 2: Determine signal type (applies hysteresis)
signal_type = self._determine_signal_type(prediction_result)

# Step 3: If signal_type is None (HOLD), signal is not generated
if signal_type is None:
    return None
```

### Примеры

#### Пример 1: Оба фильтра пройдены → сигнал генерируется

- `buy_probability = 0.7`, `sell_probability = 0.3`
- `probability_diff = 0.4 >= 0.05` → ✅ Гистерезис пройден
- `confidence = 0.7 >= 0.6` → ✅ Confidence threshold пройден
- → ✅ Сигнал генерируется: **BUY**

#### Пример 2: Гистерезис не пройден → сигнал НЕ генерируется (независимо от confidence)

- `buy_probability = 0.52`, `sell_probability = 0.48`
- `probability_diff = 0.04 < 0.05` → ❌ Гистерезис не пройден
- `confidence = 0.52` (может быть >= или < threshold)
- → ❌ Сигнал **НЕ генерируется** (HOLD) - проверка confidence не выполняется

#### Пример 3: Гистерезис пройден, но confidence низкий → сигнал НЕ генерируется

- `buy_probability = 0.6`, `sell_probability = 0.4`
- `probability_diff = 0.2 >= 0.05` → ✅ Гистерезис пройден
- `confidence = 0.5 < 0.6` → ❌ Confidence threshold не пройден
- → ❌ Сигнал **НЕ генерируется**

---

## Инвариант 3: Регрессия имеет встроенный гистерезис

### Формулировка

**Для регрессионных моделей:**

```
Если predicted_return находится в диапазоне [-threshold, threshold]:
    → signal_type = None (HOLD)
    → Сигнал НЕ генерируется
```

### Реализация

**Файл**: `model-service/src/services/intelligent_signal_generator.py`

**Метод**: `_determine_signal_type()`

```python
if predicted_return > threshold:
    return "buy"
elif predicted_return < -threshold:
    return "sell"
else:
    # HOLD: predicted return is within threshold range (hysteresis)
    return None
```

### Примеры

- `predicted_return = 0.0005`, `threshold = 0.001` → **HOLD**
- `predicted_return = 0.0015`, `threshold = 0.001` → **BUY**
- `predicted_return = -0.0015`, `threshold = 0.001` → **SELL**

---

## Инвариант 4: Calibrated thresholds обеспечивают гистерезис

### Формулировка

**Для моделей с calibrated thresholds:**

```
Если ни p_buy >= buy_threshold, ни p_sell >= sell_threshold не выполняется:
    → semantic_prediction = 0 (HOLD)
    → Сигнал НЕ генерируется
```

### Реализация

**Файл**: `model-service/src/services/model_inference.py`

**Метод**: `predict()`

```python
if probability_thresholds and isinstance(probability_thresholds, dict) and sem_probs:
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

---

## Сводная таблица инвариантов

| Инвариант | Тип модели | Условие | Результат |
|-----------|------------|---------|-----------|
| **Гистерезис (min_probability_diff)** | Классификация без thresholds | `\|buy - sell\| < min_probability_diff` | HOLD |
| **Confidence threshold** | Все модели | `confidence < min_confidence_threshold` | Сигнал не генерируется |
| **Регрессия threshold** | Регрессия | `\|predicted_return\| < threshold` | HOLD |
| **Calibrated thresholds** | Классификация с thresholds | Ни один порог не пройден | HOLD (0) |

---

## Правила применения

1. **Порядок проверок**:
   ```
   1. Гистерезис (min_probability_diff или calibrated thresholds или regression threshold)
   2. Confidence threshold
   3. Генерация сигнала
   ```

2. **Независимость**: Каждый фильтр работает независимо. Если любой фильтр не пройден, сигнал не генерируется.

3. **Приоритет**: Гистерезис применяется ПЕРЕД confidence threshold. Если гистерезис не пройден, проверка confidence не выполняется.

---

## Тестирование инвариантов

### Тест 1: Равенство вероятностей

```python
def test_equal_probabilities_hold():
    """При равенстве вероятностей сигнал не генерируется."""
    buy_probability = 0.5
    sell_probability = 0.5
    min_probability_diff = 0.05
    
    probability_diff = abs(buy_probability - sell_probability)
    assert probability_diff < min_probability_diff
    # → signal_type должен быть None (HOLD)
```

### Тест 2: Недостаточная разница

```python
def test_insufficient_difference_hold():
    """При недостаточной разнице сигнал не генерируется."""
    buy_probability = 0.52
    sell_probability = 0.48
    min_probability_diff = 0.05
    
    probability_diff = abs(buy_probability - sell_probability)
    assert probability_diff < min_probability_diff
    # → signal_type должен быть None (HOLD)
```

### Тест 3: Достаточная разница, но низкий confidence

```python
def test_sufficient_difference_low_confidence():
    """При достаточной разнице, но низком confidence сигнал не генерируется."""
    buy_probability = 0.6
    sell_probability = 0.4
    min_probability_diff = 0.05
    confidence = 0.5
    min_confidence_threshold = 0.6
    
    probability_diff = abs(buy_probability - sell_probability)
    assert probability_diff >= min_probability_diff  # Гистерезис пройден
    assert confidence < min_confidence_threshold  # Confidence не пройден
    # → Сигнал не генерируется (возвращается None)
```

---

## Изменения инвариантов

**ВАЖНО**: Инварианты не должны изменяться без явного обсуждения и одобрения. Любые изменения должны быть задокументированы с указанием:
- Причины изменения
- Дата изменения
- Автор изменения
- Влияние на существующие модели и сигналы

---

## Связанные документы

- `docs/signal-direction-analysis.md` - Анализ преобразования сигналов
- `model-service/src/services/intelligent_signal_generator.py` - Реализация генерации сигналов
- `model-service/src/services/model_inference.py` - Реализация инференса модели

