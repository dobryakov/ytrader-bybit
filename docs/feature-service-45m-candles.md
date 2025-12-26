# План изменений: Универсальная функция расчёта паттернов свечей

## Цель

Создать универсальную функцию расчёта паттернов свечей, которая автоматически определяет интервал свечей на основе `lookback_window` из Feature Registry, вместо использования отдельных функций для разных версий.

## Текущая проблема

- Три отдельные функции: `compute_all_candle_patterns_3m`, `compute_all_candle_patterns_5m`, `compute_all_candle_patterns_15m`
- Выбор функции основан на версии Feature Registry (хардкод)
- Внутри функций хардкод: всегда берутся 3 свечи, интервал свечей фиксирован (1m для 3m, 5m для 15m)
- Для v1.7.0 с `lookback_window: 45m` требуется 3 свечи по 15 минут, но текущая логика не поддерживает это

## Решение

Создать универсальную функцию `compute_all_candle_patterns()`, которая:
- Принимает `lookback_window` из Feature Registry
- Автоматически определяет интервал свечей: `candle_interval = round(lookback_minutes / 3)`
- Всегда использует 3 свечи
- Поддерживает любой `lookback_window` (3m, 15m, 45m, 50m и т.д.)

## Поддерживаемые версии

- **v1.7.0**: Использует универсальную функцию с `lookback_window="45m"` (3 свечи по 15 минут)
- **v1.5.0**: Fallback на `compute_all_candle_patterns_15m()` (если не удалось получить lookback_window)

**Примечание**: Поддержка версий ниже 1.5.0 не требуется.

## Детальный план изменений

### 1. Новая универсальная функция в `candle_patterns.py`

**Создать:**
```python
def compute_all_candle_patterns(
    rolling_windows: RollingWindows,
    lookback_window: str,  # Например, "45m", "15m"
) -> Dict[str, Optional[float]]:
    """
    Универсальная функция для расчёта паттернов свечей.
    
    Автоматически определяет интервал свечей на основе lookback_window:
    - lookback_window = 45m → интервал свечи = 15m (3 свечи по 15 минут)
    - lookback_window = 15m → интервал свечи = 5m (3 свечи по 5 минут)
    - lookback_window = 3m → интервал свечи = 1m (3 свечи по 1 минуте)
    
    Если lookback_window не делится на 3, используется округление:
    - lookback_window = 50m → интервал свечи = 17m (округление 50/3 ≈ 16.67 → 17)
    
    Args:
        rolling_windows: RollingWindows instance with kline data
        lookback_window: Lookback window string (e.g., "45m", "15m")
        
    Returns:
        Dictionary of feature name -> feature value (0.0 or 1.0 for binary, float for ratios)
    """
```

**Логика:**
1. Парсинг `lookback_window` в минуты (поддержка: s, m, h, d)
2. Определение интервала свечи: `candle_interval_minutes = round(lookback_minutes / 3)`
3. Получение свечей:
   - Попытка получить свечи нужного интервала (1m, 5m, 15m, 30m и т.д.)
   - Если нет → агрегация из 1m свечей (логика из `compute_all_candle_patterns_15m`)
   - Требуется минимум 3 свечи
4. Расчёт паттернов:
   - Взять последние 3 свечи
   - Использовать существующую логику расчёта паттернов
   - Вернуть все 32 фичи (для v1.7.0)

### 2. Вспомогательная функция парсинга в `candle_patterns.py`

```python
def _parse_lookback_window_minutes(lookback_window: str) -> Optional[int]:
    """
    Парсинг lookback_window в минуты.
    
    Поддерживает суффиксы:
    - s: секунды → минуты (деление на 60)
    - m: минуты → минуты (без изменений)
    - h: часы → минуты (умножение на 60)
    - d: дни → минуты (умножение на 24 * 60)
    
    Args:
        lookback_window: Lookback window string (e.g., "45m", "1h", "300s")
        
    Returns:
        Lookback period in minutes, or None if parsing fails
    """
```

### 3. Изменения в `feature_computer.py`

**Заменить текущий код:**
```python
if self._feature_registry_version >= "1.5.0":
    candle_pattern_features = compute_all_candle_patterns_15m(rolling_windows)
elif self._feature_registry_version >= "1.4.0":
    candle_pattern_features = compute_all_candle_patterns_5m(rolling_windows)
else:
    candle_pattern_features = compute_all_candle_patterns_3m(rolling_windows)
```

**На:**
```python
# Получить lookback_window из Feature Registry для candle pattern фичи
lookback_window = self._get_candle_pattern_lookback_window()
if lookback_window:
    candle_pattern_features = compute_all_candle_patterns(rolling_windows, lookback_window)
else:
    # Fallback для v1.5.0 (если не удалось получить lookback_window)
    candle_pattern_features = compute_all_candle_patterns_15m(rolling_windows)
```

**Добавить метод:**
```python
def _get_candle_pattern_lookback_window(self) -> Optional[str]:
    """
    Получить lookback_window для candle pattern фич из Feature Registry.
    
    Ищет первую фичу, имя которой начинается с "candle_" или "pattern_",
    и возвращает её lookback_window.
    
    Returns:
        lookback_window string (e.g., "45m", "15m") или None если не найдено
    """
    if self._feature_registry_loader is None:
        return None
    
    try:
        registry_model = self._feature_registry_loader._registry_model
        if registry_model is None:
            return None
        
        # Найти первую candle pattern фичу (начинается с "candle_" или "pattern_")
        for feature in registry_model.features:
            if feature.name.startswith("candle_") or feature.name.startswith("pattern_"):
                return feature.lookback_window
        
        return None
    except Exception:
        return None
```

**Обновить импорты:**
```python
# Добавить
from src.features.candle_patterns import compute_all_candle_patterns

# Оставить для fallback
from src.features.candle_patterns import compute_all_candle_patterns_15m

# Удалить (не используются)
# from src.features.candle_patterns import compute_all_candle_patterns_3m
# from src.features.candle_patterns import compute_all_candle_patterns_5m
```

### 4. Изменения в `offline_engine.py`

Аналогично `feature_computer.py`:
- Заменить выбор функции на вызов универсальной
- Добавить метод `_get_candle_pattern_lookback_window()`
- Fallback только на `compute_all_candle_patterns_15m()` для v1.5.0

**Заменить:**
```python
if self._feature_registry_version >= "1.5.0":
    candle_pattern_features = compute_all_candle_patterns_15m(rolling_windows)
elif self._feature_registry_version >= "1.4.0":
    candle_pattern_features = compute_all_candle_patterns_5m(rolling_windows)
else:
    candle_pattern_features = compute_all_candle_patterns_3m(rolling_windows)
```

**На:**
```python
# Получить lookback_window из Feature Registry для candle pattern фичи
lookback_window = self._get_candle_pattern_lookback_window()
if lookback_window:
    candle_pattern_features = compute_all_candle_patterns(rolling_windows, lookback_window)
else:
    # Fallback для v1.5.0 (если не удалось получить lookback_window)
    candle_pattern_features = compute_all_candle_patterns_15m(rolling_windows)
```

**Обновить импорты:**
```python
# Добавить
from src.features.candle_patterns import compute_all_candle_patterns

# Оставить для fallback
from src.features.candle_patterns import compute_all_candle_patterns_15m

# Удалить (не используются)
# from src.features.candle_patterns import compute_all_candle_patterns_3m
# from src.features.candle_patterns import compute_all_candle_patterns_5m
```

### 5. Удаление неиспользуемых функций (опционально)

Можно удалить из `candle_patterns.py`:
- `compute_all_candle_patterns_3m()` — не используется
- `compute_all_candle_patterns_5m()` — не используется

**Оставить:**
- `compute_all_candle_patterns_15m()` — для fallback v1.5.0

**Примечание**: Удаление можно выполнить после проверки, что все тесты проходят с новой функцией.

## Обратная совместимость

### v1.7.0
- Использует универсальную функцию `compute_all_candle_patterns()` с `lookback_window="45m"`
- Автоматически определяет интервал свечи: 45m / 3 = 15m
- Получает 3 свечи по 15 минут

### v1.5.0
- Fallback на `compute_all_candle_patterns_15m()` если не удалось получить lookback_window из Feature Registry
- Сохраняет существующее поведение (3 свечи по 5 минут)

## Примеры работы

### Пример 1: v1.7.0 с lookback_window="45m"
```
lookback_window = "45m"
→ lookback_minutes = 45
→ candle_interval_minutes = round(45 / 3) = 15
→ candle_interval_str = "15m"
→ Получить 3 свечи по 15 минут
→ Вычислить паттерны
```

### Пример 2: v1.5.0 с lookback_window="15m"
```
lookback_window = "15m"
→ lookback_minutes = 15
→ candle_interval_minutes = round(15 / 3) = 5
→ candle_interval_str = "5m"
→ Получить 3 свечи по 5 минут
→ Вычислить паттерны
```

### Пример 3: Граничный случай с округлением (lookback_window="50m")
```
lookback_window = "50m"
→ lookback_minutes = 50
→ candle_interval_minutes = round(50 / 3) = round(16.67) = 17
→ candle_interval_str = "17m"
→ Попытка получить 3 свечи по 17 минут
→ Если нет → агрегация из 1m свечей (нужно 51 свеча по 1 минуте)
→ Вычислить паттерны
```

## Тестирование

### Тесты для v1.7.0
- Проверка работы с `lookback_window="45m"` (3 свечи по 15 минут)
- Проверка корректности расчёта всех 32 паттернов
- Проверка агрегации из 1m свечей, если 15m свечи недоступны

### Тесты для v1.5.0
- Проверка fallback на `compute_all_candle_patterns_15m()` если lookback_window не найден
- Проверка работы с `lookback_window="15m"` через универсальную функцию

### Граничные случаи
- `lookback_window="50m"` → округление до 17m
- `lookback_window="1m"` → округление до 1m (1/3 ≈ 0.33 → 1)
- `lookback_window="100m"` → округление до 33m

## Итоговая структура файлов

```
candle_patterns.py:
  - compute_all_candle_patterns(rolling_windows, lookback_window) [НОВАЯ]
  - _parse_lookback_window_minutes(lookback_window) [НОВАЯ]
  - compute_all_candle_patterns_15m() [ОСТАВИТЬ для fallback v1.5.0]
  - compute_all_candle_patterns_3m() [УДАЛИТЬ - не используется]
  - compute_all_candle_patterns_5m() [УДАЛИТЬ - не используется]

feature_computer.py:
  - _get_candle_pattern_lookback_window() [НОВАЯ]
  - Использовать compute_all_candle_patterns() с fallback на compute_all_candle_patterns_15m()

offline_engine.py:
  - _get_candle_pattern_lookback_window() [НОВАЯ]
  - Использовать compute_all_candle_patterns() с fallback на compute_all_candle_patterns_15m()
```

## Преимущества

1. **Универсальность**: Одна функция вместо трёх
2. **Автоматизация**: Автоматический выбор интервала свечей на основе `lookback_window`
3. **Гибкость**: Работает для любого `lookback_window` (3m, 15m, 45m, 60m и т.д.)
4. **Независимость от версии**: Не зависит от версии Feature Registry (кроме fallback)
5. **Сохраняет логику**: Использует существующую логику агрегации из 1m свечей

## Риски и митигация

### Риск 1: Ошибка парсинга lookback_window
**Митигация**: Fallback на `compute_all_candle_patterns_15m()` если парсинг не удался

### Риск 2: Feature Registry не загружен
**Митигация**: Fallback на `compute_all_candle_patterns_15m()` если `_feature_registry_loader` равен None

### Риск 3: Не найдена candle pattern фича в Feature Registry
**Митигация**: Fallback на `compute_all_candle_patterns_15m()` если фича не найдена

## Порядок реализации

1. Создать функцию `_parse_lookback_window_minutes()` в `candle_patterns.py`
2. Создать функцию `compute_all_candle_patterns()` в `candle_patterns.py`
3. Добавить метод `_get_candle_pattern_lookback_window()` в `feature_computer.py`
4. Обновить логику вызова в `feature_computer.py`
5. Добавить метод `_get_candle_pattern_lookback_window()` в `offline_engine.py`
6. Обновить логику вызова в `offline_engine.py`
7. Обновить импорты в обоих файлах
8. Протестировать с v1.7.0 и v1.5.0
9. Удалить неиспользуемые функции (опционально)

