## Обновленный план реализации новых фич

### Общая информация
- Версия: `1.7.3` (новая версия registry, дополняющая и расширяющая `1.7.2`)
- Формат фич: бинарный (0.0 / 1.0)
- Новые фичи используют источник данных: `kline`

### Как работает `lookback_window`

**Важно:** `lookback_window` указывается во времени (минутах/секундах), а не в количестве свечей.

- Формат: строка типа `"20m"`, `"5m"`, `"0s"` (минуты/секунды/часы)
- При реализации проверь, что в коде действительно поддерживаются произвольные интервалы.
- **Интервал свечей: 5 минут** (используются 5-минутные свечи)
- Использование: `get_klines_for_window("5m", start_time, end_time)`, где:
  - `start_time = now - timedelta(minutes=lookback_minutes)`
  - `end_time = now`
- Количество свечей: зависит от интервала свечей и наличия данных
  - Для `lookback_window: 20m` и интервала `"5m"` → примерно 4 свечи (20 минут / 5 минут = 4)
  - Для `lookback_window: 5m` и интервала `"5m"` → примерно 1 свеча (5 минут / 5 минут = 1)
  - Может быть меньше при пропусках в данных
- Минимальные требования: проверять фактическое количество свечей перед вычислением метрик
  - Для медиан: минимум 3 свечи (для 5-минутных свечей)
  - Для анализа паттернов: точное количество зависит от логики фичи

**Пример:**
```python
# Для lookback_window: 20m с 5-минутными свечами
now = rolling_windows.last_update
start_time = now - timedelta(minutes=20)
end_time = now
klines = rolling_windows.get_klines_for_window("5m", start_time, end_time)

# klines может содержать от 0 до ~4 свечей (20 минут / 5 минут = 4 свечи)
if len(klines) < 3:
    return 0.0  # Недостаточно данных
```

---

### 1. High Vol / Low Vol (ширина свечи)

**Названия фич:**
- `volatility_range_wide` (1.0 = широкая, 0.0 = узкая)
- `volatility_range_narrow` (1.0 = узкая, 0.0 = широкая)

**Логика:**
- Lookback: 20 минут (`lookback_window: 20m`)
- Получение данных: `get_klines_for_window("5m", start_time, end_time)`, где:
  - `start_time = now - timedelta(minutes=20)`
  - `end_time = now`
- Количество свечей: примерно 4 свечи (5-минутные: 20 минут / 5 минут = 4), но может быть меньше при пропусках
- Минимальное требование: минимум 3 свечи для надежного вычисления медианы
- Метрика: `current_range = high - low` (текущая/последняя свеча)
- Сравнение: медиана `high - low` за все свечи в окне (не менее 3 свечей)
- Порог: `current_range > 1.2 × median_range` → `volatility_range_wide = 1.0`
- Порог: `current_range <= 1.2 × median_range` → `volatility_range_narrow = 1.0`
- Относительно истории: абсолютные значения `high - low`, не нормализованные к цене
- Коэффициент 1.2 для фильтрации небольших отклонений
- Если недостаточно свечей (< 3) → возвращать 0.0 для обеих фич

**Регистрация:**
```yaml
- name: volatility_range_wide
  input_sources: [kline]
  lookback_window: 20m
  lookahead_forbidden: true
  max_lookback_days: 1
  data_sources:
    - source: kline
      timestamp_required: true
- name: volatility_range_narrow
  input_sources: [kline]
  lookback_window: 20m
  lookahead_forbidden: true
  max_lookback_days: 1
  data_sources:
    - source: kline
      timestamp_required: true
```

**Модуль:** `src/features/volatility_features.py` (новый)

---

### 2. Trend / Chop (направленность движения)

**Названия фич:**
- `trend_directional` (1.0 = направленное движение, 0.0 = chop)
- `trend_chop` (1.0 = chop, 0.0 = направленное движение)

**Логика:**
- Lookback: 5 минут (`lookback_window: 5m`)
- Получение данных: `get_klines_for_window("5m", start_time, end_time)`, где:
  - `start_time = now - timedelta(minutes=5)`
  - `end_time = now`
- Количество свечей: примерно 1 свеча (5-минутные: 5 минут / 5 минут = 1), но может быть меньше при пропусках
- **Проблема:** Для анализа 5 свечей нужен lookback минимум 25 минут (5 свечей × 5 минут = 25 минут)
- **Решение:** Увеличить `lookback_window` до `25m` для получения 5 свечей
- Минимальное требование: минимум 5 свечей (для индексов 0..4)
- Метрика:
  - Итоговое смещение: `net_movement = |close_4 - open_0|` (последняя и первая свеча)
  - Суммарное движение: `total_movement = sum(|high_i - low_i|)` для всех свечей в окне
  - Efficiency ratio: `efficiency = net_movement / total_movement` (если total_movement > 0)
- Пороги:
  - `efficiency >= 0.5` → `trend_directional = 1.0` (можно поднять до 0.6 для жесткой фильтрации)
  - `efficiency < 0.35` → `trend_chop = 1.0`
  - Между 0.35 и 0.5: оба = 0.0 (неопределенное состояние)
- Направленность как факт: только абсолютное значение, без учета направления
- Если недостаточно свечей (< 5) или `total_movement == 0` → возвращать 0.0 для обеих фич

**Важно:** Для 5-минутных свечей нужно увеличить `lookback_window` до `25m` вместо `5m`, чтобы получить 5 свечей.

**Регистрация:**
```yaml
- name: trend_directional
  input_sources: [kline]
  lookback_window: 25m
  lookahead_forbidden: true
  max_lookback_days: 1
  data_sources:
    - source: kline
      timestamp_required: true
- name: trend_chop
  input_sources: [kline]
  lookback_window: 25m
  lookahead_forbidden: true
  max_lookback_days: 1
  data_sources:
    - source: kline
      timestamp_required: true
```

**Модуль:** `src/features/trend_features.py` (новый)

---

### 3. Volume Spike (спайк объема)

**Название фичи:**
- `volume_spike_detected` (1.0 = спайк, 0.0 = нет спайка)

**Логика:**
- Lookback: 20 минут (`lookback_window: 20m`)
- Получение данных: `get_klines_for_window("5m", start_time, end_time)`, где:
  - `start_time = now - timedelta(minutes=20)`
  - `end_time = now`
- Количество свечей: примерно 4 свечи (5-минутные: 20 минут / 5 минут = 4), но может быть меньше при пропусках
- Минимальное требование: минимум 3 свечи для надежного вычисления медианы
- Метрика: медиана объема за все свечи в окне (не менее 3 свечей)
- Порог: `current_volume >= 2.0 × median_volume` → `volume_spike_detected = 1.0`
- Если недостаточно свечей (< 3) → возвращать 0.0

**Регистрация:**
```yaml
- name: volume_spike_detected
  input_sources: [kline]
  lookback_window: 20m
  lookahead_forbidden: true
  max_lookback_days: 1
  data_sources:
    - source: kline
      timestamp_required: true
```

**Модуль:** `src/features/volume_features.py` (новый, или добавить в существующий модуль)

---

### 4. Sessions (London / NY / Asian)

**Названия фич:**
- `session_london_active` (1.0 = активна, 0.0 = неактивна)
- `session_ny_active` (1.0 = активна, 0.0 = неактивна)
- `session_asian_active` (1.0 = активна, 0.0 = неактивна)

**Логика:**
- Простая проверка UTC-часов (без pytz, без учета DST)
- London: 08:00–11:00 UTC (включительно)
- NY: 13:00–16:00 UTC (включительно)
- Asian: 00:00–08:00 UTC (включительно)
- Бинарный формат: если текущее время (UTC) попадает в окно → 1.0, иначе 0.0
- Использовать `timestamp.hour` из datetime объекта (UTC)

**Регистрация:**
```yaml
- name: session_london_active
  input_sources: [kline]
  lookback_window: 0s
  lookahead_forbidden: true
  max_lookback_days: 0
  data_sources:
    - source: kline
      timestamp_required: true
- name: session_ny_active
  input_sources: [kline]
  lookback_window: 0s
  lookahead_forbidden: true
  max_lookback_days: 0
  data_sources:
    - source: kline
      timestamp_required: true
- name: session_asian_active
  input_sources: [kline]
  lookback_window: 0s
  lookahead_forbidden: true
  max_lookback_days: 0
  data_sources:
    - source: kline
      timestamp_required: true
```

**Модуль:** `src/features/temporal_features.py` (расширить существующий)

**Реализация:**
```python
def compute_session_london_active(timestamp: datetime) -> float:
    """Check if London session is active (08:00-11:00 UTC)."""
    hour = timestamp.hour
    return 1.0 if 8 <= hour < 11 else 0.0

def compute_session_ny_active(timestamp: datetime) -> float:
    """Check if NY session is active (13:00-16:00 UTC)."""
    hour = timestamp.hour
    return 1.0 if 13 <= hour < 16 else 0.0

def compute_session_asian_active(timestamp: datetime) -> float:
    """Check if Asian session is active (00:00-08:00 UTC)."""
    hour = timestamp.hour
    return 1.0 if 0 <= hour < 8 else 0.0
```

---

### 5. Exhaustion (истощение импульса)

**Название фичи:**
- `exhaustion_detected` (1.0 = обнаружено, 0.0 = не обнаружено)

**Логика:**
- Lookback: 20 минут (`lookback_window: 20m`) для вычисления медиан
- Получение данных: `get_klines_for_window("5m", start_time, end_time)`, где:
  - `start_time = now - timedelta(minutes=20)` (для медиан)
  - `end_time = now`
- Анализ: текущая завершенная свеча + следующая завершенная свеча (нужно 2 завершенные свечи)
- Проверка завершенности: свеча считается завершенной, если её `timestamp` меньше текущего времени минус интервал свечи (для 5m свечи: `timestamp < now - 5 minutes`)
- Минимальное требование: минимум 3 свечи для вычисления медиан, плюс 2 завершенные свечи для анализа
- Критерии "сильной свечи" (для текущей завершенной):
  - `body_size_current > 2.0 × median_body_size` (медиана за все свечи в окне, не менее 3)
  - `volume_current > 2.0 × median_volume` (медиана за все свечи в окне, не менее 3)
- Критерии "не продолжилось" (для следующей завершенной):
  - Следующая свеча имеет маленькое тело: `body_size_next < 0.3 × body_size_current`
  - ИЛИ движение откатилось: `|close_next - close_current| / close_current < 0.001` (менее 0.1%)
  - ИЛИ противоположное направление: `(close_current > open_current) != (close_next > open_next)`
- Если сильная свеча И не продолжилось → `exhaustion_detected = 1.0`
- Если нет завершенных свечей или недостаточно данных → `exhaustion_detected = 0.0`

**Регистрация:**
```yaml
- name: exhaustion_detected
  input_sources: [kline]
  lookback_window: 20m
  lookahead_forbidden: true
  max_lookback_days: 1
  data_sources:
    - source: kline
      timestamp_required: true
```

**Модуль:** `src/features/exhaustion_features.py` (новый)

**Реализация проверки завершенности:**
```python
def _is_candle_completed(candle_timestamp: datetime, current_time: datetime, interval_minutes: int = 1) -> bool:
    """
    Check if candle is completed (closed).
    
    A candle is considered completed if its timestamp is at least one interval
    before the current time.
    """
    time_diff = (current_time - candle_timestamp).total_seconds()
    return time_diff >= (interval_minutes * 60)
```

---

## Структура реализации

### Новые файлы:
1. `src/features/volatility_features.py` — High Vol / Low Vol
2. `src/features/trend_features.py` — Trend / Chop
3. `src/features/volume_features.py` — Volume Spike
4. `src/features/exhaustion_features.py` — Exhaustion

### Изменения в существующих файлах:
1. `src/features/temporal_features.py` — добавить функции для Sessions
2. `src/services/feature_computer.py` — добавить вызовы новых функций
3. `src/services/offline_engine.py` — добавить вызовы новых функций
4. `config/versions/feature_registry_v1.7.3.yaml` — новый файл с регистрацией всех фич

### Тесты:
1. `tests/unit/test_volatility_features.py`
2. `tests/unit/test_trend_features.py`
3. `tests/unit/test_volume_features.py`
4. `tests/unit/test_exhaustion_features.py`
5. `tests/unit/test_temporal_features.py` — расширить существующие

---

## Детали реализации

### Пороги и коэффициенты:
- High Vol: коэффициент 1.2 × медиана, lookback_window: 20m (≈4 свечи), минимум 3 свечи
- Trend/Chop: efficiency >= 0.5 для trend, < 0.35 для chop, lookback_window: 25m (≈5 свечей), минимум 5 свечей
- Volume Spike: >= 2.0 × медиана, lookback_window: 20m (≈4 свечи), минимум 3 свечи
- Exhaustion: только для завершенных свечей, проверка через timestamp, lookback_window: 20m (≈4 свечи), минимум 3 свечи для медиан

### Обработка граничных случаев:
- Недостаточно данных: возвращать 0.0 для бинарных фич
- Деление на ноль: проверять перед вычислением efficiency ratio
- Завершенность свечей: проверять timestamp перед вычислением exhaustion
- Пропуски в данных: `lookback_window` указывается в минутах, но фактическое количество свечей может быть меньше из-за пропусков; проверять минимальное количество свечей перед вычислением медиан и метрик
- Временные окна: использовать `get_klines_for_window("5m", start_time, end_time)` с `start_time = now - timedelta(minutes=lookback_minutes)` для получения всех 5-минутных свечей в окне
- Расчет количества свечей: для 5-минутных свечей количество = `lookback_minutes / 5` (например, 20m → 4 свечи, 25m → 5 свечей)
- Интервал свечей: все новые фичи используют 5-минутные свечи (интервал "5m"), а не 1-минутные

