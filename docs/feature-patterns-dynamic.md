### Обновлённый общий план с поддержкой любых паттернов (не только свечных)

#### 1. Базовый интерфейс паттерна

Вводим абстракцию, не завязанную на свечи:

- **Интерфейс `MarketPattern` (название условное):**
  - **`name: str`** – имя фичи (должно совпадать с `feature.name` в Registry).
  - **`supports(feature_def: FeatureDefinition) -> bool`**  
    Говорит, умеет ли объект реализовать данную фичу (по имени, `input_sources`, типу и т.п.).
  - **`get_requirements(feature_def) -> PatternRequirements`**, где:
    - `window: timedelta` – минимальный временной интервал, который нужен этому паттерну (может быть `0s` для чисто точечных фич типа «время суток»).
    - **`inputs: dict[str, InputRequirement]`** – требования к данным по источникам:
      - пример:  
        - `"klines"` → `{base_interval: 1m, num_segments: 3}` (для свечных паттернов);
        - `"trades"` → `{window: 5m, agg: "sum"}` (для orderflow‑паттерна);
        - `"time"` → `{needs_timestamp: True}` (для временного паттерна `is_asia_session`).
    - сюда же можно добавить флаги: нужен ли аггрегированный ряд, нужны ли сырые точки и т.п.
  - **`compute(feature_def, data: dict[str, Any]) -> float | None`**  
    Получает уже подготовленные данные по источникам (аггрегированные свечи, срез трейдов, просто `timestamp`) и считает значение фичи.

> Ключевой момент: **интерфейс не знает про свечи напрямую**, он оперирует абстрактными требованиями к `inputs`.

---

#### 2. Реестр паттернов и маппинг Registry → объекты

- Создаём **реестр `pattern_objects: list[MarketPattern]`**, в котором:
  - свечные паттерны представлены как классы, реализующие:
    - `supports` для `pattern_*` / `candle_*` с `input_sources=["kline"]`;
    - `get_requirements` → `inputs["klines"] = {base_interval=1m, num_segments=X}` (X считывается/рассчитывается динамически по `feature_def`, не хардкодим 3);
  - временные паттерны (`is_asia_session`, `is_london_session`, …):
    - `supports` по имени и `input_sources=[]`;
    - `get_requirements` → `window=0s`, `inputs={"time": {needs_timestamp: True}}`;
  - любые другие (`market_regime_*`, дневные агрегаты и т.д.) описывают свои требования аналогично.
- В `compute_all_patterns_dynamic(...)`:
  - получаем список `feature_def` из Feature Registry;
  - для каждой `feature_def` ищем объект‑паттерн (`next(p for p in pattern_objects if p.supports(feature_def))`);
  - строим список пар `(feature_def, pattern_object)`.

---

#### 3. Сбор требований ко всем данным

- Для каждой пары `(feature_def, pattern_object)` вызываем:
  - `req = pattern_object.get_requirements(feature_def)`
- Получаем набор `PatternRequirements`:
  - `window: timedelta` – сколько истории нужно;
  - `inputs: dict[str, InputRequirement]` – что и в каком виде нужно по каждому источнику.
- **Агрегируем требования по источникам**:
  - на уровне всей партии фич считаем:
    - максимальный `window` по **каждому типу данных** (klines/trades/orderbook/time и т.п.);
    - для свечных/сегментных требований:
      - определяем минимальный `base_interval` (например, 1m для klines);
      - фиксируем, что нам нужны сегменты для разных `(window, num_segments)`.

---

#### 4. Универсальный слой подготовки данных (аггрегаторы)

Делаем модуль, который опирается только на `InputRequirement`, а не на «свечи» напрямую.

- **Для kline‑источника**:
  - по максимальному `window` и `base_interval` вытаскиваем из `RollingWindows` нужный диапазон (например, 1‑минутные клайны).
  - для всех запросов вида `{window=W, num_segments=X}`:
    - **делим W на X равных по времени отрезков**;
    - по каждому отрезку агрегируем базовые клайны в сегмент:
      - `open/high/low/close/volume`, плюс любые вспомогательные поля.
    - результат: `segments[(W, X)] = list[CandleSegment]`.
- **Для trades / orderflow / funding и т.п.**:
  - по максимальному `window` достаём из `RollingWindows`/источников нужные данные;
  - для требований типа `{window=W, agg="sum" | "mean" | "pct_change"}` заранее считаем необходимые аггрегаты (по аналогии с тем, как сейчас делается VWAP, volume, volatility).
- **Для временных паттернов**:
  - источник `"time"` тривиален: передаём в паттерн нормализованный `timestamp`/`datetime`.

Важно: **этот слой оперирует типизированными требованиями, а не конкретными паттернами**. Он возвращает структуру:

```python
prepared_data: dict[str, Any] = {
    "klines": { (window, num_segments): [CandleSegment, ...], ... },
    "trades": { "window=W": aggregated_struct, ... },
    "time": timestamp,
    ...
}
```

---

#### 5. Вычисление значений фич

- Для каждой пары `(feature_def, pattern_object)`:
  - из `prepared_data` достаём всё, что требуется по её `inputs`:
    - свечные паттерны – `segments[(W, X)]`;
    - временные – `time`/`timestamp`;
    - дневные – заранее посчитанные дневные агрегаты и т.д.
  - вызываем:
    - `value = pattern_object.compute(feature_def, data_for_feature)`
  - кладём в итоговый результат:
    - `features[feature_def.name] = value`.

---

#### 6. `compute_all_patterns_dynamic` как единая точка входа

- Новая функция (можно оставить имя `compute_all_candle_patterns_dynamic`, но логичнее что‑то вроде `compute_all_market_patterns`) делает:

  1. Получает `RollingWindows`, `timestamp` (через `last_update`) и `feature_definitions`.
  2. Через реестр объектов‑паттернов строит `(feature_def, pattern_object)`.
  3. Собирает и агрегирует `PatternRequirements` по всем фичам.
  4. Вызывает универсальный слой подготовки данных (аггрегаторы).
  5. Вычисляет значения всех фич через `pattern_object.compute`.
  6. Возвращает `dict[str, value]` только по фичам, объявленным в Registry.

- Свечные функции `compute_all_candle_patterns_3m/5m/15m/45m` больше **не нужны** как ядро:
  - их логика будет инкапсулирована внутри конкретных `MarketPattern`‑классов:
    - там, где раньше был жёсткий `3m` и `3 свечи`, теперь:
      - `get_requirements` говорит: окно = `feature.lookback_window`, сегментов = `X`, где X выбирается паттерном (и может зависеть от самой фичи);
      - `compute` реализует тот же самый набор формул, но уже над абстрактными сегментами.

---

#### 7. Интеграция с существующими сервисами

- **`FeatureComputer` / `HybridFeatureComputer` / `OfflineEngine`**:
  - больше не выбирают функции по `lookback_window` (`_get_candle_pattern_function` становится очень тонким);
  - вместо «выбрать 3m/15m/45m»:
    - просто вызывают `compute_all_patterns_dynamic(rolling_windows, feature_registry.features)` (или только подмножество candle/pattern‑фич).
- Для временных фич (`is_asia_session`, `is_london_session`, `is_ny_session`) и любых других паттернов:
  - они попадают в тот же механизм, просто их `MarketPattern`‑объекты имеют другие `inputs` и логику `compute`.

---

#### 8. Тестирование

- Написать юнит‑тесты для:
  - конкретных `MarketPattern`‑классов (их `get_requirements` и `compute` на искусственных `candles`/`time`);
  - универсального аггрегатора сегментов по `(window, num_segments)` для kline‑данных;
  - сквозной `compute_all_patterns_dynamic`:
    - разный `lookback_window` (включая 37m, 67s и т.п.);
    - сочетание свечных и несвечных паттернов в одном реестре.

Такой план не завязан ни на фиксированные таймфреймы 3m/15m/45m, ни на сам факт, что паттерн обязательно свечной: «паттерн» становится просто объектом, который **сам** описывает, какие данные и в каком виде ему нужны.
