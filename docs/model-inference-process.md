При `INTELLIGENT_SIGNAL_FREQUENCY=0.1` оркестратор делает один проход примерно **раз в 10 минут** и на каждом проходе выполняет такой пайплайн:

- **1. Решение “стоит ли вообще стрелять”:**
  - Проверяет **rate limit** (`SIGNAL_GENERATION_RATE_LIMIT` / `BURST_ALLOWANCE`).
  - Тянет `OrderPositionState` из позиционного сервиса.
  - Сначала пробует **take-profit правило**: если `unrealized_pnl_pct > MODEL_SERVICE_TAKE_PROFIT_PCT`, сразу генерирует `SELL`‑сигнал на закрытие позиции, без модели.
  - Если включён `SIGNAL_GENERATION_SKIP_IF_OPEN_ORDER` — может **вообще пропустить цикл**, если уже есть подходящие открытые ордера.

- **2. Модельный инференс:**
  - Берёт **последний `FeatureVector`** по инструменту из feature‑кэша (заполняется из RabbitMQ `features.live`), иначе идёт в Feature Service по REST.
  - `ModelLoader` подтягивает **активную модель** для стратегии (из БД + файлов).
  - `ModelInference.prepare_features()`:
    - Берёт фичи из Feature Service.
    - Добавляет **позиционные фичи** (размер, PnL, открытые ордера, exposure, hash’и).
    - Делает **align** под ожидания модели (добавляет недостающие колонки с дефолтами, удаляет лишнее, порядок как при трейне).
  - `ModelInference.predict()`:
    - Если модель **classification**:
      - Считает `predict_proba`.
      - Применяет **калиброванные пороги** из `training_config.probability_thresholds` (если есть): может выдать `+1` (buy), `-1` (sell) или `0` (hold).
      - Либо использует глобальную калибрацию (`MODEL_PREDICTION_USE_THRESHOLD_CALIBRATION`), либо обычный `argmax`.
      - Считает `confidence = max(probabilities)` и отдельные `buy_probability` / `sell_probability`.
    - Если модель **regression**:
      - Считает `predicted_return`.
      - Конвертирует в buy/sell/hold по `MODEL_REGRESSION_THRESHOLD`.
      - `confidence` = нормализованный модуль доходности относительно `MODEL_REGRESSION_MAX_EXPECTED_RETURN`.

- **3. Фильтры и риск‑менеджмент вокруг предсказания:**
  - Если `confidence < min_confidence_threshold` (берётся из `MODEL_QUALITY_THRESHOLD_ACCURACY`) — **сигнал отбрасывается**.
  - `_determine_signal_type()`:
    - Для классификации: сравнивает `buy_probability` vs `sell_probability` → `buy` или `sell`.
    - Для регрессии: преобразует `predicted_return` в `buy` / `sell` / `None (HOLD)`.
  - Если `signal_type is None` (HOLD для регрессии) — **ничего не делает**.
  - Считает **базовый объём** через `_calculate_amount()`:
    - Для регрессии — учитывает и `confidence`, и величину `predicted_return`.
    - Для классификации — масштабирует между `WARMUP_MIN_AMOUNT` и `WARMUP_MAX_AMOUNT` по `confidence`, + уменьшает, если уже есть позиция.
  - Прогоняет через `balance_calculator.calculate_affordable_amount()` → режет до доступного баланса.
  - Для `buy` дополнительно проверяет **лимит размера позиции** (`ORDERMANAGER_MAX_POSITION_SIZE`), может **уменьшить объём** или полностью отменить сигнал.

- **4. Итоговое действие:**
  - Если после всех фильтров и адаптаций сигнал всё ещё валиден:
    - Строит `TradingSignal` (`signal_type` = `buy`/`sell`, `amount`, `confidence`, `model_version`, снэпшот рынка из фич).
    - Прогоняет через `signal_validator`.
    - Публикует в RabbitMQ (`model-service.trading_signals`) через `signal_publisher`.
  - Дальше **order-manager** читает этот сигнал и, согласно своей логике, создаёт/модифицирует ордера на бирже (обычно лимит/маркет buy или sell на `amount`).

