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
  - Если `confidence < min_confidence_threshold` (берётся из `MODEL_ACTIVATION_THRESHOLD`) — **сигнал отбрасывается**.
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



### Ручной инференс через HTTP‑эндпоинт

Для отладки и диагностики можно руками запустить тот же пайплайн инференса, который выполняет `IntelligentOrchestrator`, через отдельный HTTP‑эндпоинт модельного сервиса.

- **Базовый URL модельного сервиса (с хоста):**
  - `http://localhost:${MODEL_SERVICE_PORT:-4500}`
- **Эндпоинт:**
  - `POST /api/v1/inference/manual`
- **Авторизация:**
  - Обязателен заголовок `X-API-Key` c значением `MODEL_SERVICE_API_KEY` (см. `env.example` и `docker-compose.yml`).
- **Параметры запроса (query):**
  - `asset` — торговая пара, например `BTCUSDT`.
  - `strategy_id` — идентификатор стратегии, например `test-strategy`.
- **Заголовки (опционально):**
  - `X-Trace-ID` — трейс‑ID для удобной склейки логов (если не передан, сгенерируется автоматически).

#### Пример вызова с хоста

```bash
export MODEL_SERVICE_PORT=${MODEL_SERVICE_PORT:-4500}
export MODEL_SERVICE_API_KEY_VALUE="$(docker compose exec -T model-service printenv MODEL_SERVICE_API_KEY | tr -d '\r')"

curl -X POST "http://localhost:${MODEL_SERVICE_PORT}/api/v1/inference/manual?asset=BTCUSDT&strategy_id=test-strategy" \
  -H "X-API-Key: ${MODEL_SERVICE_API_KEY_VALUE}" \
  -H "X-Trace-ID: manual-debug-$(date +%s)" \
  -sS | jq .
```

#### Пример вызова изнутри контейнера `model-service`

```bash
docker compose exec -T model-service bash -lc '
  curl -X POST "http://localhost:4500/api/v1/inference/manual?asset=BTCUSDT&strategy_id=test-strategy" \
    -H "X-API-Key: ${MODEL_SERVICE_API_KEY}" \
    -H "X-Trace-ID: manual-debug-$(date +%s)" \
    -sS | jq .
'
```

#### Ожидаемое поведение и ответы

- **200 OK + `TradingSignal` в теле** — сигнал успешно сгенерирован и (если прошёл валидацию) опубликован в RabbitMQ.
- **200 OK + `null` в теле** — пайплайн отработал, но сигнал отфильтрован (низкий `confidence`, HOLD‑решение модели, риск‑фильтры и т.п.). Детали причины см. в логах `model-service`.
- **401 Unauthorized** — не передан `X-API-Key` или он неверный.
- **422 Unprocessable Entity** — невалидные или отсутствующие параметры `asset` / `strategy_id`.

При ручном инференсе используется тот же пайплайн, что описан выше (fetch фич из Feature Service, подготовка фич, предсказание модели, фильтры и риск‑менеджмент), отличие только в том, что запуск идёт по HTTP‑запросу, а не по внутреннему расписанию.