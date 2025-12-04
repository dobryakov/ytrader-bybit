# Техническое задание: Feature Service

## 1. Цель системы

Feature Service — выделенный сервис для:

1. приёма и хранения сырых маркет-данных, поступающих из ws-gateway,
2. расчёта онлайн-фичей для model-service,
3. пересборки оффлайн-фичей и датасетов для обучения моделей,
4. обеспечения идентичности фичей в online и offline режимах.

Сервис отделяет вычисление признаков от логики модели, позиций и ордеров.

---

## Clarifications

### Session 2025-01-27

- Q: Message queue naming convention for market data from ws-gateway → A: Use `ws-gateway.*` prefix to match existing pattern (e.g., `ws-gateway.orderbook`, `ws-gateway.trades`)
- Q: Storage technology for raw Parquet data files → A: Local filesystem in container (mounted volumes)
- Q: API authentication method for REST endpoints → A: API Key authentication (API key in header or query parameter)
- Q: Minimum data retention period for raw data → A: 90 days (3 months) before archiving/deletion
- Q: Position Manager data source priority (REST API vs events) → A: Events as primary, REST API as fallback (events for real-time updates, REST for initialization/recovery)

---

## 2. Границы и обязанности

Feature Service обязан:

- Принимать сырые потоки Bybit (через существующие очереди/шину событий от ws-gateway).
- Поддерживать хранение raw data для последующей пересборки истории.
- Реализовать Online Feature Engine: вычисление признаков в realtime.
- Реализовать Offline Feature Engine + Dataset Builder: пересоздание фичей на исторических данных.
- Публиковать фичи в очередь для model-service.
- Предоставлять API для сборки датасета обучения (пакетно, по временным интервалам).
- Обеспечить консистентность фичей (один и тот же код расчёта и параметры).

---

## 3. Получаемые данные от Bybit (через ws-gateway)

Сервис не подключается к Bybit сам; он подписывается на внутренние потоки:

Через внутренние очереди RabbitMQ (согласовано с ws-gateway):

- `ws-gateway.orderbook` — события стакана (snapshot и delta)
- `ws-gateway.trades` — события сделок
- `ws-gateway.kline` — свечные данные (1s, 1m и другие интервалы)
- `ws-gateway.ticker` — данные тикера
- `ws-gateway.funding` — данные funding rate
- (опционально) `ws-gateway.order` или отдельная очередь для `trading.executions` — события исполнения собственных ордеров

### Требования к приёму

- Поддерживать sequence/order в orderbook:
  - корректная сборка snapshot + delta,
  - обработка рассинхронизации (рест запрос snapshot),
  - хранение всех дельт для оффлайн восстановления.

- Все сообщения снабжать внутренним timestamp + exchange timestamp.

---

## 4. Хранилище сырых данных

Требования:

- Формат: Parquet/Columnar (оптимально для оффлайн пересборки).
- Хранилище: Локальная файловая система в контейнере (mounted volumes через docker-compose).

Структуры (каталоги/файлы Parquet):

- `raw_orderbook_snapshots/` — файлы Parquet со snapshot стакана
- `raw_orderbook_deltas/` — файлы Parquet с дельтами стакана
- `raw_trades/` — файлы Parquet со сделками
- `raw_kline/` — файлы Parquet со свечными данными
- `raw_ticker/` — файлы Parquet с тикерами
- `raw_funding/` — файлы Parquet с funding rate
- `raw_executions/` — файлы Parquet с событиями исполнения ордеров

Хранить минимум 90 дней (3 месяца) — параметр конфигурируемый, по умолчанию 90 дней; поддерживать архивирование (перемещение старых файлов в архивный каталог или удаление после истечения срока хранения).

---

## 5. Feature Engine (online)

Онлайн-движок считает фичи каждые 1s, 3s, 15s, 1m.
Фичи из расчёта подаются в model-service.

### Базовые свечные/цено-фичи

- mid_price = (best_bid + best_ask)/2
- spread_abs, spread_rel
- return_1s, return_3s, return_1m
- VWAP_3s, VWAP_15s, VWAP_1m
- vol_3s, vol_15s, vol_1m
- volatility_1m, volatility_5m (std of returns)

### Orderflow (трейды)

- signed_volume_3s, 15s, 1m
- buy/sell volume ratio
- trade_count_3s
- net_aggressor_pressure = buy_vol − sell_vol

### Orderbook (top 5 / top 10)

- depth_bid_top5 / depth_ask_top5
- depth_imbalance_top5
- depth_bid_top10 / depth_ask_top10
- slope_bid, slope_ask (линейный коэффициент убывания объёмов с уровнями)
- orderbook_churn_rate (частота изменений best price)
- local_liquidity_density (объём в пределах X bps)

### Перпетуальные параметры

- funding_rate
- time_to_funding (минуты до следующего расчёта)

### Временные/мета признак

- time_of_day (числовой)
- rolling z-score (стандартизация последних N значений)

### Формат online выдачи

[timestamp, symbol, feature_vector(N)]

Через внутреннюю очередь features.live и REST /features/latest?symbol=....

---

## 6. Offline Feature Engine + Dataset Builder

Этот же движок должен уметь работать в batch-режиме:

### Функции

- Восстановить orderbook по snapshot + deltas за период.
- Восстановить все rolling windows (1s–1m).
- Посчитать фичи идентично online-версии.

### Генерация targets (целевых переменных)

**Regression targets:**

- `return_(t+N)`: будущая доходность за горизонт N
  - Формула: `return_(t+N) = (price_(t+N) - price_t) / price_t`
  - `price_t`: mid_price в момент t (время фичи)
  - `price_(t+N)`: mid_price в момент t+N (через горизонт предсказания)
  - Поддерживаемые горизонты: 1m, 5m, 15m, 1h (конфигурируемо)

**Classification targets:**

- `direction_(t+N)`: направление движения цены (up/down/flat)
  - `up`: если `return_(t+N) > threshold` (по умолчанию 0.001 = 0.1%)
  - `down`: если `return_(t+N) < -threshold`
  - `flat`: если `|return_(t+N)| <= threshold`
  - Порог `threshold` конфигурируемый (по умолчанию 0.001)

**Risk-adjusted targets (опционально):**

- `sharpe_return_(t+N)`: доходность, скорректированная на волатильность
  - Формула: `return_(t+N) / volatility_(t-N_window, t)`
  - Используется волатильность за окно до момента t (не будущая!)

**Важно:** Все targets вычисляются строго на основе данных после момента t, но не используют информацию, недоступную в момент t (no lookahead bias).

### Разделение датасета на train/validation/test

Dataset Builder должен поддерживать явное разделение на периоды:

**Time-based split (рекомендуется для временных рядов):**

- Train period: исторические данные для обучения
- Validation period: данные для валидации и подбора гиперпараметров
- Test period: финальная оценка на данных, которые модель никогда не видела

**Walk-forward validation (для временных рядов):**

- Автоматическая генерация последовательных фолдов
- Конфигурация:
  - `train_window_days`: размер окна обучения (например, 90 дней)
  - `validation_window_days`: размер окна валидации (например, 30 дней)
  - `step_days`: шаг между фолдами (например, 7 дней)
- Каждый фолд: train на [T0, T1), validation на [T1, T2)
- Последний фолд может использоваться как test set

**Random split (не рекомендуется для временных рядов, но поддерживается):**

- Случайное разделение с сохранением временного порядка внутри каждого сета
- Используется только для тестирования, не для production

Собрать датасет в parquet batch-файлы с метаданными о разделении.
Экспортировать по API /dataset/build с явным указанием периодов.

### Гарантия идентичности

- Один код расчёта для offline и online режимов.
- Один и тот же Feature Registry.

---

## 7. Feature Registry

YAML/JSON, описывающий:

- имя фичи
- входные источники (trades/orderbook/kline)
- окно (3s, 15s, 1m…)
- параметры нормализации
- порядок расчёта

### Контроль data leakage

Каждая фича в Feature Registry должна явно указывать:

- `lookback_window`: временное окно в прошлое (например, "3s", "1m", "5m")
  - Фича использует только данные из интервала [t - lookback_window, t]
- `lookahead_forbidden: true`: явный запрет на использование будущих данных
- `max_lookback_days`: максимальное окно в днях (для валидации)
- `data_sources`: список источников данных с временными метками

### Валидация при сборке датасета

- Проверка, что все фичи используют только данные до момента t
- Проверка, что target вычисляется строго на данных после t
- Логирование предупреждений при обнаружении потенциального leakage
- Автоматический отказ в сборке датасета при критических нарушениях

### Пример структуры Feature Registry

```yaml
features:
  - name: "mid_price"
    sources: ["orderbook"]
    lookback_window: "0s"  # текущий момент
    lookahead_forbidden: true
    calculation: "(best_bid + best_ask) / 2"
  
  - name: "return_1m"
    sources: ["kline"]
    lookback_window: "1m"
    lookahead_forbidden: true
    calculation: "(price_t - price_t_1m) / price_t_1m"
  
  - name: "volatility_5m"
    sources: ["kline"]
    lookback_window: "5m"
    lookahead_forbidden: true
    calculation: "std(returns[t-5m:t])"
```

Нужен для детерминированного пересборки датасета и предотвращения data leakage.

---

## 8. Интеграция с Model Service

Model-service должен быть переработан:

Убрать любую логику feature engineering из model-service.

model-service должен:

- принимать готовый feature vector;
- выполнять inference;
- выполнять обучение только на датасетах, предоставленных Feature Service;
- поддерживать модельные артефакты (weights, metadata);
- иметь API /model/train → принимает путь к датасету (или id датасета).

---

## 9. API

Public (внутренний) API Feature Service:

### Аутентификация

Все endpoints (кроме `/health`) требуют аутентификацию через API Key:

- Header: `X-API-Key: <api_key>`
- Query parameter: `?api_key=<api_key>` (альтернативный вариант)

### Online Features

1. `GET /features/latest?symbol=BTCUSDT` — последние онлайн-фичи для символа.
   - Возвращает: `{timestamp, symbol, features: {...}}`
   - Латентность: ≤ 50-70 мс

### Dataset Building

1. `POST /dataset/build` — собрать датасет за период с явным разделением на train/val/test.

   **Request body (time-based split):**

   ```json
   {
     "name": "training_dataset_v1",
     "symbol": "BTCUSDT",
     "periods": {
       "train": {
         "from": "2024-01-01T00:00:00Z",
         "to": "2024-06-30T23:59:59Z"
       },
       "validation": {
         "from": "2024-07-01T00:00:00Z",
         "to": "2024-08-31T23:59:59Z"
       },
       "test": {
         "from": "2024-09-01T00:00:00Z",
         "to": "2024-09-30T23:59:59Z"
       }
     },
     "validation_strategy": "time_based",
     "target_config": {
       "type": "return",
       "horizon": "1m",
       "threshold": 0.001
     },
     "feature_registry_version": "v1.2",
     "data_leakage_check": true,
     "output_format": "parquet"
   }
   ```

   **Request body (walk-forward validation):**

   ```json
   {
     "name": "walk_forward_dataset_v1",
     "symbol": "BTCUSDT",
     "validation_strategy": "walk_forward",
     "walk_forward_config": {
       "train_window_days": 90,
       "validation_window_days": 30,
       "step_days": 7,
       "start_date": "2024-01-01T00:00:00Z",
       "end_date": "2024-12-31T23:59:59Z",
       "min_train_samples": 1000
     },
     "target_config": {
       "type": "return",
       "horizon": "1m"
     },
     "feature_registry_version": "v1.2",
     "data_leakage_check": true
   }
   ```

   **Response:**

   ```json
   {
     "dataset_id": "ds_20241201_001",
     "status": "building",
     "estimated_completion": "2024-12-01T12:00:00Z",
     "splits": {
       "train": {"records": 50000, "period": "2024-01-01 to 2024-06-30"},
       "validation": {"records": 15000, "period": "2024-07-01 to 2024-08-31"},
       "test": {"records": 5000, "period": "2024-09-01 to 2024-09-30"}
     }
   }
   ```

2. `GET /dataset/list` — список доступных датасетов.
   - Query params: `?status=ready&symbol=BTCUSDT`
   - Возвращает список с метаданными о разделении

3. `GET /dataset/{dataset_id}` — получить метаданные датасета.
   - Возвращает: информацию о периодах, количестве записей, feature registry version, target config

4. `GET /dataset/{dataset_id}/download?split=train` — скачать датасет или его часть.
   - Query params: `split` = `train`, `validation`, `test`, или `all`
   - Возвращает: Parquet файл или ссылку на S3/объектное хранилище

### Model Evaluation

1. `POST /model/evaluate` — подготовить датасет для оценки модели на отдельном периоде.

   ```json
   {
     "dataset_id": "ds_20241201_001",
     "split": "test",
     "model_version": "v123",
     "metrics_to_compute": ["accuracy", "sharpe_ratio", "max_drawdown"]
   }
   ```

   - Возвращает готовый датасет для оценки с метаданными о метриках

### Feature Registry

1. `GET /feature-registry` — получить текущую версию Feature Registry.
2. `POST /feature-registry/reload` — перезагрузка конфигурации фичей.
3. `GET /feature-registry/validate` — валидация Feature Registry на data leakage.

### Data Quality

1. `GET /data-quality/report?symbol=BTCUSDT&from=...&to=...` — отчёт о качестве данных.
   - Возвращает: missing rate, anomaly detection, sequence gaps, рассинхронизации

### Streaming

- Очередь `features.live` — пуш онлайн-фичей в realtime.
- Очередь `features.dataset.ready` — уведомление о завершении сборки датасета.
  - Сообщение: `{dataset_id, status, splits_info, download_urls}`

---

## 10. Нефункциональные требования

- Онлайн латентность расчёта фичей: ≤ 50–70 мс.
- Возможность горизонтального масштабирования по символам.
- Идентичность онлайн/оффлайн фичей: тесты деривации.
- Стойкость к падению ws-gateway: восстановление из snapshot.
- Логи: сырые данные, ошибки seq, пропуски, рассинхронизации.

---

## 11. Data Leakage Prevention

Feature Service должен явно предотвращать data leakage (утечку будущей информации в фичи):

### Требования к Feature Registry

- Каждая фича должна явно указывать `lookback_window` — окно в прошлое
- Запрет на использование данных после момента t в фичах
- Валидация при загрузке Feature Registry: проверка временных границ

### Валидация при сборке датасета (в разделе Data Leakage Prevention)

- Автоматическая проверка всех фичей на использование только данных до момента t
- Проверка, что target вычисляется строго на данных после t
- Логирование предупреждений при обнаружении потенциального leakage
- Автоматический отказ в сборке датасета при критических нарушениях

### Правила валидации

1. Фича с `lookback_window="3s"` может использовать только данные из [t-3s, t]
2. Target `return_(t+1m)` вычисляется на основе `price_(t+1m)` — данных после t
3. Запрещено использовать фичи, которые зависят от будущих значений (например, `future_volatility`)
4. Rolling statistics (например, `rolling_z_score`) должны использовать только прошлые значения

### Логирование и мониторинг

- Логировать все проверки data leakage при сборке датасета
- Метрики: количество предупреждений, количество отказов в сборке
- Алерты при обнаружении критических нарушений

### Тестирование

- Автоматические тесты на data leakage для всех фичей в Feature Registry
- Тесты на корректность временных границ при сборке датасета
- Интеграционные тесты: сравнение online и offline фичей на идентичность

---

## 12. Model Evaluation Support

Feature Service должен поддерживать корректную оценку моделей на out-of-sample данных:

### Разделение датасета

- Явное разделение на train/validation/test периоды
- Гарантия, что test dataset не использовался при обучении
- Метаданные о разделении сохраняются вместе с датасетом

### Walk-forward validation

- Автоматическая генерация последовательных фолдов для временных рядов
- Каждый фолд: train на [T0, T1), validation на [T1, T2)
- Последний фолд может использоваться как test set
- Поддержка конфигурируемых параметров (размер окна, шаг)

### Метаданные о разделении

- Экспорт метаданных о train/val/test split вместе с датасетом
- Информация о периодах, количестве записей, feature registry version
- Гарантия реплицируемости: один и тот же запрос даёт идентичный split

### API для оценки

- `POST /model/evaluate` — подготовить датасет для оценки на отдельном периоде
- Поддержка вычисления метрик качества данных (accuracy, Sharpe ratio, max drawdown)
- Экспорт результатов оценки в структурированном формате

### Гарантии

- Test dataset никогда не используется для обучения или валидации
- Validation dataset используется только для подбора гиперпараметров
- Все метрики качества модели должны считаться на validation/test, а не на training set

---

## 13. Дополнительные требования к качеству данных

### Валидация входящих данных

- Проверка на пропуски (missing values)
- Обнаружение аномалий (outliers, резкие скачки цен)
- Проверка последовательности (sequence gaps в orderbook)
- Валидация временных меток (внутренний timestamp vs exchange timestamp)

### Мониторинг качества

- Метрики: missing rate, anomaly rate, sequence gap rate
- Автоматические алерты при превышении порогов
- API `/data-quality/report` для получения отчётов

### Обработка проблемных данных

- Стратегии обработки пропусков: интерполяция, forward fill, пропуск записи
- Обработка рассинхронизации orderbook: запрос snapshot, пересборка
- Логирование всех проблемных случаев для анализа

---

## 14. Интеграция с Model Service

### Изменения в Model Service (требования к переработке)

Model-service должен быть переработан:

Убрать любую логику feature engineering из model-service.

model-service должен:

- Принимать готовый feature vector от Feature Service (через очередь `features.live` или REST API)
- Выполнять inference на готовых фичах
- Выполнять обучение только на датасетах, предоставленных Feature Service
- Поддерживать модельные артефакты (weights, metadata)
- Иметь API `/model/train` → принимает `dataset_id` или путь к датасету

### Workflow обучения

1. Model Service запрашивает сборку датасета: `POST /dataset/build` с указанием периодов train/val/test
2. Feature Service собирает датасет с валидацией data leakage
3. Feature Service публикует уведомление в очередь `features.dataset.ready`
4. Model Service получает уведомление и скачивает датасет
5. Model Service обучает модель на train set, валидирует на validation set
6. Model Service оценивает финальное качество на test set (out-of-sample)
7. Model Service активирует модель только если метрики на test set превышают порог

### Workflow inference

1. Model Service подписывается на очередь `features.live` или опрашивает REST API `/features/latest`
2. Model Service получает готовый feature vector
3. Model Service выполняет inference на модели
4. Model Service генерирует торговый сигнал на основе предсказания

---

## 15. Position Features и их роль в Feature Service

### Обязанности Feature Service по Position Features

**Feature Service ДОЛЖЕН:**

1. **Сбор данных о позициях**
   - Подписываться на очередь `position-manager.position_updated` для получения обновлений позиций в реальном времени
   - Или опрашивать Position Manager REST API `GET /api/v1/positions/{asset}` для получения текущего состояния позиций
   - Кэшировать данные о позициях для быстрого доступа при вычислении признаков
   - Обеспечивать актуальность данных о позициях (TTL кэша ≤ 30 секунд)

2. **Вычисление position-признаков**
   - `position_size`: текущий размер позиции (положительный для long, отрицательный для short)
   - `position_size_abs`: абсолютное значение размера позиции
   - `unrealized_pnl`: нереализованная прибыль/убыток в абсолютных единицах
   - `realized_pnl`: реализованная прибыль/убыток
   - `has_position`: бинарный признак (1 если есть позиция, 0 если нет)
   - `entry_price`: средняя цена входа в позицию
   - `price_vs_entry`: процентное отклонение текущей цены от цены входа
     - Формула: `(current_price - entry_price) / entry_price * 100`
   - `total_exposure`: общая экспозиция по активу (включая открытые ордера)
   - `total_exposure_abs`: абсолютное значение общей экспозиции

3. **Интеграция position-признаков в feature vector**
   - Включать position-признаки в каждый feature vector для online inference
   - Включать position-признаки в исторические датасеты для обучения
   - Обеспечивать идентичность position-признаков в online и offline режимах
   - Использовать исторические данные о позициях для пересборки датасетов

4. **Обработка отсутствующих данных о позициях**
   - Если позиция отсутствует: устанавливать все position-признаки в 0 (кроме `entry_price`, который равен текущей цене)
   - Если данные о позиции недоступны: использовать значения по умолчанию и логировать предупреждение
   - Обеспечивать консистентность: одинаковые значения по умолчанию для online и offline

5. **Валидация position-признаков**
   - Проверять, что `entry_price > 0` если `has_position == 1`
   - Проверять, что `position_size != 0` если `has_position == 1`
   - Проверять, что `price_vs_entry` вычислен корректно относительно `entry_price`
   - Логировать предупреждения при обнаружении несоответствий

**Feature Service НЕ ДОЛЖЕН:**

- Принимать торговые решения (это ответственность Model Service и IntelligentSignalGenerator)
- Применять бизнес-правила (лимиты позиций, take profit, stop loss)
- Генерировать торговые сигналы
- Управлять позициями или ордерами

### Источники данных для Position Features

**Online режим (realtime):**

- **Position Manager Events (основной источник)**: очередь `position-manager.position_updated` для получения обновлений в реальном времени
  - Подписка на события при старте сервиса
  - Обновление кэша при каждом событии
  - Обеспечивает минимальную латентность обновления position-признаков
- **Position Manager REST API (резервный источник)**: `GET /api/v1/positions/{asset}` используется для:
  - Инициализации при старте сервиса (получение текущего состояния)
  - Восстановления после пропуска событий (если обнаружен gap в событиях)
  - Периодической синхронизации (опционально, для проверки консистентности)
- **Кэширование**: in-memory кэш с TTL ≤ 30 секунд для оптимизации запросов и обеспечения актуальности данных

**Offline режим (пересборка датасетов):**

- **Исторические данные о позициях**: хранить в базе данных или Parquet файлах
- **Восстановление состояния позиций**: реконструировать состояние позиций на каждый момент времени из исторических данных
- **Синхронизация с execution events**: использовать `execution_events` для восстановления истории позиций

### Интеграция Position Features в Feature Registry

**Пример конфигурации position-признаков в Feature Registry:**

```yaml
features:
  - name: "position_size"
    sources: ["position_manager"]
    lookback_window: "0s"  # текущее состояние
    lookahead_forbidden: true
    calculation: "position.size"
    default_value: 0.0
    
  - name: "unrealized_pnl"
    sources: ["position_manager"]
    lookback_window: "0s"
    lookahead_forbidden: true
    calculation: "position.unrealized_pnl"
    default_value: 0.0
    
  - name: "price_vs_entry"
    sources: ["position_manager", "market_data"]
    lookback_window: "0s"
    lookahead_forbidden: true
    calculation: "(current_price - entry_price) / entry_price * 100"
    default_value: 0.0
    requires: ["entry_price", "current_price"]
```

### Гарантии идентичности Position Features

**Требования к Feature Service:**

1. **Один код расчёта**: одинаковый алгоритм вычисления position-признаков для online и offline режимов
2. **Одинаковые источники данных**: использование одних и тех же источников (Position Manager) для online и исторических данных для offline
3. **Одинаковые значения по умолчанию**: если позиция отсутствует, использовать одинаковые значения по умолчанию в обоих режимах
4. **Валидация**: автоматические тесты на идентичность online и offline position-признаков

### Взаимодействие с Model Service

**Feature Service предоставляет:**

- Готовые position-признаки в feature vector для inference
- Исторические position-признаки в датасетах для обучения
- Гарантию, что position-признаки вычислены корректно и без data leakage

**Model Service использует:**

- Position-признаки как входные данные для модели (модель учится на них)
- Модель может предсказывать на основе position-признаков (например, "hold" если уже есть большая позиция)
- Model Service НЕ вычисляет position-признаки самостоятельно

**IntelligentSignalGenerator использует:**

- Position-признаки из feature vector для бизнес-логики (проверка лимитов, take profit)
- Дополнительные запросы к Position Manager для бизнес-правил (если нужно более свежее состояние)
- Модель уже учла position-признаки в своём предсказании
