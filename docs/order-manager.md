# Микросервис управления ордерами (Order Manager, Order Service)

## Общая концепция

- Микросервис ордеров (order manager) управляет ордерами bybit и поддерживает актуальную информацию об их состоянии.

## Что делает микросервис

- Хранит ордера и поддерживает их актуальное состояние.
- Получает от микросервиса модели (через очередь) высокоуровневые сигналы о торговых действиях, например, «купи BTC на 1000 USD».
- Принимает решения о создании, изменении или отмене конкретных ордеров с учётом актуального состояния баланса и текущих открытых ордеров, хранящихся в общей базе PostgreSQL.  
- Выставляет, модифицирует и отменяет ордера через REST API Bybit, обрабатывает результаты.  
- Поддерживает подписку на топики об исполнении ордеров от bybit через ws-gateway.
- Логирует операции, обновляет статус ордеров в БД и публикует события про ордера (по возможности обогащая информацию) в другую очередь для остальных микросервисов (в том числе для микросервиса модели).  
- Предоставляет устойчивый и безопасный механизм исполнения сигналов модели, обеспечивая защиту от некорректных или рискаошибочных действий.
- Имеет инструменты для принудительной актуализации текущего состояния ордеров напрямую через REST API Bybit.

## Почему это важно

- Разграничивает стратегию (модель) и исполнение, повышая надёжность и контролируемость.  
- Оптимизирует работу с биржевым API, минимизирует ошибки.  
- Обеспечивает консистентность и актуальность данных ордеров.

## Пример входного сигнала

Сигналы приходят из очереди RabbitMQ `model-service.trading_signals` в формате JSON:

```json
{
  "signal_id": "550e8400-e29b-41d4-a716-446655440000",
  "signal_type": "sell",
  "asset": "ETHUSDT",
  "amount": 500.0,
  "confidence": 0.90,
  "timestamp": "2025-11-25T21:35:00Z",
  "strategy_id": "strat-002",
  "model_version": null,
  "is_warmup": true,
  "market_data_snapshot": {
    "price": 3000.0,
    "spread": 1.5,
    "volume_24h": 1000000.0,
    "volatility": 0.02,
    "orderbook_depth": {
      "bid_depth": 100.0,
      "ask_depth": 120.0
    },
    "technical_indicators": null
  },
  "metadata": {
    "reasoning": "Warm-up signal: sell based on heuristics",
    "risk_score": 0.3,
    "randomness_level": 0.5
  },
  "trace_id": "trace-12345"
}
```

**Поля сигнала:**

- `signal_id` (string, UUID) - уникальный идентификатор сигнала
- `signal_type` (string) - тип сигнала: "buy" или "sell"
- `asset` (string) - торговый инструмент (например, "BTCUSDT", "ETHUSDT")
- `amount` (float) - сумма в quote currency (USDT), должна быть > 0
- `confidence` (float) - уверенность модели (0.0-1.0)
- `timestamp` (string, ISO 8601) - время генерации сигнала
- `strategy_id` (string) - идентификатор торговой стратегии
- `model_version` (string|null) - версия модели, использованной для генерации (null для warm-up режима)
- `is_warmup` (boolean) - флаг, указывающий на warm-up режим
- `market_data_snapshot` (object) - снапшот рыночных данных на момент генерации сигнала:
  - `price` (float) - текущая цена
  - `spread` (float) - спред bid-ask
  - `volume_24h` (float) - объем торгов за 24 часа
  - `volatility` (float) - волатильность
  - `orderbook_depth` (object|null) - глубина стакана (bid_depth, ask_depth)
  - `technical_indicators` (object|null) - технические индикаторы (RSI, MACD и т.д.)
- `metadata` (object|null) - дополнительные метаданные (reasoning, risk_score и т.д.)
- `trace_id` (string|null) - идентификатор для трейсинга запросов

## Технологический стек

- Рассмотри возможность написать этот микросервис как Ruby on Rails API-only application.
- Если данному микросервису нужен cron, исполни его внутри контейнера или в отдельном контейнере, но не относи на хост-машину.

## Форматы данных и конвенции

### Формат поля `side` (сторона ордера)

**ВАЖНО**: Поле `side` имеет разные форматы в зависимости от контекста. Это критично для предотвращения ошибок API и нарушений ограничений базы данных.

#### Формат для Bybit API
- **Формат**: `"Buy"` или `"Sell"` (только первая буква заглавная)
- **Используется в**: Всех запросах к Bybit API (`/v5/order/create` и т.д.)
- **Примеры**:
  ```python
  params = {"side": "Buy"}   # ✅ Правильно для Bybit API
  params = {"side": "Sell"}  # ✅ Правильно для Bybit API
  params = {"side": "SELL"}  # ❌ Неправильно - вызовет ошибку "Side invalid (code: 10001)"
  ```

#### Формат для базы данных
- **Формат**: `"Buy"` или `"SELL"` (все заглавные для SELL)
- **Используется в**: Всех операциях вставки/обновления в таблице `orders`
- **Ограничение БД**: `CHECK (side IN ('Buy', 'SELL'))`
- **Примеры**:
  ```python
  side_db = "Buy"   # ✅ Правильно для базы данных
  side_db = "SELL"  # ✅ Правильно для базы данных
  side_db = "Sell"  # ❌ Неправильно - нарушает constraint "chk_side"
  ```

#### Паттерн реализации

При создании ордеров используйте разные переменные для API и базы данных:

```python
# Side для Bybit API: "Buy" или "Sell" (только первая буква заглавная)
side_api = "Buy" if signal.signal_type.lower() == "buy" else "Sell"

# Side для базы данных: "Buy" или "SELL" (все заглавные для SELL по constraint)
side_db = "Buy" if signal.signal_type.lower() == "buy" else "SELL"

# Используйте side_api для запросов к Bybit API
params = {"side": side_api, ...}

# Используйте side_db при сохранении в базу данных
query = "INSERT INTO orders (side, ...) VALUES ($1, ...)"
await pool.execute(query, side_db, ...)
```

**Почему это важно**:
- Bybit API ожидает `"Sell"` (первая буква заглавная), а не `"SELL"` (все заглавные)
- Ограничение базы данных требует `"SELL"` (все заглавные), а не `"Sell"` (первая буква заглавная)
- Использование неправильного формата вызывает:
  - Bybit API: ошибку `"Side invalid (code: 10001)"`
  - База данных: нарушение `CHECK constraint "chk_side"`

**Места в коде, где это используется**:
- `order-manager/src/services/order_executor.py`: метод `_prepare_bybit_order_params()`
- `order-manager/src/services/order_executor.py`: метод `_save_order_to_database()`
- `order-manager/src/services/order_executor.py`: метод `_save_rejected_order()`
- `order-manager/src/services/signal_processor.py`: метод `_save_rejected_order()`

**См. также**: Спецификация в `specs/004-order-manager/data-model.md`, раздел "Data Format Conventions"

## Уточнения перед постановкой ордера на биржу bybit

1. Обязательные параметры для Bybit API /v5/order/create (category=spot):
   - symbol (верхний регистр, например BTCUSDT)
   - side: "Buy" 
   - orderType: "Market" или "Limit"
   - qty (string): для Market buy — количество базовой монеты (BTC) или quoteCoin (USDT) через marketUnit="quoteCoin"

2. Полная предпроверка спецификации пары через GET /v5/market/instruments-info?category=spot&symbol=BTCUSDT:
   - minOrderValue (например минимум 5 USDT для API trading)
   - maxOrderQty (например максимум 71.73 BTC для BTCUSDT)
   - qtyStep и priceFilter для точности округления
   - priceLimitRatioX/Y для market order (отклонение от текущей цены)
   - order quota MVL (например 300,000 USDT) - проверка доступных buy/sell квот

instruments-info — это спецификация торговых пар (instrument specification), критически важный эндпоинт для валидации ордеров перед размещением.​ Назначение: возвращает все торговые ограничения и параметры для конкретной пары (symbol) и категории (spot/linear/option), которые обязательны для проверки перед созданием ордера. Перед постановкой ордера нужно получить спецификацию пары из instruments-info, затем для buy проверить минимальную сумму до максимального количества по цене ask, а для sell — по цене bid с учетом шага и лимитов цены. Ответ API instruments-info нужно сохранять в БД так же, как сохраняется баланс.

Почему instruments-info обязателен перед ордером:
- Валидация qty/price — без точного знания qtyStep ордер отклонят;
- Динамические лимиты — minOrderValue, maxOrderQty меняются по парам;
- Market slippage protection — priceLimitRatioX/Y предотвращает плохие исполнения;
- API ошибки — orderQtyExceedMaxLimit, invalid_quantity_precision без него неизбежны.

3. Проверка баланса аккаунта через GET /v5/account/wallet-balance?accountType=UNIFIED&coin=USDT:
   - walletBalance (USDT) - locked - spotBorrow >= qty * currentPrice (достаточно USDT для покупки)
   - totalAvailableBalance в USD эквиваленте для оценки общей ликвидности
   - Для margin trading (isLeverage=1): totalInitialMargin, totalMaintenanceMargin
   - Ошибка при insufficient_balance (недостаточно USDT)

4. Проверка рыночной ликвидности:
   - Текущая цена и 24h volume из /v5/market/tickers?category=spot
   - Глубина orderbook (/v5/market/orderbook) для оценки slippage на ask-стороне (продавцы)
   - Market order в пределах priceLimitRatio от mark/best цены

5. Поддержка дополнительных параметров:
   - price (обязательно для Limit)
   - timeInForce: "GTC", "IOC", "FOK", "PostOnly"
   - takeProfit, stopLoss, tpLimitPrice, slLimitPrice, tpOrderType, slOrderType
   - marketUnit: "baseCoin" (BTC) или "quoteCoin" (USDT) для Market buy
   - isLeverage=1 для margin trading (требует включенной маржи)

6. Соблюдение Rate Limits:
   - /v5/order/create: 600 req/5s (IP), 50 req/s (UID)
   - Batch orders: 10/s (IP), 20/s (UID)
   - instruments-info, wallet-balance: в пределах общих лимитов

7. Обработка ошибок Bybit:
   - insufficient_balance, orderQtyExceedMaxLimit, priceNotInRange
   - order quota exceeded, invalid qty/price precision
   - Логирование с причинами отказа и рекомендациями

8. Валидация входных сигналов от Model Service:
   - qty в пределах разумных лимитов с запасом (учитывая slippage)
   - symbol существует и активен
   - confidence score выше порога для размещения

9. Структурированное логирование всех проверок:
   - Результаты баланс/ликвидность/валидация (USDT → BTC расчет)
   - Параметры отправленного ордера
   - Ответ Bybit с orderId или ошибкой
   - Trace ID propagation

10. Graceful degradation:
    - При недостатке USDT: логировать, не размещать
    - При превышении лимитов: скорректировать qty/price или пропустить
    - Retry logic для временных ошибок (rate limit, network)

## Закрытие позиции перед встречным сигналом

Order Manager поддерживает автоматическое закрытие позиции перед обработкой встречного торгового сигнала. Это обеспечивает корректное управление позициями и предотвращает конфликты между противоположными сигналами.

### Как это работает

Когда приходит торговый сигнал, противоположный текущей открытой позиции:
- **Long позиция + SELL сигнал** → позиция закрывается, затем создается новый ордер
- **Short позиция + BUY сигнал** → позиция закрывается, затем создается новый ордер

### Процесс закрытия позиции

1. **Определение встречного сигнала**: Система проверяет, является ли сигнал противоположным текущей позиции
2. **Создание ордера закрытия**: Создается Market ордер с флагом `reduceOnly=True` и количеством, равным размеру позиции
3. **Ожидание закрытия** (опционально): В зависимости от режима ожидания система может дождаться подтверждения закрытия позиции
4. **Создание нового ордера**: После закрытия (или сразу, если режим ожидания отключен) создается новый ордер по сигналу

### Настройки

#### ORDERMANAGER_CLOSE_POSITION_BEFORE_OPPOSITE_SIGNAL
- **Тип**: `boolean`
- **По умолчанию**: `true`
- **Описание**: Включить автоматическое закрытие позиции перед обработкой встречного сигнала

#### ORDERMANAGER_POSITION_CLOSE_WAIT_MODE
- **Тип**: `string`
- **По умолчанию**: `none`
- **Допустимые значения**:
  - `none` - создать ордер закрытия и сразу продолжить с новым ордером (рекомендуется)
  - `polling` - ожидать закрытия позиции, проверяя статус через Position Manager API
  - `websocket` - ожидать закрытия через WebSocket события (не реализовано)

#### ORDERMANAGER_POSITION_CLOSE_TIMEOUT_SECONDS
- **Тип**: `integer`
- **По умолчанию**: `30`
- **Описание**: Таймаут ожидания закрытия позиции в секундах (используется только при `ORDERMANAGER_POSITION_CLOSE_WAIT_MODE=polling`)

#### ORDERMANAGER_POSITION_CLOSE_MIN_SIZE_THRESHOLD
- **Тип**: `float`
- **По умолчанию**: `0.00000001`
- **Описание**: Минимальный размер позиции для закрытия. Позиции меньше этого порога игнорируются

### Примеры использования

#### Режим "none" (по умолчанию)
```env
ORDERMANAGER_CLOSE_POSITION_BEFORE_OPPOSITE_SIGNAL=true
ORDERMANAGER_POSITION_CLOSE_WAIT_MODE=none
```

В этом режиме:
- Создается ордер закрытия позиции
- Сразу после создания ордера закрытия создается новый ордер по сигналу
- Bybit обработает ордера последовательно

#### Режим "polling"
```env
ORDERMANAGER_CLOSE_POSITION_BEFORE_OPPOSITE_SIGNAL=true
ORDERMANAGER_POSITION_CLOSE_WAIT_MODE=polling
ORDERMANAGER_POSITION_CLOSE_TIMEOUT_SECONDS=30
```

В этом режиме:
- Создается ордер закрытия позиции
- Система ожидает закрытия позиции, проверяя статус каждую секунду
- После закрытия (или по истечении таймаута) создается новый ордер

### Обработка ошибок

- **Позиция уже закрыта**: Если позиция закрылась между проверкой и созданием ордера, система пропускает этап закрытия
- **Ошибка создания ордера закрытия**: Логируется ошибка, но обработка сигнала продолжается (новый ордер будет создан с `reduceOnly=True`)
- **Таймаут ожидания**: Если позиция не закрылась в течение таймаута, логируется предупреждение и создается новый ордер

### Edge cases

- **Позиция меньше порога**: Позиции меньше `ORDERMANAGER_POSITION_CLOSE_MIN_SIZE_THRESHOLD` игнорируются
- **Частичное закрытие**: Если ордер закрытия исполнился частично, система продолжит работу с оставшейся позицией
- **Множественные сигналы**: FIFO очередь по ассету гарантирует последовательную обработку сигналов

## Защита от устаревших данных Position Manager

Order Manager включает механизмы защиты от использования устаревших данных позиций, когда Position Manager отстает от реального состояния (например, при сбоях WebSocket потоков или очередей).

### Проверка актуальности данных

Перед использованием данных позиции для закрытия система проверяет их актуальность:

1. **Проверка возраста данных**: Сравнивается `position.last_updated` с текущим временем
2. **Порог устаревания**: Если данные старше `ORDERMANAGER_POSITION_MAX_AGE_SECONDS`, они считаются устаревшими
3. **Автоматическое обновление**: При обнаружении устаревших данных система запрашивает свежую информацию напрямую от Bybit API

### Fallback к Bybit API

Когда данные Position Manager устарели или недоступны, система использует прямой запрос к Bybit API:

- **Метод**: `GET /v5/position/list` с параметрами `category=linear` и `symbol={asset}`
- **Преимущества**: Получение актуального состояния позиции напрямую от биржи
- **Ограничения**: Дополнительный API запрос, но обеспечивает надежность

### Мониторинг расхождений

Система автоматически отслеживает и логирует расхождения между данными Position Manager и реальным состоянием на Bybit:

- **Логирование устаревших данных**: `position_data_stale_refreshing_from_bybit` — данные устарели, выполняется обновление
- **Логирование расхождений размеров**: `position_size_discrepancy_detected` — обнаружено значительное расхождение (> 0.0001)
- **Логирование при ошибках**: `order_creation_position_discrepancy` — расхождение обнаружено при обработке ошибки 110017

### Настройки

#### ORDERMANAGER_POSITION_MAX_AGE_SECONDS
- **Тип**: `integer`
- **По умолчанию**: `60`
- **Описание**: Максимальный возраст данных позиции в секундах. Если `position.last_updated` старше этого значения, данные считаются устаревшими и система запросит свежие данные от Bybit

#### ORDERMANAGER_ENABLE_BYBIT_POSITION_FALLBACK
- **Тип**: `boolean`
- **По умолчанию**: `true`
- **Описание**: Включить fallback к прямому запросу Bybit API для получения данных позиции, когда данные Position Manager устарели или недоступны

### Обработка ошибки 110017

При ошибке `110017` ("current position is zero, cannot fix reduce-only order qty") система:

1. **Запрашивает позицию от Bybit**: Получает актуальное состояние напрямую от биржи
2. **Логирует расхождение**: Если позиция существует на Bybit, но ошибка возникла, логируется предупреждение
3. **Retry без reduceOnly**: Удаляет флаг `reduceOnly` и повторяет создание ордера
4. **Автоматическая синхронизация**: Если включена настройка `ORDERMANAGER_AUTO_SYNC_POSITION_AFTER_BYBIT_FETCH`, асинхронно запускается синхронизация Position Manager

Это обеспечивает корректную обработку ситуаций, когда Position Manager отстает от реального состояния.

### Автоматическая синхронизация после получения данных от Bybit

После получения свежих данных позиции напрямую от Bybit (при устаревших данных или ошибке 110017), система может автоматически синхронизировать Position Manager:

1. **Fire-and-forget вызов**: Асинхронный вызов `POST /api/v1/positions/sync-bybit?force=true&asset={asset}` без ожидания ответа
2. **Фильтрация по ассету**: Синхронизируется только конкретный ассет, для которого были получены данные
3. **Обновление БД**: Position Manager обновляет свою базу данных актуальными данными от Bybit
4. **Не блокирует операцию**: Основная операция продолжается независимо от результата синхронизации

#### Настройка

#### ORDERMANAGER_AUTO_SYNC_POSITION_AFTER_BYBIT_FETCH
- **Тип**: `boolean`
- **По умолчанию**: `true`
- **Описание**: Автоматически запускать синхронизацию Position Manager (fire-and-forget) после получения позиции напрямую от Bybit. Это обновляет базу данных Position Manager актуальными данными и исправляет расхождения.

#### Преимущества

- **Исправление расхождений**: Автоматически обновляет БД Position Manager при обнаружении устаревших данных
- **Не блокирует операции**: Fire-and-forget подход не замедляет основную логику
- **Целевая синхронизация**: Синхронизирует только нужный ассет, а не все позиции
- **Устойчивость к ошибкам**: Ошибки синхронизации не влияют на основную операцию

### Примеры логов

**Устаревшие данные:**
```
position_data_stale_refreshing_from_bybit: asset=BTCUSDT, age_seconds=120, max_age=60
bybit_position_fetch_success: asset=BTCUSDT, size=1.5
```

**Обнаружено расхождение:**
```
position_size_discrepancy_detected: 
  position_manager_size=1.0, 
  bybit_size=1.5, 
  size_diff=0.5, 
  size_diff_pct=50.0
```

**Ошибка 110017 с проверкой:**
```
order_creation_reduce_only_position_zero: ret_code=110017
bybit_position_fetch_success: asset=BTCUSDT, size=0.0
order_creation_position_confirmed_closed: Bybit confirms position is closed
```
