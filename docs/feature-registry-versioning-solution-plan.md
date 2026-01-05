# План решения проблемы несоответствия версий Feature Registry

## Проблема

Модели обучены на конкретных версиях Feature Registry, но Feature Service публикует фичи активной версии в очередь `features.live`. Это приводит к:
- Вычислению фичей, которые могут быть несовместимы с активными моделями
- Работе моделей на данных несоответствующей версии (с дефолтными значениями)
- Некорректным предсказаниям и сигналам

## Цель

Model Service должен запрашивать фичи с указанием версии Feature Registry, на которой была обучена модель. Feature Service должен вычислять фичи для указанной версии on-demand через HTTP API.

## Архитектурные решения

### 1. Rolling Windows: Одно общее окно на символ

**Важно:** Rolling windows - это хранилище СЫРЫХ данных рынка (trades, klines), а не вычисленных фичей.

- **Одно rolling window на символ** (не на версию Feature Registry)
- Структура окон определяется объединением интервалов из всех используемых версий
- Данные рынка (trades, klines) обновляются один раз из WebSocket streams
- Все версии Feature Registry используют **одни и те же сырые данные**

**Как это работает для фичей с одинаковым именем, но разной шириной:**
- Фичи **вычисляются** из данных в rolling windows, а не хранятся там
- Ширина окна (lookback_window) - это параметр **функции вычисления**, а не структуры данных
- Например:
  - v1.2.0: `compute_returns(rolling_windows, window_seconds=3)` → использует данные за 3 сек
  - v1.5.0: `compute_returns(rolling_windows, window_seconds=5)` → использует данные за 5 сек
  - Оба используют **одни и те же rolling windows**, но получают данные за разные периоды через `get_trades_for_window()` или `get_klines_for_window()`
- Rolling windows хранят достаточно данных (до max_lookback_minutes_1m), чтобы покрыть все версии

### 2. Feature Computer: Пул по версиям
- FeatureComputer создается/кэшируется для каждой версии Feature Registry
- Все FeatureComputer-ы используют общие rolling windows
- Каждая версия фильтрует фичи через свой `_allowed_feature_names`

### 3. Коммуникация: HTTP API вместо очереди
- Отключить публикацию в очередь `features.live`
- Model Service всегда использует HTTP API для получения фичей
- API endpoint принимает параметр `feature_registry_version`

---

## Подробное пояснение: Rolling Windows для разных версий

### Как работают Rolling Windows

Rolling windows - это **хранилище сырых данных рынка**, а не вычисленных фичей:

```python
class RollingWindows:
    windows: Dict[str, pd.DataFrame]  # Например: {"1s": DataFrame, "3s": DataFrame, "1m": DataFrame}
    # Хранит: trades и klines с полями timestamp, price, volume, side (для trades)
    # или timestamp, open, high, low, close, volume (для klines)
```

### Вычисление фичей из Rolling Windows

Фичи **вычисляются** из данных в rolling windows через функции:

```python
# Пример: compute_returns()
def compute_returns(rolling_windows: RollingWindows, window_seconds: int, current_price: float):
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(seconds=window_seconds)  # Динамически вычисляем период
    end_time = now
    
    # Получаем данные за нужный период из rolling windows
    if window_seconds >= 60:
        data = rolling_windows.get_klines_for_window("1m", start_time, end_time)
    else:
        data = rolling_windows.get_trades_for_window(f"{window_seconds}s", start_time, end_time)
    
    # Вычисляем фичу из полученных данных
    return (current_price - first_price) / first_price
```

**Ключевой момент:** Ширина окна (lookback_window) - это параметр функции вычисления, а не структуры данных!

### Пример: Одна фича, разные версии

Предположим, у нас есть фича `returns_3s` в двух версиях Feature Registry:

**v1.2.0:**
```yaml
- name: "returns_3s"
  lookback_window: "3s"
```

**v1.5.0:**
```yaml
- name: "returns_3s"  # То же имя!
  lookback_window: "5s"  # Но другая ширина!
```

**Как это работает с одним rolling window:**

1. Rolling window хранит данные за последние N секунд (например, до max_lookback_minutes_1m)
2. При вычислении фичи для v1.2.0:
   ```python
   compute_returns(rolling_windows, window_seconds=3, current_price=...)
   # → get_trades_for_window("3s", now-3s, now) → получает данные за 3 секунды
   ```
3. При вычислении фичи для v1.5.0:
   ```python
   compute_returns(rolling_windows, window_seconds=5, current_price=...)
   # → get_trades_for_window("3s", now-5s, now) → получает данные за 5 секунд из того же окна
   ```

**Один rolling window, разные периоды выборки!**

### Почему одно окно работает

1. **Данные рынка одинаковы для всех версий** - одна и та же сделка или свеча используется всеми версиями
2. **Фичи вычисляются динамически** - функция получает нужный период через параметры
3. **Rolling windows хранит достаточно данных** - max_lookback_minutes_1m покрывает все версии
4. **Эффективность** - нет дублирования данных, одна копия на символ

### Структура окон (интервалы)

Rolling windows содержит несколько "под-окон" по интервалам:
- `windows["1s"]` - данные за последние 1 секунду
- `windows["3s"]` - данные за последние 3 секунды
- `windows["15s"]` - данные за последние 15 секунд
- `windows["1m"]` - данные за последние N минут (klines)

Эти интервалы определяются **объединением** требований всех версий:
- Если v1.2.0 требует {"1s", "3s"}
- И v1.5.0 требует {"3s", "15s"}
- То rolling windows создается с {"1s", "3s", "15s", "1m"} (1m всегда есть для klines)

Каждая версия использует нужные ей интервалы из этого набора.

---

## Детальный план реализации

### Этап 1: Feature Service - Поддержка версионирования в API

#### 1.1. Модификация API endpoint `/features/latest`

**Файл:** `feature-service/src/api/features.py`

**Изменения:**
- Добавить опциональный параметр `feature_registry_version: Optional[str] = None`
- Если версия не указана → использовать активную версию (backward compatibility)
- Если версия указана → использовать FeatureComputer для этой версии

**Код:**
```python
@router.get("/latest")
async def get_latest_features(
    symbol: str = Query(..., description="Trading pair symbol (e.g., BTCUSDT)"),
    feature_registry_version: Optional[str] = Query(None, description="Feature Registry version (default: active version)"),
) -> FeatureVector:
    """
    Get latest computed features for a symbol.
    
    If feature_registry_version is provided, computes features using that version.
    Otherwise uses active Feature Registry version.
    
    Returns 404 if features are not available for the symbol.
    """
    if _feature_computer_manager is None:
        raise HTTPException(status_code=503, detail="Feature computer manager not available")
    
    try:
        # Get FeatureComputer for specified version (or active if None)
        feature_computer = await _feature_computer_manager.get_or_create_computer(
            feature_registry_version=feature_registry_version
        )
        
        feature_vector = feature_computer.compute_features(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
        )
        
        if feature_vector is None:
            raise HTTPException(
                status_code=404,
                detail=f"Features not available for symbol: {symbol}",
            )
        
        return feature_vector
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "get_latest_features_error",
            symbol=symbol,
            feature_registry_version=feature_registry_version,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error")
```

#### 1.2. Создание FeatureComputerManager

**Файл:** `feature-service/src/services/feature_computer_manager.py` (новый)

**Назначение:**
- Управление пулом FeatureComputer-ов по версиям
- Кэширование созданных FeatureComputer-ов
- Обеспечение общих rolling windows для всех версий

**Основные методы:**
- `get_or_create_computer(feature_registry_version: Optional[str]) -> FeatureComputer`
- `get_shared_rolling_windows(symbol: str) -> RollingWindows`
- `update_window_requirements()` - объединение требований всех версий

**Структура:**
```python
class FeatureComputerManager:
    def __init__(
        self,
        orderbook_manager: OrderbookManager,
        feature_registry_version_manager: FeatureRegistryVersionManager,
        shared_rolling_windows: Dict[str, RollingWindows],  # Общие для всех
    ):
        self._orderbook_manager = orderbook_manager
        self._version_manager = feature_registry_version_manager
        self._shared_rolling_windows = shared_rolling_windows
        self._computers: Dict[str, FeatureComputer] = {}  # Версия -> Computer
        self._active_version: Optional[str] = None
        self._lock = asyncio.Lock()
    
    async def get_or_create_computer(
        self, 
        feature_registry_version: Optional[str] = None
    ) -> FeatureComputer:
        """Get or create FeatureComputer for version."""
        # Если версия не указана, используем активную
        if feature_registry_version is None:
            version_record = await self._version_manager.load_active_version()
            feature_registry_version = version_record["version"]
            self._active_version = feature_registry_version
        
        # Проверяем кэш
        if feature_registry_version in self._computers:
            return self._computers[feature_registry_version]
        
        async with self._lock:
            # Double-check после получения lock
            if feature_registry_version in self._computers:
                return self._computers[feature_registry_version]
            
            # Загружаем версию Feature Registry
            config_data = await self._version_manager.load_version(feature_registry_version)
            
            # Создаем FeatureRegistryLoader для этой версии
            loader = FeatureRegistryLoader(use_db=True, version_manager=self._version_manager)
            loader._registry_model = FeatureRegistry(**config_data)
            loader._config = config_data
            
            # Создаем FeatureComputer (использует общие rolling windows через injection)
            computer = FeatureComputer(
                orderbook_manager=self._orderbook_manager,
                feature_registry_version=feature_registry_version,
                feature_registry_loader=loader,
                shared_rolling_windows=self._shared_rolling_windows,  # НОВОЕ
            )
            
            self._computers[feature_registry_version] = computer
            await self._update_shared_rolling_windows_requirements()
            
            return computer
```

#### 1.3. Модификация FeatureComputer для общих rolling windows

**Файл:** `feature-service/src/services/feature_computer.py`

**Изменения:**
- Добавить параметр `shared_rolling_windows: Optional[Dict[str, RollingWindows]] = None`
- Если `shared_rolling_windows` предоставлен → использовать его вместо `self._rolling_windows`
- Метод `get_rolling_windows()` должен работать с общими окнами

**Код:**
```python
def __init__(
    self,
    orderbook_manager: OrderbookManager,
    feature_registry_version: str = "1.0.0",
    feature_registry_loader: Optional["FeatureRegistryLoader"] = None,
    shared_rolling_windows: Optional[Dict[str, RollingWindows]] = None,  # НОВОЕ
):
    # ...
    if shared_rolling_windows is not None:
        self._rolling_windows = shared_rolling_windows  # Используем общие
    else:
        self._rolling_windows: Dict[str, RollingWindows] = {}  # Локальные (legacy)
    
    self._uses_shared_windows = shared_rolling_windows is not None

def get_rolling_windows(self, symbol: str) -> RollingWindows:
    """Get or create rolling windows for symbol."""
    if symbol in self._rolling_windows:
        return self._rolling_windows[symbol]
    
    if self._uses_shared_windows:
        # Для общих окон - создаем с объединенными требованиями
        # (это должно делать FeatureComputerManager)
        raise RuntimeError("Shared rolling windows should be initialized by manager")
    
    # Остальная логика для локальных окон (как сейчас)
    # ...
```

#### 1.4. Инициализация общих rolling windows в main.py

**Файл:** `feature-service/src/main.py`

**Изменения:**
- Создать общие rolling windows на уровне сервиса
- Передать их в FeatureComputerManager
- Обновлять структуру окон при добавлении новых версий

**Код:**
```python
# В startup()
global shared_rolling_windows, feature_computer_manager

# Создаем общие rolling windows (один экземпляр на символ)
shared_rolling_windows: Dict[str, RollingWindows] = {}

# Создаем FeatureComputerManager
feature_computer_manager = FeatureComputerManager(
    orderbook_manager=orderbook_manager,
    feature_registry_version_manager=feature_registry_version_manager,
    shared_rolling_windows=shared_rolling_windows,
)

# Передаем в API
set_feature_computer_manager(feature_computer_manager)
```

#### 1.5. Обновление MarketDataConsumer для общих rolling windows

**Файл:** `feature-service/src/consumers/market_data_consumer.py`

**Изменения:**
- MarketDataConsumer должен обновлять общие rolling windows
- Получать ссылку на shared_rolling_windows через FeatureComputerManager

**Код:**
```python
# В _process_event() или аналогичном методе
# Вместо: rolling_windows = self._feature_computer.get_rolling_windows(symbol)
# Использовать: rolling_windows = self._feature_computer_manager.get_shared_rolling_windows(symbol)
```

---

### Этап 2: Feature Service - Отключение публикации в очередь

#### 2.1. Отключение FeatureScheduler (опционально)

**Файл:** `feature-service/src/main.py`

**Решения:**
- **Вариант A:** Полностью отключить FeatureScheduler
  - Не запускать `feature_scheduler.start()`
  - Убрать из startup
  
- **Вариант B:** Оставить scheduler только для обновления rolling windows
  - Но не публиковать фичи в очередь
  - Только обновлять общие rolling windows

**Рекомендация:** Вариант B (для поддержания rolling windows актуальными)

#### 2.2. Отключение публикации в FeatureScheduler

**Файл:** `feature-service/src/services/feature_scheduler.py`

**Изменения:**
- Добавить флаг `publish_features: bool = False`
- Если `publish_features=False`, не вызывать `feature_publisher.publish()`
- Только вычислять фичи для обновления rolling windows (если нужно)

---

### Этап 3: Model Service - Поддержка версий в HTTP запросах

#### 3.1. Модификация FeatureServiceClient

**Файл:** `model-service/src/services/feature_service_client.py`

**Изменения:**
- Добавить параметр `feature_registry_version: Optional[str]` в `get_latest_features()`
- Передавать версию в query параметрах HTTP запроса

**Код:**
```python
async def get_latest_features(
    self, 
    symbol: str, 
    feature_registry_version: Optional[str] = None,
    trace_id: Optional[str] = None
) -> Optional[FeatureVector]:
    """
    Get latest computed features for a symbol from Feature Service.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        feature_registry_version: Optional Feature Registry version to request
        trace_id: Optional trace ID for request flow tracking
    
    Returns:
        FeatureVector or None if features unavailable or error
    """
    url = f"{self.base_url}/features/latest"
    headers = {
        "X-API-Key": self.api_key,
        "Content-Type": "application/json",
    }
    params = {"symbol": symbol}
    
    if feature_registry_version:
        params["feature_registry_version"] = feature_registry_version
    
    # ... остальная логика
```

#### 3.2. Модификация IntelligentSignalGenerator

**Файл:** `model-service/src/services/intelligent_signal_generator.py`

**Изменения:**
- В методе `generate_signal()` получать `model_feature_registry_version` из `training_config`
- Передавать версию в `_get_feature_vector()`
- Убрать проверку несоответствия версий (теперь версии всегда совпадают)

**Код:**
```python
async def generate_signal(...):
    # ...
    
    # Get model's training config to get feature registry version
    model_feature_registry_version = None
    training_config = None
    if active_model and active_model.get("training_config"):
        training_config = active_model["training_config"]
        if isinstance(training_config, str):
            training_config = json.loads(training_config)
        model_feature_registry_version = training_config.get("feature_registry_version")
    
    # Get feature vector with model's version
    feature_vector = await self._get_feature_vector(
        asset=asset, 
        feature_registry_version=model_feature_registry_version,  # НОВОЕ
        trace_id=trace_id
    )
    
    # Убрать проверку несоответствия версий - теперь всегда совпадают
    # if model_feature_registry_version and model_feature_registry_version != feature_vector.feature_registry_version:
    #     logger.warning(...)
    
    # ...
```

#### 3.3. Модификация _get_feature_vector

**Файл:** `model-service/src/services/intelligent_signal_generator.py`

**Изменения:**
- Добавить параметр `feature_registry_version: Optional[str]`
- Передавать версию в `feature_service_client.get_latest_features()`
- Кэш должен учитывать версию: ключ = `(symbol, version)`

**Код:**
```python
async def _get_feature_vector(
    self, 
    asset: str, 
    feature_registry_version: Optional[str] = None,
    trace_id: Optional[str] = None
) -> Optional[FeatureVector]:
    """
    Get feature vector from Feature Service via HTTP API.
    
    Args:
        asset: Trading pair symbol
        feature_registry_version: Feature Registry version to request
        trace_id: Optional trace ID for request flow tracking
    
    Returns:
        FeatureVector or None if unavailable
    """
    # Cache key теперь включает версию
    cache_key = f"{asset}:{feature_registry_version or 'active'}"
    
    # Проверяем кэш (нужно модифицировать feature_cache для поддержки версий)
    cached_feature = await feature_cache.get_by_key(cache_key, max_age_seconds=...)
    if cached_feature:
        logger.debug("Using cached feature vector", asset=asset, version=feature_registry_version, trace_id=trace_id)
        return cached_feature
    
    # Запрос через HTTP API с указанием версии
    feature_vector = await feature_service_client.get_latest_features(
        asset, 
        feature_registry_version=feature_registry_version,
        trace_id=trace_id
    )
    
    if feature_vector:
        await feature_cache.set_by_key(cache_key, feature_vector)
    
    return feature_vector
```

#### 3.4. Модификация FeatureCache для поддержки версий

**Файл:** `model-service/src/services/feature_cache.py`

**Изменения:**
- Добавить метод `get_by_key(key: str, ...)` и `set_by_key(key: str, ...)`
- Или изменить существующие методы для поддержки составных ключей

**Код:**
```python
async def get_by_key(self, key: str, max_age_seconds: Optional[int] = None) -> Optional[FeatureVector]:
    """Get cached feature vector by custom key (e.g., 'BTCUSDT:1.2.0')."""
    # ... аналогично get(), но использует key вместо symbol

async def set_by_key(self, key: str, feature_vector: FeatureVector) -> None:
    """Cache feature vector by custom key."""
    # ... аналогично set(), но использует key вместо symbol
```

#### 3.5. Отключение FeatureConsumer

**Файл:** `model-service/src/main.py`

**Изменения:**
- Не запускать `feature_consumer.start()` даже если `FEATURE_SERVICE_USE_QUEUE=true`
- Или установить `FEATURE_SERVICE_USE_QUEUE=false` по умолчанию

---

### Этап 4: Конфигурация и настройки

#### 4.1. Обновление env.example

**Файл:** `env.example`

**Изменения:**
- Установить `FEATURE_SERVICE_USE_QUEUE=false` (или удалить, оставить только HTTP API)
- Добавить комментарии о новой архитектуре

#### 4.2. Обновление документации

**Файлы:**
- `feature-service/README.md`
- `model-service/README.md`
- `docs/feature-service.md`

**Содержание:**
- Описание новой архитектуры с версионированием
- Примеры запросов с указанием версии
- Миграция с очереди на HTTP API

---

### Этап 5: Тестирование

#### 5.1. Unit тесты

- FeatureComputerManager: создание/кэширование компьютеров
- Общие rolling windows: проверка объединения требований
- API endpoint: проверка работы с версиями

#### 5.2. Integration тесты

- Запрос фичей с разными версиями
- Проверка совместимости фичей с моделями
- Проверка кэширования по версиям

#### 5.3. E2E тесты

- Полный цикл: Model Service → Feature Service → получение фичей нужной версии
- Проверка корректности вычисления фичей для разных версий

---

## Риски и митигация

### Риск 1: Производительность при множественных версиях

**Проблема:** Много версий = много FeatureComputer-ов = больше памяти

**Митигация:**
- Кэширование FeatureComputer-ов (переиспользование)
- Ленивая загрузка версий (только при запросе)
- Очистка неиспользуемых версий (LRU cache)

### Риск 2: Объединение требований rolling windows

**Проблема:** Если версии требуют очень разные интервалы, структура окон становится большой

**Митигация:**
- Объединение только используемых версий (не всех загруженных)
- Мониторинг размера rolling windows
- Оптимизация структуры данных

### Риск 3: Backward compatibility

**Проблема:** Старые клиенты без указания версии

**Митигация:**
- Если версия не указана → использовать активную версию (как сейчас)
- Постепенная миграция клиентов

### Риск 4: Откат версий Feature Registry

**Проблема:** Что если версия была удалена/откачена?

**Митигация:**
- Проверка существования версии в API
- Возврат 404/400 если версия недоступна
- Логирование ошибок

---

## Порядок внедрения

1. ✅ **Этап 1** (Feature Service - API): Реализована поддержка версий в API, создан FeatureComputerManager
2. ⏭️ **Этап 2** (Feature Service - очередь): Отключение публикации в очередь (опционально, можно оставить для совместимости) - **НЕ ВЫПОЛНЕНО** (можно реализовать позже)
3. ✅ **Этап 3** (Model Service): Добавлена поддержка версий в клиенте и генераторе сигналов
4. ⏳ **Этап 4** (Конфигурация): Обновить настройки и документацию (в процессе)
5. ⏳ **Этап 5** (Тестирование): Полное тестирование и валидация (в процессе)

## Статус реализации

### Выполнено:

✅ **Feature Service:**
- Создан `FeatureComputerManager` (`feature-service/src/services/feature_computer_manager.py`) для управления пулом FeatureComputer-ов по версиям
- Модифицирован `FeatureComputer` для поддержки общих rolling windows через параметр `shared_rolling_windows`
- Обновлен API endpoint `/features/latest` для поддержки параметра `feature_registry_version` (опциональный, обратная совместимость сохранена)
- Обновлен `main.py` для инициализации FeatureComputerManager и общих rolling windows (только в DB-режиме)
- Rolling windows создаются автоматически при первом использовании (поддержка shared mode)
- Добавлены unit-тесты для `FeatureComputerManager` (`tests/unit/test_feature_computer_manager.py`)
- Добавлен тест для API endpoint с параметром версии (`tests/contract/test_features_api.py::test_get_latest_features_with_version_parameter`)

✅ **Model Service:**
- Модифицирован `FeatureServiceClient` для передачи `feature_registry_version` в HTTP запросах
- Модифицирован `IntelligentSignalGenerator` для использования версии из `training_config` модели
- Обновлен `_get_feature_vector` для поддержки версий и передачи их в HTTP запросы
- Добавлены методы `get_by_key()` и `set_by_key()` в `FeatureCache` для поддержки составных ключей (symbol:version)
- Убрана проверка несоответствия версий (теперь версии всегда совпадают, так как запрашивается версия модели)

✅ **Тестирование:**
- Все существующие тесты проходят
- Добавлены новые тесты для FeatureComputerManager
- Добавлены тесты для API endpoint с версионированием

### Выполнено (Этап 2):

✅ **Отключение публикации в очередь:**
- Добавлен параметр `publish_features: bool = False` в `FeatureScheduler.__init__()`
- Публикация в очередь отключена по умолчанию (значение по умолчанию `False`)
- Scheduler продолжает вычислять фичи (для поддержания rolling windows актуальными через compute_features), но не публикует их в очередь
- Обратная совместимость: можно включить публикацию, передав `publish_features=True` при создании scheduler

### Требует внимания:

⚠️ **Документация:** Обновить README и другие документы с описанием новой архитектуры (опционально)

---

## Ожидаемый результат

✅ Model Service запрашивает фичи с версией Feature Registry модели  
✅ Feature Service вычисляет фичи для указанной версии on-demand  
✅ Нет несоответствия версий между моделями и фичами  
✅ Общие rolling windows эффективно используют память  
✅ Отключена публикация в очередь `features.live` (по умолчанию, можно включить через `publish_features=True`)  

