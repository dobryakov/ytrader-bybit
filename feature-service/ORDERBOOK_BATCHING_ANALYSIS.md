# Анализ решения: Батчинг событий orderbook

## Текущая ситуация

**Поток событий:**
- ~64-67 событий в секунду (4,000-4,200 в минуту)
- Orderbook события составляют ~87% потока (~56 событий/сек)
- Обрабатываются немедленно: каждое delta применяется сразу через `OrderbookManager.apply_delta()`

**Использование orderbook:**
- Features вычисляются с интервалами: 1s, 3s, 15s, 1m (через FeatureScheduler)
- Orderbook используется для:
  - Получения `mid_price` (используется в price features)
  - Вычисления orderbook features (depth_bid_top5, depth_ask_top5, depth_imbalance_top5)
  - Все эти features требуют актуального состояния orderbook

## Предлагаемое решение: Батчинг orderbook deltas

### Концепция

**Накопление deltas в буфере:**
- Orderbook **snapshots** обрабатываются немедленно (критично для синхронизации)
- Orderbook **deltas** накапливаются в буфере по символам
- Батчи применяются каждую минуту (периодическая задача)

### Архитектурные варианты

#### Вариант 1: Батчинг в OrderbookManager (рекомендуется)

**Изменения:**
1. Добавить в `OrderbookManager`:
   - `_delta_buffer: Dict[str, List[Dict]]` - буфер для накопления deltas по символам
   - `_last_batch_apply_time: Dict[str, datetime]` - время последнего применения батча
   - `apply_delta_buffered(delta: Dict) -> None` - добавляет delta в буфер вместо немедленного применения
   - `apply_delta_batch(symbol: str) -> None` - применяет накопленные deltas для символа
   - `_batch_apply_task: Optional[asyncio.Task]` - периодическая задача для применения батчей

2. Периодическая задача (каждую минуту):
   ```python
   async def _apply_batches_periodically(self):
       while self._running:
           await asyncio.sleep(60)  # 1 минута
           for symbol in list(self._delta_buffer.keys()):
               if self._delta_buffer[symbol]:
                   await self.apply_delta_batch(symbol)
   ```

**Преимущества:**
- Минимальные изменения в существующем коде
- OrderbookManager сам управляет своим состоянием
- Легко включить/выключить батчинг через флаг

**Недостатки:**
- Orderbook будет обновляться раз в минуту (может быть устаревшим для features с интервалом 1s/3s/15s)

#### Вариант 2: Гибридный подход (применение перед compute_features)

**Изменения:**
1. Накапливать deltas в буфере (как в варианте 1)
2. Но применять батч перед вычислением features:
   ```python
   def compute_features(self, symbol: str, ...):
       # Применить накопленные deltas перед вычислением
       if self._orderbook_manager.has_pending_deltas(symbol):
           self._orderbook_manager.apply_delta_batch(symbol)
       # ... остальной код
   ```

**Преимущества:**
- Orderbook всегда актуален для feature computation
- Снижает нагрузку от частых применений deltas
- Все еще батчинг (меньше применений)

**Недостатки:**
- Более сложная логика (нужно проверять перед каждым compute_features)
- Все еще могут быть частые применения если features вычисляются часто

#### Вариант 3: Батчинг только для storage, немедленное применение для computation

**Изменения:**
1. Orderbook deltas применяются немедленно (как сейчас) для feature computation
2. Для storage (Parquet) события накапливаются и сохраняются батчами

**Преимущества:**
- Orderbook всегда актуален
- Снижает I/O нагрузку для storage
- Не влияет на feature computation

**Недостатки:**
- Не решает проблему CPU нагрузки от обработки orderbook events

## Рекомендация: Вариант 2 (Гибридный)

### Обоснование

1. **Features вычисляются часто** (1s, 3s, 15s, 1m интервалы) - нужен актуальный orderbook
2. **Батчинг снижает нагрузку** - вместо 56 применений/сек будет ~1 применение/сек (перед compute_features)
3. **Snapshots применяются немедленно** - гарантирует синхронизацию

### Реализация

**В OrderbookManager:**

```python
class OrderbookManager:
    def __init__(self):
        # ... существующий код ...
        self._delta_buffer: Dict[str, List[Dict]] = defaultdict(list)
        self._buffer_lock: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._batch_interval_seconds = 60  # Конфигурируемо
    
    async def apply_delta_buffered(self, delta: Dict) -> None:
        """Добавить delta в буфер вместо немедленного применения."""
        symbol = delta["symbol"]
        async with self._buffer_lock[symbol]:
            self._delta_buffer[symbol].append(delta)
    
    async def apply_delta_batch(self, symbol: str) -> int:
        """Применить все накопленные deltas для символа. Возвращает количество примененных."""
        async with self._buffer_lock[symbol]:
            if not self._delta_buffer[symbol]:
                return 0
            
            deltas = self._delta_buffer[symbol]
            self._delta_buffer[symbol] = []
            
            applied_count = 0
            for delta in deltas:
                if self.apply_delta(delta):  # Используем существующий метод
                    applied_count += 1
                else:
                    # Если desynchronized, нужно запросить snapshot
                    # Остальные deltas в батче будут невалидными
                    logger.warning(
                        "orderbook_batch_application_failed",
                        symbol=symbol,
                        applied_count=applied_count,
                        total_count=len(deltas),
                        failed_sequence=delta.get("sequence"),
                    )
                    break
            
            return applied_count
    
    def has_pending_deltas(self, symbol: str) -> bool:
        """Проверить, есть ли накопленные deltas."""
        return bool(self._delta_buffer.get(symbol))
```

**В FeatureComputer.update_market_data:**

```python
# Вместо немедленного применения delta:
if orderbook_type == "delta":
    # Добавить в буфер вместо немедленного применения
    await self._orderbook_manager.apply_delta_buffered(delta_data)
```

**В FeatureComputer.compute_features:**

```python
def compute_features(self, symbol: str, ...):
    # Применить накопленные deltas перед вычислением
    if self._orderbook_manager.has_pending_deltas(symbol):
        await self._orderbook_manager.apply_delta_batch(symbol)
    
    orderbook = self._orderbook_manager.get_orderbook(symbol)
    # ... остальной код
```

**Периодическая задача (опционально, для очистки старых батчей):**

```python
async def _periodic_batch_apply(self):
    """Применять батчи периодически, если features не вычисляются."""
    while self._running:
        await asyncio.sleep(60)  # Каждую минуту
        for symbol in list(self._delta_buffer.keys()):
            if self._delta_buffer[symbol]:
                # Применить только если накопилось много (защита от переполнения)
                if len(self._delta_buffer[symbol]) > 1000:
                    await self.apply_delta_batch(symbol)
```

## Ожидаемый эффект

**Снижение нагрузки:**
- Было: ~56 применений deltas в секунду × 2 символа = ~112 операций/сек
- Станет: ~1-2 применения в секунду (перед compute_features)
- **Снижение в ~50-100 раз**

**Актуальность данных:**
- Orderbook всегда актуален перед вычислением features
- Snapshots применяются немедленно
- Deltas накапливаются максимум до следующего compute_features

## Риски и ограничения

1. **Память**: Буфер deltas может расти, если features не вычисляются
   - **Решение**: Ограничить размер буфера (например, 5000 deltas), при превышении применять принудительно

2. **Sequence gaps**: Если накопилось много deltas и есть gap, весь батч может быть невалидным
   - **Решение**: Проверять sequence gaps и запрашивать snapshot при необходимости

3. **Latency**: Небольшая задержка обновления orderbook (максимум до следующего compute_features, обычно < 1 секунды)

## Альтернативные варианты батчинга

### Вариант A: Батчинг по времени (строго каждую минуту)
- Применять батчи строго каждую минуту
- **Минус**: Orderbook может быть устаревшим для features с короткими интервалами

### Вариант B: Батчинг по размеру (например, каждые 100 deltas)
- Применять когда накопилось N deltas
- **Минус**: Нет гарантии актуальности для features

### Вариант C: Комбинированный (время + размер + перед compute_features)
- Применять если: прошла 1 минута ИЛИ накопилось 1000 deltas ИЛИ вызывается compute_features
- **Плюс**: Баланс между актуальностью и производительностью

## Рекомендация по реализации

**Начать с Варианта 2 (гибридный)**, так как:
1. Гарантирует актуальность orderbook для features
2. Значительно снижает нагрузку
3. Минимальные риски
4. Легко добавить периодическую задачу для защиты от переполнения

**Дополнительные улучшения:**
- Добавить метрики: размер буфера, частота применений, latency
- Добавить конфигурацию: размер буфера, интервал принудительного применения
- Мониторинг: алерты если буфер растет слишком быстро

