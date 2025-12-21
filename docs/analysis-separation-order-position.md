# Анализ разделения ответственности: order-manager и position-manager

## Дата анализа
2025-12-21

## Текущая архитектура

### order-manager
**Подписки**:
- `ws-gateway.order` - события исполнения ордеров от Bybit

**Ответственность**:
1. Обновление таблицы `orders` (статус, filled_quantity, average_price, fees)
2. Обновление позиций через `PositionManager.update_position()` (старый класс в order-manager)
3. Создание записей в `position_orders` (связь ордеров с позициями)
4. Публикация событий в `order-manager.order_events`

**Код**:
- `order-manager/src/services/event_subscriber.py` - обработка order событий
- `order-manager/src/services/position_manager.py` - обновление позиций и создание position_orders

### position-manager
**Подписки**:
- `ws-gateway.position` - события позиций от Bybit
- `order-manager.order_events` - события исполнения ордеров от order-manager

**Ответственность**:
1. Обновление позиций из WebSocket событий (`update_position_from_websocket()`)
2. Обновление позиций из order событий (`update_position_from_order_fill()`)
3. Управление портфолио и метриками

**Код**:
- `position-manager/src/consumers/websocket_position_consumer.py` - обработка position событий
- `position-manager/src/consumers/order_position_consumer.py` - обработка order_events

## Анализ влияния на корректность работы

### ✅ Положительные аспекты

#### 1. Четкое разделение ответственности
- **order-manager** отвечает только за ордера
- **position-manager** отвечает только за позиции
- Нет дублирования логики обновления позиций
- Проще тестировать и поддерживать

#### 2. Упрощение архитектуры
- Уменьшается зависимость position-manager от order-manager (position-manager больше не обновляет позиции из order событий, а только создает связи `position_orders`)
- Меньше точек отказа
- Проще масштабировать сервисы независимо

#### 3. Источник истины для позиций
- position-manager получает данные напрямую от Bybit через WebSocket
- Не зависит от вычислений order-manager
- Использует `cumRealisedPnl` (кумулятивное значение для всей позиции) и `avgPrice` напрямую от биржи
- **Важно**: `positions.realized_pnl` (свойство позиции) обновляется напрямую из position событий, это НЕ метрика качества модели

### ⚠️ Проблемы и риски

#### 1. КРИТИЧНО: Кто создает position_orders?

**Текущая ситуация**:
- `position_orders` создается в `order-manager/src/services/position_manager.py` при обновлении позиции
- Связывает ордера с позициями через `relationship_type` (opened, increased, decreased, closed, reversed)

**Проблема при разделении**:
- order-manager больше не обновляет позиции → не создает `position_orders`
- position-manager не знает об ордерах → не может создать `position_orders`

**Влияние на prediction_trading_results**:
- `model-service` использует `position_orders` для получения `execution_price` и `relationship_type`
- Без `position_orders` невозможно определить:
  - Какой ордер закрыл позицию
  - Цену исполнения для вычисления PnL
  - Тип связи (opened, closed, etc.)

**✅ Решение (принято)**:
- **position-manager создает `position_orders`** на основе событий из `order-manager.order_events`
- order-manager публикует события в `order-manager.order_events` с полной информацией (order_id, signal_id, execution_price, etc.)
- position-manager подписывается на `order-manager.order_events` **только для создания `position_orders`** (не для обновления позиций)
- При получении order события position-manager:
  1. Находит или создает позицию в БД (по `asset`, `mode`)
  2. Определяет `relationship_type` на основе текущего состояния позиции и ордера:
     - Если позиция не существует в БД (создается впервые) → `relationship_type = "opened"` (для любого ордера: buy открывает long позицию, sell открывает short позицию)
     - Если позиция существует и пустая (`size == 0` или `size is None`) → `relationship_type = "opened"`
     - Если позиция существует и увеличивается в том же направлении (buy при long или sell при short) → `relationship_type = "increased"`
     - Если позиция существует и уменьшается (sell при long или buy при short):
       - Если размер уменьшается, но позиция не закрывается → `relationship_type = "decreased"`
       - Если позиция полностью закрывается → `relationship_type = "closed"`
       - Если позиция закрывается и меняет направление → `relationship_type = "reversed"`
  3. Создает запись в `position_orders` с полной информацией
  
  **Тип связи между позициями и ордерами**:
  - На уровне БД: **many-to-many** (таблица `position_orders` с уникальным ограничением `UNIQUE(position_id, order_id)`)
  - На уровне бизнес-логики: **1-to-many** (одна позиция → много ордеров)
  - Один ордер влияет на одну позицию (для данного `asset` и `mode`)
  - Уникальное ограничение предотвращает дублирование связей
- Детали реализации см. раздел "Рекомендуемое решение: Гибридный подход"

#### 2. КРИТИЧНО: Как position-manager узнает об исполнении ордеров?

**Текущая ситуация**:
- position-manager получает события из `order-manager.order_events`
- Обновляет позиции через `update_position_from_order_fill()`
- Вычисляет `realized_pnl_delta` на основе `execution_price` (для обновления `positions.realized_pnl` - свойства позиции)

**Проблема при разделении**:
- position-manager не получает события об исполнении ордеров
- Не может обновить позицию при исполнении ордера
- Позиция обновляется только из WebSocket position событий

**Влияние**:
- Задержка в обновлении позиций (зависит от частоты position событий от Bybit)
- Возможны расхождения между размером позиции из order событий и position событий
- `positions.realized_pnl` (свойство позиции) будет обновляться только из position событий (кумулятивное значение от биржи)

**✅ Решение (принято)**:
- **position-manager обновляет позиции только из position событий** (`ws-gateway.position`)
- position события - источник истины для позиций (размер, PnL, avgPrice от биржи)
- position-manager **НЕ обновляет позиции** из order событий (удалить `update_position_from_order_fill()`)
- Обновление позиций происходит напрямую от биржи через WebSocket, что обеспечивает:
  - Использование `cumRealisedPnl` (кумулятивное значение для всей позиции, включает комиссии) и `avgPrice` напрямую от биржи
  - `positions.realized_pnl` обновляется напрямую из position событий (это свойство позиции, не метрика модели)
  - Независимость от вычислений order-manager
  - Корректные данные о позициях
- Детали реализации см. раздел "Рекомендуемое решение: Гибридный подход"

#### 3. ВАЖНО: Синхронизация данных

**Текущая ситуация**:
- order-manager обновляет позиции на основе order событий
- position-manager обновляет позиции на основе position событий
- Есть механизм разрешения конфликтов (optimistic locking)

**Проблема при разделении**:
- position-manager обновляет позиции только из position событий
- Нет синхронизации с ордерами
- Возможны расхождения:
  - Размер позиции из position событий vs размер из ордеров
  - `average_entry_price` из position событий vs вычисленное из ордеров

**✅ Решение (принято)**:
- **position события - источник истины** для позиций (размер, PnL, avgPrice)
- position-manager обновляет позиции только из position событий
- `positions.realized_pnl` (свойство позиции) обновляется напрямую из position событий (`cumRealisedPnl` от биржи - кумулятивное значение, включает комиссии)
- `position_orders` создаются на основе order событий, но не влияют на обновление позиций
- Записи в БД могут создаваться в произвольном порядке - это нормальное поведение асинхронной системы
- Итоговое состояние всегда корректно, так как position события содержат актуальные данные от биржи
- Детали реализации см. раздел "Рекомендуемое решение: Гибридный подход" → "Обработка событий в разном порядке"

#### 4. ВАЖНО: prediction_trading_results

**Текущая ситуация**:
- `model-service` получает `OrderExecutionEvent` из `order-manager.order_events`
- Использует `position_orders` для получения `execution_price` и `relationship_type`
- Вычисляет `prediction_trading_results.realized_pnl` (метрика качества модели) на основе этих данных

**Проблема при разделении**:
- Без `position_orders` невозможно получить `execution_price` и `relationship_type`
- Без событий от order-manager невозможно получить детальную информацию об исполнении

**✅ Решение (принято) - Гибридный подход**:
- **model-service подписывается на оба источника**:
  1. `order-manager.order_events` → для `prediction_trading_results` (быстро, точно)
     - События приходят сразу после исполнения ордера (низкая задержка)
     - Использует `position_orders` для получения `execution_price` и `relationship_type`
     - Вычисляет `prediction_trading_results.realized_pnl` (метрика качества модели) на основе данных из `position_orders`
     - **Важно**: это НЕ `positions.realized_pnl` (свойство позиции), а метрика для конкретного предсказания
  2. `position-manager.position_updated` → для обновления состояния позиций (синхронизировано)
     - Получает синхронизированные данные о позициях
     - Использует `positions.realized_pnl` (свойство позиции) для генерации сигналов и принятия решений
- Это дает лучшее из обоих миров: быстрая обработка ордеров для prediction_trading_results и синхронизированные данные о позициях
- Детали реализации см. раздел "Рекомендуемое решение: Гибридный подход" → "Обработка событий в model-service"

## Оценка корректности работы

### Сценарий 1: Исполнение ордера

**Текущая архитектура**:
1. Bybit → ws-gateway.order → order-manager
2. order-manager обновляет `orders` и `positions`
3. order-manager создает `position_orders`
4. order-manager публикует в `order-manager.order_events`
5. position-manager получает событие → обновляет позицию (если нужно)
6. model-service получает событие → обновляет `prediction_trading_results`

**Предложенная архитектура (рекомендуемое решение)**:
1. Bybit → ws-gateway.order → order-manager
2. order-manager обновляет только `orders`
3. order-manager публикует в `order-manager.order_events` с полной информацией
4. Bybit → ws-gateway.position → position-manager
5. position-manager обновляет позицию из position события
6. ✅ **position-manager создает `position_orders`** на основе событий из `order-manager.order_events` (см. раздел "Рекомендуемое решение")
7. ✅ **model-service получает детальную информацию** из двух источников:
   - `order-manager.order_events` → для `prediction_trading_results` (быстро, точно)
   - `position-manager.position_updated` → для обновления состояния позиций (синхронизировано)

**Решение проблем**:
- ✅ Связь между ордерами и позициями (`position_orders`) создается position-manager при получении order событий
- ✅ Детальная информация об исполнении для `prediction_trading_results` получается из `order-manager.order_events` и `position_orders`
- ℹ️ **Особенность работы**: `position_orders` создаются после получения order события, а позиция обновляется из position события. Это нормальное поведение асинхронной системы - записи в БД могут создаваться в произвольном порядке, что не влияет на корректность работы (см. раздел "Обработка событий в разном порядке")

### Сценарий 2: Закрытие позиции

**Текущая архитектура**:
1. Исполняется SELL ордер → order-manager обновляет позицию → `relationship_type = "closed"`
2. `position_orders` содержит `execution_price` закрывающего ордера
3. model-service использует эти данные для вычисления `exit_price` и `prediction_trading_results.realized_pnl` (метрика качества модели)

**Предложенная архитектура (рекомендуемое решение)**:
1. Исполняется SELL ордер → order-manager обновляет только `orders`
2. order-manager публикует событие в `order-manager.order_events`
3. position-manager получает order событие → создает `position_orders` с `relationship_type = "closed"` и `execution_price`
4. position событие приходит позже → position-manager обновляет позицию из position события
5. ✅ **Связь закрытия позиции с ордером** определяется через `position_orders`, созданный position-manager
6. ✅ **`execution_price` закрывающего ордера** получается из `position_orders.execution_price` (см. раздел "Рекомендуемое решение")

**Решение проблем**:
- ✅ Связь закрытия позиции с конкретным ордером определяется через `position_orders` (создается position-manager)
- ✅ `execution_price` для вычисления `exit_price` в `prediction_trading_results` получается из `position_orders` через `order-manager.order_events`

## Рекомендуемое решение: Гибридный подход

#### Архитектура

**order-manager**:
- Подписывается на `ws-gateway.order`
- Обновляет только таблицу `orders` (статус, filled_quantity, average_price, fees)
- **НЕ обновляет позиции** (удалить вызов `PositionManager.update_position()`)
- **НЕ создает position_orders** (удалить создание `position_orders`)
- Публикует в `order-manager.order_events` с полной информацией:
  - `order_id` (bybit_order_id)
  - `signal_id`
  - `asset`
  - `side` (buy/sell)
  - `filled_quantity`
  - `execution_price` (average_price из order события)
  - `execution_fees`
  - `executed_at`
  - `status` (filled, partially_filled, etc.)

**position-manager**:
- Подписывается на `ws-gateway.position` (основной источник для обновления позиций)
- Подписывается на `order-manager.order_events` (только для создания `position_orders`)
- Обновляет позиции **только** из position событий (`update_position_from_websocket()`)
- **НЕ обновляет позиции** из order событий (удалить `update_position_from_order_fill()` из обработчика order_events)
- Создает `position_orders` на основе order событий:
  - Получает событие из `order-manager.order_events`
  - **Создание позиции из order события** (если позиция не существует в БД):
    - Если позиции нет в БД при получении order события, создается позиция с минимальными данными:
      - `size = size_delta` (вычисляется из `filled_quantity` и `side` ордера)
      - `average_entry_price = execution_price` (из order события)
      - `realized_pnl = NULL` (будет обновлено при получении position события)
      - `unrealized_pnl = NULL` (будет обновлено при получении position события)
    - Позиция будет обновлена актуальными данными при получении position события от Bybit
  - Находит или создает позицию в БД (по `asset`, `mode`)
  - Определяет `relationship_type` на основе текущего состояния позиции и ордера:
    - Если позиция не существует в БД (создается впервые) → `relationship_type = "opened"` (для любого ордера: buy открывает long позицию, sell открывает short позицию)
    - Если позиция существует и пустая (`size == 0` или `size is None`) → `relationship_type = "opened"`
    - Если позиция существует и увеличивается в том же направлении:
      - buy ордер при long позиции (`size > 0`) → `relationship_type = "increased"`
      - sell ордер при short позиции (`size < 0`) → `relationship_type = "increased"`
    - Если позиция существует и уменьшается в противоположном направлении (sell при long или buy при short):
      - Вычислить новый размер позиции: `new_size = current_size + size_delta` (где `size_delta` = `+filled_quantity` для buy, `-filled_quantity` для sell)
      - Если `new_size == 0` → `relationship_type = "closed"` (позиция полностью закрыта)
      - Если `new_size` имеет противоположный знак относительно `current_size` (например, `current_size > 0` и `new_size < 0`) → `relationship_type = "reversed"` (позиция закрыта и изменила направление)
      - Если `new_size` имеет тот же знак, но меньше по модулю → `relationship_type = "decreased"` (частичное закрытие позиции)
  - Создает запись в `position_orders` с:
    - `position_id`
    - `order_id` (из таблицы `orders` по `bybit_order_id`, **может быть NULL** если ордер еще не создан в БД)
      - **Важно**: `position_orders.order_id` должен допускать NULL для поддержки произвольного порядка создания объектов
      - Необходима миграция БД: изменить `order_id` на `NULLABLE` в таблице `position_orders`
    - `relationship_type` (opened, increased, decreased, closed, reversed)
    - `size_delta`: изменение размера позиции, вычисляется как:
      - Для buy ордеров: `size_delta = +filled_quantity` (увеличение позиции)
      - Для sell ордеров: `size_delta = -filled_quantity` (уменьшение позиции)
      - Где `filled_quantity` берется из order события (`order-manager.order_events.filled_quantity`)
    - `execution_price`
    - `executed_at`
  - **Обновляет `positions.total_fees`** (если доработка реализована):
    - Получает `execution_fees` из order события или из `orders.fees`
    - Обновляет `positions.total_fees = positions.total_fees + execution_fees`
    - Это метаданные, не влияющие на состояние позиции (размер, PnL, avgPrice)
  - **Обработка частичных исполнений**:
    - Для каждого ордера создается **одна запись** в `position_orders` (уникальное ограничение `UNIQUE(position_id, order_id)`)
    - При первом частичном исполнении (`partially_filled` событие) создается запись в `position_orders`
    - При последующих частичных исполнениях (`partially_filled` или `filled` события) **обновляется существующая запись**:
      - `filled_quantity` обновляется до нового кумулятивного значения
      - `relationship_type` пересчитывается на основе текущего состояния позиции после каждого частичного исполнения
      - `execution_price` обновляется до новой средней цены исполнения (если биржа предоставляет обновленное значение)
    - Это обеспечивает актуальность данных и позволяет отслеживать, как `relationship_type` может изменяться между частичными исполнениями (например, сначала "increased", потом "closed")
    - **Важно**: `relationship_type` определяется на основе **текущего состояния позиции** из БД, а не на основе предыдущего `relationship_type` в `position_orders`
  
  **Тип связи между позициями и ордерами**:
  - На уровне БД: **many-to-many** (текущее состояние - таблица `position_orders` с уникальным ограничением `UNIQUE(position_id, order_id)`)
  - На уровне бизнес-логики: **1-to-many** (одна позиция может быть связана с множеством ордеров)
  - Один ордер теоретически может быть связан с несколькими позициями (например, в hedge-mode), но на практике:
    - В режиме `one-way`: один ордер влияет только на одну позицию (для данного `asset`)
    - В режиме `hedge`: один ордер может влиять на long или short позицию, но не на обе одновременно
  - Уникальное ограничение `UNIQUE(position_id, order_id)` гарантирует, что одна пара позиция-ордер может быть создана только один раз
  - **Поддержка произвольного порядка создания**: 
    - `order_id` может быть NULL при создании `position_orders` (если ордер еще не создан в БД)
    - При создании ордера в БД можно обновить `position_orders.order_id` через SQL UPDATE
    - Уникальное ограничение должно учитывать NULL значения (NULL не равен NULL в SQL, поэтому можно создать несколько записей с `order_id = NULL` для одной позиции)
    - **Рекомендация**: использовать частичный уникальный индекс `UNIQUE(position_id, order_id) WHERE order_id IS NOT NULL` и дополнительную проверку в приложении для NULL значений

**model-service** (гибридный подход):
- Подписывается на `order-manager.order_events` - для `prediction_trading_results` (быстро, точно)
- Подписывается на `position-manager.position_updated` - для обновления состояния позиций (синхронизировано)
- Использует `position_orders` для получения `execution_price` и `relationship_type` из order событий
- Получает синхронизированные данные о позициях из position событий

#### Детали реализации

**Изменения в order-manager**:

1. `order-manager/src/services/event_subscriber.py`:
   - Удалить метод `_update_position_from_order_fill()`
   - Удалить вызов `self.position_manager.update_position()`
   - Оставить только обновление таблицы `orders`
   - Убедиться, что событие в `order-manager.order_events` содержит всю необходимую информацию
   - **После создания/обновления ордера в БД обновлять связанные `position_orders`**:
     ```python
     async def _update_position_orders_after_order_creation(
         self,
         bybit_order_id: str,
         order_id: UUID,  # внутренний UUID ордера
         trace_id: Optional[str] = None
     ) -> None:
         """Обновить position_orders.order_id после создания ордера в БД."""
         pool = await DatabaseConnection.get_pool()
         query = """
             UPDATE position_orders
             SET order_id = $1
             WHERE bybit_order_id = $2
               AND order_id IS NULL
         """
         await pool.execute(query, order_id, bybit_order_id)
     ```
   - Вызывать эту функцию после каждого INSERT/UPDATE в таблицу `orders`

2. `order-manager/src/services/position_manager.py`:
   - Можно удалить полностью (больше не используется)
   - Или оставить только для обратной совместимости, но не вызывать

**Изменения в position-manager**:

1. `position-manager/src/consumers/order_position_consumer.py`:
   - Переименовать в `position_order_linker_consumer.py` (для ясности)
   - Изменить логику: вместо `update_position_from_order_fill()` → создание `position_orders`
   - Добавить метод `_create_position_order_relationship()`:
     ```python
     async def _create_position_order_relationship(
         self,
         order_event: OrderExecutionEvent,
         trace_id: Optional[str]
     ) -> None:
         # 1. Найти order в БД по bybit_order_id (может быть NULL если ордер еще не создан)
         order = await self._find_order_by_bybit_id(order_event.order_id)
         order_id = order.id if order else None
         
         # 2. Найти или создать позицию по asset, mode
         #    Если позиции нет, создать с минимальными данными из order события:
         #    - size = size_delta (из filled_quantity и side)
         #    - average_entry_price = execution_price
         #    - realized_pnl = NULL, unrealized_pnl = NULL (будут обновлены из position события)
         position = await self._find_or_create_position(
             asset=order_event.asset,
             mode=order_event.mode,
             initial_size=size_delta,  # из order события
             initial_avg_price=order_event.execution_price,
             trace_id=trace_id
         )
         
         # 3. Определить relationship_type на основе текущего состояния позиции
         #    (см. раздел "Доработка определения relationship_type")
         
         # 4. Создать или обновить position_orders запись (INSERT ... ON CONFLICT UPDATE)
         #    order_id может быть NULL - будет обновлен позже, когда ордер будет создан
         await self._upsert_position_order(
             position_id=position.id,
             order_id=order_id,  # может быть NULL
             relationship_type=relationship_type,
             size_delta=size_delta,
             execution_price=order_event.execution_price,
             executed_at=order_event.executed_at,
             trace_id=trace_id
         )
         
         # 5. Обновить positions.total_fees (если доработка реализована):
         #    UPDATE positions SET total_fees = total_fees + execution_fees WHERE id = position_id
         
         # 6. Если order_id был NULL и ордер был найден позже, обновить position_orders.order_id
     ```

2. `position-manager/src/consumers/websocket_position_consumer.py`:
   - Оставить без изменений (уже обновляет позиции только из position событий)

3. `position-manager/src/services/position_manager.py`:
   - Удалить метод `update_position_from_order_fill()` (или оставить для обратной совместимости, но не использовать)
   - Оставить только `update_position_from_websocket()`

#### Преимущества

- Четкое разделение: order-manager - ордера, position-manager - позиции
- position-manager использует данные напрямую от биржи (position события)
- `position_orders` создаются для связи ордеров с позициями
- position-manager не обновляет позиции из order событий (только из position событий)
- model-service получает все необходимые данные через `position_orders`
- Нет дублирования логики обновления позиций

#### Недостатки

- position-manager все еще подписывается на order_events (но только для создания связей, не для обновления позиций)
- ℹ️ **Особенность работы**: `position_orders` создаются после получения order события, а позиция обновляется из position события. Это нормальное поведение асинхронной системы - записи в БД могут создаваться в произвольном порядке, что не влияет на корректность работы

#### Обработка событий в разном порядке

В асинхронной системе события могут приходить в произвольном порядке. Это нормальное поведение, не конфликт:

- **Если position событие пришло раньше order события**:
  - Позиция обновляется из position события
  - Когда придет order событие, создается `position_orders` с правильным `relationship_type` на основе текущего состояния позиции
  
- **Если order событие пришло раньше position события**:
  - Создается `position_orders` на основе текущего состояния позиции (из БД)
  - Когда придет position событие, позиция обновляется, но `position_orders` уже создан

- **В обоих случаях итоговое состояние будет корректным**, так как:
  - position события - источник истины для позиций (размер, `positions.realized_pnl` - свойство позиции, avgPrice)
  - `position_orders` создается на основе текущего состояния позиции, независимо от порядка событий
  - Записи в БД могут создаваться в произвольном порядке - это не влияет на корректность работы
  - **Важно**: `positions.realized_pnl` (свойство позиции) обновляется только из position событий, а `prediction_trading_results.realized_pnl` (метрика модели) вычисляется model-service отдельно
  - **Логирование расхождений timestamp**:
    - При получении position события проверять `event_timestamp` и сравнивать с `execution_timestamp` из связанных `position_orders`
    - Логировать предупреждение, если position событие пришло с более старым timestamp, чем order событие (может указывать на проблемы синхронизации)
    - Всегда использовать position события как источник истины, независимо от timestamp (они содержат актуальное состояние от биржи)

#### Обработка событий в model-service

**Для prediction_trading_results** (из `order-manager.order_events`):
- События приходят сразу после исполнения ордера (низкая задержка)
- Простая обработка (одно событие = один ордер)
- Использует `position_orders` для получения `execution_price` и `relationship_type`
- Вычисляет `prediction_trading_results.realized_pnl` (метрика качества модели) на основе данных из `position_orders`
- **Важно**: это НЕ `positions.realized_pnl` (свойство позиции), а PnL для конкретного предсказания (signal_id)

**Для обновления состояния позиций** (из `position-manager.position_updated`):
- Получает синхронизированные данные о позициях
- Использует для генерации сигналов и принятия решений
- Может использовать для валидации данных из order событий

#### Гарантии корректности вычислений

**1. exit_price (цена выхода)**:
- ✅ **Источник**: `position_orders.execution_price` (сохраняется position-manager при создании связи)
- ✅ **Откуда берется**: из `order-manager.order_events` → `execution_price` (average_price из order события от Bybit)
- ✅ **Гарантия**: `execution_price` в `position_orders` всегда соответствует реальной цене исполнения ордера от биржи
- ✅ **Когда устанавливается**: при создании `position_orders` position-manager сохраняет `execution_price` из order события
- ✅ **Использование**: model-service получает `exit_price` из `position_orders.execution_price` для закрывающих ордеров (`relationship_type = "closed"` или `"decreased"`)

**2. realized_pnl - ДВА РАЗНЫХ ТИПА**:

**2.1. `positions.realized_pnl` (свойство позиции)**:
- ✅ **Источник**: напрямую от биржи Bybit (`cumRealisedPnl` в position событиях WebSocket)
- ✅ **Что это**: кумулятивная реализованная прибыль/убыток для всей позиции за все время (включает комиссии)
- ✅ **Обновление**: position-manager обновляет напрямую из position событий (`update_position_from_websocket()`)
- ✅ **Использование**: управление рисками, портфолио, текущее состояние позиции
- ✅ **Гарантия**: всегда соответствует данным от биржи (кумулятивное значение, комиссии уже включены)

**2.2. `prediction_trading_results.realized_pnl` (метрика качества модели)**:
- ✅ **Источник**: вычисляется в model-service на основе:
  - `entry_price`: выбирается по приоритету:
    1. `prediction_trading_results.entry_price` (если существует) - цена входа для конкретного предсказания
    2. `position_orders.execution_price` из записи с `relationship_type = "opened"` для данного `signal_id` - реальная цена открытия позиции для этого предсказания
    3. `positions.average_entry_price` (только в крайнем случае, с предупреждением о возможной неточности) - средняя цена входа для всей позиции, которая может быть открыта несколькими предсказаниями
  - `exit_price`: из `position_orders.execution_price` (для закрывающих ордеров)
  - `filled_quantity`: из `order-manager.order_events.filled_quantity` (кумулятивное исполненное количество ордера)
  - `execution_fees`: из `order-manager.order_events.execution_fees`
- ✅ **Что это**: PnL для конкретного предсказания (signal_id), связанного с конкретным ордером
- ✅ **Формула**:
  - Для long позиции: `realized_pnl = (exit_price - entry_price) * filled_quantity - execution_fees`
  - Для short позиции: `realized_pnl = (entry_price - exit_price) * filled_quantity - execution_fees`
- ✅ **Использование**: оценка качества модели (насколько предсказания привели к прибыли)
- ✅ **Гарантия**: 
  - `exit_price` берется из `position_orders.execution_price` (реальная цена от биржи)
  - `entry_price` выбирается по приоритету (см. выше): сначала из `prediction_trading_results.entry_price`, затем из `position_orders.execution_price` для открывающего ордера данного `signal_id`, и только в крайнем случае из `positions.average_entry_price` (с предупреждением о возможной неточности, так как это средняя цена для всей позиции, которая может быть открыта несколькими предсказаниями)
  - Все значения берутся из проверенных источников (Bybit → order-manager → position-manager → model-service)
- ✅ **Разница**: это НЕ свойство позиции, а метрика для конкретного предсказания. Одна позиция может быть связана с множеством предсказаний, каждое со своим `realized_pnl`

**3. avgPrice / average_entry_price (средняя цена входа)**:
- ✅ **Источник**: `positions.average_entry_price` из position событий от Bybit
- ✅ **Откуда берется**: напрямую из WebSocket position событий (`avgPrice` от биржи)
- ✅ **Гарантия**: 
  - position-manager обновляет `average_entry_price` **только** из position событий (`update_position_from_websocket()`)
  - Используется значение `avgPrice` напрямую от биржи, без вычислений на нашей стороне
  - Это источник истины для средней цены входа позиции
- ✅ **Использование**: 
  - Для отображения в UI и метриках
  - Для валидации вычислений на основе ордеров
  - Используется для вычисления `prediction_trading_results.realized_pnl` (метрика качества модели) в `prediction_trading_results`

**Различие между двумя типами realized_pnl**:
- `positions.realized_pnl` (свойство позиции): кумулятивное значение от биржи для всей позиции
- `prediction_trading_results.realized_pnl` (метрика качества модели): PnL для конкретного предсказания (signal_id)

**Решение: model-service вычисляет `prediction_trading_results.realized_pnl`**:
- ✅ **Ответственность**: `prediction_trading_results` - это связь между предсказаниями модели и результатами торговли, зона ответственности model-service (оценка качества модели)
- ✅ **Разделение ответственности**:
  - **position-manager**: 
    - Управление позициями (размер, PnL позиции, avgPrice)
    - Обновление `positions.realized_pnl` (свойство позиции) из position событий от биржи
    - Создание `position_orders` для связи ордеров с позициями
  - **model-service**: 
    - Связь предсказаний с результатами (`prediction_trading_results`)
    - Вычисление `prediction_trading_results.realized_pnl` (метрика качества модели) для оценки качества модели
- ✅ **Источники данных**: model-service использует данные из `position_orders` (execution_price, relationship_type) и `positions` (average_entry_price) для вычисления
- ✅ **Ключевое различие**: 
  - `positions.realized_pnl` = свойство позиции (кумулятивное значение для всей позиции от биржи)
  - `prediction_trading_results.realized_pnl` = метрика качества модели (PnL для конкретного предсказания)
  - Одна позиция может быть связана с множеством предсказаний, каждое со своим `realized_pnl`

**4. Дополнительные гарантии**:
- ✅ **Синхронизация данных**: 
  - `position_orders.execution_price` всегда соответствует реальной цене исполнения от биржи
  - `positions.average_entry_price` всегда соответствует `avgPrice` от биржи
  - `positions.realized_pnl` (свойство позиции) всегда соответствует `cumRealisedPnl` от биржи (кумулятивное значение для всей позиции, включает комиссии)
- ✅ **Порядок обработки**: 
  - Не зависит от порядка прихода событий (order и position события могут приходить в любом порядке)
  - Итоговое состояние всегда корректно, так как используется источник истины (данные от биржи)
- ✅ **Валидация**: 
  - model-service может валидировать вычисленный `prediction_trading_results.realized_pnl` (метрика модели) с `positions.realized_pnl` (свойство позиции) от биржи
  - Так как `positions.realized_pnl` уже включает комиссии (от биржи в `cumRealisedPnl`), то:
    - Сумма всех `prediction_trading_results.realized_pnl` для предсказаний (signal_id), связанных с позицией через `position_orders` (с учетом комиссий), должна быть близка к `positions.realized_pnl` (с учетом округления)
    - Или альтернативная формула валидации: `(вычисленный_pnl - total_fees) ≈ positions.realized_pnl`
      - Где `вычисленный_pnl` - PnL, вычисленный на основе ордеров без учета комиссий
      - `total_fees` - сумма комиссий из `positions.total_fees` (если доработка реализована) или сумма `execution_fees` из `position_orders`
      - `positions.realized_pnl` уже включает комиссии от биржи
  - При расхождениях можно использовать значение от биржи как источник истины
  - **Логирование расхождений между order и position событиями**:
    - При обновлении позиции из position события сравнивать размер позиции с вычисленным на основе order событий
    - Логировать предупреждение при обнаружении расхождений (может указывать на другие источники изменения позиции: ручная торговля, другие системы и т.д.)
    - Всегда использовать position события как источник истины, так как они содержат актуальное состояние от биржи
    - Расхождения не являются ошибкой - это нормальное поведение при наличии нескольких источников изменения позиции

#### Терминология и маппинг полей

**Примечание о терминологии количества**:

В проекте используется следующая терминология для исполненного количества ордера:

1. **В БД и событиях order-manager**: `filled_quantity` (кумулятивное исполненное количество)
   - Это значение увеличивается при каждом частичном исполнении ордера
   - Соответствует `cumExecQty` в API Bybit
   - Используется в таблице `orders.filled_quantity` и в событиях `order-manager.order_events`

2. **В model-service**: `execution_quantity` (то же значение, но с другим названием)
   - Модель `OrderExecutionEvent` использует поле `execution_quantity`
   - При преобразовании событий: `order.filled_quantity` → `OrderExecutionEvent.execution_quantity`
   - **Требуется доработка**: для полной согласованности рекомендуется переименовать `execution_quantity` → `filled_quantity` в model-service (см. раздел "Необходимые доработки кода")

3. **Семантика**:
   - `filled_quantity` - это **кумулятивное** значение (накопленное за все частичные исполнения)
   - Для закрывающих ордеров это количество закрытой позиции
   - Для открывающих ордеров это количество открытой позиции
   - В формулах вычисления PnL используется именно это кумулятивное значение

**Примечание о терминологии PnL в position событиях Bybit**:

- **В WebSocket событиях Bybit**: поле называется `cumRealisedPnl` (cumulative realized PnL, кумулятивное значение для всей позиции, включает комиссии)
- **В нормализованных событиях ws-gateway**: нормализуется в `realised_pnl` (поддержка обоих вариантов: `cumRealisedPnl` и `realisedPnl`)
- **В БД (positions.realized_pnl)**: хранится как `realized_pnl`, обновляется из `cumRealisedPnl` от Bybit

**Маппинг полей в потоке данных**:
- Bybit WebSocket → `cumExecQty`
- ws-gateway → нормализация в `filled_quantity` (или `executed_qty`)
- order-manager → сохранение в `orders.filled_quantity` и публикация в `order.filled_quantity`
- position-manager → чтение `order.filled_quantity` (поддерживает также `execution_quantity` для обратной совместимости)
- model-service → преобразование `order.filled_quantity` → `OrderExecutionEvent.execution_quantity`

#### Хранение комиссий

**Комиссии за ордера**:
- ✅ **Сохраняются в таблице `orders`**: поле `fees` (DECIMAL(20, 8), может быть NULL)
  - Обновляется order-manager при получении order события от Bybit
  - Содержит общую сумму комиссий за ордер
- ✅ **Сохраняются в таблице `execution_events`**: поле `execution_fees` (DECIMAL(20, 8), NOT NULL)
  - Сохраняется model-service при обработке order событий
  - Используется для вычисления `prediction_trading_results.realized_pnl` (метрика качества модели) в `prediction_trading_results`
- ✅ **Сохраняются в таблице `prediction_trading_results`**: поле `fees` (DECIMAL(20, 8), DEFAULT 0)
  - Сохраняется model-service при вычислении PnL
  - Используется для учета комиссий в расчете прибыли/убытка

**Комиссии за позиции**:
- ✅ **Рекомендуемая доработка: сохранять отдельно** в таблице `positions`
  - Добавить поле `total_fees` (DECIMAL(20, 8), DEFAULT 0) в таблицу `positions`
  - Агрегировать комиссии из связанных ордеров через `position_orders`
  - Обновлять при создании `position_orders` в position-manager
- ✅ **Текущее состояние: учитываются в `positions.realized_pnl`** (свойство позиции, кумулятивное значение от биржи)
  - Биржа Bybit включает все комиссии в `cumRealisedPnl` (кумулятивное значение для всей позиции)
  - position-manager обновляет `positions.realized_pnl` напрямую из position событий (из поля `cumRealisedPnl`)
  - Можно использовать для валидации вычисленных комиссий
  - **Важно**: это НЕ `prediction_trading_results.realized_pnl` (метрика модели), а свойство позиции

**Доработка для сохранения комиссий за позиции**:

1. **Миграция БД**:
   ```sql
   ALTER TABLE positions 
   ADD COLUMN total_fees DECIMAL(20, 8) DEFAULT 0 NOT NULL;
   ```
   - Добавить поле `total_fees` для хранения кумулятивных комиссий за позицию
   - Значение по умолчанию: 0

2. **Изменения в position-manager**:
   - При создании `position_orders` в `_create_position_order_relationship()`:
     - Получить `execution_fees` из order события или из таблицы `orders.fees`
     - Обновить `positions.total_fees = positions.total_fees + execution_fees`
     - Использовать SQL: `UPDATE positions SET total_fees = total_fees + $1 WHERE id = $2`
   - При обновлении позиции из position событий:
     - Оставить `total_fees` без изменений (обновляется только при создании `position_orders`)
     - Можно использовать для валидации: `(вычисленный_pnl - total_fees) ≈ positions.realized_pnl` (где `вычисленный_pnl` - PnL, вычисленный на основе ордеров без учета комиссий, а `positions.realized_pnl` уже включает комиссии от биржи)

3. **Преимущества доработки**:
   - ✅ Прозрачность: видно точную сумму комиссий за позицию
   - ✅ Валидация: можно сравнить `total_fees` с суммой комиссий из `position_orders`
   - ✅ Аналитика: можно анализировать комиссии отдельно от PnL
   - ✅ Отладка: легче найти расхождения в расчетах

4. **Реализация в рекомендуемом решении**:
   - position-manager при создании `position_orders`:
     1. Получает `execution_fees` из order события
     2. Создает `position_orders` с информацией об ордере
     3. Обновляет `positions.total_fees = positions.total_fees + execution_fees`
   - Это не нарушает разделение ответственности:
     - position-manager обновляет позиции только из position событий (размер, PnL, avgPrice)
     - `total_fees` обновляется на основе order событий, но это метаданные, не влияющие на состояние позиции

**Как используются комиссии**:
- ✅ **При вычислении `prediction_trading_results.realized_pnl` (метрика модели) в model-service**:
  - Используется `execution_fees` из `order-manager.order_events` или `execution_events`
  - Формула зависит от направления позиции:
    - Для long позиции: `prediction_trading_results.realized_pnl = (exit_price - entry_price) * filled_quantity - execution_fees`
    - Для short позиции: `prediction_trading_results.realized_pnl = (entry_price - exit_price) * filled_quantity - execution_fees`
  - Примечание: `filled_quantity` - это кумулятивное исполненное количество ордера (см. раздел "Терминология и маппинг полей")
  - Комиссии вычитаются из прибыли/убытка для конкретного предсказания
- ✅ **В `positions.realized_pnl` (свойство позиции)**:
  - Комиссии уже включены в кумулятивное значение от биржи (`cumRealisedPnl`)
  - position-manager обновляет напрямую из position событий (поле `cumRealisedPnl`), без отдельного учета комиссий
- ✅ **При валидации данных**:
  - Можно сравнить сумму всех `prediction_trading_results.realized_pnl` (с учетом комиссий) с `positions.realized_pnl` (свойство позиции) от биржи
  - Биржа уже включает комиссии в `cumRealisedPnl`, поэтому значения должны совпадать (с учетом округления)

**Гарантии корректности комиссий**:
- ✅ **Комиссии за ордера**: берутся напрямую из order событий от Bybit
- ✅ **Комиссии в `prediction_trading_results.realized_pnl` (метрика модели)**: учитываются при вычислении в model-service
- ✅ **Комиссии в `positions.realized_pnl` (свойство позиции)** (текущее состояние): уже включены в кумулятивное значение от биржи
- ✅ **Комиссии в позициях** (с доработкой): сохраняются отдельно в `positions.total_fees`
  - Агрегируются из комиссий всех ордеров, связанных с позицией через `position_orders`
  - Обновляются position-manager при создании `position_orders`
  - Можно валидировать: сумма `total_fees` должна совпадать с суммой `orders.fees` для всех ордеров позиции
  - **Важно**: `positions.total_fees` - это метаданные позиции, не влияющие на `positions.realized_pnl` (который берется от биржи)

## Необходимые доработки кода

### 1. Унификация терминологии в model-service (рекомендуется)

**Проблема**: В model-service используется `execution_quantity`, в то время как в остальном проекте используется `filled_quantity`.

**Рекомендуемое решение**: Переименовать `execution_quantity` → `filled_quantity` в model-service для полной согласованности терминологии.

**Изменения**:

1. **Модель `OrderExecutionEvent`** (`model-service/src/models/execution_event.py`):
   ```python
   # Было:
   execution_quantity: float = Field(..., description="Executed quantity", gt=0)
   
   # Должно быть:
   filled_quantity: float = Field(..., description="Filled quantity (cumulative)", gt=0)
   ```

2. **Валидация в `execution_event_consumer.py`**:
   - Изменить проверку `execution_quantity` → `filled_quantity`
   - Обновить все места использования `event.execution_quantity` → `event.filled_quantity`

3. **Преобразование событий** (`_transform_order_manager_event()`):
   - Изменить маппинг: `filled_quantity` → `filled_quantity` (без переименования)
   - Убрать преобразование названия поля

4. **Миграция данных** (если есть таблица `execution_events`):
   ```sql
   ALTER TABLE execution_events 
   RENAME COLUMN execution_quantity TO filled_quantity;
   ```

5. **Обновление всех мест использования**:
   - `model-service/src/main.py`: `event.execution_quantity` → `event.filled_quantity`
   - `model-service/src/services/prediction_trading_linker.py`: обновить параметры и использование
   - Все тесты и спецификации

### 2. Доработка определения `relationship_type` при создании новой позиции

**Проблема**: В документации не полностью описана логика определения `relationship_type`, особенно для случая reversed.

**Рекомендуемое решение**: В `position-manager/src/consumers/position_order_linker_consumer.py` (после переименования) добавить логику:

```python
async def _create_position_order_relationship(
    self,
    order_event: OrderExecutionEvent,
    trace_id: Optional[str]
) -> None:
    # 1. Найти order в БД по bybit_order_id (может быть NULL если ордер еще не создан)
    order = await self._find_order_by_bybit_id(order_event.order_id)
    order_id = order.id if order else None
    bybit_order_id = order_event.order_id
    
    # 2. Вычислить size_delta
    if order_event.side == "buy":
        size_delta = +order_event.filled_quantity
    else:  # sell
        size_delta = -order_event.filled_quantity
    
    # 3. Найти или создать позицию по asset, mode
    #    Если позиции нет, создать с минимальными данными из order события:
    position = await self._find_or_create_position(
        asset=order_event.asset,
        mode=order_event.mode,
        initial_size=size_delta,  # из order события
        initial_avg_price=order_event.execution_price,
        trace_id=trace_id
    )
    
    # 4. Определить relationship_type на основе текущего состояния позиции
    current_size = position.size or Decimal(0)
    
    if current_size == 0:
        # Позиция пустая или только что создана
        relationship_type = "opened"
    elif (order_event.side == "buy" and current_size > 0) or \
         (order_event.side == "sell" and current_size < 0):
        # Увеличение позиции в том же направлении
        relationship_type = "increased"
    else:
        # Уменьшение или изменение направления позиции
        new_size = current_size + size_delta
        
        if new_size == 0:
            # Позиция полностью закрыта
            relationship_type = "closed"
        elif (current_size > 0 and new_size < 0) or (current_size < 0 and new_size > 0):
            # Позиция закрыта и изменила направление
            relationship_type = "reversed"
        else:
            # Частичное закрытие позиции (тот же знак, но меньше по модулю)
            relationship_type = "decreased"
    
    # 5. Создать или обновить position_orders запись
    #    order_id может быть NULL - будет обновлен позже, когда ордер будет создан
    #    Использовать INSERT ... ON CONFLICT UPDATE для обработки частичных исполнений
    await self._upsert_position_order(
        position_id=position.id,
        order_id=order_id,  # может быть NULL
        bybit_order_id=bybit_order_id,  # для связи до создания ордера
        relationship_type=relationship_type,
        size_delta=size_delta,
        execution_price=order_event.execution_price,
        executed_at=order_event.executed_at,
        trace_id=trace_id
    )
    
    # 6. Обновить positions.total_fees (если доработка реализована):
    if order_event.execution_fees:
        await self._update_position_total_fees(
            position_id=position.id,
            fees=order_event.execution_fees,
            trace_id=trace_id
        )
    
    # 7. Если order_id был NULL и ордер был найден, можно обновить position_orders.order_id
    #    (опционально, можно сделать в отдельном фоновом процессе)
```

**Приоритет**: Высокий (критично для корректной работы)

### 3. Добавление поля `size_delta` в `position_orders`

**Проблема**: В документации указано поле `size_delta`, но не описано, как оно вычисляется.

**Рекомендуемое решение**: В `_create_position_order_relationship()` добавить вычисление:

```python
# Вычисление size_delta
if order_event.side == "buy":
    size_delta = +order_event.filled_quantity
else:  # sell
    size_delta = -order_event.filled_quantity

# Создать position_orders с size_delta
await self._create_position_order(
    position_id=position.id,
    order_id=order.id,
    relationship_type=relationship_type,
    size_delta=size_delta,
    execution_price=order_event.execution_price,
    executed_at=order_event.executed_at
)
```

**Приоритет**: Средний (улучшает прозрачность данных)

### 4. Обработка частичных исполнений

**Рекомендуемое решение**: 
- Для каждого ордера создается **одна запись** в `position_orders` (уникальное ограничение `UNIQUE(position_id, order_id)`)
- При первом частичном исполнении (`partially_filled` событие) создается запись в `position_orders`
- При последующих частичных исполнениях (`partially_filled` или `filled` события) **обновляется существующая запись**:
  - Использовать SQL `INSERT ... ON CONFLICT (position_id, order_id) DO UPDATE` для обновления записи
  - Обновлять `filled_quantity`, `execution_price`, `relationship_type` на основе текущего состояния позиции
  - **Важно**: `relationship_type` пересчитывается на основе текущего состояния позиции при каждом обновлении, чтобы отражать актуальное состояние связи
- Причины:
  - `filled_quantity` - это кумулятивное значение, которое обновляется при каждом частичном исполнении
  - `relationship_type` может изменяться между частичными исполнениями (например, сначала "increased", потом "closed")
  - Уникальное ограничение гарантирует одну запись на пару (позиция, ордер)

**Приоритет**: Высокий (критично для корректной работы)

### 5. Миграция БД для поддержки произвольного порядка создания объектов

**Проблема**: Текущая схема БД требует, чтобы `order_id` был NOT NULL в таблице `position_orders`, что не позволяет создавать `position_orders` до создания ордера в БД.

**Рекомендуемое решение**: Доработать SQL схему для поддержки произвольного порядка создания объектов.

**Миграция БД**:

```sql
-- 1. Удалить старое уникальное ограничение (будет заменено частичным индексом)
ALTER TABLE position_orders 
DROP CONSTRAINT IF EXISTS uq_position_order;

-- 2. Удалить внешний ключ для order_id (будет пересоздан с поддержкой NULL)
ALTER TABLE position_orders 
DROP CONSTRAINT IF EXISTS position_orders_order_id_fkey;

-- 3. Изменить order_id на NULLABLE
ALTER TABLE position_orders 
ALTER COLUMN order_id DROP NOT NULL;

-- 4. Добавить поле bybit_order_id в position_orders для связи до создания ордера
ALTER TABLE position_orders 
ADD COLUMN IF NOT EXISTS bybit_order_id VARCHAR(255);

-- 5. Создать частичный уникальный индекс для поддержки уникальности при order_id IS NOT NULL
--    (NULL значения не будут проверяться уникальным ограничением)
CREATE UNIQUE INDEX IF NOT EXISTS position_orders_position_order_unique 
ON position_orders(position_id, order_id) 
WHERE order_id IS NOT NULL;

-- 6. Создать уникальный индекс для случая order_id IS NULL с использованием bybit_order_id
--    (предотвращает дублирование записей с NULL order_id для одной позиции)
CREATE UNIQUE INDEX IF NOT EXISTS position_orders_position_bybit_order_unique 
ON position_orders(position_id, bybit_order_id) 
WHERE order_id IS NULL AND bybit_order_id IS NOT NULL;

-- 7. Пересоздать внешний ключ с поддержкой NULL
ALTER TABLE position_orders 
ADD CONSTRAINT position_orders_order_id_fkey 
FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE;

-- 8. Создать индекс для bybit_order_id для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_position_orders_bybit_order_id 
ON position_orders(bybit_order_id) 
WHERE bybit_order_id IS NOT NULL;
```

**Обновление order_id после создания ордера**:

После создания ордера в БД (в order-manager) **обязательно обновлять** связанные `position_orders`:

```sql
-- Обновить position_orders.order_id для записей с NULL order_id
UPDATE position_orders 
SET order_id = (
    SELECT id FROM orders 
    WHERE orders.order_id = position_orders.bybit_order_id
)
WHERE bybit_order_id IS NOT NULL 
  AND order_id IS NULL 
  AND EXISTS (
      SELECT 1 FROM orders 
      WHERE orders.order_id = position_orders.bybit_order_id
  );
```

**Реализация обновления**:
- **Выполняется в order-manager после создания ордера в БД** (синхронно)
- После INSERT в таблицу `orders` в order-manager вызывается функция обновления `position_orders`
- Это обеспечивает мгновенное обновление связи между ордером и position_orders
- Если `position_orders` еще не создан (order событие пришло раньше), обновление будет выполнено позже при создании position_orders

**Порядок выполнения миграций**:

Все изменения БД должны выполняться последовательно, так как они имеют зависимости друг от друга. Рекомендуется разделить на несколько миграций с четкими зависимостями:

**Миграция 1: Подготовка таблицы position_orders для поддержки NULL**
- Номер миграции: `XXX_add_bybit_order_id_to_position_orders.sql` (где XXX - следующий номер миграции)
- Цель: Добавить поле `bybit_order_id` без изменения существующих ограничений (совместимо с текущей схемой)
- Зависимости: Нет (можно выполнить независимо)
```sql
-- Добавить поле bybit_order_id (совместимо с текущей схемой)
ALTER TABLE position_orders 
ADD COLUMN IF NOT EXISTS bybit_order_id VARCHAR(255);

-- Создать индекс для быстрого поиска (частичный индекс)
CREATE INDEX IF NOT EXISTS idx_position_orders_bybit_order_id 
ON position_orders(bybit_order_id) 
WHERE bybit_order_id IS NOT NULL;
```

**Миграция 2: Изменение order_id на NULLABLE и обновление индексов**
- Номер миграции: `XXX_make_order_id_nullable_in_position_orders.sql`
- Цель: Изменить `order_id` на NULLABLE и обновить уникальные ограничения
- Зависимости: Требует Миграцию 1 (bybit_order_id должно существовать)
```sql
-- 1. Удалить старое уникальное ограничение
ALTER TABLE position_orders 
DROP CONSTRAINT IF EXISTS uq_position_order;

-- 2. Удалить внешний ключ для order_id (будет пересоздан)
ALTER TABLE position_orders 
DROP CONSTRAINT IF EXISTS position_orders_order_id_fkey;

-- 3. Изменить order_id на NULLABLE
ALTER TABLE position_orders 
ALTER COLUMN order_id DROP NOT NULL;

-- 4. Создать частичный уникальный индекс для order_id IS NOT NULL
CREATE UNIQUE INDEX IF NOT EXISTS position_orders_position_order_unique 
ON position_orders(position_id, order_id) 
WHERE order_id IS NOT NULL;

-- 5. Создать уникальный индекс для order_id IS NULL с использованием bybit_order_id
CREATE UNIQUE INDEX IF NOT EXISTS position_orders_position_bybit_order_unique 
ON position_orders(position_id, bybit_order_id) 
WHERE order_id IS NULL AND bybit_order_id IS NOT NULL;

-- 6. Пересоздать внешний ключ с поддержкой NULL
ALTER TABLE position_orders 
ADD CONSTRAINT position_orders_order_id_fkey 
FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE;
```

**Миграция 3 (опционально): Добавление total_fees в positions**
- Номер миграции: `XXX_add_total_fees_to_positions.sql`
- Цель: Добавить поле `total_fees` для отслеживания комиссий за позицию
- Зависимости: Нет (независимая миграция)
```sql
ALTER TABLE positions 
ADD COLUMN IF NOT EXISTS total_fees DECIMAL(20, 8) DEFAULT 0 NOT NULL;
```

**Правила выполнения миграций**:
1. ✅ **Последовательность**: Миграции должны выполняться строго в указанном порядке
2. ✅ **Проверка зависимостей**: Перед выполнением миграции 2 проверить, что миграция 1 выполнена успешно
3. ✅ **Откат**: Каждая миграция должна быть обратимой (описание rollback в комментариях)
4. ✅ **Тестирование**: После каждой миграции запускать тесты для проверки корректности схемы
5. ✅ **Резервное копирование**: Создать резервную копию БД перед выполнением миграций
6. ✅ **Мониторинг**: После выполнения миграций мониторить логи приложения на наличие ошибок

**Рекомендуемый процесс развертывания**:
1. Развернуть Миграцию 1 на staging окружении
2. Протестировать работу системы с новой схемой
3. Развернуть Миграцию 2 на staging окружении
4. Протестировать работу с NULL order_id
5. Развернуть обе миграции на production в той же последовательности

**Приоритет**: Высокий (критично для корректной работы при асинхронной обработке событий)

### 6. Обработка ошибок при создании position_orders

**Проблема**: Не описано, как обрабатывать различные типы ошибок при создании `position_orders`.

**Рекомендуемое решение**: Реализовать обработку ошибок для каждого типа ситуации.

**Реализация**:

1. **Ордер не найден в БД**:
   - ✅ **Решение**: Создавать `position_orders` с `order_id = NULL` и `bybit_order_id`
   - После создания ордера в БД order-manager обновит `order_id` (см. раздел "Обновление order_id после создания ордера")

2. **Позиция не найдена в БД**:
   - ✅ **Решение**: Создавать позицию с минимальными данными из order события:
     - `size = size_delta` (из `filled_quantity` и `side` ордера)
     - `average_entry_price = execution_price`
     - `realized_pnl = NULL`, `unrealized_pnl = NULL` (будут обновлены из position события)
   - Логировать информационное сообщение о создании позиции из order события

3. **Ошибка уникальности (дубликат position_orders)**:
   - ✅ **Решение**: Использовать `INSERT ... ON CONFLICT UPDATE` для идемпотентной операции
     ```python
     query = """
         INSERT INTO position_orders (
             position_id, order_id, bybit_order_id, relationship_type,
             size_delta, execution_price, executed_at
         ) VALUES ($1, $2, $3, $4, $5, $6, $7)
         ON CONFLICT (position_id, order_id) 
         WHERE order_id IS NOT NULL
         DO UPDATE SET
             relationship_type = EXCLUDED.relationship_type,
             size_delta = EXCLUDED.size_delta,
             execution_price = EXCLUDED.execution_price,
             executed_at = EXCLUDED.executed_at
         ON CONFLICT (position_id, bybit_order_id)
         WHERE order_id IS NULL AND bybit_order_id IS NOT NULL
         DO UPDATE SET
             relationship_type = EXCLUDED.relationship_type,
             size_delta = EXCLUDED.size_delta,
             execution_price = EXCLUDED.execution_price,
             executed_at = EXCLUDED.executed_at
     """
     ```
   - Это обеспечивает идемпотентность операции и корректную обработку частичных исполнений

4. **Ошибка БД (connection lost, deadlock, constraint violation)**:
   - ✅ **Решение**: Nack с requeue в RabbitMQ для повторной доставки события
     ```python
     try:
         await self._create_position_order_relationship(order_event, trace_id)
         await message.ack()
     except (asyncio.TimeoutError, asyncpg.PostgresConnectionError, asyncpg.DeadlockDetectedError) as e:
         logger.warning(
             "position_order_creation_db_error_requeue",
             error=str(e),
             error_type=type(e).__name__,
             trace_id=trace_id
         )
         await message.nack(requeue=True)  # Повторная доставка
     except Exception as e:
         # Другие ошибки БД
         logger.error(
             "position_order_creation_db_error",
             error=str(e),
             error_type=type(e).__name__,
             trace_id=trace_id,
             exc_info=True
         )
         await message.nack(requeue=True)
     ```
   - RabbitMQ автоматически повторно доставит событие с экспоненциальной задержкой

5. **Ошибка валидации данных (некорректные данные в событии)**:
   - ✅ **Решение**: Reject без requeue + логирование ошибки
     ```python
     try:
         order_event = OrderExecutionEvent.from_message(payload)
     except (ValueError, KeyError) as e:
         logger.error(
             "position_order_event_validation_failed",
             error=str(e),
             payload_keys=list(payload.keys()) if isinstance(payload, dict) else None,
             trace_id=trace_id
         )
         await message.reject(requeue=False)  # Постоянная ошибка, не повторять
         return
     ```
   - Событие с некорректными данными не должно обрабатываться повторно
   - Опционально: отправка в Dead Letter Queue для ручной проверки

**Приоритет**: Высокий (критично для надежности системы)

### 7. Логирование расхождений для диагностики

**Проблема**: Не описано, как обрабатывать и логировать расхождения между order и position событиями.

**Рекомендуемое решение**: Добавить логирование расхождений для диагностики, но всегда использовать position события как источник истины.

**Реализация**:

1. **Логирование расхождений timestamp**:
   - При получении position события сравнить `event_timestamp` с `execution_timestamp` из связанных `position_orders`
   - Логировать предупреждение, если position событие пришло с более старым timestamp, чем order событие:
     ```python
     if position_event.event_timestamp < order_execution_timestamp:
         logger.warning(
             "position_event_older_than_order",
             asset=asset,
             position_timestamp=position_event.event_timestamp,
             order_timestamp=order_execution_timestamp,
             trace_id=trace_id
         )
     ```
   - Всегда использовать position события как источник истины, независимо от timestamp

2. **Логирование расхождений размера позиции**:
   - При обновлении позиции из position события вычислить ожидаемый размер на основе `position_orders`
   - Сравнить с фактическим размером из position события
   - Логировать предупреждение при обнаружении расхождений:
     ```python
     expected_size = calculate_size_from_position_orders(position_id)
     actual_size = position_event.size
     if abs(expected_size - actual_size) > threshold:
         logger.warning(
             "position_size_mismatch",
             asset=asset,
             expected_size=expected_size,
             actual_size=actual_size,
             difference=actual_size - expected_size,
             trace_id=trace_id,
             note="Position may have been modified by other sources (manual trading, other systems)"
         )
     ```
   - Всегда обновлять позицию данными из position события (источник истины)

**Приоритет**: Средний (улучшает диагностику и мониторинг)

## Выводы

### ✅ Разделение возможно с гибридным подходом

**Положительные аспекты**:
- Упрощает архитектуру
- Четкое разделение ответственности
- position-manager использует данные напрямую от биржи
- Все критические проблемы решены

**Решения для критических проблем**:
1. **position_orders** - создает position-manager на основе order событий
2. **Связь ордеров с позициями** - position-manager создает связи при получении order событий
3. **prediction_trading_results** - model-service использует гибридный подход (оба источника)

### Итоговая архитектура

#### Разделение ответственности

1. **order-manager**: 
   - Только ордера (`ws-gateway.order`)
   - Обновляет только таблицу `orders`
   - Публикует события в `order-manager.order_events`
   - **НЕ обновляет позиции**
   - **НЕ создает position_orders**

2. **position-manager**: 
   - Позиции из position событий (`ws-gateway.position`)
   - Подписывается на `order-manager.order_events` **только для создания `position_orders`**
   - Обновляет позиции **только** из position событий
   - Создает `position_orders` на основе order событий
   - Публикует события в `position-manager.position_updated`

3. **model-service**: 
   - Подписывается на `order-manager.order_events` → для `prediction_trading_results` (быстро, точно)
   - Подписывается на `position-manager.position_updated` → для обновления состояния позиций (синхронизировано)

**Ключевой момент**: position-manager подписывается на order_events, но **не обновляет позиции** из них, а только создает связи (`position_orders`). Позиции обновляются только из position событий от Bybit.

Это сохраняет разделение ответственности (order-manager - ордера, position-manager - позиции) и решает все критические проблемы, при этом давая model-service оптимальные источники данных для разных задач.

