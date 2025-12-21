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

## Предложенное решение

### order-manager
**Подписки**:
- `ws-gateway.order` - только события ордеров

**Ответственность**:
- Только обновление таблицы `orders`
- Публикация событий в `order-manager.order_events` (опционально, для других сервисов)
- **НЕ обновляет позиции**
- **НЕ создает position_orders**

### position-manager
**Подписки**:
- `ws-gateway.position` - только события позиций

**Ответственность**:
- Только обновление позиций из WebSocket событий
- **НЕ подписывается на order_events**

## Анализ влияния на корректность работы

### ✅ Положительные аспекты

#### 1. Четкое разделение ответственности
- **order-manager** отвечает только за ордера
- **position-manager** отвечает только за позиции
- Нет дублирования логики обновления позиций
- Проще тестировать и поддерживать

#### 2. Упрощение архитектуры
- Убирается зависимость position-manager от order-manager
- Меньше точек отказа
- Проще масштабировать сервисы независимо

#### 3. Источник истины для позиций
- position-manager получает данные напрямую от Bybit через WebSocket
- Не зависит от вычислений order-manager
- Использует `realisedPnl` и `avgPrice` напрямую от биржи

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

**Возможные решения**:
1. **position-manager создает position_orders из position событий**:
   - Проблема: position события не содержат `order_id`
   - Нужно искать ордера в БД по `asset`, `side`, `execution_price`, `timestamp`
   - Неточное сопоставление (несколько ордеров могут совпадать)

2. **order-manager публикует события с order_id, position-manager создает position_orders**:
   - order-manager публикует в `order-manager.order_events` с полной информацией
   - position-manager подписывается на эту очередь (но это нарушает разделение)
   - Или position-manager опрашивает таблицу `orders` периодически

3. **Отдельный сервис для связывания**:
   - Новый сервис подписывается на обе очереди
   - Создает `position_orders` на основе событий
   - Усложняет архитектуру

#### 2. КРИТИЧНО: Как position-manager узнает об исполнении ордеров?

**Текущая ситуация**:
- position-manager получает события из `order-manager.order_events`
- Обновляет позиции через `update_position_from_order_fill()`
- Вычисляет `realized_pnl_delta` на основе `execution_price`

**Проблема при разделении**:
- position-manager не получает события об исполнении ордеров
- Не может обновить позицию при исполнении ордера
- Позиция обновляется только из WebSocket position событий

**Влияние**:
- Задержка в обновлении позиций (зависит от частоты position событий от Bybit)
- Возможны расхождения между размером позиции из order событий и position событий
- `realized_pnl` будет обновляться только из position событий (кумулятивное значение)

**Возможные решения**:
1. **Использовать только position события**:
   - Плюсы: Источник истины от биржи, не зависит от order-manager
   - Минусы: Задержка обновления, нет детальной информации об ордерах

2. **position-manager опрашивает таблицу orders**:
   - Периодически проверяет новые исполненные ордера
   - Обновляет позиции на основе ордеров
   - Проблема: Задержка, дополнительная нагрузка на БД

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

**Влияние**:
- Нужен механизм валидации и синхронизации
- Возможны расхождения, которые нужно разрешать

#### 4. ВАЖНО: prediction_trading_results

**Текущая ситуация**:
- `model-service` получает `OrderExecutionEvent` из `order-manager.order_events`
- Использует `position_orders` для получения `execution_price` и `relationship_type`
- Вычисляет `realized_pnl_delta` на основе этих данных

**Проблема при разделении**:
- Без `position_orders` невозможно получить `execution_price` и `relationship_type`
- Без событий от order-manager невозможно получить детальную информацию об исполнении

**Возможные решения**:
1. **model-service подписывается на order-manager.order_events**:
   - Получает события об исполнении ордеров
   - Но все еще нужны `position_orders` для связи с позициями

2. **Использовать только position события**:
   - Использовать `realisedPnl` из position событий
   - Распределять между предсказаниями пропорционально
   - Проблема: Нет точной связи с конкретными ордерами

## Оценка корректности работы

### Сценарий 1: Исполнение ордера

**Текущая архитектура**:
1. Bybit → ws-gateway.order → order-manager
2. order-manager обновляет `orders` и `positions`
3. order-manager создает `position_orders`
4. order-manager публикует в `order-manager.order_events`
5. position-manager получает событие → обновляет позицию (если нужно)
6. model-service получает событие → обновляет `prediction_trading_results`

**Предложенная архитектура**:
1. Bybit → ws-gateway.order → order-manager
2. order-manager обновляет только `orders`
3. order-manager публикует в `order-manager.order_events` (опционально)
4. Bybit → ws-gateway.position → position-manager
5. position-manager обновляет позицию из position события
6. ❓ Кто создает `position_orders`?
7. ❓ Как model-service получает детальную информацию?

**Проблемы**:
- Нет связи между ордерами и позициями (`position_orders`)
- Нет детальной информации об исполнении для `prediction_trading_results`
- Задержка в обновлении позиций (зависит от частоты position событий)

### Сценарий 2: Закрытие позиции

**Текущая архитектура**:
1. Исполняется SELL ордер → order-manager обновляет позицию → `relationship_type = "closed"`
2. `position_orders` содержит `execution_price` закрывающего ордера
3. model-service использует эти данные для вычисления `exit_price` и `realized_pnl`

**Предложенная архитектура**:
1. Исполняется SELL ордер → order-manager обновляет только `orders`
2. position событие приходит позже → position-manager обновляет позицию
3. ❓ Как определить, что позиция закрыта конкретным ордером?
4. ❓ Как получить `execution_price` закрывающего ордера?

**Проблемы**:
- Невозможно точно связать закрытие позиции с конкретным ордером
- Нет `execution_price` для вычисления `exit_price` в `prediction_trading_results`

## Рекомендации

### Вариант 1: Гибридный подход (рекомендуется)

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
  - Находит или создает позицию в БД (по `asset`, `mode`)
  - Определяет `relationship_type` на основе текущего состояния позиции и ордера
  - Создает запись в `position_orders` с:
    - `position_id`
    - `order_id` (из таблицы `orders` по `bybit_order_id`)
    - `relationship_type` (opened, increased, decreased, closed, reversed)
    - `size_delta`
    - `execution_price`
    - `executed_at`

**model-service**:
- Подписывается на `order-manager.order_events` (как сейчас)
- Использует `position_orders` для получения `execution_price` и `relationship_type`
- Вычисляет `realized_pnl` на основе данных из `position_orders`

#### Детали реализации

**Изменения в order-manager**:

1. `order-manager/src/services/event_subscriber.py`:
   - Удалить метод `_update_position_from_order_fill()`
   - Удалить вызов `self.position_manager.update_position()`
   - Оставить только обновление таблицы `orders`
   - Убедиться, что событие в `order-manager.order_events` содержит всю необходимую информацию

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
         # 1. Найти order в БД по bybit_order_id
         # 2. Найти или создать позицию по asset, mode
         # 3. Определить relationship_type
         # 4. Создать position_orders запись
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
- Небольшая задержка: `position_orders` создаются после получения order события, но позиция обновляется из position события (может быть расхождение по времени)

#### Разрешение конфликтов

- Если position событие пришло раньше order события:
  - Позиция обновляется из position события
  - Когда придет order событие, создается `position_orders` с правильным `relationship_type`
  
- Если order событие пришло раньше position события:
  - Создается `position_orders` на основе текущего состояния позиции
  - Когда придет position событие, позиция обновляется, но `position_orders` уже создан

- В обоих случаях итоговое состояние будет корректным, так как position события - источник истины для позиций

### Вариант 2: Полное разделение с опросом БД

**order-manager**:
- Подписывается на `ws-gateway.order`
- Обновляет только `orders`
- Публикует в `order-manager.order_events` (для других сервисов)

**position-manager**:
- Подписывается на `ws-gateway.position`
- Периодически опрашивает таблицу `orders` для создания `position_orders`
- Обновляет позиции только из position событий

**Преимущества**:
- Полное разделение (нет подписки на order_events)
- position-manager независим от order-manager

**Недостатки**:
- Задержка в создании `position_orders`
- Дополнительная нагрузка на БД
- Сложнее синхронизировать данные

### Вариант 3: Отдельный сервис для связывания

**order-manager**:
- Подписывается на `ws-gateway.order`
- Обновляет только `orders`
- Публикует в `order-manager.order_events`

**position-manager**:
- Подписывается на `ws-gateway.position`
- Обновляет только позиции

**position-linker** (новый сервис):
- Подписывается на `order-manager.order_events` и `ws-gateway.position`
- Создает `position_orders` на основе событий
- Связывает ордера с позициями

**Преимущества**:
- Полное разделение order-manager и position-manager
- Отдельная ответственность за связывание

**Недостатки**:
- Усложнение архитектуры (еще один сервис)
- Больше точек отказа

## Выводы

### ✅ Разделение возможно, но с оговорками

**Положительные аспекты**:
- Упрощает архитектуру
- Четкое разделение ответственности
- position-manager использует данные напрямую от биржи

**Критические проблемы**:
1. **position_orders** - кто создает? (решение: position-manager на основе order событий)
2. **Связь ордеров с позициями** - как определить? (решение: position-manager создает связи)
3. **prediction_trading_results** - откуда данные? (решение: model-service подписывается на order_events)

### Рекомендуемый подход: Гибридный вариант

1. **order-manager**: только ордера, публикует события
2. **position-manager**: только позиции из position событий, но подписывается на order_events для создания `position_orders`
3. **model-service**: подписывается на order_events для получения детальной информации

**Ключевой момент**: position-manager подписывается на order_events, но **не обновляет позиции** из них, а только создает связи (`position_orders`). Позиции обновляются только из position событий от Bybit.

Это сохраняет разделение ответственности (order-manager - ордера, position-manager - позиции) и решает проблему создания `position_orders`.

