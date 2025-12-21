# Глубокий анализ проблемы с prediction_trading_results

## Дата анализа
2025-12-21

## Проблема
Модель делает правильные предсказания направления движения цены (100% точность), но все позиции показывают убыток. При этом:
- `exit_price` часто равен `entry_price` или NULL
- `realized_pnl` берется из общей позиции, а не из отдельных сделок
- Одна позиция связана с множеством предсказаний (885 записей для одной позиции)

## Путь данных от Bybit до prediction_trading_results

### 1. Источники данных от Bybit

#### 1.1. WebSocket события (ws-gateway)
**Канал**: `order` и `position`

**Данные от Bybit**:
- `order` события содержат:
  - `orderId` (bybit_order_id)
  - `orderStatus` (New, PartiallyFilled, Filled, Cancelled, Rejected)
  - `cumExecQty` (накопленное исполненное количество)
  - `avgPrice` (средняя цена исполнения)
  - `cumExecFee` (накопленные комиссии)
  - **НЕТ realized_pnl** в order событиях

- `position` события содержат:
  - `symbol` (торговая пара)
  - `size` (размер позиции)
  - `avgPrice` или `avgEntryPrice` (средняя цена входа)
  - `unrealisedPnl` (нереализованная прибыль)
  - `realisedPnl` (реализованная прибыль) - **ЭТО КУМУЛЯТИВНОЕ ЗНАЧЕНИЕ**
  - `markPrice` (маркировочная цена)
  - `mode` (one-way, hedge)

**Важно**: `realisedPnl` в position событиях - это **кумулятивная реализованная прибыль** для всей позиции, а не для отдельной сделки.

#### 1.2. REST API (для синхронизации)
- Используется для периодической синхронизации позиций
- Возвращает те же данные, что и WebSocket position события

### 2. Обработка в ws-gateway

**Файл**: `ws-gateway/src/services/positions/position_event_normalizer.py`

**Что происходит**:
1. Парсинг WebSocket событий от Bybit
2. Нормализация данных (извлечение `unrealisedPnl`, `realisedPnl`, `avgPrice`, `markPrice`)
3. Публикация в RabbitMQ очередь `ws-gateway.position`

**Данные передаются как есть** - без вычислений на нашей стороне.

### 3. Обработка в order-manager

**Файл**: `order-manager/src/services/event_subscriber.py`

**Что происходит**:
1. Получение order событий из `ws-gateway.order`
2. Обновление статуса ордера в БД (`orders` таблица)
3. Извлечение `avgPrice` и `cumExecQty` из события
4. Публикация обогащенного события в `order-manager.order_events`

**Ключевой момент**: `execution_price` берется из `avgPrice` в событии от Bybit, а не вычисляется.

**Файл**: `order-manager/src/services/position_manager.py` (старый, используется только для создания position_orders)

**Что происходит**:
1. При обновлении позиции создается запись в `position_orders`:
   - `execution_price` = `execution_price` из параметров (из order события)
   - `relationship_type` = определяется логикой (`opened`, `increased`, `decreased`, `closed`, `reversed`)
   - `size_delta` = изменение размера позиции
   - `executed_at` = время исполнения

**Важно**: `position_orders.execution_price` - это цена из order события, которая может быть `avgPrice` (средняя цена исполнения для всего ордера).

### 4. Обработка в position-manager

**Файл**: `position-manager/src/consumers/order_position_consumer.py`

**Что происходит**:
1. Получение событий из `order-manager.order_events`
2. Извлечение `execution_price` из события (приоритет: `average_price` > `price` > `execution_price`)
3. Вызов `PositionManager.update_position_from_order_fill()`

**Файл**: `position-manager/src/services/position_manager.py`

**Метод**: `update_position_from_order_fill()`

**Что происходит**:
1. Вычисление `realized_pnl_delta` для закрываемой части позиции:
   ```python
   if current_size > 0:  # Long
       realized_pnl_delta = (execution_price - current_avg_price) * closed_qty - fees
   else:  # Short
       realized_pnl_delta = (current_avg_price - execution_price) * closed_qty - fees
   ```
2. Обновление `positions.realized_pnl` = `positions.realized_pnl + realized_pnl_delta`
3. Обновление `positions.size`, `average_entry_price`

**Проблема**: `realized_pnl_delta` вычисляется на основе `execution_price` из order события, который может быть `avgPrice` (средняя цена для всего ордера), а не реальной ценой исполнения конкретной части.

**Файл**: `position-manager/src/consumers/websocket_position_consumer.py`

**Что происходит**:
1. Получение position событий из `ws-gateway.position`
2. Извлечение `realisedPnl` из события (кумулятивное значение от Bybit)
3. Вызов `PositionManager.update_position_from_websocket()`

**Метод**: `update_position_from_websocket()`

**Что происходит**:
1. Обновление `positions.realized_pnl` = `realisedPnl` из события (прямая подстановка)
2. Обновление `positions.unrealized_pnl` = `unrealisedPnl` из события
3. Обновление `positions.average_entry_price` = `avgPrice` из события (если присутствует)

**Ключевой момент**: `realized_pnl` в позиции - это **кумулятивное значение от Bybit**, а не вычисленное на нашей стороне.

### 5. Обработка в model-service

**Файл**: `model-service/src/main.py`

**Метод**: `handle_execution_event()`

**Что происходит**:
1. Получение `OrderExecutionEvent` из RabbitMQ
2. Поиск ордера в БД по `bybit_order_id`
3. Получение данных из `position_orders`:
   ```python
   correct_execution_price = (
       position_order_row["position_order_execution_price"] or
       position_order_row["order_price"] or
       position_order_row["order_average_price"] or
       Decimal(str(event.execution_price))
   )
   ```
4. Получение `position.average_entry_price` из таблицы `positions`
5. Вычисление `realized_pnl_delta`:
   ```python
   if is_long:
       realized_pnl_delta = (exit_price - entry_price) * closed_quantity - execution_fees
   else:
       realized_pnl_delta = (entry_price - exit_price) * closed_quantity - execution_fees
   ```
6. Создание/обновление `prediction_trading_results`:
   - `entry_price` = `correct_execution_price` (при создании) или `position.average_entry_price` (при обновлении)
   - `exit_price` = `correct_execution_price` (только если `relationship_type == "closed"`)
   - `realized_pnl` = накапливается через `realized_pnl_delta`

**Проблемы**:

1. **exit_price заполняется только при `relationship_type == "closed"`**:
   - Если позиция закрывается несколькими ордерами, только последний получит `exit_price`
   - Если позиция закрывается встречным ордером (не через exit signal), `exit_price` может быть NULL

2. **realized_pnl вычисляется на основе `execution_price` из position_orders**:
   - `execution_price` в `position_orders` - это `avgPrice` из order события (средняя цена для всего ордера)
   - Если ордер исполнялся по разным ценам, `avgPrice` не отражает реальную цену закрытия конкретной части позиции

3. **entry_price может быть неправильным**:
   - При создании используется `correct_execution_price` (цена из order события)
   - При обновлении используется `position.average_entry_price` (может быть из WebSocket события или вычислено)
   - Если позиция открывалась несколькими ордерами, `average_entry_price` - это средневзвешенная цена, а не цена конкретного предсказания

4. **Одна позиция связана с множеством предсказаний**:
   - Каждое предсказание создает свою запись в `prediction_trading_results`
   - Все записи связаны с одной позицией через `signal_id` → `position_orders` → `position_id`
   - При закрытии позиции все записи получают одинаковый `realized_pnl` из общей позиции

## Анализ текущей реализации

### Проблема 1: exit_price = entry_price

**Причина**:
- `exit_price` заполняется только при `relationship_type == "closed"` (строка 164 в `prediction_trading_linker.py`)
- Если позиция закрывается встречным ордером (не через exit signal), `relationship_type` может быть не "closed"
- В примерах из API: `exit_price = 2899.275`, `entry_price = 2899.275` - это означает, что `exit_price` был установлен при создании записи (использовался `correct_execution_price`), а не при закрытии

**Решение**:
- Заполнять `exit_price` при любом закрытии позиции (`relationship_type in ("closed", "decreased")` и `new_size == 0`)
- Использовать цену из `position_orders.execution_price` для закрывающего ордера

### Проблема 2: realized_pnl берется из общей позиции

**Причина**:
- `realized_pnl` в `prediction_trading_results` вычисляется как накопление `realized_pnl_delta` для каждого ордера
- Но `realized_pnl_delta` вычисляется на основе `position.average_entry_price`, который может быть средневзвешенной ценой для всей позиции
- Если позиция открывалась несколькими ордерами с разными предсказаниями, все предсказания будут использовать одну и ту же `average_entry_price`

**Решение**:
- Использовать `entry_price` из самой записи `prediction_trading_results` для вычисления `realized_pnl_delta`
- Не использовать `position.average_entry_price` для обновления существующих записей

### Проблема 3: execution_price может быть avgPrice

**Причина**:
- `execution_price` в `position_orders` берется из `avgPrice` в order событии от Bybit
- `avgPrice` - это средняя цена исполнения для всего ордера
- Если ордер исполнялся по разным ценам, `avgPrice` не отражает реальную цену закрытия конкретной части позиции

**Решение**:
- Использовать данные из Bybit REST API для получения детальной информации об исполнении ордера
- Или использовать `realisedPnl` из position событий для вычисления реальной прибыли

## Возможные решения

### Решение 1: Использовать realized_pnl напрямую из Bybit

**Идея**: Вместо вычисления `realized_pnl` на нашей стороне, использовать кумулятивное значение из position событий и распределять его между предсказаниями.

**Плюсы**:
- Используем данные напрямую от биржи (источник истины)
- Не нужно вычислять PnL самостоятельно
- Учитывает все факторы (комиссии, проскальзывание, частичное исполнение)

**Минусы**:
- `realisedPnl` в position событиях - кумулятивное значение для всей позиции
- Нужно распределять его между несколькими предсказаниями
- Нельзя точно определить, какая часть прибыли относится к какому предсказанию

**Реализация**:
1. При получении position события с `realisedPnl`:
   - Найти все открытые `prediction_trading_results` для этой позиции
   - Вычислить долю каждого предсказания в позиции (по `position_size_at_entry`)
   - Распределить `realisedPnl` пропорционально долям
   - Обновить `realized_pnl` в каждой записи

2. При закрытии позиции:
   - Использовать финальное значение `realisedPnl` из последнего position события
   - Распределить его между всеми предсказаниями

### Решение 2: Использовать детальные данные об исполнении из Bybit REST API

**Идея**: Запрашивать детальную информацию об исполнении ордера через Bybit REST API (`/v5/execution/list`).

**Плюсы**:
- Получаем точные цены исполнения для каждой части ордера
- Можем точно вычислить PnL для каждого предсказания
- Учитываем реальные цены исполнения, а не средние

**Минусы**:
- Дополнительные запросы к API (rate limits)
- Задержка в получении данных
- Нужно обрабатывать случаи, когда API недоступен

**Реализация**:
1. При получении order события с `status == "filled"`:
   - Запросить детальную информацию об исполнении через Bybit REST API
   - Получить список исполнений с точными ценами и количествами
   - Для каждого исполнения вычислить `realized_pnl_delta` на основе реальной цены

2. Обновить `prediction_trading_results` с точными данными

### Решение 3: Комбинированный подход

**Идея**: Использовать данные из Bybit для валидации и коррекции наших вычислений.

**Плюсы**:
- Сохраняем возможность вычислять PnL в реальном времени
- Используем данные от биржи для проверки и коррекции
- Более надежное решение

**Минусы**:
- Более сложная реализация
- Нужно обрабатывать расхождения между нашими вычислениями и данными от биржи

**Реализация**:
1. Вычислять `realized_pnl` на нашей стороне (как сейчас)
2. При получении position события с `realisedPnl`:
   - Сравнить наше вычисленное значение с значением от Bybit
   - Если расхождение > порога, скорректировать наши значения
   - Распределить скорректированное значение между предсказаниями

3. Периодически синхронизировать данные через REST API

## Рекомендации

### Краткосрочные (быстрые исправления)

1. **Исправить заполнение exit_price**:
   - Заполнять `exit_price` при любом закрытии позиции, а не только при `relationship_type == "closed"`
   - Использовать цену из `position_orders.execution_price` для закрывающего ордера

2. **Исправить вычисление realized_pnl**:
   - Использовать `entry_price` из самой записи `prediction_trading_results` для вычисления `realized_pnl_delta`
   - Не использовать `position.average_entry_price` для обновления существующих записей

3. **Исправить логику создания записей**:
   - Создавать запись в `prediction_trading_results` только при первом входе в позицию
   - Обновлять существующую запись при последующих ордерах, а не создавать новые

### Среднесрочные (улучшения)

1. **Использовать realized_pnl из position событий**:
   - При получении position события с `realisedPnl`, распределять его между предсказаниями
   - Использовать пропорциональное распределение на основе `position_size_at_entry`

2. **Добавить синхронизацию с Bybit REST API**:
   - Периодически запрашивать детальную информацию об исполнении ордеров
   - Корректировать наши вычисления на основе данных от биржи

### Долгосрочные (архитектурные изменения)

1. **Переработать связь между предсказаниями и позициями**:
   - Один `prediction_target` → одна запись в `prediction_trading_results`
   - Связывать через `signal_id` → `order_id` → `position_orders` → `position_id`
   - Отслеживать, какая часть позиции относится к какому предсказанию

2. **Использовать event sourcing**:
   - Сохранять все события исполнения ордеров
   - Вычислять состояние на основе событий
   - Легче отслеживать изменения и исправлять ошибки

## Выводы

1. **Основная проблема**: `realized_pnl` вычисляется на основе неточных данных (`avgPrice` вместо реальных цен исполнения)

2. **Вторичная проблема**: `exit_price` не заполняется правильно при закрытии позиции

3. **Архитектурная проблема**: Одна позиция связана с множеством предсказаний, что затрудняет точное вычисление PnL для каждого предсказания

4. **Рекомендуемое решение**: Комбинированный подход:
   - Использовать `realisedPnl` из position событий для валидации
   - Вычислять `realized_pnl` на нашей стороне для реального времени
   - Периодически синхронизировать через REST API для точности

