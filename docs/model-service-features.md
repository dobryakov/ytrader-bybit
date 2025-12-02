## Фактические данные, которые получает модель

### Данные из execution events (RabbitMQ очередь `order-manager.order_events`)

Из очереди приходят:
- `order.order_id`, `order.signal_id`, `order.asset`, `order.side`
- `order.average_price` (execution_price)
- `order.filled_quantity` (execution_quantity)
- `order.fees` (execution_fees)
- `order.executed_at`, `order.created_at` (signal_timestamp)
- `market_conditions.spread`, `market_conditions.volume_24h`, `market_conditions.volatility`
- `signal.price` (signal_price)
- `signal.strategy_id`

### Что реально используется для признаков (23 из 43)

#### Доступные признаки (23):

1. Ценовые (2):
   - `price` = `signal_price` (из execution event)
   - `price_log` = log(signal_price)

2. Спред (2):
   - `spread` = `market_conditions.spread` (из execution event)
   - `spread_percent` = (spread / price) * 100

3. Объем (2):
   - `volume_24h` = `market_conditions.volume_24h` (из execution event)
   - `volume_24h_log` = log(volume_24h + 1)

4. Волатильность (2):
   - `volatility` = `market_conditions.volatility` (из execution event)
   - `volatility_squared` = volatility²

5. Глубина стакана (3) — всегда 0:
   - `bid_depth` = 0.0
   - `ask_depth` = 0.0
   - `depth_imbalance` = 0.0

6. Технические индикаторы (9) — дефолтные значения:
   - `rsi` = 50.0
   - `macd` = 0.0
   - `macd_signal` = 0.0
   - `macd_histogram` = 0.0
   - `moving_average_20` = price
   - `moving_average_50` = price
   - `bollinger_upper` = price
   - `bollinger_lower` = price
   - `bollinger_width` = 0.0

7. Признаки исполнения (4):
   - `execution_price` = `order.average_price`
   - `execution_quantity` = `order.filled_quantity`
   - `execution_fees` = `order.fees`
   - `execution_value` = execution_price × execution_quantity

8. Проскальзывание (4):
   - `slippage` = execution_price - signal_price
   - `slippage_percent` = (slippage / signal_price) * 100
   - `price_change` = execution_price - signal_price
   - `price_change_percent` = (price_change / signal_price) * 100

9. Временные (2):
   - `execution_delay_seconds` = executed_at - signal_timestamp
   - `execution_delay_minutes` = delay_seconds / 60

10. Категориальные (4):
    - `side_buy` = 1 если buy, иначе 0
    - `side_sell` = 1 если sell, иначе 0
    - `asset_hash` = hash(asset) % 1000
    - `strategy_hash` = hash(strategy_id) % 100

11. Рыночные условия на момент исполнения (3):
    - `execution_spread` = `market_conditions.spread`
    - `execution_volume_24h` = `market_conditions.volume_24h`
    - `execution_volatility` = `market_conditions.volatility`

12. Изменения рыночных условий (3) — всегда 0:
    - `spread_change` = 0 (т.к. market_snapshot.spread = market_conditions.spread)
    - `volume_change` = 0
    - `volatility_change` = 0

13. Открытые ордера (3) — всегда 0:
    - `open_orders_count` = 0
    - `pending_buy_orders` = 0
    - `pending_sell_orders` = 0

### Итого: 23 реально используемых признака

- 20 признаков с реальными данными
- 3 признака всегда 0 (глубина стакана)
- 9 признаков с дефолтными значениями (технические индикаторы)
- 3 признака всегда 0 (изменения рыночных условий)
- 3 признака всегда 0 (открытые ордера)

### Проблема

`market_data_snapshot` из `trading_signals` не извлекается при обучении, поэтому:
- Нет технических индикаторов (RSI, MACD, MA, Bollinger)
- Нет глубины стакана (bid_depth, ask_depth)
- Нет различия между market_conditions на момент решения и на момент исполнения

В логах видно: `"Signal market data not available, using execution event market_conditions"` — используется только то, что приходит в execution event из очереди.
