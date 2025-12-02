Техническое задание: Feature Service

1. Цель системы

Feature Service — выделенный сервис для:
1. приёма и хранения сырых маркет-данных, поступающих из ws-gateway,
2. расчёта онлайн-фичей для model-service,
3. пересборки оффлайн-фичей и датасетов для обучения моделей,
4. обеспечения идентичности фичей в online и offline режимах.
Сервис отделяет вычисление признаков от логики модели, позиций и ордеров.


---

2. Границы и обязанности

Feature Service обязан:
Принимать сырые потоки Bybit (через существующие очереди/шину событий от ws-gateway).
Поддерживать хранение raw data для последующей пересборки истории.
Реализовать Online Feature Engine: вычисление признаков в realtime.
Реализовать Offline Feature Engine + Dataset Builder: пересоздание фичей на исторических данных.
Публиковать фичи в очередь для model-service.
Предоставлять API для сборки датасета обучения (пакетно, по временным интервалам).
Обеспечить консистентность фичей (один и тот же код расчёта и параметры).



---

3. Получаемые данные от Bybit (через ws-gateway)

Сервис не подключается к Bybit сам; он подписывается на внутренние потоки:
Через внутренние очереди/шину (имена абстрактные, NEEDS CLARIFICATION):
market.orderbook.snapshot
market.orderbook.delta
market.trades
market.kline.1s / market.kline.1m
market.ticker
market.funding
(опционально) trading.executions — события исполнения собственных ордеров.


Требования к приёму:

Поддерживать sequence/order в orderbook:
— корректная сборка snapshot + delta,
— обработка рассинхронизации (рест запрос snapshot),
— хранение всех дельт для оффлайн восстановления.

Все сообщения снабжать внутренним timestamp + exchange timestamp.



---

4. Хранилище сырых данных

Требования:
Формат: Parquet/Columnar (оптимально для оффлайн пересборки).

Структуры:
raw_orderbook_snapshots
raw_orderbook_deltas
raw_trades
raw_kline
raw_ticker
raw_funding
raw_executions

Хранить минимум N дней (параметр в конфигурации); поддерживать архивирование.


---

5. Feature Engine (online)

Онлайн-движок считает фичи каждые 1s, 3s, 15s, 1m.
Фичи из расчёта подаются в model-service.

Базовые свечные/цено-фичи
mid_price = (best_bid + best_ask)/2
spread_abs, spread_rel
return_1s, return_3s, return_1m
VWAP_3s, VWAP_15s, VWAP_1m
vol_3s, vol_15s, vol_1m
volatility_1m, volatility_5m (std of returns)


Orderflow (трейды)
signed_volume_3s, 15s, 1m
buy/sell volume ratio
trade_count_3s
net_aggressor_pressure = buy_vol − sell_vol


Orderbook (top 5 / top 10)
depth_bid_top5 / depth_ask_top5
depth_imbalance_top5
depth_bid_top10 / depth_ask_top10
slope_bid, slope_ask (линейный коэффициент убывания объёмов с уровнями)
orderbook_churn_rate (частота изменений best price)
local_liquidity_density (объём в пределах X bps)


Перпетуальные параметры
funding_rate
time_to_funding (минуты до следующего расчёта)


Временные/мета признак
time_of_day (числовой)
rolling z-score (стандартизация последних N значений)


Формат online выдачи
[timestamp, symbol, feature_vector(N)]
Через внутреннюю очередь features.live и REST /features/latest?symbol=....


---

6. Offline Feature Engine + Dataset Builder

Этот же движок должен уметь работать в batch-режиме:

Функции:
Восстановить orderbook по snapshot + deltas за период.
Восстановить все rolling windows (1s–1m).
Посчитать фичи идентично online-версии.

Добавить target:
return_(t+1m) или return_(t+N)
direction classification labels (up/down/flat)

Собрать датасет в parquet batch-файлы.
Экспортировать по API /dataset/build?from=...&to=....

Гарантия идентичности:
Один код расчёта для offline и online режимов.
Один и тот же Feature Registry.



---

7. Feature Registry

YAML/JSON, описывающий:
имя фичи
входные источники (trades/orderbook/kline)
окно (3s, 15s, 1m…)
параметры нормализации
порядок расчёта

Нужен для детерминированного пересборки датасета.



---






8.4. Model-service должен быть переработан:

Убрать любую логику feature engineering из model-service.

model-service должен:
принимать готовый feature vector;
выполнять inference;
выполнять обучение только на датасетах, предоставленных Feature Service;
поддерживать модельные артефакты (weights, metadata);
иметь API /model/train → принимает путь к датасету (или id датасета).




---

9. API (кратко)

Public (внутренний) API Feature Service:
1. GET /features/latest?symbol= — последние онлайн-фичи.
2. POST /dataset/build — собрать датасет за период.
3. GET /dataset/list — список доступных датасетов.
4. POST /feature-registry/reload — перезагрузка конфигурации фичей.

Streaming:
очередь features.live — пуш онлайн-фичей.
очередь features.dataset.ready — уведомление о завершении сборки датасета.


---

10. Нефункциональные требования

Онлайн латентность расчёта фичей: ≤ 50–70 мс.
Возможность горизонтального масштабирования по символам.
Идентичность онлайн/оффлайн фичей: тесты деривации.
Стойкость к падению ws-gateway: восстановление из snapshot.
Логи: сырые данные, ошибки seq, пропуски, рассинхронизации.

