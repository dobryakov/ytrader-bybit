# Сравнение балансов: БД vs Bybit API

## Описание

Скрипт `compare_balances.py` сравнивает балансы, сохраненные в базе данных через WebSocket подписку, с актуальными балансами, полученными через Bybit REST API.

## Запуск

### В Docker контейнере (рекомендуется)

```bash
# Запустить скрипт внутри контейнера ws-gateway
docker compose exec ws-gateway python /app/compare_balances.py
```

### Локально (требует настройки окружения)

```bash
cd ws-gateway
python compare_balances.py
```

## Что делает скрипт

1. **Получает балансы из БД**: Извлекает последние записи из таблицы `account_balances` для каждой монеты
2. **Получает балансы через API**: Запрашивает актуальные балансы через Bybit REST API (`/v5/account/wallet-balance`)
3. **Сравнивает балансы**: Находит расхождения между БД и API
4. **Исследует причины**: Анализирует потенциальные причины расхождений:
   - Возраст записи в БД (старые данные)
   - Пропущенные события WebSocket
   - Ошибки парсинга балансов
   - Неправильный расчет frozen balance
   - Проблемы с подпиской на wallet события

## Вывод скрипта

Скрипт выводит:
- Количество балансов в БД и API
- Список всех расхождений с деталями
- Для каждого расхождения:
  - Разницу в wallet_balance, available_balance, frozen
  - Возраст записи в БД
  - Потенциальные причины расхождения
  - Рекомендации по исправлению

## Пример вывода

```
================================================================================
BALANCE COMPARISON RESULTS
================================================================================

Database balances: 3 coins
API balances: 3 coins
Discrepancies found: 1

⚠️  DISCREPANCIES DETECTED:

================================================================================
Coin: USDT
Issue: mismatch
--------------------------------------------------------------------------------
Database Wallet Balance:  1000.50000000
API Wallet Balance:       1000.75000000
Difference:               0.25000000

Database Available:       950.00000000
API Available:            950.25000000
Difference:               0.25000000

Database Frozen:           50.50000000
API Frozen:                50.50000000
Difference:               0.00000000

DB Record Age:            125.3 seconds

--------------------------------------------------------------------------------
INVESTIGATION:
--------------------------------------------------------------------------------

Potential Causes:
  1. Database record is 2.1 minutes old - WebSocket may have missed recent updates
  2. Significant wallet balance difference (0.25) - possible parsing error or missed transaction

Recommendations:
  1. Check WebSocket connection status and subscription activity
  2. Check WebSocket wallet message parsing - verify all fields are correctly extracted
================================================================================
```

## Возможные причины расхождений

### 1. Задержка обновления
- WebSocket события могут приходить с задержкой
- Если запись в БД старая (>5 минут), возможно WebSocket пропустил обновления

### 2. Пропущенные события
- WebSocket соединение могло разорваться
- Подписка на wallet могла быть отменена
- События могли быть потеряны при обработке

### 3. Ошибки парсинга
- Неправильное извлечение полей из WebSocket сообщения
- Неправильный расчет available_balance (walletBalance - locked)
- Проблемы с обработкой пустых строк в availableToWithdraw

### 4. Различия в структуре данных
- Bybit может изменить структуру WebSocket сообщений
- API и WebSocket могут использовать разные поля для одних и тех же данных

## Рекомендации по устранению

1. **Проверить статус WebSocket соединения**:
   ```bash
   docker compose logs ws-gateway | grep -i "websocket\|wallet\|balance"
   ```

2. **Проверить активность подписки на wallet**:
   ```bash
   docker compose exec postgres psql -U ytrader -d ytrader -c "SELECT * FROM subscriptions WHERE channel_type = 'balance' AND is_active = true;"
   ```

3. **Проверить последние записи балансов**:
   ```bash
   docker compose exec postgres psql -U ytrader -d ytrader -c "SELECT coin, wallet_balance, available_balance, frozen, received_at FROM account_balances ORDER BY received_at DESC LIMIT 10;"
   ```

4. **Проверить логи обработки балансов**:
   ```bash
   docker compose logs ws-gateway | grep -i "balance_persist\|balance_parse"
   ```

5. **Переподписаться на wallet события** (если подписка неактивна):
   ```bash
   curl -X POST http://localhost:4400/api/v1/subscriptions \
     -H "X-API-Key: YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"channel_type": "balance", "requesting_service": "manual-check"}'
   ```

## Автоматизация

Можно добавить скрипт в cron для периодической проверки:

```bash
# Проверка каждые 10 минут
*/10 * * * * docker compose exec -T ws-gateway python /app/compare_balances.py >> /var/log/balance_check.log 2>&1
```

