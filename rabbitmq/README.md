# Конфигурация RabbitMQ для уменьшения персистентности на диск

## Применённые настройки

### 1. Политики Lazy Queues
Применены политики для очередей `ws-gateway.*` и `features.(live|computed)`, которые хранят сообщения в памяти и реже пишут на диск:

```bash
# Политика для ws-gateway очередей
rabbitmqctl set_policy lazy-ws-gateway "^ws-gateway\." '{"queue-mode":"lazy"}' --apply-to queues --priority 1

# Политика для features очередей
rabbitmqctl set_policy lazy-features "^features\.(live|computed)" '{"queue-mode":"lazy"}' --apply-to queues --priority 1
```

### 2. Параметры через переменные окружения

В `docker-compose.yml` настроены следующие параметры:

- `RABBITMQ_DISK_FREE_LIMIT=50MB` - минимальный свободный диск
- `RABBITMQ_QUEUE_INDEX_MAX_JOURNAL_ENTRIES=32768` - увеличивает размер журнала перед синхронизацией на диск
- `RABBITMQ_QUEUE_INDEX_EMBED_MSGS_BELOW=4096` - размер сообщений, встраиваемых в индекс (в байтах)
- `RABBITMQ_HEARTBEAT=60` - интервал heartbeat для соединений (в секундах)
- `RABBITMQ_LOG_LEVEL=warning` - уровень логирования (уменьшает запись логов на диск)

**Важные замечания**:
- Переменная `RABBITMQ_VM_MEMORY_HIGH_WATERMARK` устарела в новых версиях RabbitMQ (начиная с версии 3.8+) и была удалена из конфигурации. Согласно официальной документации RabbitMQ, настройки памяти должны управляться через конфигурационный файл `rabbitmq.conf` с использованием параметра `vm_memory_high_watermark.relative` или `vm_memory_high_watermark.absolute`.
- Конфигурационный файл `rabbitmq/config/rabbitmq.conf` в настоящее время не используется, так как вызывает ошибки `failed_to_prepare_configuration` в версии `rabbitmq:3-management-alpine`. Используются только переменные окружения, которые стабильно работают.
- Настройки памяти оставлены по умолчанию (40% от доступной памяти).

### 3. Проверка настроек

Проверить применённые настройки:

```bash
# Проверить параметры окружения
docker compose exec rabbitmq rabbitmqctl environment | grep -E "vm_memory|disk_free|queue_index|heartbeat"

# Проверить политики
docker compose exec rabbitmq rabbitmqctl list_policies

# Проверить состояние очередей
docker compose exec rabbitmq rabbitmqctl list_queues name arguments
```

## Важные замечания

⚠️ **Внимание**: Lazy queues хранят сообщения в памяти и могут быть потеряны при перезапуске RabbitMQ. Это подходит для:
- Временных данных (ticker, kline)
- Данных, которые можно восстановить (market data)
- Не критичных данных

❌ **НЕ используйте lazy queues для**:
- Критичных данных (orders, positions)
- Данных, которые нельзя потерять

## Восстановление политик после перезапуска

Политики сохраняются в базе данных RabbitMQ, но если они были удалены, их можно восстановить:

```bash
docker compose exec rabbitmq /app/rabbitmq/scripts/apply_lazy_queues.sh
```

Или вручную:

```bash
docker compose exec rabbitmq rabbitmqctl set_policy lazy-ws-gateway "^ws-gateway\." '{"queue-mode":"lazy"}' --apply-to queues --priority 1
docker compose exec rabbitmq rabbitmqctl set_policy lazy-features "^features\.(live|computed)" '{"queue-mode":"lazy"}' --apply-to queues --priority 1
```

