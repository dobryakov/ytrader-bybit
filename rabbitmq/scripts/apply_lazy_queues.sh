#!/bin/bash
# Скрипт для применения политик lazy queues к существующим очередям
# Lazy queues хранят сообщения в памяти и реже пишут на диск

RABBITMQ_USER="${RABBITMQ_USER:-guest}"
RABBITMQ_PASSWORD="${RABBITMQ_PASSWORD:-guest}"

# Применяем политику lazy queues ко всем очередям ws-gateway.*
# Эти очереди не критичны и могут быть потеряны при перезапуске
rabbitmqctl set_policy lazy-ws-gateway \
  "^ws-gateway\." \
  '{"queue-mode":"lazy"}' \
  --apply-to queues \
  --priority 1 || true

# Применяем политику к очередям features.* (кроме dataset.ready)
rabbitmqctl set_policy lazy-features \
  "^features\.(live|computed)" \
  '{"queue-mode":"lazy"}' \
  --apply-to queues \
  --priority 1 || true

echo "Lazy queue policies applied"

