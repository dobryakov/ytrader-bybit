#!/bin/bash
# Демонстрация работы Feature Service

set -e

echo "=========================================="
echo "Демонстрация Feature Service"
echo "=========================================="
echo ""

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Проверка здоровья сервиса
echo -e "${BLUE}1. Проверка здоровья сервиса${NC}"
echo "GET http://localhost:4900/health"
curl -s http://localhost:4900/health | python3 -m json.tool
echo ""

# 2. Проверка очередей RabbitMQ
echo -e "${BLUE}2. Статус очередей RabbitMQ${NC}"
echo "Очереди ws-gateway.* (источник маркет-данных):"
docker compose exec -T rabbitmq rabbitmqadmin list queues name messages consumers 2>/dev/null | grep "ws-gateway" | head -5
echo ""
echo "Очередь features.live (публикация признаков):"
docker compose exec -T rabbitmq rabbitmqadmin list queues name messages consumers 2>/dev/null | grep "features.live"
echo ""

# 3. Проверка логов - получение маркет-данных
echo -e "${BLUE}3. Логи получения маркет-данных из RabbitMQ${NC}"
echo "Последние сообщения о потреблении из очередей:"
docker compose logs feature-service --tail 50 2>&1 | grep -E "(consuming_queue|market_data_event_processed)" | tail -5
echo ""

# 4. Проверка логов - вычисление признаков
echo -e "${BLUE}4. Логи вычисления признаков${NC}"
echo "Последние вычисления признаков:"
docker compose logs feature-service --tail 100 2>&1 | grep -E "(features_computed|feature_computation)" | tail -5
echo ""

# 5. Проверка логов - публикация признаков
echo -e "${BLUE}5. Логи публикации признаков в очередь${NC}"
echo "Последние публикации:"
docker compose logs feature-service --tail 100 2>&1 | grep -E "(feature_vector_published|features_computed_and_published)" | tail -5
echo ""

# 6. REST API - получение признаков
echo -e "${BLUE}6. REST API - получение признаков${NC}"
echo "GET http://localhost:4900/features/latest?symbol=BTCUSDT"
API_KEY=$(grep FEATURE_SERVICE_API_KEY .env 2>/dev/null | cut -d'=' -f2 || echo "test-api-key")
curl -s -H "X-API-Key: ${API_KEY}" "http://localhost:4900/features/latest?symbol=BTCUSDT" | python3 -m json.tool 2>&1 | head -30
echo ""

# 7. Статистика очередей
echo -e "${BLUE}7. Статистика сообщений в очередях${NC}"
echo "Сообщений в ws-gateway.* (входящие маркет-данные):"
docker compose exec -T rabbitmq rabbitmqadmin list queues name messages 2>/dev/null | grep "ws-gateway" | awk '{print "  " $1 ": " $2 " сообщений"}' | head -5
echo ""
echo "Сообщений в features.live (вычисленные признаки):"
docker compose exec -T rabbitmq rabbitmqadmin list queues name messages 2>/dev/null | grep "features.live" | awk '{print "  " $1 ": " $2 " сообщений"}'
echo ""

echo -e "${GREEN}Демонстрация завершена!${NC}"
echo ""
echo "Для просмотра логов в реальном времени:"
echo "  docker compose logs -f feature-service"
echo ""
echo "Для проверки очередей:"
echo "  docker compose exec rabbitmq rabbitmqadmin list queues"

