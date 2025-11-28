# Quickstart: Model Service

**Feature**: Model Service - Trading Decision and ML Training Microservice  
**Date**: 2025-01-27

## Overview

This guide provides step-by-step instructions to set up and run the Model Service locally for development and testing.

## Prerequisites

- Docker and Docker Compose V2 installed
- PostgreSQL database (shared database, managed by ws-gateway)
- RabbitMQ message queue (for order execution events and trading signals)
- Order Manager microservice (for order execution events)
- WebSocket Gateway service (for market data, optional)

## Quick Setup

### 1. Clone and Navigate

```bash
cd /home/ubuntu/ytrader
git checkout 001-model-service
```

### 2. Configure Environment

Copy the example environment file and configure:

```bash
cp env.example .env
```

Edit `.env` and set the following variables:

```bash
# Database Configuration (shared PostgreSQL)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ytrader
POSTGRES_USER=ytrader
POSTGRES_PASSWORD=your_password

# RabbitMQ Configuration
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest

# Model Service Configuration
MODEL_SERVICE_PORT=4500
MODEL_SERVICE_API_KEY=your-model-service-api-key  # For REST API authentication
MODEL_SERVICE_LOG_LEVEL=INFO
MODEL_SERVICE_SERVICE_NAME=model-service

# Model Storage Configuration
MODEL_STORAGE_PATH=/models  # Directory for model files (mounted as Docker volume)

# Training Configuration
MODEL_TRAINING_MIN_DATASET_SIZE=1000  # Minimum records for training
MODEL_TRAINING_MAX_DURATION_SECONDS=1800  # 30 minutes max training time
MODEL_QUALITY_THRESHOLD_ACCURACY=0.75  # Minimum accuracy for model activation
MODEL_RETRAINING_SCHEDULE=cron(0 2 * * *)  # Daily at 2 AM (optional)

# Signal Generation Configuration
SIGNAL_GENERATION_RATE_LIMIT=60  # Signals per minute
SIGNAL_GENERATION_BURST_ALLOWANCE=10  # Burst allowance for rate limiting
WARMUP_MODE_ENABLED=true  # Enable warm-up mode when no trained model exists
WARMUP_SIGNAL_FREQUENCY=60  # Warm-up signals per minute

# Strategy Configuration
TRADING_STRATEGIES=momentum_v1,mean_reversion_v1  # Comma-separated strategy IDs
```

### 3. Start Dependencies

Start PostgreSQL and RabbitMQ using Docker Compose:

```bash
docker compose up -d postgres rabbitmq
```

Wait for services to be ready:

```bash
docker compose logs -f postgres rabbitmq
# Press Ctrl+C when services are ready
```

### 4. Run Database Migrations

**Note**: Per constitution principle II (Shared Database Strategy), the `ws-gateway` service is the single source of truth for all PostgreSQL migrations. Model service migrations are located in `ws-gateway/migrations/` (e.g., `003_create_model_versions_table.sql`, `004_create_model_quality_metrics_table.sql`).

```bash
# Migrations should be run from ws-gateway service
docker compose run --rm ws-gateway python -m migrations.run
```

Or if migrations are run automatically on startup, skip this step.

### 5. Create Model Storage Directory

```bash
mkdir -p /home/ubuntu/ytrader/model-service/models
chmod 755 /home/ubuntu/ytrader/model-service/models
```

### 6. Start Model Service

```bash
docker compose up -d model-service
```

### 7. Verify Service Health

```bash
curl http://localhost:4500/health
```

Expected response:

```json
{
  "status": "healthy",
  "timestamp": "2025-01-27T10:00:00Z",
  "details": {
    "database": "connected",
    "message_queue": "connected",
    "model_storage": "accessible",
    "active_models": 0
  }
}
```

## Basic Usage

### Check Training Status

```bash
curl -X GET "http://localhost:4500/api/v1/training/status" \
  -H "X-API-Key: your-model-service-api-key"
```

Response:

```json
{
  "is_training": false,
  "last_training": null,
  "next_scheduled_training": "2025-01-28T02:00:00Z"
}
```

### Trigger Manual Model Training

Trigger training for a strategy:

```bash
curl -X POST http://localhost:4500/api/v1/training/trigger \
  -H "X-API-Key: your-model-service-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "momentum_v1",
    "training_type": "full"
  }'
```

Response:

```json
{
  "status": "triggered",
  "training_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Training started for strategy momentum_v1"
}
```

### List Model Versions

```bash
curl -X GET "http://localhost:4500/api/v1/models?strategy_id=momentum_v1" \
  -H "X-API-Key: your-model-service-api-key"
```

Response:

```json
{
  "models": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "version": "v1",
      "model_type": "xgboost",
      "strategy_id": "momentum_v1",
      "trained_at": "2025-01-27T10:00:00Z",
      "training_duration_seconds": 1200,
      "training_dataset_size": 50000,
      "is_active": true,
      "is_warmup_mode": false,
      "quality_metrics_summary": {
        "accuracy": 0.82,
        "sharpe_ratio": 1.5
      }
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

### Get Model Version Details

```bash
curl -X GET "http://localhost:4500/api/v1/models/v1" \
  -H "X-API-Key: your-model-service-api-key"
```

### Activate Model Version

```bash
curl -X POST "http://localhost:4500/api/v1/models/v1" \
  -H "X-API-Key: your-model-service-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "momentum_v1",
    "force": false
  }'
```

### Get Model Quality Metrics

```bash
curl -X GET "http://localhost:4500/api/v1/models/v1/metrics" \
  -H "X-API-Key: your-model-service-api-key"
```

Response:

```json
{
  "model_version_id": "550e8400-e29b-41d4-a716-446655440000",
  "version": "v1",
  "metrics": [
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "metric_name": "accuracy",
      "metric_value": 0.82,
      "metric_type": "classification",
      "evaluated_at": "2025-01-27T10:00:00Z",
      "evaluation_dataset_size": 10000
    },
    {
      "id": "660e8400-e29b-41d4-a716-446655440002",
      "metric_name": "sharpe_ratio",
      "metric_value": 1.5,
      "metric_type": "trading_performance",
      "evaluated_at": "2025-01-27T10:00:00Z"
    }
  ]
}
```

### Consume Trading Signals from RabbitMQ

Trading signals are published to RabbitMQ queue `model-service.trading_signals`. To consume signals:

```bash
# Install RabbitMQ client tools (if not already installed)
docker compose exec rabbitmq rabbitmqadmin list queues

# Consume signals (example using Python)
python -c "
import pika
import json

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', 5672))
channel = connection.channel()
channel.queue_declare(queue='model-service.trading_signals')

def callback(ch, method, properties, body):
    signal = json.loads(body.decode())
    print(f'Received signal: {signal[\"signal_type\"]} {signal[\"asset\"]} amount={signal[\"amount\"]} confidence={signal[\"confidence\"]}')

channel.basic_consume(queue='model-service.trading_signals', on_message_callback=callback, auto_ack=True)
print('Waiting for trading signals...')
channel.start_consuming()
"
```

### Publish Order Execution Events (for Testing)

To test model training, you can publish order execution events to the queue consumed by the model service:

```bash
# Publish test execution event (example using Python)
python -c "
import pika
import json
from datetime import datetime

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost', 5672))
channel = connection.channel()
channel.queue_declare(queue='order-manager.order_events')

event = {
    'event_id': 'test-event-001',
    'order_id': 'test-order-001',
    'signal_id': 'test-signal-001',
    'strategy_id': 'momentum_v1',
    'asset': 'BTCUSDT',
    'side': 'buy',
    'execution_price': 50000.00,
    'execution_quantity': 0.1,
    'execution_fees': 5.00,
    'executed_at': datetime.utcnow().isoformat() + 'Z',
    'signal_price': 50010.00,
    'signal_timestamp': datetime.utcnow().isoformat() + 'Z',
    'market_conditions': {
        'spread': 1.00,
        'volume_24h': 1000000.00,
        'volatility': 0.02
    },
    'performance': {
        'slippage': -10.00,
        'slippage_percent': -0.02,
        'realized_pnl': 50.00,
        'return_percent': 0.01
    }
}

channel.basic_publish(exchange='', routing_key='order-manager.order_events', body=json.dumps(event))
print('Published execution event')
connection.close()
"
```

## Development Workflow

### Running Tests

Unit tests (run in service container):

```bash
docker compose run --rm model-service pytest tests/unit -v
```

Integration tests (run in test container):

```bash
docker compose run --rm test-model-service pytest tests/integration -v
```

End-to-end tests:

```bash
docker compose run --rm test-model-service pytest tests/e2e -v
```

### Viewing Logs

```bash
# All logs
docker compose logs -f model-service

# Last 100 lines
docker compose logs --tail 100 model-service

# Filter by trace ID
docker compose logs model-service | grep "trace_id=abc123"

# Filter by strategy
docker compose logs model-service | grep "strategy_id=momentum_v1"
```

### Database Access

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U ytrader -d ytrader

# View model versions
SELECT version, model_type, strategy_id, trained_at, is_active FROM model_versions ORDER BY trained_at DESC;

# View quality metrics
SELECT mv.version, mqm.metric_name, mqm.metric_value, mqm.evaluated_at
FROM model_quality_metrics mqm
JOIN model_versions mv ON mqm.model_version_id = mv.id
WHERE mv.version = 'v1'
ORDER BY mqm.evaluated_at DESC;

# View active models
SELECT version, strategy_id, trained_at FROM model_versions WHERE is_active = true;
```

### Model File Access

```bash
# List model files
ls -lh /home/ubuntu/ytrader/model-service/models/

# View model directory structure
find /home/ubuntu/ytrader/model-service/models -type f -name "*.json" -o -name "*.pkl"

# Check model file size
du -sh /home/ubuntu/ytrader/model-service/models/*
```

### RabbitMQ Management

Access RabbitMQ management UI:

```bash
# Open in browser
open http://localhost:15672
# Default credentials: guest/guest
```

Or use CLI:

```bash
# List queues
docker compose exec rabbitmq rabbitmqadmin list queues

# Check queue depth
docker compose exec rabbitmq rabbitmqadmin list queues name messages consumers

# Purge queue (for testing)
docker compose exec rabbitmq rabbitmqadmin purge queue name=model-service.trading_signals
```

## Troubleshooting

### Service Not Starting

1. Check service logs:
   ```bash
   docker compose logs model-service
   ```

2. Verify environment variables:
   ```bash
   docker compose exec model-service env | grep MODEL
   ```

3. Check database connection:
   ```bash
   docker compose logs model-service | grep -i "postgres\|database"
   ```

### Model Training Failing

1. Check training logs:
   ```bash
   docker compose logs model-service | grep -i "training\|model"
   ```

2. Verify sufficient training data:
   ```bash
   # Check execution events in queue
   docker compose exec rabbitmq rabbitmqadmin list queues name messages | grep execution_events
   ```

3. Check model storage permissions:
   ```bash
   docker compose exec model-service ls -la /models
   ```

### Signals Not Being Generated

1. Check if model is active:
   ```bash
   curl -X GET "http://localhost:4500/api/v1/models?is_active=true" \
     -H "X-API-Key: your-model-service-api-key"
   ```

2. Verify warm-up mode status:
   ```bash
   docker compose logs model-service | grep -i "warmup\|warm-up"
   ```

3. Check rate limiting:
   ```bash
   docker compose logs model-service | grep -i "rate limit\|throttle"
   ```

### Database Connection Issues

1. Check PostgreSQL is running:
   ```bash
   docker compose ps postgres
   ```

2. Verify connection string:
   ```bash
   docker compose logs model-service | grep -i "postgres\|database"
   ```

3. Test connection:
   ```bash
   docker compose exec postgres psql -U ytrader -d ytrader -c "SELECT COUNT(*) FROM model_versions;"
   ```

### Message Queue Issues

1. Check RabbitMQ connection:
   ```bash
   docker compose logs model-service | grep -i "rabbitmq\|queue"
   ```

2. Verify queues exist:
   ```bash
   docker compose exec rabbitmq rabbitmqadmin list queues name
   ```

3. Check queue consumers:
   ```bash
   docker compose exec rabbitmq rabbitmqadmin list queues name consumers
   ```

## Production Deployment

### Environment Configuration

1. Use strong API keys and secrets
2. Configure proper logging levels (`MODEL_SERVICE_LOG_LEVEL=WARNING`)
3. Use secure database credentials
4. Enable SSL/TLS for RabbitMQ and PostgreSQL connections
5. Configure model storage backup strategy
6. Set appropriate rate limits for signal generation
7. Configure retraining schedules based on data accumulation patterns

### Monitoring

- Monitor model training duration (target: <30 minutes for 1M records)
- Track signal generation latency (target: <5 seconds)
- Monitor model quality metrics and degradation
- Set up alerts for training failures and quality threshold breaches
- Track signal publishing success rate (target: 99.5%)
- Monitor rate limiting and throttling events

### Scaling

- Multiple instances can run (each with its own model storage)
- Model files should be shared via network storage or synchronized
- RabbitMQ queues support multiple consumers for signal distribution
- Database connection pooling handles concurrent operations
- Training operations should be coordinated to avoid conflicts

### Model Management

- Implement model version cleanup policy (keep last N versions)
- Archive old model versions to object storage
- Monitor model storage disk space
- Implement model rollback procedures
- Track model performance over time

## Next Steps

- Read the [full specification](./spec.md) for detailed requirements
- Review the [data model](./data-model.md) for database schema
- Check [API contracts](./contracts/) for complete API documentation
- See [implementation plan](./plan.md) for architecture details
- Review [research decisions](./research.md) for technology choices

## Support

For issues or questions:
1. Check service logs: `docker compose logs model-service`
2. Review health endpoint: `curl http://localhost:4500/health`
3. Check training status: `curl http://localhost:4500/api/v1/training/status -H "X-API-Key: your-key"`
4. Consult the specification and documentation in this directory

