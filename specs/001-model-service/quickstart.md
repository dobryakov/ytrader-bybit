# Quickstart: Model Service

**Feature**: Model Service - Trading Decision and ML Training Microservice  
**Date**: 2025-01-27

## Overview

This guide provides step-by-step instructions to set up and run the Model Service locally for development and testing.

## Prerequisites

- Docker and Docker Compose V2 installed
- PostgreSQL database (shared database, managed by ws-gateway)
- RabbitMQ message queue (for feature vectors, dataset notifications, and trading signals)
- Feature Service microservice (for feature vectors and training datasets)
- Order Manager microservice (for order execution, optional for training)

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

# Time-based Retraining Configuration (market-data-only training)
MODEL_RETRAINING_INTERVAL_DAYS=7  # Retrain every 7 days
MODEL_RETRAINING_TRAIN_PERIOD_DAYS=30  # Use last 30 days for training
MODEL_RETRAINING_VALIDATION_PERIOD_DAYS=7  # Use 7 days for validation
MODEL_RETRAINING_TEST_PERIOD_DAYS=1  # Use 1 day for test

# Feature Service Configuration
FEATURE_SERVICE_HOST=feature-service
FEATURE_SERVICE_PORT=4900
FEATURE_SERVICE_API_KEY=your-feature-service-api-key
FEATURE_SERVICE_USE_QUEUE=true  # Use queue subscription (true) or REST API polling (false)
FEATURE_SERVICE_FEATURE_CACHE_TTL_SECONDS=30
FEATURE_SERVICE_DATASET_BUILD_TIMEOUT_SECONDS=3600
FEATURE_SERVICE_DATASET_POLL_INTERVAL_SECONDS=60
FEATURE_SERVICE_DATASET_STORAGE_PATH=/datasets

# Signal Generation Configuration
SIGNAL_GENERATION_RATE_LIMIT=60  # Signals per minute
SIGNAL_GENERATION_BURST_ALLOWANCE=10  # Burst allowance for rate limiting
WARMUP_MODE_ENABLED=true  # Enable warm-up mode when no trained model exists
WARMUP_SIGNAL_FREQUENCY=60  # Warm-up signals per minute

# Strategy Configuration
TRADING_STRATEGIES=momentum_v1,mean_reversion_v1  # Comma-separated strategy IDs
```

### 3. Start Dependencies

Start PostgreSQL, RabbitMQ, and Feature Service using Docker Compose:

```bash
docker compose up -d postgres rabbitmq feature-service
```

Wait for services to be ready:

```bash
docker compose logs -f postgres rabbitmq feature-service
# Press Ctrl+C when services are ready
```

**Note**: Feature Service is required for Model Service to function. It provides:
- Feature vectors for signal generation (inference)
- Training datasets for model training

### 4. Run Database Migrations

**Note**: Per constitution principle II (Shared Database Strategy), the `ws-gateway` service is the single source of truth for all PostgreSQL migrations. Model service migrations are located in `ws-gateway/migrations/` (e.g., `003_create_model_versions_table.sql`, `004_create_model_quality_metrics_table.sql`).

```bash
# Migrations should be run from ws-gateway service
docker compose run --rm ws-gateway python -m migrations.run
```

Or if migrations are run automatically on startup, skip this step.

### 5. Create Storage Directories

```bash
# Model storage
mkdir -p /home/ubuntu/ytrader/model-service/models
chmod 755 /home/ubuntu/ytrader/model-service/models

# Dataset storage (for downloaded datasets from Feature Service)
mkdir -p /home/ubuntu/ytrader/model-service/datasets
chmod 755 /home/ubuntu/ytrader/model-service/datasets
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
  "next_scheduled_training": "2025-01-28T02:00:00Z",
  "queue_size": 0,
  "pending_dataset_builds": 0
}
```

### Trigger Manual Model Training

Trigger training for a strategy. This will request dataset build from Feature Service:

```bash
curl -X POST http://localhost:4500/api/v1/training/trigger \
  -H "X-API-Key: your-model-service-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "momentum_v1",
    "symbol": "BTCUSDT"
  }'
```

Response:

```json
{
  "triggered": true,
  "message": "Training triggered successfully. Dataset build requested from Feature Service. Training will start when dataset is ready.",
  "strategy_id": "momentum_v1"
}
```

**Note**: Training will start automatically when Feature Service completes dataset building and sends dataset.ready notification.

### Request Dataset Build Explicitly

You can also request dataset build without immediately triggering training:

```bash
curl -X POST http://localhost:4500/api/v1/training/dataset/build \
  -H "X-API-Key: your-model-service-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "momentum_v1",
    "symbol": "BTCUSDT"
  }'
```

Response:

```json
{
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Dataset build requested successfully. Dataset ID: 550e8400-e29b-41d4-a716-446655440000. Training will start automatically when dataset is ready.",
  "strategy_id": "momentum_v1",
  "symbol": "BTCUSDT"
}
```

This is useful for:
- Pre-building datasets before scheduled training
- Testing dataset building without triggering training
- Building datasets for multiple symbols/strategies in advance

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

### Training Workflow (Feature Service Integration)

Model training now uses Feature Service for dataset building. The workflow is:

1. **Time-based Retraining**: Training is triggered by scheduled time intervals (e.g., every 7 days)
2. **Dataset Request**: Model Service requests dataset build from Feature Service with explicit train/validation/test periods
3. **Dataset Ready Notification**: Feature Service builds dataset and publishes notification to `features.dataset.ready` queue
4. **Dataset Download**: Model Service downloads ready dataset from Feature Service
5. **Model Training**: Model is trained on downloaded dataset (market data only, no execution_events)

**Note**: Training no longer depends on execution_events accumulation. Models learn from market movements (price predictions) rather than from own trading results.

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

2. Verify Feature Service is available:
   ```bash
   # Check Feature Service health
   curl http://localhost:4900/health
   
   # Check dataset ready queue
   docker compose exec rabbitmq rabbitmqadmin list queues name messages | grep dataset.ready
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
   # Should see: features.live, features.dataset.ready, model-service.trading_signals
   ```

3. Check queue consumers:
   ```bash
   docker compose exec rabbitmq rabbitmqadmin list queues name consumers
   ```

### Feature Service Integration Issues

1. Check Feature Service connection:
   ```bash
   docker compose logs model-service | grep -i "feature.*service\|feature.*client"
   ```

2. Verify Feature Service is running:
   ```bash
   curl http://localhost:4900/health
   ```

3. Check feature queue:
   ```bash
   docker compose exec rabbitmq rabbitmqadmin list queues name messages | grep features
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

