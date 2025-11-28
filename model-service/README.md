# Model Service - Trading Decision and ML Training Microservice

A microservice that trains ML models from order execution feedback, generates trading signals using trained models or warm-up heuristics, and manages model versioning.

## Features

- **Warm-up Mode**: Generate trading signals using simple heuristics when no trained model exists
- **Model Training**: Train ML models from order execution feedback using scikit-learn and XGBoost
- **Signal Generation**: Generate intelligent trading signals using trained models
- **Model Versioning**: Track model versions, quality metrics, and manage model lifecycle
- **Real-time Processing**: Async message queue processing with RabbitMQ
- **Observability**: Structured logging with trace IDs for request flow tracking

## Technology Stack

- **Language**: Python 3.11+
- **ML Frameworks**: scikit-learn>=1.3.0, xgboost>=2.0.0
- **Data Processing**: pandas>=2.0.0
- **Message Queue**: aio-pika>=9.0.0 (RabbitMQ)
- **Database**: asyncpg>=0.29.0 (PostgreSQL)
- **Web Framework**: FastAPI>=0.104.0
- **Configuration**: pydantic-settings>=2.0.0
- **Logging**: structlog>=23.2.0

## Project Structure

```
model-service/
├── src/
│   ├── models/              # ML model definitions and training logic
│   ├── services/            # Core business logic (signal generation, training orchestration)
│   ├── api/                 # REST API endpoints (health checks, monitoring)
│   ├── consumers/           # RabbitMQ message consumers (order execution events)
│   ├── publishers/          # RabbitMQ message publishers (trading signals)
│   ├── database/            # Database access layer (model metadata, quality metrics)
│   ├── config/              # Configuration management
│   └── main.py              # Application entry point
├── tests/
│   ├── unit/                # Unit tests for models, services, utilities
│   ├── integration/         # Integration tests for database, message queue
│   └── e2e/                 # End-to-end tests for full workflows
├── models/                  # Model file storage directory (mounted volume)
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Docker and Docker Compose V2
- Python 3.11+ (for local development)
- PostgreSQL 15+ (via Docker)
- RabbitMQ 3+ (via Docker)

### Environment Variables

Copy `env.example` to `.env` and configure the following variables:

```bash
# Model Service Configuration
MODEL_SERVICE_PORT=4500
MODEL_SERVICE_API_KEY=your-api-key-here
MODEL_SERVICE_LOG_LEVEL=INFO
MODEL_SERVICE_SERVICE_NAME=model-service

# Model Storage
MODEL_STORAGE_PATH=/models

# Model Training Configuration
MODEL_TRAINING_MIN_DATASET_SIZE=1000
MODEL_TRAINING_MAX_DURATION_SECONDS=1800
MODEL_QUALITY_THRESHOLD_ACCURACY=0.75
MODEL_RETRAINING_SCHEDULE=

# Signal Generation Configuration
SIGNAL_GENERATION_RATE_LIMIT=60
SIGNAL_GENERATION_BURST_ALLOWANCE=10

# Warm-up Mode Configuration
WARMUP_MODE_ENABLED=true
WARMUP_SIGNAL_FREQUENCY=60

# Trading Strategy Configuration
TRADING_STRATEGIES=
```

See `env.example` for complete configuration options.

### Docker Setup

1. Build and start the service:

```bash
docker compose build model-service
docker compose up -d model-service
```

2. Check service health:

```bash
curl http://localhost:4500/health
```

3. View logs:

```bash
docker compose logs -f model-service
```

### Local Development Setup

1. Create a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install development tools:

```bash
pip install black flake8 mypy
```

4. Run linting:

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

5. Run tests:

```bash
pytest
```

6. Run the service locally:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 4500 --reload
```

## API Endpoints

All API endpoints (except `/health`) require authentication via the `X-API-Key` header. The API key is configured via the `MODEL_SERVICE_API_KEY` environment variable.

### Health Check

```bash
GET /health
```

Returns service health status including database, message queue, and model storage checks.

**Response:**
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

### Model Management

#### List Model Versions

```bash
GET /api/v1/models?strategy_id={strategy_id}&is_active={true|false}&limit={limit}&offset={offset}
```

List model versions with filtering and pagination.

**Query Parameters:**
- `strategy_id` (optional): Filter by trading strategy identifier
- `is_active` (optional): Filter by active status (true/false)
- `limit` (optional): Maximum number of results (1-1000, default: 100)
- `offset` (optional): Number of results to skip (default: 0)

**Response:**
```json
{
  "items": [
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
      "created_at": "2025-01-27T10:00:00Z",
      "updated_at": "2025-01-27T10:00:00Z"
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

#### Get Model Version Details

```bash
GET /api/v1/models/{version}
```

Get detailed information about a specific model version including quality metrics.

**Path Parameters:**
- `version`: Model version identifier (e.g., 'v1', 'v2.1')

**Response:**
```json
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
  "quality_metrics": [
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "metric_name": "accuracy",
      "metric_value": 0.82,
      "metric_type": "classification",
      "evaluated_at": "2025-01-27T10:00:00Z"
    }
  ]
}
```

#### Activate Model Version

```bash
POST /api/v1/models/{version}/activate
```

Activate a model version (deactivates previous active model for the strategy).

**Path Parameters:**
- `version`: Model version identifier (e.g., 'v1', 'v2.1')

**Request Body:**
```json
{
  "strategy_id": "momentum_v1"
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "version": "v1",
  "model_type": "xgboost",
  "strategy_id": "momentum_v1",
  "is_active": true,
  ...
}
```

### Quality Metrics

#### Get Model Quality Metrics

```bash
GET /api/v1/models/{version}/metrics?metric_type={type}&metric_name={name}
```

Get quality metrics for a model version.

**Path Parameters:**
- `version`: Model version identifier (e.g., 'v1', 'v2.1')

**Query Parameters:**
- `metric_type` (optional): Filter by metric type (classification, regression, trading_performance)
- `metric_name` (optional): Filter by metric name

**Response:**
```json
[
  {
    "id": "660e8400-e29b-41d4-a716-446655440001",
    "model_version_id": "550e8400-e29b-41d4-a716-446655440000",
    "metric_name": "accuracy",
    "metric_value": 0.82,
    "metric_type": "classification",
    "evaluated_at": "2025-01-27T10:00:00Z",
    "evaluation_dataset_size": 10000
  }
]
```

#### Get Time-Series Metrics

```bash
GET /api/v1/models/{version}/metrics/time-series?granularity={hour|day|week}&start_time={ISO8601}&end_time={ISO8601}&metric_names={name1,name2}
```

Get time-series metrics data for charting.

**Path Parameters:**
- `version`: Model version identifier (e.g., 'v1', 'v2.1')

**Query Parameters:**
- `granularity` (optional): Time granularity (hour, day, week, default: hour)
- `start_time` (optional): Start time in ISO 8601 format (default: 7 days ago)
- `end_time` (optional): End time in ISO 8601 format (default: now)
- `metric_names` (optional): Comma-separated list of metric names

**Response:**
```json
{
  "model_version": "v1",
  "granularity": "hour",
  "start_time": "2025-01-20T10:00:00Z",
  "end_time": "2025-01-27T10:00:00Z",
  "data_points": [
    {
      "timestamp": "2025-01-27T10:00:00Z",
      "value": 0.82,
      "metric_name": "accuracy"
    }
  ]
}
```

### Training Management

#### Get Training Status

```bash
GET /api/v1/training/status
```

Get current training status.

**Response:**
```json
{
  "is_training": false,
  "current_training": null,
  "last_training": null,
  "next_scheduled_training": null,
  "buffered_events_count": 150
}
```

#### Trigger Manual Training

```bash
POST /api/v1/training/trigger
```

Manually trigger training for a strategy.

**Request Body:**
```json
{
  "strategy_id": "momentum_v1"
}
```

**Response:**
```json
{
  "triggered": true,
  "message": "Training triggered successfully",
  "strategy_id": "momentum_v1"
}
```

### Monitoring

#### Get Model Performance Metrics

```bash
GET /api/v1/monitoring/models/performance
```

Get model performance metrics including counts and breakdowns.

**Response:**
```json
{
  "active_models_count": 2,
  "total_models_count": 10,
  "models_by_strategy": {
    "momentum_v1": 5,
    "mean_reversion_v1": 5
  },
  "models_by_type": {
    "xgboost": 8,
    "random_forest": 2
  }
}
```

#### Get System Health Details

```bash
GET /api/v1/monitoring/health
```

Get detailed system health information.

**Response:**
```json
{
  "database_connected": true,
  "message_queue_connected": true,
  "model_storage_accessible": true,
  "active_models": 2,
  "warmup_mode_enabled": false
}
```

#### Get Strategy Performance Time-Series

```bash
GET /api/v1/strategies/{strategy_id}/performance/time-series?granularity={hour|day|week}&start_time={ISO8601}&end_time={ISO8601}
```

Get strategy performance time-series data.

**Path Parameters:**
- `strategy_id`: Trading strategy identifier

**Query Parameters:**
- `granularity` (optional): Time granularity (hour, day, week, default: hour)
- `start_time` (optional): Start time in ISO 8601 format (default: 7 days ago)
- `end_time` (optional): End time in ISO 8601 format (default: now)

**Response:**
```json
{
  "strategy_id": "momentum_v1",
  "granularity": "hour",
  "start_time": "2025-01-20T10:00:00Z",
  "end_time": "2025-01-27T10:00:00Z",
  "data_points": [
    {
      "timestamp": "2025-01-27T10:00:00Z",
      "success_rate": 0.75,
      "total_pnl": 1500.50,
      "avg_pnl": 15.00,
      "total_orders": 100,
      "successful_orders": 75
    }
  ]
}
```

## Message Queues

### Consumed Queues

- `order-manager.order_events`: Order execution events for model training

### Published Queues

- `model-service.trading_signals`: Trading signals generated by the service

## Database

The service uses a shared PostgreSQL database for model metadata and quality metrics. Model files are stored on the file system in `/models/v{version}/` directory structure.

## Development

### Code Style

- **Formatter**: Black (line length: 100)
- **Linter**: Flake8 (max line length: 100)
- **Type Checking**: mypy (Python 3.11)

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# End-to-end tests only
pytest tests/e2e/
```

## License

See repository root for license information.

