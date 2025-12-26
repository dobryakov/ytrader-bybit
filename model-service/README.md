# Model Service - Trading Decision and ML Training Microservice

A microservice that trains ML models from market data, generates trading signals using trained models or warm-up heuristics, and manages model versioning.

## Features

- **Warm-up Mode**: Generate trading signals using simple heuristics when no trained model exists
- **Model Training**: Train ML models from market data using Feature Service datasets (market-data-only training approach)
- **Signal Generation**: Generate intelligent trading signals using trained models with duplicate order prevention
- **Model Versioning**: Track model versions, quality metrics, and manage model lifecycle
- **Feature Service Integration**: Receive ready feature vectors from Feature Service for both inference and training
- **Time-based Retraining**: Scheduled retraining based on time intervals instead of execution event accumulation
- **Real-time Processing**: Async message queue processing with RabbitMQ
- **Observability**: Structured logging with trace IDs for request flow tracking

## Quick Start: Retraining Models for Feature Service

**If you see warnings about missing features (e.g., `spread_percent`):**

Your model was trained before Feature Service integration and expects old feature names. **Retrain the model** to use Feature Service features directly:

```bash
# 1. Trigger training (will request dataset from Feature Service)
curl -X POST http://localhost:4500/api/v1/training/trigger \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"strategy_id": "test-strategy", "symbol": "ETHUSDT"}'

# 2. Check training status
curl -X GET "http://localhost:4500/api/v1/training/status" \
  -H "X-API-Key: your-api-key"

# 3. Activate new model (if quality threshold met, auto-activation occurs)
curl -X POST "http://localhost:4500/api/v1/models/{new_version}/activate" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"strategy_id": "test-strategy"}'
```

**Temporary workaround** (not recommended): Set `FEATURE_SERVICE_LEGACY_FEATURE_COMPATIBILITY=true` to enable automatic feature name mapping. This is only for temporary compatibility - models should be retrained.

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
│   ├── models/              # Data models (signals, feature vectors, datasets)
│   ├── services/            # Core business logic (signal generation, training orchestration)
│   ├── api/                 # REST API endpoints (health checks, monitoring)
│   ├── consumers/           # RabbitMQ message consumers (feature vectors, dataset ready notifications)
│   ├── publishers/          # RabbitMQ message publishers (trading signals)
│   ├── database/            # Database access layer (model metadata, quality metrics)
│   ├── config/              # Configuration management
│   └── main.py              # Application entry point
├── tests/
│   ├── unit/                # Unit tests for models, services, utilities
│   ├── integration/         # Integration tests for database, message queue
│   └── e2e/                 # End-to-end tests for full workflows
├── models/                  # Model file storage directory (mounted volume)
├── datasets/                # Dataset storage directory (downloaded from Feature Service)
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
MODEL_ACTIVATION_THRESHOLD=0.75

# Time-based Retraining Configuration (market-data-only training)
MODEL_RETRAINING_INTERVAL_DAYS=7
MODEL_RETRAINING_TRAIN_PERIOD_DAYS=30
MODEL_RETRAINING_VALIDATION_PERIOD_DAYS=7
MODEL_RETRAINING_TEST_PERIOD_DAYS=1

# Feature Service Configuration
FEATURE_SERVICE_HOST=feature-service
FEATURE_SERVICE_PORT=4900
FEATURE_SERVICE_API_KEY=your-feature-service-api-key
FEATURE_SERVICE_USE_QUEUE=true
FEATURE_SERVICE_FEATURE_CACHE_TTL_SECONDS=30
FEATURE_SERVICE_DATASET_BUILD_TIMEOUT_SECONDS=3600
FEATURE_SERVICE_DATASET_POLL_INTERVAL_SECONDS=60
FEATURE_SERVICE_DATASET_STORAGE_PATH=/datasets

# Legacy Feature Compatibility (for models trained before Feature Service integration)
# Set to 'true' only if you have old models that require legacy feature names (e.g., spread_percent)
# New models should be trained on Feature Service features directly (set to 'false')
# WARNING: This is a temporary workaround. Models should be retrained on Feature Service features.
FEATURE_SERVICE_LEGACY_FEATURE_COMPATIBILITY=false

# Signal Generation Configuration
SIGNAL_GENERATION_RATE_LIMIT=60
SIGNAL_GENERATION_BURST_ALLOWANCE=10

# Open Orders Check Configuration
# Skip signal generation if open order exists for same asset and strategy
# Set to 'true' to prevent duplicate orders, 'false' to allow multiple signals
SIGNAL_GENERATION_SKIP_IF_OPEN_ORDER=true

# Check only opposite direction orders when skipping (e.g., skip buy signal if sell order exists)
# Set to 'true' to check only opposite direction orders, 'false' to check all orders
# Only used when SIGNAL_GENERATION_SKIP_IF_OPEN_ORDER=true
SIGNAL_GENERATION_CHECK_OPPOSITE_ORDERS_ONLY=false

# Warm-up Mode Configuration
WARMUP_MODE_ENABLED=true
WARMUP_SIGNAL_FREQUENCY=60

# Intelligent Mode Configuration
# Frequency of model-based (intelligent) signals, in signals per minute.
# For target horizon 1800s (30 minutes) typical values:
#   - 0.05  => 1 signal / 20 minutes (консервативно)
#   - 0.067 => 1 signal / 15 minutes (чуть агрессивнее)
# If not set, defaults to 60 (1 signal per second) – для продакшена задайте осмысленное значение.
INTELLIGENT_SIGNAL_FREQUENCY=0.05

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

## Signal Generation and Duplicate Order Prevention

The model service includes a duplicate order prevention mechanism to avoid generating multiple signals for the same asset when an open order already exists. This prevents duplicate orders and helps manage risk.

### Configuration

The open orders check behavior is controlled by two configuration parameters:

- **`SIGNAL_GENERATION_SKIP_IF_OPEN_ORDER`** (default: `true`): 
  - When `true`, the service skips signal generation if there are existing open orders (status: `pending` or `partially_filled`) for the same asset and strategy.
  - When `false`, the service generates signals regardless of existing orders (may result in duplicate orders).

- **`SIGNAL_GENERATION_CHECK_OPPOSITE_ORDERS_ONLY`** (default: `false`):
  - When `true`, the service only checks for opposite direction orders (e.g., skips a buy signal if a sell order exists, or vice versa).
  - When `false`, the service checks for any open order regardless of direction (more conservative approach).
  - Only used when `SIGNAL_GENERATION_SKIP_IF_OPEN_ORDER=true`.

### Behavior Examples

1. **Default Behavior** (`SIGNAL_GENERATION_SKIP_IF_OPEN_ORDER=true`, `SIGNAL_GENERATION_CHECK_OPPOSITE_ORDERS_ONLY=false`):
   - If any open order exists for an asset, signal generation is skipped.
   - Example: A buy signal for BTCUSDT is skipped if there's already a pending buy or sell order for BTCUSDT.

2. **Opposite Orders Only** (`SIGNAL_GENERATION_SKIP_IF_OPEN_ORDER=true`, `SIGNAL_GENERATION_CHECK_OPPOSITE_ORDERS_ONLY=true`):
   - Only opposite direction orders prevent signal generation.
   - Example: A buy signal for BTCUSDT is skipped only if there's a pending sell order for BTCUSDT (buy orders don't block buy signals).

3. **Disabled** (`SIGNAL_GENERATION_SKIP_IF_OPEN_ORDER=false`):
   - Signal generation proceeds regardless of existing orders (may result in multiple orders for the same asset).

### Monitoring

Signal skip events are tracked and can be viewed via the monitoring API:

```bash
GET /api/v1/monitoring/signals/skip-metrics?asset=BTCUSDT&strategy_id=momentum_v1
```

This endpoint provides:
- Total count of skipped signals
- Breakdown by asset and strategy
- Breakdown by skip reason
- Timestamp of last metrics reset

Skip events are also logged with structured logging including asset, strategy_id, existing_order_id, order_status, and reason.

## Market Data & Balance Freshness, Signal Delay Monitoring

The model service includes additional safeguards to avoid acting on stale data and to monitor end-to-end signal latency:

- **`BALANCE_ADAPTATION_SAFETY_MARGIN`** (default: `0.95`):
  - Fraction of available balance that can be used when adapting signal amounts.
  - Example: `0.95` means at most 95% of available balance is used when adjusting order size.

- **`BALANCE_DATA_MAX_AGE_SECONDS`** (default: `60`):
  - Maximum acceptable age of balance snapshots from the `account_balances` table.
  - If balance data is older than this threshold, balance-aware adaptation is skipped and the signal is not generated to avoid using stale balances.

- **`MARKET_DATA_MAX_AGE_SECONDS`** (default: `60`):
  - Maximum acceptable age of cached market data from `MarketDataCache`.
  - If cached market data is older than this value, it is treated as stale and signal generation is skipped for the affected asset.

- **`MARKET_DATA_STALE_WARNING_THRESHOLD_SECONDS`** (default: `30`):
  - Warning threshold for market data staleness.
  - When market data age exceeds this value but is still below `MARKET_DATA_MAX_AGE_SECONDS`, warnings are logged but data is still used.

- **`SIGNAL_PROCESSING_DELAY_ALERT_THRESHOLD_SECONDS`** (default: `300`):
  - Alert threshold (in seconds) for delay between signal creation time and publication time.
  - When the delay exceeds this threshold, a warning is logged and the event is reflected in the signal processing delay metrics.

Signal processing delay metrics can be queried via the monitoring API:

```bash
GET /api/v1/monitoring/signals/processing-delay
```

The response includes total signals, average delay, maximum delay, and histogram bucket counts useful for Grafana dashboards.

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
  ],
  "confidence_threshold_info": {
    "threshold_value": 0.65,
    "threshold_source": "top_k",
    "top_k_percentage": 10,
    "metric_name": "top_k_10_confidence_threshold"
  }
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

#### Get Model Analysis (Detailed)

```bash
GET /api/v1/models/{version}/analysis
```

Get detailed analysis for a model version including predictions, baseline metrics, and top-k analysis.

**Path Parameters:**
- `version`: Model version identifier (e.g., 'v1', 'v2.1')

**Response:**
```json
{
  "model_version": "v1766688758",
  "model_id": "c66769ec-d74e-4a2d-b0a5-c097260ce069",
  "predictions": [
    {
      "split": "test",
      "count": 6351,
      "dataset_id": "b8adcd3d-fe4e-4b56-a7e2-f37021d63ed1",
      "created_at": "2025-12-25T18:52:38.388136+00:00"
    }
  ],
  "model_metrics": {
    "accuracy": 0.6486,
    "precision": 0.4403,
    "recall": 0.6486,
    "f1_score": 0.5245,
    "balanced_accuracy": 0.4958,
    "roc_auc": 0.6772,
    "pr_auc": 0.7812
  },
  "baseline_metrics": {
    "accuracy": 0.6541,
    "precision": 0.4278,
    "recall": 0.6541,
    "f1_score": 0.5173,
    "balanced_accuracy": 0.5,
    "roc_auc": 0.0,
    "pr_auc": 0.0
  },
  "top_k_metrics": [
    {
      "k": 10,
      "accuracy": 0.4646,
      "precision": 0.5087,
      "recall": 0.4646,
      "f1_score": 0.4856,
      "balanced_accuracy": 0.4263,
      "roc_auc": 0.9293,
      "pr_auc": 0.9513,
      "lift": 0.7103,
      "coverage": 0.1,
      "precision_class_1": 0.9335,
      "recall_class_1": 0.8526,
      "f1_class_1": 0.8912
    },
    {
      "k": 20,
      "accuracy": 0.5291,
      "precision": 0.5185,
      "recall": 0.5291,
      "f1_score": 0.5238,
      "balanced_accuracy": 0.4308,
      "roc_auc": 0.8564,
      "pr_auc": 0.8992,
      "lift": 0.8090,
      "coverage": 0.2,
      "precision_class_1": 0.8442,
      "recall_class_1": 0.8615,
      "f1_class_1": 0.8528
    },
    {
      "k": 30,
      "accuracy": 0.5570,
      "precision": 0.5398,
      "recall": 0.5570,
      "f1_score": 0.5483,
      "balanced_accuracy": 0.4313,
      "roc_auc": 0.8126,
      "pr_auc": 0.8736,
      "lift": 0.8515,
      "coverage": 0.3,
      "precision_class_1": 0.8361,
      "recall_class_1": 0.8626,
      "f1_class_1": 0.8491
    },
    {
      "k": 50,
      "accuracy": 0.5669,
      "precision": 0.5370,
      "recall": 0.5669,
      "f1_score": 0.5515,
      "balanced_accuracy": 0.4257,
      "roc_auc": 0.7635,
      "pr_auc": 0.8460,
      "lift": 0.8668,
      "coverage": 0.5,
      "precision_class_1": 0.8065,
      "recall_class_1": 0.8515,
      "f1_class_1": 0.8283
    }
  ],
  "comparison": {
    "accuracy": {
      "model": 0.6486,
      "baseline": 0.6541,
      "difference": -0.0055
    },
    "f1_score": {
      "model": 0.5245,
      "baseline": 0.5173,
      "difference": 0.0072
    },
    "pr_auc": {
      "model": 0.7812,
      "baseline": 0.0,
      "difference": 0.7812
    },
    "roc_auc": {
      "model": 0.6772,
      "baseline": 0.0,
      "difference": 0.6772
    }
  }
}
```

**Response Fields:**
- `predictions`: List of saved predictions with count and split information
- `model_metrics`: Main model metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
- `baseline_metrics`: Baseline metrics using majority class strategy
- `top_k_metrics`: Top-k% analysis metrics for k=10,20,30,50 (without filters)
- `comparison`: Direct comparison between model and baseline metrics with differences

**Use Cases:**
- Understanding model performance vs baseline
- Evaluating edge in top-k% predictions
- Data for threshold and filter optimization
- Ranking performance analysis

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
  "queue_size": 0,
  "pending_dataset_builds": 1
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
  "strategy_id": "momentum_v1",
  "symbol": "BTCUSDT"
}
```

**Query Parameters:**
- `strategy_id` (optional): Trading strategy identifier
- `symbol` (optional): Trading pair symbol (e.g., 'BTCUSDT'). If not provided, uses default symbol.

**Response:**
```json
{
  "triggered": true,
  "message": "Training triggered successfully. Dataset build requested from Feature Service. Training will start when dataset is ready.",
  "strategy_id": "momentum_v1"
}
```

**Note**: Training uses market data from Feature Service (not execution_events). The system will:
1. Request dataset build from Feature Service with time-based periods
2. Wait for dataset.ready notification
3. Download dataset and start training automatically

## Model Training and Evaluation Workflow

The model service uses a three-way data split (train/validation/test) for proper model evaluation and activation decisions.

### Dataset Splits

- **Train Split**: Used for model training. Contains historical market data for learning patterns.
- **Validation Split**: Used for hyperparameter tuning and early stopping during training. Metrics are saved with `dataset_split="validation"` metadata.
- **Test Split**: Used for final out-of-sample evaluation. This is the **primary metric** used for model activation decisions. Metrics are saved with `dataset_split="test"` metadata.

### Evaluation Workflow

1. **Training Phase**: Model is trained on the train split.
2. **Validation Evaluation**: Model is evaluated on the validation split. Metrics are saved for monitoring and comparison.
3. **Test Evaluation**: Model is evaluated on the test split (out-of-sample data). This provides the final quality assessment.
4. **Model Activation**: Model activation decisions use **test set metrics** (not validation metrics) to ensure generalization. If test split is unavailable, the system falls back to validation metrics with a warning.

### Class Balancing for Imbalanced Datasets

The model service automatically handles class imbalance in training datasets using XGBoost's built-in balancing mechanisms:

#### Binary Classification

For binary classification tasks (2 classes), the service uses `scale_pos_weight` parameter:

- **Formula**: `scale_pos_weight = negative_count / positive_count`
- **Purpose**: Automatically adjusts the weight of positive class samples to balance the dataset
- **Effect**: Improves recall for minority classes by giving more weight to positive samples during training
- **Example**: If negative class has 800 samples and positive class has 200 samples, `scale_pos_weight = 800/200 = 4.0`

The service logs class distribution percentages and calculated `scale_pos_weight` value for monitoring.

#### Multi-Class Classification

For multi-class classification tasks (3+ classes), the service uses `sample_weight` array:

- **Formula**: `class_weight = total_samples / (number_of_classes * class_count)`
- **Purpose**: Assigns higher weights to minority classes and lower weights to majority classes
- **Effect**: Ensures that minority classes (e.g., "down" with 20.24% of samples) receive higher weights than majority classes (e.g., "flat" with 48.78% of samples)
- **Verification**: The service verifies that minority classes receive higher weights and logs the weight ratio for monitoring

**Current Use Case**: The service handles 3-class classification (flat, up, down) where class distribution may be imbalanced. The `sample_weight` array is automatically created and passed to `model.fit()` to ensure balanced learning.

Both balancing methods are automatically applied when training XGBoost models. The service logs detailed information about class distribution, calculated weights, and weight ratios for observability.

### Quality Metrics Storage

Quality metrics are stored separately for each dataset split:
- Validation metrics: `metadata.dataset_split = "validation"`
- Test metrics: `metadata.dataset_split = "test"`

This allows querying metrics by split:
```bash
# Get test set metrics
GET /api/v1/models/{version}/metrics?dataset_split=test

# Get validation metrics
GET /api/v1/models/{version}/metrics?dataset_split=validation
```

### Fallback Behavior

If the test split is unavailable (download fails, empty split, or file not found), the system:
1. Logs a structured warning with the reason (`download_failed`, `empty_split`, `file_not_found`)
2. Falls back to validation set metrics for model activation decisions
3. Continues training workflow without interruption

This ensures backward compatibility with models that may only have validation metrics and provides operational awareness when test splits are missing.

### Mode Transition and Retraining Triggers

Both mode transition (warm-up → model-based) and retraining triggers (quality degradation detection) prefer test set metrics over validation metrics:
- **First attempt**: Query test set metrics
- **Fallback**: Use validation metrics if test set not available
- **Final fallback**: Use any available metrics (backward compatibility)

This ensures that activation and retraining decisions are based on the most reliable out-of-sample evaluation available.

#### Request Dataset Build

```bash
POST /api/v1/training/dataset/build
```

Explicitly request dataset build from Feature Service without immediately triggering training. Training will start automatically when dataset.ready notification is received.

**Request Body:**
```json
{
  "strategy_id": "momentum_v1",
  "symbol": "BTCUSDT"
}
```

**Query Parameters:**
- `strategy_id` (optional): Trading strategy identifier
- `symbol` (optional): Trading pair symbol (e.g., 'BTCUSDT'). If not provided, uses default symbol.

**Response:**
```json
{
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Dataset build requested successfully. Dataset ID: 550e8400-e29b-41d4-a716-446655440000. Training will start automatically when dataset is ready.",
  "strategy_id": "momentum_v1",
  "symbol": "BTCUSDT"
}
```

**Use Cases:**
- Pre-build datasets before scheduled training
- Test dataset building without triggering training
- Build datasets for multiple symbols/strategies in advance

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

#### Get Signal Skip Metrics

```bash
GET /api/v1/monitoring/signals/skip-metrics?asset={asset}&strategy_id={strategy_id}
```

Get metrics for signal generation skipping due to open orders.

**Query Parameters:**
- `asset` (optional): Filter by trading pair symbol (e.g., 'BTCUSDT')
- `strategy_id` (optional): Filter by trading strategy identifier

**Response:**
```json
{
  "total_skips": 42,
  "by_asset_strategy": {
    "momentum_v1:BTCUSDT": {
      "strategy_id": "momentum_v1",
      "asset": "BTCUSDT",
      "total_skips": 25,
      "by_reason": {
        "Open order exists for asset BTCUSDT": 20,
        "Open sell order exists for asset BTCUSDT": 5
      }
    }
  },
  "by_reason": {
    "Open order exists for asset BTCUSDT": 30,
    "Open sell order exists for asset BTCUSDT": 12
  },
  "last_reset": "2025-01-27T10:00:00Z"
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

## Feature Service Integration

Model Service integrates with Feature Service to receive ready feature vectors for inference and training datasets. This integration enables:

- **Market-Data-Only Training**: Models learn from market movements (price predictions) rather than from own trading results
- **Ready Features**: Feature engineering is handled by Feature Service, Model Service receives pre-computed features
- **Dataset Building**: Training datasets are built by Feature Service with explicit train/validation/test periods
- **Time-based Retraining**: Training is triggered by scheduled time intervals instead of execution event accumulation
- **Backward Compatibility**: Automatic feature name mapping for models trained before Feature Service integration

### Feature Service Workflow

1. **Inference**: Model Service requests latest features from Feature Service (via queue or REST API) for signal generation
2. **Training**: Model Service requests dataset build from Feature Service with time-based periods, waits for dataset.ready notification, downloads and trains on ready dataset

### Feature Name Compatibility and Model Retraining

**Important**: Models trained before Feature Service integration expect different feature names (e.g., `spread_percent` instead of `spread_rel`). 

**Recommended Approach**: Retrain all models using Feature Service datasets to use the new feature names directly. This ensures:
- Models use the exact features provided by Feature Service
- No feature name mapping or computation overhead
- Better model performance with properly computed features

**Legacy Compatibility Mode** (temporary workaround):
- Set `FEATURE_SERVICE_LEGACY_FEATURE_COMPATIBILITY=true` to enable automatic computation of legacy features
- This mode computes old feature names (e.g., `spread_percent`) from new Feature Service features
- **Warning**: This is a temporary solution. Models should be retrained on Feature Service features.

**How to Retrain a Model**:

1. **Request dataset build and trigger training**:
   ```bash
   curl -X POST http://localhost:4500/api/v1/training/trigger \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "strategy_id": "test-strategy",
       "symbol": "ETHUSDT"
     }'
   ```

2. **Wait for training to complete** (check status):
   ```bash
   curl -X GET "http://localhost:4500/api/v1/training/status" \
     -H "X-API-Key: your-api-key"
   ```

3. **Activate the new model** (if quality meets threshold, it activates automatically, otherwise activate manually):
   ```bash
   curl -X POST "http://localhost:4500/api/v1/models/{new_version}/activate" \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"strategy_id": "test-strategy"}'
   ```

The new model will use Feature Service feature names directly (e.g., `spread_rel`, `mid_price`, `returns_1m`) instead of legacy names.

### Configuration

See `env.example` for complete Feature Service configuration options including:
- `FEATURE_SERVICE_HOST`, `FEATURE_SERVICE_PORT`, `FEATURE_SERVICE_API_KEY`
- `FEATURE_SERVICE_USE_QUEUE`: Use queue subscription (true) or REST API polling (false)
- `FEATURE_SERVICE_DATASET_BUILD_TIMEOUT_SECONDS`: Maximum wait time for dataset build
- `FEATURE_SERVICE_DATASET_STORAGE_PATH`: Directory for downloaded dataset files

## Message Queues

### Consumed Queues

- `features.live`: Feature vectors from Feature Service (for inference)
- `features.dataset.ready`: Dataset completion notifications from Feature Service (for training)

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

