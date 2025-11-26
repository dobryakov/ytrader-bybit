# Implementation Plan: Model Service - Trading Decision and ML Training Microservice

**Branch**: `001-model-service` | **Date**: 2025-01-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-model-service/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

A microservice that trains ML models from order execution feedback, generates trading signals using trained models or warm-up heuristics, and manages model versioning. The service subscribes to order execution events via RabbitMQ, retrieves order/position state from PostgreSQL, trains models using file system storage with database metadata, and publishes trading signals back to RabbitMQ. Built with Python 3.11+ using scikit-learn and XGBoost for ML models, async message queue processing with aio-pika, asyncpg for database access, FastAPI for REST endpoints, and structured logging for observability.

## Technical Context

**Language/Version**: Python 3.11+ (async/await support for message queue processing, ML ecosystem maturity)  
**Primary Dependencies**: 
- ML Framework: `scikit-learn>=1.3.0` (comprehensive ML toolkit, preprocessing, evaluation, see research.md)
- ML Model Library: `xgboost>=2.0.0` (gradient boosting models for tabular data, see research.md)
- Data processing: `pandas>=2.0.0` for training dataset preparation and feature engineering
- Model serialization: `joblib>=1.3.0` for scikit-learn models, XGBoost native format (`.json`) for XGBoost models (see research.md)
- Message queue: `aio-pika>=9.0.0` for RabbitMQ (async client, consistent with existing services)
- Database: `asyncpg>=0.29.0` (fastest async PostgreSQL driver, consistent with existing services)
- REST API framework: `FastAPI>=0.104.0` (async support, OpenAPI generation, for health checks and monitoring endpoints)
- Configuration: `pydantic-settings>=2.0.0` for environment variable management
- Logging: `structlog>=23.2.0` for structured logging with trace IDs

**Storage**: 
- PostgreSQL (shared database for model metadata, version history, quality metrics)
- File system (model files stored as `/models/v{version}.pkl` or similar, path stored in database)

**Testing**: `pytest` with `pytest-asyncio` for async tests, `pytest-mock` for mocking  
**Target Platform**: Linux server (Docker container)  
**Project Type**: single (microservice)  
**Performance Goals**: 
- Signal generation latency: <5 seconds from receiving input data (order state, market data, or warm-up trigger)
- Signal publishing success rate: 99.5% without errors
- Order execution event processing: <10 seconds from receipt to training pipeline incorporation
- Model training completion: <30 minutes for datasets up to 1 million records
- Retraining trigger detection: <1 minute from quality threshold breach

**Constraints**: 
- Model files stored on file system with database metadata (not in database)
- Must handle retraining conflicts (cancel current training and restart with new data)
- Configurable rate limiting on signal generation with burst allowance
- Must support both online learning (incremental) and periodic batch retraining
- Must gracefully handle message queue and database failures with retry

**Scale/Scope**: 
- Multiple trading strategies (minimum 5 concurrent strategies)
- Model version history: at least 100 previous versions maintained
- Training datasets: up to 1 million records per training operation
- Multiple retraining triggers: scheduled periodic, data accumulation thresholds, quality degradation detection

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Microservices Architecture ✅
- **Status**: COMPLIANT
- **Check**: This is a new independent microservice with single responsibility (ML model training and trading signal generation)
- **Deployment**: Will be added to `docker-compose.yml` as a separate container
- **Scalability**: Service is independently deployable and testable

### II. Shared Database Strategy ✅
- **Status**: COMPLIANT
- **Check**: Uses shared PostgreSQL database for model metadata, version history, and quality metrics
- **Migrations**: Database schema changes will be managed through reversible migrations
- **Migration Ownership**: PostgreSQL migrations will be located in `ws-gateway/migrations/` (as ws-gateway is the designated owner of all PostgreSQL migrations for the shared database)
- **Justification**: Model files stored on file system (not in database) - only metadata in PostgreSQL

### III. Inter-Service Communication ✅
- **Status**: COMPLIANT
- **Check**: 
  - Event-driven: RabbitMQ queues for order execution events (consumption) and trading signals (publication)
  - REST API: May be used for health checks and monitoring endpoints (data queries/development convenience)
  - Logging: All inter-service communication will be logged with full request/response bodies and trace IDs
  - Authentication: API key authentication for service-to-service REST API calls (if implemented)

### IV. Testing Discipline ✅
- **Status**: COMPLIANT
- **Check**: 
  - Tests will run inside Docker containers (unit tests in service container, API/e2e in test containers)
  - Test containers connected to main `docker-compose.yml`
  - Automated tests will be executed after each task completion
  - Using `pytest` with `pytest-asyncio` for async testing

### V. Observability & Diagnostics ✅
- **Status**: COMPLIANT
- **Check**: 
  - Structured logging for signal generation, model training, mode transitions, message queue operations
  - Health checks for service health and model training status
  - Trace IDs for request flow tracking
  - All errors logged with sufficient context
  - Monitoring capabilities for model performance and system health

### VI. Infrastructure Standards ✅
- **Status**: COMPLIANT
- **Check**: 
  - Docker Compose V2 (`docker compose`) will be used
  - No `version:` field in `docker-compose.yml`
  - Commands run inside containers
  - Non-standard ports with explicit configuration
  - Base images reused when possible

### VII. Documentation & Configuration ✅
- **Status**: COMPLIANT
- **Check**: 
  - Configuration variables in `.env` with samples in `env.example`
  - README, quickstart, and specifications will be synchronized after changes
  - Examples using `curl` and shell scripts will be provided
  - Documentation for deploying from scratch will be maintained

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
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

**Structure Decision**: Single project structure (microservice). The service follows a modular architecture with clear separation of concerns: ML model logic, business services, API endpoints, message queue consumers/publishers, and database access. Model files are stored in a separate `models/` directory (mounted as Docker volume) while metadata is stored in PostgreSQL.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
