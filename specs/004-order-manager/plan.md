# Implementation Plan: Order Manager Microservice

**Branch**: `004-order-manager` | **Date**: 2025-01-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-order-manager/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

A microservice that receives high-level trading signals from the model service via RabbitMQ, executes them as orders on Bybit exchange, maintains accurate order state through WebSocket event subscriptions, and publishes enriched order execution events. The service implements sophisticated order management logic including order type selection (market vs limit), quantity calculation, signal-to-order relationships, position management, risk limits, and safety mechanisms. Built with Python using async architecture, FastAPI for REST endpoints, RabbitMQ for message consumption/publishing, PostgreSQL for order persistence, and Bybit REST API for order execution.

## Technical Context

**Language/Version**: Python 3.11+ (async/await support for concurrent order processing and WebSocket event handling)  
**Primary Dependencies**: 
- REST API framework: `FastAPI>=0.104.0` (async support, OpenAPI generation, API key authentication)
- Message queue: `aio-pika>=9.0.0` for RabbitMQ (async client for consuming trading signals and publishing order events)
- Database: `asyncpg>=0.29.0` (fastest async PostgreSQL driver for order and position storage)
- HTTP client: `httpx>=0.25.0` for Bybit REST API calls (async HTTP client with retry support)
- Bybit API client: Direct REST API calls using `httpx` with custom authentication wrapper (see research.md)
- Configuration: `pydantic-settings>=2.0.0` for environment variable management
- Logging: `structlog>=23.2.0` for structured logging with trace IDs
- Testing: `pytest` with `pytest-asyncio` for async tests, `pytest-mock` for mocking

**Storage**: PostgreSQL (shared database for orders, signal-order relationships, position snapshots - migrations managed in ws-gateway service per constitution)  
**Testing**: `pytest` with `pytest-asyncio` for async tests, `pytest-mock` for mocking, test containers for integration tests  
**Target Platform**: Linux server (Docker container)  
**Project Type**: single (microservice)  
**Performance Goals**: 
- Process and execute valid trading signals within 2 seconds (SC-001)
- Maintain 99% order state synchronization accuracy (SC-002)
- Publish order events within 1 second of state changes (SC-004)
- Handle WebSocket reconnections within 5 seconds (SC-006)
- Manual synchronization completes within 30 seconds (SC-007)

**Constraints**: 
- <2s signal processing latency (from queue consumption to order creation)
- <1s event publishing latency
- <5s WebSocket reconnection time
- Non-standard port (4600 range, similar to ws-gateway 4400, model-service 4500)
- Must handle Bybit API rate limits (429 errors) with exponential backoff

**Scale/Scope**: 
- Single trading account context (not multi-account)
- Per-symbol FIFO signal processing
- Support for concurrent processing of different assets
- Order state synchronization for active orders
- Position management with periodic snapshots

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Microservices Architecture ✅
- ✅ Independent microservice in Docker container
- ✅ Single responsibility: order management and execution
- ✅ Independently deployable and scalable
- ✅ Managed via unified `docker-compose.yml`

### II. Shared Database Strategy ✅
- ✅ Uses shared PostgreSQL database for orders and positions
- ✅ PostgreSQL migrations MUST be located in `ws-gateway` service (constitution requirement)
- ✅ Database schema managed through reversible migrations
- ⚠️ **ACTION REQUIRED**: Coordinate migration placement with ws-gateway service

### III. Inter-Service Communication ✅
- ✅ Event-driven: consumes from RabbitMQ queue `model-service.trading_signals`
- ✅ Event-driven: publishes to RabbitMQ queues for order events
- ✅ REST API for queries and manual operations (development convenience)
- ✅ All communication logged with trace IDs

### IV. Testing Discipline ✅
- ✅ Tests run inside Docker containers (unit tests in service container, integration tests in test containers)
- ✅ Test containers connected to main `docker-compose.yml`
- ✅ Automated tests executed after task completion

### V. Observability & Diagnostics ✅
- ✅ Structured logging with trace IDs for all operations
- ✅ Health check endpoints (health, liveness, readiness)
- ✅ Full request/response logging for HTTP requests and message queue operations
- ✅ Diagnostic output for request flow tracking

### VI. Infrastructure Standards ✅
- ✅ Docker Compose V2 (`docker compose` command)
- ✅ No `version:` field in `docker-compose.yml`
- ✅ Commands run inside containers
- ✅ Non-standard port (4600 range)
- ✅ Base images reused when possible

### VII. Documentation & Configuration ✅
- ✅ Configuration in `.env` (no hardcoded variables)
- ✅ Samples in `env.example`
- ✅ README, quickstart, and specifications synchronized
- ✅ Examples using `curl` and shell scripts

**Gate Status**: ✅ PASS (all principles satisfied, migration coordination needed)

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
order-manager/
├── src/
│   ├── models/
│   │   ├── order.py              # Order entity model
│   │   ├── trading_signal.py     # Trading signal model (received from queue)
│   │   ├── signal_order_rel.py   # Signal-order relationship model
│   │   └── position.py           # Position entity model
│   ├── services/
│   │   ├── signal_processor.py   # Signal processing and order decision logic
│   │   ├── order_executor.py     # Bybit REST API order execution
│   │   ├── order_type_selector.py # Market vs limit order selection
│   │   ├── quantity_calculator.py # Amount to quantity conversion
│   │   ├── position_manager.py   # Position state management
│   │   ├── risk_manager.py       # Risk limits enforcement
│   │   ├── order_state_sync.py   # Order state synchronization
│   │   └── event_subscriber.py   # WebSocket event subscription handler
│   ├── api/
│   │   ├── main.py               # FastAPI application
│   │   ├── routes/
│   │   │   ├── orders.py         # Order query endpoints
│   │   │   ├── positions.py      # Position query endpoints
│   │   │   ├── sync.py           # Manual synchronization endpoints
│   │   │   └── health.py         # Health check endpoints
│   │   └── middleware/
│   │       └── auth.py           # API key authentication middleware
│   ├── consumers/
│   │   └── signal_consumer.py    # RabbitMQ signal consumer
│   ├── publishers/
│   │   └── order_event_publisher.py # RabbitMQ order event publisher
│   ├── config/
│   │   ├── settings.py           # Pydantic settings
│   │   └── logging.py            # Structured logging configuration
│   ├── utils/
│   │   ├── tracing.py            # Trace ID utilities
│   │   └── bybit_client.py       # Bybit REST API client wrapper
│   └── main.py                   # Service entry point
├── tests/
│   ├── unit/
│   │   ├── test_signal_processor.py
│   │   ├── test_order_executor.py
│   │   ├── test_quantity_calculator.py
│   │   └── test_position_manager.py
│   ├── integration/
│   │   ├── test_order_flow.py
│   │   └── test_state_sync.py
│   └── contract/
│       └── test_api_contracts.py
├── Dockerfile
├── requirements.txt
└── README.md
```

**Structure Decision**: Single microservice project structure (Option 1) following the pattern established by `ws-gateway` and `model-service`. The structure separates concerns into models (data entities), services (business logic), api (REST endpoints), consumers/publishers (message queue handlers), and utilities. This aligns with the existing codebase architecture and enables clean separation of order management logic, API interfaces, and external integrations.

## Complexity Tracking

> **No violations**: All constitution principles are satisfied. Migration coordination with ws-gateway service is a procedural requirement, not a violation.
