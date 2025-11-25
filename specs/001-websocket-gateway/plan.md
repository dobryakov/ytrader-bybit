# Implementation Plan: WebSocket Gateway for Bybit Data Aggregation and Routing

**Branch**: `001-websocket-gateway` | **Date**: 2025-11-25 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-websocket-gateway/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

A microservice that establishes and maintains a single authenticated WebSocket connection to Bybit exchange, subscribes to multiple data channels (trades, tickers, order books, order statuses, balances), and routes events to subscriber services via RabbitMQ queues. The service provides a REST API for dynamic subscription management and persists critical data (balances, subscriptions) to PostgreSQL. Built with Python using async WebSocket libraries, FastAPI for REST endpoints, and structured logging for observability.

## Technical Context

**Language/Version**: Python 3.11+ (async/await support for WebSocket handling)  
**Primary Dependencies**: 
- WebSocket client: `websockets>=12.0` (standard async WebSocket library, see research.md)
- REST API framework: `FastAPI>=0.104.0` (async support, OpenAPI generation)
- Message queue: `aio-pika>=9.0.0` for RabbitMQ (async client for consistency with async architecture)
- Database: `asyncpg>=0.29.0` (fastest async PostgreSQL driver, see research.md)
- HTTP client: `httpx>=0.25.0` for Bybit REST API calls if needed
- Configuration: `pydantic-settings>=2.0.0` for environment variable management
- Logging: `structlog>=23.2.0` for structured logging with trace IDs

**Storage**: PostgreSQL (shared database for subscriptions, balance data)  
**Testing**: `pytest` with `pytest-asyncio` for async tests, `pytest-mock` for mocking  
**Target Platform**: Linux server (Docker container)  
**Project Type**: single (microservice)  
**Performance Goals**: 
- WebSocket connection uptime: 99.5% over 30 days
- Event processing latency: <100ms from receipt to queue delivery
- REST API response time: <500ms for 95% of requests
- Support 10+ concurrent subscriber services

**Constraints**: 
- Automatic reconnection within 30 seconds
- Queue retention: 24 hours or 100K messages (whichever comes first)
- PostgreSQL write failures must not block WebSocket processing
- Must handle Bybit API rate limits and connection limits

**Scale/Scope**: 
- Single WebSocket connection to Bybit (mainnet or testnet)
- Multiple concurrent subscriptions (trades, tickers, order books, order statuses, balances)
- 10+ subscriber services consuming from RabbitMQ queues
- Queue organization by event class (trades, order_status, balances, etc.)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Microservices Architecture ✅
- **Status**: COMPLIANT
- **Check**: This is a new independent microservice with single responsibility (WebSocket aggregation and routing)
- **Deployment**: Will be added to `docker-compose.yml` as a separate container
- **Scalability**: Service is independently deployable and testable

### II. Shared Database Strategy ✅
- **Status**: COMPLIANT
- **Check**: Uses shared PostgreSQL database for subscription state and balance data persistence
- **Migrations**: Database schema changes will be managed through reversible migrations
- **Justification**: No specialized database needed for this service

### III. Inter-Service Communication ✅
- **Status**: COMPLIANT
- **Check**: 
  - Event-driven: RabbitMQ queues organized by event class (trades, order_status, balances)
  - REST API: Used for subscription management (data queries/development convenience)
  - Logging: All inter-service communication will be logged with full request/response bodies and trace IDs

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
  - Structured logging for WebSocket connection events, incoming messages, REST API requests
  - Health checks for WebSocket connection status and service health
  - Trace IDs for request flow tracking
  - All errors logged with sufficient context

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
  - README and quickstart will be synchronized after changes
  - Examples using `curl` and shell scripts provided
  - Deployment documentation maintained

**Gate Status**: ✅ **PASS** - All constitution principles are followed. No violations detected.

### Post-Design Re-evaluation (Phase 1 Complete)

After completing Phase 1 design (data model, contracts, quickstart), all constitution principles remain compliant:

- **Microservices Architecture**: ✅ Service structure defined as independent microservice in `ws-gateway/` directory
- **Shared Database Strategy**: ✅ PostgreSQL schema defined in data-model.md with reversible migrations
- **Inter-Service Communication**: ✅ RabbitMQ queues organized by event class, REST API contracts defined
- **Testing Discipline**: ✅ Test structure defined (unit, integration, e2e) with container-based execution
- **Observability & Diagnostics**: ✅ Structured logging with trace IDs included in data model and API contracts
- **Infrastructure Standards**: ✅ Docker Compose V2, non-standard ports (4400), container-based execution
- **Documentation & Configuration**: ✅ Quickstart.md created, .env configuration documented, API contracts provided

**Final Gate Status**: ✅ **PASS** - Design phase complete, all principles maintained.

## Project Structure

### Documentation (this feature)

```text
specs/001-websocket-gateway/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
ws-gateway/
├── src/
│   ├── models/          # Data models (Subscription, Event, etc.)
│   ├── services/
│   │   ├── websocket/   # Bybit WebSocket connection management
│   │   ├── queue/       # RabbitMQ queue operations
│   │   ├── database/    # PostgreSQL operations
│   │   └── subscription/ # Subscription management logic
│   ├── api/             # FastAPI REST endpoints
│   ├── config/          # Configuration management
│   └── main.py          # Application entry point
├── tests/
│   ├── unit/            # Unit tests (run in service container)
│   ├── integration/     # Integration tests (test containers)
│   └── e2e/             # End-to-end tests (test containers)
├── migrations/          # Database migration scripts
├── Dockerfile
├── requirements.txt
└── README.md
```

**Structure Decision**: Single project structure (Option 1) selected. This is a microservice with a clear single responsibility, so a single source directory with organized modules is appropriate. The structure separates concerns: models, services (by domain), API layer, and configuration. Tests are organized by type (unit, integration, e2e) and will run in appropriate containers per constitution requirements.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
