# Implementation Plan: Position Management Service

**Branch**: `001-position-management` | **Date**: 2025-01-27 | **Spec**: `/specs/001-position-management/spec.md`
**Input**: Feature specification from `/specs/001-position-management/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Create a centralized Position Manager microservice for portfolio position management. The service aggregates all positions, calculates portfolio metrics (total exposure, total PnL, portfolio value), provides REST API access, integrates with Risk Manager for portfolio-level limit checking, and manages position lifecycle (creation, update, validation, snapshots). Extracts position management functionality from Order Manager to achieve better separation of concerns, scalability, and extensibility.

**Implementation Status**:
- ✅ Phase 0: Research completed (`research.md`)
- ✅ Phase 1: Design completed (`data-model.md`, `contracts/openapi.yaml`, `quickstart.md`)
- ⏳ Phase 2: Task breakdown (pending `/speckit.tasks` command)

## Technical Context

**Language/Version**: Python 3.11+  
**Primary Dependencies**: FastAPI, pydantic-settings, structlog, asyncpg (or SQLAlchemy async), aio-pika (RabbitMQ), httpx (for external API calls)  
**Storage**: PostgreSQL (shared database, tables `positions`, `position_snapshots`)  
**Testing**: pytest, pytest-asyncio, pytest-rabbitmq, httpx, testcontainers  
**Target Platform**: Linux server (Docker container)  
**Project Type**: microservice (single service)  
**Performance Goals**: 
- Position data queries: <500ms for 95% of requests
- Portfolio metrics calculations: <1s for portfolios with up to 100 positions
- Position updates reflected within 2s (order execution) and 5s (market data)
- Support 1000 concurrent position updates per minute with zero data loss
- Support querying up to 500 distinct assets simultaneously  
**Constraints**: 
- Real-time position updates from multiple sources (order execution, market data)
- Data consistency with optimistic locking (version field, retry on conflict)
- Portfolio metrics caching (TTL: 5-10 seconds)
- 1-year retention for position snapshots
- Rate limiting per API key with different tiers (100-1000 req/min)  
**Scale/Scope**: 
- Up to 500 distinct assets
- 1000 concurrent position updates per minute
- Portfolio with up to 100 positions for metrics calculations
- Historical snapshots with 1-year retention

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Microservices Architecture ✅
- **Status**: PASS
- **Rationale**: New independent microservice (`position-manager`) with single responsibility (position management), independently deployable and scalable via Docker container

### II. Shared Database Strategy ✅
- **Status**: PASS
- **Rationale**: Uses shared PostgreSQL database (tables `positions`, `position_snapshots`). Database migrations will be managed by `ws-gateway` service per constitution (PostgreSQL migration ownership). Requires migration to add `current_price` and `version` fields to `positions` table before deployment.

### III. Inter-Service Communication ✅
- **Status**: PASS
- **Rationale**: Event-driven communication via RabbitMQ (consuming from `ws-gateway.position` and `order-manager.order_executed`, publishing to `position-manager.*` queues). REST API for queries (Model Service, Risk Manager). All communication logged with trace IDs.

### IV. Testing Discipline ✅
- **Status**: PASS
- **Rationale**: Tests run inside Docker containers (unit tests in service container, integration/e2e tests in separate test containers). Test structure: `position-manager/tests/unit/`, `integration/`, `e2e/`. Uses pytest, pytest-asyncio, testcontainers.

### V. Observability & Diagnostics ✅
- **Status**: PASS
- **Rationale**: Structured logging with structlog and trace IDs. Health check endpoint (`/health`). Logging of all position operations, validation results, rate limit exceedances. Diagnostic output for request flow tracking.

### VI. Infrastructure Standards ✅
- **Status**: PASS
- **Rationale**: Docker Compose V2 (`docker compose`), non-standard port (4800), explicit configuration via environment variables. Commands run inside containers. Base images reused when possible.

### VII. Documentation & Configuration ✅
- **Status**: PASS
- **Rationale**: Configuration via `.env` with `env.example` sample. Environment variable prefix: `POSITION_MANAGER_*`. README, quickstart, and specifications will be synchronized. Examples using `curl` provided.

**Overall Gate Status**: ✅ **PASS** - All constitution principles satisfied

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
position-manager/
├── Dockerfile
├── docker-compose.yml (add to main)
├── requirements.txt
├── README.md
├── env.example
├── src/
│   ├── main.py
│   ├── config/
│   │   ├── settings.py
│   │   ├── database.py
│   │   ├── rabbitmq.py
│   │   └── logging.py
│   ├── models/
│   │   ├── position.py
│   │   ├── portfolio.py
│   │   └── __init__.py
│   ├── services/
│   │   ├── position_manager.py
│   │   ├── portfolio_manager.py
│   │   ├── position_event_consumer.py
│   │   └── position_sync.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── positions.py
│   │   │   ├── portfolio.py
│   │   │   └── health.py
│   │   └── middleware/
│   │       ├── auth.py
│   │       └── logging.py
│   ├── consumers/
│   │   ├── order_position_consumer.py
│   │   └── websocket_position_consumer.py
│   ├── publishers/
│   │   └── position_event_publisher.py
│   ├── tasks/
│   │   ├── position_snapshot_task.py
│   │   ├── position_snapshot_cleanup_task.py
│   │   └── position_validation_task.py
│   └── utils/
│       └── tracing.py
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

**Structure Decision**: Single microservice project structure. All source code in `position-manager/src/` with clear separation: `config/` for configuration, `models/` for data models, `services/` for business logic, `api/` for REST endpoints, `consumers/` for RabbitMQ consumers, `publishers/` for event publishing, `tasks/` for background tasks, `utils/` for utilities. Tests organized by type: `unit/`, `integration/`, `e2e/`.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | No violations | All constitution principles satisfied |

## Generated Artifacts

### Phase 0: Research ✅

**File**: `specs/001-position-management/research.md`

**Contents**:
- Technology stack decisions (Python 3.11+, FastAPI, PostgreSQL, RabbitMQ)
- Design patterns (optimistic locking, caching, rate limiting)
- Integration patterns (RabbitMQ events, external API)
- Database schema decisions
- Performance optimization decisions
- Testing strategy decisions
- Security decisions

**Status**: Complete - All technical decisions documented with rationale and alternatives.

### Phase 1: Design ✅

**Files**:
1. **`specs/001-position-management/data-model.md`**
   - Entity definitions (Position, PositionSnapshot, Portfolio)
   - Field specifications, constraints, validation rules
   - State transitions and relationships
   - Database indexes and migration requirements
   - Data access patterns

2. **`specs/001-position-management/contracts/openapi.yaml`**
   - OpenAPI 3.0.3 specification
   - All REST API endpoints documented
   - Request/response schemas
   - Authentication and error handling
   - Query parameters and filtering

3. **`specs/001-position-management/quickstart.md`**
   - Setup instructions
   - Environment configuration
   - Database migration steps
   - Basic usage examples with curl
   - Event-driven integration examples
   - Testing instructions
   - Troubleshooting guide

**Status**: Complete - All design artifacts generated.

### Agent Context Update ✅

**File**: `.cursor/rules/specify-rules.mdc`

**Updates**:
- Added Python 3.11+ language context
- Added FastAPI, pydantic-settings, structlog, asyncpg, aio-pika, httpx framework context
- Added PostgreSQL database context

**Status**: Complete - Agent context updated for Cursor IDE.

## Next Steps

1. **Phase 2: Task Breakdown** - Run `/speckit.tasks` command to generate `tasks.md` with detailed implementation tasks
2. **Implementation** - Begin development based on tasks.md
3. **Database Migration** - Create migration in ws-gateway service for `current_price` and `version` fields
4. **Integration** - Integrate with Order Manager, Model Service, and Risk Manager

## Branch Information

- **Branch**: `001-position-management`
- **Plan Path**: `specs/001-position-management/plan.md`
- **Spec Path**: `specs/001-position-management/spec.md`
- **Generated Artifacts**:
  - `specs/001-position-management/research.md`
  - `specs/001-position-management/data-model.md`
  - `specs/001-position-management/contracts/openapi.yaml`
  - `specs/001-position-management/quickstart.md`
