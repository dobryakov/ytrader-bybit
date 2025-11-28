# Implementation Plan: Grafana Monitoring Dashboard

**Branch**: `001-grafana-monitoring` | **Date**: 2025-01-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-grafana-monitoring/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

A Grafana monitoring dashboard service running in a separate Docker container that provides visual monitoring of the trading system. The dashboard displays recent trading signals, order execution status, model state and quality metrics, RabbitMQ queue health, system health status, WebSocket connection metrics, and event history. Grafana connects to PostgreSQL (read-only user), RabbitMQ Management API, and service REST APIs to collect monitoring data. Authentication is handled via basic auth (username/password) configurable via `.env` file. The service runs on a non-standard port (4700) and is accessible from external networks.

**Note on Trading Signals**: Phase 1 uses RabbitMQ Management API to monitor queue metrics as a proxy for trading signal activity. Phase 2 (future) will migrate to querying trading signals directly from PostgreSQL database once model-service implements signal persistence (tasks T089-T090 in model-service).

## Technical Context

**Language/Version**: Grafana (containerized, official Grafana image version 10.4.0 or latest stable)  
**Primary Dependencies**: 
- Grafana official Docker image (`grafana/grafana:10.4.0`)
- PostgreSQL data source plugin (built-in)
- HTTP/JSON data source (built-in) for RabbitMQ Management API and REST API endpoints
- Basic authentication (built-in Grafana auth, configurable via environment variables)

**Storage**: 
- PostgreSQL (read-only connection for querying trading signals, orders, execution_events, model_versions, model_quality_metrics tables)
- No local storage required (Grafana uses mounted volumes for dashboards/configs)

**Testing**: 
- Integration tests: Verify Grafana container starts, data sources connect, dashboards load
- Manual testing: Verify dashboard displays data correctly from all sources
- Container health checks: Verify Grafana /api/health endpoint

**Target Platform**: Linux server (Docker container)  
**Project Type**: single (monitoring service container)

**Performance Goals**: 
- Dashboard queries complete within 3 seconds for recent data (last 100-200 records)
- Dashboard auto-refresh every 60 seconds (configurable)
- Support up to 10 concurrent dashboard users without performance degradation

**Constraints**: 
- Must not require modifications to existing services (read-only access to existing data sources)
- Must handle data source connection failures gracefully (show connection status, continue with cached data)
- Authentication credentials must be configurable via `.env` file
- Must use non-standard port (4700) for Grafana UI access
- Read-only PostgreSQL user with limited permissions (only SELECT on required tables)

**Scale/Scope**: 
- Single Grafana instance serving monitoring dashboards
- 7 main dashboard panels: trading signals, orders, model state/metrics, queue monitoring, health status, WebSocket metrics, event history
- Monitoring ~5 services: ws-gateway, model-service, order-manager, postgres, rabbitmq
- Supporting 2-10 concurrent operators viewing dashboards

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Microservices Architecture ✅
- **Requirement**: All functionality MUST be organized as independent microservices running in Docker containers under a unified `docker-compose.yml`
- **Compliance**: Grafana runs as a separate Docker container in `docker-compose.yml` with clear single responsibility (monitoring and visualization)
- **Status**: PASS

### II. Shared Database Strategy ✅
- **Requirement**: All microservices MUST use a shared PostgreSQL database for data exchange and persistence. Database schema changes MUST be managed through reversible migrations whenever possible
- **Compliance**: 
  - Grafana connects to the shared PostgreSQL database using a read-only user (no schema changes)
  - No migrations required (read-only access to existing tables)
  - PostgreSQL migrations remain in `ws-gateway` service per constitution
- **Status**: PASS

### III. Inter-Service Communication ✅
- **Requirement**: Event-driven communication queues is PREFERRED, HTTP REST API MAY be used for data queries. All inter-service communication MUST be logged
- **Compliance**: 
  - Grafana uses HTTP REST API for health checks and model statistics (query-only, acceptable per constitution)
  - Grafana uses RabbitMQ Management API for queue monitoring (query-only)
  - Grafana reads from PostgreSQL (query-only)
  - All queries are read-only; no event publishing required
- **Status**: PASS

### IV. Testing Discipline ✅
- **Requirement**: Tests MUST run inside Docker containers. Unit tests run in service containers; API and e2e tests run in separate test containers
- **Compliance**: 
  - Integration tests will run in Docker container
  - Health check tests verify Grafana container functionality
  - Dashboard rendering tests verify data source connectivity
- **Status**: PASS

### V. Observability & Diagnostics ✅
- **Requirement**: All services MUST implement structured logging. Health checks MUST be implemented for monitoring
- **Compliance**: 
  - Grafana provides built-in health endpoint (`/api/health`)
  - Grafana logs are available via Docker Compose logs
  - This service IS observability (monitors other services)
- **Status**: PASS

### VI. Infrastructure Standards ✅
- **Requirement**: Docker Compose V2 MUST be used. The `version:` field MUST NOT be included. Commands MUST run inside containers. Services MUST use non-standard ports
- **Compliance**: 
  - Uses Docker Compose V2 (`docker compose`)
  - No `version:` field in docker-compose.yml
  - Runs in Docker container
  - Uses non-standard port 4700 for Grafana UI
- **Status**: PASS

### VII. Documentation & Configuration ✅
- **Requirement**: Configuration variables MUST NOT be hardcoded; they MUST be moved to `.env` with samples in `env.example`. README, quickstart, and specifications MUST be synchronized
- **Compliance**: 
  - All Grafana configuration (credentials, data source URLs, ports) via `.env` file
  - `env.example` will include Grafana configuration variables
  - Documentation will be updated (quickstart.md, README)
- **Status**: PASS

### Gate Status: ✅ ALL GATES PASSED

## Project Structure

### Documentation (this feature)

```text
specs/001-grafana-monitoring/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
grafana/
├── Dockerfile           # Grafana container definition (or use official image)
├── provisioning/        # Grafana provisioning configs
│   ├── datasources/     # Data source configurations (PostgreSQL, RabbitMQ, HTTP)
│   │   └── datasources.yml
│   └── dashboards/      # Dashboard JSON definitions
│       └── trading-system-monitoring.json
├── dashboards/          # Alternative: dashboard provisioning via files
│   └── trading-system-monitoring.json
└── grafana.ini          # Custom Grafana configuration (if needed)

tests/
└── integration/
    └── test_grafana.py  # Integration tests for Grafana container and dashboards
```

**Structure Decision**: Single monitoring service container. Grafana uses provisioning configuration files for data sources and dashboards, eliminating need for manual dashboard configuration. All configuration is version-controlled and applied on container startup. No custom application code required - Grafana container with provisioning files is sufficient.

## Future Phases

### Phase 2: Enhanced Trading Signals Dashboard (Post-Database Migration)

**Prerequisites**: 
- Model-service tasks T089-T090 completed (trading_signals table created, signal publisher persists to database)
- Database migration `010_create_trading_signals_table.sql` applied

**Goal**: Migrate trading signals dashboard panel from RabbitMQ Management API (queue metrics proxy) to direct PostgreSQL queries for detailed signal visualization.

**Implementation Tasks**:
1. Update Grafana dashboard SQL query for trading signals panel to query `trading_signals` table instead of using RabbitMQ queue metrics
2. Add detailed signal fields: signal_id, asset, side, price, confidence, timestamp, strategy_id, model_version, is_warmup, market_data_snapshot
3. Implement time-range filtering with proper indexes (timestamp DESC index)
4. Add signal filtering capabilities (by strategy_id, asset, model_version, is_warmup)
5. Update dashboard queries to use PostgreSQL data source for signals (remove RabbitMQ dependency for signals)
6. Maintain RabbitMQ Management API monitoring for queue health (separate panel, not replaced)

**Benefits**:
- Detailed signal information with all attributes (confidence, model_version, market_data_snapshot)
- Better query performance with indexed database queries
- Historical signal analysis capabilities
- Filtering and searching signals by various criteria
- Consistent data source pattern (PostgreSQL) for all trading data

**Dependencies**:
- Model-service must complete T089 (create trading_signals table migration)
- Model-service must complete T090 (extend signal publisher to persist signals)
- Database migration must be applied to production database

**Out of Scope for Phase 1**: 
- This phase is deferred until model-service implements signal persistence per research.md decision (Option B: persist signals to database)

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
