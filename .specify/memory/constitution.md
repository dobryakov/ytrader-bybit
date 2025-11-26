<!--
Sync Impact Report:
Version change: 1.0.0 → 1.1.0 (minor: added PostgreSQL migration ownership clarification)
Modified principles: II. Shared Database Strategy (added PostgreSQL migration ownership clause)
Added sections: None
Templates requiring updates:
  ✅ plan-template.md - Constitution Check section aligns with principles
  ✅ spec-template.md - No changes needed (generic template)
  ✅ tasks-template.md - No changes needed (generic template)
  ✅ agent-file-template.md - No changes needed (generic template)
Follow-up TODOs: None
-->

# YTrader Constitution

## Core Principles

### I. Microservices Architecture
All functionality MUST be organized as independent microservices running in Docker containers under a unified `docker-compose.yml`. Each microservice MUST have a clear, single responsibility (e.g., WebSocket aggregation, order management, model training). Services MUST be independently deployable, scalable, and testable. New services MUST be added incrementally as the platform evolves.

### II. Shared Database Strategy
All microservices MUST use a shared PostgreSQL database for data exchange and persistence. Specialized databases (e.g., vector databases for ML, time-series databases) MAY be used for specific tasks when justified. Database schema changes MUST be managed through reversible migrations whenever possible; irreversible migrations require explicit approval.

**PostgreSQL Migration Ownership**: Since the PostgreSQL database is shared across multiple services in the project, the `ws-gateway` service is designated as the single source of truth for all PostgreSQL migrations and schema structure. All PostgreSQL migrations (including those for existing or future services) MUST be located in the `ws-gateway` service to ensure centralized management and consistency. Other database types (e.g., vector databases for ML models, specialized databases) MAY maintain their own migrations within their respective service containers.

### III. Inter-Service Communication
Event-driven communication queues is the PREFERRED method for inter-service messaging, with events organized by entity and class for scalability. HTTP REST API MAY be used for data queries or development convenience. All inter-service communication MUST be logged with full request/response bodies or major attributes, including trace IDs for request flow tracking.

### IV. Testing Discipline (NON-NEGOTIABLE)
Tests MUST run inside Docker containers, never on the host. Unit tests run in service containers; API and e2e tests run in separate test containers connected to the main `docker-compose.yml`. Test containers MUST be rebuilt when dependencies change. After completing each task, relevant automated tests MUST be executed. If using Playwright, browsers MUST be installed in the test container using proper base images.

### V. Observability & Diagnostics
All services MUST implement structured logging for incoming/outgoing HTTP requests, WebSocket messages, and key operations. Health checks MUST be implemented for monitoring key functions. Diagnostic output and trace IDs MUST be included to track request flow through logs. All available logs (scripts, tests, proxies, browser console, docker-compose) MUST be used to identify error causes.

### VI. Infrastructure Standards
Docker Compose V2 (`docker compose`) MUST be used instead of V1 (`docker-compose`). The `version:` field MUST NOT be included in `docker-compose.yml`. Commands MUST run inside containers whenever possible (exceptions: `docker`, `compose`, `curl`, `sh`). Services MUST use non-standard ports (not 80, 8080, 443) with explicit configuration. Base images SHOULD be reused across containers when possible.

### VII. Documentation & Configuration
Configuration variables MUST NOT be hardcoded; they MUST be moved to `.env` with samples in `env.example` (without leading dot). After making changes, README, quickstart, and specifications MUST be synchronized. Convenient examples using `curl` and shell scripts MUST be provided. Documentation for deploying from scratch (container builds, migrations, utilities, data generation) MUST be maintained and kept up-to-date.

## Infrastructure & Containers

All services MUST be deployed in Docker and managed via `docker-compose.yml`. When services include both frontend and backend, the frontend container serves browser requests and proxies them to the backend. WebSocket connections SHOULD be passed to the frontend container and proxied to the backend container when possible. After making changes, evaluate whether a container rebuild (`build`) is required instead of just a restart (`run`). Periodic tasks (cron jobs) MUST be documented with ready-to-use `crontab` syntax for manual execution and, if possible, automated within a dedicated `scheduler` container.

## Development Workflow

When modifying services, API contracts MUST be validated and impact on related components assessed. SDKs and programmatic clients MUST be verified to remain up-to-date whenever services change. Health checks based on `curl` MUST ensure `curl` is included in the Docker image. Use MCP servers (e.g., context7) to search for libraries and utilities. Always use the latest Bybit documentation and SDK, and leverage MCP context7 for making informed decisions.

## Governance

This constitution supersedes all other development practices. All PRs and reviews MUST verify compliance with these principles. Amendments require documentation of the rationale, approval, and a migration plan if breaking changes are involved. Complexity additions MUST be justified with simpler alternatives considered and rejected for documented reasons.

**Version**: 1.1.0 | **Ratified**: 2025-11-25 | **Last Amended**: 2025-01-27
