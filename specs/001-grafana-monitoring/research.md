# Research: Grafana Monitoring Dashboard

**Feature**: Grafana Monitoring Dashboard  
**Date**: 2025-01-27  
**Phase**: Phase 0 - Outline & Research

## Overview

This document consolidates research findings for implementing a Grafana monitoring dashboard service. All "NEEDS CLARIFICATION" items from the Technical Context section are resolved here.

---

## Research Findings

### 1. Grafana Docker Image Version

**Decision**: Use `grafana/grafana:latest` or `grafana/grafana:10.4.0` (latest stable LTS version)

**Rationale**: 
- Grafana 10.x series is the current stable LTS (Long Term Support) version as of 2025
- Official Grafana Docker image provides all necessary plugins and dependencies
- Using `latest` tag allows automatic updates, but pinning to specific version (10.4.0) is recommended for production stability
- Official image includes PostgreSQL, HTTP/JSON, and Prometheus data sources out of the box

**Alternatives Considered**:
- Grafana Enterprise: Not needed for basic monitoring requirements
- Custom Grafana build: Unnecessary complexity, official image is sufficient

**Implementation Notes**:
- Use official `grafana/grafana` image from Docker Hub
- Pin to specific version for production: `grafana/grafana:10.4.0`
- Container runs on port 3000 internally, map to external port 4700 via docker-compose

---

### 2. RabbitMQ Queue Monitoring Approach

**Decision**: Use Grafana's built-in HTTP/JSON data source to query RabbitMQ Management API

**Rationale**:
- RabbitMQ Management API provides RESTful endpoints for queue metrics (`/api/queues`)
- Grafana HTTP data source can query JSON APIs directly
- No dedicated RabbitMQ plugin needed - standard HTTP data source supports this use case
- Management API is enabled by default in `rabbitmq:3-management-alpine` image already used in docker-compose.yml
- Provides queue metrics: message count, publish rate, consume rate, consumer count

**Alternatives Considered**:
- Prometheus + RabbitMQ exporter: Overhead of adding Prometheus for single use case
- Custom Grafana plugin: Unnecessary complexity, HTTP data source is sufficient
- Direct RabbitMQ protocol: Management API is simpler and provides all needed metrics

**Implementation Notes**:
- RabbitMQ Management API endpoint: `http://rabbitmq:15672/api/queues`
- Authentication: Basic auth using `RABBITMQ_USER` and `RABBITMQ_PASSWORD` from `.env`
- Configure HTTP data source in Grafana provisioning with authentication headers
- Query all queues (including system queues like `amq.*`) via Management API

---

### 3. PostgreSQL Data Source Configuration

**Decision**: Use Grafana's built-in PostgreSQL data source plugin

**Rationale**:
- PostgreSQL data source is included in official Grafana image
- Supports SQL queries directly in dashboard panels
- Read-only user access is sufficient for monitoring queries
- No additional plugins or configuration required

**Implementation Notes**:
- Create read-only PostgreSQL user for Grafana with SELECT permissions on required tables
- Configure data source in Grafana provisioning with connection string
- Tables to query: `execution_events`, `orders`, `model_versions`, `model_quality_metrics`, `subscriptions`
- Credentials stored in `.env` file (separate from service database credentials)

---

### 4. Trading Signals Data Source

**Decision**: Query recent trading signals from RabbitMQ queue metrics OR create a temporary solution to persist recent signals

**Rationale**:
- Trading signals are published to RabbitMQ queue `model-service.trading_signals` but not persisted to database
- Two approaches:
  1. **Option A (Preferred for Phase 1)**: Query RabbitMQ Management API to see queue message count and recent message headers (limited signal details)
  2. **Option B**: Create a lightweight consumer service that persists recent signals (last 100-200) to a temporary table
  
**Alternatives Considered**:
- Modify model-service to persist signals: Out of scope for Phase 1 (per spec: "no major modifications to existing services")
- Query queue directly: RabbitMQ Management API provides limited message content access (only headers/metadata)

**Implementation Notes**:
- **Phase 1 Approach**: Use RabbitMQ Management API to monitor queue health and message count
- For detailed signal viewing, consider Option B in future phase or extend model-service to optionally persist signals
- Dashboard will show queue metrics (message count, consumption rate) as proxy for signal activity

---

### 5. HTTP Data Source for REST API Endpoints

**Decision**: Use Grafana's built-in HTTP/JSON data source for service health checks and model statistics

**Rationale**:
- Grafana HTTP data source can query any REST API endpoint
- Supports authentication via headers (API keys)
- Can parse JSON responses for dashboard panels
- No additional plugins required

**Implementation Notes**:
- Configure separate HTTP data source entries for:
  - ws-gateway health endpoint: `http://ws-gateway:4400/health`
  - model-service health/statistics: `http://model-service:4500/health` and `/api/v1/models/statistics` (if available)
  - order-manager health: `http://order-manager:4600/health`
- Authentication: Use API keys from `.env` file (e.g., `WS_GATEWAY_API_KEY`) in HTTP headers
- Response parsing: Use JSON path expressions in dashboard queries

---

### 6. Grafana Authentication Configuration

**Decision**: Use Grafana's built-in basic authentication (username/password)

**Rationale**:
- Built-in Grafana auth is sufficient for single admin user access
- Credentials configurable via environment variables (`GF_SECURITY_ADMIN_USER`, `GF_SECURITY_ADMIN_PASSWORD`)
- Simple to configure, no external auth provider needed for Phase 1

**Alternatives Considered**:
- LDAP/OAuth: Overkill for single admin user
- API key only: Less secure, basic auth provides UI access control

**Implementation Notes**:
- Configure admin credentials via environment variables in docker-compose.yml
- Values from `.env` file: `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` (default: admin/admin)
- Authentication required for all dashboard access

---

### 7. Dashboard Provisioning Strategy

**Decision**: Use Grafana provisioning (datasources.yml and dashboard JSON files) for automated configuration

**Rationale**:
- Provisioning allows automated setup of data sources and dashboards on container startup
- Version-controlled configuration (YAML/JSON files in git)
- No manual dashboard creation required
- Eliminates need for Grafana UI configuration

**Alternatives Considered**:
- Manual configuration via UI: Error-prone, not version-controlled, requires manual setup
- Grafana API calls: More complex, provisioning is simpler

**Implementation Notes**:
- Create `grafana/provisioning/datasources/datasources.yml` for data source configuration
- Create `grafana/provisioning/dashboards/dashboards.yml` for dashboard provisioning
- Place dashboard JSON files in `grafana/dashboards/` directory
- Mount provisioning directories as volumes in docker-compose.yml

---

### 8. Grafana Configuration File Strategy

**Decision**: Use environment variables for Grafana configuration where possible; custom `grafana.ini` only if needed

**Rationale**:
- Grafana supports configuration via environment variables (prefix: `GF_`)
- Environment variables are easier to manage via `.env` file
- Only use `grafana.ini` for settings not available via environment variables

**Implementation Notes**:
- Configure via environment variables:
  - `GF_SECURITY_ADMIN_USER` and `GF_SECURITY_ADMIN_PASSWORD` for admin credentials
  - `GF_SERVER_HTTP_PORT=3000` (internal port, external mapping via docker-compose)
  - `GF_INSTALL_PLUGINS` (if any plugins needed)
- Mount `grafana.ini` only if custom settings required beyond environment variables

---

## Technology Stack Summary

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Grafana Image | `grafana/grafana` | 10.4.0 (or latest stable) | Official image with all required data sources |
| PostgreSQL Data Source | Built-in | Included | Standard PostgreSQL plugin |
| HTTP/JSON Data Source | Built-in | Included | For REST API and RabbitMQ Management API |
| RabbitMQ Monitoring | HTTP data source + Management API | N/A | RabbitMQ Management API endpoint |
| Authentication | Basic Auth | Built-in | Single admin user via environment variables |
| Configuration | Provisioning + Environment Variables | N/A | Automated setup via files and env vars |

---

## Dependencies Summary

**Container Dependencies**:
- `grafana/grafana:10.4.0` - Official Grafana Docker image
- Existing services: `postgres`, `rabbitmq`, `ws-gateway`, `model-service`, `order-manager`

**Network Dependencies**:
- Access to PostgreSQL (read-only user)
- Access to RabbitMQ Management API (port 15672)
- Access to service REST APIs (ports 4400, 4500, 4600)

**Configuration Dependencies**:
- PostgreSQL read-only user credentials (`.env`)
- RabbitMQ Management API credentials (`.env`)
- Service API keys for authenticated endpoints (`.env`)
- Grafana admin credentials (`.env`)

---

## Resolved Clarifications

✅ **Grafana Version**: Use `grafana/grafana:10.4.0` (or latest stable)  
✅ **RabbitMQ Data Source**: HTTP data source querying Management API  
✅ **Trading Signals Display**: Use RabbitMQ queue metrics (message count, rates) as proxy; detailed signal viewing may require future enhancement  
✅ **Authentication Method**: Basic auth (username/password) via environment variables  
✅ **Configuration Approach**: Provisioning files + environment variables  
✅ **Dashboard Setup**: Automated via provisioning, no manual UI configuration required

---

## Next Steps

1. Generate `data-model.md` with entities for monitoring data
2. Generate API contracts for data source configurations
3. Generate `quickstart.md` with deployment instructions
4. Update agent context with Grafana technology

