# Research: Position Management Service

**Date**: 2025-01-27  
**Feature**: Position Management Service  
**Branch**: `001-position-management`

## Overview

This document consolidates research findings and technical decisions for implementing the Position Management Service. All technical choices are documented with rationale and alternatives considered.

## Technology Stack Decisions

### Decision: Python 3.11+ with FastAPI

**Rationale**:
- FastAPI provides async/await support for high-performance I/O operations
- Automatic OpenAPI documentation generation
- Type hints and Pydantic validation for data integrity
- Excellent performance (comparable to Node.js and Go)
- Strong ecosystem for async database and messaging libraries

**Alternatives Considered**:
- **Node.js/Express**: Rejected - team expertise in Python, existing services use Python
- **Go**: Rejected - higher development complexity, existing codebase is Python
- **Django**: Rejected - synchronous by default, FastAPI better for async I/O

### Decision: PostgreSQL with asyncpg or SQLAlchemy async

**Rationale**:
- Shared database strategy per constitution (all services use shared PostgreSQL)
- Async support required for non-blocking I/O
- asyncpg provides best performance for async PostgreSQL operations
- SQLAlchemy async provides ORM benefits if complex queries needed

**Alternatives Considered**:
- **MongoDB**: Rejected - shared database strategy requires PostgreSQL
- **Redis for primary storage**: Rejected - PostgreSQL required for ACID guarantees and complex queries
- **SQLAlchemy sync**: Rejected - async required for performance

**Decision**: Use asyncpg for direct async database access (better performance) with optional SQLAlchemy async for complex ORM needs.

### Decision: RabbitMQ with aio-pika

**Rationale**:
- Event-driven communication preferred per constitution
- aio-pika provides async RabbitMQ client for Python
- Supports async/await patterns
- Reliable message delivery and acknowledgment

**Alternatives Considered**:
- **Kafka**: Rejected - overkill for current scale, RabbitMQ already in use
- **Redis Pub/Sub**: Rejected - less reliable than RabbitMQ for critical events
- **pika (sync)**: Rejected - async required for non-blocking operations

### Decision: structlog for Structured Logging

**Rationale**:
- Structured logging required per constitution (Principle V)
- Trace ID support for request flow tracking
- JSON output for log aggregation systems
- Context propagation for async operations

**Alternatives Considered**:
- **Python logging**: Rejected - lacks structured output and context propagation
- **loguru**: Rejected - structlog better for structured logging in microservices

## Design Patterns

### Decision: Optimistic Locking with Version Field

**Rationale**:
- Required per spec (FR-014) to handle concurrent position updates
- Version field increments on each update
- Retry strategy: up to 3 retries with exponential backoff (100ms, 200ms, 400ms)
- Prevents lost updates in high-concurrency scenarios

**Implementation Pattern**:
```python
# Pseudocode
def update_position(position_id, updates):
    for attempt in range(3):
        position = get_position(position_id)
        if position.version != expected_version:
            wait(100 * (2 ** attempt) ms)  # Exponential backoff
            continue
        position.update(updates)
        position.version += 1
        if save_position(position):
            return success
    raise OptimisticLockException()
```

**Alternatives Considered**:
- **Pessimistic locking**: Rejected - database-level locks reduce concurrency
- **No locking**: Rejected - risk of lost updates with concurrent modifications

### Decision: In-Memory Metrics Caching with TTL

**Rationale**:
- Portfolio metrics calculation can be expensive (aggregation across all positions)
- Cache TTL: 5-10 seconds (configurable via `POSITION_MANAGER_METRICS_CACHE_TTL`)
- Cache invalidation on position updates
- Optional Redis for distributed caching (future enhancement)

**Implementation Pattern**:
```python
# Pseudocode
cache = {}
cache_ttl = 10  # seconds

def get_portfolio_metrics():
    cache_key = "portfolio_metrics"
    if cache_key in cache and not cache_expired(cache_key):
        return cache[cache_key]
    metrics = calculate_metrics_from_db()
    cache[cache_key] = (metrics, time.now() + cache_ttl)
    return metrics

def invalidate_cache():
    cache.clear()
```

**Alternatives Considered**:
- **No caching**: Rejected - performance requirements (<1s for portfolio metrics) require caching
- **Database caching table**: Rejected - in-memory faster, database adds latency
- **Redis from start**: Rejected - in-memory sufficient for single-instance, Redis adds complexity

### Decision: Rate Limiting Per API Key with Tiers

**Rationale**:
- Required per spec for API security
- Different tiers: Model Service (100 req/min), Risk Manager (200 req/min), UI (1000 req/min)
- Configuration via environment variables: `POSITION_MANAGER_RATE_LIMIT_OVERRIDES`
- HTTP 429 response on limit exceedance with `Retry-After` header

**Implementation Pattern**:
- Use `slowapi` or `fastapi-limiter` for FastAPI rate limiting
- Store rate limit state in memory (or Redis for distributed)
- Per-API-Key tracking with sliding window or token bucket algorithm

**Alternatives Considered**:
- **No rate limiting**: Rejected - security requirement per spec
- **Global rate limit**: Rejected - different consumers have different needs
- **IP-based rate limiting**: Rejected - API key-based more appropriate for service-to-service communication

## Integration Patterns

### Decision: RabbitMQ Event Consumption Pattern

**Rationale**:
- Event-driven communication preferred per constitution
- Consume from multiple queues: `ws-gateway.position`, `order-manager.order_executed`
- Publish to queues: `position-manager.position_updated`, `position-manager.portfolio_updated`, `position-manager.position_snapshot_created`
- Async consumers with aio-pika for non-blocking message processing

**Implementation Pattern**:
```python
# Pseudocode
async def consume_websocket_events():
    async with aio_pika.connect_robust() as connection:
        channel = await connection.channel()
        queue = await channel.declare_queue("ws-gateway.position")
        async for message in queue:
            try:
                await process_websocket_position_event(message)
                await message.ack()
            except Exception as e:
                await message.nack(requeue=True)
                log.error("Failed to process event", error=e)
```

**Alternatives Considered**:
- **REST API polling**: Rejected - inefficient, event-driven preferred
- **WebSocket direct connection**: Rejected - RabbitMQ provides better reliability and decoupling

### Decision: External Price API Integration (Bybit REST API)

**Rationale**:
- Required when `markPrice` missing or stale in WebSocket events
- Bybit REST API `/v5/market/tickers` endpoint for current price
- Retry strategy: 3 retries with exponential backoff (1s, 2s, 4s)
- Timeout: 5s per request
- Fallback to last known price if all retries fail

**Implementation Pattern**:
```python
# Pseudocode
async def get_current_price(asset, retries=3):
    for attempt in range(retries):
        try:
            response = await httpx.get(
                f"https://api.bybit.com/v5/market/tickers?symbol={asset}",
                timeout=5.0
            )
            return parse_price(response.json())
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                return get_last_known_price(asset) or None
```

**Alternatives Considered**:
- **No fallback**: Rejected - system must handle missing prices gracefully
- **Synchronous requests**: Rejected - async required for non-blocking I/O
- **Different exchange API**: Rejected - Bybit is the trading exchange in use

## Database Schema Decisions

### Decision: Add `current_price` and `version` Fields to `positions` Table

**Rationale**:
- `current_price`: Store latest `markPrice` from WebSocket events for portfolio calculations
- `version`: Required for optimistic locking (FR-014)
- Migration must be created in `ws-gateway` service per constitution (PostgreSQL migration ownership)

**Migration Location**: `ws-gateway/migrations/` (per constitution Principle II)

**Schema Changes**:
```sql
ALTER TABLE positions 
ADD COLUMN current_price DECIMAL(20, 8) NULL,
ADD COLUMN version INTEGER NOT NULL DEFAULT 1;

CREATE INDEX idx_positions_current_price ON positions(current_price);
CREATE INDEX idx_positions_version ON positions(version);
```

**Alternatives Considered**:
- **Separate price table**: Rejected - adds complexity, current_price is position attribute
- **No version field**: Rejected - optimistic locking requires version tracking

### Decision: Position Snapshot Retention (1 year)

**Rationale**:
- Required per spec: 1-year retention for historical tracking
- Cleanup job runs on service startup
- Deletes snapshots older than `POSITION_MANAGER_SNAPSHOT_RETENTION_DAYS` (365 days)

**Implementation Pattern**:
```python
# Pseudocode
async def cleanup_old_snapshots():
    cutoff_date = datetime.now() - timedelta(days=365)
    deleted = await db.execute(
        "DELETE FROM position_snapshots WHERE created_at < :cutoff",
        {"cutoff": cutoff_date}
    )
    log.info(f"Deleted {deleted} old snapshots")
```

**Alternatives Considered**:
- **No retention limit**: Rejected - unbounded growth would fill database
- **Archive to cold storage**: Considered but rejected for MVP - can be added later

## Performance Optimization Decisions

### Decision: Database Indexes for Position Queries

**Rationale**:
- Performance requirements: <500ms for 95% of position queries
- Composite index on `(asset, mode)` for position identity lookups
- Index on `current_price` for portfolio calculations
- Index on `version` for optimistic locking checks

**Indexes Required**:
```sql
CREATE INDEX idx_positions_asset ON positions(asset);
CREATE INDEX idx_positions_mode ON positions(mode);
CREATE INDEX idx_positions_asset_mode ON positions(asset, mode);
CREATE INDEX idx_positions_current_price ON positions(current_price);
CREATE INDEX idx_positions_version ON positions(version);
```

**Alternatives Considered**:
- **No indexes**: Rejected - performance requirements cannot be met without indexes
- **Full table scans**: Rejected - unacceptable performance for 500+ assets

### Decision: Async Database Queries

**Rationale**:
- Non-blocking I/O required for handling 1000 concurrent updates per minute
- asyncpg or SQLAlchemy async for async database operations
- Connection pooling for efficient resource usage

**Alternatives Considered**:
- **Synchronous queries**: Rejected - blocks event loop, poor concurrency
- **Thread pool**: Rejected - async/await more efficient for I/O-bound operations

## Testing Strategy Decisions

### Decision: Test Structure (unit/integration/e2e)

**Rationale**:
- Per constitution Principle IV: tests run in Docker containers
- Unit tests in service container, integration/e2e in separate test containers
- testcontainers for isolating database and RabbitMQ in tests

**Test Structure**:
```
position-manager/tests/
├── unit/           # Fast, no external dependencies
├── integration/    # Database and RabbitMQ (testcontainers)
└── e2e/            # Full flow tests
```

**Alternatives Considered**:
- **All tests in one container**: Rejected - violates constitution, slower execution
- **Tests on host**: Rejected - violates constitution Principle IV

## Security Decisions

### Decision: API Key Authentication

**Rationale**:
- Required per spec for REST API security
- Header: `X-API-Key: <api_key>`
- Configuration: `POSITION_MANAGER_API_KEY`
- Validation on each request

**Implementation Pattern**:
```python
# Pseudocode
async def api_key_auth(request: Request):
    api_key = request.headers.get("X-API-Key")
    if api_key != settings.POSITION_MANAGER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key
```

**Alternatives Considered**:
- **No authentication**: Rejected - security requirement
- **OAuth2/JWT**: Rejected - overkill for service-to-service communication, API key simpler
- **Basic auth**: Rejected - API key more appropriate for programmatic access

## Summary

All technical decisions have been made based on:
1. Constitution principles (shared database, event-driven communication, testing in containers)
2. Feature specification requirements (optimistic locking, rate limiting, caching)
3. Performance goals (<500ms queries, <1s portfolio metrics, 1000 concurrent updates/min)
4. Existing architecture (Python services, PostgreSQL, RabbitMQ)

No "NEEDS CLARIFICATION" items remain. All technology choices are justified with rationale and alternatives considered.

