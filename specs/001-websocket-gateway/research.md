# Research: WebSocket Gateway Technology Decisions

**Feature**: WebSocket Gateway for Bybit Data Aggregation and Routing  
**Date**: 2025-11-25  
**Purpose**: Resolve technical clarification items from implementation plan

## Research Questions

### 1. WebSocket Client Library Selection

**Question**: Should we use `websockets` or `aiohttp` for WebSocket client connections to Bybit?

**Decision**: `websockets` library

**Rationale**:
- `websockets` is the standard Python library for async WebSocket connections, specifically designed for asyncio
- Lightweight and optimized for WebSocket protocol (RFC 6455)
- Excellent async/await support with clean API
- Well-maintained and widely used in production systems
- Better performance for WebSocket-only use cases compared to `aiohttp` (which is a full HTTP client/server framework)
- Native support for connection management, ping/pong heartbeats, and reconnection patterns
- Works seamlessly with FastAPI's async ecosystem

**Alternatives Considered**:
- `aiohttp`: Full-featured HTTP client/server framework that also supports WebSocket, but heavier and more complex for WebSocket-only use case
- `pybit`: Bybit's official Python SDK - evaluated but may not provide sufficient control over connection lifecycle and reconnection logic required by our specifications

**Implementation Notes**:
- Use `websockets` version 12.0+ for Python 3.11+ compatibility
- Implement custom reconnection logic with exponential backoff (max 30 seconds per requirement)
- Use `websockets` ping/pong mechanism for heartbeat maintenance

---

### 2. RabbitMQ Client Library Selection

**Question**: Should we use `pika` (sync) or `aio-pika` (async) for RabbitMQ operations?

**Decision**: `aio-pika` (async)

**Rationale**:
- `aio-pika` is the async version of `pika`, designed for asyncio
- Consistent with our async architecture (FastAPI, async WebSocket, async database)
- Better performance for concurrent operations (publishing multiple events simultaneously)
- Non-blocking I/O prevents event processing delays when publishing to queues
- Seamless integration with async/await patterns throughout the codebase
- Supports connection pooling and async context managers
- Better resource utilization under high event throughput

**Alternatives Considered**:
- `pika`: Synchronous library, would require thread pools or blocking operations, creating bottlenecks in async event processing pipeline
- `celery`: Task queue framework - overkill for simple message publishing, adds unnecessary complexity

**Implementation Notes**:
- Use `aio-pika` version 9.0+ for Python 3.11+ compatibility
- Implement connection pooling for multiple queue publishers
- Use async context managers for proper resource cleanup
- Configure queue durability and retention policies (24 hours or 100K messages)

---

### 3. PostgreSQL Database Driver Selection

**Question**: Should we use `asyncpg` or `SQLAlchemy` with async driver for PostgreSQL operations?

**Decision**: `asyncpg`

**Rationale**:
- `asyncpg` is the fastest async PostgreSQL driver for Python, written in Cython
- Direct async protocol implementation (not a wrapper around sync driver)
- Lower latency and higher throughput for database operations
- Minimal overhead - critical for high-frequency event processing
- Simple, direct API well-suited for straightforward CRUD operations
- Better performance for write-heavy workloads (balance persistence, subscription state)
- Sufficient for our use case (subscriptions, balance data) without ORM complexity

**Alternatives Considered**:
- `SQLAlchemy` with async driver: More feature-rich ORM, better for complex queries and relationships, but adds overhead and complexity
- `psycopg3` async: Good alternative but `asyncpg` has better performance benchmarks for async workloads

**Implementation Notes**:
- Use `asyncpg` version 0.29+ for Python 3.11+ compatibility
- Implement connection pooling for concurrent database operations
- Use prepared statements for frequently executed queries (subscription lookups, balance updates)
- Handle database write failures gracefully (log and continue, per requirement FR-017)

---

### 4. Bybit WebSocket API Integration

**Question**: How should we integrate with Bybit WebSocket API?

**Decision**: Direct WebSocket connection using `websockets` library with custom Bybit protocol implementation

**Rationale**:
- Bybit WebSocket API uses standard WebSocket protocol with JSON messages
- Custom implementation provides full control over connection lifecycle, reconnection, and subscription management
- Allows precise implementation of requirements (automatic resubscription, heartbeat, error handling)
- Better observability and logging of all WebSocket interactions
- No dependency on potentially outdated or incompatible SDK versions

**Alternatives Considered**:
- Official Bybit Python SDK (`pybit`): May not provide sufficient control over connection management and may have different reconnection behavior than required
- Third-party WebSocket wrappers: Add unnecessary abstraction layer

**Implementation Notes**:
- Follow Bybit WebSocket API documentation for authentication (API key signature)
- Implement subscription message format per Bybit specifications
- Handle Bybit-specific message formats (topic subscriptions, event types)
- Support both mainnet and testnet environments via configuration
- Implement proper authentication flow (signature generation, auth message)

---

### 5. Event Queue Organization Strategy

**Question**: How should events be organized in RabbitMQ queues?

**Decision**: Separate queues per event class (trades, order_status, balances, tickers, orderbook)

**Rationale**:
- Aligns with requirement FR-010: "queues organized by event class"
- Enables independent scaling and processing of different event types
- Allows different retention policies per event class if needed in future
- Simplifies subscriber routing (subscribers consume from specific queues)
- Better observability (queue metrics per event type)
- Supports fan-out delivery (multiple subscribers per queue) naturally

**Alternatives Considered**:
- Single queue with event type routing: Would require message routing logic and complicate subscriber filtering
- Topic-based routing: More complex, not needed for current requirements

**Implementation Notes**:
- Queue naming convention: `ws-gateway.{event_class}` (e.g., `ws-gateway.trades`, `ws-gateway.order_status`)
- Configure queue durability and retention (24 hours or 100K messages)
- Use RabbitMQ exchanges for fan-out if multiple subscribers per queue
- Include event metadata (event_id, event_type, timestamp) in message payload

---

## Summary of Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Language | Python | 3.11+ | Async/await support, ecosystem maturity |
| WebSocket Client | `websockets` | 12.0+ | Standard async WebSocket library |
| REST API | `FastAPI` | 0.104+ | Async support, OpenAPI generation |
| Message Queue Client | `aio-pika` | 9.0+ | Async RabbitMQ client |
| Database Driver | `asyncpg` | 0.29+ | Fastest async PostgreSQL driver |
| HTTP Client | `httpx` | 0.25+ | Async HTTP client (if needed for Bybit REST API) |
| Configuration | `pydantic-settings` | 2.0+ | Type-safe configuration management |
| Logging | `structlog` | 23.2+ | Structured logging with trace IDs |
| Testing | `pytest` + `pytest-asyncio` | Latest | Async test support |

## Dependencies Summary

**Core Dependencies**:
- `websockets>=12.0` - WebSocket client
- `fastapi>=0.104.0` - REST API framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `aio-pika>=9.0.0` - RabbitMQ async client
- `asyncpg>=0.29.0` - PostgreSQL async driver
- `pydantic-settings>=2.0.0` - Configuration management
- `structlog>=23.2.0` - Structured logging
- `httpx>=0.25.0` - Async HTTP client (optional, for Bybit REST API if needed)

**Development Dependencies**:
- `pytest>=7.4.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async test support
- `pytest-mock>=3.11.0` - Mocking utilities
- `black>=23.0.0` - Code formatting
- `mypy>=1.5.0` - Type checking

## Next Steps

All technical clarifications resolved. Proceed to Phase 1: Design & Contracts.

