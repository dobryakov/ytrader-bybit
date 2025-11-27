# Tasks: Order Manager Microservice

**Input**: Design documents from `/specs/004-order-manager/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Per Constitution Principle IV (Testing Discipline), tests MUST run inside Docker containers. Test tasks are organized separately to maintain clear separation between implementation and testing phases. Unit tests run in service containers; integration and e2e tests run in separate test containers connected to the main `docker-compose.yml`. After completing each implementation task, relevant automated tests MUST be executed.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `order-manager/src/`, `order-manager/tests/` at repository root
- Paths shown below follow the structure defined in plan.md

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project structure per implementation plan in order-manager/
- [X] T002 Initialize Python project with dependencies in order-manager/requirements.txt (FastAPI, httpx, aio-pika, asyncpg, pydantic-settings, structlog, pytest, pytest-asyncio)
- [X] T003 [P] Create Dockerfile for order-manager service in order-manager/Dockerfile
- [X] T004 [P] Create docker-compose.yml service definition for order-manager (port 4600)
- [X] T005 [P] Configure environment variables in env.example (ORDERMANAGER_*, BYBIT_*, database, RabbitMQ, WebSocket gateway, order execution, risk limits, retry, order type selection, position management configs)
- [X] T006 [P] Create README.md in order-manager/README.md with setup instructions
- [X] T007 [P] Configure logging infrastructure in order-manager/src/config/logging.py (structlog with trace IDs)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T008 Setup database connection pool in order-manager/src/config/database.py (asyncpg connection pool)
- [X] T009 [P] Implement configuration management in order-manager/src/config/settings.py (pydantic-settings with all configuration categories from research.md)
- [X] T010 [P] Create base models structure in order-manager/src/models/__init__.py
- [X] T011 [P] Implement trace ID utilities in order-manager/src/utils/tracing.py (trace ID generation and propagation)
- [X] T012 [P] Implement Bybit REST API client wrapper in order-manager/src/utils/bybit_client.py (httpx async client with HMAC-SHA256 authentication, retry logic with exponential backoff for 429 errors)
- [X] T013 [P] Setup FastAPI application structure in order-manager/src/api/main.py (FastAPI app initialization, middleware setup)
- [X] T014 [P] Implement API key authentication middleware in order-manager/src/api/middleware/auth.py (X-API-Key header validation)
- [X] T015 [P] Setup RabbitMQ connection in order-manager/src/config/rabbitmq.py (aio-pika connection and channel management)
- [X] T016 Create service entry point in order-manager/src/main.py (service startup, dependency initialization, graceful shutdown)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Execute Trading Signals (Priority: P1) 🎯 MVP

**Goal**: Receive high-level trading signals from the model service and execute them as actual orders on Bybit exchange, ensuring proper order creation, modification, and cancellation based on current balance and open positions.

**Independent Test**: Can be fully tested by sending a trading signal to the service and verifying that an appropriate order is created on Bybit and stored in the database with correct status.

### Implementation for User Story 1

- [X] T017 [P] [US1] Create Order model in order-manager/src/models/order.py (Order entity with all fields from data-model.md)
- [X] T018 [P] [US1] Create Trading Signal model in order-manager/src/models/trading_signal.py (in-memory structure for consumed signals)
- [X] T019 [P] [US1] Create Signal-Order Relationship model in order-manager/src/models/signal_order_rel.py (SignalOrderRelationship entity)
- [X] T020 [P] [US1] Create Position model in order-manager/src/models/position.py (Position entity with one-way and hedge-mode support)
- [X] T021 [US1] Implement order type selector service in order-manager/src/services/order_type_selector.py (market vs limit selection logic per research.md decisions)
- [X] T022 [US1] Implement quantity calculator service in order-manager/src/services/quantity_calculator.py (amount to quantity conversion with tick size/lot size precision per research.md)
- [X] T023 [US1] Implement position manager service in order-manager/src/services/position_manager.py (position query, update, validation logic)
- [X] T024 [US1] Implement risk manager service in order-manager/src/services/risk_manager.py (risk limits enforcement: max exposure, max order size, position size limits)
- [X] T025 [US1] Implement signal processor service in order-manager/src/services/signal_processor.py (signal validation, per-symbol FIFO queue, order decision logic, cancellation strategy)
- [X] T026 [US1] Implement order executor service in order-manager/src/services/order_executor.py (Bybit REST API order creation, modification, cancellation with retry logic and dry-run mode support)
- [X] T027 [US1] Implement signal consumer in order-manager/src/consumers/signal_consumer.py (RabbitMQ consumer for model-service.trading_signals queue, signal processing orchestration)
- [X] T028 [US1] Add signal validation logic (required fields, data types, value ranges) in order-manager/src/services/signal_processor.py
- [X] T029 [US1] Add balance checking logic before order creation in order-manager/src/services/risk_manager.py
- [X] T030 [US1] Add order cancellation logic for existing orders when new signals arrive in order-manager/src/services/signal_processor.py
- [X] T031 [US1] Add logging for all signal processing and order execution operations in order-manager/src/services/signal_processor.py and order-manager/src/services/order_executor.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently - signals can be received, processed, and executed as orders on Bybit

---

## Phase 4: User Story 2 - Maintain Order State Accuracy (Priority: P1)

**Goal**: Keep order status information synchronized with Bybit exchange, updating order states when they change (filled, partially filled, cancelled, rejected) so that the system always has accurate information about order execution.

**Independent Test**: Can be fully tested by subscribing to order execution events from WebSocket gateway and verifying that order states in the database are updated correctly when events are received.

### Implementation for User Story 2

- [ ] T032 [US2] Implement order state synchronization service in order-manager/src/services/order_state_sync.py (startup reconciliation, manual sync, active order query from Bybit API)
- [ ] T033 [US2] Implement event subscriber service in order-manager/src/services/event_subscriber.py (WebSocket event subscription handler, order state update logic from execution events)
- [ ] T034 [US2] Implement order state update logic when execution events received in order-manager/src/services/event_subscriber.py (handle filled, partially_filled, cancelled, rejected events)
- [ ] T035 [US2] Implement startup reconciliation procedure in order-manager/src/services/order_state_sync.py (query active orders from Bybit, compare with database, sync discrepancies)
- [ ] T036 [US2] Add WebSocket gateway subscription management (subscribe to order execution events on startup) in order-manager/src/services/event_subscriber.py
- [ ] T037 [US2] Add position update logic when orders are filled in order-manager/src/services/position_manager.py (update position size, average price, PnL when order execution events received)
- [ ] T038 [US2] Add logging for order state synchronization operations in order-manager/src/services/order_state_sync.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - orders are created and their states are accurately maintained

---

## Phase 5: User Story 3 - Publish Order Events (Priority: P2)

**Goal**: Publish enriched order execution events to RabbitMQ queue for other microservices (especially the model service) so they can track order outcomes, learn from execution results, and adjust trading strategies accordingly.

**Independent Test**: Can be fully tested by verifying that when an order state changes, an enriched event is published to the appropriate queue with all relevant order and execution details.

### Implementation for User Story 3

- [ ] T039 [US3] Implement order event publisher service in order-manager/src/publishers/order_event_publisher.py (RabbitMQ publisher for enriched order events)
- [ ] T040 [US3] Add event enrichment logic (execution price, fees, market conditions, timing) in order-manager/src/publishers/order_event_publisher.py
- [ ] T041 [US3] Integrate event publishing when order states change in order-manager/src/services/event_subscriber.py (publish on filled, partially_filled, cancelled, rejected)
- [ ] T042 [US3] Add event publishing for order rejections with rejection reason in order-manager/src/services/signal_processor.py
- [ ] T043 [US3] Add event publishing for order modifications with before/after state comparison in order-manager/src/services/order_executor.py
- [ ] T044 [US3] Add logging for event publishing operations in order-manager/src/publishers/order_event_publisher.py

**Checkpoint**: At this point, User Stories 1, 2, AND 3 should all work independently - orders are created, states are maintained, and events are published

---

## Phase 6: User Story 4 - Safety and Risk Protection (Priority: P1)

**Goal**: Provide protection against incorrect or risky trading actions, ensuring that signals are validated before execution and that orders cannot exceed available balance or violate configured risk limits.

**Independent Test**: Can be fully tested by sending invalid or risky signals and verifying that they are rejected with appropriate safety checks, without creating orders on the exchange.

### Implementation for User Story 4

- [ ] T045 [US4] Enhance signal validation with comprehensive parameter validation in order-manager/src/services/signal_processor.py (negative amounts, invalid assets, confidence range, required fields)
- [ ] T046 [US4] Enhance balance validation to prevent orders exceeding available balance in order-manager/src/services/risk_manager.py
- [ ] T047 [US4] Implement risk limit checks (max exposure, max order size, position size limits) in order-manager/src/services/risk_manager.py
- [ ] T048 [US4] Add duplicate signal handling (track signal processing state, reject duplicates if previous succeeded, allow retry if previous failed) in order-manager/src/services/signal_processor.py
- [ ] T049 [US4] Add conflict resolution for simultaneous signals in order-manager/src/services/signal_processor.py (queue handling for same asset)
- [ ] T050 [US4] Add error handling and rejection logging with clear error messages in order-manager/src/services/signal_processor.py
- [ ] T051 [US4] Add safety mechanism logging for all rejections in order-manager/src/services/risk_manager.py

**Checkpoint**: At this point, all user stories should work together - orders are created safely with proper validation and risk protection

---

## Phase 7: REST API Endpoints (Supporting All User Stories)

**Purpose**: Provide REST API endpoints for querying orders, positions, and manual synchronization

### Implementation for REST API

- [ ] T052 [P] Implement health check endpoint in order-manager/src/api/routes/health.py (/health, /live, /ready with dependency status)
- [ ] T053 [P] Implement order query endpoints in order-manager/src/api/routes/orders.py (GET /api/v1/orders with filtering, pagination, sorting; GET /api/v1/orders/{order_id})
- [ ] T054 [P] Implement position query endpoints in order-manager/src/api/routes/positions.py (GET /api/v1/positions with filtering; GET /api/v1/positions/{asset})
- [ ] T055 [P] Implement manual synchronization endpoint in order-manager/src/api/routes/sync.py (POST /api/v1/sync with scope parameter)
- [ ] T056 Integrate all routes into FastAPI application in order-manager/src/api/main.py
- [ ] T057 Add API endpoint logging (request/response logging with trace IDs) in order-manager/src/api/middleware/logging.py

**Checkpoint**: REST API endpoints are available for querying and manual operations

---

## Phase 8: Position Management Enhancements

**Purpose**: Implement hybrid position storage strategy with periodic snapshots and validation

### Implementation for Position Management

- [ ] T058 [P] Create Position Snapshot model in order-manager/src/models/position.py (PositionSnapshot entity)
- [ ] T059 Implement position snapshot creation logic in order-manager/src/services/position_manager.py (periodic snapshots based on ORDERMANAGER_POSITION_SNAPSHOT_INTERVAL config)
- [ ] T060 Implement position validation logic (compute position from order history, compare with stored state) in order-manager/src/services/position_manager.py
- [ ] T061 Add background task for periodic position snapshots in order-manager/src/main.py (scheduler for snapshot creation)
- [ ] T062 Add background task for periodic position validation in order-manager/src/main.py (scheduler for validation based on ORDERMANAGER_POSITION_VALIDATION_INTERVAL config)
- [ ] T063 Add discrepancy handling when computed position differs from stored position in order-manager/src/services/position_manager.py

**Checkpoint**: Position management with snapshots and validation is operational

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T064 [P] Update README.md with complete setup and usage instructions in order-manager/README.md
- [ ] T065 [P] Synchronize quickstart.md with actual implementation in specs/004-order-manager/quickstart.md
- [ ] T066 Code cleanup and refactoring (review all services for consistency)
- [ ] T067 [P] Add comprehensive error handling across all services (consistent error responses)
- [ ] T068 [P] Add performance monitoring and metrics (order processing latency, queue depth, API response times)
- [ ] T069 Security hardening (API key rotation, secure credential storage, input sanitization)
- [ ] T070 Run quickstart.md validation (verify all steps work end-to-end)
- [ ] T071 Add database migration files in ws-gateway/migrations/ (XXX_create_orders_table.sql, XXX_create_signal_order_relationships_table.sql, XXX_create_positions_table.sql, XXX_create_position_snapshots_table.sql per constitution requirement)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2)
- **REST API (Phase 7)**: Depends on User Stories 1-4 (needs models and services)
- **Position Management (Phase 8)**: Depends on User Story 1 (position model and manager)
- **Polish (Phase 9)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - Depends on User Story 1 (needs Order model and order creation logic)
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - Depends on User Story 1 (needs order state changes) and User Story 2 (needs order state updates)
- **User Story 4 (P1)**: Can start after Foundational (Phase 2) - Integrates with User Story 1 (enhances signal processing and risk checks)

### Within Each User Story

- Models before services
- Services before consumers/publishers
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, User Stories 1 and 4 can start in parallel (both P1, minimal overlap)
- Models within a story marked [P] can run in parallel
- REST API endpoints marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members (with coordination)

---

## Parallel Example: User Story 1

```bash
# Launch all models for User Story 1 together:
Task: "Create Order model in order-manager/src/models/order.py"
Task: "Create Trading Signal model in order-manager/src/models/trading_signal.py"
Task: "Create Signal-Order Relationship model in order-manager/src/models/signal_order_rel.py"
Task: "Create Position model in order-manager/src/models/position.py"

# After models, services can be developed in parallel where dependencies allow:
Task: "Implement order type selector service in order-manager/src/services/order_type_selector.py"
Task: "Implement quantity calculator service in order-manager/src/services/quantity_calculator.py"
Task: "Implement position manager service in order-manager/src/services/position_manager.py"
Task: "Implement risk manager service in order-manager/src/services/risk_manager.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Execute Trading Signals)
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo (Order state accuracy)
4. Add User Story 4 → Test independently → Deploy/Demo (Safety mechanisms)
5. Add User Story 3 → Test independently → Deploy/Demo (Event publishing)
6. Add REST API → Test independently → Deploy/Demo (Query capabilities)
7. Add Position Management → Test independently → Deploy/Demo (Full feature set)
8. Polish → Final release

Each story adds value without breaking previous stories.

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (core execution)
   - Developer B: User Story 4 (safety - can work in parallel with US1)
   - Developer C: User Story 2 (state sync - after US1 models ready)
3. After US1 and US4 complete:
   - Developer A: User Story 3 (event publishing)
   - Developer B: REST API endpoints
   - Developer C: Position Management enhancements
4. Stories complete and integrate independently

---

## Task Summary

- **Total Tasks**: 71
- **Setup Phase**: 7 tasks
- **Foundational Phase**: 9 tasks
- **User Story 1 (P1)**: 15 tasks
- **User Story 2 (P1)**: 7 tasks
- **User Story 3 (P2)**: 6 tasks
- **User Story 4 (P1)**: 7 tasks
- **REST API Phase**: 6 tasks
- **Position Management Phase**: 6 tasks
- **Polish Phase**: 8 tasks

### Parallel Opportunities Identified

- **Setup**: 4 parallel tasks (T003-T006)
- **Foundational**: 8 parallel tasks (T009-T016)
- **User Story 1**: 4 parallel model tasks (T017-T020)
- **REST API**: 4 parallel endpoint tasks (T052-T055)
- **Polish**: 3 parallel tasks (T064, T065, T067)

### Independent Test Criteria

- **User Story 1**: Send trading signal → verify order created on Bybit and stored in database
- **User Story 2**: Subscribe to order events → verify order states updated correctly in database
- **User Story 3**: Trigger order state change → verify enriched event published to queue
- **User Story 4**: Send invalid/risky signal → verify rejection with safety checks, no order created

### Suggested MVP Scope

**MVP = Phase 1 + Phase 2 + Phase 3 (User Story 1 only)**

This delivers the core functionality: receiving trading signals and executing them as orders on Bybit exchange. All other user stories can be added incrementally.

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Database migrations MUST be created in ws-gateway/migrations/ per constitution requirement
- All configuration via .env with pydantic-settings validation
- All operations must include structured logging with trace IDs
- Dry-run mode must be supported throughout order execution logic
- Per-symbol FIFO queue for signal processing (different assets can process in parallel)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence

