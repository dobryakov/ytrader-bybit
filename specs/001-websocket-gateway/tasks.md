---
description: "Task list for WebSocket Gateway feature implementation"
---

# Tasks: WebSocket Gateway for Bybit Data Aggregation and Routing

**Input**: Design documents from `/specs/001-websocket-gateway/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Per constitution principle IV (Testing Discipline), automated tests MUST be executed after completing each task. Unit tests run in service containers; API and e2e tests run in separate test containers connected to main docker-compose.yml. Test tasks are integrated into each phase where applicable.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `ws-gateway/src/`, `ws-gateway/tests/` at repository root
- Paths shown below follow the structure defined in plan.md

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project structure per implementation plan in ws-gateway/
- [X] T002 Initialize Python project with dependencies in ws-gateway/requirements.txt
- [X] T003 [P] Create Dockerfile for ws-gateway service in ws-gateway/Dockerfile
- [X] T004 [P] Create docker-compose.yml entry for ws-gateway service
- [X] T005 [P] Configure linting and formatting tools (black, mypy) in ws-gateway/
- [X] T006 [P] Create README.md in ws-gateway/README.md
- [X] T007 [P] Add environment variables to env.example for ws-gateway configuration

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

**Note on PostgreSQL Migrations**: Per constitution principle II (Shared Database Strategy), the `ws-gateway` service is the single source of truth for all PostgreSQL migrations. All PostgreSQL schema changes (including those for other services in the project) MUST be located in `ws-gateway/migrations/`. Other database types (e.g., vector databases for ML models) may maintain their own migrations within their respective service containers.

- [X] T008 Setup database schema and migrations framework in ws-gateway/migrations/
- [X] T009 [P] Create database migration for subscriptions table in ws-gateway/migrations/001_create_subscriptions_table.sql
- [X] T010 [P] Create database migration for account_balances table in ws-gateway/migrations/002_create_account_balances_table.sql
- [X] T011 [P] Implement configuration management using pydantic-settings in ws-gateway/src/config/settings.py
- [X] T012 [P] Setup structured logging infrastructure with structlog in ws-gateway/src/config/logging.py
- [X] T013 [P] Create base database connection pool using asyncpg in ws-gateway/src/services/database/connection.py
- [X] T014 [P] Create base RabbitMQ connection using aio-pika in ws-gateway/src/services/queue/connection.py
- [X] T015 [P] Setup error handling and exception classes in ws-gateway/src/exceptions.py
- [X] T016 [P] Create base FastAPI application structure in ws-gateway/src/main.py
- [X] T017 [P] Implement health check endpoint GET /health in ws-gateway/src/api/health.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Establish Reliable WebSocket Connection to Bybit Exchange (Priority: P1) 🎯 MVP

**Goal**: The system establishes and maintains a single, authenticated WebSocket connection to the Bybit exchange (mainnet or testnet). The connection automatically reconnects when interrupted and maintains heartbeat signals to ensure continuous operation.

**Independent Test**: Can be fully tested by establishing a connection to Bybit testnet, verifying authentication succeeds, and confirming the connection remains active for a sustained period. The system should automatically recover from network interruptions within acceptable timeframes.

### Implementation for User Story 1

- [X] T018 [P] [US1] Create WebSocket connection state model in ws-gateway/src/models/websocket_state.py
- [X] T019 [US1] Implement Bybit WebSocket authentication logic in ws-gateway/src/services/websocket/auth.py
- [X] T020 [US1] Implement WebSocket connection manager with websockets library in ws-gateway/src/services/websocket/connection.py
- [X] T021 [US1] Implement automatic reconnection logic with 30-second timeout in ws-gateway/src/services/websocket/reconnection.py
- [X] T022 [US1] Implement heartbeat mechanism for connection maintenance in ws-gateway/src/services/websocket/heartbeat.py
- [X] T023 [US1] Integrate WebSocket connection into main application lifecycle in ws-gateway/src/main.py
- [X] T024 [US1] Add logging for WebSocket connection events in ws-gateway/src/services/websocket/connection.py
- [X] T025 [US1] Update health check endpoint to include WebSocket connection status in ws-gateway/src/api/health.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently. The service should successfully connect to Bybit testnet, authenticate, and maintain the connection with automatic reconnection.

---

## Phase 4: User Story 2 - Subscribe to Exchange Data Channels and Receive Events (Priority: P1)

**Goal**: The system subscribes to multiple data channels from Bybit (trades, tickers, order books, order statuses, balances, and others) and receives structured event data. Subscription information is stored in PostgreSQL to enable automatic resubscription after reconnection.

**Independent Test**: Can be fully tested by subscribing to at least one channel type (e.g., trades), verifying events are received with proper structure (unique identifiers, event types, timestamps, payloads), and confirming subscription state is preserved for reconnection scenarios.

### Implementation for User Story 2

- [X] T026 [P] [US2] Create Subscription model in ws-gateway/src/models/subscription.py
- [X] T027 [P] [US2] Create Event model (in-memory structure) in ws-gateway/src/models/event.py
- [X] T028 [US2] Implement subscription database operations in ws-gateway/src/services/database/subscription_repository.py
- [X] T029 [US2] Implement subscription management service in ws-gateway/src/services/subscription/subscription_service.py
- [X] T030 [US2] Implement Bybit subscription message formatting in ws-gateway/src/services/websocket/subscription.py
- [X] T031 [US2] Implement event parsing and validation from Bybit messages in ws-gateway/src/services/websocket/event_parser.py
- [X] T032 [US2] Integrate subscription management with WebSocket connection in ws-gateway/src/services/websocket/connection.py
- [X] T033 [US2] Implement automatic resubscription after reconnection in ws-gateway/src/services/websocket/reconnection.py
- [X] T034 [US2] Add logging for subscription events and received messages in ws-gateway/src/services/websocket/event_parser.py
- [X] T035 [US2] Update health check to include active subscriptions count in ws-gateway/src/api/health.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently. The service should connect to Bybit, subscribe to channels, receive events, and automatically resubscribe after reconnection.

---

## Phase 5: User Story 3 - Manage Dynamic Subscriptions via REST API (Priority: P2)

**Goal**: Other microservices can request new subscriptions or cancel existing ones through a REST API authenticated with API keys. The system processes these requests and updates its active subscriptions accordingly.

**Independent Test**: Can be fully tested by having another service make REST API calls to add a subscription, verify the subscription becomes active, then cancel it and confirm events stop being received for that channel.

### Implementation for User Story 3

- [ ] T036 [P] [US3] Implement API key authentication middleware in ws-gateway/src/api/middleware/auth.py
- [ ] T037 [US3] Implement POST /api/v1/subscriptions endpoint in ws-gateway/src/api/v1/subscriptions.py
- [ ] T038 [US3] Implement GET /api/v1/subscriptions endpoint with filtering in ws-gateway/src/api/v1/subscriptions.py
- [ ] T039 [US3] Implement GET /api/v1/subscriptions/{subscription_id} endpoint in ws-gateway/src/api/v1/subscriptions.py
- [ ] T040 [US3] Implement DELETE /api/v1/subscriptions/{subscription_id} endpoint in ws-gateway/src/api/v1/subscriptions.py
- [ ] T041 [US3] Implement DELETE /api/v1/subscriptions/by-service/{service_name} endpoint in ws-gateway/src/api/v1/subscriptions.py
- [ ] T042 [US3] Implement subscription request validation in ws-gateway/src/api/v1/schemas.py
- [ ] T043 [US3] Integrate REST API endpoints with subscription service in ws-gateway/src/api/v1/subscriptions.py
- [ ] T044 [US3] Add error handling and appropriate HTTP status codes in ws-gateway/src/api/v1/subscriptions.py
- [ ] T045 [US3] Add logging for REST API requests and responses in ws-gateway/src/api/middleware/logging.py

**Checkpoint**: At this point, User Stories 1, 2, AND 3 should all work independently. The service should allow other microservices to manage subscriptions via REST API.

---

## Phase 6: User Story 4 - Deliver Events to Subscribers via Queues (Priority: P2)

**Goal**: The system places received events into appropriate queues, organized by event class, and ensures subscribers (model service, order manager service, and others) receive fresh, structured events from these queues.

**Independent Test**: Can be fully tested by subscribing to a channel, verifying events appear in the appropriate queue, and having a subscriber service consume events from the queue to confirm they are properly structured and delivered.

### Implementation for User Story 4

- [ ] T046 [US4] Implement queue publisher service using aio-pika in ws-gateway/src/services/queue/publisher.py
- [ ] T047 [US4] Implement queue initialization and configuration (durability, retention) in ws-gateway/src/services/queue/setup.py
- [ ] T048 [US4] Implement event routing logic to determine target queue by event class in ws-gateway/src/services/queue/router.py
- [ ] T049 [US4] Integrate queue publishing with event processing pipeline in ws-gateway/src/services/websocket/event_processor.py
- [ ] T050 [US4] Implement queue naming convention (ws-gateway.{event_class}) in ws-gateway/src/services/queue/setup.py
- [ ] T051 [US4] Configure queue retention limits (24 hours or 100K messages) in ws-gateway/src/services/queue/setup.py
- [ ] T051a [US4] Implement queue retention monitoring and cleanup logic in ws-gateway/src/services/queue/retention.py (monitor queue age/size, discard messages exceeding limits per FR-019)
- [ ] T052 [US4] Add logging for queue publishing operations in ws-gateway/src/services/queue/publisher.py
- [ ] T053 [US4] Handle queue connection failures gracefully (log and continue) in ws-gateway/src/services/queue/publisher.py

**Checkpoint**: At this point, User Stories 1, 2, 3, AND 4 should all work independently. Events should be delivered to RabbitMQ queues organized by event class.

---

## Phase 7: User Story 5 - Store Critical Data Directly to Database (Priority: P3)

**Goal**: Certain types of incoming data (such as account balances and account balance information) are immediately persisted to PostgreSQL for reliable record-keeping, independent of queue delivery.

**Independent Test**: Can be fully tested by receiving balance or account data events, verifying they are written to the database, and confirming the data is accurate and timestamped.

### Implementation for User Story 5

- [ ] T054 [P] [US5] Create AccountBalance model in ws-gateway/src/models/account_balance.py
- [ ] T055 [US5] Implement account balance database operations in ws-gateway/src/services/database/balance_repository.py
- [ ] T056 [US5] Implement balance persistence service in ws-gateway/src/services/database/balance_service.py
- [ ] T057 [US5] Integrate balance persistence with event processing pipeline in ws-gateway/src/services/websocket/event_processor.py
- [ ] T058 [US5] Implement balance validation (non-negative, sum consistency) in ws-gateway/src/services/database/balance_service.py
- [ ] T059 [US5] Handle database write failures gracefully (log and continue, per FR-017) in ws-gateway/src/services/database/balance_service.py
- [ ] T060 [US5] Add logging for balance persistence operations in ws-gateway/src/services/database/balance_service.py

**Checkpoint**: At this point, User Stories 1, 2, 3, 4, AND 5 should all work independently. Balance events should be persisted to PostgreSQL.

---

## Phase 8: User Story 6 - Log Activities for Monitoring and Debugging (Priority: P3)

**Goal**: The system logs all significant activities including WebSocket connection events, incoming messages, REST API requests, and system state changes to enable monitoring and troubleshooting.

**Independent Test**: Can be fully tested by performing various operations (connect, subscribe, receive events, handle API requests) and verifying appropriate log entries are created with sufficient detail for troubleshooting.

### Implementation for User Story 6

- [ ] T061 [US6] Enhance WebSocket connection logging with trace IDs in ws-gateway/src/services/websocket/connection.py
- [ ] T062 [US6] Enhance event receipt logging with full message details in ws-gateway/src/services/websocket/event_parser.py
- [ ] T063 [US6] Enhance REST API request/response logging with trace IDs in ws-gateway/src/api/middleware/logging.py
- [ ] T064 [US6] Add error logging with sufficient context throughout the application
- [ ] T065 [US6] Implement trace ID generation and propagation in ws-gateway/src/utils/tracing.py
- [ ] T066 [US6] Add structured logging for system state changes in ws-gateway/src/services/websocket/connection.py

**Checkpoint**: All user stories should now be independently functional with comprehensive logging.

---

## Phase 8.5: Edge Case Handling

**Purpose**: Explicit handling of edge cases identified in spec.md

**Edge Case Coverage Mapping**:

- [ ] EC1 [Edge Cases] Handle extended exchange API unavailability: Implement circuit breaker pattern and exponential backoff in ws-gateway/src/services/websocket/reconnection.py (covers spec.md edge case: "exchange API temporarily unavailable")
- [ ] EC2 [Edge Cases] Handle malformed messages: Add message validation and error handling in ws-gateway/src/services/websocket/event_parser.py (covers spec.md edge case: "malformed message formats")
- [ ] EC3 [Edge Cases] Handle queue capacity limits: Implement queue monitoring and alerting in ws-gateway/src/services/queue/retention.py (covers spec.md edge case: "queue storage reaches capacity")
- [ ] EC4 [Edge Cases] Handle authentication failures: Add credential validation and error recovery in ws-gateway/src/services/websocket/auth.py (covers spec.md edge case: "authentication failures or expired credentials")
- [ ] EC5 [Edge Cases] Handle conflicting subscription configurations: Implement conflict resolution logic in ws-gateway/src/services/subscription/subscription_service.py (covers spec.md edge case: "multiple services request conflicting subscriptions")
- [ ] EC6 [Edge Cases] Handle slow/unavailable PostgreSQL: Already covered by T059 (database write failures), verify graceful degradation
- [ ] EC7 [Edge Cases] Handle slow subscriber consumption: Add queue backlog monitoring and alerting in ws-gateway/src/services/queue/monitoring.py (covers spec.md edge case: "subscriber consumes slower than events arrive")
- [ ] EC8 [Edge Cases] Handle exchange endpoint timeouts: Add timeout handling in ws-gateway/src/services/websocket/connection.py (covers spec.md edge case: "timeouts or unresponsive endpoints")

**Note**: Some edge cases are implicitly covered by existing tasks (e.g., EC6 by T059). This phase makes coverage explicit and adds monitoring where needed.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T067 [P] Update README.md with setup and usage instructions in ws-gateway/README.md
- [ ] T068 [P] Synchronize quickstart.md with implemented features in specs/001-websocket-gateway/quickstart.md
- [ ] T069 [P] Code cleanup and refactoring across all modules
- [ ] T070 [P] Performance optimization (connection pooling, async operations)
- [ ] T071 [P] Add comprehensive error handling and user-friendly error messages
- [ ] T072 [P] Security hardening (API key validation, input sanitization)
- [ ] T073 [P] Run quickstart.md validation and update if needed
- [ ] T074 [P] Implement monitoring and metrics collection for success criteria validation:
  - T074a: Add WebSocket connection uptime tracking (SC-001) in ws-gateway/src/services/websocket/monitoring.py
  - T074b: Add event processing success rate tracking (SC-002) in ws-gateway/src/services/websocket/event_processor.py
  - T074c: Add event delivery latency tracking (SC-003) in ws-gateway/src/services/queue/publisher.py
  - T074d: Add balance persistence latency tracking (SC-005) in ws-gateway/src/services/database/balance_service.py
  - T074e: Add REST API response time tracking (SC-006) in ws-gateway/src/api/middleware/metrics.py
  - T074f: Add resubscription timing tracking (SC-007) in ws-gateway/src/services/websocket/reconnection.py
- [ ] T075 [P] Documentation updates in ws-gateway/README.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Depends on User Story 1 (needs WebSocket connection to subscribe)
- **User Story 3 (P2)**: Depends on User Story 2 (needs subscription system to manage)
- **User Story 4 (P2)**: Depends on User Story 2 (needs events to deliver)
- **User Story 5 (P3)**: Depends on User Story 2 (needs events to persist)
- **User Story 6 (P3)**: Can start after Foundational (Phase 2) - Enhances all previous stories

### Within Each User Story

- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes:
  - User Story 1 can start
  - After User Story 1 completes, User Stories 2, 3, 4 can potentially run in parallel (with coordination)
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members (with dependency awareness)

---

## Parallel Example: User Story 2

```bash
# Launch all models for User Story 2 together:
Task: "Create Subscription model in ws-gateway/src/models/subscription.py"
Task: "Create Event model (in-memory structure) in ws-gateway/src/models/event.py"
```

---

## Parallel Example: Foundational Phase

```bash
# Launch all foundational setup tasks together:
Task: "Create database migration for subscriptions table in ws-gateway/migrations/001_create_subscriptions_table.sql"
Task: "Create database migration for account_balances table in ws-gateway/migrations/002_create_account_balances_table.sql"
Task: "Implement configuration management using pydantic-settings in ws-gateway/src/config/settings.py"
Task: "Setup structured logging infrastructure with structlog in ws-gateway/src/config/logging.py"
Task: "Create base database connection pool using asyncpg in ws-gateway/src/services/database/connection.py"
Task: "Create base RabbitMQ connection using aio-pika in ws-gateway/src/services/queue/connection.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1 & 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (WebSocket Connection)
4. Complete Phase 4: User Story 2 (Subscriptions & Events)
5. **STOP and VALIDATE**: Test User Stories 1 & 2 independently
6. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (Basic Connection)
3. Add User Story 2 → Test independently → Deploy/Demo (Event Reception)
4. Add User Story 3 → Test independently → Deploy/Demo (REST API)
5. Add User Story 4 → Test independently → Deploy/Demo (Queue Delivery)
6. Add User Story 5 → Test independently → Deploy/Demo (Balance Persistence)
7. Add User Story 6 → Test independently → Deploy/Demo (Logging)
8. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (WebSocket Connection)
3. Once User Story 1 is done:
   - Developer A: User Story 2 (Subscriptions)
   - Developer B: User Story 3 (REST API) - can start after US2 foundation
   - Developer C: User Story 4 (Queue Delivery) - can start after US2 foundation
4. Once User Stories 2-4 are done:
   - Developer A: User Story 5 (Balance Persistence)
   - Developer B: User Story 6 (Logging)
5. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- User Story 2 depends on User Story 1 (needs connection to subscribe)
- User Stories 3 and 4 can start after User Story 2 foundation is laid
- User Story 5 depends on User Story 2 (needs events to persist)
- User Story 6 enhances all previous stories with logging

---

## Task Summary

- **Total Tasks**: 88 (75 original + 13 new: T051a, EC1-EC8, T074a-T074f)
- **Setup Phase**: 7 tasks
- **Foundational Phase**: 10 tasks
- **User Story 1 (P1)**: 8 tasks
- **User Story 2 (P1)**: 10 tasks
- **User Story 3 (P2)**: 10 tasks
- **User Story 4 (P2)**: 9 tasks (added T051a for queue retention enforcement)
- **User Story 5 (P3)**: 7 tasks
- **User Story 6 (P3)**: 6 tasks
- **Edge Case Handling (Phase 8.5)**: 8 tasks (EC1-EC8, EC6 is verification note)
- **Polish Phase**: 13 tasks (T074 expanded to T074a-T074f for monitoring)

**Suggested MVP Scope**: User Stories 1 & 2 (WebSocket Connection + Subscriptions & Events) - 18 implementation tasks plus setup and foundational phases.

