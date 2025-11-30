# Position Management Service

**Service**: Position Manager  
**Port**: 4800  
**Branch**: `001-position-management`

## Overview

The Position Management Service provides centralized portfolio position management for the trading system. It aggregates all positions, calculates portfolio metrics (total exposure, total PnL, portfolio value), provides REST API access, and manages position lifecycle (creation, update, validation, snapshots).

## Features

- **Real-Time Position Tracking**: Access current position data and portfolio metrics in real-time
- **Automatic Position Updates**: Automatically updates positions from multiple sources (order execution, market data)
- **Portfolio Risk Management Support**: Portfolio-level exposure and profit/loss metrics for risk management components
- **Historical Position Tracking**: Position snapshots for analytics and model training
- **Position Data Validation**: Periodic validation and synchronization with authoritative sources

## Quick Start

### Prerequisites

- Docker and Docker Compose V2 installed
- Access to shared PostgreSQL database
- Access to shared RabbitMQ
- API key for authentication

### Setup

1. **Environment Configuration**: Copy environment variables from root `env.example` to `.env` (in project root)

2. **Database Migration**: Run migration to add `current_price` and `version` fields to `positions` table (see `quickstart.md`)

3. **Build and Start**:
   ```bash
   docker compose build position-manager
   docker compose up -d position-manager
   ```

4. **Verify Health**:
   ```bash
   curl http://localhost:4800/health
   ```

## API Documentation

See `specs/001-position-management/contracts/openapi.yaml` for complete API specification.

### Key Endpoints

- `GET /api/v1/positions` - Get all positions
- `GET /api/v1/positions/{asset}` - Get position by asset
- `GET /api/v1/portfolio` - Get portfolio metrics
- `GET /api/v1/portfolio/exposure` - Get portfolio exposure
- `GET /api/v1/portfolio/pnl` - Get portfolio PnL
- `POST /api/v1/positions/{asset}/validate` - Validate position
- `POST /api/v1/positions/{asset}/snapshot` - Create position snapshot
- `GET /api/v1/positions/{asset}/snapshots` - Get position snapshots

## Architecture

- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Database**: PostgreSQL (shared)
- **Message Queue**: RabbitMQ (shared)
- **Logging**: structlog with trace IDs

## Project Structure

```
position-manager/
├── Dockerfile
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
│   ├── services/
│   ├── api/
│   ├── consumers/
│   ├── publishers/
│   ├── tasks/
│   └── utils/
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

## Testing

Run tests inside Docker containers:

```bash
# Unit tests
docker compose exec position-manager pytest tests/unit/ -v

# Integration tests
docker compose exec test-container pytest tests/integration/ -v

# E2E tests
docker compose exec test-container pytest tests/e2e/ -v
```

## Documentation

- **Quickstart**: `specs/001-position-management/quickstart.md`
- **Data Model**: `specs/001-position-management/data-model.md`
- **API Contracts**: `specs/001-position-management/contracts/openapi.yaml`
- **Implementation Plan**: `specs/001-position-management/plan.md`
- **Feature Specification**: `specs/001-position-management/spec.md`

## Database Indexes

The following indexes are required for optimal performance (migration handled by ws-gateway service):

- `idx_positions_asset` on `asset`
- `idx_positions_mode` on `mode`
- `idx_positions_asset_mode` on `(asset, mode)`
- `idx_positions_current_price` on `current_price`
- `idx_positions_version` on `version`

## Performance Goals

- Position queries: <500ms for 95% of requests
- Portfolio metrics: <1s for portfolios with up to 100 positions
- Position updates: reflected within 2s (order execution) and 5s (market data)

## Support

For issues or questions:
1. Check service logs: `docker compose logs position-manager`
2. Check health endpoint: `curl http://localhost:4800/health`
3. Review feature specification: `specs/001-position-management/spec.md`

