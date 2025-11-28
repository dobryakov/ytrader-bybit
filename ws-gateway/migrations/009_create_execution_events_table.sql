-- Migration: Create execution_events table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Store order execution events for time-series queries and quality evaluation

CREATE TABLE IF NOT EXISTS execution_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID NOT NULL,
    strategy_id VARCHAR(100) NOT NULL,
    asset VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    execution_price DECIMAL(20, 8) NOT NULL CHECK (execution_price > 0),
    execution_quantity DECIMAL(20, 8) NOT NULL CHECK (execution_quantity > 0),
    execution_fees DECIMAL(20, 8) NOT NULL CHECK (execution_fees >= 0),
    executed_at TIMESTAMP NOT NULL,
    signal_price DECIMAL(20, 8) NOT NULL,
    signal_timestamp TIMESTAMP NOT NULL,
    performance JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_side CHECK (side IN ('buy', 'sell'))
);

-- Indexes for time-series queries
CREATE INDEX IF NOT EXISTS idx_execution_events_executed_at ON execution_events(executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_execution_events_signal_id ON execution_events(signal_id);
CREATE INDEX IF NOT EXISTS idx_execution_events_strategy_id ON execution_events(strategy_id);
CREATE INDEX IF NOT EXISTS idx_execution_events_strategy_executed_at ON execution_events(strategy_id, executed_at DESC);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_execution_events_strategy_executed_at;
-- DROP INDEX IF EXISTS idx_execution_events_strategy_id;
-- DROP INDEX IF EXISTS idx_execution_events_signal_id;
-- DROP INDEX IF EXISTS idx_execution_events_executed_at;
-- DROP TABLE IF EXISTS execution_events;

