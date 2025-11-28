-- Migration: Create trading_signals table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Store trading signals for Grafana dashboard queries and monitoring

CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID NOT NULL UNIQUE,
    strategy_id VARCHAR(100) NOT NULL,
    asset VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    price DECIMAL(20, 8) NOT NULL CHECK (price > 0),
    confidence DECIMAL(5, 4) CHECK (confidence >= 0 AND confidence <= 1),
    timestamp TIMESTAMP NOT NULL,
    model_version VARCHAR(50),
    is_warmup BOOLEAN NOT NULL DEFAULT false,
    market_data_snapshot JSONB,
    metadata JSONB,
    trace_id VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_side CHECK (side IN ('buy', 'sell')),
    CONSTRAINT chk_confidence CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);

-- Indexes for Grafana dashboard queries
CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp ON trading_signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_signal_id ON trading_signals(signal_id);
CREATE INDEX IF NOT EXISTS idx_trading_signals_strategy_id ON trading_signals(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trading_signals_asset ON trading_signals(asset);
CREATE INDEX IF NOT EXISTS idx_trading_signals_strategy_timestamp ON trading_signals(strategy_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_asset_timestamp ON trading_signals(asset, timestamp DESC);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_trading_signals_asset_timestamp;
-- DROP INDEX IF EXISTS idx_trading_signals_strategy_timestamp;
-- DROP INDEX IF EXISTS idx_trading_signals_asset;
-- DROP INDEX IF EXISTS idx_trading_signals_strategy_id;
-- DROP INDEX IF EXISTS idx_trading_signals_signal_id;
-- DROP INDEX IF EXISTS idx_trading_signals_timestamp;
-- DROP TABLE IF EXISTS trading_signals;

