-- Migration: Add prediction horizon fields to trading_signals
-- Reversible: Yes
-- Purpose: Store horizon for easier queries (denormalization for performance)

ALTER TABLE trading_signals 
    ADD COLUMN IF NOT EXISTS prediction_horizon_seconds INTEGER,
    ADD COLUMN IF NOT EXISTS target_timestamp TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_trading_signals_target_timestamp ON trading_signals(target_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_horizon ON trading_signals(prediction_horizon_seconds);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_trading_signals_horizon;
-- DROP INDEX IF EXISTS idx_trading_signals_target_timestamp;
-- ALTER TABLE trading_signals DROP COLUMN IF EXISTS target_timestamp;
-- ALTER TABLE trading_signals DROP COLUMN IF EXISTS prediction_horizon_seconds;

