-- Migration: Add rejection fields to trading_signals table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Store rejected signals (low confidence) as full signals with rejection metadata

ALTER TABLE trading_signals 
ADD COLUMN IF NOT EXISTS is_rejected BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN IF NOT EXISTS rejection_reason TEXT,
ADD COLUMN IF NOT EXISTS effective_threshold DECIMAL(5, 4) CHECK (effective_threshold >= 0 AND effective_threshold <= 1);

-- Index for filtering rejected signals
CREATE INDEX IF NOT EXISTS idx_trading_signals_is_rejected ON trading_signals(is_rejected, timestamp DESC);

-- Index for querying by rejection reason
CREATE INDEX IF NOT EXISTS idx_trading_signals_rejection_reason ON trading_signals(rejection_reason) 
    WHERE rejection_reason IS NOT NULL;

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_trading_signals_rejection_reason;
-- DROP INDEX IF EXISTS idx_trading_signals_is_rejected;
-- ALTER TABLE trading_signals 
-- DROP COLUMN IF EXISTS effective_threshold,
-- DROP COLUMN IF EXISTS rejection_reason,
-- DROP COLUMN IF EXISTS is_rejected;

