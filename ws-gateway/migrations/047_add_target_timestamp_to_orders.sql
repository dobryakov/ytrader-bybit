-- Migration: Add target_timestamp column to orders table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Store target timestamp for order closure based on prediction horizon

ALTER TABLE orders
ADD COLUMN IF NOT EXISTS target_timestamp TIMESTAMP;

COMMENT ON COLUMN orders.target_timestamp IS 'Target timestamp for order closure based on prediction horizon. When reached, the position should be closed.';

CREATE INDEX IF NOT EXISTS idx_orders_target_timestamp ON orders(target_timestamp) WHERE target_timestamp IS NOT NULL;

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_orders_target_timestamp;
-- ALTER TABLE orders DROP COLUMN IF EXISTS target_timestamp;

