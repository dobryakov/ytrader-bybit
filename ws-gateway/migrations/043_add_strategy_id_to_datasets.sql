-- Migration: Add strategy_id to datasets table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Link datasets to trading strategies

-- Add new column
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS strategy_id VARCHAR(100);

-- Create index for fast lookup by strategy_id
CREATE INDEX IF NOT EXISTS idx_datasets_strategy_id ON datasets(strategy_id);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_datasets_strategy_id;
-- ALTER TABLE datasets DROP COLUMN IF EXISTS strategy_id;

