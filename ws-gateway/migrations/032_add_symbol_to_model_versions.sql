-- Migration: Add symbol column to model_versions table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Allow models to be bound to specific trading symbols (e.g., BTCUSDT, ETHUSDT)
--          in addition to strategy_id. This enables symbol-specific models.

-- Add symbol column (nullable for backward compatibility with existing models)
ALTER TABLE model_versions 
ADD COLUMN IF NOT EXISTS symbol VARCHAR(20);

-- Create index for fast lookup by strategy_id + symbol
CREATE INDEX IF NOT EXISTS idx_model_versions_strategy_symbol 
ON model_versions(strategy_id, symbol) 
WHERE symbol IS NOT NULL;

-- Drop old unique index for active models (only by strategy_id)
DROP INDEX IF EXISTS idx_model_versions_unique_active;

-- Create new unique index for active models by strategy_id + symbol
-- This ensures only one active model per (strategy_id, symbol) combination
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_unique_active 
ON model_versions(strategy_id, symbol, is_active) 
WHERE is_active = true AND symbol IS NOT NULL;

-- For backward compatibility: allow models without symbol (universal models)
-- Only one active model per strategy_id if symbol is NULL
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_unique_active_no_symbol 
ON model_versions(strategy_id, is_active) 
WHERE is_active = true AND symbol IS NULL;

-- Update existing active index to include symbol in WHERE clause
-- (idx_model_versions_active is used for filtering, not uniqueness)
DROP INDEX IF EXISTS idx_model_versions_active;
CREATE INDEX IF NOT EXISTS idx_model_versions_active 
ON model_versions(strategy_id, symbol, is_active) 
WHERE is_active = true;

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_model_versions_active;
-- DROP INDEX IF EXISTS idx_model_versions_unique_active_no_symbol;
-- DROP INDEX IF EXISTS idx_model_versions_unique_active;
-- DROP INDEX IF EXISTS idx_model_versions_strategy_symbol;
-- ALTER TABLE model_versions DROP COLUMN IF EXISTS symbol;
-- CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_unique_active 
--   ON model_versions(strategy_id, is_active) WHERE is_active = true;
-- CREATE INDEX IF NOT EXISTS idx_model_versions_active 
--   ON model_versions(strategy_id, is_active) WHERE is_active = true;

