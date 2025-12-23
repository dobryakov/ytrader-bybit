-- Migration: Fix unique index for model_versions to handle empty strings in strategy_id
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Ensure only one active model per (strategy_id, symbol) combination,
--          including cases where strategy_id is empty string or NULL

-- First, deactivate duplicate active models, keeping only the most recent one per (strategy_id, symbol)
-- For models with symbol IS NOT NULL
UPDATE model_versions mv1
SET is_active = false
WHERE mv1.is_active = true
  AND mv1.symbol IS NOT NULL
  AND EXISTS (
    SELECT 1
    FROM model_versions mv2
    WHERE mv2.is_active = true
      AND mv2.symbol IS NOT NULL
      AND COALESCE(mv2.strategy_id, '') = COALESCE(mv1.strategy_id, '')
      AND mv2.symbol = mv1.symbol
      AND mv2.trained_at > mv1.trained_at
  );

-- For models with symbol IS NULL (universal models)
UPDATE model_versions mv1
SET is_active = false
WHERE mv1.is_active = true
  AND mv1.symbol IS NULL
  AND EXISTS (
    SELECT 1
    FROM model_versions mv2
    WHERE mv2.is_active = true
      AND mv2.symbol IS NULL
      AND COALESCE(mv2.strategy_id, '') = COALESCE(mv1.strategy_id, '')
      AND mv2.trained_at > mv1.trained_at
  );

-- Drop existing unique indexes
DROP INDEX IF EXISTS idx_model_versions_unique_active;
DROP INDEX IF EXISTS idx_model_versions_unique_active_no_symbol;

-- Recreate unique indexes using COALESCE to normalize empty strings and NULL
-- For models with symbol IS NOT NULL
CREATE UNIQUE INDEX idx_model_versions_unique_active 
ON model_versions(COALESCE(strategy_id, ''), symbol, is_active) 
WHERE is_active = true AND symbol IS NOT NULL;

-- For models with symbol IS NULL (universal models)
CREATE UNIQUE INDEX idx_model_versions_unique_active_no_symbol 
ON model_versions(COALESCE(strategy_id, ''), is_active) 
WHERE is_active = true AND symbol IS NULL;

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_model_versions_unique_active_no_symbol;
-- DROP INDEX IF EXISTS idx_model_versions_unique_active;
-- CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_unique_active 
--   ON model_versions(strategy_id, symbol, is_active) 
--   WHERE is_active = true AND symbol IS NOT NULL;
-- CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_unique_active_no_symbol 
--   ON model_versions(strategy_id, is_active) 
--   WHERE is_active = true AND symbol IS NULL;

