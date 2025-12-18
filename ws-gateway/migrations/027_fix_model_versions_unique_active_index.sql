-- Migration: Fix unique index for model_versions to handle NULL strategy_id correctly
-- Date: 2025-12-16
-- Reversible: Yes (see rollback section at bottom)
--
-- Problem: The original unique index idx_model_versions_unique_active didn't work correctly
-- for NULL strategy_id values because PostgreSQL treats NULL != NULL in unique constraints.
-- This allowed multiple active models for the same strategy (when strategy_id is NULL).
--
-- Solution: Recreate the index using COALESCE to treat NULL as empty string,
-- ensuring only one active model per strategy (including NULL/default strategy).

-- Drop the old index
DROP INDEX IF EXISTS idx_model_versions_unique_active;

-- Create new index with COALESCE to handle NULL strategy_id
CREATE UNIQUE INDEX idx_model_versions_unique_active 
ON model_versions(COALESCE(strategy_id, ''), is_active) 
WHERE is_active = true;

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_model_versions_unique_active;
-- CREATE UNIQUE INDEX idx_model_versions_unique_active 
-- ON model_versions(strategy_id, is_active) 
-- WHERE is_active = true;

