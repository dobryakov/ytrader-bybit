-- Migration: Modify feature_registry_versions table for file-based storage
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Change from storing config JSONB in DB to storing file_path, making YAML files the source of truth
-- Note: This migration removes config JSONB column and adds file_path column

-- Step 1: Add file_path column (nullable first to allow migration)
ALTER TABLE feature_registry_versions
ADD COLUMN IF NOT EXISTS file_path VARCHAR(500);

-- Step 2: Backfill file_path from existing config JSONB if migrating from old schema
-- For active version: use legacy file path /app/config/feature_registry.yaml (if exists)
-- For inactive versions: use versioned path /app/config/versions/feature_registry_v{version}.yaml
-- NOTE: This migration only sets file_path metadata in DB. Actual YAML files must be created
-- by application migration script (migrate_legacy_registry.py) BEFORE or AFTER this migration.
-- Files are the source of truth - DB only stores metadata (file_path pointer).

-- Set file_path for active version (if exists) to legacy file location
UPDATE feature_registry_versions
SET file_path = '/app/config/feature_registry.yaml'
WHERE file_path IS NULL AND is_active = true;

-- Set file_path for inactive versions to versioned location
UPDATE feature_registry_versions
SET file_path = '/app/config/versions/feature_registry_v' || version || '.yaml'
WHERE file_path IS NULL AND is_active = false;

-- Step 3: Make file_path NOT NULL (after backfill)
ALTER TABLE feature_registry_versions
ALTER COLUMN file_path SET NOT NULL;

-- Step 4: Add index on file_path for quick lookups
CREATE INDEX IF NOT EXISTS idx_feature_registry_file_path ON feature_registry_versions(file_path);

-- Step 5: Remove config JSONB column (if exists)
-- Note: This will fail if there are rows with config but no file_path (should not happen after backfill)
ALTER TABLE feature_registry_versions
DROP COLUMN IF EXISTS config;

-- Rollback (reverse migration):
-- Step 1: Add config JSONB column back (nullable first)
-- ALTER TABLE feature_registry_versions
-- ADD COLUMN IF NOT EXISTS config JSONB;
--
-- Step 2: Backfill config from file_path if files exist (requires application logic)
-- Note: This would require reading YAML files and converting to JSONB
-- This is a complex operation that should be done via application migration script
--
-- Step 3: Make config NOT NULL (after backfill)
-- ALTER TABLE feature_registry_versions
-- ALTER COLUMN config SET NOT NULL;
--
-- Step 4: Drop file_path column
-- ALTER TABLE feature_registry_versions
-- DROP COLUMN IF EXISTS file_path;
--
-- Step 5: Drop index on file_path
-- DROP INDEX IF EXISTS idx_feature_registry_file_path;

