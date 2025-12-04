-- Migration: Create feature_registry_versions table for Feature Service
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Stores Feature Registry configuration versions with validation, activation, and rollback support

CREATE TABLE IF NOT EXISTS feature_registry_versions (
    version VARCHAR(50) PRIMARY KEY,
    config JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT false,
    validated_at TIMESTAMP,
    validation_errors TEXT[],
    loaded_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100),
    activated_by VARCHAR(100),
    rollback_from VARCHAR(50),
    previous_version VARCHAR(50),
    schema_version VARCHAR(50),
    migration_script TEXT,
    compatibility_warnings TEXT[],
    breaking_changes TEXT[],
    activation_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_feature_registry_active ON feature_registry_versions(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_feature_registry_created_at ON feature_registry_versions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feature_registry_previous_version ON feature_registry_versions(previous_version);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_feature_registry_previous_version;
-- DROP INDEX IF EXISTS idx_feature_registry_created_at;
-- DROP INDEX IF EXISTS idx_feature_registry_active;
-- DROP TABLE IF EXISTS feature_registry_versions;

