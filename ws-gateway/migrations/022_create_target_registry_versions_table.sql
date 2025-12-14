-- Migration: Create target_registry_versions table for Target Registry
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Stores target configuration versions for dataset building (similar to Feature Registry)

CREATE TABLE IF NOT EXISTS target_registry_versions (
    version VARCHAR(50) PRIMARY KEY,
    config JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100),
    activated_at TIMESTAMP,
    activated_by VARCHAR(100),
    activation_reason TEXT,
    description TEXT,
    file_path VARCHAR(500),
    validated_at TIMESTAMP,
    validation_errors TEXT[],
    loaded_at TIMESTAMP,
    previous_version VARCHAR(50),
    
    CONSTRAINT chk_target_config_type CHECK (
        config ? 'type' AND 
        config->>'type' IN ('regression', 'classification', 'risk_adjusted')
    ),
    CONSTRAINT chk_target_config_horizon CHECK (
        config ? 'horizon' AND 
        (config->>'horizon')::integer > 0
    )
);

CREATE INDEX IF NOT EXISTS idx_target_registry_active ON target_registry_versions(is_active);
CREATE INDEX IF NOT EXISTS idx_target_registry_created_at ON target_registry_versions(created_at DESC);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_target_registry_created_at;
-- DROP INDEX IF EXISTS idx_target_registry_active;
-- DROP TABLE IF EXISTS target_registry_versions;

