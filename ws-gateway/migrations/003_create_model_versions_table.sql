-- Migration: Create model_versions table
-- Reversible: Yes (see rollback section at bottom)

CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version VARCHAR(50) NOT NULL UNIQUE,
    file_path VARCHAR(500) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    strategy_id VARCHAR(100),
    trained_at TIMESTAMP NOT NULL DEFAULT NOW(),
    training_duration_seconds INTEGER,
    training_dataset_size INTEGER,
    training_config JSONB,
    is_active BOOLEAN NOT NULL DEFAULT false,
    is_warmup_mode BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_model_type CHECK (model_type IN ('xgboost', 'random_forest', 'logistic_regression', 'sgd_classifier')),
    CONSTRAINT chk_version_format CHECK (version ~ '^v\d+(\.\d+)?$'),
    CONSTRAINT chk_file_path CHECK (file_path LIKE '/models/%')
);

CREATE INDEX IF NOT EXISTS idx_model_versions_version ON model_versions(version);
CREATE INDEX IF NOT EXISTS idx_model_versions_strategy_id ON model_versions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(strategy_id, is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_model_versions_trained_at ON model_versions(trained_at DESC);

-- Unique constraint: only one active model per strategy
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_versions_unique_active ON model_versions(strategy_id, is_active) WHERE is_active = true;

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_model_versions_unique_active;
-- DROP INDEX IF EXISTS idx_model_versions_trained_at;
-- DROP INDEX IF EXISTS idx_model_versions_active;
-- DROP INDEX IF EXISTS idx_model_versions_strategy_id;
-- DROP INDEX IF EXISTS idx_model_versions_version;
-- DROP TABLE IF EXISTS model_versions;

