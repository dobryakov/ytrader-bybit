-- Migration: Create prediction_targets table
-- Reversible: Yes
-- Purpose: Store predictions and actual target values with flexible JSONB structure

CREATE TABLE IF NOT EXISTS prediction_targets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID NOT NULL REFERENCES trading_signals(signal_id) ON DELETE CASCADE,
    
    -- Timestamps
    prediction_timestamp TIMESTAMP NOT NULL,
    target_timestamp TIMESTAMP NOT NULL,
    
    -- Registry versions (for reproducibility)
    model_version VARCHAR(50) NOT NULL,
    feature_registry_version VARCHAR(50) NOT NULL,
    target_registry_version VARCHAR(50) NOT NULL,
    
    -- Full target configuration snapshot (JSONB)
    -- Stores complete config from target_registry_versions.config
    -- Allows understanding what config was used even if registry changes
    target_config JSONB NOT NULL,
    
    -- Predicted values (JSONB - flexible structure)
    -- Structure depends on target_config.type and target_config.computation.preset
    predicted_values JSONB NOT NULL,
    
    -- Actual values (JSONB - filled after target_timestamp)
    -- Structure matches predicted_values
    actual_values JSONB,
    
    -- Metadata for actual values computation
    actual_values_computed_at TIMESTAMP,
    actual_values_computation_error TEXT,
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_target_config_structure CHECK (
        target_config ? 'type' AND 
        target_config ? 'horizon' AND
        target_config->>'type' IN ('regression', 'classification', 'risk_adjusted') AND
        (target_config->>'horizon')::integer > 0
    ),
    CONSTRAINT chk_predicted_values_not_empty CHECK (
        jsonb_typeof(predicted_values) = 'object' AND
        predicted_values != '{}'::jsonb
    ),
    CONSTRAINT chk_target_timestamp_after_prediction CHECK (
        target_timestamp >= prediction_timestamp
    )
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_prediction_targets_signal_id ON prediction_targets(signal_id);
CREATE INDEX IF NOT EXISTS idx_prediction_targets_prediction_timestamp ON prediction_targets(prediction_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_targets_target_timestamp ON prediction_targets(target_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_targets_target_registry_version ON prediction_targets(target_registry_version);
CREATE INDEX IF NOT EXISTS idx_prediction_targets_model_version ON prediction_targets(model_version);
CREATE INDEX IF NOT EXISTS idx_prediction_targets_feature_registry_version ON prediction_targets(feature_registry_version);
-- Partial index to speed up lookups of pending computations (without using non-immutable functions)
CREATE INDEX IF NOT EXISTS idx_prediction_targets_pending_computation ON prediction_targets(target_timestamp) 
    WHERE actual_values_computed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_prediction_targets_target_config_type ON prediction_targets((target_config->>'type'));
CREATE INDEX IF NOT EXISTS idx_prediction_targets_target_config_horizon ON prediction_targets(((target_config->>'horizon')::integer));
CREATE INDEX IF NOT EXISTS idx_prediction_targets_target_config_preset ON prediction_targets((target_config->'computation'->>'preset'))
    WHERE target_config->'computation' IS NOT NULL;

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_prediction_targets_target_config_preset;
-- DROP INDEX IF EXISTS idx_prediction_targets_target_config_horizon;
-- DROP INDEX IF EXISTS idx_prediction_targets_target_config_type;
-- DROP INDEX IF EXISTS idx_prediction_targets_pending_computation;
-- DROP INDEX IF EXISTS idx_prediction_targets_feature_registry_version;
-- DROP INDEX IF EXISTS idx_prediction_targets_model_version;
-- DROP INDEX IF EXISTS idx_prediction_targets_target_registry_version;
-- DROP INDEX IF EXISTS idx_prediction_targets_target_timestamp;
-- DROP INDEX IF EXISTS idx_prediction_targets_prediction_timestamp;
-- DROP INDEX IF EXISTS idx_prediction_targets_signal_id;
-- DROP TABLE IF EXISTS prediction_targets;

