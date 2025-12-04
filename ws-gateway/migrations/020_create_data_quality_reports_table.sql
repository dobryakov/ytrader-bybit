-- Migration: Create data_quality_reports table for Feature Service
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Stores data quality reports tracking missing data, anomalies, sequence gaps, and desynchronization events

CREATE TABLE IF NOT EXISTS data_quality_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    missing_rate DECIMAL(5, 4) NOT NULL,
    anomaly_rate DECIMAL(5, 4) NOT NULL,
    sequence_gaps INTEGER NOT NULL DEFAULT 0,
    desynchronization_events INTEGER NOT NULL DEFAULT 0,
    anomaly_details JSONB,
    sequence_gap_details JSONB,
    recommendations TEXT[],
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_period CHECK (period_start < period_end),
    CONSTRAINT chk_rates CHECK (
        missing_rate >= 0 AND missing_rate <= 1 AND
        anomaly_rate >= 0 AND anomaly_rate <= 1
    ),
    CONSTRAINT chk_counts CHECK (
        sequence_gaps >= 0 AND desynchronization_events >= 0
    )
);

CREATE INDEX IF NOT EXISTS idx_data_quality_symbol ON data_quality_reports(symbol);
CREATE INDEX IF NOT EXISTS idx_data_quality_period ON data_quality_reports(period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_data_quality_created_at ON data_quality_reports(created_at DESC);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_data_quality_created_at;
-- DROP INDEX IF EXISTS idx_data_quality_period;
-- DROP INDEX IF EXISTS idx_data_quality_symbol;
-- DROP TABLE IF EXISTS data_quality_reports;

