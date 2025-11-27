-- Migration: Create positions table for order-manager service
-- Reversible: Yes (see rollback section at bottom)

CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asset VARCHAR(20) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    average_entry_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8),
    mode VARCHAR(20) NOT NULL DEFAULT 'one-way',
    long_size DECIMAL(20, 8),
    short_size DECIMAL(20, 8),
    long_avg_price DECIMAL(20, 8),
    short_avg_price DECIMAL(20, 8),
    last_updated TIMESTAMP NOT NULL DEFAULT NOW(),
    last_snapshot_at TIMESTAMP,
    
    CONSTRAINT chk_mode CHECK (mode IN ('one-way', 'hedge')),
    CONSTRAINT chk_avg_entry_price CHECK (average_entry_price IS NULL OR average_entry_price > 0),
    CONSTRAINT chk_long_avg_price CHECK (long_avg_price IS NULL OR long_avg_price > 0),
    CONSTRAINT chk_short_avg_price CHECK (short_avg_price IS NULL OR short_avg_price > 0),
    CONSTRAINT uq_position_asset_mode UNIQUE (asset, mode)
);

CREATE INDEX IF NOT EXISTS idx_positions_asset ON positions(asset);
CREATE INDEX IF NOT EXISTS idx_positions_mode ON positions(mode);
CREATE INDEX IF NOT EXISTS idx_positions_last_updated ON positions(last_updated);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_positions_last_updated;
-- DROP INDEX IF EXISTS idx_positions_mode;
-- DROP INDEX IF EXISTS idx_positions_asset;
-- DROP TABLE IF EXISTS positions;

