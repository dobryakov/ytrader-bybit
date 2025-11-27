-- Migration: Create position_snapshots table for order-manager service
-- Reversible: Yes (see rollback section at bottom)

CREATE TABLE IF NOT EXISTS position_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    position_id UUID NOT NULL,
    asset VARCHAR(20) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    average_entry_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8),
    mode VARCHAR(20) NOT NULL,
    long_size DECIMAL(20, 8),
    short_size DECIMAL(20, 8),
    snapshot_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_mode CHECK (mode IN ('one-way', 'hedge')),
    CONSTRAINT chk_avg_entry_price CHECK (average_entry_price IS NULL OR average_entry_price > 0),
    CONSTRAINT fk_position FOREIGN KEY (position_id) REFERENCES positions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_position_snapshots_position_id ON position_snapshots(position_id);
CREATE INDEX IF NOT EXISTS idx_position_snapshots_asset ON position_snapshots(asset);
CREATE INDEX IF NOT EXISTS idx_position_snapshots_timestamp ON position_snapshots(snapshot_timestamp);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_position_snapshots_timestamp;
-- DROP INDEX IF EXISTS idx_position_snapshots_asset;
-- DROP INDEX IF EXISTS idx_position_snapshots_position_id;
-- DROP TABLE IF EXISTS position_snapshots;

