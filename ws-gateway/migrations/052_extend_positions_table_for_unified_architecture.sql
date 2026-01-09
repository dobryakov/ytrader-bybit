-- Migration: 052_extend_positions_table_for_unified_architecture.sql
-- Purpose: Extend positions table with all fields from position-manager-2.0.md specification
--          Add indexes for active and historical positions
--          Add unique index for active positions to prevent duplicates
-- Note: This migration is owned by ws-gateway service (PostgreSQL migration ownership)

BEGIN;

-- Add all new fields from specification
ALTER TABLE positions
    -- Financial indicators from Bybit
    ADD COLUMN IF NOT EXISTS leverage DECIMAL(10, 2),
    ADD COLUMN IF NOT EXISTS position_value DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS liq_price DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS bust_price DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS take_profit DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS stop_loss DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS cum_realised_pnl DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS cum_unrealised_pnl DECIMAL(20, 8),
    
    -- Fees and margins
    ADD COLUMN IF NOT EXISTS opening_fees DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS closing_fees DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS margin_used DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS available_margin DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS maintenance_margin DECIMAL(20, 8),
    
    -- Volume tracking
    ADD COLUMN IF NOT EXISTS max_size DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS min_size DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS total_volume_traded DECIMAL(20, 8) NOT NULL DEFAULT 0,
    
    -- Price tracking
    ADD COLUMN IF NOT EXISTS first_entry_price DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS last_entry_price DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS exit_price DECIMAL(20, 8),
    
    -- PnL metrics
    ADD COLUMN IF NOT EXISTS peak_unrealized_pnl DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS peak_unrealized_pnl_at TIMESTAMP,
    ADD COLUMN IF NOT EXISTS worst_unrealized_pnl DECIMAL(20, 8),
    ADD COLUMN IF NOT EXISTS worst_unrealized_pnl_at TIMESTAMP,
    
    -- Metadata
    ADD COLUMN IF NOT EXISTS source VARCHAR(50),
    ADD COLUMN IF NOT EXISTS last_sync_with_bybit TIMESTAMP,
    ADD COLUMN IF NOT EXISTS bybit_position_data JSONB;

-- Ensure default values for NOT NULL fields
UPDATE positions SET total_volume_traded = 0 WHERE total_volume_traded IS NULL;
UPDATE positions SET unrealized_pnl = 0 WHERE unrealized_pnl IS NULL;
UPDATE positions SET realized_pnl = 0 WHERE realized_pnl IS NULL;
UPDATE positions SET total_fees = 0 WHERE total_fees IS NULL;

-- Ensure created_at is set for existing rows
UPDATE positions SET created_at = last_updated WHERE created_at IS NULL;

-- Add constraint for size and closed_at relationship
ALTER TABLE positions
    DROP CONSTRAINT IF EXISTS chk_size_closed,
    ADD CONSTRAINT chk_size_closed CHECK (
        (closed_at IS NULL) OR (closed_at IS NOT NULL AND size = 0)
    );

-- Remove old unique constraint on (asset, mode) - we'll replace it with conditional unique index
ALTER TABLE positions
    DROP CONSTRAINT IF EXISTS uq_position_asset_mode;

-- Create indexes for active positions
CREATE INDEX IF NOT EXISTS idx_positions_asset_mode ON positions(asset, mode);
CREATE INDEX IF NOT EXISTS idx_positions_closed_at ON positions(closed_at DESC);
CREATE INDEX IF NOT EXISTS idx_positions_active ON positions(asset, mode) WHERE closed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_positions_historical ON positions(asset, mode, closed_at DESC) WHERE closed_at IS NOT NULL;

-- CRITICAL: Unique index for active positions to prevent duplicates
-- This ensures only one active position per (asset, mode) combination
CREATE UNIQUE INDEX IF NOT EXISTS idx_positions_active_unique 
ON positions(asset, mode) 
WHERE closed_at IS NULL;

COMMIT;

-- Rollback section
-- WARNING: Dropping columns will remove data. Use with caution.
-- To rollback this migration, run:
--
-- BEGIN;
-- DROP INDEX IF EXISTS idx_positions_active_unique;
-- DROP INDEX IF EXISTS idx_positions_historical;
-- DROP INDEX IF EXISTS idx_positions_active;
-- DROP INDEX IF EXISTS idx_positions_closed_at;
-- DROP INDEX IF EXISTS idx_positions_asset_mode;
-- ALTER TABLE positions DROP CONSTRAINT IF EXISTS chk_size_closed;
-- ALTER TABLE positions DROP COLUMN IF EXISTS bybit_position_data;
-- ALTER TABLE positions DROP COLUMN IF EXISTS last_sync_with_bybit;
-- ALTER TABLE positions DROP COLUMN IF EXISTS source;
-- ALTER TABLE positions DROP COLUMN IF EXISTS worst_unrealized_pnl_at;
-- ALTER TABLE positions DROP COLUMN IF EXISTS worst_unrealized_pnl;
-- ALTER TABLE positions DROP COLUMN IF EXISTS peak_unrealized_pnl_at;
-- ALTER TABLE positions DROP COLUMN IF EXISTS peak_unrealized_pnl;
-- ALTER TABLE positions DROP COLUMN IF EXISTS exit_price;
-- ALTER TABLE positions DROP COLUMN IF EXISTS last_entry_price;
-- ALTER TABLE positions DROP COLUMN IF EXISTS first_entry_price;
-- ALTER TABLE positions DROP COLUMN IF EXISTS total_volume_traded;
-- ALTER TABLE positions DROP COLUMN IF EXISTS min_size;
-- ALTER TABLE positions DROP COLUMN IF EXISTS max_size;
-- ALTER TABLE positions DROP COLUMN IF EXISTS maintenance_margin;
-- ALTER TABLE positions DROP COLUMN IF EXISTS available_margin;
-- ALTER TABLE positions DROP COLUMN IF EXISTS margin_used;
-- ALTER TABLE positions DROP COLUMN IF EXISTS closing_fees;
-- ALTER TABLE positions DROP COLUMN IF EXISTS opening_fees;
-- ALTER TABLE positions DROP COLUMN IF EXISTS cum_unrealised_pnl;
-- ALTER TABLE positions DROP COLUMN IF EXISTS cum_realised_pnl;
-- ALTER TABLE positions DROP COLUMN IF EXISTS stop_loss;
-- ALTER TABLE positions DROP COLUMN IF EXISTS take_profit;
-- ALTER TABLE positions DROP COLUMN IF EXISTS bust_price;
-- ALTER TABLE positions DROP COLUMN IF EXISTS liq_price;
-- ALTER TABLE positions DROP COLUMN IF EXISTS position_value;
-- ALTER TABLE positions DROP COLUMN IF EXISTS leverage;
-- ALTER TABLE positions ADD CONSTRAINT uq_position_asset_mode UNIQUE (asset, mode);
-- COMMIT;

