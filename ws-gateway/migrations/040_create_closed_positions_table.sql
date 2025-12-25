-- Migration: 040_create_closed_positions_table.sql
-- Purpose: Create table to store history of closed positions
-- Note: This migration is owned by ws-gateway service (PostgreSQL migration ownership)

BEGIN;

CREATE TABLE IF NOT EXISTS closed_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_position_id UUID NOT NULL,
    asset VARCHAR(20) NOT NULL,
    mode VARCHAR(20) NOT NULL,
    
    -- Position state at closure
    final_size DECIMAL(20, 8) NOT NULL DEFAULT 0,  -- Always 0 for closed positions, but stored for completeness
    average_entry_price DECIMAL(20, 8),  -- Entry price for PnL calculation
    exit_price DECIMAL(20, 8),  -- Price at closure (from current_price at close time)
    current_price DECIMAL(20, 8),  -- Same as exit_price, kept for consistency with positions table
    
    -- PnL at closure
    realized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
    unrealized_pnl_at_close DECIMAL(20, 8) NOT NULL DEFAULT 0,
    
    -- Hedge mode fields
    long_size DECIMAL(20, 8),
    short_size DECIMAL(20, 8),
    long_avg_price DECIMAL(20, 8),
    short_avg_price DECIMAL(20, 8),
    
    -- Fees (if available)
    total_fees DECIMAL(20, 8),
    
    -- Metadata
    opened_at TIMESTAMP NOT NULL,  -- created_at from original position
    closed_at TIMESTAMP NOT NULL,  -- When position was closed
    version INTEGER NOT NULL,  -- Version at closure
    
    CONSTRAINT chk_mode CHECK (mode IN ('one-way', 'hedge')),
    CONSTRAINT chk_avg_entry_price CHECK (average_entry_price IS NULL OR average_entry_price > 0),
    CONSTRAINT chk_exit_price CHECK (exit_price IS NULL OR exit_price > 0),
    CONSTRAINT chk_long_avg_price CHECK (long_avg_price IS NULL OR long_avg_price > 0),
    CONSTRAINT chk_short_avg_price CHECK (short_avg_price IS NULL OR short_avg_price > 0)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_closed_positions_asset ON closed_positions(asset);
CREATE INDEX IF NOT EXISTS idx_closed_positions_closed_at ON closed_positions(closed_at DESC);
CREATE INDEX IF NOT EXISTS idx_closed_positions_opened_at ON closed_positions(opened_at);
CREATE INDEX IF NOT EXISTS idx_closed_positions_asset_closed_at ON closed_positions(asset, closed_at DESC);
CREATE INDEX IF NOT EXISTS idx_closed_positions_original_position_id ON closed_positions(original_position_id);

COMMIT;

-- Rollback section
-- WARNING: Dropping table will remove all closed position history. Use with caution.
-- To rollback this migration, run:
--
-- BEGIN;
-- DROP INDEX IF EXISTS idx_closed_positions_original_position_id;
-- DROP INDEX IF EXISTS idx_closed_positions_asset_closed_at;
-- DROP INDEX IF EXISTS idx_closed_positions_opened_at;
-- DROP INDEX IF EXISTS idx_closed_positions_closed_at;
-- DROP INDEX IF EXISTS idx_closed_positions_asset;
-- DROP TABLE IF EXISTS closed_positions;
-- COMMIT;

