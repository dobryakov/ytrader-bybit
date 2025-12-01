-- Migration: Create position_states table for exit strategy evaluation
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Tracks position lifecycle data (entry price, peak price, highest PnL) for exit strategy evaluation

CREATE TABLE IF NOT EXISTS position_states (
    asset VARCHAR(20) PRIMARY KEY,
    entry_price DECIMAL(20, 8) NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    peak_price DECIMAL(20, 8) NOT NULL,
    highest_unrealized_pnl DECIMAL(20, 8) NOT NULL DEFAULT 0,
    last_exit_signal_time TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_entry_price_positive CHECK (entry_price > 0),
    CONSTRAINT chk_peak_price_positive CHECK (peak_price > 0)
);

CREATE INDEX IF NOT EXISTS idx_position_states_entry_time ON position_states(entry_time);
CREATE INDEX IF NOT EXISTS idx_position_states_updated_at ON position_states(updated_at);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_position_states_updated_at;
-- DROP INDEX IF EXISTS idx_position_states_entry_time;
-- DROP TABLE IF EXISTS position_states;

