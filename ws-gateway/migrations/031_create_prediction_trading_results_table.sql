-- Migration: Create prediction_trading_results table
-- Reversible: Yes
-- Purpose: Link predictions with actual trading PnL for model quality evaluation

CREATE TABLE IF NOT EXISTS prediction_trading_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_target_id UUID NOT NULL REFERENCES prediction_targets(id) ON DELETE CASCADE,
    signal_id UUID NOT NULL REFERENCES trading_signals(signal_id) ON DELETE CASCADE,
    
    -- Trading PnL metrics
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    fees DECIMAL(20, 8) DEFAULT 0,
    
    -- Entry/exit information
    entry_price DECIMAL(20, 8),
    exit_price DECIMAL(20, 8),
    entry_signal_id UUID REFERENCES trading_signals(signal_id),
    exit_signal_id UUID REFERENCES trading_signals(signal_id),
    
    -- Position information
    position_id UUID REFERENCES positions(id),
    position_size_at_entry DECIMAL(20, 8),
    position_size_at_exit DECIMAL(20, 8),
    
    -- Timestamps
    entry_timestamp TIMESTAMP,
    exit_timestamp TIMESTAMP,
    
    -- Status
    is_closed BOOLEAN NOT NULL DEFAULT false,
    is_partial_close BOOLEAN NOT NULL DEFAULT false,
    
    -- Metadata
    computed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_pnl_consistency CHECK (
        total_pnl = realized_pnl + unrealized_pnl
    ),
    CONSTRAINT chk_entry_exit_prices CHECK (
        (entry_price IS NULL OR entry_price > 0) AND
        (exit_price IS NULL OR exit_price > 0)
    )
);

-- Indexes
CREATE INDEX idx_prediction_trading_results_prediction_target_id ON prediction_trading_results(prediction_target_id);
CREATE INDEX idx_prediction_trading_results_signal_id ON prediction_trading_results(signal_id);
CREATE INDEX idx_prediction_trading_results_entry_signal_id ON prediction_trading_results(entry_signal_id);
CREATE INDEX idx_prediction_trading_results_exit_signal_id ON prediction_trading_results(exit_signal_id);
CREATE INDEX idx_prediction_trading_results_position_id ON prediction_trading_results(position_id);
CREATE INDEX idx_prediction_trading_results_is_closed ON prediction_trading_results(is_closed);
CREATE INDEX idx_prediction_trading_results_total_pnl ON prediction_trading_results(total_pnl DESC);
CREATE INDEX idx_prediction_trading_results_computed_at ON prediction_trading_results(computed_at DESC);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_prediction_trading_results_computed_at;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_total_pnl;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_is_closed;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_position_id;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_exit_signal_id;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_entry_signal_id;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_signal_id;
-- DROP INDEX IF EXISTS idx_prediction_trading_results_prediction_target_id;
-- DROP TABLE IF EXISTS prediction_trading_results;

