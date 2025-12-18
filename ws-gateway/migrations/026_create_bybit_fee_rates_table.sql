-- Migration: Create bybit_fee_rates table for caching Bybit trading fee rates
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Store per-symbol maker/taker fee rates so that Order Manager can
--          avoid calling Bybit API for fee information on every order.
--
-- Notes:
-- - This table is owned by ws-gateway service per constitution (shared DB).
-- - Data is periodically refreshed by Order Manager via Bybit REST API.
--

CREATE TABLE IF NOT EXISTS bybit_fee_rates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(50) NOT NULL,
    market_type VARCHAR(32) NOT NULL,
    maker_fee_rate DECIMAL(20, 10) NOT NULL,
    taker_fee_rate DECIMAL(20, 10) NOT NULL,
    last_synced_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_bybit_fee_rates_symbol_market_type UNIQUE (symbol, market_type)
);

CREATE INDEX IF NOT EXISTS idx_bybit_fee_rates_symbol_market_type
    ON bybit_fee_rates(symbol, market_type);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_bybit_fee_rates_symbol_market_type;
-- DROP TABLE IF EXISTS bybit_fee_rates;



