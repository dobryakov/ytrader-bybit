-- Migration: Create instrument_info table for Bybit instruments-info persistence
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Store normalized instruments-info data for symbols used by Order Manager
--          so that order validation and risk logic can rely on cached limits instead
--          of calling Bybit API on every request.
--
-- Notes:
-- - This table is owned by ws-gateway service per constitution (shared DB migrations).
-- - Data is periodically refreshed by Order Manager service via Bybit REST API.
-- - raw_response keeps full instruments-info payload for future extensions.

CREATE TABLE IF NOT EXISTS instrument_info (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(50) NOT NULL UNIQUE,
    base_coin VARCHAR(20),
    quote_coin VARCHAR(20),
    status VARCHAR(32),
    lot_size DECIMAL(20, 8) NOT NULL,
    min_order_qty DECIMAL(20, 8) NOT NULL,
    max_order_qty DECIMAL(20, 8) NOT NULL,
    min_order_value DECIMAL(20, 8) NOT NULL,
    price_tick_size DECIMAL(20, 8) NOT NULL,
    price_limit_ratio_x DECIMAL(20, 8),
    price_limit_ratio_y DECIMAL(20, 8),
    raw_response JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_instrument_info_symbol
    ON instrument_info(symbol);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_instrument_info_symbol;
-- DROP TABLE IF EXISTS instrument_info;


