-- Migration: Create orders table for order-manager service
-- Reversible: Yes (see rollback section at bottom)

CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id VARCHAR(100) NOT NULL UNIQUE,
    signal_id UUID NOT NULL,
    asset VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    filled_quantity DECIMAL(20, 8) NOT NULL DEFAULT 0,
    average_price DECIMAL(20, 8),
    fees DECIMAL(20, 8),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    executed_at TIMESTAMP,
    trace_id VARCHAR(100),
    is_dry_run BOOLEAN NOT NULL DEFAULT false,
    
    CONSTRAINT chk_side CHECK (side IN ('Buy', 'SELL')),
    CONSTRAINT chk_order_type CHECK (order_type IN ('Market', 'Limit')),
    CONSTRAINT chk_status CHECK (status IN ('pending', 'partially_filled', 'filled', 'cancelled', 'rejected', 'dry_run')),
    CONSTRAINT chk_quantity_positive CHECK (quantity > 0),
    CONSTRAINT chk_filled_quantity CHECK (filled_quantity >= 0 AND filled_quantity <= quantity),
    CONSTRAINT chk_price_limit_order CHECK (
        (order_type = 'Limit' AND price IS NOT NULL AND price > 0) OR
        (order_type = 'Market')
    )
);

CREATE INDEX IF NOT EXISTS idx_orders_signal_id ON orders(signal_id);
CREATE INDEX IF NOT EXISTS idx_orders_asset ON orders(asset);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_orders_order_id;
-- DROP INDEX IF EXISTS idx_orders_created_at;
-- DROP INDEX IF EXISTS idx_orders_status;
-- DROP INDEX IF EXISTS idx_orders_asset;
-- DROP INDEX IF EXISTS idx_orders_signal_id;
-- DROP TABLE IF EXISTS orders;

