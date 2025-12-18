-- Migration: Create position_orders table
-- Reversible: Yes
-- Purpose: Link positions with orders that created/modified/closed them

CREATE TABLE IF NOT EXISTS position_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    position_id UUID NOT NULL REFERENCES positions(id) ON DELETE CASCADE,
    order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    relationship_type VARCHAR(20) NOT NULL CHECK (relationship_type IN ('opened', 'increased', 'decreased', 'closed', 'reversed')),
    size_delta DECIMAL(20, 8) NOT NULL,
    execution_price DECIMAL(20, 8) NOT NULL CHECK (execution_price > 0),
    executed_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT uq_position_order UNIQUE (position_id, order_id)
);

CREATE INDEX idx_position_orders_position_id ON position_orders(position_id);
CREATE INDEX idx_position_orders_order_id ON position_orders(order_id);
CREATE INDEX idx_position_orders_executed_at ON position_orders(executed_at DESC);
CREATE INDEX idx_position_orders_relationship_type ON position_orders(relationship_type);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_position_orders_relationship_type;
-- DROP INDEX IF EXISTS idx_position_orders_executed_at;
-- DROP INDEX IF EXISTS idx_position_orders_order_id;
-- DROP INDEX IF EXISTS idx_position_orders_position_id;
-- DROP TABLE IF EXISTS position_orders;

