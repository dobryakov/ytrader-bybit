-- Migration: Create signal_order_relationships table for order-manager service
-- Reversible: Yes (see rollback section at bottom)

CREATE TABLE IF NOT EXISTS signal_order_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id UUID NOT NULL,
    order_id UUID NOT NULL,
    relationship_type VARCHAR(20) NOT NULL,
    execution_sequence INTEGER,
    allocation_amount DECIMAL(20, 8),
    allocation_quantity DECIMAL(20, 8),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_relationship_type CHECK (relationship_type IN ('one_to_one', 'one_to_many', 'many_to_one')),
    CONSTRAINT chk_execution_sequence CHECK (execution_sequence IS NULL OR execution_sequence >= 1),
    CONSTRAINT chk_allocation_amount CHECK (allocation_amount IS NULL OR allocation_amount > 0),
    CONSTRAINT chk_allocation_quantity CHECK (allocation_quantity IS NULL OR allocation_quantity > 0),
    CONSTRAINT uq_signal_order UNIQUE (signal_id, order_id),
    CONSTRAINT fk_order FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_signal_order_rel_signal_id ON signal_order_relationships(signal_id);
CREATE INDEX IF NOT EXISTS idx_signal_order_rel_order_id ON signal_order_relationships(order_id);
CREATE INDEX IF NOT EXISTS idx_signal_order_rel_type ON signal_order_relationships(relationship_type);

-- Rollback (reverse migration):
-- DROP INDEX IF EXISTS idx_signal_order_rel_type;
-- DROP INDEX IF EXISTS idx_signal_order_rel_order_id;
-- DROP INDEX IF EXISTS idx_signal_order_rel_signal_id;
-- DROP TABLE IF EXISTS signal_order_relationships;

