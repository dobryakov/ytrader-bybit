-- Migration: Add rejection_reason column to orders table
-- Reversible: Yes (see rollback section at bottom)
-- Purpose: Store reason for order rejection to enable analysis and monitoring

ALTER TABLE orders
ADD COLUMN IF NOT EXISTS rejection_reason TEXT;

COMMENT ON COLUMN orders.rejection_reason IS 'Reason for order rejection (for orders with status=rejected)';

-- Rollback (reverse migration):
-- ALTER TABLE orders DROP COLUMN IF EXISTS rejection_reason;

