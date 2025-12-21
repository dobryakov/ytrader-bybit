-- Migration: Add total_fees column to positions table
-- Reversible: Yes
-- Purpose: Track cumulative fees for positions separately from realized_pnl
--          This allows validation and transparency of fee tracking

-- Add total_fees column with default value 0
ALTER TABLE positions 
ADD COLUMN IF NOT EXISTS total_fees DECIMAL(20, 8) DEFAULT 0 NOT NULL;

-- Rollback (reverse migration):
-- ALTER TABLE positions DROP COLUMN IF EXISTS total_fees;

