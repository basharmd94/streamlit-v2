-- Drop mv_imtrn_movements
-- Run AFTER mv_issues_daily_item and mv_stock_movement have been created
-- and the Python code has been reverted to use mv_stock_movement.
--
-- mv_imtrn_movements is replaced by:
--   mv_stock_movement      — comprehensive signed ledger (onhand_before / balance)
--   mv_issues_daily_item   — IS--, ISS- depletion events (new)
--   mv_sales_daily_item    — DO-- depletion events (unchanged)
--   mv_returns_daily_item  — SR-- etc. return events (unchanged)

DROP MATERIALIZED VIEW IF EXISTS mv_imtrn_movements;
