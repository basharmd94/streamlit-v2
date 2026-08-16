-- ─────────────────────────────────────────────────────────────────────────────
-- mv_stock_movement
-- Rebuilt: August 2026
--
-- Row-level signed stock-movement ledger — ONE ROW PER imtrn ROW, all ZIDs,
-- all transaction types. No aggregation. Used to compute onhand_before /
-- onhand_end for the FIFO engine in Batch P&L and Batch Con.
--
-- Item code resolution uses caitem.xdrawing (NOT packcode — that column does
-- not exist on caitem). xdrawing serves a dual purpose:
--   - In 100009 caitem: stores the 100001-equivalent item code (cross-ZID map).
--   - In 100001 caitem: stores a variant-consolidation code.
-- Skip xdrawing when null, blank, 'NO', or starting with 'KH' (colour/variant
-- suffixes, not real mappings) — fall back to the native xitem code.
--
-- INNER JOIN caitem: an imtrn row with no caitem match is dropped (matches
-- the original pre-2026 mv_stock_movement behaviour).
--
-- stockqty / stockvalue are SIGNED: positive = inflow, negative = outflow.
--
-- Output columns (backward-compatible with old mv_stock_movement):
--   zid, year, month, date, docnum, project, itemcode, itemname,
--   itemgroup, warehouse, stockqty, stockvalue
--
-- Refresh schedule: add to programmatic MV refresh list.
-- ─────────────────────────────────────────────────────────────────────────────

DROP MATERIALIZED VIEW IF EXISTS mv_stock_movement;

CREATE MATERIALIZED VIEW mv_stock_movement AS
SELECT
    i.zid,
    i.xyear AS year,
    i.xper  AS month,
    i.xdate AS date,
    i.xdocnum AS docnum,
    i.xproj AS project,
    CASE
        WHEN ci.xdrawing IS NOT NULL
         AND ci.xdrawing::text <> ''::text
         AND ci.xdrawing::text <> 'NO'::text
         AND LEFT(ci.xdrawing::text, 2) <> 'KH'::text
        THEN ci.xdrawing::text
        ELSE i.xitem::text
    END AS itemcode,
    ci.xdesc  AS itemname,
    ci.xgitem AS itemgroup,
    i.xwh     AS warehouse,
    i.xqty * i.xsign::numeric AS stockqty,
    i.xval * i.xsign::numeric AS stockvalue
FROM imtrn i
JOIN caitem ci ON i.xitem::text = ci.xitem::text AND i.zid = ci.zid;

-- ── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX ON mv_stock_movement (zid, itemcode, date);
CREATE INDEX ON mv_stock_movement (zid, date);
CREATE INDEX ON mv_stock_movement (itemcode, date);
CREATE INDEX ON mv_stock_movement (zid, itemcode, warehouse, date);
