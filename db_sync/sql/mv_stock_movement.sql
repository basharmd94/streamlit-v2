-- ─────────────────────────────────────────────────────────────────────────────
-- mv_stock_movement
-- Rebuilt: August 2026
--
-- Comprehensive signed stock-movement ledger — every imtrn row, all ZIDs,
-- all transaction types.  Used to compute onhand_before / onhand_end for
-- the FIFO engine in Batch P&L and Batch Con.
--
-- Changes vs. original MV:
--   1. Packcode resolution now uses caitem.packcode (not ci.xdrawing).
--      xdrawing is reserved for cross-ZID item mapping (100009→100001);
--      packcode is the correct intra-ZID consolidated-item resolver.
--
--   2. No cross-ZID item code remapping (xdrawing not applied here).
--      100009 rows keep their native xitem after packcode resolution.
--      The batch P&L engine loads both zids separately and combines them
--      in Python, where the cross-ZID relationship is handled explicitly.
--
--   3. No ZID filter — MV covers all ZIDs so a single refresh serves
--      every entity.  Query-time WHERE zid = %s applied by get_stock_movement_data.
--
-- stockqty / stockvalue are SIGNED: positive = inflow, negative = outflow.
-- This matches SUM(xqty * xsign) / SUM(xval * xsign) convention.
--
-- Output columns (backward-compatible with old mv_stock_movement):
--   zid, year, month, date, docnum, project, itemcode, itemname,
--   itemgroup, warehouse, stockqty, stockvalue
--
-- Refresh schedule: add to programmatic MV refresh list.
-- For data-only changes (no schema change): REFRESH MATERIALIZED VIEW mv_stock_movement;
-- For definition changes: DROP + CREATE (this script).
-- ─────────────────────────────────────────────────────────────────────────────

DROP MATERIALIZED VIEW IF EXISTS mv_stock_movement;

CREATE MATERIALIZED VIEW mv_stock_movement AS
WITH stock AS (
    SELECT
        i.zid::TEXT                                            AS zid,
        i.xyear                                                AS year,
        i.xper                                                 AS month,
        i.xdate::date                                          AS date,
        i.xdocnum                                              AS docnum,
        i.xproj                                                AS project,

        -- ── Packcode resolution ───────────────────────────────────────────
        -- Use caitem.packcode to consolidate packaging variants to their
        -- canonical item code.  Skip packcode when null, blank, 'NO', or
        -- starting with 'KH' (those are colour/variant suffixes, not packs).
        CASE
            WHEN c.packcode IS NOT NULL
             AND TRIM(c.packcode) <> ''
             AND c.packcode <> 'NO'
             AND LEFT(c.packcode, 2) <> 'KH'
            THEN c.packcode
            ELSE i.xitem
        END                                                    AS itemcode,

        c.xdesc                                                AS itemname,
        c.xgitem                                               AS itemgroup,
        i.xwh                                                  AS warehouse,

        -- Signed quantities: positive = stock in, negative = stock out
        SUM(i.xqty * i.xsign::NUMERIC)                        AS stockqty,
        SUM(i.xval * i.xsign::NUMERIC)                        AS stockvalue

    FROM imtrn i
    LEFT JOIN caitem c
        ON  i.xitem = c.xitem
        AND i.zid   = c.zid
    GROUP BY
        i.zid,
        i.xyear,
        i.xper,
        i.xdate::date,
        i.xdocnum,
        i.xproj,
        CASE
            WHEN c.packcode IS NOT NULL
             AND TRIM(c.packcode) <> ''
             AND c.packcode <> 'NO'
             AND LEFT(c.packcode, 2) <> 'KH'
            THEN c.packcode
            ELSE i.xitem
        END,
        c.xdesc,
        c.xgitem,
        i.xwh
)
SELECT * FROM stock;

-- ── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX ON mv_stock_movement (zid, itemcode, date);
CREATE INDEX ON mv_stock_movement (zid, date);
CREATE INDEX ON mv_stock_movement (itemcode, date);
CREATE INDEX ON mv_stock_movement (zid, itemcode, warehouse, date);
