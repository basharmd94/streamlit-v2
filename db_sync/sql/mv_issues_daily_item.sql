-- ─────────────────────────────────────────────────────────────────────────────
-- mv_issues_daily_item
-- Created: August 2026
--
-- Daily internal-issue quantities per (zid, itemcode, date).
-- Covers IS-- and ISS- document types — internal stock draw-downs,
-- raw-material issues to production, and any other IS-type write-downs.
--
-- Mirrors the structure of mv_returns_daily_item and mv_sales_daily_item so
-- all three depletion-event MVs are consistent and can be combined in the
-- FIFO engine's _build_daily_events pass.
--
-- issue_qty is stored as a POSITIVE value (outflow implied by txn type).
-- The FIFO engine treats it the same way it treats returnqty — a quantity
-- that is subtracted from cumulative stock at that date.
--
-- Packcode resolution: items whose caitem.packcode is non-null and not 'NO'
-- or 'KH*' are mapped to their consolidated code, consistent with the CLAUDE.md
-- SQL rule and with mv_sales_daily_item / mv_stock_movement.
--
-- Run: DROP + CREATE (structure change) OR REFRESH (data-only refresh).
-- Add to programmatic MV refresh list alongside mv_sales_daily_item.
-- ─────────────────────────────────────────────────────────────────────────────

DROP MATERIALIZED VIEW IF EXISTS mv_issues_daily_item;

CREATE MATERIALIZED VIEW mv_issues_daily_item AS
SELECT
    i.zid::TEXT                                                AS zid,

    -- ── Packcode resolution ───────────────────────────────────────────────
    -- Same CASE as mv_stock_movement and mv_sales_daily_item.
    CASE
        WHEN c.packcode IS NOT NULL
         AND TRIM(c.packcode) <> ''
         AND c.packcode <> 'NO'
         AND LEFT(c.packcode, 2) <> 'KH'
        THEN c.packcode
        ELSE i.xitem
    END                                                        AS itemcode,

    i.xdate::date                                              AS date,
    i.xyear                                                    AS year,
    i.xper                                                     AS month,

    -- Positive quantity (outflow implied by IS-- / ISS- doctype)
    SUM(i.xqty)                                                AS issue_qty,
    SUM(i.xval)                                                AS issue_val

FROM imtrn i
LEFT JOIN caitem c
    ON  i.xitem = c.xitem
    AND i.zid   = c.zid
WHERE i.xdoctype IN ('IS--', 'ISS-')
GROUP BY
    i.zid,
    CASE
        WHEN c.packcode IS NOT NULL
         AND TRIM(c.packcode) <> ''
         AND c.packcode <> 'NO'
         AND LEFT(c.packcode, 2) <> 'KH'
        THEN c.packcode
        ELSE i.xitem
    END,
    i.xdate::date,
    i.xyear,
    i.xper;

-- ── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX ON mv_issues_daily_item (zid, itemcode, date);
CREATE INDEX ON mv_issues_daily_item (zid, date);
CREATE INDEX ON mv_issues_daily_item (itemcode, date);
