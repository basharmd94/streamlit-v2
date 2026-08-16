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
-- Item code resolution uses caitem.xdrawing (NOT packcode — that column does
-- not exist on caitem). Same CASE as mv_stock_movement: this is what remaps
-- 100009 issue events onto their 100001-equivalent item codes so the FIFO
-- engine sees them without needing an explicit ZID filter.
--
-- Run: DROP + CREATE (structure change) OR REFRESH (data-only refresh).
-- Add to programmatic MV refresh list alongside mv_sales_daily_item.
-- ─────────────────────────────────────────────────────────────────────────────

DROP MATERIALIZED VIEW IF EXISTS mv_issues_daily_item;

CREATE MATERIALIZED VIEW mv_issues_daily_item AS
SELECT
    i.zid::text AS zid,

    -- ── xdrawing resolution ─────────────────────────────────────────────────
    -- Same CASE as mv_stock_movement.
    CASE
        WHEN ci.xdrawing IS NOT NULL
         AND ci.xdrawing::text <> ''::text
         AND ci.xdrawing::text <> 'NO'::text
         AND LEFT(ci.xdrawing::text, 2) <> 'KH'::text
        THEN ci.xdrawing::text
        ELSE i.xitem::text
    END AS itemcode,

    i.xdate::date AS date,
    i.xyear AS year,
    i.xper  AS month,

    -- Positive quantity (outflow implied by IS-- / ISS- doctype)
    SUM(i.xqty) AS issue_qty,
    SUM(i.xval) AS issue_val

FROM imtrn i
JOIN caitem ci ON i.xitem::text = ci.xitem::text AND i.zid = ci.zid
WHERE i.xdoctype IN ('IS--', 'ISS-')
GROUP BY
    i.zid,
    CASE
        WHEN ci.xdrawing IS NOT NULL
         AND ci.xdrawing::text <> ''::text
         AND ci.xdrawing::text <> 'NO'::text
         AND LEFT(ci.xdrawing::text, 2) <> 'KH'::text
        THEN ci.xdrawing::text
        ELSE i.xitem::text
    END,
    i.xdate::date,
    i.xyear,
    i.xper;

-- ── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX ON mv_issues_daily_item (zid, itemcode, date);
CREATE INDEX ON mv_issues_daily_item (zid, date);
CREATE INDEX ON mv_issues_daily_item (itemcode, date);
