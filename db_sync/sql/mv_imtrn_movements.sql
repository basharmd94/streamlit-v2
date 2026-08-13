-- ─────────────────────────────────────────────────────────────────────────────
-- mv_imtrn_movements
-- Rebuilt: August 2026
--
-- Classifies every imtrn row by txn_type and adds cross-ZID item-code mapping
-- so 100009 depletion events are visible under their 100001 item codes.
--
-- Changes vs. original MV:
--   1. Cross-ZID item code mapping via caitem.xdrawing
--        100009 item HPI000115 → itemcode '2145'  (example)
--      Allows FIFO engine querying ZID 100001 to see depletions that physically
--      occurred in 100009 (ISS / DO-- on the same shared inventory).
--
--   2. Local GRN receipts now classified as 'purchase' instead of 'adjustment'.
--      Identified by: ximtrnnum LIKE 'RE--%' AND xdocnum LIKE 'GRN%'
--      (xdoctype is a numeric warehouse-form code that varies by site and cannot
--       be enumerated; the ximtrnnum + xdocnum combination is reliable.)
--
--   3. Manufacturing-output receipts (xdoctype='RE--', xdocnum LIKE 'MO--%')
--      reclassified 'cust_return' → 'adjustment'.
--      These are finished-goods receipts from production orders, not customer
--      returns.  The old classification was inflating return_qty in the FIFO
--      engine and suppressing net depletion (higher remaining_qty than actual).
--
-- Run on the server after any schema change, then REFRESH the MV.
-- ─────────────────────────────────────────────────────────────────────────────

DROP MATERIALIZED VIEW IF EXISTS mv_imtrn_movements;

CREATE MATERIALIZED VIEW mv_imtrn_movements AS
WITH cross_zid_map AS (
    -- For each 100009 item that carries a xdrawing link to a 100001 item code.
    -- xdrawing in 100009's caitem stores the corresponding 100001 item code.
    SELECT
        ci.xitem    AS code_100009,   -- 100009 native code   e.g. 'HPI000115'
        ci.xdrawing AS code_100001    -- 100001 equivalent    e.g. '2145'
    FROM caitem ci
    WHERE ci.zid::TEXT = '100009'
      AND ci.xdrawing IS NOT NULL
      AND TRIM(ci.xdrawing) <> ''
)
SELECT
    i.zid::TEXT                                                          AS zid,
    i.ximtrnnum,
    i.xdocnum,

    -- ── Item code ────────────────────────────────────────────────────────────
    -- For 100009 cross-ZID items, use the 100001-equivalent code (xdrawing).
    -- All other rows keep their native xitem.
    COALESCE(
        CASE
            WHEN i.zid::TEXT = '100009' AND m.code_100001 IS NOT NULL
            THEN m.code_100001
        END,
        i.xitem
    )                                                                    AS itemcode,

    i.xwh                                                                AS warehouse,
    i.xdate                                                              AS txn_date,
    i.xyear                                                              AS year,
    i.xper                                                               AS month,
    i.xqty,
    i.xval,
    i.xsign,
    i.xqty * i.xsign::NUMERIC                                           AS net_qty,
    i.xval * i.xsign::NUMERIC                                           AS net_val,

    -- ── Transaction type ─────────────────────────────────────────────────────
    CASE
        -- Import GRN (unchanged from original)
        WHEN i.xdoctype = 'IGRN'
            THEN 'purchase'

        -- Local GRN receipt: ximtrnnum='RE--*', xdocnum='GRN*'.
        -- xdoctype is a numeric warehouse-form code (0040, 0045, 0095 …) that
        -- varies by site — cannot be enumerated.  ximtrnnum + xdocnum reliable.
        -- Covers 100001 local GRNs AND 100009 intercompany receipts from 100001.
        WHEN i.ximtrnnum LIKE 'RE--%' AND i.xdocnum LIKE 'GRN%'
            THEN 'purchase'

        -- Customer-facing sales (delivery orders)
        WHEN i.xdoctype = 'DO--'
            THEN 'sale'

        -- Customer returns (all known return document types)
        WHEN i.xdoctype IN ('SR--', 'SRE-', 'RECA', 'RECT', 'REC-', 'DSR-')
            THEN 'cust_return'

        -- Manufacturing-output receipts (xdoctype='RE--', xdocnum='MO--*').
        -- Changed from 'cust_return' → 'adjustment': these are finished-goods
        -- receipts from production, not customer returns.
        WHEN i.xdoctype = 'RE--'
            THEN 'adjustment'

        -- Internal issues (raw material draw-down / production)
        WHEN i.xdoctype IN ('IS--', 'ISS-')
            THEN 'issue'

        -- Warehouse transfers
        WHEN i.xdoctype = 'TO--'
            THEN 'transfer'

        -- Purchase returns to supplier
        WHEN i.xdoctype = 'PRE-'
            THEN 'purch_return'

        -- Catch-all (numeric-doctype adjustments, etc.)
        ELSE 'adjustment'
    END                                                                  AS txn_type

FROM imtrn i
LEFT JOIN cross_zid_map m
    ON  i.xitem     = m.code_100009
    AND i.zid::TEXT = '100009';   -- join only for 100009 rows

-- ── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX ON mv_imtrn_movements (zid, itemcode, txn_date);
CREATE INDEX ON mv_imtrn_movements (zid, txn_type, txn_date);
CREATE INDEX ON mv_imtrn_movements (itemcode, txn_date);
