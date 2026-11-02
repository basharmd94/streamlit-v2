-- Fixit ERP database ("fixit" — restored from the Fixit `da` backup, NOT the HMBR `da`).
-- Item master + current stock + average monthly sales, one row per (zid, item).
-- Modeled on HMBR's `final_items_view` (same name and column names so app queries can be pointed at this DB),
-- but the business rules are Fixit's own — they were verified against this data, not copied from HMBR.
--
--   Companies kept: 100001 Fixit Gulshan (store), 100002 Fixit Central (central distribution warehouse — STOCK counts,
--   but it makes no customer sales so its avg_monthly_sales is forced to 0) and 100003 Ecommerce Warehouse (online shop).
--   Deliberately left out (owner decision): 100004 Fixit Green Road — store closed down.
--   (100000 is a 9-item placeholder with no stock; it is still listed.)
--
--   stock              = SUM(xqty * xsign) over the zid's own main warehouse only (list below).
--                        Excluded on purpose (owner decision): blank xwh rows (100001, -322 units), 'xfwh' (100003, 11 units),
--                        'Damage Store' (100002, 16 units) and 'General Store(Fixit Central)' (100002, nets to 0).
--   avg_monthly_sales  = NET units sold over the last 3 COMPLETE calendar months, / 3.0 — same window and divisor as HMBR's
--                        view. The current partial month is excluded.
--                        Net = customer-order issues (imtrn xdoctype 'IS--', docnum CO--nnnnnn) MINUS stock put back by
--                        'SRE-' documents, i.e. -SUM(xqty * xsign) over ('IS--','SRE-').
--                        Only 100001 and 100003 are measured. 100002 (Central) is excluded from the sales CTE on purpose: its
--                        'SRE-' documents are stock put back with no matching sale, which would show as negative sales.
--                        'IS--' lines reconcile exactly to opodt order lines. 'SRE-' is both genuine customer returns
--                        (header action 'Sales Return') and order corrections ("CO--241039 Change to CO--241041" — old order
--                        reversed, replacement order issues the goods again); netting it removes the double count.
--                        NOT counted as sales: PRE- (purchase return to supplier), ISS- (stock adjustment),
--                        TO-- (transfer between branches), REC-/IGRN (receipts).
--   Items newer than 3 months are averaged over 3 anyway (understated) — same as HMBR.
--
--   PRICE CHAIN: Central buys from outside (local PO -> GRN-nnnnnn, international IP -> IGRNnnnnnn), then sells to
--   Gulshan / Ecommerce at a transfer rate, who sell at their own std_price.
--   central_buy_price / central_buy_date   = what CENTRAL last paid a supplier for the item: latest confirmed GRN/IGRN line
--                        (pogdt/pogrn, zid 100002), net BDT per stock unit = xlineamt / xqty. Equals the unit value booked
--                        into imtrn (99.95% of lines checked); import rates are already landed BDT. Same value on every
--                        zid's row for that item code (it is Central's price). NULL = Central never bought it (e.g. opening stock only).
--   store_buy_price / store_buy_date       = the rate at which THIS store last bought the item from Central, i.e. Central's
--                        sale price to it: latest confirmed GRN in the store's own books with supplier SUP-000001 (= Central
--                        in both stores). Only filled on 100001 / 100003 rows. Agrees with Central's TO-- invoice rate for the
--                        same shipment (99.9%); the store's GRN is used because Central's TO-- lines are patchy in older years.
--   Both are "latest ever" — check the date columns, many prices are years old.
--
-- The view is a snapshot: re-run `REFRESH MATERIALIZED VIEW CONCURRENTLY public.final_items_view;`
-- to move the 3-month window and pick up new stock. A fresh restore of the backup drops this view — re-run this script.
-- Changing the definition below needs this whole script (DROP + CREATE); REFRESH alone keeps the old definition.

DROP MATERIALIZED VIEW IF EXISTS public.final_items_view;

CREATE MATERIALIZED VIEW public.final_items_view AS
WITH stk AS (
    SELECT im.zid, im.xitem, SUM(im.xqty * im.xsign) AS stock
    FROM imtrn im
    WHERE (im.zid, im.xwh) IN ((100001, 'Fixit Gulshan'),
                               (100002, 'Fixit Central'),
                               (100003, 'Ecommerce Warehouse'))
    GROUP BY im.zid, im.xitem
),
cbuy AS (
    SELECT DISTINCT ON (d.xitem) d.xitem, g.xdate AS buy_date, round(d.xlineamt / d.xqty, 4) AS buy_price
    FROM pogdt d
    JOIN pogrn g ON g.zid = d.zid AND g.xgrnnum = d.xgrnnum
    WHERE g.zid = 100002 AND g.xstatusgrn = '5-Confirmed'
      AND d.xqty > 0 AND d.xlineamt > 0 AND g.xdate < '2900-01-01'
    ORDER BY d.xitem, g.xdate DESC, g.xgrnnum DESC
),
sbuy AS (
    SELECT DISTINCT ON (g.zid, d.xitem) g.zid, d.xitem, g.xdate AS buy_date, round(d.xlineamt / d.xqty, 4) AS buy_price
    FROM pogdt d
    JOIN pogrn g ON g.zid = d.zid AND g.xgrnnum = d.xgrnnum
    WHERE g.zid IN (100001, 100003) AND g.xsup = 'SUP-000001' AND g.xstatusgrn = '5-Confirmed'
      AND d.xqty > 0 AND d.xlineamt > 0 AND g.xdate < '2900-01-01'
    ORDER BY g.zid, d.xitem, g.xdate DESC, g.xgrnnum DESC
),
sal AS (
    SELECT im.zid, im.xitem, SUM(-im.xqty * im.xsign) / 3.0 AS avg_monthly_sales
    FROM imtrn im
    WHERE im.zid IN (100001, 100003)
      AND im.xdoctype IN ('IS--', 'SRE-')
      AND im.xdate >= (date_trunc('month', current_date) - interval '3 months')::date
      AND im.xdate <  date_trunc('month', current_date)::date
    GROUP BY im.zid, im.xitem
)
SELECT ci.zid,
       ci.xitem                          AS item_id,
       ci.xdesc                          AS item_name,
       ci.xgitem                         AS item_group,
       COALESCE(ci.xstdprice, 0.0)       AS std_price,
       ci.xunitstk                       AS stock_unit,
       COALESCE(stk.stock, 0::numeric)   AS stock,
       COALESCE(sal.avg_monthly_sales, 0::numeric) AS avg_monthly_sales,
       cbuy.buy_price                    AS central_buy_price,
       cbuy.buy_date                     AS central_buy_date,
       sbuy.buy_price                    AS store_buy_price,
       sbuy.buy_date                     AS store_buy_date
FROM caitem ci
LEFT JOIN stk ON stk.zid = ci.zid AND stk.xitem = ci.xitem
LEFT JOIN sal ON sal.zid = ci.zid AND sal.xitem = ci.xitem
LEFT JOIN cbuy ON cbuy.xitem = ci.xitem
LEFT JOIN sbuy ON sbuy.zid = ci.zid AND sbuy.xitem = ci.xitem
WHERE ci.zid <> 100004
ORDER BY ci.zid, ci.xitem;

-- (zid, item_id) is unique (caitem PK is (zid, xitem)) — required for REFRESH ... CONCURRENTLY.
CREATE UNIQUE INDEX final_items_view_zid_item_id_idx ON public.final_items_view (zid, item_id);
