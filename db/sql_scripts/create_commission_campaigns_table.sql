-- ─────────────────────────────────────────────────────────────────────────────
-- commission_campaigns — app-owned table for the "🎯 Product / Stock Clearance /
-- Slow-Moving Campaign" section of Commissions (views/commissions.py). Same
-- convention as crm_call_log / marketing_leads / page_permissions: created once,
-- written directly by the app via core/db.py, no separate role/grants needed.
--
-- Holds campaign DEFINITIONS only -- no salesman/DO/payout numbers are stored
-- here. Payout is computed LIVE every time a campaign is opened, from this row
-- plus existing sales/collection data (processing/commission_campaigns.py) --
-- see commission_tracking_design.md §B.1/3/4 for the full spec and the
-- 2026-09-19 correction from an earlier "store the computed result" framing.
--
-- Salesman -> payout-group membership is deliberately NOT here -- it's
-- derived LIVE from prmst.xdisease (salesman's own current area) matched
-- against cacus.xstate (that area's zone classification), confirmed
-- 2026-09-20 -- see processing/commission_campaigns.py::derive_spid_group_map.
-- (An earlier version of this used a hand-maintained
-- data/commission_payout_groups.json roster instead -- superseded, since
-- group membership now comes straight from the ERP with nothing to keep in
-- sync by hand.) payout_groups below only carries each group's own deadline
-- FOR THIS campaign.
--
-- Schema revised 2026-09-20 twice, both before this table was ever deployed
-- to the live server (confirmed empty before each revision -- no ALTER/
-- migration needed, straight DROP+recreate both times):
--   1) product_codes/rate_per_unit/cap_per_salesman_per_product collapsed
--      into one product_rates JSONB (rate AND cap varying per product), and
--      recipient_type added -- the commission can go to the salesman OR the
--      customer on the DO; the collection/deadline eligibility logic is
--      identical either way, only who gets paid differs.
--   2) SAME-DAY CORRECTION: the per-product cap in (1) was a
--      misunderstanding -- the user clarified the cap is meant to be ONE
--      value for the whole campaign (a ceiling on a recipient's TOTAL
--      payout summed across every product in the campaign), not a separate
--      cap per product. product_rates collapsed further to {itemcode: rate}
--      (plain float, no nested object) and a new top-level `cap` column
--      holds the single campaign-wide ceiling.
--
-- Run this once against the live app database:
--   psql -h <host> -U <user> -d <dbname> -f db/sql_scripts/create_commission_campaigns_table.sql
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS commission_campaigns (
    id                SERIAL PRIMARY KEY,
    campaign_name     VARCHAR(200) NOT NULL,
    campaign_type     VARCHAR(50),          -- free-text label: 'Product-Specific' /
                                             -- 'Stock Clearance' / 'Slow-Moving' -- same
                                             -- mechanism regardless, just a UI category
    recipient_type    VARCHAR(20) NOT NULL DEFAULT 'salesman',  -- 'salesman' or 'customer'
    product_rates     JSONB NOT NULL,       -- {"ITEMCODE1": 5.0, "ITEMCODE2": 3.0}
                                             -- rate = per-unit INCENTIVE/discount amount
                                             -- (e.g. 5 or 3 BDT), NOT the product's sales
                                             -- price -- varies per product, cap does not
                                             -- (see `cap` column below)
    cap               NUMERIC(18,2),        -- ONE ceiling for the whole campaign -- caps a
                                             -- single recipient's TOTAL payout, summed across
                                             -- every product in this campaign. NULL = no cap.
    window_start      DATE NOT NULL,        -- sales window: which DOs are eligible
    window_end        DATE NOT NULL,
    payout_groups     JSONB NOT NULL,       -- {"Dhaka retail": "2026-10-15", "District": "2026-10-31"}
                                             -- group name -> this campaign's own collection
                                             -- deadline for that group -- always keyed off
                                             -- the DO's own salesman's group, regardless of
                                             -- recipient_type (deadline is a logistics/
                                             -- territory concept, not a recipient concept)
    uptick_baseline_months INTEGER NOT NULL DEFAULT 3,
    created_by        VARCHAR(50),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes             TEXT
);

-- No "IF NOT EXISTS" here -- this live server predates Postgres 9.5, where
-- CREATE INDEX IF NOT EXISTS isn't valid syntax (same constraint documented
-- for marketing_leads' setup scripts in CLAUDE.md). Safe to run once; errors
-- (not a silent no-op) if run twice.
CREATE INDEX commission_campaigns_created_idx ON commission_campaigns (created_at);
