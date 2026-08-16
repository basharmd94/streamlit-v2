-- ─────────────────────────────────────────────────────────────────────────────
-- Marketing Leads CRM — app-owned tables (NOT synced from the ERP via db_sync).
-- Same convention as crm_call_log / users / page_permissions: created once,
-- directly on the app's own Postgres DB, and written to by the Streamlit app.
--
-- Run this once against the live app database:
--   psql -h <host> -U <user> -d <dbname> -f db/sql_scripts/create_marketing_leads_tables.sql
-- ─────────────────────────────────────────────────────────────────────────────

-- ── marketing_leads ─────────────────────────────────────────────────────────
-- One row per lead. Sourced from Facebook Lead Ads CSV exports (or any future
-- lead source with the same shape). zid is NOT in the export — it is set at
-- upload time from the uploader's active ZID (navbar selection).
--
-- fb_lead_id is the lead-gen platform's own id (CSV column "id"). It is the
-- join key back to cacus.xurl once a lead is converted to a real customer in
-- the ERP: staff manually paste this id into the new customer's xurl field,
-- and the app looks that up live (no cus_code column is stored here — it can
-- change/appear at any time on the ERP side, so it's always read fresh).
--
-- Any CSV column not in the fixed schema below (campaign forms carry different
-- custom questions — e.g. a Bengali "what type of institution" question that
-- won't be the same across forms) is preserved as {question: answer} in
-- extra_fields rather than requiring a schema change per new lead form.
CREATE TABLE IF NOT EXISTS marketing_leads (
    id                  SERIAL PRIMARY KEY,
    zid                 VARCHAR(10)   NOT NULL,
    fb_lead_id          VARCHAR(50)   NOT NULL,
    created_time        TIMESTAMPTZ,
    ad_id               VARCHAR(50),
    ad_name             VARCHAR(255),
    adset_id            VARCHAR(50),
    adset_name          VARCHAR(255),
    campaign_id         VARCHAR(50),
    campaign_name       VARCHAR(255),
    form_id             VARCHAR(50),
    form_name           VARCHAR(255),
    is_organic          BOOLEAN,
    platform            VARCHAR(20),
    full_name           VARCHAR(255),
    work_phone_number   VARCHAR(50),
    company_name        VARCHAR(255),
    street_address      TEXT,
    job_title           VARCHAR(255),
    inbox_url           TEXT,
    lead_status         VARCHAR(50),   -- the lead-gen platform's own status (e.g. "complete")
    extra_fields        JSONB,         -- any non-standard CSV columns, as {question: answer}
    lead_stage          VARCHAR(30)   NOT NULL DEFAULT 'New',  -- internal CRM pipeline stage
    uploaded_by         VARCHAR(50),
    uploaded_at         TIMESTAMPTZ   NOT NULL DEFAULT now(),
    UNIQUE (zid, fb_lead_id)
);

CREATE INDEX IF NOT EXISTS marketing_leads_zid_created_idx ON marketing_leads (zid, created_time DESC);
CREATE INDEX IF NOT EXISTS marketing_leads_fb_lead_id_idx  ON marketing_leads (fb_lead_id);
CREATE INDEX IF NOT EXISTS marketing_leads_zid_stage_idx   ON marketing_leads (zid, lead_stage);


-- ── marketing_lead_call_log ─────────────────────────────────────────────────
-- Call history against a lead — separate from crm_call_log (which is keyed
-- on cusid, a real ERP customer code). Mirrors crm_call_log's shape and adds
-- next_visit_date, which the customer call log does not have.
CREATE TABLE IF NOT EXISTS marketing_lead_call_log (
    id               SERIAL PRIMARY KEY,
    zid              VARCHAR(10)  NOT NULL,
    lead_id          INTEGER      NOT NULL REFERENCES marketing_leads(id) ON DELETE CASCADE,
    called_at        TIMESTAMP    NOT NULL DEFAULT now(),
    called_by        VARCHAR(50),
    outcome          VARCHAR(50),
    next_visit_date  DATE,
    notes            TEXT
);

CREATE INDEX IF NOT EXISTS marketing_lead_call_log_lead_idx ON marketing_lead_call_log (lead_id, called_at DESC);
CREATE INDEX IF NOT EXISTS marketing_lead_call_log_zid_idx  ON marketing_lead_call_log (zid, called_at DESC);
CREATE INDEX IF NOT EXISTS marketing_lead_call_log_nvd_idx  ON marketing_lead_call_log (next_visit_date);
