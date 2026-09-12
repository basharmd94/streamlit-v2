-- Bulk Messaging campaign tables — non-destructive addition to an existing
-- whatsapp_webhooks database, e.g.:
--   psql -h <host> -U <user> -d whatsapp_webhooks -f add_bulk_campaign_tables.sql
-- (see schema.sql for a fresh-setup version, which now includes these same
-- tables). Syntax and every constraint below verified against a throwaway
-- local Postgres 14 database loaded from schema.sql first, then dropped —
-- not run against the real whatsapp_webhooks database from this environment,
-- which has no connection to it. Pre-9.5 Postgres server, same as
-- every other migration script in this repo — no `IF NOT EXISTS` on
-- CREATE TABLE/INDEX, no `ON CONFLICT`. Safe to run once; errors (not a
-- silent no-op) if run twice.
--
-- STATUS: already run for real against the live database — these three
-- tables exist there now. This file is kept accurate as a from-scratch
-- reference (and so schema.sql, which includes the same DDL, stays right
-- for a fresh setup) but re-running it against the live database will now
-- just error on "table already exists" rather than do anything. The
-- cooldown_days column below was added to the CREATE TABLE here AFTER the
-- live tables were created — see add_campaign_cooldown_column.sql for the
-- separate ALTER that actually needs to run against the live database.
--
-- Dedup / crash-safety note: campaign_recipients rows start 'pending' and
-- the send worker only ever processes rows still 'pending' for a given
-- campaign. This means "just re-run the campaign after a crash" (per the
-- build plan) needs NO separate resume logic — a crash mid-run simply
-- leaves some rows 'sent'/'failed' and the rest 'pending'; re-running the
-- worker against the same campaign_id naturally skips the ones already
-- attempted and picks up only what's left. The UNIQUE (campaign_id, cusid)
-- constraint is the second half of the guard — Phase 2 can't insert the
-- same customer into the same campaign twice even if its own insert step
-- were accidentally run twice.

CREATE TABLE campaigns (
    id                  BIGSERIAL PRIMARY KEY,
    zid                 TEXT NOT NULL,
    template_name       TEXT NOT NULL,
    template_id         TEXT NOT NULL,        -- WhatsFly's short internal `id` field, the one the send call actually needs — NOT the longer `template_id` the template-list response returns for the same template (see CLAUDE.md's "naming trap" note)
    header_image_url    TEXT,
    variable_mapping    JSONB NOT NULL DEFAULT '{}',  -- {"CUSNAME": "customer attribute: cusname", ...} — a readable record of how each placeholder was SOURCED, for audit only; the actual values sent live per-recipient below
    filters_used        JSONB NOT NULL DEFAULT '[]',  -- snapshot of the exact filter list + window_months active when this campaign's audience was built
    cooldown_days       INTEGER NOT NULL DEFAULT 30,  -- per-template resend cooldown (days) in effect when this campaign's audience was built — see add_campaign_cooldown_column.sql for the note on scope/why this is persisted per-campaign, not a code constant
    status              TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed')),
    total_recipients    INTEGER NOT NULL,
    created_by          TEXT NOT NULL,        -- st.session_state.username
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ
);
CREATE INDEX idx_campaigns_status ON campaigns(status);
CREATE INDEX idx_campaigns_zid ON campaigns(zid);

CREATE TABLE campaign_recipients (
    id              BIGSERIAL PRIMARY KEY,
    campaign_id     BIGINT NOT NULL REFERENCES campaigns(id),
    zid             TEXT NOT NULL,            -- denormalized from campaigns — cusid is only unique within a zid, so this is needed for any lookup that doesn't join back through campaigns
    cusid           TEXT NOT NULL,
    cusname         TEXT,                     -- denormalized snapshot of this app's own customer name — NOT from whatsapp_webhooks' own contacts table, which has no cusid concept at all (see CLAUDE.md)
    phone_number    TEXT NOT NULL,            -- WhatsApp-format number actually used (880...) — the one the send call was made to
    variables_sent  JSONB,                    -- {"CUSNAME": "actual value sent", ...} — the real substituted values (contrast with campaigns.variable_mapping, which only describes the SOURCE, not the value)
    status          TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed')),
    failure_type    TEXT CHECK (failure_type IN ('permanent', 'transient')),  -- NULL unless status = 'failed'
    error_detail    TEXT,                     -- raw error message/code from WhatsFly — NULL unless status = 'failed'
    wamid           TEXT,                     -- links to messages.wamid once sent — NULL until a successful send; this is how delivered/read status (arriving later via the existing webhook path) gets joined back onto a campaign
    sent_at         TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (campaign_id, cusid)
);
CREATE INDEX idx_campaign_recipients_campaign_id ON campaign_recipients(campaign_id);
CREATE INDEX idx_campaign_recipients_status ON campaign_recipients(status);
CREATE UNIQUE INDEX idx_campaign_recipients_wamid ON campaign_recipients(wamid) WHERE wamid IS NOT NULL;

CREATE TABLE contact_opt_outs (
    id              BIGSERIAL PRIMARY KEY,
    zid             TEXT NOT NULL,
    cusid           TEXT NOT NULL,
    opted_out_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    opted_out_by    TEXT NOT NULL,            -- st.session_state.username
    reason          TEXT,
    UNIQUE (zid, cusid)
);
