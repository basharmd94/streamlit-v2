-- WhatsApp webhook receiver schema — a separate PostgreSQL database, isolated
-- from the main app's `da` database (see
-- WhatsApp_Integration_docs/whatsapp-webhook-build.md, "Database" section).
--
-- Run once against a fresh database, e.g.:
--   createdb whatsapp_webhooks
--   psql -h localhost -U postgres -d whatsapp_webhooks -f schema.sql

CREATE TABLE webhook_events (
    id BIGSERIAL PRIMARY KEY,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    signature_valid BOOLEAN NOT NULL,
    raw_payload JSONB NOT NULL,
    processed_at TIMESTAMPTZ,
    processing_status TEXT NOT NULL DEFAULT 'pending' -- pending | processed | failed
);

CREATE TABLE contacts (
    id BIGSERIAL PRIMARY KEY,
    phone_number TEXT UNIQUE NOT NULL,
    wa_id TEXT,
    name TEXT,
    customer_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE messages (
    id BIGSERIAL PRIMARY KEY,
    wamid TEXT UNIQUE NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    phone_number_id TEXT NOT NULL,
    contact_phone TEXT NOT NULL REFERENCES contacts(phone_number),
    message_type TEXT NOT NULL,
    template_name TEXT,
    content JSONB,
    current_status TEXT,
    message_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_messages_contact_phone ON messages(contact_phone);
CREATE INDEX idx_messages_current_status ON messages(current_status);

CREATE TABLE message_status_events (
    id BIGSERIAL PRIMARY KEY,
    wamid TEXT NOT NULL REFERENCES messages(wamid),
    status TEXT NOT NULL,
    error_code TEXT,
    error_title TEXT,
    event_timestamp TIMESTAMPTZ NOT NULL,
    webhook_event_id BIGINT REFERENCES webhook_events(id),
    UNIQUE (wamid, status)
);
CREATE INDEX idx_status_events_wamid ON message_status_events(wamid);

CREATE TABLE templates (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    language TEXT,
    status TEXT,
    last_checked_at TIMESTAMPTZ,
    UNIQUE (name, language)
);

CREATE TABLE account_alerts (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Bulk Messaging campaign tables — see add_bulk_campaign_tables.sql for the
-- non-destructive version of this same DDL, for adding onto an existing
-- deployed database. Kept identical in both places; if the schema is
-- revised, revise it in both files.

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
    completed_at        TIMESTAMPTZ,
    actual_cost         NUMERIC(12, 2)  -- entered/updated later from Campaign History (Phase 4) — see add_campaign_cost_column.sql for why this can never be known at send time
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

-- Per-template variable mapping (Phase 6) -- see
-- add_template_variable_mappings_table.sql for the non-destructive
-- version of this same DDL, for adding onto an existing deployed
-- database. Kept identical in both places.
CREATE TABLE template_variable_mappings (
    id              BIGSERIAL PRIMARY KEY,
    template_id     TEXT NOT NULL,
    template_name   TEXT NOT NULL,
    variable_name   TEXT NOT NULL,
    source_type     TEXT NOT NULL CHECK (source_type IN ('customer_attribute', 'flat_value')),
    source_key      TEXT,
    created_by      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (template_id, variable_name)
);
CREATE INDEX idx_template_variable_mappings_template_id ON template_variable_mappings(template_id);
