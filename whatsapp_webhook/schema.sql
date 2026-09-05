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
