-- A hand-curated customer list for WhatsFly Bulk Messaging -- an
-- alternative to the filter-builder audience (Marketing > Bulk Messaging)
-- for when the target audience is better picked by hand than by any
-- filter: browse the full customer directory for a ZID (no area/salesman/
-- score/etc. filters -- deliberately just the raw list) and pick
-- individual customers onto a saved list.
--
-- One curated list per ZID (not multiple named lists -- kept simple per
-- explicit ask; revisit if more than one saved list is ever needed).
--
-- Run once, against the existing whatsapp_webhooks database (same one
-- the campaign tables live in), e.g.:
--   psql -h <host> -U <admin> -d whatsapp_webhooks -f add_curated_list_table.sql
-- Then run add_curated_list_grants.sql to give the existing
-- streamlit_campaign_writer role access to this new table.
--
-- Pre-9.5 Postgres server, same as every other migration script in this
-- repo -- no `IF NOT EXISTS`, no `ON CONFLICT` (adding a customer is an
-- UPDATE-then-INSERT-if-0-rows from the Python side, matching every
-- other upsert in this app). Safe to run once; errors (not a silent
-- no-op) if run twice.

CREATE TABLE curated_list_contacts (
    id              BIGSERIAL PRIMARY KEY,
    zid             TEXT NOT NULL,
    cusid           TEXT NOT NULL,
    cusname         TEXT,
    cusmobile       TEXT,       -- cacus.xmobile, as-is from cacus_directory
    whatsapp        TEXT,       -- cacus.xtaxnum, as-is from cacus_directory -- resolving which of the two to actually message is left to whatever consumes this list later, same as everywhere else this app handles the two fields
    area            TEXT,
    added_by        TEXT NOT NULL,
    added_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (zid, cusid)
);
CREATE INDEX idx_curated_list_contacts_zid ON curated_list_contacts(zid);
