-- ─────────────────────────────────────────────────────────────────────────────
-- View usage logging — app-owned table, powers the admin-only "📊 Usage
-- Stats" page (views/usage_stats.py). Same convention as crm_call_log /
-- marketing_leads / page_permissions: created once, directly on the app's
-- own Postgres DB, written to by the Streamlit app itself (core/db.py) --
-- no separate role/grants needed, this app's single DB connection already
-- has full access to its own database.
--
-- One row per genuine page/mode TRANSITION (not one row per Streamlit
-- rerun) -- see processing/usage_log.py::log_view for the dedup logic.
-- "Time spent on a view" is derived after the fact as the gap to the NEXT
-- row in the same session_id, since Streamlit has no real page-exit event
-- to log an explicit duration at write time.
--
-- Run this once against the live app database:
--   psql -h <host> -U <user> -d <dbname> -f db/sql_scripts/create_view_usage_log_table.sql
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS view_usage_log (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(36)   NOT NULL,  -- one UUID per browser session (processing/usage_log.py::ensure_session_id)
    username    VARCHAR(50)   NOT NULL,
    user_role   VARCHAR(30),
    zid         VARCHAR(10),             -- active business at the time this view was opened
    page        VARCHAR(100)  NOT NULL,  -- top-level menu item, e.g. "Marketing Analysis"
    view_path   VARCHAR(255),            -- the page's own primary radio selection, if it has one
    entered_at  TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX view_usage_log_entered_idx  ON view_usage_log (entered_at);
CREATE INDEX view_usage_log_session_idx  ON view_usage_log (session_id, entered_at);
CREATE INDEX view_usage_log_username_idx ON view_usage_log (username, entered_at);
CREATE INDEX view_usage_log_page_idx     ON view_usage_log (page, entered_at);


-- Grants "Usage Stats" page access to the admin role only -- without this
-- row, even an admin's menu won't show it (app.py::navigation filters the
-- menu through auth.check_page_access, which reads this table). Plain
-- INSERT, no ON CONFLICT -- pre-9.5 Postgres server, same as every other
-- one-time setup script in this repo. Safe to run once; errors (not a
-- silent no-op) if run twice.
INSERT INTO page_permissions (role, page_name) VALUES ('admin', 'Usage Stats');
