-- ─────────────────────────────────────────────────────────────────────────────
-- Grants "Commissions" page access to the admin role only -- without this
-- row, even an admin's sidebar menu won't show it (app.py::navigation
-- filters the menu through auth.check_page_access, which reads
-- page_permissions). Same convention as create_view_usage_log_table.sql's
-- own grant for "Usage Stats" -- a new, compensation-sensitive page starts
-- admin-only; widen to other roles later if/when that's actually wanted.
--
-- Plain INSERT, no ON CONFLICT -- pre-9.5 Postgres server, same as every
-- other one-time setup script in this repo. Safe to run once; errors (not
-- a silent no-op) if run twice.
--
-- Run this once against the live app database:
--   psql -h <host> -U <user> -d <dbname> -f db/sql_scripts/grant_commissions_page_permission.sql
-- ─────────────────────────────────────────────────────────────────────────────

INSERT INTO page_permissions (role, page_name) VALUES ('admin', 'Commissions');
