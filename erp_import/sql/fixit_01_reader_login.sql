-- FIXIT SERVER -- a read-only login for the online-shop importer (fixit_pull.py, which runs on the HMBR server).
-- Run once on the FIXIT server's database (the one that holds final_items_view) as a superuser. Written for PostgreSQL 9.4.
-- Change CHANGE_ME, then put the same password in the HMBR server's erp_import.ini under [fixit].
-- It can read final_items_view and nothing else.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'um_fixit_reader') THEN
        CREATE ROLE um_fixit_reader LOGIN PASSWORD 'CHANGE_ME';
    END IF;
END
$$;

-- put the real database name here (the one \c is connected to); the line below uses the current one
DO $$
BEGIN
    EXECUTE 'GRANT CONNECT ON DATABASE ' || quote_ident(current_database()) || ' TO um_fixit_reader';
END
$$;
GRANT USAGE ON SCHEMA public TO um_fixit_reader;
GRANT SELECT ON public.final_items_view TO um_fixit_reader;

-- CHECK (expect 1 row, and a count): 
SELECT has_table_privilege('um_fixit_reader', 'public.final_items_view', 'SELECT') AS can_read_view;
