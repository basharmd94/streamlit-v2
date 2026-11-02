-- OPTIONAL -- a database login that can do only what the online-shop importer needs (instead of reusing the ERP/Streamlit login).
-- Run once on the HMBR server as a superuser, connected to database `da`. Written for PostgreSQL 9.4.
-- Change CHANGE_ME below, then put the same password in erp_import.ini ([database] user = um_importer).
--
-- What it may do:
--   read    the ERP tables the importer looks at (customer, supplier, employee, stock, orders, price tiers, ...)
--   write   caitem, xcodes (items and their price category), opmob (the online orders), poord/poodt (the open purchase orders)
--   own     its um_* work tables: it creates them itself on the first run (that is why it gets CREATE on schema public)
-- It cannot touch accounting (gl*), customers, suppliers, delivery or receipt documents.
-- To remove it later:  DROP OWNED BY um_importer;  DROP ROLE um_importer;   (this also drops the um_* tables it created)

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'um_importer') THEN
        CREATE ROLE um_importer LOGIN PASSWORD 'CHANGE_ME';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE da TO um_importer;
GRANT USAGE, CREATE ON SCHEMA public TO um_importer;

-- read only
GRANT SELECT ON cacus, casup, prmst, imtrn, opspprc, opord, opodt, opdor, opddt, opcrn, pogrn, pogdt, final_items_view TO um_importer;

-- read and write
GRANT SELECT, INSERT, UPDATE, DELETE ON caitem, xcodes, opmob, poord, poodt TO um_importer;
