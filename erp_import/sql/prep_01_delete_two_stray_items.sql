-- PREP 1 of 2 -- delete the two stray test items 04-00001 and 04-00002 from company 100007.
-- Run on the HMBR server, database `da` (psql or pgAdmin query tool). Written for PostgreSQL 9.4. Safe to run twice.
--
-- Why first: the importer numbers new Fixit items 04-00001, 04-00002, ... (highest existing + 1), so these two codes must be
-- free before the first real run, otherwise real Fixit items would be numbered after two meaningless ones.

-- 1) LOOK FIRST. Expect 2 rows (0 rows = already deleted). They should have no SKU (xalias) and no stock/sales.
SELECT xitem, xalias, xdesc, xitemnew, xgitem, xpricecat, xstdprice
FROM caitem
WHERE zid = 100007 AND xitem IN ('04-00001', '04-00002');

-- 2) THE DELETE. One block: if anything at all still points at these codes it STOPS and deletes nothing.
DO $$
DECLARE
    refs integer;
BEGIN
    SELECT (SELECT count(*) FROM imtrn  WHERE zid = 100007 AND xitem IN ('04-00001', '04-00002'))
         + (SELECT count(*) FROM opmob  WHERE zid = 100007 AND xitem IN ('04-00001', '04-00002'))
         + (SELECT count(*) FROM opodt  WHERE zid = 100007 AND (xitem IN ('04-00001', '04-00002') OR xcode IN ('04-00001', '04-00002')))
         + (SELECT count(*) FROM opddt  WHERE zid = 100007 AND (xitem IN ('04-00001', '04-00002') OR xcode IN ('04-00001', '04-00002')))
         + (SELECT count(*) FROM opcdt  WHERE zid = 100007 AND (xitem IN ('04-00001', '04-00002') OR xcode IN ('04-00001', '04-00002')))
         + (SELECT count(*) FROM poodt  WHERE zid = 100007 AND (xitem IN ('04-00001', '04-00002') OR xcode IN ('04-00001', '04-00002')))
         + (SELECT count(*) FROM pogdt  WHERE zid = 100007 AND (xitem IN ('04-00001', '04-00002') OR xcode IN ('04-00001', '04-00002')))
      INTO refs;

    IF refs > 0 THEN
        RAISE EXCEPTION 'STOPPED: % rows in stock / order / purchase tables still use 04-00001 or 04-00002. Nothing was deleted.', refs;
    END IF;

    -- the price-category code of these two items (only if no other 100007 item shares it)
    DELETE FROM xcodes
    WHERE zid = 100007 AND xtype = 'Price Category'
      AND xcode IN (SELECT xpricecat FROM caitem
                    WHERE zid = 100007 AND xitem IN ('04-00001', '04-00002') AND coalesce(xpricecat, '') <> '')
      AND xcode NOT IN (SELECT xpricecat FROM caitem
                        WHERE zid = 100007 AND xitem NOT IN ('04-00001', '04-00002') AND xpricecat IS NOT NULL);

    -- only a code with NO SKU: a real Fixit item the importer created later always carries its Fixit code in xalias
    DELETE FROM caitem
    WHERE zid = 100007 AND xitem IN ('04-00001', '04-00002') AND coalesce(xalias, '') = '';
END
$$;

-- 3) CHECK. Expect 0 rows.
SELECT xitem, xalias, xdesc FROM caitem WHERE zid = 100007 AND xitem IN ('04-00001', '04-00002');
