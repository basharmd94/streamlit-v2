# core/queries.py
# All SQL query builders. Every function returns (query_string, params_tuple).
# No f-string interpolation of user-supplied values.

from typing import Iterable, Tuple, Dict, Any


def _build_in_clause(id_iterable: Iterable[int]) -> Tuple[str, Tuple[int, ...]]:
    seen, ordered_ids = set(), []
    for zid in id_iterable:
        if zid not in seen:
            ordered_ids.append(zid)
            seen.add(zid)
    placeholders = ", ".join(["%s"] * len(ordered_ids))
    return placeholders, tuple(ordered_ids)


def get_sales_data(filters=None):
    filters = filters or {}
    query = """SELECT
                sales.zid,
                sales.ordernumber as voucher,
                sales.date,
                EXTRACT(YEAR FROM sales.date) AS year,
                EXTRACT(MONTH FROM sales.date) AS month,
                sales.sp_id as spid,
                employee.spname,
                sales.cusid,
                cacus.cusname,
                cacus.cuscity as area,
                sales.itemcode,
                caitem.itemname,
                caitem.itemgroup2 as itemgroup,
                sales.quantity,
                sales.altsales,
                sales.proddiscount,
                sales.totalsales,
                sales.cost
            FROM
                sales
            JOIN
                employee ON sales.sp_id = employee.spid AND sales.zid = employee.zid
            JOIN
                cacus ON sales.cusid = cacus.cusid AND sales.zid = cacus.zid
            JOIN
                caitem ON sales.itemcode = caitem.itemcode AND sales.zid = caitem.zid
            WHERE
                sales.zid = %s"""

    params = [filters["zid"][0]]

    if filters.get("year"):
        placeholders = ",".join(["%s"] * len(filters["year"]))
        query += f" AND EXTRACT(YEAR FROM sales.date) IN ({placeholders})"
        params.extend(filters["year"])

    if filters.get("month"):
        placeholders = ",".join(["%s"] * len(filters["month"]))
        query += f" AND EXTRACT(MONTH FROM sales.date) IN ({placeholders})"
        params.extend(filters["month"])

    if filters.get("spname"):
        raw_vals = [str(v) for v in filters["spname"]]
        spids, spnames = [], []
        for v in raw_vals:
            if " - " in v:
                code, name = v.split(" - ", 1)
                code, name = code.strip(), name.strip()
                if code:
                    spids.append(code)
                if name:
                    spnames.append(name)
            else:
                name = v.strip()
                if name:
                    spnames.append(name)

        conditions = []
        if spids:
            placeholders = ",".join(["%s"] * len(spids))
            conditions.append(f"sales.sp_id IN ({placeholders})")
            params.extend(spids)
        if spnames:
            placeholders = ",".join(["%s"] * len(spnames))
            conditions.append(f"employee.spname IN ({placeholders})")
            params.extend(spnames)
        if conditions:
            query += " AND (" + " OR ".join(conditions) + ")"

    if filters.get("cusname"):
        raw_vals = [str(v) for v in filters["cusname"]]
        cusids, cusnames = [], []
        for v in raw_vals:
            if " - " in v:
                code, name = v.split(" - ", 1)
                code, name = code.strip(), name.strip()
                if code:
                    cusids.append(code)
                if name:
                    cusnames.append(name)
            else:
                name = v.strip()
                if name:
                    cusnames.append(name)

        conditions = []
        if cusids:
            placeholders = ",".join(["%s"] * len(cusids))
            conditions.append(f"sales.cusid IN ({placeholders})")
            params.extend(cusids)
        if cusnames:
            placeholders = ",".join(["%s"] * len(cusnames))
            conditions.append(f"cacus.cusname IN ({placeholders})")
            params.extend(cusnames)
        if conditions:
            query += " AND (" + " OR ".join(conditions) + ")"

    if filters.get("itemname"):
        raw_vals = [str(v) for v in filters["itemname"]]
        itemcodes, itemnames = [], []
        for v in raw_vals:
            if " - " in v:
                code, name = v.split(" - ", 1)
                code, name = code.strip(), name.strip()
                if code:
                    itemcodes.append(code)
                if name:
                    itemnames.append(name)
            else:
                name = v.strip()
                if name:
                    itemnames.append(name)

        conditions = []
        if itemcodes:
            placeholders = ",".join(["%s"] * len(itemcodes))
            conditions.append(f"sales.itemcode IN ({placeholders})")
            params.extend(itemcodes)
        if itemnames:
            placeholders = ",".join(["%s"] * len(itemnames))
            conditions.append(f"caitem.itemname IN ({placeholders})")
            params.extend(itemnames)
        if conditions:
            query += " AND (" + " OR ".join(conditions) + ")"

    if filters.get("area"):
        placeholders = ",".join(["%s"] * len(filters["area"]))
        query += f" AND cacus.cuscity IN ({placeholders})"
        params.extend(filters["area"])

    if filters.get("itemgroup"):
        placeholders = ",".join(["%s"] * len(filters["itemgroup"]))
        query += f" AND caitem.itemgroup2 IN ({placeholders})"
        params.extend(filters["itemgroup"])

    return query, tuple(params)


def get_return_data(filters=None):
    filters = filters or {}
    query = """SELECT
                return.zid,
                return.revoucher,
                return.date,
                EXTRACT(YEAR FROM return.date) AS year,
                EXTRACT(MONTH FROM return.date) AS month,
                return.sp_id as spid,
                employee.spname,
                return.cusid,
                cacus.cusname,
                cacus.cuscity as area,
                return.itemcode,
                return.reason,
                caitem.itemname,
                caitem.itemgroup2 as itemgroup,
                return.returnqty,
                return.treturnamt,
                return.returncost
            FROM
                return
            JOIN
                employee ON return.sp_id = employee.spid AND return.zid = employee.zid
            JOIN
                cacus ON return.cusid = cacus.cusid AND return.zid = cacus.zid
            JOIN
                caitem ON return.itemcode = caitem.itemcode AND return.zid = caitem.zid
            WHERE
                return.zid = %s"""

    params = [filters["zid"][0]]

    if filters.get("year"):
        placeholders = ",".join(["%s"] * len(filters["year"]))
        query += f" AND EXTRACT(YEAR FROM return.date) IN ({placeholders})"
        params.extend(filters["year"])

    if filters.get("month"):
        placeholders = ",".join(["%s"] * len(filters["month"]))
        query += f" AND EXTRACT(MONTH FROM return.date) IN ({placeholders})"
        params.extend(filters["month"])

    if filters.get("spname"):
        raw_vals = [str(v) for v in filters["spname"]]
        spids, spnames = [], []
        for v in raw_vals:
            if " - " in v:
                code, name = v.split(" - ", 1)
                code, name = code.strip(), name.strip()
                if code:
                    spids.append(code)
                if name:
                    spnames.append(name)
            else:
                name = v.strip()
                if name:
                    spnames.append(name)

        conditions = []
        if spids:
            placeholders = ",".join(["%s"] * len(spids))
            conditions.append(f"return.sp_id IN ({placeholders})")
            params.extend(spids)
        if spnames:
            placeholders = ",".join(["%s"] * len(spnames))
            conditions.append(f"employee.spname IN ({placeholders})")
            params.extend(spnames)
        if conditions:
            query += " AND (" + " OR ".join(conditions) + ")"

    if filters.get("cusname"):
        raw_vals = [str(v) for v in filters["cusname"]]
        cusids, cusnames = [], []
        for v in raw_vals:
            if " - " in v:
                code, name = v.split(" - ", 1)
                code, name = code.strip(), name.strip()
                if code:
                    cusids.append(code)
                if name:
                    cusnames.append(name)
            else:
                name = v.strip()
                if name:
                    cusnames.append(name)

        conditions = []
        if cusids:
            placeholders = ",".join(["%s"] * len(cusids))
            conditions.append(f"return.cusid IN ({placeholders})")
            params.extend(cusids)
        if cusnames:
            placeholders = ",".join(["%s"] * len(cusnames))
            conditions.append(f"cacus.cusname IN ({placeholders})")
            params.extend(cusnames)
        if conditions:
            query += " AND (" + " OR ".join(conditions) + ")"

    if filters.get("itemname"):
        raw_vals = [str(v) for v in filters["itemname"]]
        itemcodes, itemnames = [], []
        for v in raw_vals:
            if " - " in v:
                code, name = v.split(" - ", 1)
                code, name = code.strip(), name.strip()
                if code:
                    itemcodes.append(code)
                if name:
                    itemnames.append(name)
            else:
                name = v.strip()
                if name:
                    itemnames.append(name)

        conditions = []
        if itemcodes:
            placeholders = ",".join(["%s"] * len(itemcodes))
            conditions.append(f"return.itemcode IN ({placeholders})")
            params.extend(itemcodes)
        if itemnames:
            placeholders = ",".join(["%s"] * len(itemnames))
            conditions.append(f"caitem.itemname IN ({placeholders})")
            params.extend(itemnames)
        if conditions:
            query += " AND (" + " OR ".join(conditions) + ")"

    if filters.get("area"):
        placeholders = ",".join(["%s"] * len(filters["area"]))
        query += f" AND cacus.cuscity IN ({placeholders})"
        params.extend(filters["area"])

    if filters.get("itemgroup"):
        placeholders = ",".join(["%s"] * len(filters["itemgroup"]))
        query += f" AND caitem.itemgroup2 IN ({placeholders})"
        params.extend(filters["itemgroup"])

    return query, tuple(params)


def get_collection_data(filters=None):
    """
    Returns one row per collection voucher with:
      - spid/spname from gldetail.sp_id (the newly added column)
      - fallback: if gldetail.sp_id is NULL/blank, uses the most recent
        salesman from the sales table for that customer (last-INOP rule)
    """
    filters = filters or {}
    zid = filters["zid"][0]

    # ── CTE 1: aggregate collection vouchers, grab direct sp_id ──────────────
    rct_conditions = [
        "gd.zid = %s",
        "gm.usage = 'AR'",
        "(gd.voucher LIKE 'RCT-%%' OR gd.voucher LIKE 'CRCT%%'"
        " OR gd.voucher LIKE 'STJV%%' OR gd.voucher LIKE 'BRCT%%'"
        " OR gd.voucher LIKE 'JV--%%')",
    ]
    rct_params: list = [zid]

    if filters.get("year"):
        ph = ",".join(["%s"] * len(filters["year"]))
        rct_conditions.append(f"gh.year IN ({ph})")
        rct_params.extend(filters["year"])

    if filters.get("month"):
        ph = ",".join(["%s"] * len(filters["month"]))
        rct_conditions.append(f"gh.month IN ({ph})")
        rct_params.extend(filters["month"])

    rct_where = " AND ".join(rct_conditions)

    # ── CTE 2: last salesman per customer from sales table ────────────────────
    last_sp_params: list = [zid]

    # ── Outer WHERE (cusname / area / spname filters) ─────────────────────────
    outer_conditions: list = []
    outer_params: list = []

    if filters.get("cusname"):
        raw_vals = [str(v) for v in filters["cusname"]]
        cusids, cusnames = [], []
        for v in raw_vals:
            if " - " in v:
                code, name = v.split(" - ", 1)
                if code.strip(): cusids.append(code.strip())
                if name.strip(): cusnames.append(name.strip())
            else:
                if v.strip(): cusnames.append(v.strip())
        conds = []
        if cusids:
            ph = ",".join(["%s"] * len(cusids))
            conds.append(f"r.cusid IN ({ph})")
            outer_params.extend(cusids)
        if cusnames:
            ph = ",".join(["%s"] * len(cusnames))
            conds.append(f"cc.cusname IN ({ph})")
            outer_params.extend(cusnames)
        if conds:
            outer_conditions.append("(" + " OR ".join(conds) + ")")

    if filters.get("area"):
        ph = ",".join(["%s"] * len(filters["area"]))
        outer_conditions.append(f"cc.cuscity IN ({ph})")
        outer_params.extend(filters["area"])

    if filters.get("spname"):
        raw_vals = [str(v) for v in filters["spname"]]
        spids, spnames = [], []
        for v in raw_vals:
            if " - " in v:
                code, name = v.split(" - ", 1)
                if code.strip(): spids.append(code.strip())
                if name.strip(): spnames.append(name.strip())
            else:
                if v.strip(): spnames.append(v.strip())
        conds = []
        if spids:
            ph = ",".join(["%s"] * len(spids))
            conds.append(f"COALESCE(r.direct_sp_id, ls.sp_id) IN ({ph})")
            outer_params.extend(spids)
        if spnames:
            ph = ",".join(["%s"] * len(spnames))
            conds.append(f"e.spname IN ({ph})")
            outer_params.extend(spnames)
        if conds:
            outer_conditions.append("(" + " OR ".join(conds) + ")")

    outer_where = ("\n    WHERE " + " AND ".join(outer_conditions)) if outer_conditions else ""

    query = f"""
    WITH rct_raw AS (
        SELECT
            gd.zid,
            gd.voucher,
            gd.ac_sub::text          AS cusid,
            gh.date,
            gh.year,
            gh.month,
            ABS(SUM(gd.value))       AS value,
            MAX(NULLIF(TRIM(gd.sp_id::text), '')) AS direct_sp_id
        FROM gldetail gd
        JOIN glheader gh ON gd.voucher  = gh.voucher  AND gd.zid = gh.zid
        JOIN glmst    gm ON gd.ac_code  = gm.ac_code  AND gd.zid = gm.zid
        WHERE {rct_where}
        GROUP BY gd.zid, gd.voucher, gd.ac_sub, gh.date, gh.year, gh.month
    ),
    last_sale_sp AS (
        -- Most recent salesman per customer from the sales table (last-INOP rule)
        SELECT DISTINCT ON (cusid::text)
            cusid::text   AS cusid,
            sp_id::text   AS sp_id
        FROM sales
        WHERE zid = %s
          AND sp_id IS NOT NULL
          AND TRIM(sp_id::text) != ''
        ORDER BY cusid::text, date DESC
    )
    SELECT
        r.zid,
        r.voucher                                       AS glvoucher,
        r.cusid,
        cc.cusname,
        cc.cuscity                                      AS area,
        r.date,
        r.year,
        r.month,
        r.value,
        COALESCE(r.direct_sp_id, ls.sp_id)             AS spid,
        e.spname
    FROM rct_raw r
    JOIN  cacus    cc ON r.cusid  = cc.cusid::text AND r.zid = cc.zid
    LEFT JOIN last_sale_sp ls ON r.cusid = ls.cusid
    LEFT JOIN employee     e
           ON COALESCE(r.direct_sp_id, ls.sp_id) = e.spid::text
          AND e.zid = r.zid{outer_where}
    """

    all_params = tuple(rct_params + last_sp_params + outer_params)
    return query, all_params


def get_ar_data(filters=None):
    filters = filters or {}
    zid = filters["zid"][0]

    query = """
    SELECT
      glmst.zid,
      gldetail.project       AS project,
      gldetail.voucher       AS voucher,
      gldetail.ac_sub        AS cusid,
      cacus.cusname          AS cusname,
      cacus.cuscity          AS area,
      glheader.date          AS date,
      glheader.year          AS year,
      glheader.month         AS month,
      SUM(gldetail.value)    AS value
    FROM glmst
    JOIN gldetail
      ON glmst.ac_code = gldetail.ac_code
     AND glmst.zid     = gldetail.zid
    JOIN glheader
      ON gldetail.voucher = glheader.voucher
     AND glheader.zid      = glmst.zid
    JOIN cacus
      ON gldetail.ac_sub = cacus.cusid
     AND cacus.zid       = glmst.zid
    WHERE glmst.zid        = %s
      AND glmst.usage      = 'AR'
    """
    params = [zid]

    if filters.get("cusname"):
        raw_vals = [str(v) for v in filters["cusname"]]
        cusids, cusnames = [], []
        for v in raw_vals:
            if " - " in v:
                code, name = v.split(" - ", 1)
                code, name = code.strip(), name.strip()
                if code:
                    cusids.append(code)
                if name:
                    cusnames.append(name)
            else:
                name = v.strip()
                if name:
                    cusnames.append(name)

        conditions = []
        if cusids:
            placeholders = ",".join(["%s"] * len(cusids))
            conditions.append(f"gldetail.ac_sub IN ({placeholders})")
            params.extend(cusids)
        if cusnames:
            placeholders = ",".join(["%s"] * len(cusnames))
            conditions.append(f"cacus.cusname IN ({placeholders})")
            params.extend(cusnames)
        if conditions:
            query += "\n  AND (" + " OR ".join(conditions) + ")"

    if filters.get("area"):
        placeholders = ",".join(["%s"] * len(filters["area"]))
        query += f"\n  AND cacus.cuscity IN ({placeholders})"
        params.extend(filters["area"])

    if filters.get("year"):
        placeholders = ",".join(["%s"] * len(filters["year"]))
        query += f"\n  AND glheader.year IN ({placeholders})"
        params.extend([str(y) for y in filters["year"]])

    if filters.get("month"):
        placeholders = ",".join(["%s"] * len(filters["month"]))
        query += f"\n  AND glheader.month IN ({placeholders})"
        params.extend([str(m) for m in filters["month"]])

    query += """
    GROUP BY
      glmst.zid,
      gldetail.project,
      gldetail.voucher,
      gldetail.ac_sub,
      cacus.cusname,
      cacus.cuscity,
      glheader.date,
      glheader.year,
      glheader.month
    ORDER BY date, voucher;
    """
    return query, tuple(params)


def get_purchase_data(filters=None):
    query = """SELECT
                purchase.zid,
                purchase.combinedate,
                purchase.povoucher,
                purchase.grnvoucher,
                CASE
                    WHEN caitem.packcode IS NOT NULL AND caitem.packcode <> '' AND caitem.packcode != 'NO' THEN caitem.packcode
                    ELSE purchase.itemcode
                END AS itemcode,
                caitem.itemname,
                purchase.shipmentname,
                purchase.quantity,
                purchase.cost,
                purchase.status
            FROM
                purchase
            JOIN
                caitem ON purchase.itemcode =  caitem.itemcode AND purchase.zid = caitem.zid
            WHERE
                purchase.zid IN (%s,%s)
                AND purchase.povoucher LIKE 'IP--%%'
                AND purchase.status IN ('5-Received','1-Open')
            """
    return query


def get_product_inventory_data(filters=None):
    query = """SELECT
                stock.zid,
                stock.year,
                stock.month,
                CASE
                    WHEN caitem.packcode IS NOT NULL
                    AND caitem.packcode <> ''
                    AND caitem.packcode != 'NO'
                    AND LEFT(caitem.packcode,2) != 'KH' THEN caitem.packcode
                    ELSE stock.itemcode
                END AS itemcode,
                caitem.itemname,
                caitem.itemgroup,
                stock.warehouse,
                stock.stockqty,
                stock.stockvalue
            FROM
                stock
            JOIN
                caitem ON stock.itemcode = caitem.itemcode AND stock.zid = caitem.zid
            WHERE
                stock.zid = (%s)"""
    return query


def get_inventory_value_data(filters=None):
    return """SELECT
                stock_value.zid,
                stock_value.year,
                stock_value.month,
                stock_value.warehouse,
                stock_value.stockvalue
            FROM stock_value WHERE zid = (%s)"""


def get_stock_flow_data(filters=None):
    # Apply the same packcode CASE logic used in get_product_inventory_data
    # so that itemcodes here always match those in inv_df (from stock).
    return """SELECT
                sf.zid,
                sf.year,
                sf.month,
                sf.warehouse,
                CASE
                    WHEN ca.packcode IS NOT NULL
                     AND ca.packcode <> ''
                     AND ca.packcode != 'NO'
                     AND LEFT(ca.packcode, 2) != 'KH' THEN ca.packcode
                    ELSE sf.itemcode
                END AS itemcode,
                sf.qty_in,
                sf.qty_out,
                sf.net_qty,
                sf.val_in,
                sf.val_out,
                sf.net_val
              FROM stock_flow sf
              LEFT JOIN caitem ca
                ON sf.itemcode = ca.itemcode AND sf.zid = ca.zid
              WHERE sf.zid = (%s)"""


def get_stock_movement_data(filters=None) -> Tuple[str, tuple]:
    filters = filters or {}
    zid = filters["zid"][0]

    sql = """
        SELECT
            stock.zid,
            stock.year,
            stock.month,
            stock.date::date AS date,
            stock.docnum,
            stock.project,
            CASE
                WHEN caitem.packcode IS NOT NULL
                 AND caitem.packcode <> ''
                 AND caitem.packcode != 'NO'
                 AND LEFT(caitem.packcode,2) != 'KH' THEN caitem.packcode
                ELSE stock.itemcode
            END AS itemcode,
            caitem.itemname,
            caitem.itemgroup,
            stock.warehouse,
            stock.stockqty,
            stock.stockvalue
        FROM stock
        JOIN caitem ON stock.itemcode = caitem.itemcode AND stock.zid = caitem.zid
        WHERE stock.zid = %s
    """
    return sql, (zid,)


def get_payment_data():
    return """SELECT
                glmst.zid as zid,
                gldetail.voucher as glvoucher,
                gldetail.ac_sub as account,
                glmst.ac_name as description,
                glheader.date,
                glheader.year,
                glheader.month,
                SUM(gldetail.value) as value
            FROM
                glmst
            JOIN
                gldetail ON glmst.ac_code = gldetail.ac_code AND glmst.zid = gldetail.zid
            JOIN
                glheader ON gldetail.voucher = glheader.voucher AND glmst.zid = glheader.zid
            WHERE
                glmst.zid = 100001
                AND gldetail.zid = 100001
                AND glheader.zid = 100001
                AND (
                    gldetail.voucher LIKE 'CPAY%%'
                    OR gldetail.voucher LIKE 'BPAY%%'
                    OR gldetail.voucher LIKE 'STJV%%'
                )
            GROUP BY
                glmst.zid,
                gldetail.voucher,
                gldetail.ac_sub,
                glmst.ac_name,
                glheader.date,
                glheader.year,
                glheader.month
            HAVING
                SUM(gldetail.value) > 0
            ORDER BY
                SUM(gldetail.value) DESC"""


# ─────────────────────────────────────────────────────────────────────────────
# Simple table fetchers
# ─────────────────────────────────────────────────────────────────────────────

def get_cacus_simple(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    sql = """
        SELECT
            zid,
            cusid::text AS cusid,
            cusname,
            COALESCE(cuscity,'') AS cuscity
        FROM cacus
        WHERE zid = %s
    """
    return sql, (zid,)


def get_gldetail_simple(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    project = filters.get("project")
    sql = """
        SELECT
            zid,
            voucher,
            ac_code,
            ac_sub::text AS ac_sub,
            value::numeric AS value
        FROM gldetail
        WHERE zid = %s
        AND project = %s
    """
    return sql, (zid, project)


def get_glheader_simple(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    sql = """
        SELECT
            zid,
            voucher,
            date::date AS date,
            year::int  AS year,
            month::int AS month
        FROM glheader
        WHERE zid = %s
    """
    return sql, (zid,)


def get_glmst_simple(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    sql = """
        SELECT
            zid,
            ac_code,
            ac_name,
            ac_type,
            COALESCE(ac_lv1, '') AS ac_lv1,
            COALESCE(ac_lv2, '') AS ac_lv2
        FROM glmst
        WHERE zid = %s
    """
    return sql, (zid,)


def get_casup_simple(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    sql = """
        SELECT
            zid,
            supid::text  AS supid,
            supname
        FROM casup
        WHERE zid = %s
    """
    return sql, (zid,)


# ─────────────────────────────────────────────────────────────────────────────
# GL / Financial query builders (moved from db/db_utils.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_gl_details(zid, project=None, year=None, smonth=None, emonth=None,
                   is_bs=False, is_project=False) -> Tuple[str, tuple]:
    """
    Builds the GL detail query. Returns (sql, params).
    Replaces db_utils.get_gl_details() — same logic, standardized return format.
    """
    where = [
        "glmst.zid = %s",
        "gldetail.zid = %s",
        "glheader.zid = %s"
    ]
    params = [zid, zid, zid]

    select = ["glmst.zid", "glmst.ac_code", "glheader.year",
              "glheader.month", "SUM(gldetail.value) as sum"]
    group_by = ["glmst.zid", "glmst.ac_code", "glheader.year",
                "glheader.month", "glmst.usage"]

    if is_project:
        select.extend(["glmst.ac_type", "glmst.ac_lv1", "glmst.ac_lv2", "glmst.usage"])
        group_by.extend(["glmst.ac_type", "glmst.ac_lv1", "glmst.ac_lv2"])
        where.append("gldetail.project = %s")
        params.append(project)

    if is_bs:
        where.append("(glmst.ac_type = 'Asset' OR glmst.ac_type = 'Liability')")
    else:
        where.append("(glmst.ac_type = 'Income' OR glmst.ac_type = 'Expenditure')")

    if year:
        where.append("glheader.year = %s")
        params.append(year)

    if is_bs:
        if emonth:
            where.append("glheader.month <= %s")
            params.append(emonth)
    else:
        if smonth:
            where.append("glheader.month >= %s")
            params.append(smonth)
        if emonth:
            where.append("glheader.month <= %s")
            params.append(emonth)

    sql = f"""
        SELECT {", ".join(select)}
        FROM glmst
        JOIN gldetail ON glmst.ac_code = gldetail.ac_code
        JOIN glheader ON gldetail.voucher = glheader.voucher
        WHERE {" AND ".join(where)}
        GROUP BY {", ".join(group_by)}
        ORDER BY glheader.month ASC, glmst.ac_type
    """
    return sql, tuple(params)


def get_gl_master(zid) -> Tuple[str, tuple]:
    """Returns (sql, params) for the GL account master."""
    sql = """
        SELECT ac_code, ac_name, ac_type, ac_lv1, ac_lv2, ac_lv3, ac_lv4
        FROM glmst WHERE glmst.zid = %s
    """
    return sql, (zid,)


def get_caitem_data(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    """Returns item master: itemcode, itemname, itemgroup, packcode."""
    zid = filters["zid"][0]
    sql = """
        SELECT
            zid,
            itemcode,
            itemname,
            COALESCE(itemgroup, '') AS itemgroup,
            COALESCE(packcode,  '') AS packcode
        FROM caitem
        WHERE zid = %s
    """
    return sql, (zid,)
