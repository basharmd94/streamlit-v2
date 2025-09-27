# db/sql_scripts.py

from typing import Iterable, Tuple, Dict, Any, List

def _build_in_clause(id_iterable: Iterable[int]) -> Tuple[str, Tuple[int, ...]]:
    # 1) Remove duplicates while preserving order
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
                
    # Append filters to the WHERE clause
    if filters.get("year"):
        query += f" AND EXTRACT(YEAR FROM sales.date) IN ({','.join(map(str, filters['year']))})"
    if filters.get("month"):
        query += f" AND EXTRACT(MONTH FROM sales.date) IN ({','.join(map(str, filters['month']))})"
    if filters.get("spname"):
        names = "','".join(filters["spname"])
        query += f" AND employee.spname IN ('{names}')"
    if filters.get("cusname"):
        names = "','".join(filters["cusname"])
        query += f" AND cacus.cusname IN ('{names}')"
    if filters.get("itemname"):
        names = "','".join(filters["itemname"])
        query += f" AND caitem.itemname IN ('{names}')"
    if filters.get("area"):
        names = "','".join(filters["area"])
        query += f" AND cacus.cuscity IN ('{names}')"
    if filters.get("itemgroup"):
        names = "','".join(filters["itemgroup"])
        query += f" AND caitem.itemgroup2 IN ('{names}')"
    return query

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

    # Append filters to the WHERE clause
    if filters.get("year"):
        query += f" AND EXTRACT(YEAR FROM return.date) IN ({','.join(map(str, filters['year']))})"
    if filters.get("month"):
        query += f" AND EXTRACT(MONTH FROM return.date) IN ({','.join(map(str, filters['month']))})"
    if filters.get("spname"):
        names = "','".join(filters["spname"])
        query += f" AND employee.spname IN ('{names}')"
    if filters.get("cusname"):
        names = "','".join(filters["cusname"])
        query += f" AND cacus.cusname IN ('{names}')"
    if filters.get("itemname"):
        names = "','".join(filters["itemname"])
        query += f" AND caitem.itemname IN ('{names}')"
    if filters.get("area"):
        names = "','".join(filters["area"])
        query += f" AND cacus.cuscity IN ('{names}')"
    if filters.get("itemgroup"):
        names = "','".join(filters["itemgroup"])
        query += f" AND caitem.itemgroup2 IN ('{names}')"
    return query

def get_collection_data(filters=None):
    filters = filters or {}

    query = """
        SELECT
            glmst.zid,
            gldetail.voucher AS glvoucher,
            gldetail.ac_sub AS cusid,
            cacus.cusname as cusname,
            glheader.date,
            glheader.year,
            glheader.month,
            ABS(SUM(gldetail.value)) AS value
        FROM glmst
        JOIN gldetail ON glmst.ac_code = gldetail.ac_code AND glmst.zid = gldetail.zid
        JOIN glheader ON gldetail.voucher = glheader.voucher AND glmst.zid = glheader.zid
        JOIN cacus ON gldetail.ac_sub = cacus.cusid AND gldetail.zid = cacus.zid
        WHERE glmst.zid = %s
        AND (
            gldetail.voucher LIKE 'JV--%%'
            OR gldetail.voucher LIKE 'RCT-%%'
            OR gldetail.voucher LIKE 'CRCT%%'
            OR gldetail.voucher LIKE 'STJV%%'
            OR gldetail.voucher LIKE 'BRCT%%'
        )
        AND glmst.usage IN ('AR')
    """

    # Optional filter clauses
    if filters.get("year"):
        query += f" AND glheader.year IN ({','.join(map(str, filters['year']))})"

    if filters.get("month"):
        query += f" AND glheader.month IN ({','.join(map(str, filters['month']))})"

    if filters.get("cusname"):
        names = "','".join(filters["cusname"])
        query += f" AND cacus.cusname IN ('{names}')"

    if filters.get("area"):
        areas = "','".join(filters["area"])
        query += f" AND cacus.cuscity IN ('{areas}')"

    query += """
        GROUP BY
            glmst.zid,
            gldetail.voucher,
            gldetail.ac_sub,
            cacus.cusname,
            glmst.usage,
            glheader.date,
            glheader.year,
            glheader.month
    """
    return query

def get_ar_data(filters=None):
    """
    Returns a (query, params) tuple that fetches AR data
    with dynamic cusname, area, year, and month filters,
    but without a running_balance column.
    """
    filters = filters or {}
    zid = filters["zid"]

    # Base SELECT without running_balance
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

    # Optional filters
    if filters.get("cusname"):
        names = filters["cusname"]
        placeholders = ", ".join(["%s"] * len(names))
        query += f"\n  AND cacus.cusname IN ({placeholders})"
        params.extend(names)

    if filters.get("area"):
        areas = filters["area"]
        placeholders = ", ".join(["%s"] * len(areas))
        query += f"\n  AND cacus.cuscity IN ({placeholders})"
        params.extend(areas)

    if filters.get("year"):
        years = [str(y) for y in filters["year"]]
        placeholders = ", ".join(["%s"] * len(years))
        query += f"\n  AND glheader.year IN ({placeholders})"
        params.extend(years)

    if filters.get("month"):
        months = [str(m) for m in filters["month"]]
        placeholders = ", ".join(["%s"] * len(months))
        query += f"\n  AND glheader.month IN ({placeholders})"
        params.extend(months)

    # Grouping and ordering
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
                CASE
                    WHEN caitem.packcode IS NOT NULL 
                    AND caitem.packcode <> '' 
                    AND caitem.packcode != 'NO'
                    AND LEFT(caitem.packcode,2) != 'KH' THEN caitem.packcode
                    ELSE stock.itemcode
                END AS itemcode,
                caitem.itemname,
                caitem.itemgroup,
                stock.stockqty,
                stock.stockvalue
            FROM 
                stock
            JOIN
                caitem ON stock.itemcode = caitem.itemcode AND stock.zid = caitem.zid
            WHERE 
                stock.zid IN  (%s,%s)"""

    return query

def get_inventory_value_data():
    return """SELECT 
                stock_value.zid,
                stock_value.year,
                stock_value.month,
                stock_value.warehouse,
                stock_value.stockvalue
            FROM stock_value WHERE zid = (%s)"""

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
# Simple table fetchers (one table each; let pandas do the work)
# ─────────────────────────────────────────────────────────────────────────────
from typing import Dict, Any, Tuple

def get_cacus_simple(filters: Dict[str, Any]) -> Tuple[str, Tuple[Any, ...]]:
    """
    Return all customers for a zid.
    """
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

def get_gldetail_simple(filters: Dict[str, Any]) -> Tuple[str, Tuple[Any, ...]]:
    """
    Return raw GL detail (no joins, no filters) for a zid.
    """
    zid = filters["zid"][0]
    sql = """
        SELECT
            zid,
            voucher,
            ac_code,
            ac_sub::text AS ac_sub,
            value::numeric AS value
        FROM gldetail
        WHERE zid = %s
    """
    return sql, (zid,)

def get_glheader_simple(filters: Dict[str, Any]) -> Tuple[str, Tuple[Any, ...]]:
    """
    Return GL headers for a zid (date + year/month).
    """
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

def get_glmst_simple(filters: Dict[str, Any]) -> Tuple[str, Tuple[Any, ...]]:
    """
    Return account master for a zid.
    """
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

def get_casup_simple(filters: Dict[str, Any]) -> Tuple[str, Tuple[Any, ...]]:
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