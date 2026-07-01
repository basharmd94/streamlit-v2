# core/queries.py
# All SQL query builders. Every function returns (query_string, params_tuple).
# No f-string interpolation of user-supplied values.
#
# Queries run directly against hmbr (ERP) tables.
# Column/table mapping from hmbr → stream2 aliases (so downstream code is unchanged):
#   glmst:    xacc→ac_code, xdesc→ac_name, xacctype→ac_type, xhrc1-4→ac_lv1-4, xaccusage→usage
#   gldetail: xacc→ac_code, xsub→ac_sub, xsp→sp_id, xproj→project, xvoucher→voucher, xrow→row, xprime→value
#   glheader: xvoucher→voucher, xdate→date, xyear→year, xper→month
#   cacus:    xcus→cusid, xshort→cusname, xadd2→cusadd, xcity→cuscity, xmobile→cusmobile
#   caitem:   xitem→itemcode, xdesc→itemname, xgitem→itemgroup, xabc→itemgroup2, xdrawing→packcode
#   casup:    xsup→supid, xorg→supname, xcity→supadd
#   prmst:    table renamed from employee; xemp→spid, xname→spname, xdept→department, xdesig→designation
#   zbusiness: table renamed from business; zorg→org, xshort→name
#   purchase:  derived — poord JOIN poodt LEFT JOIN pogrn
#   sales:     derived — opdor LEFT JOIN opddt LEFT JOIN imtrn
#   return:    derived — opcdt/opcrn/imtrn UNION ALL imtemptdt/imtemptrn/imtrn
#   stock:     derived — imtrn GROUP BY per transaction (xqty*xsign, xval*xsign)
#   stock_value: derived — imtrn GROUP BY year/month/warehouse
#   stock_flow:  derived — imtrn GROUP BY year/month/item/warehouse with in/out CASE

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
    query = """
    WITH s AS (
        SELECT
            opddt.zid,
            opdor.xdornum  AS ordernumber,
            opdor.xdate    AS date,
            opdor.xsp      AS sp_id,
            opdor.xcus     AS cusid,
            opddt.xitem    AS itemcode,
            opddt.xqty     AS quantity,
            opddt.xdtwotax AS altsales,
            opddt.xdtdisc  AS proddiscount,
            opddt.xlineamt AS totalsales,
            imtrn.xval     AS cost
        FROM opdor
        LEFT JOIN opddt ON opdor.xdornum = opddt.xdornum AND opdor.zid = opddt.zid
        LEFT JOIN imtrn ON opdor.xdornum  = imtrn.xdocnum
                       AND opddt.xdornum  = imtrn.xdocnum
                       AND opddt.xitem    = imtrn.xitem
                       AND opddt.zid      = imtrn.zid
                       AND opdor.zid      = imtrn.zid
                       AND opddt.xrow     = imtrn.xdocrow
        WHERE opdor.zid = %s
    )
    SELECT
        s.zid,
        s.ordernumber AS voucher,
        s.date,
        EXTRACT(YEAR  FROM s.date) AS year,
        EXTRACT(MONTH FROM s.date) AS month,
        s.sp_id  AS spid,
        p.xname  AS spname,
        s.cusid,
        c.xshort   AS cusname,
        c.xmobile  AS cusmobile,
        c.xtaxnum  AS whatsapp,
        c.xcity    AS area,
        s.itemcode,
        ci.xdesc AS itemname,
        ci.xabc  AS itemgroup,
        s.quantity,
        s.altsales,
        s.proddiscount,
        s.totalsales,
        s.cost
    FROM s
    JOIN prmst  p  ON s.sp_id    = p.xemp  AND s.zid = p.zid
    JOIN cacus  c  ON s.cusid    = c.xcus   AND s.zid = c.zid
    JOIN caitem ci ON s.itemcode = ci.xitem AND s.zid = ci.zid
    WHERE 1=1"""

    params = [filters["zid"][0]]

    if filters.get("year"):
        placeholders = ",".join(["%s"] * len(filters["year"]))
        query += f" AND EXTRACT(YEAR FROM s.date) IN ({placeholders})"
        params.extend(filters["year"])

    if filters.get("month"):
        placeholders = ",".join(["%s"] * len(filters["month"]))
        query += f" AND EXTRACT(MONTH FROM s.date) IN ({placeholders})"
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
            conditions.append(f"s.sp_id IN ({placeholders})")
            params.extend(spids)
        if spnames:
            placeholders = ",".join(["%s"] * len(spnames))
            conditions.append(f"p.xname IN ({placeholders})")
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
            conditions.append(f"s.cusid IN ({placeholders})")
            params.extend(cusids)
        if cusnames:
            placeholders = ",".join(["%s"] * len(cusnames))
            conditions.append(f"c.xshort IN ({placeholders})")
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
            conditions.append(f"s.itemcode IN ({placeholders})")
            params.extend(itemcodes)
        if itemnames:
            placeholders = ",".join(["%s"] * len(itemnames))
            conditions.append(f"ci.xdesc IN ({placeholders})")
            params.extend(itemnames)
        if conditions:
            query += " AND (" + " OR ".join(conditions) + ")"

    if filters.get("area"):
        placeholders = ",".join(["%s"] * len(filters["area"]))
        query += f" AND c.xcity IN ({placeholders})"
        params.extend(filters["area"])

    if filters.get("itemgroup"):
        placeholders = ",".join(["%s"] * len(filters["itemgroup"]))
        query += f" AND ci.xabc IN ({placeholders})"
        params.extend(filters["itemgroup"])

    return query, tuple(params)


def get_return_data(filters=None):
    filters = filters or {}
    # Two zid params: one per UNION branch in the CTE
    query = """
    WITH ret AS (
        SELECT
            opcdt.zid,
            opcrn.xcrnnum  AS revoucher,
            opcrn.xdate    AS date,
            opcrn.xcus     AS cusid,
            opcrn.xemp     AS sp_id,
            opcdt.xitem    AS itemcode,
            opcrn.xreason  AS reason,
            opcdt.xqty     AS returnqty,
            opcdt.xlineamt AS treturnamt,
            imtrn.xval     AS returncost
        FROM opcdt
        LEFT JOIN opcrn ON opcrn.xcrnnum = opcdt.xcrnnum AND opcrn.zid = opcdt.zid
        LEFT JOIN imtrn ON opcrn.xcrnnum  = imtrn.xdocnum
                       AND opcdt.xcrnnum  = imtrn.xdocnum
                       AND opcdt.xitem    = imtrn.xitem
                       AND opcdt.zid      = imtrn.zid
                       AND opcrn.zid      = imtrn.zid
                       AND opcdt.xrow     = imtrn.xdocrow
        WHERE opcdt.zid = %s

        UNION ALL

        SELECT
            imtemptdt.zid,
            imtemptrn.ximtmptrn AS revoucher,
            imtemptrn.xdate     AS date,
            imtemptrn.xcus      AS cusid,
            imtemptrn.xemp      AS sp_id,
            imtemptdt.xitem     AS itemcode,
            imtemptrn.xrem      AS reason,
            imtemptdt.xqtyord   AS returnqty,
            imtemptdt.xlineamt  AS treturnamt,
            imtrn.xval          AS returncost
        FROM imtemptdt
        LEFT JOIN imtemptrn ON imtemptrn.ximtmptrn = imtemptdt.ximtmptrn AND imtemptrn.zid = imtemptdt.zid
        LEFT JOIN imtrn ON imtemptrn.ximtmptrn  = imtrn.xdocnum
                       AND imtemptdt.ximtmptrn  = imtrn.xdocnum
                       AND imtemptdt.xitem       = imtrn.xitem
                       AND imtemptdt.zid         = imtrn.zid
                       AND imtemptrn.zid         = imtrn.zid
                       AND imtemptdt.xtorlno     = imtrn.xdocrow
        WHERE imtemptdt.zid = %s
    )
    SELECT
        ret.zid,
        ret.revoucher,
        ret.date,
        EXTRACT(YEAR  FROM ret.date) AS year,
        EXTRACT(MONTH FROM ret.date) AS month,
        ret.sp_id   AS spid,
        p.xname     AS spname,
        ret.cusid,
        c.xshort    AS cusname,
        c.xmobile   AS cusmobile,
        c.xcity     AS area,
        ret.itemcode,
        ret.reason,
        ci.xdesc    AS itemname,
        ci.xabc     AS itemgroup,
        ret.returnqty,
        ret.treturnamt,
        ret.returncost
    FROM ret
    JOIN prmst  p  ON ret.sp_id    = p.xemp  AND ret.zid = p.zid
    JOIN cacus  c  ON ret.cusid    = c.xcus   AND ret.zid = c.zid
    JOIN caitem ci ON ret.itemcode = ci.xitem AND ret.zid = ci.zid
    WHERE 1=1"""

    zid = filters["zid"][0]
    params = [zid, zid]

    if filters.get("year"):
        placeholders = ",".join(["%s"] * len(filters["year"]))
        query += f" AND EXTRACT(YEAR FROM ret.date) IN ({placeholders})"
        params.extend(filters["year"])

    if filters.get("month"):
        placeholders = ",".join(["%s"] * len(filters["month"]))
        query += f" AND EXTRACT(MONTH FROM ret.date) IN ({placeholders})"
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
            conditions.append(f"ret.sp_id IN ({placeholders})")
            params.extend(spids)
        if spnames:
            placeholders = ",".join(["%s"] * len(spnames))
            conditions.append(f"p.xname IN ({placeholders})")
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
            conditions.append(f"ret.cusid IN ({placeholders})")
            params.extend(cusids)
        if cusnames:
            placeholders = ",".join(["%s"] * len(cusnames))
            conditions.append(f"c.xshort IN ({placeholders})")
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
            conditions.append(f"ret.itemcode IN ({placeholders})")
            params.extend(itemcodes)
        if itemnames:
            placeholders = ",".join(["%s"] * len(itemnames))
            conditions.append(f"ci.xdesc IN ({placeholders})")
            params.extend(itemnames)
        if conditions:
            query += " AND (" + " OR ".join(conditions) + ")"

    if filters.get("area"):
        placeholders = ",".join(["%s"] * len(filters["area"]))
        query += f" AND c.xcity IN ({placeholders})"
        params.extend(filters["area"])

    if filters.get("itemgroup"):
        placeholders = ",".join(["%s"] * len(filters["itemgroup"]))
        query += f" AND ci.xabc IN ({placeholders})"
        params.extend(filters["itemgroup"])

    return query, tuple(params)


def get_collection_data(filters=None):
    """
    Returns one row per collection voucher with:
      - spid/spname from gldetail.xsp (the direct sp column)
      - fallback: if gldetail.xsp is NULL/blank, uses the most recent
        salesman from opdor for that customer (last-INOP rule)

    SIGN NOTE (read this before touching `value` below):
    AR is credited (xprime negative) when a customer pays, so a real
    collection nets negative. `value` is computed as -SUM(xprime) so a
    real collection becomes positive — NOT ABS(SUM(xprime)). Until
    2026-06-28 this used ABS(), which silently flipped any voucher whose
    net was a positive (debit-to-AR) amount — e.g. a correction/reversal
    booked under JV--/STJV/ADJV — into a *positive* "collection" instead
    of correctly subtracting it. Verified against zid 100001: ABS() had
    been overstating total collections by ~359M (3.71B vs the
    sign-correct 3.35B), almost entirely from JV-- vouchers (40% of which
    net positive). If collection totals ever look off again, check here
    first — and check whether any *new* voucher prefix added to the list
    below also needs this same sign treatment.
    """
    filters = filters or {}
    zid = filters["zid"][0]

    # ── CTE 1: aggregate collection vouchers, grab direct sp_id ──────────────
    rct_conditions = [
        "gd.zid = %s",
        "gm.xaccusage = 'AR'",
        "(gd.xvoucher LIKE 'RCT-%%' OR gd.xvoucher LIKE 'CRCT%%'"
        " OR gd.xvoucher LIKE 'STJV%%' OR gd.xvoucher LIKE 'BRCT%%'"
        " OR gd.xvoucher LIKE 'JV--%%' OR gd.xvoucher LIKE 'ADJV%%')",
    ]
    rct_params: list = [zid]

    if filters.get("year"):
        ph = ",".join(["%s"] * len(filters["year"]))
        rct_conditions.append(f"gh.xyear IN ({ph})")
        rct_params.extend(filters["year"])

    if filters.get("month"):
        ph = ",".join(["%s"] * len(filters["month"]))
        rct_conditions.append(f"gh.xper IN ({ph})")
        rct_params.extend(filters["month"])

    rct_where = " AND ".join(rct_conditions)

    # ── CTE 2: last salesman per customer from opdor ──────────────────────────
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
            conds.append(f"cc.xshort IN ({ph})")
            outer_params.extend(cusnames)
        if conds:
            outer_conditions.append("(" + " OR ".join(conds) + ")")

    if filters.get("area"):
        ph = ",".join(["%s"] * len(filters["area"]))
        outer_conditions.append(f"cc.xcity IN ({ph})")
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
            conds.append(f"e.xname IN ({ph})")
            outer_params.extend(spnames)
        if conds:
            outer_conditions.append("(" + " OR ".join(conds) + ")")

    outer_where = ("\n    WHERE " + " AND ".join(outer_conditions)) if outer_conditions else ""

    query = f"""
    WITH rct_raw AS (
        SELECT
            gd.zid,
            gd.xvoucher                                          AS voucher,
            gd.xsub::text                                        AS cusid,
            gh.xdate                                             AS date,
            gh.xyear                                             AS year,
            gh.xper                                              AS month,
            -SUM(gd.xprime)                                      AS value,  -- sign-preserving; see docstring above — do NOT change to ABS()
            MAX(NULLIF(TRIM(gd.xsp::text), ''))                  AS direct_sp_id
        FROM gldetail gd
        JOIN glheader gh ON gd.xvoucher = gh.xvoucher AND gd.zid = gh.zid
        JOIN glmst    gm ON gd.xacc     = gm.xacc     AND gd.zid = gm.zid
        WHERE {rct_where}
        GROUP BY gd.zid, gd.xvoucher, gd.xsub, gh.xdate, gh.xyear, gh.xper
    ),
    last_sale_sp AS (
        SELECT DISTINCT ON (opdor.xcus::text)
            opdor.xcus::text AS cusid,
            opdor.xsp::text  AS sp_id
        FROM opdor
        WHERE opdor.zid = %s
          AND opdor.xsp IS NOT NULL
          AND TRIM(opdor.xsp::text) != ''
        ORDER BY opdor.xcus::text, opdor.xdate DESC
    )
    SELECT
        r.zid,
        r.voucher                                       AS glvoucher,
        r.cusid,
        cc.xshort                                       AS cusname,
        cc.xcity                                        AS area,
        r.date,
        r.year,
        r.month,
        r.value,
        COALESCE(r.direct_sp_id, ls.sp_id)             AS spid,
        e.xname                                         AS spname
    FROM rct_raw r
    JOIN  cacus    cc ON r.cusid  = cc.xcus::text AND r.zid = cc.zid
    LEFT JOIN last_sale_sp ls ON r.cusid = ls.cusid
    LEFT JOIN prmst        e
           ON COALESCE(r.direct_sp_id, ls.sp_id) = e.xemp::text
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
      gldetail.xproj         AS project,
      gldetail.xvoucher      AS voucher,
      gldetail.xsub          AS cusid,
      cacus.xshort           AS cusname,
      cacus.xcity            AS area,
      glheader.xdate         AS date,
      glheader.xyear         AS year,
      glheader.xper          AS month,
      SUM(gldetail.xprime)   AS value
    FROM glmst
    JOIN gldetail
      ON glmst.xacc    = gldetail.xacc
     AND glmst.zid     = gldetail.zid
    JOIN glheader
      ON gldetail.xvoucher = glheader.xvoucher
     AND glheader.zid      = glmst.zid
    JOIN cacus
      ON gldetail.xsub = cacus.xcus
     AND cacus.zid     = glmst.zid
    WHERE glmst.zid        = %s
      AND glmst.xaccusage  = 'AR'
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
            conditions.append(f"gldetail.xsub IN ({placeholders})")
            params.extend(cusids)
        if cusnames:
            placeholders = ",".join(["%s"] * len(cusnames))
            conditions.append(f"cacus.xshort IN ({placeholders})")
            params.extend(cusnames)
        if conditions:
            query += "\n  AND (" + " OR ".join(conditions) + ")"

    if filters.get("area"):
        placeholders = ",".join(["%s"] * len(filters["area"]))
        query += f"\n  AND cacus.xcity IN ({placeholders})"
        params.extend(filters["area"])

    if filters.get("year"):
        placeholders = ",".join(["%s"] * len(filters["year"]))
        query += f"\n  AND glheader.xyear IN ({placeholders})"
        params.extend([str(y) for y in filters["year"]])

    if filters.get("month"):
        placeholders = ",".join(["%s"] * len(filters["month"]))
        query += f"\n  AND glheader.xper IN ({placeholders})"
        params.extend([str(m) for m in filters["month"]])

    query += """
    GROUP BY
      glmst.zid,
      gldetail.xproj,
      gldetail.xvoucher,
      gldetail.xsub,
      cacus.xshort,
      cacus.xcity,
      glheader.xdate,
      glheader.xyear,
      glheader.xper
    ORDER BY glheader.xdate, gldetail.xvoucher;
    """
    return query, tuple(params)


def get_purchase_data(filters=None):
    # Returns query string only; caller supplies (zid1, zid2) params.
    query = """
    WITH purchase AS (
        SELECT
            poord.zid,
            COALESCE(pogrn.xdate, poord.xdate) AS combinedate,
            poord.xpornum   AS povoucher,
            pogrn.xgrnnum   AS grnvoucher,
            poodt.xitem     AS itemcode,
            poord.xcounterno AS shipmentname,
            poodt.xqtyord   AS quantity,
            poodt.xrate     AS cost,
            poord.xstatuspor AS status
        FROM poord
        JOIN poodt ON poord.xpornum = poodt.xpornum AND poord.zid = poodt.zid
        LEFT JOIN pogrn ON poord.xpornum = pogrn.xpornum AND poord.zid = pogrn.zid
        WHERE poord.zid IN (%s, %s)
          AND poord.xstatuspor IN ('5-Received','1-Open')
    )
    SELECT
        purchase.zid,
        purchase.combinedate,
        purchase.povoucher,
        purchase.grnvoucher,
        CASE
            WHEN ci.xdrawing IS NOT NULL AND ci.xdrawing <> '' AND ci.xdrawing != 'NO' THEN ci.xdrawing
            ELSE purchase.itemcode
        END AS itemcode,
        ci.xdesc AS itemname,
        purchase.shipmentname,
        purchase.quantity,
        purchase.cost,
        purchase.status
    FROM purchase
    JOIN caitem ci ON purchase.itemcode = ci.xitem AND purchase.zid = ci.zid
    WHERE purchase.povoucher LIKE 'IP--%%'
    """
    return query


def get_product_inventory_data(filters=None):
    # Returns query string only; caller supplies (zid,) param.
    query = """
    WITH stock AS (
        SELECT
            imtrn.zid,
            imtrn.xyear  AS year,
            imtrn.xper   AS month,
            imtrn.xitem  AS itemcode,
            imtrn.xwh    AS warehouse,
            SUM(imtrn.xqty * imtrn.xsign) AS stockqty,
            SUM(imtrn.xval * imtrn.xsign) AS stockvalue
        FROM imtrn
        WHERE imtrn.zid = %s
        -- Monthly aggregate only — xdate/xdocnum/xproj/ximtrnnum were previously
        -- in GROUP BY but not in SELECT, so every ximtrnnum (unique PK) formed its
        -- own "group" with SUM() = identity. This returned 2M rows instead of
        -- ~tens-of-thousands of monthly totals and was 2x slower than necessary.
        GROUP BY imtrn.zid, imtrn.xitem, imtrn.xwh, imtrn.xyear, imtrn.xper
    )
    SELECT
        stock.zid,
        stock.year,
        stock.month,
        CASE
            WHEN ci.xdrawing IS NOT NULL
             AND ci.xdrawing <> ''
             AND ci.xdrawing != 'NO'
             AND LEFT(ci.xdrawing, 2) != 'KH' THEN ci.xdrawing
            ELSE stock.itemcode
        END AS itemcode,
        ci.xdesc  AS itemname,
        ci.xgitem AS itemgroup,
        stock.warehouse,
        stock.stockqty,
        stock.stockvalue
    FROM stock
    JOIN caitem ci ON stock.itemcode = ci.xitem AND stock.zid = ci.zid
    """
    return query


def get_inventory_value_data(filters=None):
    # Returns query string only; caller supplies (zid,) param.
    return """
    WITH stock_value AS (
        SELECT
            imtrn.zid,
            imtrn.xyear AS year,
            imtrn.xper  AS month,
            imtrn.xwh   AS warehouse,
            SUM(imtrn.xval * imtrn.xsign) AS stockvalue
        FROM imtrn
        WHERE imtrn.zid = %s
        GROUP BY imtrn.zid, imtrn.xyear, imtrn.xper, imtrn.xwh
    )
    SELECT zid, year, month, warehouse, stockvalue
    FROM stock_value
    """


def get_stock_flow_data(filters=None):
    # Returns query string only; caller supplies (zid,) param.
    return """
    WITH stock_flow AS (
        SELECT
            imtrn.zid,
            imtrn.xyear AS year,
            imtrn.xper  AS month,
            imtrn.xitem AS itemcode,
            imtrn.xwh   AS warehouse,
            SUM(CASE WHEN imtrn.xsign > 0 THEN imtrn.xqty  ELSE 0            END) AS qty_in,
            SUM(CASE WHEN imtrn.xsign < 0 THEN -imtrn.xqty ELSE 0            END) AS qty_out,
            SUM(imtrn.xqty * imtrn.xsign)                                         AS net_qty,
            SUM(CASE WHEN imtrn.xsign > 0 THEN imtrn.xval  ELSE 0            END) AS val_in,
            SUM(CASE WHEN imtrn.xsign < 0 THEN -imtrn.xval ELSE 0            END) AS val_out,
            SUM(imtrn.xval * imtrn.xsign)                                         AS net_val
        FROM imtrn
        WHERE imtrn.zid = %s
        GROUP BY imtrn.zid, imtrn.xitem, imtrn.xwh, imtrn.xyear, imtrn.xper
    )
    SELECT
        sf.zid,
        sf.year,
        sf.month,
        sf.warehouse,
        CASE
            WHEN ca.xdrawing IS NOT NULL
             AND ca.xdrawing <> ''
             AND ca.xdrawing != 'NO'
             AND LEFT(ca.xdrawing, 2) != 'KH' THEN ca.xdrawing
            ELSE sf.itemcode
        END AS itemcode,
        sf.qty_in,
        sf.qty_out,
        sf.net_qty,
        sf.val_in,
        sf.val_out,
        sf.net_val
    FROM stock_flow sf
    LEFT JOIN caitem ca ON sf.itemcode = ca.xitem AND sf.zid = ca.zid
    """


def get_stock_movement_data(filters=None) -> Tuple[str, tuple]:
    filters = filters or {}
    zid = filters["zid"][0]

    sql = """
    WITH stock AS (
        SELECT
            imtrn.zid,
            imtrn.xyear      AS year,
            imtrn.xper       AS month,
            imtrn.xdate      AS date,
            imtrn.xdocnum    AS docnum,
            imtrn.xproj      AS project,
            imtrn.xitem      AS itemcode,
            imtrn.xwh        AS warehouse,
            SUM(imtrn.xqty * imtrn.xsign) AS stockqty,
            SUM(imtrn.xval * imtrn.xsign) AS stockvalue
        FROM imtrn
        WHERE imtrn.zid = %s
        GROUP BY imtrn.zid, imtrn.xitem, imtrn.xwh, imtrn.xyear, imtrn.xper,
                 imtrn.xdate, imtrn.xdocnum, imtrn.xproj, imtrn.ximtrnnum
    )
    SELECT
        stock.zid,
        stock.year,
        stock.month,
        stock.date::date AS date,
        stock.docnum,
        stock.project,
        CASE
            WHEN ci.xdrawing IS NOT NULL
             AND ci.xdrawing <> ''
             AND ci.xdrawing != 'NO'
             AND LEFT(ci.xdrawing, 2) != 'KH' THEN ci.xdrawing
            ELSE stock.itemcode
        END AS itemcode,
        ci.xdesc  AS itemname,
        ci.xgitem AS itemgroup,
        stock.warehouse,
        stock.stockqty,
        stock.stockvalue
    FROM stock
    JOIN caitem ci ON stock.itemcode = ci.xitem AND stock.zid = ci.zid
    """
    return sql, (zid,)


def get_payment_data():
    return """
    SELECT
        glmst.zid                AS zid,
        gldetail.xvoucher        AS glvoucher,
        gldetail.xsub            AS account,
        glmst.xdesc              AS description,
        glheader.xdate           AS date,
        glheader.xyear           AS year,
        glheader.xper            AS month,
        SUM(gldetail.xprime)     AS value
    FROM glmst
    JOIN gldetail ON glmst.xacc     = gldetail.xacc     AND glmst.zid = gldetail.zid
    JOIN glheader ON gldetail.xvoucher = glheader.xvoucher AND glmst.zid = glheader.zid
    WHERE glmst.zid     = 100001
      AND gldetail.zid  = 100001
      AND glheader.zid  = 100001
      AND (
          gldetail.xvoucher LIKE 'CPAY%%'
          OR gldetail.xvoucher LIKE 'BPAY%%'
          OR gldetail.xvoucher LIKE 'STJV%%'
      )
    GROUP BY
        glmst.zid,
        gldetail.xvoucher,
        gldetail.xsub,
        glmst.xdesc,
        glheader.xdate,
        glheader.xyear,
        glheader.xper
    HAVING SUM(gldetail.xprime) > 0
    ORDER BY SUM(gldetail.xprime) DESC"""


# ─────────────────────────────────────────────────────────────────────────────
# Simple table fetchers
# ─────────────────────────────────────────────────────────────────────────────

def get_cacus_simple(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    sql = """
        SELECT
            zid,
            xcus::text           AS cusid,
            xshort               AS cusname,
            COALESCE(xcity, '')  AS cuscity
        FROM cacus
        WHERE zid = %s
    """
    return sql, (zid,)


def get_cacus_directory(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    sql = """
        SELECT
            xcus::text              AS cusid,
            xshort                  AS cusname,
            COALESCE(xmobile,  '')  AS cusmobile,
            COALESCE(xtaxnum,  '')  AS whatsapp,
            COALESCE(xcity,    '')  AS area
        FROM cacus
        WHERE zid = %s
        ORDER BY xshort
    """
    return sql, (zid,)


def get_final_items_view(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    """
    Query the final_items_view database view.
    Returns item_id, item_name, item_group, stock filtered by zid.
    """
    zid = filters["zid"][0]
    sql = """
        SELECT
            item_id,
            item_name,
            item_group,
            stock
        FROM final_items_view
        WHERE zid = %s
        ORDER BY item_name
    """
    return sql, (zid,)


def get_gldetail_simple(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    project = filters.get("project")
    sql = """
        SELECT
            zid,
            xvoucher             AS voucher,
            xacc                 AS ac_code,
            xsub::text           AS ac_sub,
            xprime::numeric      AS value
        FROM gldetail
        WHERE zid = %s
          AND xproj = %s
    """
    return sql, (zid, project)


def get_glheader_simple(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    sql = """
        SELECT
            zid,
            xvoucher         AS voucher,
            xdate::date      AS date,
            xyear::int       AS year,
            xper::int        AS month
        FROM glheader
        WHERE zid = %s
    """
    return sql, (zid,)


def get_glmst_simple(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    sql = """
        SELECT
            zid,
            xacc                        AS ac_code,
            xdesc                       AS ac_name,
            xacctype                    AS ac_type,
            COALESCE(xhrc1, '')         AS ac_lv1,
            COALESCE(xhrc2, '')         AS ac_lv2
        FROM glmst
        WHERE zid = %s
    """
    return sql, (zid,)


def get_casup_simple(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    sql = """
        SELECT
            zid,
            xsup::text  AS supid,
            xorg        AS supname
        FROM casup
        WHERE zid = %s
    """
    return sql, (zid,)


# ─────────────────────────────────────────────────────────────────────────────
# GL / Financial query builders
# ─────────────────────────────────────────────────────────────────────────────

def get_gl_details(zid, project=None, year=None, smonth=None, emonth=None,
                   is_bs=False, is_project=False) -> Tuple[str, tuple]:
    where = [
        "glmst.zid = %s",
        "gldetail.zid = %s",
        "glheader.zid = %s"
    ]
    params = [zid, zid, zid]

    select = ["glmst.zid", "glmst.xacc AS ac_code", "glheader.xyear AS year",
              "glheader.xper AS month", "SUM(gldetail.xprime) as sum"]
    group_by = ["glmst.zid", "glmst.xacc", "glheader.xyear",
                "glheader.xper", "glmst.xaccusage", "glmst.xacctype"]

    if is_project:
        select.extend(["glmst.xacctype AS ac_type", "glmst.xhrc1 AS ac_lv1",
                        "glmst.xhrc2 AS ac_lv2", "glmst.xaccusage AS usage"])
        group_by.extend(["glmst.xhrc1", "glmst.xhrc2"])
        where.append("gldetail.xproj = %s")
        params.append(project)

    if is_bs:
        where.append(
            "(glmst.xacctype IN ('Asset', 'Liability') "
            "OR (glmst.xacctype IS NULL "
            "    AND LEFT(glmst.xacc, 2) IN ('01','02','03','09','10','11','12','13')))"
        )
    else:
        where.append(
            "(glmst.xacctype IN ('Income', 'Expenditure') "
            "OR (glmst.xacctype IS NULL "
            "    AND LEFT(glmst.xacc, 2) IN ('04','05','06','07','08','14','15')))"
        )

    if year:
        where.append("glheader.xyear = %s")
        params.append(year)

    if is_bs:
        if emonth:
            where.append("glheader.xper <= %s")
            params.append(emonth)
    else:
        if smonth:
            where.append("glheader.xper >= %s")
            params.append(smonth)
        if emonth:
            where.append("glheader.xper <= %s")
            params.append(emonth)

    sql = f"""
        SELECT {", ".join(select)}
        FROM glmst
        JOIN gldetail ON glmst.xacc     = gldetail.xacc
        JOIN glheader ON gldetail.xvoucher = glheader.xvoucher
        WHERE {" AND ".join(where)}
        GROUP BY {", ".join(group_by)}
        ORDER BY glheader.xper ASC, glmst.xacctype
    """
    return sql, tuple(params)


def get_gl_mtd(zid, year: int, month: int) -> Tuple[str, tuple]:
    """
    MTD Income Statement: SUM(xprime) per xacc from gldetail for a single
    year/month.  IS accounts only (Income + Expenditure, inferred where NULL).
    Used for the MTD IS dashboard in views/financial.py.
    """
    sql = """
        SELECT
            glmst.xacc           AS ac_code,
            glmst.xdesc          AS ac_name,
            SUM(gldetail.xprime) AS sum
        FROM glmst
        JOIN gldetail ON glmst.xacc        = gldetail.xacc
                     AND glmst.zid         = gldetail.zid
        JOIN glheader ON gldetail.xvoucher = glheader.xvoucher
                     AND gldetail.zid      = glheader.zid
        WHERE glmst.zid      = %s
          AND glheader.xyear = %s
          AND glheader.xper  = %s
          AND (
              glmst.xacctype IN ('Income', 'Expenditure')
              OR (glmst.xacctype IS NULL
                  AND LEFT(glmst.xacc, 2) IN ('04','05','06','07','08','14','15'))
          )
        GROUP BY glmst.xacc, glmst.xdesc
        ORDER BY glmst.xacc
    """
    return sql, (zid, year, month)


def get_gl_master(zid) -> Tuple[str, tuple]:
    sql = """
        SELECT xacc  AS ac_code,
               xdesc AS ac_name,
               xacctype AS ac_type,
               xhrc1 AS ac_lv1,
               xhrc2 AS ac_lv2,
               xhrc3 AS ac_lv3,
               xhrc4 AS ac_lv4
        FROM glmst WHERE glmst.zid = %s
    """
    return sql, (zid,)


def get_vat_breakdown_gl(
    zid,
    project=None,
    year_list=None,
    smonth: int = 1,
    emonth: int = 12,
) -> Tuple[str, tuple]:
    params: list = [zid]
    where = ["gldetail.zid = %s", "gldetail.xacc IN ('01050007', '06290001')"]

    if project:
        where.append("gldetail.xproj = %s")
        params.append(project)

    if year_list:
        placeholders = ", ".join(["%s"] * len(year_list))
        where.append(f"glheader.xyear IN ({placeholders})")
        params.extend(year_list)

    where.append("glheader.xper >= %s")
    params.append(smonth)
    where.append("glheader.xper <= %s")
    params.append(emonth)

    sql = f"""
        SELECT
            gldetail.xacc        AS ac_code,
            glheader.xyear       AS year,
            glheader.xper        AS month,
            gldetail.xvoucher    AS voucher,
            gldetail.xprime      AS value
        FROM gldetail
        JOIN glheader ON gldetail.xvoucher = glheader.xvoucher
                      AND gldetail.zid     = glheader.zid
        WHERE {" AND ".join(where)}
    """
    return sql, tuple(params)


def get_ind_hh_net_sales(
    zid,
    year_list=None,
    smonth: int = 1,
    emonth: int = 12,
) -> Tuple[str, tuple]:
    year_list = list(year_list) if year_list else []

    s_year_clause = ""
    r_year_clause = ""
    if year_list:
        placeholders = ", ".join(["%s"] * len(year_list))
        s_year_clause = f"AND EXTRACT(YEAR FROM s.date)::int IN ({placeholders})"
        r_year_clause = f"AND EXTRACT(YEAR FROM r.date)::int IN ({placeholders})"

    # CTE zid params: sales_raw(1) + return_raw UNION branch1(1) + branch2(1)
    s_params = year_list + [smonth, emonth]
    r_params = year_list + [smonth, emonth]

    sql = f"""
        WITH sales_raw AS (
            SELECT
                opddt.zid,
                opdor.xdate    AS date,
                opddt.xlineamt AS totalsales,
                opddt.xitem    AS itemcode
            FROM opdor
            LEFT JOIN opddt ON opdor.xdornum = opddt.xdornum AND opdor.zid = opddt.zid
            WHERE opdor.zid = %s
        ),
        return_raw AS (
            SELECT
                opcdt.zid,
                opcrn.xdate    AS date,
                opcdt.xlineamt AS treturnamt,
                opcdt.xitem    AS itemcode
            FROM opcdt
            LEFT JOIN opcrn ON opcrn.xcrnnum = opcdt.xcrnnum AND opcrn.zid = opcdt.zid
            WHERE opcdt.zid = %s

            UNION ALL

            SELECT
                imtemptdt.zid,
                imtemptrn.xdate    AS date,
                imtemptdt.xlineamt AS treturnamt,
                imtemptdt.xitem    AS itemcode
            FROM imtemptdt
            LEFT JOIN imtemptrn ON imtemptrn.ximtmptrn = imtemptdt.ximtmptrn
                                AND imtemptrn.zid       = imtemptdt.zid
            WHERE imtemptdt.zid = %s
        ),
        sales_agg AS (
            SELECT EXTRACT(YEAR FROM s.date)::int AS year,
                   SUM(s.totalsales) AS total_sales
            FROM sales_raw s
            JOIN caitem ci ON s.itemcode = ci.xitem AND s.zid = ci.zid
            WHERE 1=1
              {s_year_clause}
              AND EXTRACT(MONTH FROM s.date)::int BETWEEN %s AND %s
              AND ci.xgitem = 'Industrial & Household'
            GROUP BY 1
        ),
        return_agg AS (
            SELECT EXTRACT(YEAR FROM r.date)::int AS year,
                   SUM(r.treturnamt) AS total_returns
            FROM return_raw r
            JOIN caitem ci ON r.itemcode = ci.xitem AND r.zid = ci.zid
            WHERE 1=1
              {r_year_clause}
              AND EXTRACT(MONTH FROM r.date)::int BETWEEN %s AND %s
              AND ci.xgitem = 'Industrial & Household'
            GROUP BY 1
        )
        SELECT
            COALESCE(s.year, r.year)                                          AS year,
            COALESCE(s.total_sales, 0) - COALESCE(r.total_returns, 0)        AS net_sales
        FROM sales_agg s
        FULL OUTER JOIN return_agg r ON s.year = r.year
        ORDER BY 1
    """
    return sql, tuple([zid, zid, zid] + s_params + r_params)


def get_ind_hh_net_cost(
    zid,
    year_list=None,
    smonth: int = 1,
    emonth: int = 12,
) -> Tuple[str, tuple]:
    year_list = list(year_list) if year_list else []

    s_year_clause = ""
    r_year_clause = ""
    if year_list:
        placeholders = ", ".join(["%s"] * len(year_list))
        s_year_clause = f"AND EXTRACT(YEAR FROM s.date)::int IN ({placeholders})"
        r_year_clause = f"AND EXTRACT(YEAR FROM r.date)::int IN ({placeholders})"

    s_params = year_list + [smonth, emonth]
    r_params = year_list + [smonth, emonth]

    sql = f"""
        WITH sales_raw AS (
            SELECT
                opddt.zid,
                opdor.xdate AS date,
                imtrn.xval  AS cost,
                opddt.xitem AS itemcode
            FROM opdor
            LEFT JOIN opddt ON opdor.xdornum = opddt.xdornum AND opdor.zid = opddt.zid
            LEFT JOIN imtrn ON opdor.xdornum  = imtrn.xdocnum
                           AND opddt.xdornum  = imtrn.xdocnum
                           AND opddt.xitem    = imtrn.xitem
                           AND opddt.zid      = imtrn.zid
                           AND opdor.zid      = imtrn.zid
                           AND opddt.xrow     = imtrn.xdocrow
            WHERE opdor.zid = %s
        ),
        return_raw AS (
            SELECT
                opcdt.zid,
                opcrn.xdate AS date,
                imtrn.xval  AS returncost,
                opcdt.xitem AS itemcode
            FROM opcdt
            LEFT JOIN opcrn ON opcrn.xcrnnum = opcdt.xcrnnum AND opcrn.zid = opcdt.zid
            LEFT JOIN imtrn ON opcrn.xcrnnum  = imtrn.xdocnum
                           AND opcdt.xcrnnum  = imtrn.xdocnum
                           AND opcdt.xitem    = imtrn.xitem
                           AND opcdt.zid      = imtrn.zid
                           AND opcrn.zid      = imtrn.zid
                           AND opcdt.xrow     = imtrn.xdocrow
            WHERE opcdt.zid = %s

            UNION ALL

            SELECT
                imtemptdt.zid,
                imtemptrn.xdate AS date,
                imtrn.xval      AS returncost,
                imtemptdt.xitem AS itemcode
            FROM imtemptdt
            LEFT JOIN imtemptrn ON imtemptrn.ximtmptrn = imtemptdt.ximtmptrn
                                AND imtemptrn.zid       = imtemptdt.zid
            LEFT JOIN imtrn ON imtemptrn.ximtmptrn  = imtrn.xdocnum
                           AND imtemptdt.ximtmptrn   = imtrn.xdocnum
                           AND imtemptdt.xitem        = imtrn.xitem
                           AND imtemptdt.zid          = imtrn.zid
                           AND imtemptrn.zid          = imtrn.zid
                           AND imtemptdt.xtorlno      = imtrn.xdocrow
            WHERE imtemptdt.zid = %s
        ),
        sales_cost AS (
            SELECT EXTRACT(YEAR FROM s.date)::int AS year,
                   SUM(s.cost) AS total_cost
            FROM sales_raw s
            JOIN caitem ci ON s.itemcode = ci.xitem AND s.zid = ci.zid
            WHERE 1=1
              {s_year_clause}
              AND EXTRACT(MONTH FROM s.date)::int BETWEEN %s AND %s
              AND ci.xgitem = 'Industrial & Household'
            GROUP BY 1
        ),
        return_cost AS (
            SELECT EXTRACT(YEAR FROM r.date)::int AS year,
                   SUM(r.returncost) AS total_returncost
            FROM return_raw r
            JOIN caitem ci ON r.itemcode = ci.xitem AND r.zid = ci.zid
            WHERE 1=1
              {r_year_clause}
              AND EXTRACT(MONTH FROM r.date)::int BETWEEN %s AND %s
              AND ci.xgitem = 'Industrial & Household'
            GROUP BY 1
        )
        SELECT
            COALESCE(s.year, r.year)                                          AS year,
            COALESCE(s.total_cost, 0) - COALESCE(r.total_returncost, 0)      AS net_cost
        FROM sales_cost s
        FULL OUTER JOIN return_cost r ON s.year = r.year
        ORDER BY 1
    """
    return sql, tuple([zid, zid, zid] + s_params + r_params)


def get_all_businesses() -> Tuple[str, tuple]:
    sql = "SELECT zid, zorg AS org FROM zbusiness WHERE zid <> 1 ORDER BY zid"
    return sql, ()


def get_caitem_data(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    sql = """
        SELECT
            zid,
            xitem                       AS itemcode,
            xdesc                       AS itemname,
            COALESCE(xgitem, '')        AS itemgroup,
            COALESCE(xdrawing,  '')     AS packcode
        FROM caitem
        WHERE zid = %s
    """
    return sql, (zid,)


def get_opmob_pending(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    zid = filters["zid"][0]
    sql = """
        SELECT
            om.xcus        AS cusid,
            c.xshort       AS cusname,
            om.xemp        AS spid,
            om.xitem       AS itemcode,
            ci.xdesc       AS itemname,
            om.xlinetotal  AS linetotal,
            om.xdate       AS order_date
        FROM opmob om
        LEFT JOIN cacus  c  ON om.xcus  = c.xcus  AND om.zid = c.zid
        LEFT JOIN caitem ci ON om.xitem = ci.xitem AND om.zid = ci.zid
        WHERE om.zid = %s
          AND om.xstatusord = 'Order Created'
          AND EXTRACT(YEAR  FROM om.xdate)::int = EXTRACT(YEAR  FROM CURRENT_DATE)::int
          AND EXTRACT(MONTH FROM om.xdate)::int = EXTRACT(MONTH FROM CURRENT_DATE)::int
    """
    return sql, (zid,)


def get_ar_due_ledger(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    """Row-level AR ledger for Salesman Due (Collection Analysis).

    Mirrors jupyter_audits/HM_36_*_Due.py: every AR posting with the salesman
    that should be credited for it (own xsp, falling back to the xsp on the
    related INOP sales voucher), plus customer city/state and salesman name.
    """
    zid = filters["zid"][0]
    project = filters.get("project")
    sql = """
        SELECT
            gd.zid,
            gd.xvoucher,
            gh.xdate,
            gd.xrow,
            EXTRACT(YEAR  FROM gh.xdate)::int AS year,
            EXTRACT(MONTH FROM gh.xdate)::int AS month,
            gd.xsub,
            cacus.xshort AS customer_name,
            cacus.xcity,
            cacus.xstate,
            gd.xprime,
            COALESCE(gd.xsp, sp_source.inop_sp) AS xsp,
            prmst.xname AS salesman_name
        FROM gldetail gd
        JOIN glheader gh
          ON gd.zid = gh.zid
         AND gd.xvoucher = gh.xvoucher
        LEFT JOIN cacus
          ON gd.zid = cacus.zid
         AND gd.xsub = cacus.xcus
        LEFT JOIN (
            SELECT DISTINCT ON (zid, xvoucher) zid, xvoucher, xsp AS inop_sp
            FROM gldetail
            WHERE xaccusage != 'AR'
              AND xvoucher LIKE 'INOP%%'
              AND xsp IS NOT NULL
              AND xsp != ''
            ORDER BY zid, xvoucher, xrow
        ) sp_source
          ON gd.zid = sp_source.zid
         AND gd.xvoucher = sp_source.xvoucher
        LEFT JOIN prmst
          ON gd.zid = prmst.zid
         AND COALESCE(gd.xsp, sp_source.inop_sp) = prmst.xemp
        WHERE gd.zid = %s
          AND gh.zid = %s
          AND gd.xaccusage = 'AR'
          AND gd.xproj = %s
          AND (
                gd.xvoucher LIKE 'INOP%%'
             OR gd.xvoucher LIKE 'RCT%%'
             OR gd.xvoucher LIKE 'BRCT%%'
             OR gd.xvoucher LIKE 'CRCT%%'
             OR gd.xvoucher LIKE 'SRJV%%'
             OR gd.xvoucher LIKE 'SRT%%'
             OR gd.xvoucher LIKE 'JV%%'
             OR gd.xvoucher LIKE 'IMSA%%'
             OR gd.xvoucher LIKE 'STJV%%'
             OR gd.xvoucher LIKE 'CPAY%%'
             OR gd.xvoucher LIKE 'PAY%%'
             OR gd.xvoucher LIKE 'CHQ%%'
             OR gd.xvoucher LIKE 'ADJV%%'
             OR gd.xvoucher LIKE 'TR%%'
             OR gd.xvoucher LIKE 'BPAY%%'
             OR gd.xvoucher LIKE 'BTJV%%'
          )
        ORDER BY gd.xsub, gh.xdate, gd.xrow, gd.xvoucher
    """
    return sql, (zid, zid, project)


def get_cacus_master(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    """Customer master for Salesman Due name/area lookups."""
    zid = filters["zid"][0]
    sql = """
        SELECT
            xcus   AS cusid,
            xshort AS cusname,
            xcity,
            xstate
        FROM cacus
        WHERE zid = %s
    """
    return sql, (zid,)


def get_prmst_simple(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    """Salesman master for Salesman Due name lookups."""
    zid = filters["zid"][0]
    sql = """
        SELECT
            xemp  AS spid,
            xname AS spname
        FROM prmst
        WHERE zid = %s
    """
    return sql, (zid,)


# ─────────────────────────────────────────────────────────────────────────────
# Manufacturing Analysis (moord/moodt) — 100000 / 100005 / 100009
# ─────────────────────────────────────────────────────────────────────────────

def get_mo_header_data(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    """MO header (moord): one row per manufacturing order — the finished good
    produced, qty produced, and the MO's open date. Only 'Completed' MOs are
    returned — xdatecom is unreliable (always 2999-12-31 in this DB, even for
    completed orders), so xdatemo is the only usable date and status filter
    happens here rather than per-caller.
    """
    zid = filters["zid"][0]
    sql = """
        SELECT
            moord.zid,
            moord.xmoord     AS monumber,
            moord.xitem      AS itemcode,
            moord.xqtyprd    AS qtyprd,
            moord.xunit      AS unit,
            moord.xstatusmor AS status,
            moord.xdatemo    AS date,
            EXTRACT(YEAR  FROM moord.xdatemo)::int AS year,
            EXTRACT(MONTH FROM moord.xdatemo)::int AS month,
            ci.xdesc         AS itemname,
            ci.xgitem        AS itemgroup
        FROM moord
        LEFT JOIN caitem ci ON moord.xitem = ci.xitem AND moord.zid = ci.zid
        WHERE moord.zid = %s
          AND moord.xstatusmor = 'Completed'
    """
    return sql, (zid,)


def get_mo_detail_data(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    """MO detail (moodt): one row per BOM/raw-material line consumed against
    an MO. No date of its own — join to mo_header on (zid, monumber) in the
    processing layer to get the MO's date/status/finished-good context.
    """
    zid = filters["zid"][0]
    sql = """
        SELECT
            moodt.zid,
            moodt.xmoord   AS monumber,
            moodt.xmorlno  AS lineno,
            moodt.xitem    AS itemcode,
            moodt.xwh      AS warehouse,
            moodt.xqty     AS qty,
            moodt.xqtyord  AS qtyord,
            moodt.xunit    AS unit,
            moodt.xrate    AS rate,
            ci.xdesc       AS itemname,
            ci.xgitem      AS itemgroup
        FROM moodt
        LEFT JOIN caitem ci ON moodt.xitem = ci.xitem AND moodt.zid = ci.zid
        WHERE moodt.zid = %s
    """
    return sql, (zid,)


def get_admin_expense_monthly(filters: Dict[str, Any]) -> Tuple[str, tuple]:
    """Total Office & Administrative expense (GL ac_code prefix '06') per
    month, for allocating overhead across finished goods by their share of
    that month's total production material cost (Manufacturing Analysis ->
    FG Costing tab).
    """
    zid = filters["zid"][0]
    sql = """
        SELECT
            glheader.zid,
            glheader.xyear::int AS year,
            glheader.xper::int  AS month,
            SUM(gldetail.xprime) AS value
        FROM gldetail
        JOIN glheader
          ON gldetail.xvoucher = glheader.xvoucher
         AND gldetail.zid      = glheader.zid
        WHERE gldetail.zid = %s
          AND LEFT(gldetail.xacc, 2) = '06'
        GROUP BY glheader.zid, glheader.xyear, glheader.xper
    """
    return sql, (zid,)
