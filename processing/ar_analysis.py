"""
AR Analysis Processing Module
Translated from HM_36_*_Due.py scripts (da database) to stream2 schema.

Column mapping from da-database originals:
    xvoucher     -> gldetail.voucher
    xsub         -> gldetail.ac_sub
    xprime       -> gldetail.value
    xsp          -> resolved via sales.sp_id (reliable) then last_sale_sp fallback
    xaccusage    -> glmst.usage  (via JOIN on ac_code)
    xrow         -> gldetail.row  (added to stream2)
    xdate        -> glheader.date
    xcity        -> cacus.cuscity
    xstate       -> cacus.cusstate
    customer_name-> cacus.cusname
    salesman_name-> employee.spname

Key fix: gldetail.sp_id in stream2 contains dirty data (partial codes, names).
We resolve salesman using the sales table which is reliable.
"""

import numpy as np
import pandas as pd
from datetime import date
from core.db import get_dataframe


# ---------------------------------------------------------------------------
# 1. Data fetch
# ---------------------------------------------------------------------------

def fetch_ar_data(zid, project, till_date=None):
    """Fetch AR GL ledger rows from stream2 for the given business/project.

    Salesman (xsp) is resolved using:
      1. sales.sp_id for INOP vouchers (most reliable — same source as all other analysis)
      2. last_sale_sp: most recent sales.sp_id per customer, as fallback for non-INOP rows

    xrow (gldetail.row) is included for deterministic intra-voucher ordering.

    Parameters
    ----------
    zid : str | int
    project : str   e.g. 'GULSHAN TRADING'
    till_date : date | None   Upper bound for glheader.date. Defaults to today.

    Returns
    -------
    pd.DataFrame | None
        Columns: zid, xvoucher, xdate, year, month, xsub, customer_name,
                 xcity, xstate, xprime, xrow, xsp, salesman_name
    """
    till_str = str(till_date) if till_date is not None else str(date.today())

    sql = """
WITH inop_sp AS (
    -- Reliable salesman per INOP voucher from the sales table
    SELECT DISTINCT ON (zid, ordernumber)
        zid,
        ordernumber AS voucher,
        sp_id::text AS sp_id
    FROM sales
    WHERE zid = %s
      AND sp_id IS NOT NULL
      AND TRIM(sp_id::text) != ''
    ORDER BY zid, ordernumber
),
last_sale_sp AS (
    -- Fallback: most recent salesman per customer from sales table
    SELECT DISTINCT ON (zid, cusid)
        zid,
        cusid::text AS cusid,
        sp_id::text AS sp_id
    FROM sales
    WHERE zid = %s
      AND sp_id IS NOT NULL
      AND TRIM(sp_id::text) != ''
    ORDER BY zid, cusid, date DESC
)
SELECT
    gd.zid,
    gd.voucher                          AS xvoucher,
    gh.date                             AS xdate,
    EXTRACT(YEAR  FROM gh.date)::int    AS year,
    EXTRACT(MONTH FROM gh.date)::int    AS month,
    gd.ac_sub                           AS xsub,
    COALESCE(cacus.cusname, gd.ac_sub)  AS customer_name,
    cacus.cuscity                       AS xcity,
    cacus.cusstate                      AS xstate,
    gd.value                            AS xprime,
    gd.row                              AS xrow,
    COALESCE(inop_sp.sp_id, ls.sp_id)   AS xsp,
    e.spname                            AS salesman_name
FROM gldetail gd
JOIN glmst gm ON gd.ac_code = gm.ac_code AND gd.zid = gm.zid
JOIN glheader gh ON gd.voucher = gh.voucher AND gd.zid = gh.zid
LEFT JOIN cacus ON gd.ac_sub = cacus.cusid AND gd.zid = cacus.zid
LEFT JOIN inop_sp ON gd.zid = inop_sp.zid AND gd.voucher = inop_sp.voucher
LEFT JOIN last_sale_sp ls ON gd.zid = ls.zid AND gd.ac_sub::text = ls.cusid
LEFT JOIN employee e
  ON COALESCE(inop_sp.sp_id, ls.sp_id) = e.spid::text
 AND e.zid = gd.zid
WHERE gd.zid = %s
  AND gm.usage = 'AR'
  AND gd.project = %s
  AND gh.date <= %s
  AND (
       gd.voucher LIKE 'INOP%%' OR gd.voucher LIKE 'RCT%%'  OR gd.voucher LIKE 'BRCT%%'
    OR gd.voucher LIKE 'CRCT%%' OR gd.voucher LIKE 'SRJV%%' OR gd.voucher LIKE 'SRT%%'
    OR gd.voucher LIKE 'JV%%'   OR gd.voucher LIKE 'IMSA%%' OR gd.voucher LIKE 'STJV%%'
    OR gd.voucher LIKE 'CPAY%%' OR gd.voucher LIKE 'PAY%%'  OR gd.voucher LIKE 'CHQ%%'
    OR gd.voucher LIKE 'ADJV%%' OR gd.voucher LIKE 'TR%%'   OR gd.voucher LIKE 'BPAY%%'
    OR gd.voucher LIKE 'BTJV%%'
  )
ORDER BY gd.ac_sub, gh.date, gd.row, gd.voucher
"""
    params = (zid, zid, zid, project, till_str)
    return get_dataframe(sql, params)


# ---------------------------------------------------------------------------
# 2. Employee list
# ---------------------------------------------------------------------------

def get_employee_list(zid):
    """Return DataFrame with (spid, spname) for the given ZID."""
    sql = "SELECT spid::text AS spid, spname FROM employee WHERE zid = %s ORDER BY spname"
    return get_dataframe(sql, (zid,))


# ---------------------------------------------------------------------------
# 3. Customer DataFrame preparation
# ---------------------------------------------------------------------------

def prep_customer_df(
    df,
    customer_col="xsub",
    date_col="xdate",
    voucher_col="xvoucher",
    amount_col="xprime",
    salesman_col="xsp",
    row_col="xrow",
):
    """Port of prep_customer_df_fillxsp_drop_ob_day_and_running.

    Returns (cleaned_df, missing_diag_df).
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    d = df.copy()

    # 1. Clean types
    d[customer_col] = d[customer_col].astype(str)
    d[voucher_col]  = d[voucher_col].astype(str).fillna("")
    d[date_col]     = pd.to_datetime(d[date_col], errors="coerce")
    d[amount_col]   = pd.to_numeric(d[amount_col], errors="coerce").fillna(0.0)

    # 2. xrow: numeric for deterministic sorting (now present in stream2 as 'xrow')
    if row_col in d.columns:
        d[row_col] = pd.to_numeric(d[row_col], errors="coerce")
        if d[row_col].isna().any():
            d[row_col] = d[row_col].fillna(pd.Series(np.arange(len(d)), index=d.index))
        _row_col = row_col
    else:
        d["_row_fallback"] = np.arange(len(d))
        _row_col = "_row_fallback"

    # 3. Normalize blank xsp -> NaN
    d[salesman_col] = d[salesman_col].replace(r"^\s*$", np.nan, regex=True)

    # 4. Extract voucher prefix
    v = d[voucher_col].astype(str).fillna("")
    has_dash = v.str.contains("-", na=False)
    d["vprefix"] = np.where(
        has_dash,
        v.str.split("-", n=1).str[0],
        v.str.extract(r"^([A-Za-z]+)", expand=False),
    )
    d["vprefix"] = d["vprefix"].fillna("")

    is_ob = d["vprefix"].eq("OB")

    # 5. Latest OB date per customer
    ob_dates = (
        d.loc[is_ob & d[date_col].notna(), [customer_col, date_col]]
         .groupby(customer_col, sort=False)[date_col]
         .max()
         .rename("latest_ob_date")
         .reset_index()
    )
    d = d.merge(ob_dates, on=customer_col, how="left")

    # 6. Drop non-OB rows that fall on the same date as latest OB
    drop_mask = (
        d["latest_ob_date"].notna()
        & d[date_col].notna()
        & (d[date_col] == d["latest_ob_date"])
        & (~is_ob)
    )
    d = d.loc[~drop_mask].copy()

    # 7. Sort deterministically: customer, date, xrow, voucher
    d = d.sort_values(
        [customer_col, date_col, _row_col, voucher_col], kind="mergesort"
    )

    # 8. Fill xsp: bfill then ffill within each customer group
    d[salesman_col] = (
        d.groupby(customer_col, sort=False)[salesman_col]
         .transform(lambda s: s.bfill().ffill())
    )

    # 9. Diagnostics: customers still missing xsp
    missing_diag = (
        d.loc[d[salesman_col].isna(), [customer_col]]
         .drop_duplicates()
         .rename(columns={customer_col: "Customer Code"})
         .reset_index(drop=True)
    )

    # 10. Running balance per customer
    d["running_balance"] = d.groupby(customer_col, sort=False)[amount_col].cumsum()

    # 11. Clean up temp columns
    d = d.drop(columns=["vprefix", "latest_ob_date"], errors="ignore")

    return d, missing_diag


# ---------------------------------------------------------------------------
# 4. Latest sale & collection report
# ---------------------------------------------------------------------------

def make_latest_sale_collection_report(
    df_clean,
    customer_col="xsub",
    customer_city_col="xcity",
    customer_state_col="xstate",
    salesman_col="xsp",
    date_col="xdate",
    voucher_col="xvoucher",
    amount_col="xprime",
    running_col="running_balance",
    balance_tol=100.0,
    show_collection_as_positive=True,
):
    """Port of make_latest_sale_collection_report from the original script."""
    if df_clean is None or df_clean.empty:
        return pd.DataFrame()

    d = df_clean.copy()
    d[date_col]   = pd.to_datetime(d[date_col], errors="coerce")
    d[amount_col] = pd.to_numeric(d[amount_col], errors="coerce").fillna(0.0)

    # Voucher prefix
    v = d[voucher_col].astype(str).fillna("")
    has_dash = v.str.contains("-", na=False)
    d["prefix"] = np.where(
        has_dash,
        v.str.split("-", n=1).str[0],
        v.str.extract(r"^([A-Za-z]+)", expand=False),
    )

    # Sort cols
    row_c = "xrow" if "xrow" in d.columns else voucher_col
    sort_cols = [customer_col, date_col, row_c]

    # Current balance (last running_balance per customer)
    if running_col in d.columns:
        bal = (
            d.sort_values(sort_cols, kind="mergesort")
             .groupby(customer_col, as_index=False)[running_col]
             .last()
             .rename(columns={running_col: "Current Balance"})
        )
    else:
        bal = (
            d.groupby(customer_col, as_index=False)[amount_col]
             .sum()
             .rename(columns={amount_col: "Current Balance"})
        )

    # Latest INOP (sale)
    sales_df = d[d["prefix"] == "INOP"].copy()
    last_sale = (
        sales_df.sort_values(sort_cols, kind="mergesort")
                .groupby(customer_col, as_index=False)
                .last()[[customer_col, date_col, amount_col, salesman_col]]
                .rename(columns={date_col: "Sales Date", amount_col: "Sale Amount",
                                 salesman_col: "Salesman Code"})
    )

    # Latest collection (negative amount)
    coll_df = d[d[amount_col] < 0].copy()
    last_coll = (
        coll_df.sort_values(sort_cols, kind="mergesort")
               .groupby(customer_col, as_index=False)
               .last()[[customer_col, date_col, amount_col]]
               .rename(columns={date_col: "Latest Collection Date",
                                amount_col: "Latest Collection Amount"})
    )
    if show_collection_as_positive and not last_coll.empty:
        last_coll["Latest Collection Amount"] = last_coll["Latest Collection Amount"].abs()

    # Fallback salesman (for customers with no INOP rows)
    last_sp = (
        d.sort_values(sort_cols, kind="mergesort")
         .groupby(customer_col, as_index=False)[salesman_col]
         .last()
         .rename(columns={salesman_col: "_fallback_sp"})
    )

    # Customer location
    cust_loc = (
        d[[customer_col, customer_city_col, customer_state_col]]
        .drop_duplicates(subset=[customer_col])
    )

    # Customer name
    if "customer_name" in d.columns:
        cust_name = d[[customer_col, "customer_name"]].drop_duplicates(subset=[customer_col])
    else:
        cust_name = None

    # Salesman name
    if "salesman_name" in d.columns:
        sp_name_df = (
            d.sort_values(sort_cols, kind="mergesort")
             .groupby(customer_col, as_index=False)["salesman_name"]
             .last()
        )
    else:
        sp_name_df = None

    # Merge
    report = (
        bal
        .merge(last_sale, on=customer_col, how="left")
        .merge(last_coll,  on=customer_col, how="left")
        .merge(last_sp,    on=customer_col, how="left")
        .merge(cust_loc,   on=customer_col, how="left")
    )
    if cust_name is not None:
        report = report.merge(cust_name, on=customer_col, how="left")
    if sp_name_df is not None:
        report = report.merge(sp_name_df, on=customer_col, how="left")

    # Resolve salesman code & name
    report["Salesman Code"] = report["Salesman Code"].fillna(report["_fallback_sp"])
    report.drop(columns=["_fallback_sp"], inplace=True)

    if "salesman_name" in report.columns:
        report.rename(columns={"salesman_name": "Salesman Name"}, inplace=True)
    else:
        report["Salesman Name"] = np.nan

    # Filter near-zero balances
    report = report[report["Current Balance"].abs() >= balance_tol].copy()

    # Round
    for c in ["Sale Amount", "Latest Collection Amount", "Current Balance"]:
        if c in report.columns:
            report[c] = pd.to_numeric(report[c], errors="coerce").round(2)

    # Rename / final column order
    report.rename(columns={
        customer_col:          "Customer Code",
        customer_city_col:     "City",
        customer_state_col:    "State",
        "customer_name":       "Customer Name",
    }, inplace=True)

    out_cols = [
        "Salesman Code", "Salesman Name", "Customer Code", "Customer Name",
        "City", "State", "Sales Date", "Sale Amount",
        "Latest Collection Date", "Latest Collection Amount", "Current Balance",
    ]
    for col in out_cols:
        if col not in report.columns:
            report[col] = np.nan

    return (
        report[out_cols]
        .sort_values(["Salesman Code", "Current Balance"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 5. FIFO trickle-down balances  (vectorised — exact port of original script)
# ---------------------------------------------------------------------------

def build_trickledown_balances(
    df_clean,
    till_date=None,
    customer_col="xsub",
    salesman_col="xsp",
    date_col="xdate",
    amount_col="xprime",
    months_back=4,
    opening_label="Opening balance",
    min_abs_balance_keep=100.0,
):
    """Vectorised FIFO credit trickle-down — exact port of
    build_trickledown_balances_customer_credits from the original script.

    Uses till_date as the reference date for the reporting window
    (mirrors the script running on that specific date).
    """
    if df_clean is None or df_clean.empty:
        return pd.DataFrame()

    d = df_clean.copy()

    d[date_col]   = pd.to_datetime(d[date_col], errors="coerce").dt.normalize()
    d[amount_col] = pd.to_numeric(d[amount_col], errors="coerce").fillna(0.0)
    d[customer_col] = d[customer_col].astype(str)
    d[salesman_col] = d[salesman_col].astype(str)

    d = d.loc[d[date_col].notna()].copy()

    # Reporting window anchored to till_date (not today)
    as_of = pd.Timestamp(till_date).normalize() if till_date is not None else pd.Timestamp.today().normalize()
    current_period = as_of.to_period("M")

    window_periods = [current_period - (months_back - 1 - i) for i in range(months_back)]
    window_labels  = [f"{p.year}_{p.month:02d}" for p in window_periods]
    window_set     = set(window_periods)
    window_start   = window_periods[0]

    # Ignore future-dated rows (relative to till_date)
    d = d.loc[d[date_col].dt.to_period("M") <= current_period].copy()
    d["ym"] = d[date_col].dt.to_period("M")

    # 1) Sales per (customer, salesman, month) — positives only
    sales_sp = (
        d.loc[d[amount_col] > 0]
         .groupby([customer_col, salesman_col, "ym"], as_index=False)[amount_col]
         .sum()
         .rename(columns={amount_col: "sales_sp"})
    )

    if sales_sp.empty:
        return pd.DataFrame()

    # Total sales per (customer, month) across salesmen
    sales_cus = (
        sales_sp.groupby([customer_col, "ym"], as_index=False)["sales_sp"]
                .sum()
                .rename(columns={"sales_sp": "sales_cus"})
    )

    # 2) Credits per (customer, month) — negatives only
    credits = (
        d.loc[d[amount_col] < 0]
         .groupby([customer_col, "ym"], as_index=False)[amount_col]
         .sum()
         .rename(columns={amount_col: "credit_sum"})
    )
    credits["credit_mag"] = -credits["credit_sum"]

    cus_month = (
        sales_cus
        .merge(credits[[customer_col, "ym", "credit_mag"]], on=[customer_col, "ym"], how="outer")
        .fillna({"sales_cus": 0.0, "credit_mag": 0.0})
    )

    # 3) FIFO trickle-down per customer
    def _fifo_customer(g):
        g = g.sort_values("ym").copy()
        credit_left = float(g["credit_mag"].sum())
        rem = np.empty(len(g), dtype=float)
        for i, sale in enumerate(g["sales_cus"].to_numpy(dtype=float)):
            if credit_left <= 0:
                rem[i] = sale
            elif credit_left >= sale:
                rem[i] = 0.0
                credit_left -= sale
            else:
                rem[i] = sale - credit_left
                credit_left = 0.0
        g["rem_cus_month"] = rem
        return g[[customer_col, "ym", "sales_cus", "rem_cus_month"]]

    cus_month_rem = (
        cus_month.groupby(customer_col, sort=False, group_keys=False)
                 .apply(_fifo_customer)
    )

    # 4) Allocate remaining balance to salesmen proportionally
    alloc = sales_sp.merge(
        cus_month_rem[[customer_col, "ym", "sales_cus", "rem_cus_month"]],
        on=[customer_col, "ym"], how="left",
    )
    alloc["share"] = np.where(
        alloc["sales_cus"] > 0,
        alloc["sales_sp"] / alloc["sales_cus"],
        0.0,
    )
    alloc["rem_sp_month"] = (alloc["rem_cus_month"].fillna(0.0) * alloc["share"]).astype(float)

    # Opening balance per (salesman, customer) = months before window_start
    opening = (
        alloc.loc[alloc["ym"] < window_start]
             .groupby([salesman_col, customer_col], as_index=False)["rem_sp_month"]
             .sum()
             .rename(columns={"rem_sp_month": opening_label})
    )

    # Window months pivot
    win = alloc.loc[alloc["ym"].isin(window_set)].copy()
    win["ym_label"] = win["ym"].apply(lambda p: f"{p.year}_{p.month:02d}")

    win_pivot = (
        win.pivot_table(
            index=[salesman_col, customer_col],
            columns="ym_label",
            values="rem_sp_month",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    win_pivot.columns.name = None

    for col in window_labels:
        if col not in win_pivot.columns:
            win_pivot[col] = 0.0

    report = win_pivot.merge(opening, on=[salesman_col, customer_col], how="left")
    report[opening_label] = report[opening_label].fillna(0.0)

    ordered_cols = [salesman_col, customer_col, opening_label] + window_labels
    report = report[ordered_cols].copy()

    # Filter by min_abs_balance_keep
    if min_abs_balance_keep is not None:
        total = report[[opening_label] + window_labels].sum(axis=1)
        report = report.loc[total.abs() >= float(min_abs_balance_keep)].copy()

    # Round
    for c in [opening_label] + window_labels:
        report[c] = pd.to_numeric(report[c], errors="coerce").fillna(0.0).round(2)

    # total_due
    report["total_due"] = report[[opening_label] + window_labels].sum(axis=1).round(2)

    # Enrich with salesman_name and customer details
    if "salesman_name" in d.columns:
        sp_names = (
            d.dropna(subset=[salesman_col])
             .groupby(salesman_col)["salesman_name"]
             .last()
             .reset_index()
        )
        report = report.merge(sp_names, on=salesman_col, how="left")
    else:
        report["salesman_name"] = np.nan

    for src in ("customer_name", "xcity", "xstate"):
        if src in d.columns:
            info = d.groupby(customer_col)[src].first().reset_index()
            report = report.merge(info, on=customer_col, how="left")
        else:
            report[src] = np.nan

    # Final column order matching the original script
    fixed = [salesman_col, "salesman_name", customer_col, "customer_name",
             "xcity", "xstate", opening_label]
    extra = [c for c in report.columns if c not in fixed + ["total_due"]]
    report = report[fixed + extra + ["total_due"]].sort_values(
        [salesman_col, "total_due"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)

    return report


# ---------------------------------------------------------------------------
# 6. Salesman due summary (main due report — HMBR format)
# ---------------------------------------------------------------------------

def build_main_due_report(report_cc):
    """Build grouped main due report with Retail/District market split.

    Mirrors build_main_due_report_grouped from the HMBR script:
      - Groups by (Salesman ID, Salesman Name, Market)
      - Appends District Due / Retail Due subtotal rows
      - Appends Grand Total row
      - Month columns renamed from YYYY_MM to 'Mon-YYYY' format
    """
    if report_cc is None or report_cc.empty:
        return pd.DataFrame()

    import calendar as _cal

    d = report_cc.copy()

    # Detect dynamic month columns (YYYY_MM)
    month_cols = sorted([
        c for c in d.columns
        if isinstance(c, str) and len(c) == 7
        and c[:4].isdigit() and c[4] == "_" and c[5:].isdigit()
    ])

    # Market classification: xstate contains "retail" (case-insensitive) → Retail, else District
    area_col = "xstate" if "xstate" in d.columns else None
    if area_col:
        d["Market"] = np.where(
            d[area_col].astype(str).str.contains("retail", case=False, na=False),
            "Retail",
            "District",
        )
    else:
        d["Market"] = "District"

    # Rename columns to display format
    sp_col   = "xsp"
    name_col = "salesman_name" if "salesman_name" in d.columns else None

    def _month_label(col):
        y, m = int(col[:4]), int(col[5:])
        return f"{_cal.month_abbr[m]}-{y}"

    month_rename = {c: _month_label(c) for c in month_cols}
    d.rename(columns=month_rename, inplace=True)
    readable_months = list(month_rename.values())

    d.rename(columns={
        sp_col:            "Salesman ID",
        "Opening balance": "Opening Balance",
        "total_due":       "Total Due",
    }, inplace=True)
    if name_col:
        d.rename(columns={name_col: "Salesman Name"}, inplace=True)
    else:
        d["Salesman Name"] = ""

    # Ensure numerics
    for c in ["Opening Balance"] + readable_months + ["Total Due"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    detail_cols = ["Salesman ID", "Salesman Name", "Market", "Opening Balance"] \
                  + readable_months + ["Total Due"]
    d = d[[c for c in detail_cols if c in d.columns]]

    # Group by salesman + Market
    num_cols = ["Opening Balance"] + readable_months + ["Total Due"]
    num_cols = [c for c in num_cols if c in d.columns]
    grouped = (
        d.groupby(["Salesman ID", "Salesman Name", "Market"], as_index=False, dropna=False)[num_cols]
         .sum()
    )

    # Market subtotal rows
    market_summary = (
        grouped.groupby("Market", as_index=False)[num_cols].sum()
    )
    market_summary["Salesman ID"]   = ""
    market_summary["Salesman Name"] = ""
    market_summary["Market"] = market_summary["Market"].replace({
        "Retail": "Retail Due", "District": "District Due"
    })

    # Grand total row
    grand_row = {c: grouped[c].sum() for c in num_cols}
    grand_row.update({"Salesman ID": "", "Salesman Name": "Grand Total", "Market": ""})
    grand_total = pd.DataFrame([grand_row])

    final = pd.concat([grouped, market_summary, grand_total], ignore_index=True)
    all_cols = ["Salesman ID", "Salesman Name", "Market", "Opening Balance"] \
               + readable_months + ["Total Due"]
    for col in all_cols:
        if col not in final.columns:
            final[col] = np.nan
    return final[[c for c in all_cols if c in final.columns]]


# ---------------------------------------------------------------------------
# 7. Find missing customers
# ---------------------------------------------------------------------------

def find_missing_customers(report_df, report_cc):
    """Return rows in report_df whose (Customer Code, Salesman Code) pair is
    absent from report_cc — matching the original script's logic."""
    if report_df is None or report_df.empty:
        return pd.DataFrame()
    if report_cc is None or report_cc.empty:
        return report_df.copy()

    existing_customers = set(report_cc["xsub"].dropna().astype(str).unique())
    return (
        report_df[~report_df["Customer Code"].astype(str).isin(existing_customers)]
        .reset_index(drop=True)
    )
