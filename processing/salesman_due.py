# processing/salesman_due.py
# Faithful port of jupyter_audits/HM_36_*_Due.py (Zepto / HMBR / GI) into the
# Analytics pipeline for the "Salesman Due" report in Collection Analysis.

import numpy as np
import pandas as pd

_OPENING_LABEL = "Opening balance"
_MONTHS_BACK = 4
_BALANCE_TOL = 100.0
_MIN_ABS_BALANCE_KEEP = 100.0


def _is_month_col(col) -> bool:
    return (
        isinstance(col, str)
        and len(col) == 7
        and col[:4].isdigit()
        and col[4] == "_"
        and col[5:].isdigit()
    )


def prep_ar_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """Port of prep_customer_df_fillxsp_drop_ob_day_and_running.

    Cleans dtypes, drops non-OB rows on a customer's latest OB date, fills
    missing salesman codes (bfill then ffill per customer), and computes a
    running balance per customer.
    """
    customer_col, date_col, voucher_col = "xsub", "xdate", "xvoucher"
    amount_col, salesman_col, row_col = "xprime", "xsp", "xrow"

    d = df.copy()

    d[customer_col] = d[customer_col].astype(str)
    d[voucher_col] = d[voucher_col].astype(str).fillna("")
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[amount_col] = pd.to_numeric(d[amount_col], errors="coerce").fillna(0.0)

    d[row_col] = pd.to_numeric(d[row_col], errors="coerce")
    if d[row_col].isna().any():
        d[row_col] = d[row_col].fillna(pd.Series(np.arange(len(d)), index=d.index))

    d[salesman_col] = d[salesman_col].replace(r"^\s*$", np.nan, regex=True)

    d["vprefix"] = d[voucher_col].str.split("-", n=1).str[0]
    mask_no_dash = ~d[voucher_col].str.contains("-", na=False)
    d.loc[mask_no_dash, "vprefix"] = d.loc[mask_no_dash, voucher_col].str.extract(
        r"^([A-Za-z]+)", expand=False
    )
    d["vprefix"] = d["vprefix"].fillna("")

    is_ob = d["vprefix"].eq("OB")

    ob_date = (
        d.loc[is_ob & d[date_col].notna(), [customer_col, date_col]]
        .groupby(customer_col, sort=False)[date_col]
        .max()
        .rename("latest_ob_date")
        .reset_index()
    )
    d = d.merge(ob_date, on=customer_col, how="left")

    # Drop non-OB rows landing on the customer's latest OB date (keep the OB row itself).
    drop_mask = (
        d["latest_ob_date"].notna()
        & d[date_col].notna()
        & (d[date_col] == d["latest_ob_date"])
        & (~is_ob)
    )
    d = d.loc[~drop_mask].copy()

    d = d.sort_values([customer_col, date_col, row_col, voucher_col], kind="mergesort")

    # Beginning gaps -> bfill, then carry forward -> ffill, per customer.
    d[salesman_col] = d.groupby(customer_col, sort=False)[salesman_col].transform(
        lambda s: s.bfill().ffill()
    )

    d["running_balance"] = d.groupby(customer_col, sort=False)[amount_col].cumsum()

    d = d.drop(columns=["vprefix", "latest_ob_date"], errors="ignore")

    return d


def build_latest_sale_collection_report(
    df_clean: pd.DataFrame, balance_tol: float = _BALANCE_TOL
) -> pd.DataFrame:
    """Port of make_latest_sale_collection_report (codes only)."""
    customer_col, date_col, voucher_col = "xsub", "xdate", "xvoucher"
    amount_col, salesman_col, running_col = "xprime", "xsp", "running_balance"
    customer_city_col, customer_state_col = "xcity", "xstate"

    d = df_clean.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[amount_col] = pd.to_numeric(d[amount_col], errors="coerce").fillna(0.0)

    v = d[voucher_col].astype(str).fillna("")
    has_dash = v.str.contains("-", na=False)
    d["prefix"] = np.where(
        has_dash,
        v.str.split("-", n=1).str[0],
        v.str.extract(r"^([A-Za-z]+)", expand=False),
    )

    sort_cols = [customer_col, date_col, "xrow"]

    bal = (
        d.sort_values(sort_cols, kind="mergesort")
        .groupby(customer_col, as_index=False)[running_col]
        .last()
        .rename(columns={running_col: "Current Balance"})
    )

    sales = d[d["prefix"] == "INOP"].copy()
    last_sale = (
        sales.sort_values(sort_cols, kind="mergesort")
        .groupby(customer_col, as_index=False)
        .last()[[customer_col, date_col, amount_col, salesman_col]]
        .rename(
            columns={
                date_col: "Sales Date",
                amount_col: "Sale Amount",
                salesman_col: "Salesman Code",
            }
        )
    )

    col = d[d[amount_col] < 0].copy()
    last_col = (
        col.sort_values(sort_cols, kind="mergesort")
        .groupby(customer_col, as_index=False)
        .last()[[customer_col, date_col, amount_col]]
        .rename(
            columns={
                date_col: "Latest Collection Date",
                amount_col: "Latest Collection Amount",
            }
        )
    )
    last_col["Latest Collection Amount"] = last_col["Latest Collection Amount"].abs()

    last_sp = (
        d.sort_values(sort_cols, kind="mergesort")
        .groupby(customer_col, as_index=False)[salesman_col]
        .last()
        .rename(columns={salesman_col: "_fallback_sp"})
    )

    cust_loc = d[[customer_col, customer_city_col, customer_state_col]].drop_duplicates(
        subset=[customer_col]
    )

    report = (
        bal.merge(last_sale, on=customer_col, how="left")
        .merge(last_col, on=customer_col, how="left")
        .merge(last_sp, on=customer_col, how="left")
        .merge(cust_loc, on=customer_col, how="left")
    )

    report["Salesman Code"] = report["Salesman Code"].fillna(report["_fallback_sp"])
    report = report.drop(columns="_fallback_sp")

    report = report[report["Current Balance"].abs() >= balance_tol].copy()

    for c in ["Sale Amount", "Latest Collection Amount", "Current Balance"]:
        report[c] = pd.to_numeric(report[c], errors="coerce").round(2)

    report = report.rename(
        columns={
            customer_col: "Customer Code",
            customer_city_col: "City",
            customer_state_col: "State",
        }
    )

    report = report[
        [
            "Salesman Code",
            "Customer Code",
            "City",
            "State",
            "Sales Date",
            "Sale Amount",
            "Latest Collection Date",
            "Latest Collection Amount",
            "Current Balance",
        ]
    ].sort_values(["Salesman Code", "Current Balance"], ascending=[True, False]).reset_index(
        drop=True
    )

    return report


def build_trickledown_balances(
    df_clean: pd.DataFrame,
    months_back: int = _MONTHS_BACK,
    opening_label: str = _OPENING_LABEL,
    min_abs_balance_keep: float = _MIN_ABS_BALANCE_KEEP,
) -> pd.DataFrame:
    """Port of build_trickledown_balances_customer_credits.

    FIFO trickle-down of customer credits across months, then allocated to
    salesmen proportional to their sales share within each month.
    """
    customer_col, salesman_col, date_col, amount_col = "xsub", "xsp", "xdate", "xprime"

    d = df_clean.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce").dt.normalize()
    d[amount_col] = pd.to_numeric(d[amount_col], errors="coerce").fillna(0.0)
    d[customer_col] = d[customer_col].astype(str)
    d[salesman_col] = d[salesman_col].astype(str)

    d = d.loc[d[date_col].notna()].copy()

    as_of = pd.Timestamp.today().normalize()
    current_period = as_of.to_period("M")

    window_periods = [current_period - (months_back - 1 - i) for i in range(months_back)]
    window_labels = [f"{p.year}_{p.month:02d}" for p in window_periods]
    window_set = set(window_periods)
    window_start = window_periods[0]

    # ignore any future-dated rows
    d = d.loc[d[date_col].dt.to_period("M") <= current_period].copy()
    d["ym"] = d[date_col].dt.to_period("M")

    # Sales per (customer, salesman, month): positives only.
    sales_sp = (
        d.loc[d[amount_col] > 0]
        .groupby([customer_col, salesman_col, "ym"], as_index=False)[amount_col]
        .sum()
        .rename(columns={amount_col: "sales_sp"})
    )

    # Total sales per (customer, month), summed across salesmen.
    sales_cus = (
        sales_sp.groupby([customer_col, "ym"], as_index=False)["sales_sp"]
        .sum()
        .rename(columns={"sales_sp": "sales_cus"})
    )

    # Credits per (customer, month): negatives only (customer-level).
    credits = (
        d.loc[d[amount_col] < 0]
        .groupby([customer_col, "ym"], as_index=False)[amount_col]
        .sum()
        .rename(columns={amount_col: "credit_sum"})
    )
    credits["credit_mag"] = -credits["credit_sum"]

    cus_month = sales_cus.merge(
        credits[[customer_col, "ym", "credit_mag"]],
        on=[customer_col, "ym"],
        how="outer",
    ).fillna({"sales_cus": 0.0, "credit_mag": 0.0})

    def _fifo_customer(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("ym").copy()

        credit_left = float(g["credit_mag"].sum())
        rem = np.empty(len(g), dtype=float)

        sales_arr = g["sales_cus"].to_numpy(dtype=float)
        for i, sale in enumerate(sales_arr):
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

    cus_month_rem = cus_month.groupby(customer_col, sort=False, group_keys=False).apply(
        _fifo_customer
    )

    # Allocate remaining customer-month balance to salesmen by their sales share.
    alloc = sales_sp.merge(
        cus_month_rem[[customer_col, "ym", "sales_cus", "rem_cus_month"]],
        on=[customer_col, "ym"],
        how="left",
    )

    alloc["share"] = np.where(
        alloc["sales_cus"] > 0, alloc["sales_sp"] / alloc["sales_cus"], 0.0
    )
    alloc["rem_sp_month"] = (alloc["rem_cus_month"].fillna(0.0) * alloc["share"]).astype(float)

    # Opening = sum(rem_sp_month) for months before the reporting window.
    opening = (
        alloc.loc[alloc["ym"] < window_start]
        .groupby([salesman_col, customer_col], as_index=False)["rem_sp_month"]
        .sum()
        .rename(columns={"rem_sp_month": opening_label})
    )

    win = alloc.loc[alloc["ym"].isin(window_set)].copy()
    win["ym_label"] = win["ym"].apply(lambda p: f"{p.year}_{p.month:02d}")

    win_pivot = win.pivot_table(
        index=[salesman_col, customer_col],
        columns="ym_label",
        values="rem_sp_month",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()

    for c in window_labels:
        if c not in win_pivot.columns:
            win_pivot[c] = 0.0

    report = win_pivot.merge(opening, on=[salesman_col, customer_col], how="left")
    report[opening_label] = report[opening_label].fillna(0.0)

    ordered_cols = [salesman_col, customer_col, opening_label] + window_labels
    report = report[ordered_cols].copy()

    if min_abs_balance_keep is not None:
        total = report[[opening_label] + window_labels].sum(axis=1)
        report = report.loc[total.abs() >= float(min_abs_balance_keep)].copy()

    for c in [opening_label] + window_labels:
        report[c] = pd.to_numeric(report[c], errors="coerce").fillna(0.0).round(2)

    return report


def attach_names_and_area(
    report_cc: pd.DataFrame,
    df_customer: pd.DataFrame,
    df_salesman: pd.DataFrame,
    opening_label: str = _OPENING_LABEL,
) -> pd.DataFrame:
    """Merge customer/salesman names + city/state, then rename to report_cc_with_names schema."""
    merged = report_cc.merge(
        df_customer[["Customer Code", "Customer Name", "xcity", "xstate"]],
        left_on="xsub",
        right_on="Customer Code",
        how="left",
    ).merge(
        df_salesman[["Salesman Code", "Salesman Name"]],
        left_on="xsp",
        right_on="Salesman Code",
        how="left",
    )

    merged = merged.drop(columns=["Customer Code", "Salesman Code"])

    fixed_cols = ["xsp", "Salesman Name", "xsub", "Customer Name", "xcity", "xstate", opening_label]
    month_cols = [c for c in merged.columns if c not in fixed_cols]
    merged = merged[fixed_cols + month_cols]

    merged = merged.rename(
        columns={
            "Salesman Name": "salesman_name",
            "Customer Name": "customer_name",
            "xcity": "city",
            "xstate": "area",
        }
    )

    return merged


def add_total_due_column(df: pd.DataFrame, opening_label: str = _OPENING_LABEL) -> pd.DataFrame:
    """Port of add_total_due_column: sums Opening Balance + dynamic YYYY_MM month columns."""
    d = df.copy()

    month_cols = [c for c in d.columns if _is_month_col(c)]

    sum_cols = []
    if opening_label in d.columns:
        sum_cols.append(opening_label)
    sum_cols.extend(month_cols)

    d["total_due"] = (
        d[sum_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).round(2)
    )

    return d


def group_by_salesman_area_city(
    df: pd.DataFrame,
    salesman_cols=("xsp", "salesman_name"),
    area_col="area",
    city_col="city",
    opening_label: str = _OPENING_LABEL,
    total_due_col="total_due",
) -> pd.DataFrame:
    """Port of group_by_salesman_area_city."""
    d = df.copy()

    month_cols = [c for c in d.columns if _is_month_col(c)]

    sum_cols = []
    if opening_label in d.columns:
        sum_cols.append(opening_label)
    sum_cols.extend(month_cols)
    if total_due_col in d.columns:
        sum_cols.append(total_due_col)

    for c in sum_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    grouped = d.groupby(list(salesman_cols) + [area_col, city_col], dropna=False, as_index=False)[
        sum_cols
    ].sum()

    grouped[sum_cols] = grouped[sum_cols].round(2)

    grouped = grouped.sort_values(
        [salesman_cols[1], area_col, city_col], kind="mergesort"
    ).reset_index(drop=True)

    return grouped


def _month_label(col: str) -> str:
    y, m = col.split("_")
    return pd.to_datetime(f"{y}-{m}-01").strftime("%b-%Y")


def build_main_due_report_market(report_salesman_area_city: pd.DataFrame) -> pd.DataFrame:
    """Port of build_main_due_report_grouped (HMBR / GI): Retail/District market split,
    per-market summary rows, and a grand total row."""
    d = report_salesman_area_city.copy()

    d["Market"] = np.where(
        d["area"].str.contains("retail", case=False, na=False), "Retail", "District"
    )

    month_cols = sorted([c for c in d.columns if _is_month_col(c)])
    month_rename = {c: _month_label(c) for c in month_cols}
    d = d.rename(columns=month_rename)
    readable_months = list(month_rename.values())

    d = d.rename(
        columns={
            "xsp": "Salesman ID",
            "salesman_name": "Salesman Name",
            _OPENING_LABEL: "Opening Balance",
            "total_due": "Total Due",
        }
    )

    detail_cols = ["Salesman ID", "Salesman Name", "Market", "Opening Balance"] + readable_months + [
        "Total Due"
    ]
    d = d[detail_cols]

    for c in ["Opening Balance"] + readable_months + ["Total Due"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    grouped = d.groupby(["Salesman ID", "Salesman Name", "Market"], as_index=False)[
        ["Opening Balance"] + readable_months + ["Total Due"]
    ].sum()

    market_summary = grouped.groupby("Market", as_index=False)[
        ["Opening Balance"] + readable_months + ["Total Due"]
    ].sum()

    market_summary["Salesman ID"] = ""
    market_summary["Salesman Name"] = ""
    market_summary["Market"] = market_summary["Market"].replace(
        {"Retail": "Retail Due", "District": "District Due"}
    )

    grand_total = pd.DataFrame(
        [
            {
                "Salesman ID": "",
                "Salesman Name": "Grand Total",
                "Market": "",
                **{c: grouped[c].sum() for c in ["Opening Balance"] + readable_months},
                "Total Due": grouped["Total Due"].sum(),
            }
        ]
    )
    grand_total = grand_total[detail_cols]

    return pd.concat([grouped, market_summary, grand_total], ignore_index=True)


def build_main_due_report_simple(report_salesman_area_city: pd.DataFrame) -> pd.DataFrame:
    """Port of the Zepto main_due_report: salesman totals + a grand "Total" row,
    no Retail/District market split."""
    d = report_salesman_area_city.drop(columns=["area", "city"])
    d = d.groupby(["xsp", "salesman_name"]).sum().reset_index()

    numeric_columns = d.select_dtypes(include=["number"]).columns
    totals = d[numeric_columns].sum()

    total_row = {"xsp": "Total", "salesman_name": ""}
    for c in numeric_columns:
        total_row[c] = totals[c]

    return pd.concat([d, pd.DataFrame([total_row])], ignore_index=True)


def finalize_report_df(
    report_df: pd.DataFrame, df_customer: pd.DataFrame, df_salesman: pd.DataFrame
) -> pd.DataFrame:
    """Port of section 8.1: merge in Customer Name / Salesman Name, reorder, drop xcity/xstate."""
    merged = pd.merge(report_df, df_customer, on="Customer Code", how="left")
    merged = pd.merge(merged, df_salesman, on="Salesman Code", how="left")

    cols = list(merged.columns)
    cols.remove("Salesman Name")
    cols.insert(cols.index("Salesman Code") + 1, "Salesman Name")
    cols.remove("Customer Name")
    cols.insert(cols.index("Customer Code") + 1, "Customer Name")
    merged = merged[cols]

    merged = merged.drop(columns=["xcity", "xstate"])

    return merged


def build_missing_customers(
    report_df_final: pd.DataFrame, report_cc_with_names: pd.DataFrame
) -> pd.DataFrame:
    """Port of section 8.2: rows in report_df not present in report_cc_with_names."""
    return report_df_final[
        ~report_df_final.set_index(["Customer Code", "Salesman Code"]).index.isin(
            report_cc_with_names.set_index(["xsub", "xsp"]).index
        )
    ].reset_index(drop=True)


def build_salesman_due_reports(
    ar_df: pd.DataFrame,
    cacus_df: pd.DataFrame,
    prmst_df: pd.DataFrame,
    market_split: bool,
) -> dict:
    """End-to-end port of jupyter_audits/HM_36_*_Due.py.

    market_split=True uses the HMBR/GI main_due_report (Retail/District split
    + market summary + grand total). market_split=False uses the Zepto
    main_due_report (salesman totals + single "Total" row).

    Returns a dict with keys: main_due_report, report_df, report_cc_with_names,
    missing_customers_df.
    """
    if ar_df is None or ar_df.empty:
        raise ValueError("No AR customer data found")

    df_customer = cacus_df.rename(columns={"cusid": "Customer Code", "cusname": "Customer Name"})
    df_customer["Customer Code"] = df_customer["Customer Code"].astype(str)

    df_salesman = prmst_df.rename(columns={"spid": "Salesman Code", "spname": "Salesman Name"})
    df_salesman["Salesman Code"] = df_salesman["Salesman Code"].astype(str)

    df_clean = prep_ar_ledger(ar_df)

    report_df = build_latest_sale_collection_report(df_clean)
    report_cc = build_trickledown_balances(df_clean)

    report_cc_with_names = attach_names_and_area(report_cc, df_customer, df_salesman)
    report_cc_with_names = add_total_due_column(report_cc_with_names)

    report_salesman_area_city = group_by_salesman_area_city(report_cc_with_names)

    if market_split:
        main_due_report = build_main_due_report_market(report_salesman_area_city)
    else:
        main_due_report = build_main_due_report_simple(report_salesman_area_city)

    report_df_final = finalize_report_df(report_df, df_customer, df_salesman)
    missing_customers_df = build_missing_customers(report_df_final, report_cc_with_names)

    return {
        "main_due_report": main_due_report,
        "report_df": report_df_final,
        "report_cc_with_names": report_cc_with_names,
        "missing_customers_df": missing_customers_df,
    }
