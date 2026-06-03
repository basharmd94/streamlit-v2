import calendar
import pandas as pd


def _month_label(year: int, month: int) -> str:
    return f"{calendar.month_abbr[month]}-{str(year)[-2:]}"


def _build_pivot(sales_df: pd.DataFrame, returns_df: pd.DataFrame, id_cols: list) -> pd.DataFrame:
    """
    Build a wide-format net-sales pivot (rows = id_cols groups, columns = month labels).
    Assumes sales_df already has 'final_sales' (computed by data_copy_add_columns).
    Net sales = final_sales - treturnamt.
    """
    period_cols = ["year", "month"]

    sales_df = sales_df.copy()
    sales_df["year"] = pd.to_numeric(sales_df["year"], errors="coerce").astype("Int64")
    sales_df["month"] = pd.to_numeric(sales_df["month"], errors="coerce").astype("Int64")
    sales_df = sales_df.dropna(subset=["year", "month"])

    # Aggregate sales by id + period
    grp_cols = [c for c in id_cols + period_cols if c in sales_df.columns]
    sale_grp = (
        sales_df
        .groupby(grp_cols, dropna=False)["final_sales"]
        .sum()
        .reset_index()
        .rename(columns={"final_sales": "_sales"})
    )

    # Aggregate returns on keys that exist in both returns_df and sale_grp
    ret_candidate_keys = ["spid", "cusid", "itemcode"] + period_cols
    ret_keys = [c for c in ret_candidate_keys if c in returns_df.columns and c in sale_grp.columns]

    if not returns_df.empty and "treturnamt" in returns_df.columns and ret_keys:
        r = returns_df.copy()
        r["year"] = pd.to_numeric(r["year"], errors="coerce").astype("Int64")
        r["month"] = pd.to_numeric(r["month"], errors="coerce").astype("Int64")
        r = r.dropna(subset=["year", "month"])
        ret_grp = (
            r.groupby(ret_keys, dropna=False)["treturnamt"]
            .sum()
            .reset_index()
        )
        merged = sale_grp.merge(ret_grp, on=ret_keys, how="left")
    else:
        merged = sale_grp.copy()
        merged["treturnamt"] = 0.0

    merged["treturnamt"] = merged["treturnamt"].fillna(0.0)
    merged["net_sales"] = merged["_sales"] - merged["treturnamt"]
    merged["year"] = merged["year"].astype(int)
    merged["month"] = merged["month"].astype(int)
    merged["_period"] = merged.apply(lambda r: _month_label(r["year"], r["month"]), axis=1)

    pivot_id_cols = [c for c in id_cols if c in merged.columns]
    pivot = merged.pivot_table(
        index=pivot_id_cols,
        columns="_period",
        values="net_sales",
        aggfunc="sum",
        fill_value=0,
    )
    pivot.columns.name = None

    # Sort columns chronologically
    period_order = (
        merged[["year", "month", "_period"]]
        .drop_duplicates()
        .sort_values(["year", "month"])
    )
    seen: set = set()
    ordered_cols = []
    for _, row in period_order.iterrows():
        lbl = row["_period"]
        if lbl in pivot.columns and lbl not in seen:
            ordered_cols.append(lbl)
            seen.add(lbl)

    pivot = pivot[ordered_cols]
    pivot["Total"] = pivot.sum(axis=1)
    return pivot.reset_index()


def build_customer_wise_monthly(sales_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """Report 1: one row per (salesman × customer), monthly net sales columns."""
    if sales_df.empty or "final_sales" not in sales_df.columns:
        return pd.DataFrame()
    id_cols = ["spid", "spname", "cusid", "cusname", "cusmobile", "whatsapp", "area"]
    return _build_pivot(sales_df, returns_df, id_cols)


def build_customer_product_monthly(sales_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """Report 2: one row per (salesman × customer × product), monthly net sales columns."""
    if sales_df.empty or "final_sales" not in sales_df.columns:
        return pd.DataFrame()
    id_cols = ["spid", "spname", "cusid", "cusname", "cusmobile", "whatsapp", "area", "itemcode", "itemname"]
    return _build_pivot(sales_df, returns_df, id_cols)
