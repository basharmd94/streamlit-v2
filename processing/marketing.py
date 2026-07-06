import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _yoy_growth_same_month(
    df: pd.DataFrame, value_col: str, years: list, metric: str
) -> pd.DataFrame:
    """Same-month YoY growth — eliminates partial-year bias.

    Instead of comparing annual totals (which makes a partial current year look
    like shrinkage), we compare each calendar month individually against the
    same month in the prior year, then average those per-month growth rates.

    - Only months where the PRIOR year had sales (base > 0) count toward the avg.
    - Months where prior year = 0 but current year > 0 are skipped (new activity
      in that month — including them as inf would inflate the average unfairly).
    - A customer with NO valid base months at all (entirely new) → "New ↑" (inf).
    - 3+ years: same logic applied to each consecutive year pair, then averaged.
    """
    col_name = f"yoy_{metric}_growth_pct"

    # Ensure numeric year/month
    df = df.copy()
    df["year"]  = pd.to_numeric(df["year"],  errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)

    years_sorted = sorted(
        y for y in years
        if pd.notna(y) and y in df["year"].dropna().unique()
    )
    if len(years_sorted) < 2:
        return pd.DataFrame(columns=["cusid"])

    # Monthly totals per customer
    monthly = (
        df.groupby(["cusid", "year", "month"])[value_col]
        .sum()
        .reset_index()
    )

    pair_avgs = []   # one Series (cusid → mean growth %) per consecutive year pair

    for i in range(1, len(years_sorted)):
        y0, y1 = years_sorted[i - 1], years_sorted[i]

        y0_m = (monthly[monthly["year"] == y0][["cusid", "month", value_col]]
                .rename(columns={value_col: "v0"}))
        y1_m = (monthly[monthly["year"] == y1][["cusid", "month", value_col]]
                .rename(columns={value_col: "v1"}))

        # Left-join current year onto prior year on cusid+month
        joined = y1_m.merge(y0_m, on=["cusid", "month"], how="left")
        joined["v0"] = joined["v0"].fillna(0.0)

        # Only months where the base (prior year) had sales
        valid = joined[joined["v0"] > 0].copy()
        if valid.empty:
            continue

        valid["pct"] = (valid["v1"] - valid["v0"]) / valid["v0"] * 100
        pair_avg = valid.groupby("cusid")["pct"].mean()
        pair_avgs.append(pair_avg)

    if not pair_avgs:
        # No customer had a valid base month across any transition
        new_cus = monthly[monthly["year"] == years_sorted[-1]]["cusid"].unique()
        if len(new_cus):
            return pd.DataFrame({"cusid": new_cus, col_name: np.inf})
        return pd.DataFrame(columns=["cusid", col_name])

    # Average across all year-pair transitions
    combined = pd.concat(pair_avgs, axis=1)
    avg = combined.mean(axis=1).round(1)
    avg.name = col_name
    result = avg.reset_index()

    # Customers in the latest year but not captured above → fully new → inf
    all_in_result = set(result["cusid"])
    last_year_cus = set(monthly[monthly["year"] == years_sorted[-1]]["cusid"])
    new_cus = last_year_cus - all_in_result
    if new_cus:
        result = pd.concat(
            [result, pd.DataFrame({"cusid": list(new_cus), col_name: np.inf})],
            ignore_index=True,
        )

    return result


def _sales_metrics(sales_df: pd.DataFrame, years: list) -> pd.DataFrame:
    """Per-customer sales aggregates, YoY, order interval, activity rate."""
    if sales_df is None or sales_df.empty:
        return pd.DataFrame(columns=[
            "cusid", "cusname", "area", "spname",
            "total_sales", "order_count",
            "avg_order_interval_days", "monthly_activity_rate",
        ])

    s = _ensure_date(sales_df, "date")
    s["altsales"] = pd.to_numeric(s["altsales"], errors="coerce").fillna(0.0)

    if "year" not in s.columns:
        s["year"] = s["date"].dt.year
    if "month" not in s.columns:
        s["month"] = s["date"].dt.month

    s["year"] = pd.to_numeric(s["year"], errors="coerce")
    s["month"] = pd.to_numeric(s["month"], errors="coerce")

    if years:
        s = s[s["year"].isin(years)]

    # ── totals ──────────────────────────────────────────────────────────────
    totals = (
        s.groupby("cusid")
        .agg(
            total_sales=("altsales", "sum"),
            cusname=("cusname", "first"),
            area=("area", "first"),
            spname=("spname", "first"),
        )
        .reset_index()
    )

    # ── YoY growth (same-month comparison to avoid partial-year bias) ────────
    yoy = _yoy_growth_same_month(s, "altsales", years, "sales")

    # ── avg order interval + order count ─────────────────────────────────────
    # Distinct order-dates per customer (not line-item rows)
    order_dates = (
        s[["cusid", "date"]]
        .dropna(subset=["date"])
        .drop_duplicates()
        .sort_values(["cusid", "date"])
    )
    order_dates["interval"] = order_dates.groupby("cusid")["date"].diff().dt.days

    order_count = (
        order_dates.groupby("cusid")["date"]
        .count()
        .reset_index()
        .rename(columns={"date": "order_count"})
    )
    interval = (
        order_dates.groupby("cusid")["interval"]
        .mean()
        .round(1)
        .reset_index()
        .rename(columns={"interval": "avg_order_interval_days"})
    )

    # ── monthly activity rate ────────────────────────────────────────────────
    total_months = len(years) * 12 if years else 0
    if total_months > 0:
        s["ym"] = (
            s["year"].astype("Int64").astype(str)
            + "-"
            + s["month"].astype("Int64").astype(str).str.zfill(2)
        )
        active = (
            s.groupby("cusid")["ym"]
            .nunique()
            .reset_index()
            .rename(columns={"ym": "active_months"})
        )
        active["monthly_activity_rate"] = (
            active["active_months"] / total_months * 100
        ).round(1)
        activity = active[["cusid", "monthly_activity_rate"]]
    else:
        activity = pd.DataFrame(columns=["cusid", "monthly_activity_rate"])

    # ── merge ────────────────────────────────────────────────────────────────
    result = totals.merge(order_count, on="cusid", how="left")
    if "cusid" in yoy.columns and len(yoy.columns) > 1:
        result = result.merge(yoy, on="cusid", how="left")
    result = result.merge(interval, on="cusid", how="left")
    result = result.merge(activity, on="cusid", how="left")

    return result


def _collection_metrics(
    sales_df: pd.DataFrame,
    collection_df: pd.DataFrame,
    years: list,
) -> pd.DataFrame:
    """Per-customer collection totals, YoY, avg days between, avg days to."""
    if collection_df is None or collection_df.empty:
        return pd.DataFrame(
            columns=["cusid", "total_collection", "coll_event_count",
                     "avg_days_to_collection", "avg_days_between_collections"]
        )

    c = _ensure_date(collection_df, "date")
    c["value"] = pd.to_numeric(c["value"], errors="coerce").fillna(0.0)

    if "year" not in c.columns:
        c["year"] = c["date"].dt.year
    c["year"] = pd.to_numeric(c["year"], errors="coerce")

    if years:
        c = c[c["year"].isin(years)]

    total_coll = (
        c.groupby("cusid")["value"]
        .sum()
        .reset_index()
        .rename(columns={"value": "total_collection"})
    )

    # ── collection event count ───────────────────────────────────────────────
    coll_event_count = (
        c[c["date"].notna()]
        .groupby("cusid")["date"]
        .count()
        .reset_index()
        .rename(columns={"date": "coll_event_count"})
    )

    # ── YoY growth (same-month comparison to avoid partial-year bias) ────────
    yoy_c = _yoy_growth_same_month(c, "value", years, "collection")

    # ── avg_days_between_collections ────────────────────────────────────────
    c_sorted = c[c["date"].notna()].sort_values(["cusid", "date"])
    c_sorted["days_between"] = (
        c_sorted.groupby("cusid")["date"].diff().dt.days
    )
    avg_between = (
        c_sorted.groupby("cusid")["days_between"]
        .mean()
        .round(1)
        .reset_index()
        .rename(columns={"days_between": "avg_days_between_collections"})
    )

    # ── avg_days_to_collection ───────────────────────────────────────────────
    avg_days_to = _compute_avg_days_to_collection(sales_df, c, years)

    # ── merge ────────────────────────────────────────────────────────────────
    result = total_coll.merge(coll_event_count, on="cusid", how="left")
    if "cusid" in yoy_c.columns and len(yoy_c.columns) > 1:
        result = result.merge(yoy_c, on="cusid", how="left")
    result = result.merge(avg_between, on="cusid", how="left")
    result = result.merge(avg_days_to, on="cusid", how="left")

    return result


def _compute_avg_days_to_collection(
    sales_df: pd.DataFrame,
    collection_df: pd.DataFrame,
    years: list,
) -> pd.DataFrame:
    """Lightweight reimplementation of the CP logic — no returns data needed."""
    if sales_df is None or sales_df.empty:
        return pd.DataFrame(columns=["cusid", "avg_days_to_collection"])

    s = _ensure_date(sales_df, "date")
    s["altsales"] = pd.to_numeric(s["altsales"], errors="coerce").fillna(0.0)
    if "year" not in s.columns:
        s["year"] = s["date"].dt.year
    if years:
        s = s[s["year"].isin(years)]
    if s.empty:
        return pd.DataFrame(columns=["cusid", "avg_days_to_collection"])

    sales_events = s[["cusid", "date", "altsales"]].copy()
    sales_events["type"] = "sale"
    coll_events = collection_df[["cusid", "date"]].copy()
    coll_events["altsales"] = 0.0
    coll_events["type"] = "collection"

    combined = (
        pd.concat([sales_events, coll_events], ignore_index=True)
        .sort_values(["cusid", "date"])
    )

    last_sale = {}
    total_days = defaultdict(float)
    count = defaultdict(int)

    for _, row in combined.iterrows():
        cid = row["cusid"]
        if row["type"] == "sale":
            last_sale[cid] = row["date"]
        elif row["type"] == "collection" and cid in last_sale:
            diff = (row["date"] - last_sale[cid]).days
            if diff >= 0:
                total_days[cid] += diff
                count[cid] += 1

    rows = [
        {"cusid": cid, "avg_days_to_collection": round(total_days[cid] / count[cid], 1)}
        for cid in count
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["cusid", "avg_days_to_collection"])


def _ar_balance(ar_df: pd.DataFrame) -> pd.DataFrame:
    """Sum xprime per customer — debit-positive = customer owes money."""
    if ar_df is None or ar_df.empty:
        return pd.DataFrame(columns=["cusid", "current_balance"])

    a = ar_df.copy()
    a["xprime"] = pd.to_numeric(a["xprime"], errors="coerce").fillna(0.0)
    a = a.rename(columns={"xsub": "cusid"})

    balance = (
        a.groupby("cusid")["xprime"]
        .sum()
        .round(2)
        .reset_index()
        .rename(columns={"xprime": "current_balance"})
    )
    return balance


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=86400)
def build_customer_marketing_table(
    sales_df: pd.DataFrame,
    collection_df: pd.DataFrame,
    ar_df: pd.DataFrame,
    selected_years: tuple,
) -> pd.DataFrame:
    """Assemble full per-customer marketing metrics table."""
    years = list(selected_years) if selected_years else []

    sales_part = _sales_metrics(sales_df, years)
    coll_part = _collection_metrics(sales_df, collection_df, years)
    bal_part = _ar_balance(ar_df)

    result = sales_part.merge(coll_part, on="cusid", how="outer")
    result = result.merge(bal_part, on="cusid", how="left")

    # Fill missing name/area from collection side
    if "cusname" not in result.columns:
        result["cusname"] = np.nan
    if not result["cusname"].notna().all() and collection_df is not None and not collection_df.empty:
        cus_names = (
            collection_df[["cusid", "cusname"]]
            .dropna()
            .drop_duplicates("cusid")
            .set_index("cusid")["cusname"]
        )
        mask = result["cusname"].isna()
        result.loc[mask, "cusname"] = result.loc[mask, "cusid"].map(cus_names)

    # Friendly column order (helper count cols kept for view-level formatting)
    ordered = [
        "cusid", "cusname", "area", "spname",
        "total_sales", "total_collection",
        "yoy_sales_growth_pct", "yoy_collection_growth_pct",
        "avg_days_to_collection", "avg_days_between_collections",
        "avg_order_interval_days", "monthly_activity_rate",
        "current_balance",
        # helper columns (used for display context, not shown to user)
        "order_count", "coll_event_count",
    ]
    present = [c for c in ordered if c in result.columns]
    extra = [c for c in result.columns if c not in ordered]
    result = result[present + extra]

    result = result.sort_values("total_sales", ascending=False, na_position="last").reset_index(drop=True)
    return result
