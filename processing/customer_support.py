# processing/customer_support.py
# Data helpers for the Customer Support (CRM) view.
# All balance logic mirrors prep_ar_ledger in salesman_due.py exactly so that
# the customer ledger balance matches the Salesman Due trickle-down.

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from processing.salesman_due import prep_ar_ledger

# ZIDs that have customer-facing AR (project per ZID matches Salesman Due)
_ZID_PROJECT: dict[str, str] = {
    "100001": "GULSHAN TRADING",
    "100000": "GI Corporation",
    "100005": "Zepto Chemicals",
}

# Voucher-prefix → human transaction type (first match wins)
_TXN_RULES: list[tuple[tuple[str, ...], str]] = [
    (("INOP",),                                                "Delivery"),
    (("RCT", "BRCT", "CRCT", "BPAY", "CPAY", "PAY", "CHQ",
      "BTJV"),                                                 "Collection"),
    (("SRT", "SRJV", "IMSA"),                                 "Return"),
    (("ADJV", "JV", "STJV", "TR"),                            "Adjustment"),
]


def classify_txn_type(voucher: str) -> str:
    v = str(voucher).upper()
    for prefixes, label in _TXN_RULES:
        if any(v.startswith(p) for p in prefixes):
            return label
    return "Other"


# ─── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=1800)
def load_all_ar_ledgers() -> pd.DataFrame:
    """Load AR due ledger for every ZID, run prep_ar_ledger per ZID, combine.

    Uses the same project filter as Salesman Due so the running_balance column
    is identical to the Salesman Due trickle-down for each customer.
    """
    from core.analytics import Analytics

    dfs: list[pd.DataFrame] = []
    for zid, project in _ZID_PROJECT.items():
        df = Analytics("ar_due_ledger", zid=zid, project=project, filters={}).data
        if df is None or df.empty:
            continue
        df = df.copy()
        df["zid"] = str(zid)
        df_clean = prep_ar_ledger(df)
        dfs.append(df_clean)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=1800)
def load_all_sales_7day() -> pd.DataFrame:
    """Sales line items from the last 7 days across all ZIDs (from MV).

    Used to build the DO-detail table in the Customer Support expander.
    """
    from core.analytics import Analytics

    dfs: list[pd.DataFrame] = []
    for zid in _ZID_PROJECT:
        df = Analytics("sales_7day", zid=zid, filters={}).data
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["zid"] = str(zid)
        dfs.append(tmp)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    out["date"]  = pd.to_datetime(out["date"],  errors="coerce")
    out["cusid"] = out["cusid"].astype(str)
    return out


@st.cache_data(show_spinner=False, ttl=1800)
def load_all_cacus() -> pd.DataFrame:
    """Customer directory (mobile + whatsapp) for all ZIDs."""
    from core.analytics import Analytics

    dfs: list[pd.DataFrame] = []
    for zid in _ZID_PROJECT:
        df = Analytics("cacus_directory", zid=zid, filters={}).data
        if df is None or df.empty:
            continue
        df = df.copy()
        df["zid"] = str(zid)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ─── 7-day activity feed ──────────────────────────────────────────────────────

def build_7day_feed(
    ar_df: pd.DataFrame,
    cacus_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per (zid, customer, voucher) in the last 7 days.

    Aggregates GL lines to voucher level so each CRM row maps to one
    transaction the specialist will tick off.
    """
    if ar_df.empty:
        return pd.DataFrame()

    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=13)  # inclusive last 14 days

    ar = ar_df.copy()
    ar["xdate"] = pd.to_datetime(ar["xdate"], errors="coerce")
    ar["xprime"] = pd.to_numeric(ar["xprime"], errors="coerce").fillna(0.0)
    ar["zid"] = ar["zid"].astype(str)
    ar["xsub"] = ar["xsub"].astype(str)

    recent = ar[ar["xdate"] >= cutoff].copy()
    if recent.empty:
        return pd.DataFrame()

    recent["txn_type"] = recent["xvoucher"].apply(classify_txn_type)

    # Aggregate GL lines → voucher level (one CRM row per voucher per customer)
    agg: dict[str, object] = {
        "xdate":         ("xdate", "first"),
        "customer_name": ("customer_name", "first"),
        "xcity":         ("xcity", "first"),
        "salesman_name": ("salesman_name", "first"),
        "txn_type":      ("txn_type", "first"),
        "xprime":        ("xprime", "sum"),
    }
    feed = (
        recent
        .groupby(["zid", "xsub", "xvoucher"], as_index=False)
        .agg(**agg)
    )

    # Join contact info (mobile + whatsapp) from cacus
    if not cacus_df.empty:
        contact = (
            cacus_df[["zid", "cusid", "cusmobile", "whatsapp"]]
            .copy()
            .assign(cusid=lambda d: d["cusid"].astype(str))
            .rename(columns={"cusid": "xsub"})
        )
        feed = feed.merge(contact, on=["zid", "xsub"], how="left")

    feed = feed.sort_values(
        ["xdate", "zid", "xsub"], ascending=[False, True, True]
    ).reset_index(drop=True)

    return feed


# ─── Customer 6-month ledger ──────────────────────────────────────────────────

def build_customer_ledger(
    ar_df: pd.DataFrame,
    zid: str,
    cusid: str,
) -> pd.DataFrame:
    """Full GL-line ledger for one (zid, customer).

    running_balance was computed from ALL history by prep_ar_ledger, so the
    final row's balance matches the Salesman Due trickle-down exactly.
    Display is limited to last 6 months in the view layer.
    """
    if ar_df.empty:
        return pd.DataFrame()

    mask = (
        (ar_df["zid"].astype(str) == str(zid))
        & (ar_df["xsub"].astype(str) == str(cusid))
    )
    df = ar_df[mask].copy()
    if df.empty:
        return df

    df["xdate"] = pd.to_datetime(df["xdate"], errors="coerce")
    df["txn_type"] = df["xvoucher"].apply(classify_txn_type)
    df = df.sort_values(["xdate", "xrow", "xvoucher"]).reset_index(drop=True)
    return df


# ─── Latest Sales & Collection (same pipeline as Salesman Due) ───────────────

def build_latest_sc_for_zid(
    ar_df_cleaned: pd.DataFrame,
    zid: str,
    cacus_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build Latest Sales & Collection for one ZID using the identical pipeline
    as Collection Analysis → Salesman Due → Latest sale & collection tab.

    ar_df_cleaned must already be the output of load_all_ar_ledgers() (i.e.
    prep_ar_ledger has been applied per ZID and running_balance is present).
    This guarantees balances, latest sale, and latest collection all match
    the Salesman Due report exactly.
    """
    from processing.salesman_due import build_latest_sale_collection_report

    zid_df = ar_df_cleaned[ar_df_cleaned["zid"].astype(str) == str(zid)].copy()
    if zid_df.empty:
        return pd.DataFrame()

    report = build_latest_sale_collection_report(zid_df)
    if report.empty:
        return pd.DataFrame()

    # Salesman name from the MV-joined AR ledger (mv_ar_transactions already joins prmst)
    sp_lookup = (
        zid_df.dropna(subset=["xsp", "salesman_name"])
        [["xsp", "salesman_name"]]
        .drop_duplicates("xsp", keep="last")
        .set_index("xsp")["salesman_name"]
        .to_dict()
    )
    report["salesman_name"] = report["Salesman Code"].map(sp_lookup).fillna("")

    # Customer name + mobile from cacus
    zid_cacus = (
        cacus_df[cacus_df["zid"].astype(str) == str(zid)]
        [["cusid", "cusname", "cusmobile"]]
        .copy()
        .assign(cusid=lambda d: d["cusid"].astype(str))
        .rename(columns={"cusid": "Customer Code", "cusname": "customer_name"})
    )
    report["Customer Code"] = report["Customer Code"].astype(str)
    report = report.merge(zid_cacus, on="Customer Code", how="left")

    # Days since sale / collection
    today = pd.Timestamp.today().normalize()
    report["last_sale_date"] = pd.to_datetime(report["Sales Date"], errors="coerce")
    report["last_coll_date"] = pd.to_datetime(report["Latest Collection Date"], errors="coerce")
    report["days_since_sale"] = (today - report["last_sale_date"]).dt.days
    report["days_since_coll"] = (today - report["last_coll_date"]).dt.days

    out = report.rename(columns={
        "Customer Code":            "cusid",
        "Salesman Code":            "spid",
        "City":                     "city",
        "Sale Amount":              "last_sale_amount",
        "Latest Collection Amount": "last_coll_amount",
        "Current Balance":          "current_balance",
    })

    keep = [
        "cusid", "customer_name", "cusmobile",
        "spid", "salesman_name", "city",
        "last_sale_date", "last_sale_amount", "days_since_sale",
        "last_coll_date", "last_coll_amount", "days_since_coll",
        "current_balance",
    ]
    return out[[c for c in keep if c in out.columns]].reset_index(drop=True)


# ─── Merged SC table (100001 + 100000) ───────────────────────────────────────

def build_merged_sc_table(
    df_100001: pd.DataFrame,
    df_100000: pd.DataFrame,
) -> pd.DataFrame:
    """Combine SC tables for 100001 and 100000 into one grouped table.

    A customer present in both ZIDs appears as two adjacent rows. Groups are
    sorted by max(days_since_sale) across the group, descending, so the most
    overdue customer cluster floats to the top regardless of which ZID holds
    the longer balance.
    """
    parts: list[pd.DataFrame] = []
    for zid, df in [("100001", df_100001), ("100000", df_100000)]:
        if df is not None and not df.empty:
            tmp = df.copy()
            tmp["zid"] = zid
            parts.append(tmp)

    if not parts:
        return pd.DataFrame()

    merged = pd.concat(parts, ignore_index=True)

    days_col = pd.to_numeric(merged.get("days_since_sale", pd.Series(dtype=float)), errors="coerce").fillna(0)
    merged["_days"] = days_col
    group_rank = merged.groupby("cusid")["_days"].max().rename("_group_rank")
    merged = (
        merged
        .merge(group_rank, on="cusid")
        .sort_values(["_group_rank", "cusid", "zid"], ascending=[False, True, True])
        .drop(columns=["_days", "_group_rank"])
        .reset_index(drop=True)
    )
    return merged
