# views/glpmt_shared.py
# Shared "App Collections" panel — payments salesmen enter directly into the
# mobile Ordering app (table: glpmt), staged pending reconciliation into the
# real GL ledger. Same panel is mounted in both Collection Analysis and
# Target Management, so this module is the single source of truth for it.
#
# Redesigned (2026-09-16) per explicit ask, once it became clear the point
# was never "list every payment promise a salesman logged" but to verify
# the whole story: a delivery (DO) happened, the salesman logged the
# customer's promised payment date on their behalf, and (hopefully) a real
# collection (RCT) came in afterward. Now one row per customer -- their
# latest DO, latest RCT, latest glpmt entry, and current AR balance -- see
# processing/glpmt_reconciliation.py for the actual reconciliation logic.

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.analytics import Analytics
from processing import common
from processing.glpmt_reconciliation import build_glpmt_reconciliation


@st.cache_data(show_spinner=False, ttl=300)
def _load_glpmt(zid: str) -> pd.DataFrame:
    df = Analytics("glpmt", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_sales_full(zid: str) -> pd.DataFrame:
    """Full sales history (no year/month filter) -- needed to find each
    customer's single latest DO regardless of when it happened, same
    "load full history, slice in memory" pattern used elsewhere in this
    app for similar all-time lookups."""
    df = Analytics("sales", zid=zid, filters={}).data
    if df is None or df.empty:
        return pd.DataFrame()
    (df,) = common.data_copy_add_columns(df)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def _load_collection_full(zid: str) -> pd.DataFrame:
    """Full collection-voucher history (mv_collection_vouchers) -- needed
    to find each customer's single latest RCT regardless of when it
    happened."""
    df = Analytics("collection", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_ar_ledger_full(zid: str) -> pd.DataFrame:
    """Full AR ledger (mv_ar_transactions) -- feeds the same Current
    Balance calc Collection Analysis -> Salesman Due already uses
    (processing/salesman_due.py), reused unchanged here rather than
    recomputed."""
    df = Analytics("ar_due_ledger", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


def render_glpmt_panel(zid: str, key_suffix: str = "") -> None:
    """Filters: Salesman (emp code), Customer, Date of Entry (range) --
    applied to the RAW glpmt entries first, same as before the redesign.
    Always sorted by date of entry, latest first — per spec, not
    user-toggleable."""
    st.subheader("📲 App Collections")
    st.caption(
        "Did the whole story complete? A delivery (DO) happened, the salesman logged the "
        "customer's promised payment date on their behalf, and — hopefully — a real collection "
        "(RCT) came in after. One row per customer: their latest DO, latest RCT, latest promised "
        "payment, and current balance, side by side."
    )

    df = _load_glpmt(str(zid))
    if df.empty:
        st.info("No app-entered payments found for this business.")
        return

    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["paydate"]     = pd.to_datetime(df["paydate"], errors="coerce")

    # ── Filters ────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        sp_opts = sorted(df["spid"].dropna().unique().tolist())
        sp_name_map = df.drop_duplicates("spid").set_index("spid")["spname"].to_dict()
        sel_sp = st.multiselect(
            "Salesman (Emp Code)",
            sp_opts,
            format_func=lambda x: f"{x} — {sp_name_map.get(x, '')}",
            key=f"glpmt_sp{key_suffix}",
        )
    with f2:
        cus_opts = sorted(df["cusid"].dropna().unique().tolist())
        cus_name_map = df.drop_duplicates("cusid").set_index("cusid")["cusname"].to_dict()
        sel_cus = st.multiselect(
            "Customer",
            cus_opts,
            format_func=lambda x: f"{x} — {cus_name_map.get(x, '')}",
            key=f"glpmt_cus{key_suffix}",
        )
    with f3:
        entry_dates = df["entry_time"].dt.date.dropna()
        date_range = None
        if not entry_dates.empty:
            date_range = st.date_input(
                "Date of Entry (range)",
                value=(entry_dates.min(), entry_dates.max()),
                key=f"glpmt_daterange{key_suffix}",
            )

    filtered = df.copy()
    if sel_sp:
        filtered = filtered[filtered["spid"].isin(sel_sp)]
    if sel_cus:
        filtered = filtered[filtered["cusid"].isin(sel_cus)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["entry_time"].dt.date >= start) & (filtered["entry_time"].dt.date <= end)
        ]

    if filtered.empty:
        st.info("No app-entered payments match these filters.")
        return

    sales_df = _load_sales_full(str(zid))
    collection_df = _load_collection_full(str(zid))
    ar_ledger_df = _load_ar_ledger_full(str(zid))

    recon = build_glpmt_reconciliation(filtered, sales_df, collection_df, ar_ledger_df)
    if recon.empty:
        st.info(
            "No customer here still has a real balance — every match either fell below the "
            "near-zero threshold or has no matching AR ledger row at all."
        )
        return

    # Sorted by date of entry, latest first — non-negotiable per spec.
    recon = recon.sort_values("entry_time", ascending=False).reset_index(drop=True)

    st.caption(
        f"**{len(recon):,}** customer(s) — customers whose AR balance is already near zero are "
        "left out entirely (nothing left to chase)."
    )

    show = recon.rename(columns={
        "cusid":       "Cust Code",
        "cusname":     "Customer",
        "spid":        "Emp Code",
        "spname":      "Salesman",
        "do_date":     "Latest DO",
        "do_number":   "DO Number",
        "do_amount":   "DO Amount",
        "rct_date":    "Latest RCT",
        "rct_number":  "RCT Number",
        "rct_amount":  "RCT Amount",
        "balance":     "Customer Balance",
        "paydate":     "Latest Payment Date",
        "entry_time":  "Entered Date",
        "payamt":      "Amount Entered",
        "paytype":     "Type",
        "bankdetail":  "Bank Detail",
        "paystatus":   "Status",
        "remarks":     "Remark",
    })
    show_cols = [
        "Cust Code", "Customer", "Emp Code", "Salesman",
        "Latest DO", "DO Number", "DO Amount",
        "Latest RCT", "RCT Number", "RCT Amount", "RCT After DO",
        "Customer Balance",
        "Latest Payment Date", "Payment After DO",
        "Entered Date", "Amount Entered", "Type", "Bank Detail", "Status", "Remark",
    ]
    show = show[[c for c in show_cols if c in show.columns]]

    st.dataframe(
        show,
        column_config={
            "Latest DO":           st.column_config.DateColumn("Latest DO", format="YYYY-MM-DD"),
            "DO Amount":           st.column_config.NumberColumn("DO Amount", format="%.2f"),
            "Latest RCT":          st.column_config.DateColumn("Latest RCT", format="YYYY-MM-DD"),
            "RCT Amount":          st.column_config.NumberColumn("RCT Amount", format="%.2f"),
            "Customer Balance":    st.column_config.NumberColumn("Customer Balance", format="%.2f"),
            "Latest Payment Date": st.column_config.DateColumn("Latest Payment Date", format="YYYY-MM-DD"),
            "Entered Date":        st.column_config.DatetimeColumn("Entered Date", format="YYYY-MM-DD HH:mm"),
            "Amount Entered":      st.column_config.NumberColumn("Amount Entered", format="%.2f"),
        },
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "✅ / ⚠️ = the date genuinely came after the DO date / did not (or was at/before it). "
        "Blank = nothing to compare yet (one side of the check has no date on file)."
    )

    st.download_button(
        "⬇ Download CSV",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name=f"app_collections_{zid}.csv",
        mime="text/csv",
        key=f"glpmt_dl{key_suffix}",
    )
