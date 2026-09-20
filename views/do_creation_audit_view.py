# views/do_creation_audit_view.py
"""
DO Creation Audit — shared panel mounted identically in Customer Data View
("🚦 DO Creation Audit" mode, for SOPs) and Target Management (for sales
managers) — same filters, same four tables, same sort. Edit this module,
not either call site, to change behavior in both places at once (same
convention as views/glpmt_shared.py's App Collections panel).

Scope is whatever ZID is currently active (st.session_state.zid / the zid
passed in) — NOT hardcoded to 100001+100000, unlike Commission Tracking.
opmob (mobile orders) has real activity on 100005 too (confirmed against
real Postgres, if only 63 rows), so this audit follows the page's normal
single-ZID scoping rather than presuming a combined 100001+100000 SOP team.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.analytics import Analytics
from processing import do_creation_audit as dca

_TABLE_SPECS = [
    ("🆕 Pending Order Creation", "🆕", "No SOP action yet on this mobile order."),
    ("📦 Pending DO Creation", "📦", "Order created (COMO), but no delivery order (DO) yet."),
    ("🚚 Partial Fulfillment", "🚚", "A DO exists, but the order isn't fully delivered and nothing further has shipped recently."),
    ("⚠️ Order Never Persisted", "⚠️", "opmob says a COMO was created, but it never actually landed in opord — a system gap, not an SOP miss."),
]


@st.cache_data(show_spinner=False, ttl=300)
def _load_raw(zid: str, cutoff_date: str) -> dict:
    filt = {"cutoff_date": [cutoff_date]}
    return {
        "pending_order": Analytics("do_audit_pending_order", zid=zid, filters=filt).data,
        "pending_do": Analytics("do_audit_pending_do", zid=zid, filters=filt).data,
        "partial": Analytics("do_audit_partial_fulfillment", zid=zid, filters=filt).data,
        "unpersisted": Analytics("do_audit_unpersisted", zid=zid, filters=filt).data,
    }


def render_do_creation_audit_panel(zid: str, key_suffix: str = "") -> None:
    st.subheader("🚦 DO Creation Audit")
    st.caption(
        "Catches orders stuck anywhere in the mobile-order → COMO → DO pipeline — a mobile "
        "order nobody's acted on, a COMO with no delivery order yet, an order that's only "
        "partly shipped, or a COMO that was reported created but never actually saved. "
        "Each table only shows rows stuck longer than a holiday-aware threshold (Friday and "
        "any configured public holiday don't count against the clock) — a Thursday order "
        "won't look overdue until Saturday."
    )

    window_days = st.slider(
        "Time window (days)", min_value=7, max_value=90, value=dca.DEFAULT_WINDOW_DAYS, step=1,
        key=f"doaudit_window{key_suffix}",
    )
    cutoff_date = (pd.Timestamp.today().normalize() - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")

    with st.spinner("Auditing the order pipeline…"):
        raw = _load_raw(str(zid), cutoff_date)

    pending_order = dca.build_pending_order_creation(raw["pending_order"] if raw["pending_order"] is not None else pd.DataFrame())
    pending_do = dca.build_pending_do_creation(raw["pending_do"] if raw["pending_do"] is not None else pd.DataFrame())
    partial = dca.build_partial_fulfillment(raw["partial"] if raw["partial"] is not None else pd.DataFrame())
    unpersisted = dca.build_unpersisted(raw["unpersisted"] if raw["unpersisted"] is not None else pd.DataFrame())

    _render_pending_order_table(pending_order, key_suffix, zid)
    _render_pending_do_table(pending_do, key_suffix, zid)
    _render_partial_table(partial, key_suffix, zid)
    _render_unpersisted_table(unpersisted, key_suffix, zid)


def _download(df: pd.DataFrame, label: str, filename: str, key: str) -> None:
    st.download_button(
        f"⬇ Download {label} CSV", df.to_csv(index=False).encode("utf-8"),
        file_name=filename, mime="text/csv", key=key,
    )


def _render_pending_order_table(df: pd.DataFrame, key_suffix: str, zid: str) -> None:
    st.markdown("### 🆕 Pending Order Creation")
    st.caption("A mobile order came in and no SOP has turned it into an order (COMO) yet.")
    if df.empty:
        st.success("Nothing overdue — every mobile order has been acted on in time.")
        return
    show = df.rename(columns={
        "invoiceno": "Submission", "cusid": "Cust Code", "cusname": "Customer",
        "spid": "Emp Code", "spname": "Salesman", "order_date": "Date",
        "ztime": "Submitted At", "item_count": "Items", "total_qty": "Total Qty",
        "total_value": "Total Value", "hours_pending": "Hours Pending",
    })
    cols = ["Cust Code", "Customer", "Emp Code", "Salesman", "Date", "Submitted At",
            "Items", "Total Qty", "Total Value", "Hours Pending"]
    show = show[[c for c in cols if c in show.columns]]
    st.dataframe(
        show,
        column_config={
            "Submitted At": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Total Value": st.column_config.NumberColumn(format="%.2f"),
            "Hours Pending": st.column_config.NumberColumn(format="%.1f"),
        },
        hide_index=True, width="stretch",
    )
    st.caption(f"**{len(show):,}** submission(s) overdue.")
    _download(show, "Pending Order Creation", f"pending_order_creation_{zid}.csv", f"doaudit_dl1{key_suffix}")


def _render_pending_do_table(df: pd.DataFrame, key_suffix: str, zid: str) -> None:
    st.markdown("### 📦 Pending DO Creation")
    st.caption("The order was created, but stock verification/packaging never produced a DO.")
    if df.empty:
        st.success("Nothing overdue — every created order has a delivery order in progress.")
        return
    show = df.rename(columns={
        "xordernum": "Order (COMO)", "cusid": "Cust Code", "cusname": "Customer",
        "spid": "Emp Code", "spname": "Salesman", "xdate": "Date", "ztime": "Order Created At",
        "order_amount": "Order Amount", "hours_pending": "Hours Pending",
    })
    cols = ["Order (COMO)", "Cust Code", "Customer", "Emp Code", "Salesman",
            "Date", "Order Created At", "Order Amount", "Hours Pending"]
    show = show[[c for c in cols if c in show.columns]]
    st.dataframe(
        show,
        column_config={
            "Order Created At": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Order Amount": st.column_config.NumberColumn(format="%.2f"),
            "Hours Pending": st.column_config.NumberColumn(format="%.1f"),
        },
        hide_index=True, width="stretch",
    )
    st.caption(f"**{len(show):,}** order(s) overdue.")
    _download(show, "Pending DO Creation", f"pending_do_creation_{zid}.csv", f"doaudit_dl2{key_suffix}")


def _render_partial_table(df: pd.DataFrame, key_suffix: str, zid: str) -> None:
    st.markdown("### 🚚 Partial Fulfillment")
    st.caption("At least one DO shipped, but the order isn't fully delivered and nothing further has moved recently.")
    if df.empty:
        st.success("Nothing overdue — every partly-shipped order is still actively moving.")
        return
    show = df.rename(columns={
        "xordernum": "Order (COMO)", "cusid": "Cust Code", "cusname": "Customer",
        "spid": "Emp Code", "spname": "Salesman", "xdate": "Date",
        "ordered": "Qty Ordered", "delivered": "Qty Delivered", "remaining": "Qty Remaining",
        "last_do_ztime": "Last DO At", "hours_pending": "Hours Since Last DO",
    })
    cols = ["Order (COMO)", "Cust Code", "Customer", "Emp Code", "Salesman", "Date",
            "Qty Ordered", "Qty Delivered", "Qty Remaining", "Last DO At", "Hours Since Last DO"]
    show = show[[c for c in cols if c in show.columns]]
    st.dataframe(
        show,
        column_config={
            "Last DO At": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Hours Since Last DO": st.column_config.NumberColumn(format="%.1f"),
        },
        hide_index=True, width="stretch",
    )
    st.caption(f"**{len(show):,}** order(s) overdue.")
    _download(show, "Partial Fulfillment", f"partial_fulfillment_{zid}.csv", f"doaudit_dl3{key_suffix}")


def _render_unpersisted_table(df: pd.DataFrame, key_suffix: str, zid: str) -> None:
    st.markdown("### ⚠️ Order Never Persisted")
    st.caption(
        "opmob shows a real COMO number was generated ('Order Created'), but that order never "
        "actually landed in opord. Nothing an SOP can do to move this forward — it's a system "
        "gap worth reporting, not a missed step."
    )
    if df.empty:
        st.success("None found in this window.")
        return
    show = df.rename(columns={
        "invoiceno": "Submission", "xordernum": "Order (COMO)", "cusid": "Cust Code",
        "cusname": "Customer", "spid": "Emp Code", "spname": "Salesman",
        "xdate": "Date", "ztime": "Submitted At", "item_count": "Items", "total_value": "Total Value",
    })
    cols = ["Order (COMO)", "Cust Code", "Customer", "Emp Code", "Salesman",
            "Date", "Submitted At", "Items", "Total Value"]
    show = show[[c for c in cols if c in show.columns]]
    st.dataframe(
        show,
        column_config={
            "Submitted At": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Total Value": st.column_config.NumberColumn(format="%.2f"),
        },
        hide_index=True, width="stretch",
    )
    st.caption(f"**{len(show):,}** order(s) found.")
    _download(show, "Order Never Persisted", f"order_never_persisted_{zid}.csv", f"doaudit_dl4{key_suffix}")
