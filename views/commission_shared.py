# views/commission_shared.py
# Shared "one consolidated Before/After table" renderer — used by Product
# Tracking (views/commissions.py) and the B.1/3/4 campaign's own product
# Before/After table (views/commission_campaigns_view.py). Both feed the
# same shape (see processing/commissions.py::build_product_comparisons and
# processing/commission_campaigns.py::compute_campaign_product_before_after).
#
# Redesigned 2026-09-20 from one table PER product into a single table
# (products as rows, Before / Before Avg (Prorated) / After / Change as
# columns, for ONE dropdown-picked metric at a time) — explicit ask:
# "instead of looking at five tables, I would like it to be in one table...
# the before after can only be in the column header." Total Sales/Qty Sold
# (gross) dropped from the picker per the same ask ("total revenue is not
# very necessary") — only the three metrics below are selectable.

from __future__ import annotations

import pandas as pd
import streamlit as st

_METRIC_OPTIONS = {
    "Net Units Sold": ("net_qty", False),
    "Returns": ("qty_returned", False),
    "Net Revenue": ("net_revenue", True),
}


def _fmt(v, is_revenue: bool, signed: bool = False) -> str:
    if v is None or pd.isna(v):
        return "—"
    prefix = "৳" if is_revenue else ""
    return f"{prefix}{v:+,.0f}" if signed else f"{prefix}{v:,.0f}"


def render_before_after_table(rows: list, key_suffix: str = "") -> None:
    """`rows`: list of per-product dicts (build_product_comparisons /
    compute_campaign_product_before_after's shared output shape) —
    {itemcode, itemname, itemgroup, current_stock, before_days, after_days,
    before, before_avg_prorated, after}, each of the three metric dicts
    holding qty_sold/sales_revenue/qty_returned/net_qty/net_revenue.

    Renders ONE table: one row per product, Before / Before Avg (Prorated) /
    After / Change columns for whichever metric a selectbox above picks."""
    if not rows:
        st.info("No data.")
        return

    metric_label = st.selectbox("Metric", list(_METRIC_OPTIONS.keys()), key=f"cmn_metric{key_suffix}")
    metric_key, is_revenue = _METRIC_OPTIONS[metric_label]

    table = []
    for r in rows:
        before_v = r["before"][metric_key]
        avg_v = r["before_avg_prorated"][metric_key]
        after_v = r["after"][metric_key]
        change_v = after_v - avg_v
        table.append({
            "Item Code": r["itemcode"],
            "Item Name": r["itemname"] or "(name unavailable)",
            "Item Group": r["itemgroup"] or "—",
            "Current Stock": _fmt(r.get("current_stock"), is_revenue=False),
            "Before": _fmt(before_v, is_revenue),
            "Before Avg (Prorated)": _fmt(avg_v, is_revenue),
            "After": _fmt(after_v, is_revenue),
            "Change": _fmt(change_v, is_revenue, signed=True),
        })

    st.caption(
        "Before Avg (Prorated) scales Before to match how many days After has actually "
        "covered so far — a fair baseline since Before and After are usually different-"
        "length windows. Change = After − Prorated Avg."
    )
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
