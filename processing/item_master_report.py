# processing/item_master_report.py
"""
Item Master Report — admin-only, Usage Stats -> "📦 Item Master Report".
Every caitem row for 100001/100009/100000, related to current stock (100001
and 100009 combined via xdrawing, mirroring final_items_view's own logic but
without its xgitem whitelist — this report deliberately shows every product,
not a curated subset), every opspprc price tier, and 5-year IP-- (import)
purchase frequency. See core/queries.py::get_item_master_report for the SQL
and its own reasoning.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.analytics import Analytics


@st.cache_data(show_spinner=False, ttl=900)
def load_item_master_report() -> pd.DataFrame:
    df = Analytics("item_master_report", zid="100001", filters={}).data
    return df if df is not None else pd.DataFrame()


def apply_filters(
    df: pd.DataFrame,
    zids: list | None = None,
    xgitems: list | None = None,
    xabcs: list | None = None,
    xdrawings: list | None = None,
) -> pd.DataFrame:
    """All filters are simple isin() narrows — empty selection means no filter."""
    d = df
    if zids:
        d = d[d["zid"].astype(str).isin([str(z) for z in zids])]
    if xgitems:
        d = d[d["xgitem"].isin(xgitems)]
    if xabcs:
        d = d[d["xabc"].isin(xabcs)]
    if xdrawings:
        d = d[d["xdrawing"].isin(xdrawings)]
    return d
