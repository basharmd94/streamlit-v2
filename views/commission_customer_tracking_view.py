# views/commission_customer_tracking_view.py
# UI for "Customer Tracking" — two independent, manager-curated customer
# watchlists (Collection / Returns), added 2026-10-10 per explicit ask.
# Not a payout mechanism, so it doesn't use the commission_campaigns table
# at all — just a JSON watchlist (processing/customer_tracking.py), same
# convention as Product Tracking's commission_product_watchlist.json.
#
# Setup (admin Commissions page, read_only=False): two comma-separated-code
# text inputs, one per watchlist, admin-only — mirrors Product Tracking's
# exact setup/results split (see views/commissions.py::_sections).
#
# Results (Target Management -> Commission Results, read_only=True):
#   - Collection Watchlist: Customer Score (same engine as Marketing ->
#     Customer Scoring) + Latest Sale & Collection (same engine as
#     Collection Analysis -> Salesman Due), both filtered to the watchlist,
#     then a customer picker showing that one customer's AR ledger (same
#     "AR Customer Ledger" pattern already used in Collection Analysis's
#     own customer-ledger drill-down).
#   - Returns Watchlist: has each watched customer returned anything THIS
#     MONTH — a summary row per customer (Yes/No + count + amount) so a
#     customer with zero returns is still visible, plus the actual return
#     line detail for whoever did.

from __future__ import annotations

import pandas as pd
import streamlit as st

from processing import common, customer_tracking as ctk
from processing import marketing as mkt
from processing import salesman_due as sd
from views.marketing import _load_ar_balance, _load_cacus

# Target Management/Commissions only ever runs against these 3 businesses
# (100009 has no sales team, 100007 is a separate e-commerce catalog, not
# part of this feature) — same mapping convention as
# commission_customer_best_performer_view.py's own _PROJ_BY_ZID.
_PROJ_BY_ZID = {"100001": "GULSHAN TRADING", "100000": "GI Corporation", "100005": "Zepto Chemicals"}


# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_sales_year(zid: str, year: int) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("sales", zid=zid, filters={"year": [year]}).data
    if df is None or df.empty:
        return pd.DataFrame()
    (df,) = common.data_copy_add_columns(df.copy())
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def _load_collection_year(zid: str, year: int) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("collection", zid=zid, filters={"year": [year]}).data
    return df if df is not None else pd.DataFrame()


def _score_customers(zid: str, year: int) -> pd.DataFrame:
    """Customer Score via the exact same build_customer_marketing_table
    engine Marketing -> Customer Scoring uses — current year only, no
    admin-configurable window (this isn't a payout, just a status readout
    for whoever's on the watchlist)."""
    sales_df = _load_sales_year(zid, year)
    collection_df = _load_collection_year(zid, year)
    ar_df = _load_ar_balance(zid, _PROJ_BY_ZID.get(str(zid), ""))
    cacus_df = _load_cacus(zid)
    if sales_df.empty:
        return pd.DataFrame()
    return mkt.build_customer_marketing_table(
        sales_df=sales_df, collection_df=collection_df, ar_df=ar_df,
        selected_years=(year,), cacus_df=cacus_df if not cacus_df.empty else None,
    )


@st.cache_data(ttl=3600, show_spinner="Building Latest Sale & Collection report…")
def _load_latest_sale_collection(zid: str) -> pd.DataFrame:
    """Same build_salesman_due_reports pipeline as Collection Analysis ->
    Salesman Due -> Latest Sale & Collection (views/collection.py::
    _load_salesman_due_reports) — reproduced here (can't import a private
    `_`-prefixed function cross-module) so the watchlist filters this
    feature's own copy of the identical computation."""
    from core.analytics import Analytics
    project = _PROJ_BY_ZID.get(str(zid), "")
    ar_df = Analytics("ar_due_ledger", zid=zid, project=project, filters={}).data
    if ar_df is None or ar_df.empty:
        return pd.DataFrame()
    cacus_df = Analytics("cacus_master", zid=zid, filters={}).data
    if cacus_df is None:
        cacus_df = pd.DataFrame(columns=["cusid", "cusname", "xcity", "xstate"])
    prmst_df = Analytics("prmst_simple", zid=zid, filters={}).data
    if prmst_df is None:
        prmst_df = pd.DataFrame(columns=["spid", "spname"])
    glpmt_df = Analytics("glpmt", zid=zid, filters={}).data
    if glpmt_df is None:
        glpmt_df = pd.DataFrame()
    promise_df = Analytics("cus_delivery_payment_promise", zid=zid, filters={}).data
    if promise_df is None:
        promise_df = pd.DataFrame()
    market_split = str(zid) in ("100000", "100001")
    reports = sd.build_salesman_due_reports(
        ar_df, cacus_df, prmst_df, market_split, glpmt_df=glpmt_df, promise_df=promise_df
    )
    return reports.get("report_df", pd.DataFrame())


@st.cache_data(show_spinner=False, ttl=3600)
def _load_ar_ledger_raw(zid: str) -> pd.DataFrame:
    """mv_ar_vouchers rows (cusid/date/value directly, no renaming needed) —
    the exact same Analytics("ar", ...) source Collection Analysis's own
    "Customer Ledger" -> AR Customer Ledger section uses
    (views/collection.py::_render_cust_ledger_body). NOT "ar_due_ledger" —
    that query's raw columns are xsub/xdate/xprime (needs salesman_due.
    prep_ar_ledger to even get a running balance), a different shape that
    doesn't have a plain cusid/date/value to filter on directly."""
    from core.analytics import Analytics
    df = Analytics("ar", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_returns_month(zid: str, year: int, month: int) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("return", zid=zid, filters={"year": [year], "month": [month]}).data
    return df if df is not None else pd.DataFrame()


# ── Setup (admin Commissions page) ──────────────────────────────────────────

def _render_setup(zid: str, key_suffix: str) -> None:
    st.caption(
        "Enter customer codes as a comma-separated list (e.g. `CUS-005596, CUS-008441`) for "
        "each watchlist. Results show in Target Management → 💰 Commission Results → "
        "👀 Customer Tracking."
    )
    if st.session_state.get("user_role") != "admin":
        st.info("Setup is admin-only.")
        return

    watchlist = ctk.load_watchlist(zid)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📲 Collection Watchlist**")
        st.caption(
            "Customers to keep an eye on for payment — results show their Customer Score, "
            "Latest Sale & Collection, and a full AR ledger drill-down."
        )
        text = st.text_area(
            "Customer codes (comma-separated)",
            value=", ".join(watchlist["collection"]),
            key=f"ct_coll_input{key_suffix}", height=120,
        )
        if st.button("💾 Save Collection Watchlist", key=f"ct_coll_save{key_suffix}"):
            codes = ctk.parse_codes(text)
            ctk.save_group(zid, "collection", codes)
            st.success(f"Saved {len(codes)} customer(s) to the Collection watchlist.")
            st.rerun()

    with col2:
        st.markdown("**↩️ Returns Watchlist**")
        st.caption(
            "Customers flagged by your manager as prone to return products — results show "
            "whether each one returned anything this month."
        )
        text2 = st.text_area(
            "Customer codes (comma-separated)",
            value=", ".join(watchlist["returns"]),
            key=f"ct_ret_input{key_suffix}", height=120,
        )
        if st.button("💾 Save Returns Watchlist", key=f"ct_ret_save{key_suffix}"):
            codes2 = ctk.parse_codes(text2)
            ctk.save_group(zid, "returns", codes2)
            st.success(f"Saved {len(codes2)} customer(s) to the Returns watchlist.")
            st.rerun()


# ── Results (Target Management -> Commission Results) ──────────────────────

def _render_collection_results(zid: str, codes: list, key_suffix: str) -> None:
    st.subheader("📲 Collection Watchlist")
    if not codes:
        st.info("No customers on the Collection watchlist yet. Ask an admin to add some via the Commissions page.")
        return

    today = pd.Timestamp.today().normalize()

    st.markdown("**Customer Score**")
    with st.spinner("Loading customer score…"):
        score_df = _score_customers(str(zid), int(today.year))
    score_wl = (
        score_df[score_df["cusid"].astype(str).isin(codes)].copy()
        if not score_df.empty else pd.DataFrame()
    )
    if score_wl.empty:
        st.caption(f"No {today.year} scoring data found for these customers.")
    else:
        show_cols = [c for c in ["cusid", "cusname", "composite_score", "current_balance"] if c in score_wl.columns]
        disp = score_wl[show_cols].rename(columns={
            "cusid": "Customer Code", "cusname": "Customer Name",
            "composite_score": "Customer Score", "current_balance": "Current Balance",
        })
        if "Customer Score" in disp.columns:
            disp = disp.sort_values("Customer Score", ascending=False)
        st.dataframe(disp, use_container_width=True)

    st.divider()
    st.markdown("**Latest Sale & Collection** (watchlist only)")
    with st.spinner("Loading Latest Sale & Collection…"):
        report_df = _load_latest_sale_collection(str(zid))
    report_wl = (
        report_df[report_df["Customer Code"].astype(str).isin(codes)].copy()
        if not report_df.empty else pd.DataFrame()
    )
    if report_wl.empty:
        st.caption(
            "No AR activity found for these customers here — near-zero-balance customers are "
            "excluded from this table, same as Collection Analysis → Salesman Due."
        )
    else:
        st.caption(f"Showing {report_wl['Customer Code'].nunique()} of {len(codes)} watchlist customers with AR activity.")
        st.dataframe(report_wl, use_container_width=True)

    st.divider()
    st.markdown("**Customer Ledger**")
    cacus_df = _load_cacus(str(zid))
    name_lookup = cacus_df.set_index("cusid")["cusname"].to_dict() if not cacus_df.empty else {}
    options = [f"{c} - {name_lookup.get(c, '')}" for c in codes]
    sel = st.selectbox("Select customer", options, key=f"ct_coll_ledger_sel{key_suffix}")
    sel_code = sel.split(" - ")[0]
    with st.spinner("Loading ledger…"):
        ar_raw = _load_ar_ledger_raw(str(zid))
    ledger = (
        ar_raw[ar_raw["cusid"].astype(str) == sel_code].sort_values("date").copy()
        if not ar_raw.empty else pd.DataFrame()
    )
    if ledger.empty:
        st.info(f"No ledger entries found for {sel_code}.")
    else:
        ledger["ending_balance"] = ledger["value"].cumsum()
        st.dataframe(ledger, use_container_width=True)


def _render_returns_results(zid: str, codes: list, key_suffix: str) -> None:
    st.subheader("↩️ Returns Watchlist")
    st.caption(
        "Customers flagged by your manager as prone to return products — tracking their "
        "returns here confirms whether this month's returns from them were managed properly."
    )
    if not codes:
        st.info("No customers on the Returns watchlist yet. Ask an admin to add some via the Commissions page.")
        return

    today = pd.Timestamp.today().normalize()
    cacus_df = _load_cacus(str(zid))
    name_lookup = cacus_df.set_index("cusid")["cusname"].to_dict() if not cacus_df.empty else {}

    with st.spinner("Loading this month's returns…"):
        returns_df = _load_returns_month(str(zid), int(today.year), int(today.month))
    wl_returns = (
        returns_df[returns_df["cusid"].astype(str).isin(codes)].copy()
        if not returns_df.empty else pd.DataFrame()
    )

    rows = []
    for code in codes:
        sub = wl_returns[wl_returns["cusid"].astype(str) == code] if not wl_returns.empty else pd.DataFrame()
        rows.append({
            "Customer Code": code,
            "Customer Name": name_lookup.get(code, "—"),
            "Returned This Month": "Yes" if len(sub) else "No",
            "Return Count": int(len(sub)),
            "Return Amount": float(sub["treturnamt"].sum()) if len(sub) else 0.0,
        })
    summary_df = pd.DataFrame(rows).sort_values(
        ["Returned This Month", "Return Amount"], ascending=[False, False]
    )
    st.markdown(f"**{today:%B %Y} Summary**")
    st.dataframe(summary_df, use_container_width=True)

    st.divider()
    st.markdown("**Return Line Detail**")
    if wl_returns.empty:
        st.info("None of the watchlist customers have a return this month.")
        return
    detail_cols = [c for c in [
        "date", "revoucher", "cusid", "cusname", "itemcode", "itemname", "returnqty", "treturnamt", "reason",
    ] if c in wl_returns.columns]
    detail = wl_returns[detail_cols].rename(columns={
        "date": "Date", "revoucher": "Return Voucher", "cusid": "Customer Code", "cusname": "Customer Name",
        "itemcode": "Item Code", "itemname": "Item Name", "returnqty": "Qty",
        "treturnamt": "Amount", "reason": "Reason",
    }).sort_values("Date", ascending=False)
    st.dataframe(detail, use_container_width=True)


def _render_results(zid: str, key_suffix: str) -> None:
    watchlist = ctk.load_watchlist(zid)
    sub = st.radio(
        "Tracking Type", ["📲 Collection Watchlist", "↩️ Returns Watchlist"],
        key=f"ct_sub{key_suffix}", horizontal=True,
    )
    st.divider()
    if sub == "📲 Collection Watchlist":
        _render_collection_results(zid, watchlist["collection"], key_suffix)
    else:
        _render_returns_results(zid, watchlist["returns"], key_suffix)


def render(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    st.subheader("👀 Customer Tracking")
    if not read_only:
        _render_setup(str(zid), key_suffix)
        return
    _render_results(str(zid), key_suffix)
