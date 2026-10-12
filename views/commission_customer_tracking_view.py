# views/commission_customer_tracking_view.py
# UI for Customer Tracking — two manager-curated customer watchlists
# (Collection / Returns) plus a standalone Promised Payment list.
#
# Originally built 2026-10-10 nested under Commission Results; moved to its
# own direct Target Management radio and redesigned 2026-10-12 per explicit
# spec — see CLAUDE.md's "Customer Tracking" section for the full writeup.
# Not a payout mechanism, so it doesn't use the commission_campaigns table
# at all — just a JSON watchlist (processing/customer_tracking.py), same
# convention as Product Tracking's commission_product_watchlist.json.
#
# Setup (admin Commissions page, read_only=False): two comma-separated-code
# text inputs, one per watchlist, admin-only, still scoped per the sidebar's
# active ZID — unchanged from the original build.
#
# Results (Target Management -> "👀 Customer Tracking", read_only=True,
# called DIRECTLY, no longer nested under the Commission Campaign picker):
# always pooled 100001+100000 (the two ZIDs sharing one field sales team),
# per explicit ask — ignores whatever `zid` the caller passes.
#   - Collection Watchlist: Customer Score (pooled, same engine as
#     Marketing -> Customer Scoring) + Latest Sale & Collection (one row per
#     ZID a shared customer appears in — concatenated, not balance-merged,
#     same non-merging convention Collection Analysis's own combined-scope
#     views already use), light-red-highlighted where a balance has been
#     outstanding more than 3 days since the customer's own Sales Date, then
#     a customer picker showing that one customer's AR ledger pooled across
#     both ZIDs (per-ZID running balance, rows interleaved chronologically).
#   - Returns Watchlist: the FULL list of customers matching the "DO today,
#     return posted the very next WORKING day" refusal pattern in the
#     trailing 14 days (opcrn.xdornum -> opdor.xdornum, holiday-aware via
#     processing/customer_tracking.py::compute_next_day_return_matches) —
#     not filtered down to the watchlist; watchlist members are
#     light-red-highlighted within that full list instead, per explicit ask.
#   - Promised Payment: every pooled customer with a Promised Payment date
#     set in the trailing 3 months (reusing the Latest Sale & Collection
#     pipeline's own already-gated "Promised Payment" column — no new
#     query), light-red-highlighted where that date has passed and a
#     balance is still outstanding.

from __future__ import annotations

import pandas as pd
import streamlit as st

from processing import common, customer_tracking as ctk
from processing import marketing as mkt
from processing import salesman_due as sd
from processing.holidays import _get_holidays
from views.marketing import _load_ar_balance, _load_cacus

# Always pooled — the two ZIDs sharing one field sales team/customer base,
# per explicit ask ("All these need to be 100001 and 100000 combined
# marking"). Fixed, not derived from whichever ZID happens to be active in
# the sidebar when Target Management's Customer Tracking radio is opened.
_ZIDS = ["100001", "100000"]
_PROJ_BY_ZID = {"100001": "GULSHAN TRADING", "100000": "GI Corporation", "100005": "Zepto Chemicals"}

_RETURNS_WINDOW_DAYS = 14
_PROMISE_WINDOW_MONTHS = 3


# ── Cached single-ZID loaders ────────────────────────────────────────────────

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


@st.cache_data(ttl=3600, show_spinner="Building Latest Sale & Collection report…")
def _load_latest_sale_collection(zid: str) -> pd.DataFrame:
    """Same build_salesman_due_reports pipeline as Collection Analysis ->
    Salesman Due -> Latest Sale & Collection (views/collection.py::
    _load_salesman_due_reports) — reproduced here (can't import a private
    `_`-prefixed function cross-module). Already includes "Delivery Date" /
    "Promised Payment" via merge_delivery_payment_promise, which the
    Promised Payment panel reuses directly — no separate query needed."""
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
    """mv_ar_vouchers rows (cusid/date/value directly) — the exact same
    Analytics("ar", ...) source Collection Analysis's own "Customer Ledger"
    -> AR Customer Ledger section uses."""
    from core.analytics import Analytics
    df = Analytics("ar", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_do_next_day_returns_raw(zids: tuple, start_date: str, end_date: str) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics(
        "do_next_day_returns", zid=list(zids),
        filters={"start_date": start_date, "end_date": end_date},
    ).data
    return df if df is not None else pd.DataFrame()


# ── Pooled (100001+100000) wrappers ─────────────────────────────────────────

def _pooled_sales_year(year: int) -> pd.DataFrame:
    frames = [df for z in _ZIDS if not (df := _load_sales_year(z, year)).empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pooled_collection_year(year: int) -> pd.DataFrame:
    frames = [df for z in _ZIDS if not (df := _load_collection_year(z, year)).empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pooled_ar_balance() -> pd.DataFrame:
    frames = []
    for z in _ZIDS:
        df = _load_ar_balance(z, _PROJ_BY_ZID.get(z, ""))
        if df is not None and not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pooled_cacus() -> pd.DataFrame:
    frames = [df for z in _ZIDS if not (df := _load_cacus(z)).empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["cusid"], keep="first")


def _score_customers(year: int) -> pd.DataFrame:
    """Customer Score via the exact same build_customer_marketing_table
    engine Marketing -> Customer Scoring uses, pooled across 100001+100000
    (same pooling A.1b's own customer scoring uses, for the same reason —
    one shared customer base). Current year only — this isn't a payout, just
    a status readout for whoever's on the watchlist."""
    sales_df = _pooled_sales_year(year)
    if sales_df.empty:
        return pd.DataFrame()
    collection_df = _pooled_collection_year(year)
    ar_df = _pooled_ar_balance()
    cacus_df = _pooled_cacus()
    return mkt.build_customer_marketing_table(
        sales_df=sales_df, collection_df=collection_df, ar_df=ar_df,
        selected_years=(year,), cacus_df=cacus_df if not cacus_df.empty else None,
    )


def _pooled_latest_sale_collection() -> pd.DataFrame:
    """Concatenated, NOT balance-merged — a customer shared by both ZIDs
    shows one row per ZID, each with that business's own AR balance, tagged
    by a "ZID" column. Same non-merging convention Collection Analysis's own
    combined-scope Salesman Due / Customer Ledger views already use (two
    separately-tagged results, not one reconciled number)."""
    frames = []
    for z in _ZIDS:
        df = _load_latest_sale_collection(z)
        if not df.empty:
            d = df.copy()
            d.insert(0, "ZID", z)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pooled_ar_ledger(cusid: str) -> pd.DataFrame:
    """One customer's AR ledger across both ZIDs, interleaved
    chronologically — ending_balance computed PER ZID (groupby before
    cumsum) since the two businesses are separate GL books; blindly
    cumsum-ing the pooled rows in display order would mix two unrelated
    running balances into one meaningless number."""
    frames = []
    for z in _ZIDS:
        raw = _load_ar_ledger_raw(z)
        if raw.empty:
            continue
        sub = raw[raw["cusid"].astype(str) == str(cusid)].copy()
        if not sub.empty:
            sub["ZID"] = z
            frames.append(sub)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True).sort_values(["ZID", "date"])
    # mv_ar_vouchers.value is a NUMERIC column — arrives as Decimal/object
    # dtype, which pandas' cython cumsum can't operate on directly (same
    # class of bug as WhatsFly Bulk Messaging's own Decimal-dtype fix
    # elsewhere in this app).
    d["value"] = pd.to_numeric(d["value"], errors="coerce").fillna(0.0)
    d["ending_balance"] = d.groupby("ZID")["value"].cumsum()
    return d.sort_values("date").reset_index(drop=True)


def _pooled_do_next_day_returns(window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    raw = _load_do_next_day_returns_raw(
        tuple(_ZIDS), window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d"),
    )
    holidays = _get_holidays()
    return ctk.compute_next_day_return_matches(raw, holidays, window_start, window_end)


# ── Setup (admin Commissions page) ──────────────────────────────────────────

def _render_setup(zid: str, key_suffix: str) -> None:
    st.caption(
        "Enter customer codes as a comma-separated list (e.g. `CUS-005596, CUS-008441`) for "
        "each watchlist. Results show in Target Management → 👀 Customer Tracking (always "
        "combined 100001 + 100000)."
    )
    if st.session_state.get("user_role") != "admin":
        st.info("Setup is admin-only.")
        return

    watchlist = ctk.load_watchlist(zid)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📲 Collection Watchlist**")
        st.caption(
            "Customers flagged for advance payment — results highlight any of them with a "
            "balance still pending more than 3 days after their last sale."
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
            "Customers flagged by your manager as prone to refuse delivery — highlighted "
            "within the full list of customers matching that pattern in the last 14 days."
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


# ── Results (Target Management -> 👀 Customer Tracking) ─────────────────────

def _render_collection_results(codes: list) -> None:
    st.subheader("📲 Collection Watchlist")
    st.caption("Pooled 100001 + 100000.")
    if not codes:
        st.info("No customers on the Collection watchlist yet. Ask an admin to add some via the Commissions page.")
        return

    today = pd.Timestamp.today().normalize()

    st.markdown("**Customer Score**")
    with st.spinner("Loading customer score…"):
        score_df = _score_customers(int(today.year))
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
    st.caption(
        "🔴 Light red = balance still pending more than 3 days after the customer's own Sales Date. "
        "One row per ZID a customer appears in (not balance-merged)."
    )
    with st.spinner("Loading Latest Sale & Collection…"):
        report_df = _pooled_latest_sale_collection()
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
        st.dataframe(ctk.style_overdue_balance(report_wl, days_threshold=3, today=today), use_container_width=True)

    st.divider()
    st.markdown("**Customer Ledger**")
    cacus_df = _pooled_cacus()
    name_lookup = cacus_df.set_index("cusid")["cusname"].to_dict() if not cacus_df.empty else {}
    options = [f"{c} - {name_lookup.get(c, '')}" for c in codes]
    sel = st.selectbox("Select customer", options, key="ct_coll_ledger_sel")
    sel_code = sel.split(" - ")[0]
    with st.spinner("Loading ledger…"):
        ledger = _pooled_ar_ledger(sel_code)
    if ledger.empty:
        st.info(f"No ledger entries found for {sel_code}.")
    else:
        st.dataframe(ledger, use_container_width=True)


def _render_returns_results(codes: list) -> None:
    st.subheader("↩️ Returns Watchlist")
    st.caption(
        "Pooled 100001 + 100000. Customers who refuse delivery on the spot show up here as a "
        "DO posted one day, with the return (SR) posted the very next WORKING day "
        f"(Thursday deliveries expect a Saturday return, skipping Friday) — trailing "
        f"{_RETURNS_WINDOW_DAYS} days. This is the full list, not just the watchlist — "
        "🔴 light red marks whoever is also on your manager's watchlist."
    )
    today = pd.Timestamp.today().normalize()
    window_start = today - pd.Timedelta(days=_RETURNS_WINDOW_DAYS)
    with st.spinner("Finding next-working-day return candidates…"):
        matches = _pooled_do_next_day_returns(window_start, today)

    if matches.empty:
        st.info(f"No DO→next-working-day-return pattern found in the trailing {_RETURNS_WINDOW_DAYS} days.")
        return

    disp = matches.rename(columns={
        "zid": "ZID", "xdornum": "DO Voucher", "do_date": "DO Date",
        "xcrnnum": "Return Voucher", "sr_date": "Return Date",
        "cusid": "Customer Code", "cusname": "Customer Name",
        "return_amount": "Return Amount", "day_gap": "Day Gap",
    })
    st.caption(f"{disp['Customer Code'].nunique()} distinct customer(s), {len(disp)} matching DO/return pair(s).")
    st.dataframe(ctk.style_watchlist_rows(disp, codes, customer_col="Customer Code"), use_container_width=True)
    st.download_button(
        "⬇ Download CSV", disp.to_csv(index=False).encode("utf-8"),
        file_name=f"returns_watchlist_{today:%Y_%m_%d}.csv", mime="text/csv", key="dl_ct_returns",
    )

    if not codes:
        st.caption("No customers on the Returns watchlist yet — nothing to highlight above. Ask an admin to add some via the Commissions page.")


def _render_promised_payment_results() -> None:
    st.subheader("💳 Promised Payment")
    st.caption(
        "Pooled 100001 + 100000. Every customer with a promised payment date "
        f"(Latest Sale & Collection's own \"Promised Payment\" column) set in the trailing "
        f"{_PROMISE_WINDOW_MONTHS} months. 🔴 Light red = that date has already passed and a "
        "balance is still outstanding."
    )
    with st.spinner("Loading Latest Sale & Collection…"):
        report_df = _pooled_latest_sale_collection()
    promised = ctk.filter_promised_payment_window(report_df, months_back=_PROMISE_WINDOW_MONTHS)
    if promised.empty:
        st.info(f"No promised payment dates found in the trailing {_PROMISE_WINDOW_MONTHS} months.")
        return

    show_cols = [c for c in [
        "ZID", "Customer Code", "Customer Name", "Salesman Code", "Salesman Name",
        "Sales Date", "Current Balance", "Delivery Date", "Promised Payment",
    ] if c in promised.columns]
    disp = promised[show_cols]
    st.caption(f"{disp['Customer Code'].nunique()} distinct customer(s), {len(disp)} row(s).")
    st.dataframe(ctk.style_promised_payment_overdue(disp), use_container_width=True)
    st.download_button(
        "⬇ Download CSV", disp.to_csv(index=False).encode("utf-8"),
        file_name=f"promised_payment_{pd.Timestamp.today():%Y_%m_%d}.csv", mime="text/csv", key="dl_ct_promise",
    )


def _render_results() -> None:
    zids_pooled = ctk.load_watchlist_pooled(_ZIDS)
    sub = st.radio(
        "Tracking Type",
        ["📲 Collection Watchlist", "↩️ Returns Watchlist", "💳 Promised Payment"],
        key="ct_sub", horizontal=True,
    )
    st.divider()
    if sub == "📲 Collection Watchlist":
        _render_collection_results(zids_pooled["collection"])
    elif sub == "↩️ Returns Watchlist":
        _render_returns_results(zids_pooled["returns"])
    else:
        _render_promised_payment_results()


def render(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    st.subheader("👀 Customer Tracking")
    if not read_only:
        _render_setup(str(zid), key_suffix)
        return
    _render_results()
