# views/customer_support.py
# Customer Support view — 14-day activity feed + Latest Sales & Collection
# with PostgreSQL-backed call log.

from __future__ import annotations

import pandas as pd
import streamlit as st

from processing import customer_support as cs
from processing.common import normalize_phone_cols
from views.call_log_shared import (
    OUTCOMES as _OUTCOMES,
    _OUTCOME_BADGE,
    blue_header as _blue_header,
    BLUE_FOOTER as _BLUE_FOOTER,
    load_call_logs as _load_call_logs,
    get_call_logs_cached as _get_call_logs_cached,
    fetch_last_calllog as _fetch_last_calllog,
    last_calllog_map as _last_calllog_map,
    bust_call_log_cache as _bust_call_log_cache,
    save_call_log as _save_call_log,
    delete_call_log_entry as _delete_call_log,
    render_call_log_panel as _render_call_log_panel,
)

_3M_DAYS = 92   # ~3 calendar months for ledger display window

_FEED_COLS = [
    "zid", "xdate", "xsub", "customer_name", "xcity",
    "cusmobile", "whatsapp", "salesman_name",
    "xvoucher", "txn_type", "xprime",
]
_FEED_RENAME = {
    "zid": "ZID", "xdate": "Date", "xsub": "Cust Code",
    "customer_name": "Customer", "xcity": "Area",
    "cusmobile": "Mobile", "whatsapp": "WhatsApp",
    "salesman_name": "Salesman", "xvoucher": "Voucher",
    "txn_type": "Type", "xprime": "Amount",
}
_LEDGER_COLS   = ["xdate", "xvoucher", "txn_type", "xprime", "running_balance"]
_LEDGER_RENAME = {
    "xdate": "Date", "xvoucher": "Voucher", "txn_type": "Type",
    "xprime": "Amount", "running_balance": "Balance",
}
_ZID_LABEL = {
    "100001": "100001 · HMBR Tools",
    "100000": "100000 · GI Corporation",
    "100005": "100005 · Zepto Chemicals",
}
# Blue panel CSS — applied once per page render inside display_customer_support
_BLUE_CSS = """<style>
/* Blue border on every expander on this page */
div[data-testid="stExpander"] > details {
    border: 1.5px solid #2E86C1 !important;
    border-radius: 8px !important;
}
div[data-testid="stExpander"] > details > summary {
    background: #1A5276 !important;
    border-radius: 6px !important;
}
div[data-testid="stExpander"] > details > summary *,
div[data-testid="stExpander"] > details > summary {
    color: #FFFFFF !important;
}
div[data-testid="stExpander"] > details > summary svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}
</style>"""


# ── Call-coverage matrix ───────────────────────────────────────────────────────

def _render_coverage_matrix(
    df: pd.DataFrame,
    cl_map: dict,
    key_suffix: str,
    has_type: bool = False,
) -> None:
    """Collapsed expander — at-a-glance call-coverage pivot for owner oversight."""
    with st.expander("📊 Call Coverage Matrix", expanded=False):
        options = ["Salesman", "City", "Outcomes"]
        if has_type:
            options.append("Type")
        dim = st.radio(
            "Group by", options, horizontal=True, key=f"cov_dim_{key_suffix}",
        )
        matrix = cs.build_callcoverage_matrix(df, cl_map, dim)
        if matrix.empty:
            st.info("No data to build coverage matrix.")
            return
        if dim in ("Salesman", "City"):
            caption = "Cells: called / total unique customers · rows = days since last sale, highest first"
        elif dim == "Type":
            caption = "Cells: called / total unique customers with that transaction type · rows = days since transaction"
        else:
            caption = "Cells: count of customers with that outcome · 'Not Called' = no log entry · rows = days since last sale"
        st.caption(caption)
        st.dataframe(matrix, width="stretch")


# ── Public entry point ─────────────────────────────────────────────────────────

def display_customer_support(zid, project):
    st.title("📞 Customer Support")
    st.markdown(_BLUE_CSS, unsafe_allow_html=True)
    radio = st.radio(
        "View",
        ["📋 14-Day Activity", "📊 Latest Sales & Collection"],
        horizontal=True,
        key="cs_radio",
    )
    if radio == "📋 14-Day Activity":
        _render_14day_activity()
    else:
        _render_latest_sales_collection()


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading AR ledgers…", ttl=1800)
def _ar_data() -> pd.DataFrame:
    return cs.load_all_ar_ledgers()


@st.cache_data(show_spinner="Loading customer contacts…", ttl=1800)
def _cacus_data() -> pd.DataFrame:
    return cs.load_all_cacus()


@st.cache_data(show_spinner="Loading recent activity…", ttl=1800)
def _sales_14day_data() -> pd.DataFrame:
    return cs.load_all_sales_7day()


@st.cache_data(show_spinner="Building Sales & Collection table…", ttl=1800)
def _sc_data(zid: str) -> pd.DataFrame:
    return cs.build_latest_sc_for_zid(_ar_data(), zid, _cacus_data())



# ── Radio 1: 14-Day Activity ───────────────────────────────────────────────────

def _render_14day_activity():
    ar_df    = _ar_data()
    cacus_df = _cacus_data()

    if ar_df is None or ar_df.empty:
        st.warning("No AR data available.")
        return

    feed = cs.build_7day_feed(ar_df, cacus_df)
    if feed.empty:
        st.info("No customer transactions in the last 14 days.")
        return

    feed["_xdate"] = pd.to_datetime(feed["xdate"], errors="coerce").dt.date
    feed = feed.sort_values("xdate", ascending=False).reset_index(drop=True)
    _feed_full = feed.copy()

    # Prepare matrix data from full unfiltered feed now — survives any date/type filter.
    _feed_matrix = _feed_full.copy()
    _feed_matrix["cusid"] = _feed_matrix["xsub"].astype(str)
    _feed_matrix["days_since_sale"] = (
        pd.Timestamp.today().normalize()
        - pd.to_datetime(_feed_matrix["xdate"], errors="coerce")
    ).dt.days
    _feed_matrix = _feed_matrix.rename(columns={"xcity": "city"})
    _cl_map_14d = _last_calllog_map(_feed_matrix["cusid"].unique().tolist())

    fc1, fc2 = st.columns([2, 2])
    unique_dates = sorted(feed["_xdate"].dropna().unique(), reverse=True)
    date_opts    = ["All dates"] + [d.strftime("%Y-%m-%d") for d in unique_dates]
    sel_date_str = fc1.selectbox("Date", date_opts, key="cs_activity_date")
    type_opts    = ["All Types"] + sorted(feed["txn_type"].dropna().unique().tolist())
    sel_type     = fc2.selectbox("Type", type_opts, key="cs_type_filter")

    if sel_date_str != "All dates":
        import datetime as _dt
        sel_date_obj = _dt.date.fromisoformat(sel_date_str)
        feed = feed[feed["_xdate"] == sel_date_obj]
    if sel_type != "All Types" and "txn_type" in feed.columns:
        feed = feed[feed["txn_type"] == sel_type]

    feed = feed.drop(columns=["_xdate"])

    if feed.empty:
        label = sel_date_str if sel_date_str != "All dates" else "the selected period"
        st.info(
            f"No vouchers for {label}"
            + (f" of type '{sel_type}'" if sel_type != "All Types" else "") + "."
        )
    else:
        disp_cols = [c for c in _FEED_COLS if c in feed.columns]
        disp = normalize_phone_cols(feed[disp_cols].copy()).rename(columns=_FEED_RENAME)

        # Inject last-called date, outcome, notes columns
        _cl_map = _last_calllog_map(feed["xsub"].astype(str).unique().tolist())
        insert_at = disp.columns.get_loc("Cust Code") + 1
        disp.insert(insert_at,     "Last Called", disp["Cust Code"].astype(str).map(lambda c: (_cl_map.get(c) or {}).get("last_called")))
        disp.insert(insert_at + 1, "Outcome",     disp["Cust Code"].astype(str).map(lambda c: (_cl_map.get(c) or {}).get("outcome")))
        disp.insert(insert_at + 2, "Notes",       disp["Cust Code"].astype(str).map(lambda c: (_cl_map.get(c) or {}).get("notes")))

        st.caption(
            f"**{len(feed):,}** vouchers"
            + (f" — {sel_date_str}" if sel_date_str != "All dates" else " — last 14 days")
            + (f", type: {sel_type}" if sel_type != "All Types" else "")
            + " · sorted latest first · Outcome/Notes = most recent call log entry per customer"
        )
        st.dataframe(
            disp,
            column_config={
                "Date":        st.column_config.DateColumn("Date",        format="YYYY-MM-DD"),
                "Amount":      st.column_config.NumberColumn("Amount",    format="%.0f"),
                "Last Called": st.column_config.DateColumn("Last Called", format="YYYY-MM-DD"),
                "Outcome":     st.column_config.TextColumn("Outcome"),
                "Notes":       st.column_config.TextColumn("Notes"),
            },
            width="stretch",
            hide_index=True,
        )

    st.markdown("---")
    with st.expander("📦 Customer DO Detail & Ledger", expanded=True):
        feed_g = _feed_full[["zid", "xsub", "customer_name"]].drop_duplicates().copy()
        feed_g["zid"] = feed_g["zid"].astype(str)

        paired_opts = (
            feed_g[feed_g["zid"].isin(["100001", "100000"])]
            .sort_values("customer_name")
            .groupby("xsub", as_index=False)
            .agg(customer_name=("customer_name", "first"))
            .assign(
                group="100001+100000",
                label=lambda d: (
                    "100001+100000 | " + d["xsub"].astype(str) + " | "
                    + d["customer_name"].fillna("").astype(str)
                ),
            )
        )
        zepto_opts = (
            feed_g[feed_g["zid"] == "100005"]
            .groupby("xsub", as_index=False)
            .agg(customer_name=("customer_name", "first"))
            .assign(
                group="100005",
                label=lambda d: (
                    "100005 | " + d["xsub"].astype(str) + " | "
                    + d["customer_name"].fillna("").astype(str)
                ),
            )
        )
        cust_opts = (
            pd.concat([paired_opts, zepto_opts], ignore_index=True)
            .sort_values("label")
            .reset_index(drop=True)
        )

        sel_label = st.selectbox(
            "Select customer",
            ["— pick a customer —"] + cust_opts["label"].tolist(),
            key="cs_ledger_sel",
        )

        if sel_label and sel_label != "— pick a customer —":
            sel_row   = cust_opts[cust_opts["label"] == sel_label].iloc[0]
            sel_cusid = str(sel_row["xsub"])
            sel_group = str(sel_row["group"])
            sel_name  = str(sel_row["customer_name"])

            st.markdown("##### Deliveries — Last 14 Days (All Entities)")
            _render_do_detail(_feed_full, sel_cusid)

            st.markdown("---")

            _3M_CAPTION = (
                "Balance is cumulative from all history; only the last 3 months "
                "of transactions are displayed. Final balance matches Salesman Due."
            )
            if sel_group == "100001+100000":
                with st.expander("3-Month AR Ledger — 100001 · HMBR Tools", expanded=False):
                    st.caption(_3M_CAPTION)
                    _render_ledger(ar_df, "100001", sel_cusid, "_100001")
                with st.expander("3-Month AR Ledger — 100000 · GI Corporation", expanded=False):
                    st.caption(_3M_CAPTION)
                    _render_ledger(ar_df, "100000", sel_cusid, "_100000")
            else:
                with st.expander("3-Month AR Ledger — 100005 · Zepto Chemicals", expanded=False):
                    st.caption(_3M_CAPTION)
                    _render_ledger(ar_df, "100005", sel_cusid, "_100005")

            st.markdown("---")
            _render_call_log_panel(sel_cusid, "100001", sel_name, key_suffix="_14d")

    st.markdown("---")
    _render_coverage_matrix(_feed_matrix, _cl_map_14d, key_suffix="14d", has_type=True)


def _render_do_detail(feed: pd.DataFrame, cusid: str):
    sales_df = _sales_14day_data()
    if sales_df is None or sales_df.empty:
        st.info("No delivery line items found in the last 14 days.")
        return

    cust_sales = sales_df[sales_df["cusid"] == cusid].copy()
    if cust_sales.empty:
        st.info("No DO line items for this customer in the last 14 days.")
        return

    inop_rows = feed[
        (feed["xsub"].astype(str) == cusid) & (feed["txn_type"] == "Delivery")
    ][["zid", "xdate", "xvoucher"]].copy()
    inop_rows["_d"] = pd.to_datetime(inop_rows["xdate"], errors="coerce").dt.date
    inop_map: dict = (
        inop_rows.groupby(["zid", "_d"])["xvoucher"]
        .apply(lambda s: ", ".join(s.astype(str).unique()))
        .to_dict()
    )

    cust_sales["_d"] = cust_sales["date"].dt.date
    cust_sales["INOP Voucher"] = cust_sales.apply(
        lambda r: inop_map.get((r["zid"], r["_d"]), "—"), axis=1
    )
    cust_sales = cust_sales.sort_values(
        ["zid", "date", "voucher", "itemname"]
    ).reset_index(drop=True)

    disp = cust_sales[
        ["zid", "date", "voucher", "INOP Voucher", "itemname", "quantity", "altsales"]
    ].rename(columns={
        "zid": "ZID", "date": "Date", "voucher": "DO Number",
        "itemname": "Product", "quantity": "Qty", "altsales": "Amount",
    })

    st.caption(f"{len(disp):,} line item(s) across all entities")
    try:
        st.dataframe(
            disp.style.format(
                {"Date": "{:%Y-%m-%d}", "Qty": "{:,.0f}", "Amount": "{:,.0f}"},
                na_rep="—",
            ),
            width="stretch",
            hide_index=True,
        )
    except Exception:
        st.dataframe(disp, width="stretch", hide_index=True)


# ── Radio 2: Latest Sales & Collection ────────────────────────────────────────

def _sc_status(days) -> str:
    try:
        d = int(days)
    except (TypeError, ValueError):
        return ""
    if d > 30:
        return ">30"
    if d >= 24:
        return ">24"
    return ""


def _render_merged_sc_table(
    df_merged: pd.DataFrame,
    days_min: int | None,
    salesman_filter: str | None,
    table_key: str,
) -> None:
    """Render the combined 100001+100000 SC table with call log panel below."""
    if df_merged.empty:
        st.info("No customers with an outstanding balance.")
        return

    # Apply salesman filter first — restricts to one salesman's customers.
    # Keep a salesman-scoped copy for the coverage matrix below.
    df_for_matrix = df_merged.copy()
    if salesman_filter and "salesman_name" in df_for_matrix.columns:
        df_for_matrix = df_for_matrix[df_for_matrix["salesman_name"] == salesman_filter]

    df = df_for_matrix.copy()

    if days_min and "days_since_sale" in df.columns:
        qualifying = df[df["days_since_sale"].fillna(0) >= days_min]["cusid"].unique()
        df = df[df["cusid"].isin(qualifying)]

    if df.empty:
        st.info("No customers match the current filters.")
        return

    df["_status"] = df["days_since_sale"].apply(_sc_status)

    # Build cl_map from the salesman-scoped (pre-days-filter) set so the
    # coverage matrix shows all of that salesman's customers, not just the
    # days-filtered subset.
    _cl_map = _last_calllog_map(df_for_matrix["cusid"].astype(str).unique().tolist())
    df["last_called"] = df["cusid"].astype(str).map(lambda c: (_cl_map.get(c) or {}).get("last_called"))
    df["outcome"]     = df["cusid"].astype(str).map(lambda c: (_cl_map.get(c) or {}).get("outcome"))
    df["notes"]       = df["cusid"].astype(str).map(lambda c: (_cl_map.get(c) or {}).get("notes"))

    col_order = [
        "_status", "zid", "cusid", "customer_name", "last_called", "outcome", "notes",
        "cusmobile", "spid", "salesman_name", "city",
        "days_since_sale", "last_sale_date", "last_sale_amount",
        "days_since_coll", "last_coll_date", "last_coll_amount",
        "current_balance",
    ]
    disp_cols = [c for c in col_order if c in df.columns]
    disp = normalize_phone_cols(df[disp_cols].copy()).rename(columns={
        "_status": "⚠", "zid": "ZID", "cusid": "Cust Code",
        "customer_name": "Customer", "last_called": "Last Called",
        "outcome": "Outcome", "notes": "Notes",
        "cusmobile": "Mobile",
        "spid": "SP Code", "salesman_name": "Salesman", "city": "City",
        "days_since_sale": "Days Sale", "last_sale_date": "Latest Sale Date",
        "last_sale_amount": "Sale Amt", "days_since_coll": "Days Coll",
        "last_coll_date": "Latest Coll Date", "last_coll_amount": "Last Coll Amt",
        "current_balance": "Balance",
    })

    unique_cust = df[["cusid", "customer_name"]].drop_duplicates("cusid")
    sm_label = f" · Salesman: **{salesman_filter}**" if salesman_filter else ""
    st.caption(
        f"**{len(unique_cust):,}** customers · **{len(df):,}** rows (100001+100000)"
        + sm_label
        + "  ·  >24 = 24+ days  ·  >30 = 30+ days  ·  sorted: most overdue group first"
        "  ·  Outcome/Notes = most recent call log entry"
    )
    st.dataframe(
        disp,
        column_config={
            "⚠":               st.column_config.TextColumn("⚠", width="small"),
            "Last Called":      st.column_config.DateColumn("Last Called",      format="YYYY-MM-DD"),
            "Outcome":          st.column_config.TextColumn("Outcome"),
            "Notes":            st.column_config.TextColumn("Notes"),
            "Latest Sale Date": st.column_config.DateColumn("Latest Sale Date", format="YYYY-MM-DD"),
            "Latest Coll Date": st.column_config.DateColumn("Latest Coll Date", format="YYYY-MM-DD"),
            "Sale Amt":         st.column_config.NumberColumn("Sale Amt",        format="%.0f"),
            "Last Coll Amt":    st.column_config.NumberColumn("Last Coll Amt",   format="%.0f"),
            "Balance":          st.column_config.NumberColumn("Balance",          format="%.0f"),
            "Days Sale":        st.column_config.NumberColumn("Days Sale",        format="%d"),
            "Days Coll":        st.column_config.NumberColumn("Days Coll",        format="%d"),
        },
        width="stretch",
        hide_index=True,
    )

    # ── Call log panel ────────────────────────────────────────────────────────
    cust_options = (
        unique_cust
        .sort_values("customer_name")
        .apply(lambda r: f"{r['cusid']} · {r['customer_name']}", axis=1)
        .tolist()
    )
    sel = st.selectbox(
        "Select customer to view / log calls",
        ["— pick a customer —"] + cust_options,
        key=f"cs_clog_sel_{table_key}",
    )
    if sel and sel != "— pick a customer —":
        sel_cusid = sel.split(" · ")[0]
        sel_name  = unique_cust.loc[unique_cust["cusid"] == sel_cusid, "customer_name"].iloc[0]
        _render_call_log_panel(sel_cusid, "100001", sel_name, key_suffix=f"_{table_key}")

    st.markdown("---")
    _render_coverage_matrix(df_for_matrix, _cl_map, key_suffix=table_key)


def _render_sc_table_zepto(
    df: pd.DataFrame,
    days_min: int | None,
    salesman_filter: str | None,
) -> None:
    if df.empty:
        st.info(f"No customers with an outstanding balance for {_ZID_LABEL['100005']}.")
        return

    # Apply salesman filter first; keep salesman-scoped copy for coverage matrix.
    df_full = df.copy()
    if salesman_filter and "salesman_name" in df_full.columns:
        df_full = df_full[df_full["salesman_name"] == salesman_filter]

    df = df_full.copy()

    if days_min and "days_since_sale" in df.columns:
        df = df[df["days_since_sale"].fillna(0) >= days_min]

    if df.empty:
        st.info("No customers match the current filters.")
        return

    df = df.copy()
    df["_status"] = df["days_since_sale"].apply(_sc_status)

    # Build cl_map from full pre-filter set so coverage matrix covers all customers.
    _cl_map = _last_calllog_map(df_full["cusid"].astype(str).unique().tolist())
    df["last_called"] = df["cusid"].astype(str).map(lambda c: (_cl_map.get(c) or {}).get("last_called"))
    df["outcome"]     = df["cusid"].astype(str).map(lambda c: (_cl_map.get(c) or {}).get("outcome"))
    df["notes"]       = df["cusid"].astype(str).map(lambda c: (_cl_map.get(c) or {}).get("notes"))

    col_order = [
        "_status", "cusid", "customer_name", "last_called", "outcome", "notes",
        "cusmobile", "spid", "salesman_name", "city",
        "days_since_sale", "last_sale_date", "last_sale_amount",
        "days_since_coll", "last_coll_date", "last_coll_amount",
        "current_balance",
    ]
    disp_cols = [c for c in col_order if c in df.columns]
    disp = normalize_phone_cols(df[disp_cols].copy()).rename(columns={
        "_status": "⚠", "cusid": "Cust Code", "customer_name": "Customer",
        "last_called": "Last Called", "outcome": "Outcome", "notes": "Notes",
        "cusmobile": "Mobile", "spid": "SP Code", "salesman_name": "Salesman",
        "city": "City", "days_since_sale": "Days Sale",
        "last_sale_date": "Latest Sale Date", "last_sale_amount": "Sale Amt",
        "days_since_coll": "Days Coll", "last_coll_date": "Latest Coll Date",
        "last_coll_amount": "Last Coll Amt", "current_balance": "Balance",
    })

    sm_label_z = f" · Salesman: **{salesman_filter}**" if salesman_filter else ""
    st.caption(
        f"**{len(disp):,}** customers with outstanding balance"
        + sm_label_z
        + "  ·  >24 = 24+ days  ·  >30 = 30+ days"
        "  ·  Outcome/Notes = most recent call log entry"
    )
    st.dataframe(
        disp,
        column_config={
            "⚠":               st.column_config.TextColumn("⚠", width="small"),
            "Last Called":      st.column_config.DateColumn("Last Called",      format="YYYY-MM-DD"),
            "Outcome":          st.column_config.TextColumn("Outcome"),
            "Notes":            st.column_config.TextColumn("Notes"),
            "Latest Sale Date": st.column_config.DateColumn("Latest Sale Date", format="YYYY-MM-DD"),
            "Latest Coll Date": st.column_config.DateColumn("Latest Coll Date", format="YYYY-MM-DD"),
            "Sale Amt":         st.column_config.NumberColumn("Sale Amt",        format="%.0f"),
            "Last Coll Amt":    st.column_config.NumberColumn("Last Coll Amt",   format="%.0f"),
            "Balance":          st.column_config.NumberColumn("Balance",          format="%.0f"),
            "Days Sale":        st.column_config.NumberColumn("Days Sale",        format="%d"),
            "Days Coll":        st.column_config.NumberColumn("Days Coll",        format="%d"),
        },
        width="stretch",
        hide_index=True,
    )

    cust_options = (
        df[["cusid", "customer_name"]]
        .drop_duplicates("cusid")
        .sort_values("customer_name")
        .apply(lambda r: f"{r['cusid']} · {r['customer_name']}", axis=1)
        .tolist()
    )
    sel = st.selectbox(
        "Select customer to view / log calls",
        ["— pick a customer —"] + cust_options,
        key="cs_clog_sel_zepto",
    )
    if sel and sel != "— pick a customer —":
        sel_cusid = sel.split(" · ")[0]
        sel_name  = df.loc[df["cusid"] == sel_cusid, "customer_name"].iloc[0]
        _render_call_log_panel(sel_cusid, "100005", sel_name, key_suffix="_zepto")

    st.markdown("---")
    _render_coverage_matrix(df_full, _cl_map, key_suffix="zepto")


def _render_latest_sales_collection():
    df_100001 = _sc_data("100001")
    df_100000 = _sc_data("100000")
    df_100005 = _sc_data("100005")

    if df_100001 is None and df_100000 is None and df_100005 is None:
        st.warning("AR ledger data unavailable.")
        return

    days_opts = {"All Days": None, "7+ days": 7, "14+ days": 14, "24+ days": 24, "30+ days": 30}

    # ── HMBR + GI ──────────────────────────────────────────────────────────────
    st.markdown("#### HMBR Tools (100001) + GI Corporation (100000)")

    df_merged = cs.build_merged_sc_table(
        df_100001 if df_100001 is not None else pd.DataFrame(),
        df_100000 if df_100000 is not None else pd.DataFrame(),
    )

    # Build salesman list from merged data so the selectbox is always populated.
    sm_names_ab = sorted(
        df_merged["salesman_name"].dropna().astype(str).unique().tolist()
    ) if not df_merged.empty and "salesman_name" in df_merged.columns else []

    fc1, fc2 = st.columns(2)
    sel_days_ab = days_opts[fc1.selectbox(
        "Days since sale", list(days_opts.keys()), index=0, key="cs_sc_days_ab",
    )]
    sel_sm_raw_ab = fc2.selectbox(
        "Salesman", ["All Salesmen"] + sm_names_ab, index=0, key="cs_sc_sm_ab",
    )
    sel_sm_ab = None if sel_sm_raw_ab == "All Salesmen" else sel_sm_raw_ab

    _render_merged_sc_table(df_merged, sel_days_ab, sel_sm_ab, table_key="ab")

    # ── Zepto ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Zepto Chemicals (100005)")

    df_z = df_100005 if df_100005 is not None else pd.DataFrame()

    sm_names_z = sorted(
        df_z["salesman_name"].dropna().astype(str).unique().tolist()
    ) if not df_z.empty and "salesman_name" in df_z.columns else []

    fz1, fz2 = st.columns(2)
    sel_days_z = days_opts[fz1.selectbox(
        "Days since sale (100005)", list(days_opts.keys()), index=0, key="cs_sc_days_z",
    )]
    sel_sm_raw_z = fz2.selectbox(
        "Salesman (100005)", ["All Salesmen"] + sm_names_z, index=0, key="cs_sc_sm_z",
    )
    sel_sm_z = None if sel_sm_raw_z == "All Salesmen" else sel_sm_raw_z

    _render_sc_table_zepto(df_z, sel_days_z, sel_sm_z)


# ── Ledger helper ──────────────────────────────────────────────────────────────

def _render_ledger(ar_df: pd.DataFrame, zid: str, cusid: str, key_suffix: str = ""):
    ledger = cs.build_customer_ledger(ar_df, zid, cusid)
    if ledger.empty:
        st.info("No ledger data found for this customer.")
        return

    current_bal = (
        float(ledger["running_balance"].iloc[-1])
        if "running_balance" in ledger.columns else None
    )
    cust_name = ledger["customer_name"].iloc[0] if "customer_name" in ledger.columns else cusid

    col1, col2, col3 = st.columns(3)
    col1.metric("Customer", cust_name)
    col2.metric("ZID", zid)
    if current_bal is not None:
        col3.metric("Current AR Balance", f"{current_bal:,.0f}")

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=_3M_DAYS)
    disp   = ledger[ledger["xdate"] >= cutoff].copy()

    l_cols   = [c for c in _LEDGER_COLS if c in disp.columns]
    l_rename = {k: v for k, v in _LEDGER_RENAME.items() if k in l_cols}
    disp     = disp[l_cols].rename(columns=l_rename)

    st.caption(f"{len(disp):,} transaction row(s) in last 3 months")
    try:
        st.dataframe(
            disp.style.format(
                {"Date": "{:%Y-%m-%d}", "Amount": "{:,.0f}", "Balance": "{:,.0f}"},
                na_rep="—",
            ),
            width="stretch",
            hide_index=True,
        )
    except Exception:
        st.dataframe(disp, width="stretch", hide_index=True)

    st.download_button(
        "⬇ Download Ledger CSV",
        disp.to_csv(index=False).encode("utf-8"),
        file_name=f"ledger_{zid}_{cusid}.csv",
        mime="text/csv",
        key=f"dl_cs_ledger{key_suffix}",
    )
