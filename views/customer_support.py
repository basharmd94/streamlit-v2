# views/customer_support.py
# Customer Support view — 14-day activity feed + Latest Sales & Collection
# with PostgreSQL-backed call log.

from __future__ import annotations

import pandas as pd
import streamlit as st

from processing import customer_support as cs

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
_OUTCOMES = [
    "Promised", "Paid", "Not answered", "Dispute",
    "Delivered", "Not Delivered", "Returned", "Other",
]

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

# Reusable blue panel HTML wrappers for call log sections
def _blue_header(title: str) -> str:
    return (
        f'<div style="background:#EBF5FB;border:1.5px solid #2E86C1;'
        f'border-radius:8px 8px 0 0;padding:8px 14px 6px;margin-bottom:0;">'
        f'<span style="color:#1A5276;font-weight:500;font-size:14px;">{title}</span>'
        f'</div>'
        f'<div style="border:1.5px solid #2E86C1;border-top:none;'
        f'border-radius:0 0 8px 8px;padding:10px 14px 4px;margin-bottom:12px;">'
    )

_BLUE_FOOTER = '</div>'


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


# ── Call log DB helpers ────────────────────────────────────────────────────────

def _load_call_logs(cusid: str) -> pd.DataFrame:
    from core.queries import get_call_logs
    from core.db import get_data
    sql, params = get_call_logs(cusid)
    records, cols = get_data(sql, *params)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records, columns=cols)


def _get_call_logs_cached(cusid: str) -> pd.DataFrame:
    """Serve call logs from session_state — avoids DB hit on every filter keystroke."""
    key = f"_calllog_{cusid}"
    if key not in st.session_state:
        st.session_state[key] = _load_call_logs(cusid)
    return st.session_state[key]


def _bust_call_log_cache(cusid: str) -> None:
    st.session_state.pop(f"_calllog_{cusid}", None)
    # also invalidate the bulk last-called cache so table columns update immediately
    for k in list(st.session_state.keys()):
        if k.startswith("_lastcalled_"):
            del st.session_state[k]


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_last_called_dates(cusids_key: tuple) -> dict:
    """Return {cusid: date_str} for a sorted, frozen tuple of customer codes."""
    if not cusids_key:
        return {}
    from core.db import get_data
    placeholders = ", ".join(["%s"] * len(cusids_key))
    sql = (
        f"SELECT cusid, MAX(called_at)::date AS last_called "
        f"FROM crm_call_log WHERE cusid IN ({placeholders}) GROUP BY cusid"
    )
    records, _ = get_data(sql, *cusids_key)
    if not records:
        return {}
    return {str(r[0]): str(r[1]) for r in records}


def _last_called_map(cusids: "list[str]") -> dict:
    """Serve from session_state so filter keystrokes skip the DB."""
    key = "_lastcalled_" + ",".join(sorted(set(cusids)))
    if key not in st.session_state:
        st.session_state[key] = _fetch_last_called_dates(
            tuple(sorted(set(cusids)))
        )
    return st.session_state[key]


def _save_call_log(zid: str, cusid: str, outcome: str, notes: str) -> bool:
    from core.queries import insert_call_log
    from core.db import execute_write
    sql, params = insert_call_log(
        zid, cusid,
        st.session_state.get("username", ""),
        outcome, notes,
    )
    return execute_write(sql, params)


def _delete_call_log(log_id: int) -> bool:
    from core.queries import delete_call_log
    from core.db import execute_write
    sql, params = delete_call_log(log_id)
    return execute_write(sql, params)


# ── Call log panel ─────────────────────────────────────────────────────────────

_OUTCOME_BADGE = {
    "Paid":         ("background:#D5F5E3;color:#1E8449;", "Paid"),
    "Promised":     ("background:#FDEBD0;color:#A04000;", "Promised"),
    "Not answered": ("background:#F2F3F4;color:#5D6D7E;", "Not answered"),
    "Dispute":      ("background:#FADBD8;color:#A93226;", "Dispute"),
}

def _render_call_log_panel(
    cusid: str,
    zid: str,
    customer_name: str,
    key_suffix: str = "",
) -> None:
    """Blue-bordered call log section: history + add-new form."""
    logs_df = _get_call_logs_cached(cusid)

    # ── Blue header + history as HTML ─────────────────────────────────────────
    entries_html = ""
    if logs_df.empty:
        entries_html = '<p style="color:#7F8C8D;font-style:italic;font-size:13px;margin:0 0 6px;">No calls logged yet.</p>'
    else:
        for _, row in logs_df.iterrows():
            ts = (
                pd.to_datetime(row["called_at"]).strftime("%Y-%m-%d %H:%M")
                if pd.notna(row.get("called_at")) else "—"
            )
            by      = str(row.get("called_by") or "—")
            outcome = str(row.get("outcome") or "—")
            notes   = str(row.get("notes") or "")
            style, label = _OUTCOME_BADGE.get(outcome, ("background:#F2F3F4;color:#5D6D7E;", outcome))
            badge = (
                f'<span style="{style}padding:2px 7px;border-radius:10px;'
                f'font-size:11px;font-weight:500;">{label}</span>'
            )
            note_line = f'<div style="font-size:13px;color:#2C3E50;margin:2px 0 6px 0;">{notes}</div>' if notes else ""
            entries_html += (
                f'<div style="border-left:3px solid #2E86C1;padding:3px 0 3px 10px;margin-bottom:6px;">'
                f'<span style="font-size:11px;color:#7F8C8D;">{ts} · <strong>{by}</strong></span> &nbsp;{badge}'
                f'{note_line}</div>'
            )

    st.markdown(
        _blue_header(f"📞 Call Log — {customer_name} ({cusid})")
        + entries_html
        + _BLUE_FOOTER,
        unsafe_allow_html=True,
    )

    # ── Delete picker (only if there are logs) ────────────────────────────────
    if not logs_df.empty:
        del_options = {
            f"{pd.to_datetime(r['called_at']).strftime('%Y-%m-%d %H:%M')} · {r.get('outcome','—')} · {str(r.get('notes',''))[:40]}": int(r["id"])
            for _, r in logs_df.iterrows()
        }
        dc1, dc2 = st.columns([5, 1])
        del_sel = dc1.selectbox(
            "Delete a log entry",
            ["— select to delete —"] + list(del_options.keys()),
            key=f"del_sel_{cusid}{key_suffix}",
            label_visibility="collapsed",
        )
        if dc2.button("🗑 Delete", key=f"del_btn_{cusid}{key_suffix}") and del_sel != "— select to delete —":
            if _delete_call_log(del_options[del_sel]):
                _bust_call_log_cache(cusid)
                st.rerun()

    # ── Add new log form ──────────────────────────────────────────────────────
    with st.form(f"call_log_form_{cusid}{key_suffix}", clear_on_submit=True):
        fc1, fc2, fc3 = st.columns([2, 4, 1])
        outcome  = fc1.selectbox("Outcome", _OUTCOMES)
        notes    = fc2.text_input("Notes", placeholder="What did they say?")
        fc3.markdown("<br>", unsafe_allow_html=True)
        if fc3.form_submit_button("Save"):
            if _save_call_log(zid, cusid, outcome, notes):
                _bust_call_log_cache(cusid)
                st.success("Call logged.")
                st.rerun()
            else:
                st.error("Failed to save — check DB connection.")


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
        return

    disp_cols = [c for c in _FEED_COLS if c in feed.columns]
    disp = feed[disp_cols].copy().rename(columns=_FEED_RENAME)

    # Inject last-called date column
    _lc_map = _last_called_map(feed["xsub"].astype(str).unique().tolist())
    disp.insert(
        disp.columns.get_loc("Cust Code") + 1,
        "Last Called",
        disp["Cust Code"].astype(str).map(_lc_map),
    )

    st.caption(
        f"**{len(feed):,}** vouchers"
        + (f" — {sel_date_str}" if sel_date_str != "All dates" else " — last 14 days")
        + (f", type: {sel_type}" if sel_type != "All Types" else "")
        + " · sorted latest first"
    )
    st.dataframe(
        disp,
        column_config={
            "Date":        st.column_config.DateColumn("Date",        format="YYYY-MM-DD"),
            "Amount":      st.column_config.NumberColumn("Amount",    format="%.0f"),
            "Last Called": st.column_config.DateColumn("Last Called", format="YYYY-MM-DD"),
        },
        use_container_width=True,
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
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ── Radio 2: Latest Sales & Collection ────────────────────────────────────────

def _sc_status(days) -> str:
    try:
        d = int(days)
    except (TypeError, ValueError):
        return ""
    if d > 30:
        return "🔴"
    if d >= 24:
        return "⚠️"
    return ""


def _render_merged_sc_table(
    df_merged: pd.DataFrame,
    days_min: int | None,
    cust_filter: str | None,
    table_key: str,
) -> None:
    """Render the combined 100001+100000 SC table with call log panel below."""
    if df_merged.empty:
        st.info("No customers with an outstanding balance.")
        return

    df = df_merged.copy()

    if days_min and "days_since_sale" in df.columns:
        qualifying = df[df["days_since_sale"].fillna(0) >= days_min]["cusid"].unique()
        df = df[df["cusid"].isin(qualifying)]

    if cust_filter and "customer_name" in df.columns:
        mask = (
            df["customer_name"].str.contains(cust_filter, case=False, na=False)
            | df["cusid"].astype(str).str.contains(cust_filter, case=False, na=False)
        )
        df = df[df["cusid"].isin(df[mask]["cusid"].unique())]

    if df.empty:
        st.info("No customers match the current filters.")
        return

    df["_status"] = df["days_since_sale"].apply(_sc_status)

    # Inject last-called date
    _lc_map = _last_called_map(df["cusid"].astype(str).unique().tolist())
    df["last_called"] = df["cusid"].astype(str).map(_lc_map)

    col_order = [
        "_status", "zid", "cusid", "customer_name", "last_called", "cusmobile",
        "spid", "salesman_name", "city",
        "days_since_sale", "last_sale_date", "last_sale_amount",
        "days_since_coll", "last_coll_date", "last_coll_amount",
        "current_balance",
    ]
    disp_cols = [c for c in col_order if c in df.columns]
    disp = df[disp_cols].copy().rename(columns={
        "_status": "⚠", "zid": "ZID", "cusid": "Cust Code",
        "customer_name": "Customer", "last_called": "Last Called",
        "cusmobile": "Mobile",
        "spid": "SP Code", "salesman_name": "Salesman", "city": "City",
        "days_since_sale": "Days Sale", "last_sale_date": "Latest Sale Date",
        "last_sale_amount": "Sale Amt", "days_since_coll": "Days Coll",
        "last_coll_date": "Latest Coll Date", "last_coll_amount": "Last Coll Amt",
        "current_balance": "Balance",
    })

    unique_cust = df[["cusid", "customer_name"]].drop_duplicates("cusid")
    st.caption(
        f"**{len(unique_cust):,}** customers · **{len(df):,}** rows (100001+100000)"
        "  ·  ⚠️ = 24-30 days  ·  🔴 = >30 days  ·  sorted: most overdue group first"
    )
    st.dataframe(
        disp,
        column_config={
            "⚠":               st.column_config.TextColumn("⚠", width="small"),
            "Last Called":      st.column_config.DateColumn("Last Called",      format="YYYY-MM-DD"),
            "Latest Sale Date": st.column_config.DateColumn("Latest Sale Date", format="YYYY-MM-DD"),
            "Latest Coll Date": st.column_config.DateColumn("Latest Coll Date", format="YYYY-MM-DD"),
            "Sale Amt":         st.column_config.NumberColumn("Sale Amt",        format="%.0f"),
            "Last Coll Amt":    st.column_config.NumberColumn("Last Coll Amt",   format="%.0f"),
            "Balance":          st.column_config.NumberColumn("Balance",          format="%.0f"),
            "Days Sale":        st.column_config.NumberColumn("Days Sale",        format="%d"),
            "Days Coll":        st.column_config.NumberColumn("Days Coll",        format="%d"),
        },
        use_container_width=True,
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


def _render_sc_table_zepto(
    df: pd.DataFrame,
    days_min: int | None,
    cust_filter: str | None,
) -> None:
    if df.empty:
        st.info(f"No customers with an outstanding balance for {_ZID_LABEL['100005']}.")
        return

    if days_min and "days_since_sale" in df.columns:
        df = df[df["days_since_sale"].fillna(0) >= days_min]
    if cust_filter and "customer_name" in df.columns:
        df = df[
            df["customer_name"].str.contains(cust_filter, case=False, na=False)
            | df["cusid"].astype(str).str.contains(cust_filter, case=False, na=False)
        ]

    if df.empty:
        st.info("No customers match the current filters.")
        return

    df = df.copy()
    df["_status"] = df["days_since_sale"].apply(_sc_status)

    # Inject last-called date
    _lc_map = _last_called_map(df["cusid"].astype(str).unique().tolist())
    df["last_called"] = df["cusid"].astype(str).map(_lc_map)

    col_order = [
        "_status", "cusid", "customer_name", "last_called", "cusmobile",
        "spid", "salesman_name", "city",
        "days_since_sale", "last_sale_date", "last_sale_amount",
        "days_since_coll", "last_coll_date", "last_coll_amount",
        "current_balance",
    ]
    disp_cols = [c for c in col_order if c in df.columns]
    disp = df[disp_cols].copy().rename(columns={
        "_status": "⚠", "cusid": "Cust Code", "customer_name": "Customer",
        "last_called": "Last Called", "cusmobile": "Mobile",
        "spid": "SP Code", "salesman_name": "Salesman",
        "city": "City", "days_since_sale": "Days Sale",
        "last_sale_date": "Latest Sale Date", "last_sale_amount": "Sale Amt",
        "days_since_coll": "Days Coll", "last_coll_date": "Latest Coll Date",
        "last_coll_amount": "Last Coll Amt", "current_balance": "Balance",
    })

    st.caption(
        f"**{len(disp):,}** customers with outstanding balance"
        "  ·  ⚠️ = 24-30 days  ·  🔴 = >30 days"
    )
    st.dataframe(
        disp,
        column_config={
            "⚠":               st.column_config.TextColumn("⚠", width="small"),
            "Last Called":      st.column_config.DateColumn("Last Called",      format="YYYY-MM-DD"),
            "Latest Sale Date": st.column_config.DateColumn("Latest Sale Date", format="YYYY-MM-DD"),
            "Latest Coll Date": st.column_config.DateColumn("Latest Coll Date", format="YYYY-MM-DD"),
            "Sale Amt":         st.column_config.NumberColumn("Sale Amt",        format="%.0f"),
            "Last Coll Amt":    st.column_config.NumberColumn("Last Coll Amt",   format="%.0f"),
            "Balance":          st.column_config.NumberColumn("Balance",          format="%.0f"),
            "Days Sale":        st.column_config.NumberColumn("Days Sale",        format="%d"),
            "Days Coll":        st.column_config.NumberColumn("Days Coll",        format="%d"),
        },
        use_container_width=True,
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


def _render_latest_sales_collection():
    df_100001 = _sc_data("100001")
    df_100000 = _sc_data("100000")
    df_100005 = _sc_data("100005")

    if df_100001 is None and df_100000 is None and df_100005 is None:
        st.warning("AR ledger data unavailable.")
        return

    days_opts = {"All Days": None, "7+ days": 7, "14+ days": 14, "24+ days": 24, "30+ days": 30}

    st.markdown("#### HMBR Tools (100001) + GI Corporation (100000)")
    fc1, fc2 = st.columns(2)
    sel_days_ab = days_opts[fc1.selectbox(
        "Days since sale", list(days_opts.keys()), index=0, key="cs_sc_days_ab",
    )]
    sel_cust_ab = (fc2.text_input(
        "Customer filter", placeholder="name or code…", key="cs_sc_cust_ab",
    ).strip() or None)

    df_merged = cs.build_merged_sc_table(
        df_100001 if df_100001 is not None else pd.DataFrame(),
        df_100000 if df_100000 is not None else pd.DataFrame(),
    )
    _render_merged_sc_table(df_merged, sel_days_ab, sel_cust_ab, table_key="ab")

    st.markdown("---")
    st.markdown("#### Zepto Chemicals (100005)")
    fz1, fz2 = st.columns(2)
    sel_days_z = days_opts[fz1.selectbox(
        "Days since sale (100005)", list(days_opts.keys()), index=0, key="cs_sc_days_z",
    )]
    sel_cust_z = (fz2.text_input(
        "Customer filter (100005)", placeholder="name or code…", key="cs_sc_cust_z",
    ).strip() or None)

    _render_sc_table_zepto(
        df_100005 if df_100005 is not None else pd.DataFrame(),
        sel_days_z, sel_cust_z,
    )


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
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download Ledger CSV",
        disp.to_csv(index=False).encode("utf-8"),
        file_name=f"ledger_{zid}_{cusid}.csv",
        mime="text/csv",
        key=f"dl_cs_ledger{key_suffix}",
    )
