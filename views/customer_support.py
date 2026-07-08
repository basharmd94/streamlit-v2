# views/customer_support.py
# Customer Support (CRM) view — 7-day activity feed + per-customer AR ledger
# + persistent CRM log (confirmed / remarks) saved to data/crm_log.json.

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from processing import customer_support as cs

_6M_DAYS = 183  # ~6 calendar months

# ── Column display config ──────────────────────────────────────────────────────

_FEED_COLS = [
    "zid", "xdate", "xsub", "customer_name", "xcity",
    "cusmobile", "whatsapp", "salesman_name",
    "xvoucher", "txn_type", "xprime",
    "confirmed", "remarks",
]

_FEED_RENAME = {
    "zid":           "ZID",
    "xdate":         "Date",
    "xsub":          "Cust Code",
    "customer_name": "Customer",
    "xcity":         "Area",
    "cusmobile":     "Mobile",
    "whatsapp":      "WhatsApp",
    "salesman_name": "Salesman",
    "xvoucher":      "Voucher",
    "txn_type":      "Type",
    "xprime":        "Amount",
    "confirmed":     "✓",
    "remarks":       "Remarks",
}

_LEDGER_COLS = [
    "xdate", "xvoucher", "txn_type", "xprime", "running_balance",
]

_LEDGER_RENAME = {
    "xdate":           "Date",
    "xvoucher":        "Voucher",
    "txn_type":        "Type",
    "xprime":          "Amount",
    "running_balance": "Balance",
}


# ── Public entry point ─────────────────────────────────────────────────────────

def display_customer_support(zid, project):
    st.title("📞 Customer Support")

    radio = st.radio(
        "View",
        ["📋 7-Day Activity", "📁 CRM Log"],
        horizontal=True,
        key="cs_radio",
    )

    if radio == "📋 7-Day Activity":
        _render_7day_activity()
    else:
        _render_crm_log()


# ── Cached loaders (module-level so they survive rerenders) ───────────────────

@st.cache_data(show_spinner="Loading AR ledgers…", ttl=1800)
def _ar_data() -> pd.DataFrame:
    return cs.load_all_ar_ledgers()


@st.cache_data(show_spinner="Loading customer contacts…", ttl=1800)
def _cacus_data() -> pd.DataFrame:
    return cs.load_all_cacus()


@st.cache_data(show_spinner="Loading recent deliveries…", ttl=1800)
def _sales_7day_data() -> pd.DataFrame:
    return cs.load_all_sales_7day()


# ── Radio 1: 7-Day Activity ────────────────────────────────────────────────────

def _render_7day_activity():
    ar_df   = _ar_data()
    cacus_df = _cacus_data()

    if ar_df is None or ar_df.empty:
        st.warning("No AR data available.")
        return

    feed = cs.build_7day_feed(ar_df, cacus_df)
    if feed.empty:
        st.info("No customer transactions in the last 7 days.")
        return

    # ── Date filter ───────────────────────────────────────────────────────────
    today_d = pd.Timestamp.today().date()
    sel_date = st.date_input("Show activity for date", value=today_d, key="cs_activity_date")
    feed["_xdate"] = pd.to_datetime(feed["xdate"], errors="coerce").dt.date
    feed = feed[feed["_xdate"] == sel_date].drop(columns=["_xdate"])
    if feed.empty:
        st.info(f"No vouchers on {pd.Timestamp(sel_date).strftime('%d %b %Y')}.")
        return

    # ── CRM log: load once per session and track when ─────────────────────────
    if "cs_loaded_at" not in st.session_state:
        crm_entries, loaded_at = cs.load_crm_log()
        st.session_state["cs_crm_entries"] = crm_entries
        st.session_state["cs_loaded_at"]   = loaded_at
    else:
        crm_entries = st.session_state["cs_crm_entries"]

    # Build per-row CRM key: zid_cusid_voucher
    feed["_key"] = (
        feed["zid"].astype(str) + "_" +
        feed["xsub"].astype(str) + "_" +
        feed["xvoucher"].astype(str)
    )

    # Pre-populate editable columns from saved CRM entries
    feed["confirmed"] = feed["_key"].map(
        lambda k: crm_entries.get(k, {}).get("confirmed", False)
    )
    feed["remarks"] = feed["_key"].map(
        lambda k: crm_entries.get(k, {}).get("remarks", "")
    )

    # Snapshot the feed (keys + original values) so save handler can align rows
    st.session_state["_cs_feed"] = feed.reset_index(drop=True)

    # ── Build display dataframe ───────────────────────────────────────────────
    disp_cols = [c for c in _FEED_COLS if c in feed.columns]
    disp = feed[disp_cols].copy().rename(columns=_FEED_RENAME)

    disabled = [c for c in disp.columns if c not in ("✓", "Remarks")]

    col_cfg = {
        "✓": st.column_config.CheckboxColumn(
            "✓ Confirmed",
            help="Tick to confirm: Delivery received / Collection received / Return acknowledged",
            default=False,
        ),
        "Remarks": st.column_config.TextColumn("Remarks", width="large"),
        "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
        "Amount": st.column_config.NumberColumn("Amount", format="%.0f"),
    }

    st.caption(
        f"Last 7 days — **{len(feed):,}** vouchers across all entities. "
        "Tick ✓ and add remarks, then press Save."
    )

    edited = st.data_editor(
        disp,
        column_config=col_cfg,
        disabled=disabled,
        use_container_width=True,
        hide_index=True,
        key="cs_7day_editor",
        num_rows="fixed",
    )

    # ── Save controls ─────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 4])
    force_save = col2.checkbox(
        "Force save (overwrite if another user saved since your page loaded)",
        key="cs_force_save",
    )
    save_clicked = col1.button("💾 Save CRM Updates", type="primary")

    if save_clicked:
        _do_save(edited, force_save)

    # ── Customer DO detail + 6-month ledger (inside expander) ────────────────
    st.markdown("---")
    with st.expander("📦 Customer DO Detail & Ledger", expanded=True):
        # ── Build grouped customer options ─────────────────────────────────
        # 100001 and 100000 share the same field sales team and customer base
        # so they are grouped together.  100005 (Zepto) is always separate.
        feed_g = feed[["zid", "xsub", "customer_name"]].drop_duplicates().copy()
        feed_g["zid"] = feed_g["zid"].astype(str)

        paired_opts = (
            feed_g[feed_g["zid"].isin(["100001", "100000"])]
            .sort_values("customer_name")
            .groupby("xsub", as_index=False)
            .agg(customer_name=("customer_name", "first"))
            .assign(
                group="100001+100000",
                label=lambda d: (
                    "100001+100000 | " +
                    d["xsub"].astype(str) + " | " +
                    d["customer_name"].fillna("").astype(str)
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
                    "100005 | " +
                    d["xsub"].astype(str) + " | " +
                    d["customer_name"].fillna("").astype(str)
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

            # ── DO line items (all entities, last 7 days) ─────────────────
            st.markdown("##### Deliveries — Last 7 Days (All Entities)")
            _render_do_detail(feed, sel_cusid)

            st.markdown("---")

            # ── 6-month AR ledger(s) ──────────────────────────────────────
            _6M_CAPTION = (
                "Balance is cumulative from all history; only the last 6 months "
                "of transactions are displayed. Final balance matches Salesman Due."
            )
            if sel_group == "100001+100000":
                st.markdown("##### 6-Month AR Ledger — 100001 · GULSHAN TRADING")
                st.caption(_6M_CAPTION)
                _render_ledger(ar_df, "100001", sel_cusid, "_100001")

                st.markdown("---")

                st.markdown("##### 6-Month AR Ledger — 100000 · GI Corporation")
                st.caption(_6M_CAPTION)
                _render_ledger(ar_df, "100000", sel_cusid, "_100000")
            else:
                st.markdown("##### 6-Month AR Ledger — 100005 · Zepto Chemicals")
                st.caption(_6M_CAPTION)
                _render_ledger(ar_df, "100005", sel_cusid, "_100005")


def _render_do_detail(feed: pd.DataFrame, cusid: str):
    """DO line items for `cusid` across all ZIDs in the last 7 days.

    Attempts to cross-reference INOP vouchers from the 7-day AR feed using
    (zid, date) as the link key, since each DO generates one INOP GL entry.
    """
    sales_df = _sales_7day_data()
    if sales_df is None or sales_df.empty:
        st.info("No delivery line items found in the last 7 days.")
        return

    cust_sales = sales_df[sales_df["cusid"] == cusid].copy()
    if cust_sales.empty:
        st.info("No DO line items for this customer in the last 7 days.")
        return

    # Build INOP voucher map: (zid, date) → comma-separated INOP vouchers
    inop_rows = feed[
        (feed["xsub"].astype(str) == cusid) &
        (feed["txn_type"] == "Delivery")
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
        "zid":      "ZID",
        "date":     "Date",
        "voucher":  "DO Number",
        "itemname": "Product",
        "quantity": "Qty",
        "altsales": "Amount",
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


def _do_save(edited: pd.DataFrame, force: bool):
    snapshot: pd.DataFrame = st.session_state.get("_cs_feed", pd.DataFrame())
    if snapshot.empty:
        st.error("Session snapshot missing — please reload the page.")
        return

    new_entries: dict = {}
    delete_keys: set  = set()
    existing_crm = st.session_state.get("cs_crm_entries", {})

    for i in range(len(edited)):
        snap_row  = snapshot.iloc[i]
        edit_row  = edited.iloc[i]
        key       = str(snap_row["_key"])

        confirmed = bool(edit_row.get("✓", False))
        remarks   = str(edit_row.get("Remarks", "") or "").strip()

        if not confirmed and not remarks:
            # Row cleared — remove from log if it previously existed
            if key in existing_crm:
                delete_keys.add(key)
            continue

        new_entries[key] = {
            "zid":       str(snap_row["zid"]),
            "cusid":     str(snap_row["xsub"]),
            "voucher":   str(snap_row["xvoucher"]),
            "txn_type":  str(snap_row.get("txn_type", "")),
            "confirmed": confirmed,
            "remarks":   remarks,
        }

    if not new_entries and not delete_keys:
        st.info("Nothing to save — tick ✓ or add remarks first.")
        return

    loaded_at = st.session_state.get("cs_loaded_at", datetime.min.replace(tzinfo=timezone.utc))
    username  = st.session_state.get("username", "unknown")

    success, msg = cs.save_crm_log(new_entries, loaded_at, username, force=force, delete_keys=delete_keys)
    if success:
        st.success(msg)
        # Refresh session cache so the next Save sees the new timestamp
        crm_entries, new_loaded_at = cs.load_crm_log()
        st.session_state["cs_crm_entries"] = crm_entries
        st.session_state["cs_loaded_at"]   = new_loaded_at
    else:
        st.warning(msg)


def _render_ledger(ar_df: pd.DataFrame, zid: str, cusid: str, key_suffix: str = ""):
    ledger = cs.build_customer_ledger(ar_df, zid, cusid)
    if ledger.empty:
        st.info("No ledger data found for this customer.")
        return

    # Current balance = last running_balance in full history
    current_bal = (
        float(ledger["running_balance"].iloc[-1])
        if "running_balance" in ledger.columns
        else None
    )
    cust_name = ledger["customer_name"].iloc[0] if "customer_name" in ledger.columns else cusid

    col1, col2, col3 = st.columns(3)
    col1.metric("Customer", cust_name)
    col2.metric("ZID", zid)
    if current_bal is not None:
        col3.metric("Current AR Balance", f"{current_bal:,.0f}")

    # Display last 6 months only
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=_6M_DAYS)
    disp = ledger[ledger["xdate"] >= cutoff].copy()

    l_cols   = [c for c in _LEDGER_COLS if c in disp.columns]
    l_rename = {k: v for k, v in _LEDGER_RENAME.items() if k in l_cols}
    disp = disp[l_cols].rename(columns=l_rename)

    st.caption(f"{len(disp):,} transaction row(s) in last 6 months")

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


# ── Radio 2: CRM Log ──────────────────────────────────────────────────────────

def _render_crm_log():
    crm_entries, _ = cs.load_crm_log()

    if not crm_entries:
        st.info("No CRM entries saved yet.")
        return

    log_df = pd.DataFrame(crm_entries.values())
    if log_df.empty:
        st.info("CRM log is empty.")
        return

    # Enrich with customer names
    cacus_df = _cacus_data()
    if not cacus_df.empty and "cusid" in log_df.columns:
        cn = (
            cacus_df[["zid", "cusid", "cusname"]]
            .copy()
            .assign(cusid=lambda d: d["cusid"].astype(str))
        )
        log_df["cusid"] = log_df["cusid"].astype(str)
        log_df = log_df.merge(cn, on=["zid", "cusid"], how="left")

    # Customer filter
    cust_opts = (
        log_df[["zid", "cusid"] + (["cusname"] if "cusname" in log_df.columns else [])]
        .drop_duplicates()
        .sort_values(["zid"] + (["cusname"] if "cusname" in log_df.columns else ["cusid"]))
    )
    if "cusname" in cust_opts.columns:
        cust_opts["label"] = (
            cust_opts["zid"] + " | " +
            cust_opts["cusid"] + " — " +
            cust_opts["cusname"].fillna("")
        )
    else:
        cust_opts["label"] = cust_opts["zid"] + " | " + cust_opts["cusid"]

    sel = st.selectbox(
        "Filter by customer",
        ["— All customers —"] + cust_opts["label"].tolist(),
        key="cs_log_filter",
    )

    if sel != "— All customers —":
        sel_zid   = sel.split(" | ")[0].strip()
        sel_cusid = sel.split(" | ")[1].split(" — ")[0].strip()
        log_df = log_df[
            (log_df["zid"] == sel_zid) &
            (log_df["cusid"].astype(str) == sel_cusid)
        ]

    if log_df.empty:
        st.info("No CRM entries for this customer.")
        return

    if "logged_at" in log_df.columns:
        log_df["logged_at"] = pd.to_datetime(log_df["logged_at"], errors="coerce")
        log_df = log_df.sort_values("logged_at", ascending=False)

    show_cols = (
        ["logged_at", "zid", "cusid"]
        + (["cusname"] if "cusname" in log_df.columns else [])
        + ["voucher", "txn_type", "confirmed", "remarks", "logged_by"]
    )
    show_cols = [c for c in show_cols if c in log_df.columns]

    rename = {
        "logged_at": "Logged At",
        "zid":       "ZID",
        "cusid":     "Cust Code",
        "cusname":   "Customer",
        "voucher":   "Voucher",
        "txn_type":  "Type",
        "confirmed": "✓",
        "remarks":   "Remarks",
        "logged_by": "Logged By",
    }

    disp = log_df[show_cols].rename(columns=rename).reset_index(drop=True)
    st.caption(f"{len(disp):,} CRM entr{'y' if len(disp)==1 else 'ies'}")

    try:
        st.dataframe(
            disp.style.format({"Logged At": "{:%Y-%m-%d %H:%M}"}, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download CRM Log CSV",
        disp.to_csv(index=False).encode("utf-8"),
        file_name="crm_log_export.csv",
        mime="text/csv",
        key="dl_cs_log",
    )
