# views/usage_stats.py
# Admin-only "📊 Usage Stats" page — reports on view_usage_log
# (processing/usage_log.py). Gated the same way every other page in this
# app is: a page_permissions row for 'admin' only (see
# db/sql_scripts/create_view_usage_log_table.sql), not a hardcoded role
# check — consistent with how access control works everywhere else here.

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from processing import usage_log as ul
from processing import item_master_report as imr
from visualization.common_v import plot_bar_chart


def display_usage_stats_page() -> None:
    st.title("📊 Usage Stats")

    _view_mode = st.radio(
        "View", ["📊 Usage Log", "📦 Item Master Report"],
        horizontal=True, key="us_view_mode",
    )
    ul.log_view("Usage Stats", _view_mode)
    st.markdown("---")

    if _view_mode == "📦 Item Master Report":
        _render_item_master_report()
        return

    _render_usage_log()


def _render_item_master_report() -> None:
    st.caption(
        "Every product in caitem for 100001/100009/100000, related to current stock "
        "(100001+100009 combined via xdrawing) and 5-year import (IP--) purchase frequency. "
        "One row per price tier — an item with 3 opspprc tiers shows 3 rows."
    )

    with st.spinner("Loading item master…"):
        df = imr.load_item_master_report()

    if df.empty:
        st.info("No data available.")
        return

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        zid_opts = sorted(df["zid"].astype(str).unique().tolist())
        sel_zid = st.multiselect(
            "ZID", zid_opts,
            format_func=lambda z: {"100001": "100001 — HMBR/Gulshan Trading",
                                    "100009": "100009 — Gulshan Packaging",
                                    "100000": "100000 — GI Corporation"}.get(z, z),
            key="imr_zid",
        )
    with f2:
        gitem_opts = sorted(df["xgitem"].dropna().unique().tolist())
        sel_gitem = st.multiselect("Item Group (xgitem)", gitem_opts, key="imr_gitem")
    with f3:
        abc_opts = sorted(df["xabc"].dropna().unique().tolist())
        sel_abc = st.multiselect("ABC Class (xabc)", abc_opts, key="imr_abc")
    with f4:
        drawing_opts = sorted(df["xdrawing"].dropna().unique().tolist())
        sel_drawing = st.multiselect("Cross-ZID Link (xdrawing)", drawing_opts, key="imr_drawing")

    filtered = imr.apply_filters(
        df, zids=sel_zid, xgitems=sel_gitem, xabcs=sel_abc, xdrawings=sel_drawing,
    )

    if filtered.empty:
        st.warning("No items match these filters.")
        return

    show = filtered.rename(columns={
        "zid": "ZID", "xitem": "Item Code", "xdesc": "Description", "xlong": "Long Description",
        "xgitem": "Item Group", "xabc": "ABC Class", "xdrawing": "Cross-ZID Link (xdrawing)",
        "std_price": "Std Price", "tier_qty": "Tier Qty", "tier_disc": "Tier Disc",
        "own_zid_stock": "Own ZID Stock", "combined_100001_plus_100009_stock": "Combined 100001+100009 Stock",
        "purchase_count_last_5yr_ip_only": "Purchase Count (5yr, IP-- only)",
    })
    st.caption(f"{len(show):,} row(s) — {show['Item Code'].nunique():,} distinct item(s).")
    st.dataframe(
        show,
        column_config={
            "Std Price": st.column_config.NumberColumn(format="%.2f"),
            "Tier Qty": st.column_config.NumberColumn(format="%.2f"),
            "Tier Disc": st.column_config.NumberColumn(format="%.2f"),
            "Own ZID Stock": st.column_config.NumberColumn(format="%.2f"),
            "Combined 100001+100009 Stock": st.column_config.NumberColumn(format="%.2f"),
        },
        width="stretch", hide_index=True,
    )
    st.download_button(
        "⬇ Download CSV",
        show.to_csv(index=False).encode("utf-8"),
        file_name="item_master_report.csv", mime="text/csv",
        key="imr_dl",
    )


def _render_usage_log() -> None:
    st.caption(
        "Which pages/modes actually get used, by whom, and for how long. Data is retained on a "
        "rolling 1-year basis — anything older than that is cleaned up automatically."
    )

    today = datetime.now().date()
    c1, c2 = st.columns(2)
    start_date = c1.date_input(
        "From", value=today - timedelta(days=30),
        min_value=today - timedelta(days=365), max_value=today, key="us_start",
    )
    end_date = c2.date_input(
        "To", value=today,
        min_value=today - timedelta(days=365), max_value=today, key="us_end",
    )
    if start_date > end_date:
        st.error("From date must be before To date.")
        return

    # end_date is inclusive in the UI but the query's upper bound is
    # exclusive (entered_at < end_date) — push it one day forward so the
    # selected end day's own events aren't dropped.
    raw = ul.load_usage_events(start_date, end_date + timedelta(days=1))
    if raw.empty:
        st.info("No usage recorded in this date range yet.")
        return
    df = ul.compute_durations(raw)

    # ── Overview ────────────────────────────────────────────────────────
    st.markdown("**📈 Overview**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Views", f"{len(df):,}")
    m2.metric("Active Users", f"{df['username'].nunique():,}")
    m3.metric("Sessions", f"{df['session_id'].nunique():,}")
    st.caption(
        "\"Views\" = genuine page/mode transitions, not raw clicks — switching filters within the "
        "same view doesn't count again. Avg/Total time excludes the last view of each session (there's "
        "no way to know when someone closed the tab, so it's left blank rather than guessed)."
    )

    # ── Most-used pages ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🏆 Most-Used Pages**")
    page_summary = ul.build_page_summary(df)
    plot_bar_chart(page_summary, x_axis="Page", y_axis="Views", title="Views by Page")
    st.dataframe(
        page_summary,
        column_config={
            "Avg Seconds": st.column_config.NumberColumn("Avg Seconds", format="%.0f"),
            "Total Seconds": st.column_config.NumberColumn("Total Seconds", format="%.0f"),
        },
        width="stretch", hide_index=True,
    )

    # ── Most-used pages by mode ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🔍 Most-Used Pages by Mode**")
    mode_summary = ul.build_page_mode_summary(df)
    if mode_summary.empty:
        st.info("No radio-level detail recorded for this range.")
    else:
        st.dataframe(
            mode_summary,
            column_config={
                "Avg Seconds": st.column_config.NumberColumn("Avg Seconds", format="%.0f"),
                "Total Seconds": st.column_config.NumberColumn("Total Seconds", format="%.0f"),
            },
            width="stretch", hide_index=True,
        )

    # ── Usage by user ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**👤 Usage by User**")
    user_summary = ul.build_user_summary(df)
    st.dataframe(
        user_summary,
        column_config={
            "Total Seconds": st.column_config.NumberColumn("Total Seconds", format="%.0f"),
            "Last Active": st.column_config.DatetimeColumn("Last Active", format="YYYY-MM-DD HH:mm"),
        },
        width="stretch", hide_index=True,
    )
    st.download_button(
        "⬇ Download Usage by User (CSV)",
        user_summary.to_csv(index=False).encode("utf-8"),
        file_name=f"usage_by_user_{start_date}_{end_date}.csv", mime="text/csv",
        key="us_user_dl",
    )

    # ── Usage by role ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🧑‍🤝‍🧑 Usage by Role**")
    role_summary = ul.build_role_summary(df)
    st.dataframe(
        role_summary,
        column_config={"Total Seconds": st.column_config.NumberColumn("Total Seconds", format="%.0f")},
        width="stretch", hide_index=True,
    )

    # ── Usage over time ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**📅 Usage Over Time**")
    trend = ul.build_daily_trend(df)
    plot_bar_chart(trend, x_axis="Date", y_axis="Views", title="Views per Day")

    # ── Least-used pages ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**📉 Least-Used Pages**")
    st.caption("Every page in the app's menu, including any with zero views in this range.")
    least_used = ul.build_least_used_pages(df)
    st.dataframe(least_used, width="stretch", hide_index=True)
