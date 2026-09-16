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
from visualization.common_v import plot_bar_chart


def display_usage_stats_page() -> None:
    ul.log_view("Usage Stats")

    st.title("📊 Usage Stats")
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
