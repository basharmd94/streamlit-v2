# processing/usage_log.py
"""
Per-view usage logging — powers the admin-only "📊 Usage Stats" page
(views/usage_stats.py). Tracks which page + primary radio-level mode each
user opens, and when, so usage patterns (most/least-used pages, per-user
activity, time spent) can be reported on.

App-owned table (view_usage_log, the main `da` database — see
db/sql_scripts/create_view_usage_log_table.sql), same convention as
crm_call_log/marketing_leads/page_permissions: written directly by this
app via core/db.py, no separate role/grants needed (this app's single DB
connection already has full access to its own database).

Streamlit has no real "page exit" event — there's no unload hook the way
a normal web app has. "Time spent on a view" is instead DERIVED after the
fact, as the gap between one view's entered_at and the NEXT view's
entered_at within the same session (see compute_durations below) — the
very last view in a session gets no computed duration, since there's
genuinely no way to know when the user closed the tab. This is the
standard practical approach for this kind of lightweight session
analytics in a rerun-based framework, not a compromise unique to this app.
"""

import random
import uuid

import pandas as pd
import streamlit as st

from core.db import execute_write, get_dataframe

# One-year rolling retention, per explicit ask — self-maintaining (see
# _CLEANUP_PROBABILITY below), no separate scheduled job needed (this app
# has no background scheduler; every write happens inside a normal page
# request/response cycle).
_RETENTION_DAYS = 365

# Probability of running the retention cleanup on any given log_view call.
# ~1 in 500 view-logs triggers a sweep — cheap (one indexed DELETE), and a
# day or two of "late" cleanup doesn't matter for a rolling retention
# window that isn't compliance-driven.
_CLEANUP_PROBABILITY = 1 / 500

# Keep in sync with app.py's own `menu` list — used by the Least-Used
# Pages report so a page with ZERO views in the selected window still
# shows up (rather than silently vanishing because it has no rows to
# group by at all).
ALL_PAGES = [
    "Home",
    "Overall Sales Analysis",
    "Customer Data View",
    "Overall Margin Analysis",
    "Collection Analysis",
    "Purchase Analysis",
    "Basket Analysis",
    "Financial Statements",
    "Target Management",
    "Accounting Analysis",
    "Inventory Analysis",
    "Manufacturing Analysis",
    "Marketing Analysis",
    "Customer Support",
    "Commissions",
]


def ensure_session_id() -> str:
    """One UUID per browser session, generated once and reused for every
    view logged in that session — lets usage be grouped/sequenced per
    session (session count, session length, view-to-view duration)."""
    if "_usage_session_id" not in st.session_state:
        st.session_state["_usage_session_id"] = str(uuid.uuid4())
    return st.session_state["_usage_session_id"]


def log_view(page: str, view_path: str = None) -> None:
    """Logs one (page, view_path) entry for the current user/session —
    but ONLY when it's actually different from the last thing logged in
    this session. Streamlit reruns the whole script on ANY interaction
    (typing in an unrelated filter, clicking an unrelated button
    elsewhere on the page), so without this guard every such rerun would
    insert a duplicate row for a view the user hasn't actually navigated
    away from. Silently does nothing if the user isn't authenticated yet
    (e.g. still on the login page) — never raises, since a logging
    failure should never break the page it's instrumenting."""
    try:
        if not st.session_state.get("authenticated"):
            return
        username = st.session_state.get("username")
        if not username:
            return

        key = (page, view_path)
        if st.session_state.get("_usage_last_logged") == key:
            return
        st.session_state["_usage_last_logged"] = key

        session_id = ensure_session_id()
        execute_write(
            """
            INSERT INTO view_usage_log (session_id, username, user_role, zid, page, view_path)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                session_id, username, st.session_state.get("user_role"),
                str(st.session_state.get("zid") or ""), page, view_path,
            ),
        )

        if random.random() < _CLEANUP_PROBABILITY:
            _cleanup_old_rows()
    except Exception:
        # Usage logging is a side effect, never the main point of whatever
        # page called this — a DB hiccup here must not break that page.
        pass


def _cleanup_old_rows() -> None:
    """One-year rolling retention sweep — see _RETENTION_DAYS above."""
    execute_write("DELETE FROM view_usage_log WHERE entered_at < now() - interval '365 days'")


def load_usage_events(start_date, end_date) -> pd.DataFrame:
    """Raw view_usage_log rows with entered_at in [start_date, end_date) —
    for views/usage_stats.py to aggregate in pandas, same "SQL for raw
    fetch, pandas for transform" pattern used throughout this app."""
    df = get_dataframe(
        """
        SELECT session_id, username, user_role, zid, page, view_path, entered_at
        FROM view_usage_log
        WHERE entered_at >= %s AND entered_at < %s
        ORDER BY session_id, entered_at
        """,
        (start_date, end_date),
    )
    return df if df is not None else pd.DataFrame()


def compute_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a duration_seconds column — the gap to the NEXT logged view
    within the same session. The last view of each session gets NaN (see
    module docstring — there's no way to know when they actually left)."""
    if df.empty:
        return df
    d = df.sort_values(["session_id", "entered_at"]).copy()
    d["entered_at"] = pd.to_datetime(d["entered_at"])
    d["next_entered_at"] = d.groupby("session_id")["entered_at"].shift(-1)
    d["duration_seconds"] = (d["next_entered_at"] - d["entered_at"]).dt.total_seconds()
    return d.drop(columns=["next_entered_at"])


def build_page_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per page: view count, avg/total time spent (seconds).
    avg/total exclude the NaN "last view of a session" rows automatically
    (pandas .mean()/.sum() skip NaN)."""
    if df.empty:
        return pd.DataFrame(columns=["Page", "Views", "Avg Seconds", "Total Seconds"])
    g = (
        df.groupby("page")
        .agg(views=("page", "size"), avg_seconds=("duration_seconds", "mean"), total_seconds=("duration_seconds", "sum"))
        .reset_index()
        .rename(columns={"page": "Page", "views": "Views", "avg_seconds": "Avg Seconds", "total_seconds": "Total Seconds"})
        .sort_values("Views", ascending=False)
        .reset_index(drop=True)
    )
    return g


def build_page_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Same as build_page_summary, one level finer: (page, view_path).
    Rows with no view_path (a page with no internal radio) are dropped —
    those are already fully covered by build_page_summary itself."""
    if df.empty:
        return pd.DataFrame(columns=["Page", "Mode", "Views", "Avg Seconds", "Total Seconds"])
    d = df[df["view_path"].notna()]
    if d.empty:
        return pd.DataFrame(columns=["Page", "Mode", "Views", "Avg Seconds", "Total Seconds"])
    g = (
        d.groupby(["page", "view_path"])
        .agg(views=("page", "size"), avg_seconds=("duration_seconds", "mean"), total_seconds=("duration_seconds", "sum"))
        .reset_index()
        .rename(columns={
            "page": "Page", "view_path": "Mode", "views": "Views",
            "avg_seconds": "Avg Seconds", "total_seconds": "Total Seconds",
        })
        .sort_values("Views", ascending=False)
        .reset_index(drop=True)
    )
    return g


def build_user_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per user: total views, distinct pages touched, total time,
    and when they were last active — per explicit ask."""
    if df.empty:
        return pd.DataFrame(columns=["User", "Role", "Views", "Distinct Pages", "Total Seconds", "Last Active"])
    g = (
        df.groupby(["username", "user_role"])
        .agg(
            views=("page", "size"),
            distinct_pages=("page", "nunique"),
            total_seconds=("duration_seconds", "sum"),
            last_active=("entered_at", "max"),
        )
        .reset_index()
        .rename(columns={
            "username": "User", "user_role": "Role", "views": "Views",
            "distinct_pages": "Distinct Pages", "total_seconds": "Total Seconds",
            "last_active": "Last Active",
        })
        .sort_values("Views", ascending=False)
        .reset_index(drop=True)
    )
    return g


def build_role_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per role: total views, distinct users active."""
    if df.empty:
        return pd.DataFrame(columns=["Role", "Views", "Distinct Users", "Total Seconds"])
    g = (
        df.groupby("user_role")
        .agg(views=("page", "size"), distinct_users=("username", "nunique"), total_seconds=("duration_seconds", "sum"))
        .reset_index()
        .rename(columns={
            "user_role": "Role", "views": "Views", "distinct_users": "Distinct Users",
            "total_seconds": "Total Seconds",
        })
        .sort_values("Views", ascending=False)
        .reset_index(drop=True)
    )
    return g


def build_daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day: view count, distinct active users."""
    if df.empty:
        return pd.DataFrame(columns=["Date", "Views", "Distinct Users"])
    d = df.copy()
    d["date"] = pd.to_datetime(d["entered_at"]).dt.date
    g = (
        d.groupby("date")
        .agg(views=("page", "size"), distinct_users=("username", "nunique"))
        .reset_index()
        .rename(columns={"date": "Date", "views": "Views", "distinct_users": "Distinct Users"})
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return g


def build_least_used_pages(df: pd.DataFrame) -> pd.DataFrame:
    """Every page in ALL_PAGES, sorted fewest-views-first — including
    pages with ZERO views in the selected window (which wouldn't appear
    at all from a plain groupby), so a genuinely unused feature is
    visible rather than silently absent from the report."""
    counts = df.groupby("page").size() if not df.empty else pd.Series(dtype=int)
    out = pd.DataFrame({"Page": ALL_PAGES})
    out["Views"] = out["Page"].map(counts).fillna(0).astype(int)
    return out.sort_values("Views").reset_index(drop=True)
