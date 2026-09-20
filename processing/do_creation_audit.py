# processing/do_creation_audit.py
"""
DO Creation Audit — catches orders an SOP forgot to move forward through the
real pipeline: a mobile order comes into opmob, an SOP turns it into a COMO
order (opord), and once stock is verified/packed a delivery order (opdor,
"DO") gets created for the truck. Three genuine gaps, plus a fourth
system-level anomaly, all scoped to 100001+100000 (they share one field
sales team) and a rolling window (default 30 days):

1. Pending Order Creation  — opmob still 'New', SOP hasn't acted at all.
2. Pending DO Creation     — a COMO exists, no DO has been created yet.
3. Partial Fulfillment     — at least one DO exists, but the order isn't
   fully delivered and no further DO has landed in a while.
4. Order Never Persisted   — opmob says 'Order Created' (a real COMO number
   was generated) but that COMO never actually landed in opord. A system
   gap, not an SOP miss — the SOP did act, the backend just didn't persist
   it. No staleness threshold applies here (there's no "still in progress"
   state to wait out; it will never resolve on its own).

Each of tables 1-3 only shows a row once it's been stuck for longer than a
holiday-aware threshold (processing/holidays.py::working_hours_elapsed) — a
Thursday-evening order doesn't start looking overdue until Saturday, since
Friday (and any configured public holiday) contributes zero elapsed hours.
Confirmed by the user: same threshold treatment for both Stage 1 and Stage 2.
"""

from __future__ import annotations

import pandas as pd

from processing.holidays import _get_holidays, working_hours_elapsed

DEFAULT_WINDOW_DAYS = 30

STAGE1_THRESHOLD_HOURS = 24  # Pending Order Creation
STAGE2_THRESHOLD_HOURS = 24  # Pending DO Creation
STAGE3_THRESHOLD_HOURS = 24  # Partial Fulfillment, measured from last DO activity


def _flag_overdue(df: pd.DataFrame, ztime_col: str, threshold_hours: float) -> pd.DataFrame:
    """Adds hours_pending (holiday-aware) and keeps only rows past threshold_hours."""
    if df.empty:
        return df
    d = df.copy()
    d[ztime_col] = pd.to_datetime(d[ztime_col], errors="coerce")
    d = d.dropna(subset=[ztime_col])
    if d.empty:
        return d
    holidays = _get_holidays()
    now = pd.Timestamp.now()
    d["hours_pending"] = d[ztime_col].apply(
        lambda ts: working_hours_elapsed(ts, now, holidays)
    )
    return d[d["hours_pending"] >= threshold_hours].sort_values("hours_pending", ascending=False).reset_index(drop=True)


def build_pending_order_creation(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Table 1 — opmob rows still 'New', past 24 holiday-aware working hours."""
    return _flag_overdue(raw_df, "ztime", STAGE1_THRESHOLD_HOURS)


def build_pending_do_creation(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Table 2 — mobile-sourced opord with no DO yet, past 24 holiday-aware
    working hours since the order (COMO) was created."""
    return _flag_overdue(raw_df, "ztime", STAGE2_THRESHOLD_HOURS)


def build_partial_fulfillment(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Table 3 — at least one DO exists but the order isn't fully delivered,
    and no further DO has landed in the past 24 holiday-aware working hours
    (measured from the most recent DO's own ztime, not the order's creation
    time — an order that's actively still shipping shouldn't show up here)."""
    return _flag_overdue(raw_df, "last_do_ztime", STAGE3_THRESHOLD_HOURS)


def build_unpersisted(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Table 4 — opmob says 'Order Created' but the COMO never landed in
    opord. No threshold — this never resolves on its own, so every row in
    the window is shown regardless of age."""
    if raw_df.empty:
        return raw_df
    d = raw_df.copy()
    d["ztime"] = pd.to_datetime(d["ztime"], errors="coerce")
    return d.sort_values("ztime", ascending=False).reset_index(drop=True)
