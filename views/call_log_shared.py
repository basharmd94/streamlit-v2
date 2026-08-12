# views/call_log_shared.py
# Shared call-log helpers imported by customer_support and marketing (inactive outreach).
# All DB state for crm_call_log lives here; views only call the public functions.

from __future__ import annotations

import pandas as pd
import streamlit as st


# ── Constants ──────────────────────────────────────────────────────────────────

OUTCOMES = [
    "Promised", "Paid", "Not answered", "Dispute",
    "Delivered", "Not Delivered", "Returned", "Other",
]

_OUTCOME_BADGE = {
    "Paid":         ("background:#D5F5E3;color:#1E8449;",   "Paid"),
    "Promised":     ("background:#FDEBD0;color:#A04000;",   "Promised"),
    "Not answered": ("background:#F2F3F4;color:#5D6D7E;",   "Not answered"),
    "Dispute":      ("background:#FADBD8;color:#A93226;",   "Dispute"),
}


# ── Blue panel HTML helpers ────────────────────────────────────────────────────

def blue_header(title: str) -> str:
    return (
        f'<div style="background:#EBF5FB;border:1.5px solid #2E86C1;'
        f'border-radius:8px 8px 0 0;padding:8px 14px 6px;margin-bottom:0;">'
        f'<span style="color:#1A5276;font-weight:500;font-size:14px;">{title}</span>'
        f'</div>'
        f'<div style="border:1.5px solid #2E86C1;border-top:none;'
        f'border-radius:0 0 8px 8px;padding:10px 14px 4px;margin-bottom:12px;">'
    )


BLUE_FOOTER = '</div>'


# ── DB helpers ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def load_call_logs(cusid: str) -> pd.DataFrame:
    """Fetch all call log entries for a customer, newest first.
    Cached with a 30-second TTL (shared across all sessions) so fresh entries
    from any tab or user appear within 30 s, and busting via load_call_logs.clear()
    forces an immediate re-fetch."""
    from core.queries import get_call_logs
    from core.db import get_data
    sql, params = get_call_logs(cusid)
    records, cols = get_data(sql, *params)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records, columns=cols)


def get_call_logs_cached(cusid: str) -> pd.DataFrame:
    """Return per-customer call logs via the shared @st.cache_data cache."""
    return load_call_logs(cusid)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_last_calllog(cusids_key: tuple) -> dict:
    """Return {cusid: {last_called, outcome, notes}} for a frozen tuple of cusids.
    Uses DISTINCT ON to return only the most-recent entry per customer."""
    if not cusids_key:
        return {}
    from core.db import get_data
    placeholders = ", ".join(["%s"] * len(cusids_key))
    sql = (
        f"SELECT DISTINCT ON (cusid) cusid, called_at::date, outcome, notes "
        f"FROM crm_call_log WHERE cusid IN ({placeholders}) "
        f"ORDER BY cusid, called_at DESC"
    )
    records, _ = get_data(sql, *cusids_key)
    if not records:
        return {}
    return {
        str(r[0]): {
            "last_called": str(r[1]) if r[1] is not None else None,
            "outcome":     str(r[2]) if r[2] is not None else None,
            "notes":       str(r[3]) if r[3] is not None else None,
        }
        for r in records
    }


def last_calllog_map(cusids: "list[str]") -> dict:
    """Serve from session_state so filter keystrokes skip the DB."""
    key = "_lastcalllog_" + ",".join(sorted(set(cusids)))
    if key not in st.session_state:
        st.session_state[key] = fetch_last_calllog(tuple(sorted(set(cusids))))
    return st.session_state[key]


def bust_call_log_cache(cusid: str) -> None:  # noqa: ARG001 (cusid kept for API compat)
    """Invalidate the shared @cache_data memos so every session sees fresh history."""
    load_call_logs.clear()          # clears history for all customers (small table, fine)
    fetch_last_calllog.clear()      # clears last-calllog summary column
    for k in list(st.session_state.keys()):
        if k.startswith("_lastcalllog_"):
            del st.session_state[k]


def save_call_log(zid: str, cusid: str, outcome: str, notes: str) -> bool:
    from core.queries import insert_call_log
    from core.db import execute_write
    sql, params = insert_call_log(
        zid, cusid,
        st.session_state.get("username", ""),
        outcome, notes,
    )
    return execute_write(sql, params)


def delete_call_log_entry(log_id: int) -> bool:
    from core.queries import delete_call_log
    from core.db import execute_write
    sql, params = delete_call_log(log_id)
    return execute_write(sql, params)


# ── Rendered panel ─────────────────────────────────────────────────────────────

def render_call_log_panel(
    cusid: str,
    zid: str,
    customer_name: str,
    key_suffix: str = "",
) -> None:
    """Blue-bordered call log section: history (with caller username) + add-new form."""
    logs_df = get_call_logs_cached(cusid)

    entries_html = ""
    if logs_df.empty:
        entries_html = (
            '<p style="color:#7F8C8D;font-style:italic;font-size:13px;margin:0 0 6px;">'
            'No calls logged yet.</p>'
        )
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
            note_line = (
                f'<div style="font-size:13px;color:#2C3E50;margin:2px 0 6px 0;">{notes}</div>'
                if notes else ""
            )
            entries_html += (
                f'<div style="border-left:3px solid #2E86C1;padding:3px 0 3px 10px;margin-bottom:6px;">'
                f'<span style="font-size:11px;color:#7F8C8D;">{ts} · <strong>{by}</strong></span>'
                f' &nbsp;{badge}{note_line}</div>'
            )

    st.markdown(
        blue_header(f"📞 Call Log — {customer_name} ({cusid})")
        + entries_html
        + BLUE_FOOTER,
        unsafe_allow_html=True,
    )

    if not logs_df.empty:
        del_options = {
            f"{pd.to_datetime(r['called_at']).strftime('%Y-%m-%d %H:%M')} · "
            f"{r.get('outcome','—')} · {str(r.get('notes',''))[:40]}": int(r["id"])
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
            if delete_call_log_entry(del_options[del_sel]):
                bust_call_log_cache(cusid)
                st.rerun()

    with st.form(f"call_log_form_{cusid}{key_suffix}", clear_on_submit=True):
        fc1, fc2, fc3 = st.columns([2, 4, 1])
        outcome = fc1.selectbox("Outcome", OUTCOMES)
        notes   = fc2.text_input("Notes", placeholder="What did they say?")
        fc3.markdown("<br>", unsafe_allow_html=True)
        if fc3.form_submit_button("Save"):
            if save_call_log(zid, cusid, outcome, notes):
                bust_call_log_cache(cusid)
                st.success("Call logged.")
                st.rerun()
            else:
                st.error("Failed to save — check DB connection.")


def render_call_log_readonly(cusid: str, customer_name: str) -> None:
    """Read-only call log history — shows logged calls with no add/delete controls.
    Used in Collection Analysis and anywhere a view-only audit trail is needed."""
    logs_df = get_call_logs_cached(cusid)

    if logs_df.empty:
        entries_html = (
            '<p style="color:#7F8C8D;font-style:italic;font-size:13px;margin:0 0 6px;">'
            'No calls logged yet for this customer.</p>'
        )
    else:
        entries_html = ""
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
            note_line = (
                f'<div style="font-size:13px;color:#2C3E50;margin:2px 0 6px 0;">{notes}</div>'
                if notes else ""
            )
            entries_html += (
                f'<div style="border-left:3px solid #2E86C1;padding:3px 0 3px 10px;margin-bottom:6px;">'
                f'<span style="font-size:11px;color:#7F8C8D;">{ts} · <strong>{by}</strong></span>'
                f' &nbsp;{badge}{note_line}</div>'
            )

    st.markdown(
        blue_header(f"📞 Call Log — {customer_name} ({cusid})")
        + entries_html
        + BLUE_FOOTER,
        unsafe_allow_html=True,
    )
