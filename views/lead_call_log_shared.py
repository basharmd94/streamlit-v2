# views/lead_call_log_shared.py
# Call-log helpers for Marketing Leads — same panel look/behavior as
# views/call_log_shared.py (Customer Support), keyed on lead_id instead of
# cusid, in a separate table (marketing_lead_call_log) since a lead is not
# yet a customer. Adds next_visit_date, which the customer call log doesn't have.

from __future__ import annotations

import pandas as pd
import streamlit as st

from views.call_log_shared import blue_header, BLUE_FOOTER


# ── Lead-specific outcomes ───────────────────────────────────────────────────
# Distinct from Customer Support's OUTCOMES (Paid/Delivered/Returned are
# order-fulfilment states — meaningless before a lead has become a customer).
# Roughly ordered along a qualify -> sample -> close funnel. "Follow-up
# Requested" and "Promised to Order" are additions beyond what was asked for,
# to cover an answered-but-undecided call and a soft pre-close commitment —
# both fill real gaps in the funnel and pair naturally with the next_visit_date
# field already on this form.
LEAD_OUTCOMES = [
    "Not Answered",
    "Not Interested",
    "B2C",
    "Wrong Lead",
    "Asked to Submit Sample",
    "Sample Submitted",
    "Still Using Sample – Will Contact After",
    "Follow-up Requested",
    "Promised to Order",
    "Deal Completed",
]

_LEAD_OUTCOME_BADGE = {
    "Deal Completed":                          ("background:#D5F5E3;color:#1E8449;", "Deal Completed"),
    "Sample Submitted":                        ("background:#D5F5E3;color:#1E8449;", "Sample Submitted"),
    "Promised to Order":                       ("background:#FDEBD0;color:#A04000;", "Promised to Order"),
    "Asked to Submit Sample":                  ("background:#FDEBD0;color:#A04000;", "Asked to Submit Sample"),
    "Still Using Sample – Will Contact After": ("background:#FCF3CF;color:#7D6608;", "Using Sample – Follow Up Later"),
    "Follow-up Requested":                     ("background:#FCF3CF;color:#7D6608;", "Follow-up Requested"),
    "Not Answered":                            ("background:#F2F3F4;color:#5D6D7E;", "Not Answered"),
    "Not Interested":                          ("background:#FADBD8;color:#A93226;", "Not Interested"),
    "Wrong Lead":                              ("background:#FADBD8;color:#A93226;", "Wrong Lead"),
    "B2C":                                     ("background:#FADBD8;color:#A93226;", "B2C"),
}


# ── DB helpers ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def load_lead_call_logs(lead_id: int) -> pd.DataFrame:
    """Fetch all call log entries for one lead, newest first. 30s TTL, shared
    across sessions — mirrors load_call_logs in call_log_shared.py."""
    from core.queries import get_lead_call_logs
    from core.db import get_data
    sql, params = get_lead_call_logs(lead_id)
    records, cols = get_data(sql, *params)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records, columns=cols)


def get_lead_call_logs_cached(lead_id: int) -> pd.DataFrame:
    return load_lead_call_logs(lead_id)


@st.cache_data(ttl=60, show_spinner=False)
def load_all_lead_call_logs(zid: str) -> pd.DataFrame:
    """All call log rows for a ZID, joined with lead name/company — feeds
    Table 2 (all call logs, filterable by date called / next visit date)."""
    from core.queries import get_all_lead_call_logs
    from core.db import get_data
    sql, params = get_all_lead_call_logs(zid)
    records, cols = get_data(sql, *params)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records, columns=cols)


def bust_lead_call_log_cache() -> None:
    load_lead_call_logs.clear()
    load_all_lead_call_logs.clear()


def save_lead_call_log(
    zid: str, lead_id: int, outcome: str, next_visit_date, notes: str
) -> bool:
    from core.queries import insert_lead_call_log
    from core.db import execute_write
    sql, params = insert_lead_call_log(
        zid, lead_id,
        st.session_state.get("username", ""),
        outcome, next_visit_date, notes,
    )
    return execute_write(sql, params)


def delete_lead_call_log_entry(log_id: int) -> bool:
    from core.queries import delete_lead_call_log
    from core.db import execute_write
    sql, params = delete_lead_call_log(log_id)
    return execute_write(sql, params)


# ── Rendered panel ─────────────────────────────────────────────────────────────

def render_lead_call_log_panel(
    lead_id: int,
    zid: str,
    lead_name: str,
    key_suffix: str = "",
) -> None:
    """Blue-bordered call log section for a lead — history (with caller
    username + next visit date) + add-new form. Same visual language as
    render_call_log_panel in Customer Support."""
    logs_df = get_lead_call_logs_cached(lead_id)

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
            nvd     = row.get("next_visit_date")
            nvd_str = (
                pd.to_datetime(nvd).strftime("%Y-%m-%d")
                if pd.notna(nvd) else None
            )
            style, label = _LEAD_OUTCOME_BADGE.get(outcome, ("background:#F2F3F4;color:#5D6D7E;", outcome))
            badge = (
                f'<span style="{style}padding:2px 7px;border-radius:10px;'
                f'font-size:11px;font-weight:500;">{label}</span>'
            )
            nvd_badge = (
                f'<span style="background:#EAF2F8;color:#1A5276;padding:2px 7px;'
                f'border-radius:10px;font-size:11px;font-weight:500;margin-left:4px;">'
                f'📅 Next visit: {nvd_str}</span>'
                if nvd_str else ""
            )
            note_line = (
                f'<div style="font-size:13px;color:#2C3E50;margin:2px 0 6px 0;">{notes}</div>'
                if notes else ""
            )
            entries_html += (
                f'<div style="border-left:3px solid #2E86C1;padding:3px 0 3px 10px;margin-bottom:6px;">'
                f'<span style="font-size:11px;color:#7F8C8D;">{ts} · <strong>{by}</strong></span>'
                f' &nbsp;{badge}{nvd_badge}{note_line}</div>'
            )

    st.markdown(
        blue_header(f"📞 Call Log — {lead_name} (lead #{lead_id})")
        + entries_html
        + BLUE_FOOTER,
        unsafe_allow_html=True,
    )

    if not logs_df.empty and st.session_state.get("user_role") == "admin":
        del_options = {
            f"{pd.to_datetime(r['called_at']).strftime('%Y-%m-%d %H:%M')} · "
            f"{r.get('outcome','—')} · {str(r.get('notes',''))[:40]}": int(r["id"])
            for _, r in logs_df.iterrows()
        }
        dc1, dc2 = st.columns([5, 1])
        del_sel = dc1.selectbox(
            "Delete a log entry",
            ["— select to delete —"] + list(del_options.keys()),
            key=f"lead_del_sel_{lead_id}{key_suffix}",
            label_visibility="collapsed",
        )
        if dc2.button("🗑 Delete", key=f"lead_del_btn_{lead_id}{key_suffix}") and del_sel != "— select to delete —":
            if delete_lead_call_log_entry(del_options[del_sel]):
                bust_lead_call_log_cache()
                st.rerun()

    with st.form(f"lead_call_log_form_{lead_id}{key_suffix}", clear_on_submit=True):
        fc1, fc2, fc3, fc4 = st.columns([2, 2, 3, 1])
        outcome = fc1.selectbox("Outcome", LEAD_OUTCOMES, key=f"lead_outcome_{lead_id}{key_suffix}")
        next_visit = fc2.date_input(
            "Next visit", value=None, key=f"lead_nvd_{lead_id}{key_suffix}",
        )
        notes = fc3.text_input("Notes", placeholder="What did they say?", key=f"lead_notes_{lead_id}{key_suffix}")
        fc4.markdown("<br>", unsafe_allow_html=True)
        if fc4.form_submit_button("Save"):
            if save_lead_call_log(zid, lead_id, outcome, next_visit, notes):
                bust_lead_call_log_cache()
                st.success("Call logged.")
                st.rerun()
            else:
                st.error("Failed to save — check DB connection.")
