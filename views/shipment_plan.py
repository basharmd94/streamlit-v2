# views/shipment_plan.py
# Purchase Analysis -> "🗓 Shipment Plan". The commercial officer's input to the cash projection:
#   * open import purchases (IP--, status 1-Open, WHOLE-shipment value) with an expected arrival month
#     that can be moved for holidays / delays (Chinese New Year, the two Eids, Pooja) and a note;
#   * extra planned shipments (whole-shipment vendor value + month) to size future shipments;
#   * holiday / disruption notes.
# Everything is saved to data/shipment_plan.json (settings, not results). The Financial Statements
# pages are NOT needed for this — the projection reads the saved plan.
# New items may have no cost in the ERP yet: the cost must be on the open purchase itself, so any
# open IP with a line lacking a cost / not in the catalogue / not adding up to its total is flagged.

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from core.analytics import Analytics
from processing import cash_projection as cp


def _month_options(first: tuple, n: int = 24) -> list:
    return [cp.month_str(m) for m in [first] + cp.next_months(first, n - 1)]


def render_shipment_plan(zid: str) -> None:
    zid = str(zid)
    st.subheader("🗓 Shipment Plan — input to the cash projection")
    st.caption("Open import purchases carry the **whole shipment's** value (they are split between HMBR and Gulshan Packaging only "
               "when the goods arrive). Set when each one is expected to arrive — move it for the Chinese New Year, the two Eids or the "
               "Pooja holidays — and add shipments you are planning. The finance team's projection uses this plan.")

    now = datetime.now()
    first = (now.year, now.month)
    plan = cp.load_plan(zid)
    months = _month_options(first)

    orders = Analytics("open_import_orders", zid=zid).data
    orders = orders if orders is not None else pd.DataFrame()

    # ── open import purchases ────────────────────────────────────────────────────
    st.markdown("#### Open import purchases")
    sched = cp.build_shipment_schedule(orders, {"overrides": {}, "planned": []}, first)   # as the ERP has them
    overrides = {}
    if orders.empty:
        st.info("No open import purchases (IP--) with a shipment number.")
    else:
        for r in orders.itertuples(index=False):
            po = str(r.po)
            cur = plan["overrides"].get(po, {})
            base = sched[sched["ref"].str.startswith(po)].iloc[0]
            warn = base["warning"]
            with st.container(border=True):
                c = st.columns([2.2, 1.6, 1.6, 1.6, 3])
                c[0].markdown(f"**{po}**  \n{str(r.shipment).split(',')[0]}  \nERP date {pd.Timestamp(r.po_date):%d-%b-%Y}")
                c[1].metric("Value in the ERP", f"{float(r.total or 0):,.0f}")
                default_month = cp.month_str(base["month"])
                chosen_default = cur.get("month") if cur.get("month") in months else (default_month if default_month in months else months[0])
                month = c[2].selectbox("Expected arrival", months, index=months.index(chosen_default), key=f"sp_m_{po}")
                value = c[3].number_input("Value override (0 = ERP)", min_value=0.0, value=float(cur.get("value") or 0.0),
                                          step=100000.0, key=f"sp_v_{po}")
                note = c[4].text_input("Note (e.g. Eid delay)", value=cur.get("note", ""), key=f"sp_n_{po}")
                if warn:
                    st.warning(f"⚠️ Cost data incomplete on {po}: {warn}. New items need their cost on the open purchase — "
                               "please complete it before this plan is relied on.")
                if pd.Timestamp(r.po_date) < pd.Timestamp(first[0], first[1], 1):
                    st.caption("The ERP date is already past; the projection treats it as arriving this month unless you move it.")
                overrides[po] = {"month": month, "value": value or None, "note": note}

    # ── planned shipments ───────────────────────────────────────────────────────────
    st.markdown("#### Planned shipments (not yet in the ERP)")
    planned = list(plan.get("planned", []))
    for i, p in enumerate(planned):
        c = st.columns([3, 2, 2, 3, 1])
        c[0].markdown(p.get("ref", "Planned shipment"))
        c[1].markdown(f"{float(p.get('value', 0)):,.0f}")
        c[2].markdown(p.get("month", ""))
        c[3].markdown(p.get("note", "") or "—")
        if c[4].button("✕", key=f"sp_pdel_{zid}_{i}"):
            planned.pop(i)
            cp.save_plan(zid, {**plan, "planned": planned, "overrides": overrides or plan["overrides"]}, st.session_state.get("username"))
            st.rerun()
    with st.form(f"sp_pform_{zid}", clear_on_submit=True):
        c = st.columns([3, 2, 2, 3])
        ref = c[0].text_input("Name (e.g. MDKF009/26)")
        val = c[1].number_input("Whole-shipment vendor value", min_value=0.0, step=100000.0)
        mon = c[2].selectbox("Expected arrival", months)
        note = c[3].text_input("Note")
        if st.form_submit_button("➕ Add planned shipment") and val > 0:
            planned.append({"ref": ref or "Planned shipment", "value": val, "month": mon, "note": note})
            cp.save_plan(zid, {**plan, "planned": planned, "overrides": overrides or plan["overrides"]}, st.session_state.get("username"))
            st.rerun()

    # ── holidays / disruptions ───────────────────────────────────────────────────────
    st.markdown("#### Holidays & disruptions to keep in mind")
    holidays = list(plan.get("holidays", []))
    for i, h in enumerate(holidays):
        c = st.columns([3, 2, 2, 3, 1])
        c[0].markdown(h.get("label", ""))
        c[1].markdown(str(h.get("from", "")))
        c[2].markdown(str(h.get("to", "")))
        c[3].markdown(h.get("note", "") or "—")
        if c[4].button("✕", key=f"sp_hdel_{zid}_{i}"):
            holidays.pop(i)
            cp.save_plan(zid, {**plan, "holidays": holidays, "overrides": overrides or plan["overrides"]}, st.session_state.get("username"))
            st.rerun()
    with st.form(f"sp_hform_{zid}", clear_on_submit=True):
        c = st.columns([3, 2, 2, 3])
        label = c[0].text_input("Holiday (e.g. Chinese New Year, Eid-ul-Fitr, Pooja)")
        d1 = c[1].date_input("From", value=now.date())
        d2 = c[2].date_input("To", value=now.date())
        hnote = c[3].text_input("Effect on shipments")
        if st.form_submit_button("➕ Add holiday") and label:
            holidays.append({"label": label, "from": str(d1), "to": str(d2), "note": hnote})
            cp.save_plan(zid, {**plan, "holidays": holidays, "overrides": overrides or plan["overrides"]}, st.session_state.get("username"))
            st.rerun()

    # ── save ───────────────────────────────────────────────────────────────────────────
    if st.button("💾 Save shipment plan", type="primary", key=f"sp_save_{zid}"):
        cp.save_plan(zid, {"overrides": overrides, "planned": planned, "holidays": holidays}, st.session_state.get("username"))
        st.success("Shipment plan saved — the cash projection now uses it.")
    if plan.get("updated_at"):
        st.caption(f"Last saved {plan['updated_at']} by {plan.get('updated_by') or '—'}")
