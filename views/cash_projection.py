# views/cash_projection.py
# Cash Model -> "📈 Projection" (a clean table) and "⚙️ Settings" (every input, last tab).
#   * Projection: the WHOLE Final Model (all actual months up to the latest completed month) with
#     12 projected months appended on the same rows; notes at the end (to be defined).
#   * Settings: shipments in the projection, import financing, drivers, other cash lines, known items,
#     debts & payment priority, cash floor. Only settings are saved (data/cash_projection_settings.json);
#     the shipment plan is entered by the commercial officer in Purchase Analysis -> "🗓 Shipment Plan".
# All maths lives in processing/cash_projection.py.

from __future__ import annotations

import calendar
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from core.analytics import Analytics
from processing import cash_model as cm
from processing import cash_projection as cp
from views import cash_model as cmv

_DRIVER_LABELS = {
    "pay_order_pct": ("Pay order (capitalised duty) / vendor value", "pct"),
    "charges_pct": ("L/C charges & other import costs / vendor value", "pct"),
    "rebate_pct": ("VAT/AT on pay order (reclaimable, cash) / vendor value", "pct"),
    "sales_price_per_landed": ("Inventory at sales price per taka of landed + local purchases", "x"),
    "pct_sold": ("% of opening inventory sold (net sales)", "pct"),
    "return_rate": ("Returns / gross sales", "pct"),
    "discount_rate": ("Discount / gross sales", "pct"),
    "delivery_per_gross": ("Deliveries out of stock / gross sales", "x"),
    "sr_per_return": ("Returns back into stock / returns (AR)", "x"),
    "collections_pct": ("Customer receipts / net sales", "pct"),
}
_MODES = {"instalment": "fixed amount per month", "lump": "one amount in a month", "sweep": "whatever cash is left", "off": "not paid"}


def _current_month() -> tuple:
    now = datetime.now()
    return (now.year, now.month)


@st.cache_data(show_spinner=False, ttl=1800)
def _history(members, year: int, month: int):
    """Final Model through the CURRENT month (month to date) + what the projection needs from the app.

    The last column is the current month to date; everything before it is complete. The projection's
    averages and anchors (closing cash, debt balances, LTR) use the last COMPLETED month.
    """
    zids = [z for z, _ in members]
    projects = [p for _, p in members]
    m = cmv.build_cash_model(zids, projects, year, month)
    months, cols, final = m["months"], m["cols"], m["final"]
    has_mtd = bool(m.get("mtd_col"))
    done_idx = -2 if has_mtd else -1
    last_done = months[done_idx]
    ltr, debts = 0.0, {}
    for z, p in members:
        one = cmv._app_cash_one(z, p, year, month)
        ltr += abs(one.get("ltr", {}).get(last_done, 0.0))
        for code, mp in (one.get("debts") or {}).items():
            debts[code] = debts.get(code, 0.0) + abs(mp.get(last_done, 0.0))
    app_close = float(cp._row(final, "Closing cash & CE — APP (source of truth)", [cols[done_idx]])[0])
    return {"final": final, "cols": cols, "months": months, "shipments": m["shipments"], "has_mtd": has_mtd,
            "done_cols": cols[:done_idx + 1] if has_mtd else cols, "last_done": last_done,
            "app_close": app_close, "ltr_open": ltr, "app_debts": debts}


@st.cache_data(show_spinner=False, ttl=600)
def _open_orders(zid: str):
    d = Analytics("open_import_orders", zid=zid).data
    return d if d is not None else pd.DataFrame()


def prepare(ent: str, zids: list, projects: list):
    """Everything both tabs need (history, plan, shipment schedule); None when it cannot be built."""
    year, month = _current_month()
    try:
        with st.spinner("Building the history and projection…"):
            h = _history(tuple(zip(zids, projects)), year, month)
    except Exception as e:
        st.warning(f"Could not build the projection base for {ent}: {e}")
        return None
    first = cp.next_months(h["last_done"], 1)[0]            # the first projected month = the current month
    plan = cp.load_plan(str(ent))
    schedule = cp.build_shipment_schedule(_open_orders(str(ent)), plan, first)
    return {"ent": str(ent), "h": h, "plan": plan, "schedule": schedule, "first": first,
            "saved": cp._read(cp.SETTINGS_PATH).get(str(ent), {})}


# ── ⚙️ Settings ──────────────────────────────────────────────────────────────

def _shipments_block(ctx) -> None:
    sch, plan = ctx["schedule"], ctx["plan"]
    st.markdown("#### Shipments in the projection")
    st.caption("From the open import purchases and the commercial officer's plan (Purchase Analysis → 🗓 Shipment Plan). "
               "Values are whole-shipment vendor values.")
    if sch.empty:
        st.info("No open import purchases and no planned shipments — imports will be zero.")
    else:
        show = sch.copy()
        show["month"] = show["month"].map(cp.month_str)
        show["value"] = show["value"].map(lambda v: f"{v:,.0f}")
        st.dataframe(show.rename(columns={"ref": "Shipment", "source": "Source", "value": "Whole-shipment value",
                                          "month": "Arrival", "note": "Note", "warning": "Cost-data check"}),
                     use_container_width=True, hide_index=True)
    if plan.get("holidays"):
        st.caption("Holidays / disruptions noted: " + " · ".join(
            f"{x.get('label')} ({x.get('from')} → {x.get('to')})" for x in plan["holidays"]))


def _imports_block(ent, s) -> None:
    imp = s["imports"]
    st.markdown("#### Import financing — L/C margin → LTR → TT")
    st.caption("Per shipment of whole vendor value V: L/C value = L/C % × V; the margin is paid when the L/C is opened; the rest of the "
               "L/C value becomes an LTR when the documents arrive (the bank pays the supplier — no cash) and is repaid within the LTR "
               "limit; the remaining part of V is paid by TT the month after the LTR. Customs and charges fall in the arrival month.")
    c = st.columns(4)
    imp["lc_pct"] = c[0].slider("L/C value % of shipment", 30, 80, int(round(imp["lc_pct"] * 100)), key=f"cp_lc_{ent}") / 100
    imp["margin_pct"] = c[1].slider("L/C margin % of L/C value", 5, 50, int(round(imp["margin_pct"] * 100)), key=f"cp_mg_{ent}") / 100
    imp["ltr_months"] = c[2].slider("LTR repaid within (months)", 1, 3, int(imp["ltr_months"]), key=f"cp_ltrm_{ent}")
    imp["tt_after_ltr"] = c[3].slider("TT paid N months after LTR", 0, 3, int(imp["tt_after_ltr"]), key=f"cp_tt_{ent}")
    c = st.columns(4)
    imp["lc_lead"] = c[0].slider("L/C opened N months before arrival", 0, 4, int(imp["lc_lead"]), key=f"cp_lcl_{ent}")
    imp["ltr_doc_lead"] = c[1].slider("Documents (LTR) N months before arrival", 0, 3, int(imp["ltr_doc_lead"]), key=f"cp_dl_{ent}")
    c = st.columns(4)
    imp["ltr_rate"] = c[1].number_input("LTR interest % p.a.", value=float(imp.get("ltr_rate", 0.1455)) * 100, step=0.25,
                                        key=f"cp_ltrr_{ent}", help="Accrues monthly on the outstanding LTR, paid in cash at quarter-end "
                                                                   "(HMBR debt model).") / 100
    imp["margin_funding"] = c[2].radio("L/C margin paid from", ["cc", "cash"], index=["cc", "cash"].index(imp["margin_funding"]),
                                       horizontal=True, key=f"cp_mf_{ent}",
                                       help="'cc' = the Dhaka CC up to its target headroom, the rest in cash.")
    imp["customs_funding"] = c[3].radio("Customs paid from", ["cc", "cash"], index=["cc", "cash"].index(imp["customs_funding"]),
                                        horizontal=True, key=f"cp_cf_{ent}")


def _drivers_block(ent, s, auto) -> None:
    st.markdown("#### Drivers — measured from the last 3 months / shipments, override if you wish")
    for k, (label, kind) in _DRIVER_LABELS.items():
        cur = s["drivers"].get(k, {"mode": "auto", "value": None})
        a = auto.get(k, 0.0)
        c = st.columns([4, 1.4, 1.4, 1.6])
        c[0].markdown(label)
        c[1].markdown(f"auto: **{a:.1%}**" if kind == "pct" else f"auto: **{a:.2f}×**")
        mode = c[2].selectbox("mode", ["auto", "manual"], index=0 if cur.get("mode") != "manual" else 1,
                              key=f"cp_dm_{k}_{ent}", label_visibility="collapsed")
        if kind == "pct":
            v = c[3].number_input("value %", value=float((cur.get("value") if cur.get("value") is not None else a) * 100),
                                  step=0.5, key=f"cp_dv_{k}_{ent}", label_visibility="collapsed")
            val = v / 100
        else:
            val = c[3].number_input("value ×", value=float(cur.get("value") if cur.get("value") is not None else a),
                                    step=0.01, key=f"cp_dv_{k}_{ent}", label_visibility="collapsed")
        s["drivers"][k] = {"mode": mode, "value": float(val)}


def _lines_block(ent, s, line_labels) -> None:
    st.markdown("#### Other cash lines — average of the last 3 months, adjustable")
    st.caption("avg3 = average of the last 3 actual months · zero = nothing expected (one-off lines start here; put known amounts in "
               "the list below) · fixed = a set amount per month. 'Adjust %' trims or raises a line (e.g. −10 = pay a little less). "
               "With the debt model on, the CC / loan repayment lines and the Malek interest are driven by the debt settings instead.")
    for direction, labels in line_labels.items():
        st.markdown("**Inflows**" if direction == "in" else "**Payments**")
        for lab in labels:
            cfg = {**cp.default_line_setting(lab, direction), **s["lines"].get(lab, {})}
            c = st.columns([4, 1.4, 1.4, 1.6])
            c[0].markdown(lab.strip())
            mode = c[1].selectbox("mode", ["avg3", "zero", "fixed"], index=["avg3", "zero", "fixed"].index(cfg["mode"]),
                                  key=f"cp_lm_{direction}_{lab}_{ent}", label_visibility="collapsed")
            adj = c[2].number_input("adjust %", value=float(cfg["adjust_pct"]), step=5.0,
                                    key=f"cp_la_{direction}_{lab}_{ent}", label_visibility="collapsed")
            fx = c[3].number_input("fixed", value=float(cfg["fixed"]), step=10000.0,
                                   key=f"cp_lf_{direction}_{lab}_{ent}", label_visibility="collapsed")
            s["lines"][lab] = {"mode": mode, "adjust_pct": adj, "fixed": fx}


def _known_block(ent, s, line_labels) -> None:
    st.markdown("#### Known items — amounts owed to us / by us (with month)")
    items = list(s.get("known_items", []))
    for i, it in enumerate(items):
        c = st.columns([3, 1, 2, 2, 3, 1])
        c[0].markdown(it.get("label", ""))
        c[1].markdown("in" if it.get("direction") == "in" else "out")
        c[2].markdown(f"{float(it.get('amount', 0)):,.0f}")
        c[3].markdown(str(it.get("month", "")))
        c[4].markdown((it.get("line") or "").strip() or "—")
        if c[5].button("✕", key=f"cp_kdel_{ent}_{i}"):
            items.pop(i)
            s["known_items"] = items
            cp.save_settings(ent, s)
            st.rerun()
    with st.form(f"cp_kform_{ent}", clear_on_submit=True):
        c = st.columns([3, 1, 2, 2, 3])
        label = c[0].text_input("Label")
        direction = c[1].selectbox("Direction", ["in", "out"])
        amount = c[2].number_input("Amount", min_value=0.0, step=10000.0)
        month = c[3].text_input("Month (YYYY-MM)")
        line = c[4].selectbox("Goes into line", ["(Other inflows / Other outflows)"] + [l for ls in line_labels.values() for l in ls])
        if st.form_submit_button("Add known item") and label and cp.parse_month(month) and amount > 0:
            items.append({"label": label, "direction": direction, "amount": amount, "month": month,
                          "line": None if line.startswith("(") else line})
            s["known_items"] = items
            cp.save_settings(ent, s)
            st.rerun()
    s["known_items"] = items


def _debts_block(ent, s, ctx) -> None:
    debts = s["debts"]
    st.markdown("#### Debts & payment priority")
    if not debts.get("enabled"):
        st.caption("The debt model (limits, interest, priority) is configured for the HMBR group only so far; this business uses the "
                   "3-month averages for its loan repayment lines.")
        return
    st.caption("Each month, cash above the floor pays the obligations **in priority order** (1 = first); what cannot be paid is carried as "
               "arrears. If cash would fall below the floor the Dhaka CC is drawn, up to its target (or the sanctioned limit if allowed). "
               "CC interest is added to the CC balance at quarter-end (no cash). Balances come from the app unless overridden.")
    items = debts["items"]
    app_bal = cp.debt_opening_balances({"items": {k: {"balance": None} for k in items}}, ctx["h"]["app_debts"])
    order = {k: i + 1 for i, k in enumerate(debts.get("priority", []))}
    head = st.columns([2.4, 1.2, 1.5, 1.6, 1.6, 1.4, 0.9])
    for col, t in zip(head, ["Instrument", "Balance (app)", "Balance override", "How it is paid", "Amount / month", "Interest % p.a.", "Priority"]):
        col.markdown(f"**{t}**")
    for k in list(order) + [k for k in items if k not in order]:
        it = items[k]
        c = st.columns([2.4, 1.2, 1.5, 1.6, 1.6, 1.4, 0.9])
        c[0].markdown(cp.DEBT_NAMES.get(k, k))
        c[1].markdown(f"{app_bal.get(k, 0.0):,.0f}")
        ov = c[2].number_input("override", value=float(it["balance"]) if it.get("balance") not in (None, "") else 0.0, step=100000.0,
                               key=f"cp_db_{k}_{ent}", label_visibility="collapsed", help="0 = use the app balance")
        it["balance"] = ov if ov > 0 else None
        keys = list(_MODES)
        it["mode"] = c[3].selectbox("mode", keys, index=keys.index(it.get("mode", "off")), format_func=lambda m: _MODES[m],
                                    key=f"cp_dmode_{k}_{ent}", label_visibility="collapsed")
        if it["mode"] == "instalment":
            it["monthly"] = c[4].number_input("monthly", value=float(it.get("monthly") or 0.0), step=50000.0,
                                              key=f"cp_dmon_{k}_{ent}", label_visibility="collapsed")
        elif it["mode"] == "lump":
            it["amount"] = c[4].number_input("amount", value=float(it.get("amount") or 0.0), step=50000.0,
                                             key=f"cp_damt_{k}_{ent}", label_visibility="collapsed")
        else:
            c[4].markdown("—")
        it["rate"] = c[5].number_input("rate %", value=float(it.get("rate") or 0.0) * 100, step=0.25,
                                       key=f"cp_drate_{k}_{ent}", label_visibility="collapsed") / 100
        order[k] = c[6].number_input("prio", min_value=1, max_value=20, value=int(order.get(k, 99)) if order.get(k, 99) <= 20 else len(order),
                                     step=1, key=f"cp_dprio_{k}_{ent}", label_visibility="collapsed")
        if it["mode"] == "lump":
            it["due_month"] = st.text_input(f"{cp.DEBT_NAMES.get(k, k)} — due month (YYYY-MM)", value=it.get("due_month", ""),
                                            key=f"cp_ddue_{k}_{ent}")
    debts["priority"] = [k for k, _ in sorted(order.items(), key=lambda kv: kv[1])]

    st.markdown("**Dhaka CC limits**")
    dh = items["dhaka"]
    c = st.columns(3)
    dh["target"] = c[0].number_input("Target — keep the CC under", value=float(dh.get("target") or 0.0), step=500000.0, key=f"cp_dh_t_{ent}")
    dh["limit"] = c[1].number_input("Sanctioned limit", value=float(dh.get("limit") or 0.0), step=1000000.0, key=f"cp_dh_l_{ent}")
    dh["use_limit"] = c[2].checkbox("Allow drawing up to the sanctioned limit (instead of the target)", value=bool(dh.get("use_limit")),
                                    key=f"cp_dh_u_{ent}")
    debts["lc_limit"] = st.number_input("L/C limit (an L/C counts until its LTR is drawn)", value=float(debts.get("lc_limit") or 0.0),
                                        step=1000000.0, key=f"cp_lclim_{ent}")

    st.markdown("**Money deposited straight into the bank loans** (no cash — each one goes only where you enter it; "
                "nothing is spilled over automatically)")
    dest_names = {"ucb": "UCB CC", "dhaka": "Dhaka CC", "cash": "Cash (bank balance)"}
    deps = list(debts.get("deposits", []))
    for i, d in enumerate(deps):
        c = st.columns([3, 2, 2, 2, 1])
        c[0].markdown(d.get("label", ""))
        c[1].markdown(f"{float(d.get('amount', 0)):,.0f}")
        c[2].markdown(d.get("month") or "⚠️ month not set — not in the projection")
        c[3].markdown(dest_names.get(d.get("dest", "ucb"), "UCB CC"))
        if c[4].button("✕", key=f"cp_depdel_{ent}_{i}"):
            deps.pop(i)
            debts["deposits"] = deps
            cp.save_settings(ent, s)
            st.rerun()
    with st.form(f"cp_depform_{ent}", clear_on_submit=True):
        c = st.columns([3, 2, 2, 2])
        label = c[0].text_input("Label (e.g. Hospital repayment — UCB part)")
        amount = c[1].number_input("Amount", min_value=0.0, step=1000000.0)
        month = c[2].text_input("Expected month (YYYY-MM)")
        dest = c[3].selectbox("Goes to", list(dest_names), format_func=lambda k: dest_names[k])
        if st.form_submit_button("Add deposit") and label and amount > 0:
            deps.append({"label": label, "amount": amount, "month": month if cp.parse_month(month) else "", "dest": dest})
            debts["deposits"] = deps
            cp.save_settings(ent, s)
            st.rerun()
    debts["deposits"] = deps


def render_settings(ctx) -> dict:
    """All the inputs of the projection; returns the live settings dict (saved only on the button)."""
    ent, h = ctx["ent"], ctx["h"]
    s = cp._merge(cp.default_settings_for(ent), ctx["saved"])
    st.caption("Every input of the projection lives here, so the Projection tab stays a clean table. Changes apply immediately to the "
               "projection; **Save settings** keeps them for next time. Only settings are saved — figures are always recomputed.")
    _shipments_block(ctx)
    _imports_block(ent, s)
    auto = cp.auto_drivers(h["final"], h["done_cols"], h["shipments"])
    _drivers_block(ent, s, auto)
    line_labels = {
        "in": [r[2] for r in cm.FINAL_MODEL_ROWS if r[0] == "cash" and r[1] == "add" and r[2] and r[2].startswith("   ")],
        "out": [r[2] for r in cm.FINAL_MODEL_ROWS if r[0] == "cash" and r[1] == "out" and r[2] and r[2].startswith("   ")
                and r[2] not in ("   International: L/C margin & charges", "   International: TT / advances to vendors",
                                 "   International: pay order (customs tax & VAT)", "   Bank loans & LTR repaid")],
    }
    if s["debts"].get("enabled"):
        engine_rows = {"   Bank overdraft (CC) repaid", "   Bank overdraft (CC) drawn", "   Other-party loans (0904: Malek Jobeda, Water Lily …)",
                       "   Loans repaid (directors, group, others)", "   Interest & bank charges"}
        line_labels["out"] = [l for l in line_labels["out"] if l not in engine_rows - {"   Interest & bank charges"}]
        line_labels["in"] = [l for l in line_labels["in"] if l != "   Bank overdraft (CC) drawn"]
    _lines_block(ent, s, line_labels)
    _known_block(ent, s, line_labels)
    _debts_block(ent, s, ctx)
    st.markdown("#### Cash floor")
    s["cash_floor"] = st.number_input("Minimum cash to keep — the CC is drawn to hold it; surplus above it pays the debts",
                                      value=float(s.get("cash_floor", 0.0)), step=500000.0, key=f"cp_floor_{ent}")
    b1, b2, _ = st.columns([1, 1, 6])
    if b1.button("💾 Save settings", key=f"cp_save_{ent}"):
        cp.save_settings(ent, s)
        st.success("Settings saved.")
    if b2.button("↩️ Reset to defaults", key=f"cp_reset_{ent}"):
        cp.save_settings(ent, {})
        for k in [k for k in st.session_state if str(k).startswith("cp_") and str(k).endswith(f"_{ent}")]:
            del st.session_state[k]
        st.rerun()
    return s


# ── 📈 Projection (clean) ─────────────────────────────────────────────────────

def render_projection(ctx, settings) -> None:
    h, sch = ctx["h"], ctx["schedule"]
    done = h["done_cols"]
    now = datetime.now()
    mtd = ({"col": h["cols"][-1], "elapsed": now.day, "days": calendar.monthrange(now.year, now.month)[1]}
           if h["has_mtd"] else None)
    table, pcols, info = cp.project(h["final"], done, h["last_done"], h["app_close"], sch, settings,
                                    ltr_open=h["ltr_open"], shipments_hist=h["shipments"], app_debts=h["app_debts"], mtd=mtd)
    trans = (f" · **{pcols[0]}** is the transition month: what has been entered so far is real, the rest is projected "
             f"(sales linearly from the {mtd['elapsed']} days elapsed of {mtd['days']}; other lines to their average when that is larger)"
             if mtd else "")
    st.caption(f"Actuals **{done[0]} → {done[-1]}** ({len(done)} complete months, as in the Final Model){trans} · "
               f"**{len(pcols)} months projected** from the app's closing cash of {h['app_close']:,.0f}. Inputs are in the ⚙️ Settings tab.")
    bad = sch[sch["warning"] != ""] if not sch.empty else sch
    if len(bad):
        st.warning("⚠️ Open purchases with incomplete cost data — " + "; ".join(f"{r.ref}: {r.warning}" for r in bad.itertuples())
                   + ". The cost must be on the open purchase (commercial officer).")
    if info.get("shortfall_months"):
        st.error("Cash is short of the floor (the Dhaka CC has no headroom left) in: " + ", ".join(info["shortfall_months"]))
    elif info["floor_breaches"]:
        st.error("Closing cash falls below the floor in: " + ", ".join(info["floor_breaches"]))
    dbg = info.get("debts")
    if dbg is not None and float(np.sum(dbg["dep_unapplied"])) > 1.0:
        months_ = [cm.month_label(m) for m, x in zip(info["proj_months"], dbg["dep_unapplied"]) if x > 1.0]
        st.warning(f"A deposit is larger than the balance of the loan it was entered for ({', '.join(months_)}): "
                   f"{float(np.sum(dbg['dep_unapplied'])):,.0f} is not applied anywhere — enter where the rest goes (Settings → deposits).")
    if info.get("lc_limit_breaches"):
        st.warning("L/C value open exceeds the L/C limit in: " + ", ".join(info["lc_limit_breaches"]))

    sty = cmv._style(table, done + pcols).set_properties(subset=pcols, **{"background-color": "rgba(80,140,255,0.10)"})
    st.dataframe(sty, use_container_width=True, hide_index=True, height=min(35 * (len(table) + 1) + 3, 3200),
                 column_config={"Line": st.column_config.TextColumn("Line", pinned=True, width=430)})
    st.download_button("⬇️ Download projection (Excel)", data=cm.to_excel_bytes({"Projection": table}),
                       file_name=f"cash_projection_{ctx['ent']}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown("#### Notes")
    st.caption("Notes will be added here once the projection has been studied.")
