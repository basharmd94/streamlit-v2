"""Cash projection — pure pandas engine for the Cash Model's "📈 Projection" tab.

Builds 12 projected months on the SAME rows as the Final Model, continuing from the last
actual month. Nothing here is stored as a result: only the settings (finance) and the
shipment plan (commercial officer) are saved (JSON); every figure is recomputed from actuals
plus those settings.

Chain (driver based, each driver defaults to the history and can be overridden):
    shipment plan (whole-shipment vendor value + arrival month)
      -> landed cost (pay order %, charges % of vendor value, last 3 shipments)
      -> inventory receipts valued at average sales price (sales price per landed taka)
      -> net sales = % of opening inventory sold (last 3 months); gross = net / (1 - returns - discount)
      -> collections = ratio of net sales (last 3 months)
      -> payments: shipment-driven import payments + averages of the other lines (adjustable)
      -> cash

Import financing (user-confirmed, per shipment of whole vendor value V):
    L/C value = lc_pct * V (50-60% of V); margin = margin_pct * L/C value, paid when the L/C is
    opened (`lc_lead` months before arrival). The rest of the L/C value is an LTR created when the
    documents arrive (`ltr_doc_lead` before arrival; the bank pays the supplier, no cash) and repaid
    within `ltr_months` (max 3). The balance of V (the TT part) is paid by TT the month after the
    LTR (`tt_after_ltr`). Customs (pay order incl. the reclaimable VAT/AT) and charges fall in the
    arrival month. The L/C margin and customs can be funded from cash or from the bank overdraft
    (CC): from CC no cash moves (it shows as a memo and is repaid later through "CC repaid").

No st.* calls (processing-layer rule).
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from processing import cash_model as cm

SETTINGS_PATH = "data/cash_projection_settings.json"
PLAN_PATH = "data/shipment_plan.json"
HORIZON_MONTHS = 12
AVG_N = 3                         # months / shipments averaged for every "auto" driver

# ── defaults ────────────────────────────────────────────────────────────────
# Lines that are lumpy (one-off) default to zero instead of a 3-month average, which would
# repeat a single big payment every month; known amounts are added through `known_items`.
_LUMPY_IN = {"   Hospital loan repayments (capital inflow)", "   Loans received (directors, group, others)",
             "   Bank overdraft (CC) drawn", "   Other-party funds received (0904)",
             "   Import advances refunded / VAT rebate received", "   Other income & interest",
             "   Other inflows", "   GI cash passing through HMBR (09060001)"}
_LUMPY_OUT = {"   Other-party loans (0904: Malek Jobeda, Water Lily …)", "   Loans given (group, directors, others)",
              "   Loans repaid (directors, group, others)", "   Capital expenditure (assets)",
              "   Other outflows", "   GI cash passing through HMBR (09060001)"}

DEFAULT_SETTINGS: Dict = {
    "horizon_months": HORIZON_MONTHS,
    "imports": {
        "lc_pct": 0.55,            # L/C value as a share of the shipment's vendor value (50-60%)
        "margin_pct": 0.20,        # L/C margin paid in cash/CC when the L/C is opened
        "lc_lead": 1,              # months before arrival that the L/C is opened (margin paid)
        "ltr_doc_lead": 1,         # months before arrival that documents come in (LTR created)
        "ltr_months": 3,           # LTR must be paid within this many months (max 3)
        "tt_after_ltr": 1,         # TT balance paid this many months after the LTR is created
        "ltr_rate": 0.1455,        # LTR interest p.a.; accrues monthly on the outstanding LTR, paid in cash at quarter-end
        "margin_funding": "cc",    # 'cc' | 'cash'   (2026 actuals: the margin was paid from the overdraft)
        "customs_funding": "cc",   # 'cc' | 'cash'
    },
    # each driver: auto = measured from the last 3 months/shipments, or a manual value
    "drivers": {k: {"mode": "auto", "value": None} for k in
                ("pay_order_pct", "charges_pct", "rebate_pct", "sales_price_per_landed", "pct_sold",
                 "return_rate", "discount_rate", "delivery_per_gross", "sr_per_return", "collections_pct")},
    "debts": {"enabled": False},   # see HMBR_DEBTS; only the HMBR group has its instruments configured so far
    "lines": {},                   # label -> {"mode": "avg3"|"zero"|"fixed", "adjust_pct": 0, "fixed": 0}
    "known_items": [],             # {"label", "direction": "in"|"out", "amount", "month": "YYYY-MM", "line"}
    "cash_floor": 0.0,
}


# App accounts behind each debt instrument (balances are read from the app, liabilities are negative there)
DEBT_ACCOUNTS = {"dhaka": "10010003", "ucb": "10010006", "mdod": "10010007", "malek": "09040007",
                 "sunflower": "09040001", "mdloan": "10020001", "fixit": "10020002", "director": "10020016"}
DEBT_NAMES = {"dhaka": "CC Dhaka Bank", "ucb": "CC UCB (bulk, over its limit)", "mdod": "Overdraft by MD Sir",
              "malek": "Malek Jobeda Foundation", "sunflower": "Sunflower", "mdloan": "Loan from MD Sir",
              "fixit": "Loan from Fix-iT", "director": "Loan from Director (Md Anil Saddat)"}
# Row of the cash statement each instrument's CASH repayment lands in
DEBT_CASH_ROW = {"dhaka": "   Bank overdraft (CC) repaid", "ucb": "   Bank overdraft (CC) repaid",
                 "mdod": "   Bank overdraft (CC) repaid", "malek": "   Other-party loans (0904: Malek Jobeda, Water Lily …)",
                 "sunflower": "   Other-party loans (0904: Malek Jobeda, Water Lily …)",
                 "mdloan": "   Loans repaid (directors, group, others)", "fixit": "   Loans repaid (directors, group, others)",
                 "director": "   Loans repaid (directors, group, others)"}

# HMBR (100001 + 100009) terms as given by the owner. mode: instalment (a fixed total per month, interest
# included) | lump (one amount in due_month) | sweep (takes whatever surplus cash is left) | off.
HMBR_DEBTS: Dict = {
    "enabled": True,
    "lc_limit": 15_000_000,        # L/C limit; an L/C counts until its LTR is drawn
    "items": {
        "malek": {"mode": "instalment", "monthly": 1_000_000, "rate": 0.12, "balance": None},        # ~750k principal + ~250k interest
        "sunflower": {"mode": "instalment", "monthly": 500_000, "rate": 0.0, "balance": None},
        "mdloan": {"mode": "lump", "amount": 600_000, "due_month": "2026-11", "rate": 0.0, "balance": None},
        "mdod": {"mode": "instalment", "monthly": 200_000, "rate": 0.0, "balance": None},            # 400k/month split 100001 / 100000
        "ucb": {"mode": "off", "rate": 0.1455, "limit": 25_000_000, "balance": None},                # bulk, no longer used; paid by MANUAL entries only
        "dhaka": {"mode": "sweep", "rate": 0.1455, "target": 12_500_000, "limit": 50_000_000,
                  "use_limit": False, "balance": None},                                              # keep under the 12.5M target
        "fixit": {"mode": "off", "rate": 0.0, "balance": None},                                      # settled against goods, no timeline
        "director": {"mode": "sweep", "rate": 0.0, "balance": None},                                 # not a priority, paid when cash allows
    },
    "priority": ["malek", "sunflower", "mdloan", "mdod", "ucb", "dhaka", "director", "fixit"],
    # money coming in that is deposited straight into a bank loan account (no cash): applied to UCB first, then Dhaka CC
    # each deposit names its destination ('ucb' | 'dhaka' | 'cash'): nothing is spilled over automatically
    "deposits": [{"label": "Hospital repayment", "amount": 30_000_000, "month": "", "dest": "ucb"}],
}


def default_settings_for(zid: str) -> Dict:
    d = copy.deepcopy(DEFAULT_SETTINGS)
    if str(zid) == "100001":
        d["debts"] = copy.deepcopy(HMBR_DEBTS)
        d["cash_floor"] = 1_000_000.0          # kept AFTER payments; payouts come first
    return d


def default_line_setting(label: str, direction: str) -> Dict:
    lumpy = _LUMPY_IN if direction == "in" else _LUMPY_OUT
    return {"mode": "zero" if label in lumpy else "avg3", "adjust_pct": 0.0, "fixed": 0.0}


# ── JSON persistence (settings and plan; results are never saved) ───────────
def _read(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _write(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _merge(base: Dict, over: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(zid: str, path: str = SETTINGS_PATH) -> Dict:
    """Saved settings of the business (defaults filled in for anything missing)."""
    return _merge(default_settings_for(zid), _read(path).get(str(zid), {}))


def save_settings(zid: str, settings: Dict, path: str = SETTINGS_PATH) -> None:
    allv = _read(path)
    allv[str(zid)] = settings
    _write(path, allv)


def load_plan(zid: str, path: str = PLAN_PATH) -> Dict:
    p = _read(path).get(str(zid), {})
    return {"overrides": p.get("overrides", {}), "planned": p.get("planned", []),
            "holidays": p.get("holidays", []),
            "uniform": {"enabled": False, "value": 0.0, "include_open": False, **(p.get("uniform") or {})},
            "updated_by": p.get("updated_by"), "updated_at": p.get("updated_at")}


def save_plan(zid: str, plan: Dict, user: Optional[str] = None, path: str = PLAN_PATH) -> None:
    allv = _read(path)
    plan = dict(plan, updated_by=user, updated_at=datetime.now().isoformat(timespec="seconds"))
    allv[str(zid)] = plan
    _write(path, allv)


# ── months ───────────────────────────────────────────────────────────────────
def next_months(last: Tuple[int, int], n: int) -> List[Tuple[int, int]]:
    y, m = last
    out = []
    for _ in range(n):
        m += 1
        if m > 12:
            y, m = y + 1, 1
        out.append((y, m))
    return out


def parse_month(text: str) -> Optional[Tuple[int, int]]:
    try:
        y, m = str(text)[:7].split("-")
        return int(y), int(m)
    except (ValueError, AttributeError):
        return None


def month_str(ym: Tuple[int, int]) -> str:
    return f"{ym[0]}-{ym[1]:02d}"


# ── shipment schedule: open IPs + overrides + planned ones ───────────────────
def build_shipment_schedule(open_orders: pd.DataFrame, plan: Dict,
                            first_month: Tuple[int, int]) -> pd.DataFrame:
    """One row per shipment in the projection: ref, source, value (WHOLE shipment), month, note, warning.

    * Open IPs (status 1-Open, whole-shipment value) arrive in the month of their date; a date
      before the first projected month is treated as arriving in the first projected month.
    * The commercial officer's override can move the month (holidays, delays) or restate the value.
    * Planned shipments are extra shipments entered in the plan.
    * "Uniform" (plan["uniform"]): when enabled, ONE whole-shipment value applies to every planned
      shipment (and to the open IPs too when `include_open`); when disabled each shipment carries its own.
    A warning is attached when an open IP lacks cost data (zero rate, item not in the catalogue,
    lines not adding up to the total) — new items must carry their cost on the open purchase.
    """
    cols = ["ref", "source", "value", "month", "note", "warning"]
    rows = []
    ov = (plan or {}).get("overrides", {})
    uni = {"enabled": False, "value": 0.0, "include_open": False, **((plan or {}).get("uniform") or {})}
    uni_value = float(uni.get("value") or 0.0) if uni.get("enabled") else 0.0
    if open_orders is not None and not open_orders.empty:
        open_orders = open_orders.copy()
        for c in ("total", "lines_value", "lines", "lines_no_rate", "lines_not_in_catalog"):
            open_orders[c] = pd.to_numeric(open_orders[c], errors="coerce").fillna(0.0)
        for r in open_orders.itertuples(index=False):
            o = ov.get(str(r.po), {})
            m = parse_month(o.get("month")) or (pd.Timestamp(r.po_date).year, pd.Timestamp(r.po_date).month)
            if m < first_month:
                m = first_month
            value = o.get("value")
            value = float(value) if value not in (None, "") else float(r.total or 0)
            if uni_value > 0 and uni.get("include_open"):
                value = uni_value
            warn = []
            if int(r.lines or 0) == 0:
                warn.append("no item lines")
            if int(r.lines_no_rate or 0) > 0:
                warn.append(f"{int(r.lines_no_rate)} line(s) without a cost")
            if int(r.lines_not_in_catalog or 0) > 0:
                warn.append(f"{int(r.lines_not_in_catalog)} item(s) not in the catalogue")
            if int(r.lines or 0) and abs(float(r.lines_value or 0) - float(r.total or 0)) > max(1.0, 0.001 * float(r.total or 0)):
                warn.append("lines do not add up to the order total")
            if str(getattr(r, "currency", "BDT") or "BDT") != "BDT":
                warn.append(f"currency {r.currency}")
            rows.append({"ref": f"{r.po} ({str(r.shipment).split(',')[0]})", "source": "Open IP", "value": value,
                         "month": m, "note": o.get("note", ""), "warning": "; ".join(warn)})
    for p in (plan or {}).get("planned", []):
        m = parse_month(p.get("month"))
        value = uni_value if uni_value > 0 else float(p.get("value") or 0.0)
        if m is None or value <= 0:
            continue
        rows.append({"ref": p.get("ref") or "Planned shipment", "source": "Planned", "value": value,
                     "month": max(m, first_month), "note": p.get("note", ""), "warning": ""})
    return pd.DataFrame(rows, columns=cols).sort_values("month").reset_index(drop=True) if rows else pd.DataFrame(columns=cols)


# ── history -> automatic driver values ───────────────────────────────────────
def _row(final: pd.DataFrame, label: str, cols: List[str], nth: int = 0) -> np.ndarray:
    sel = final[final["Line"].str.rstrip() == label.rstrip()]
    if len(sel) <= nth:
        return np.zeros(len(cols))
    return pd.to_numeric(sel.iloc[nth][cols], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _ratio(num: np.ndarray, den: np.ndarray, n: int = AVG_N) -> float:
    d = float(den[-n:].sum())
    return float(num[-n:].sum()) / d if abs(d) > 1e-9 else 0.0


def shipment_cost_ratios(shipments: pd.DataFrame, n: int = AVG_N) -> Dict[str, float]:
    """pay order / charges / rebate transfer as a share of vendor value over the last `n` settled shipments."""
    out = {"pay_order_pct": 0.0, "charges_pct": 0.0, "rebate_pct": 0.0}
    if shipments is None or shipments.empty:
        return out
    s = shipments[shipments["Settlement matched"].astype(bool)].sort_values("GRN date").tail(n)
    v = float(pd.to_numeric(s["Vendor value"], errors="coerce").sum())
    if v <= 0:
        return out
    for key, col in (("pay_order_pct", "Pay order (capitalised)"), ("charges_pct", "Other import costs"),
                     ("rebate_pct", "VAT/AT rebate transfer")):
        out[key] = float(pd.to_numeric(s[col], errors="coerce").fillna(0).sum()) / v
    return out


def auto_drivers(final: pd.DataFrame, actual_cols: List[str], shipments: pd.DataFrame) -> Dict[str, float]:
    """Every driver measured from the last 3 actual months (or shipments)."""
    g = lambda label, nth=0: _row(final, label, actual_cols, nth)
    gross, net = g("   Gross sales (trade value)"), g("Net sales (after returns)")
    opening = g("Opening inventory (prior month closing)")
    ret, disc = -g("   − Customer returns (AR credits, SRJV/SRT-)"), -g("   − Discount")
    d = {
        "pct_sold": _ratio(net, opening), "return_rate": _ratio(ret, gross), "discount_rate": _ratio(disc, gross),
        "delivery_per_gross": _ratio(-g("   Deliveries (DO--)"), gross),
        "sr_per_return": _ratio(g("   Customer returns (SR--)"), ret),
        "sales_price_per_landed": _ratio(g("   Goods received & other receipts"),
                                         g("Landed cost of shipments received") + g("   Trading purchases received (items that are sold)")),
        "collections_pct": _ratio(g("   Customer receipts"), net),
    }
    d.update(shipment_cost_ratios(shipments))
    return d


def resolve_drivers(auto: Dict[str, float], settings: Dict) -> Dict[str, float]:
    out = {}
    for k, a in auto.items():
        o = (settings.get("drivers", {}) or {}).get(k, {})
        out[k] = float(o["value"]) if o.get("mode") == "manual" and o.get("value") is not None else float(a)
    return out


# ── projection ───────────────────────────────────────────────────────────────
def _idx(base: int, offset: int, T: int) -> Optional[int]:
    i = base + offset
    return i if 0 <= i < T else None


def import_schedule(schedule: pd.DataFrame, proj_months: List[Tuple[int, int]], imp: Dict,
                    drv: Dict[str, float], ltr_open: float) -> Dict[str, np.ndarray]:
    """Monthly import arrays (length T) from the shipment schedule. All amounts are positive."""
    T = len(proj_months)
    pos = {m: i for i, m in enumerate(proj_months)}
    z = lambda: np.zeros(T)
    a = {k: z() for k in ("vendor", "pay_order", "charges", "rebate", "landed", "margin", "customs",
                          "tt", "ltr_draw", "ltr_repay", "lc_exposure")}
    for r in schedule.itertuples(index=False) if schedule is not None and not schedule.empty else []:
        ia = pos.get(r.month)
        if ia is None:
            continue
        V = float(r.value)
        lc_value = imp["lc_pct"] * V
        a["vendor"][ia] += V
        a["pay_order"][ia] += drv["pay_order_pct"] * V
        a["charges"][ia] += drv["charges_pct"] * V
        a["rebate"][ia] += drv["rebate_pct"] * V
        a["landed"][ia] += V * (1 + drv["pay_order_pct"] + drv["charges_pct"])
        a["customs"][ia] += (drv["pay_order_pct"] + drv["rebate_pct"]) * V          # cash = duty + reclaimable VAT/AT
        for key, off, amt in (("margin", -int(imp["lc_lead"]), imp["margin_pct"] * lc_value),
                              ("ltr_draw", -int(imp["ltr_doc_lead"]), (1 - imp["margin_pct"]) * lc_value),
                              ("tt", -int(imp["ltr_doc_lead"]) + int(imp["tt_after_ltr"]), V - lc_value)):
            i = _idx(ia, off, T)
            if i is not None:
                a[key][i] += amt
        o_idx, d_idx = ia - int(imp["lc_lead"]), ia - int(imp["ltr_doc_lead"])
        for t in range(max(o_idx, 0), min(max(d_idx, o_idx), T - 1) + 1):    # open from the L/C opening until its LTR is drawn
            a["lc_exposure"][t] += lc_value
        if d_idx >= 0:                                    # an LTR drawn before the projection is in `ltr_open`
            i = _idx(d_idx, int(imp["ltr_months"]), T)
            if i is not None:
                a["ltr_repay"][i] += (1 - imp["margin_pct"]) * lc_value
    n = max(1, min(int(imp["ltr_months"]), T))        # LTR already outstanding is paid off within the limit
    if ltr_open > 0:
        a["ltr_repay"][:n] += ltr_open / n
    # TT for a shipment whose TT month falls before the projection is treated as already paid
    return a


def run_debts(proj_months: List[Tuple[int, int]], base_net: np.ndarray, margin_req: np.ndarray,
              customs_req: np.ndarray, opening_cash: float, balances: Dict[str, float],
              debts: Dict, floor: float, margin_funding: str, customs_funding: str) -> Dict[str, np.ndarray]:
    """Month-by-month cash / bank-debt engine for the HMBR group.

    base_net      cash flow of everything that is NOT debt service and NOT the L/C margin / customs
                  (collections, inflows, operating payments, TT, LTR repayments ...)
    margin_req    L/C margin due this month; customs_req customs (duty + VAT/AT) due this month
    Order each month:
      1. margin / customs go on the Dhaka CC up to its headroom (target ceiling, or the sanctioned
         limit when `use_limit`), the rest is paid in cash;
      2. deposits (e.g. the hospital repayment) go to the destination ENTERED for each one (UCB CC,
         Dhaka CC or cash) — nothing is spilled over automatically; no cash for the CC destinations;
      3. payouts come first: the obligations are paid in PRIORITY order from the cash on hand (instalment /
         lump: whatever is due plus arrears; sweep: only cash above the floor); unpaid amounts are arrears;
      4. then, if cash is below the floor, Dhaka CC is drawn up to the headroom to restore it;
      5. at quarter-end the accrued CC interest is added to the CC balances (no cash).
    Returns arrays (length T), amounts positive unless stated.
    """
    T = len(proj_months)
    items = debts["items"]
    prio = [k for k in debts.get("priority", []) if k in items]
    z = lambda: np.zeros(T)
    keys = list(items)
    out = {k: z() for k in ("cash", "draw", "cc_margin", "cc_customs", "cash_margin", "cash_customs", "dep_cash",
                           "dep_noncash", "dep_unapplied", "int_cc", "malek_interest", "shortfall")}
    out["pay"] = {k: z() for k in keys}                      # cash paid per instrument (incl. interest part for Malek)
    out["princ"] = {k: z() for k in keys}                    # principal reduction per instrument
    out["bal"] = {k: z() for k in keys}                      # month-end balance per instrument
    bal = {k: float(balances.get(k, 0.0) or 0.0) for k in keys}
    arrears = {k: 0.0 for k in keys}
    accrued = {"dhaka": 0.0, "ucb": 0.0}
    cash = float(opening_cash)
    deps_by_month: Dict[Tuple[int, int], List[Dict]] = {}
    for d in debts.get("deposits", []) or []:
        m = parse_month(d.get("month"))
        if m and float(d.get("amount") or 0) > 0:
            deps_by_month.setdefault(m, []).append(d)

    dh = items.get("dhaka", {})
    ceiling = float(dh.get("limit") if dh.get("use_limit") else dh.get("target") or 0.0)
    for t, ym in enumerate(proj_months):
        # 1. L/C margin and customs: onto the Dhaka CC up to its headroom
        headroom = max(ceiling - bal.get("dhaka", 0.0), 0.0)
        cc_m = min(margin_req[t], headroom) if margin_funding == "cc" else 0.0
        headroom -= cc_m
        cc_c = min(customs_req[t], headroom) if customs_funding == "cc" else 0.0
        out["cc_margin"][t], out["cc_customs"][t] = cc_m, cc_c
        out["cash_margin"][t], out["cash_customs"][t] = margin_req[t] - cc_m, customs_req[t] - cc_c
        bal["dhaka"] = bal.get("dhaka", 0.0) + cc_m + cc_c
        # 2. deposits go where they were entered (no automatic spill-over)
        for d in deps_by_month.get(ym, []):
            amt, dest = float(d["amount"]), d.get("dest", "ucb")
            if dest == "cash":
                out["dep_cash"][t] += amt
            else:
                take = min(amt, bal.get(dest, 0.0))
                bal[dest] = bal.get(dest, 0.0) - take
                out["dep_noncash"][t] += take
                out["dep_unapplied"][t] += amt - take
        pre = cash + base_net[t] - out["cash_margin"][t] - out["cash_customs"][t] + out["dep_cash"][t]
        # 3. payouts first, in priority order, from the cash on hand
        avail = max(pre, 0.0)
        for k in prio:
            it = items[k]
            mode = it.get("mode", "off")
            rate_m = float(it.get("rate") or 0.0) / 12.0
            if mode == "off" or bal[k] <= 0 and arrears[k] <= 0:
                continue
            if mode == "sweep":
                pay = min(bal[k], max(avail - floor, 0.0))      # sweeps only take cash above the floor
                out["pay"][k][t], out["princ"][k][t] = pay, pay
                bal[k] -= pay
                avail -= pay
                continue
            interest = bal[k] * rate_m if k not in ("dhaka", "ucb") else 0.0
            sched = float(it.get("monthly") or 0.0) if mode == "instalment" else (
                float(it.get("amount") or 0.0) if parse_month(it.get("due_month")) == ym else 0.0)
            due = min(sched + arrears[k], bal[k] + interest)
            pay = min(due, avail)
            int_paid = min(interest, pay)
            princ = pay - int_paid
            bal[k] += interest - int_paid - princ          # unpaid interest is added to the balance
            arrears[k] = max(due - pay, 0.0)
            out["pay"][k][t], out["princ"][k][t] = pay, princ
            if k == "malek":
                out["malek_interest"][t] = int_paid
            avail -= pay
        paid = sum(out["pay"][k][t] for k in keys)
        cash = pre - paid
        # 4. below the floor after the payouts -> draw the Dhaka CC up to its headroom
        if cash < floor:
            avail_cc = max(ceiling - bal.get("dhaka", 0.0), 0.0)
            draw = min(floor - cash, avail_cc)
            out["draw"][t] = draw
            bal["dhaka"] = bal.get("dhaka", 0.0) + draw
            cash += draw
        out["cash"][t] = cash
        out["shortfall"][t] = max(floor - cash, 0.0)
        # 5. quarterly CC interest is added to the balance at quarter-end (not paid in cash)
        for k in ("dhaka", "ucb"):
            accrued[k] += bal.get(k, 0.0) * float(items.get(k, {}).get("rate") or 0.0) / 12.0
            if ym[1] in (3, 6, 9, 12):
                bal[k] = bal.get(k, 0.0) + accrued[k]
                out["int_cc"][t] += accrued[k]
                accrued[k] = 0.0
        for k in keys:
            out["bal"][k][t] = bal[k]
    return out


def debt_opening_balances(debts: Dict, app_debts: Dict[str, float]) -> Dict[str, float]:
    """Auto balance from the app (absolute value of the account balance) unless overridden in the settings."""
    out = {}
    for k, it in debts["items"].items():
        o = it.get("balance")
        out[k] = float(o) if o not in (None, "") else abs(float((app_debts or {}).get(DEBT_ACCOUNTS.get(k), 0.0) or 0.0))
    return out


def project(final: pd.DataFrame, actual_cols: List[str], last_actual: Tuple[int, int],
            app_last_closing: float, schedule: pd.DataFrame, settings: Dict,
            ltr_open: float = 0.0, shipments_hist: Optional[pd.DataFrame] = None,
            app_debts: Optional[Dict[str, float]] = None, mtd: Optional[Dict] = None
            ) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Final Model rows with `horizon` projected columns appended (labelled 'Mon-YY (P)').

    Returns (table, projected_column_names, info) where info carries the resolved drivers,
    the lowest projected closing cash and the memo arrays.

    Transition month: when `mtd` = {"col": <column of `final` holding the current month to date>,
    "elapsed": days passed, "days": days in the month} is given, the first projected month is that
    month and is REAL + PROJECTION:
      * a line already entered shows what was entered; a line expected to reach its average
        shows the average when that is larger (outflows: the more negative of the two);
      * sales are projected linearly: month-to-date x days in month / days elapsed; deliveries,
        returns back into stock and collections then follow the usual ratios (with no sales entered
        yet this month the normal %-of-inventory-sold projection is used instead);
      * goods already received are real, open IPs expected this month are added on top.
    """
    T = int(settings.get("horizon_months", HORIZON_MONTHS))
    proj_months = next_months(last_actual, T)
    pcols = [cm.month_label(m) + " (P)" for m in proj_months]
    if mtd:
        pcols[0] = cm.month_label(proj_months[0]) + " (MTD+P)"
    f_lin = (float(mtd["days"]) / max(float(mtd["elapsed"]), 1.0)) if mtd else 1.0
    act = (lambda label, nth=0: float(_row(final, label, [mtd["col"]], nth)[0])) if mtd else (lambda label, nth=0: 0.0)
    imp = settings["imports"]
    auto = auto_drivers(final, actual_cols, shipments_hist)
    drv = resolve_drivers(auto, settings)
    A = import_schedule(schedule, proj_months, imp, drv, ltr_open)

    g = lambda label, nth=0: _row(final, label, actual_cols, nth)
    avg3 = lambda label, nth=0: float(g(label, nth)[-AVG_N:].mean()) if len(actual_cols) else 0.0

    P: Dict[Tuple[str, int], np.ndarray] = {}            # (label, nth) -> projected vector
    def put(label, vec, nth=0):
        P[(label, nth)] = np.asarray(vec, dtype=float)

    zeros = lambda: np.zeros(T)
    # purchases ---------------------------------------------------------------
    local = np.full(T, avg3("   Trading purchases received (items that are sold)"))
    local_act = act("   Trading purchases received (items that are sold)")
    if mtd:
        local[0] = max(local_act, local[0])
    def with_mtd(vec, label):                  # goods already received this month (real) + open IPs still to arrive
        v = np.array(vec, dtype=float)
        if mtd:
            v[0] += act(label)
        return v
    put("   Vendor value of shipments received", with_mtd(A["vendor"], "   Vendor value of shipments received"))
    put("   + Pay order — capitalised customs duty", with_mtd(A["pay_order"], "   + Pay order — capitalised customs duty"))
    put("   + L/C charges & other import costs (insurance, C&F, freight, carrying)",
        with_mtd(A["charges"], "   + L/C charges & other import costs (insurance, C&F, freight, carrying)"))
    put("   VAT/AT on pay order transferred to rebate claim (reclaimable, not cost)",
        with_mtd(A["rebate"], "   VAT/AT on pay order transferred to rebate claim (reclaimable, not cost)"))
    put("Landed cost of shipments received", with_mtd(A["landed"], "Landed cost of shipments received"))
    put("   Trading purchases received (items that are sold)", local)

    # inventory & sales (month by month because opening depends on the previous closing) ----
    opening, closing = zeros(), zeros()
    gross, disc, rets, net = zeros(), zeros(), zeros(), zeros()
    delivered, sr_in, recv = zeros(), zeros(), zeros()
    manuf = np.full(T, avg3("   Manufactured (RE--)"))
    other_iss = np.full(T, avg3("   Other issues, adjustments & returns to suppliers"))
    prev = float(g("Closing inventory")[-1]) if len(actual_cols) else 0.0
    if mtd:
        manuf[0] = max(act("   Manufactured (RE--)"), manuf[0])
        other_iss[0] = min(act("   Other issues, adjustments & returns to suppliers"), other_iss[0])
    for t in range(T):
        opening[t] = prev
        if mtd and t == 0 and act("   Gross sales (trade value)") > 0:   # linear: month-to-date x days in month / days elapsed
            gross[t] = act("   Gross sales (trade value)") * f_lin
            disc[t] = -act("   − Discount") * f_lin
            rets[t] = -act("   − Customer returns (AR credits, SRJV/SRT-)") * f_lin
            net[t] = gross[t] - disc[t] - rets[t]
        else:
            net[t] = drv["pct_sold"] * prev
            denom = 1.0 - drv["return_rate"] - drv["discount_rate"]
            gross[t] = net[t] / denom if denom > 0.05 else net[t]
            disc[t], rets[t] = drv["discount_rate"] * gross[t], drv["return_rate"] * gross[t]
        delivered[t] = -drv["delivery_per_gross"] * gross[t]
        sr_in[t] = drv["sr_per_return"] * rets[t]
        if mtd and t == 0:                                    # goods already received (real) + open IPs expected this month
            recv[t] = (act("   Goods received & other receipts") + drv["sales_price_per_landed"] * A["landed"][t]
                       + drv["sales_price_per_landed"] * max(local[t] - local_act, 0.0))
        else:
            recv[t] = drv["sales_price_per_landed"] * (A["landed"][t] + local[t])
        closing[t] = prev + delivered[t] + sr_in[t] + manuf[t] + recv[t] + other_iss[t]
        prev = closing[t]
    put("Opening inventory (prior month closing)", opening)
    put("   Revaluation at this month's average prices", zeros())
    put("   Deliveries (DO--)", delivered)
    put("   Customer returns (SR--)", sr_in)
    put("   Manufactured (RE--)", manuf)
    put("   Goods received & other receipts", recv)
    put("   Other issues, adjustments & returns to suppliers", other_iss)
    put("   Internal transfers between the combined businesses (net)", zeros())
    put("Closing inventory", closing)
    put("   Gross sales (trade value)", gross)
    put("   − Discount", -disc)
    put("   − Customer returns (AR credits, SRJV/SRT-)", -rets)
    put("Net sales (after returns)", net)
    with np.errstate(divide="ignore", invalid="ignore"):
        put("% of opening inventory sold", np.where(opening > 0, net / opening, np.nan))

    # cash ----------------------------------------------------------------------
    line_cfg = settings.get("lines", {}) or {}
    known = {}                                            # (line label) -> vector of known amounts
    for k in settings.get("known_items", []) or []:
        m = parse_month(k.get("month"))
        if m not in proj_months:
            continue
        sign = 1.0 if k.get("direction") == "in" else -1.0
        lab = k.get("line") or ("   Other inflows" if sign > 0 else "   Other outflows")
        known.setdefault(lab, zeros())[proj_months.index(m)] += sign * abs(float(k.get("amount") or 0))

    def trans(vec, label, direction, nth=0):
        """Transition month: what was already entered wins when it exceeds the projection (outflows: the more negative)."""
        if mtd:
            a = act(label, nth)
            vec[0] = max(a, vec[0]) if direction == "in" else min(a, vec[0])
        return vec

    def line_vec(label, direction, nth=0, use_trans=True):
        cfg = {**default_line_setting(label, direction), **line_cfg.get(label, {})}
        if cfg["mode"] == "zero":
            v = zeros()
        elif cfg["mode"] == "fixed":
            v = np.full(T, float(cfg.get("fixed") or 0.0))
        else:
            v = np.full(T, avg3(label, nth))
        v = v * (1.0 + float(cfg.get("adjust_pct") or 0.0) / 100.0)
        if use_trans:
            v = trans(v, label, direction, nth)
        return v + known.get(label, zeros())

    cust = trans(drv["collections_pct"] * net, "   Customer receipts", "in")
    coll = {"   Customer receipts": cust,
            "   Recognized agents & other receivables": line_vec("   Recognized agents & other receivables", "in")}
    for lab, v in coll.items():
        put(lab, v + known.get(lab, zeros()) if lab == "   Customer receipts" else v)
    put("Total collections", sum(coll.values()) + known.get("   Customer receipts", zeros()))

    add_labels = [r[2] for r in cm.FINAL_MODEL_ROWS if r[0] == "cash" and r[1] == "add" and r[2].startswith("   ")]
    add_vecs = {}
    for lab in add_labels:
        nth = 0
        add_vecs[lab] = line_vec(lab, "in", nth)
        put(lab, add_vecs[lab], nth)
    tot_add = sum(add_vecs.values()) if add_vecs else zeros()
    put("Total additional inflows", tot_add)
    total_in = P[("Total collections", 0)] + tot_add
    put("TOTAL CASH IN", total_in)

    out_labels = [r[2] for r in cm.FINAL_MODEL_ROWS if r[0] == "cash" and r[1] == "out" and r[2].startswith("   ")]
    debts = settings.get("debts", {}) or {}
    use_debts = bool(debts.get("enabled"))
    floor = float(settings.get("cash_floor") or 0.0)
    cc_draw = zeros()
    dbg = None
    # rows the debt engine owns when enabled
    engine_row_labels = {"   Bank overdraft (CC) repaid", "   Other-party loans (0904: Malek Jobeda, Water Lily …)",
                         "   Loans repaid (directors, group, others)", "   Interest & bank charges",
                         "   International: L/C margin & charges", "   International: pay order (customs tax & VAT)"}
    driven = {
        "   International: L/C margin & charges": -A["margin"] if imp["margin_funding"] == "cash" else zeros(),
        "   International: TT / advances to vendors": -A["tt"],
        "   International: pay order (customs tax & VAT)": -A["customs"] if imp["customs_funding"] == "cash" else zeros(),
        "   Bank loans & LTR repaid": -A["ltr_repay"],
    }
    out_vecs = {}
    for lab in out_labels:
        nth = 1 if lab == "   GI cash passing through HMBR (09060001)" else 0
        if use_debts and lab in engine_row_labels:
            continue
        v = (trans(np.array(driven[lab], dtype=float), lab, "out") + known.get(lab, zeros())) if lab in driven \
            else line_vec(lab, "out", nth)
        out_vecs[lab] = v
        put(lab, v, nth)

    if use_debts:
        bal0 = debt_opening_balances(debts, app_debts or {})
        in_nodebt = P[("Total collections", 0)] + tot_add                         # already without the CC draw
        base_net = in_nodebt + sum(out_vecs.values())
        bank_chg = line_vec("   Interest & bank charges", "out", use_trans=False)   # bank charges (3-month average)
        ltr_out_open = np.concatenate([[ltr_open], np.maximum(ltr_open + np.cumsum(A["ltr_draw"]) - np.cumsum(A["ltr_repay"]), 0.0)[:-1]])
        accr = ltr_out_open * float(imp.get("ltr_rate") or 0.0) / 12.0
        ltr_int = np.zeros(T)
        run = 0.0
        for t, ym in enumerate(proj_months):                                            # accrues monthly, paid in cash at quarter-end
            run += accr[t]
            if ym[1] in (3, 6, 9, 12):
                ltr_int[t], run = run, 0.0
        bank_chg = bank_chg - ltr_int

        def engine(extra0: float):
            bn = (base_net + bank_chg).copy()
            bn[0] += extra0
            return run_debts(proj_months, bn, A["margin"], A["customs"], app_last_closing, bal0,
                             debts, floor, imp["margin_funding"], imp["customs_funding"])

        def engine_rows(d):
            pay_ = d["pay"]
            grp = lambda row: sum(pay_[k] for k in pay_ if DEBT_CASH_ROW.get(k) == row)
            return {
                "   Bank overdraft (CC) repaid": -grp("   Bank overdraft (CC) repaid"),
                "   Other-party loans (0904: Malek Jobeda, Water Lily …)": -(grp("   Other-party loans (0904: Malek Jobeda, Water Lily …)") - d["malek_interest"]),
                "   Loans repaid (directors, group, others)": -grp("   Loans repaid (directors, group, others)"),
                "   Interest & bank charges": bank_chg - d["malek_interest"],
                "   International: L/C margin & charges": -d["cash_margin"],
                "   International: pay order (customs tax & VAT)": -d["cash_customs"],
            }, {"   Bank overdraft (CC) drawn": d["draw"], "   Hospital loan repayments (capital inflow)": d["dep_cash"]}

        dbg = engine(0.0)
        excess_out, excess_in = {}, {}
        if mtd:
            # What was already paid / received this month (real) on the engine-owned rows, over and above what
            # the engine would pay, is added as a real flow and the engine is re-run so cash and the later
            # months see it (row = engine value + the excess = the larger of the two).
            rows_out, rows_in = engine_rows(dbg)
            excess_out = {lab: min(act(lab) - float(v[0]), 0.0) for lab, v in rows_out.items()}
            excess_in = {lab: max(act(lab) - float(add_vecs[lab][0] + v[0]), 0.0) for lab, v in rows_in.items()}
            dbg = engine(sum(excess_out.values()) + sum(excess_in.values()))
        rows_out, rows_in = engine_rows(dbg)
        cc_draw = dbg["draw"]
        for lab, v in rows_out.items():
            v = v + known.get(lab, zeros())
            v[0] += excess_out.get(lab, 0.0)
            out_vecs[lab] = v
            put(lab, v)
        # the CC draw and any deposit left over after the loans are paid are cash inflows
        for lab, v in rows_in.items():
            add_vecs[lab] = add_vecs[lab] + v
            add_vecs[lab][0] += excess_in.get(lab, 0.0)
            put(lab, add_vecs[lab])
        tot_add = sum(add_vecs.values())
        put("Total additional inflows", tot_add)
        total_in = P[("Total collections", 0)] + tot_add
        put("TOTAL CASH IN", total_in)
    total_out = sum(out_vecs.values())
    put("TOTAL CASH OUT", total_out)

    net_change = total_in + total_out
    open_cash, close_cash = zeros(), zeros()
    prev_c = float(app_last_closing)
    for t in range(T):
        open_cash[t] = prev_c
        close_cash[t] = prev_c + net_change[t]
        prev_c = close_cash[t]
    put("Opening cash & CE (app)", open_cash)
    put("NET CHANGE IN CASH", net_change)
    put("Closing cash — model (opening + net change)", close_cash)

    # memo (projection only): nothing below moves cash
    ltr_out = np.maximum(ltr_open + np.cumsum(A["ltr_draw"]) - np.cumsum(A["ltr_repay"]), 0.0)
    if use_debts:
        lim = float(debts.get("lc_limit") or 0.0)
        cc_margin, cc_customs = -dbg["cc_margin"], -dbg["cc_customs"]
    else:
        lim = 0.0
        cc_margin = -A["margin"] if imp["margin_funding"] == "cc" else zeros()
        cc_customs = -A["customs"] if imp["customs_funding"] == "cc" else zeros()
    memo = {
        "MEMO — projected, no cash moved": None,
        "   L/C margin paid from bank overdraft (CC)": cc_margin,
        "   Customs (pay order incl. VAT/AT) paid from bank overdraft (CC)": cc_customs,
        "   LTR drawn — bank pays the supplier": A["ltr_draw"],
        "   LTR outstanding at month-end": ltr_out,
        "   L/C value open (limit applies until its LTR is drawn)": A["lc_exposure"],
    }
    if use_debts:
        memo.update({
            "   Interest charged to the CC accounts (quarter-end, added to the balance)": -dbg["int_cc"],
            "   Deposits straight into the CC accounts (e.g. hospital repayment)": dbg["dep_noncash"],
            "DEBT BALANCES (month-end)": None,
        })
        for k in debts.get("priority", []) + [k for k in debts["items"] if k not in debts.get("priority", [])]:
            if k in dbg["bal"]:
                memo["   " + DEBT_NAMES.get(k, k)] = dbg["bal"][k]
        memo["   Cash short of the floor (not covered by the CC)"] = dbg["shortfall"]

    # assemble the table: every Final Model row, actual columns untouched, projected appended
    seen: Dict[str, int] = {}
    rows = []
    for _, r in final.iterrows():
        label = str(r["Line"])
        key_label = label.rstrip() if label else ""
        nth = seen.get(key_label, 0)
        seen[key_label] = nth + 1
        vec = P.get((label, nth)) if label else None
        if vec is None and label:
            vec = P.get((label.rstrip(), nth))
        base = {c: r[c] for c in ["Line", "kind"] + list(actual_cols)}
        for t, pc in enumerate(pcols):
            base[pc] = (np.nan if vec is None or r["kind"] in ("header", "blank", "check") else round(float(vec[t]), 6) + 0.0)  # +0.0: never "-0"
        rows.append(base)
    table = pd.DataFrame(rows, columns=["Line", "kind"] + list(actual_cols) + pcols)
    for label, vec in memo.items():
        row = {"Line": label, "kind": "header" if vec is None else "memo", **{c: np.nan for c in actual_cols}}
        for t, pc in enumerate(pcols):
            row[pc] = np.nan if vec is None else round(float(vec[t]), 6) + 0.0
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)

    info = {"auto": auto, "drivers": drv, "proj_months": proj_months, "closing_cash": close_cash,
            "min_cash": float(close_cash.min()) if T else 0.0,
            "min_cash_month": cm.month_label(proj_months[int(close_cash.argmin())]) if T else None,
            "floor_breaches": [cm.month_label(m) for m, c in zip(proj_months, close_cash)
                               if c < float(settings.get("cash_floor") or 0.0)],
            "imports": A, "debts": dbg, "transition": bool(mtd),
            "lc_limit_breaches": ([cm.month_label(m) for m, e in zip(proj_months, A["lc_exposure"])
                                   if e > float((settings.get("debts") or {}).get("lc_limit") or 0) > 0]),
            "shortfall_months": ([cm.month_label(m) for m, x in zip(proj_months, dbg["shortfall"]) if x > 1.0] if dbg else [])}
    return table, pcols, info


# ── worked examples for the Settings tab (what a change would do) ────────────
# Every example is computed with the same engine from the CURRENT settings, so it can never drift from
# what the projection really does. Pure pandas; the view only formats and displays.
def _fmt0(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.0f}"


def _one_shipment(imp: Dict, drv: Dict[str, float], value: float, arrival_idx: int, T: int, base=(2026, 9)):
    months = next_months(base, T)
    sched = pd.DataFrame([{"ref": "example", "source": "Planned", "value": value, "month": months[arrival_idx],
                           "note": "", "warning": ""}])
    return months, import_schedule(sched, months, imp, drv, 0.0)


def example_import(imp: Dict, drv: Dict[str, float], value: float = 10_000_000.0) -> pd.DataFrame:
    """Cash steps of one example shipment now and with the L/C value 10 points higher."""
    T, ia = 12, 5
    rows = []
    variants = [("Now", imp), ("If L/C % +10 pts", {**imp, "lc_pct": min(imp["lc_pct"] + 0.10, 1.0)})]
    res = {name: _one_shipment(v, drv, value, ia, T)[1] for name, v in variants}
    steps = [
        ("L/C margin paid", f"{int(imp['lc_lead'])} month(s) before arrival", "margin", ia - int(imp["lc_lead"])),
        ("LTR drawn — bank pays the supplier (no cash)", f"{int(imp['ltr_doc_lead'])} month(s) before arrival", "ltr_draw", ia - int(imp["ltr_doc_lead"])),
        ("TT paid to the supplier", f"{int(imp['ltr_doc_lead']) * -1 + int(imp['tt_after_ltr']):+d} month(s) vs arrival", "tt",
         ia - int(imp["ltr_doc_lead"]) + int(imp["tt_after_ltr"])),
        ("Customs: duty + VAT/AT", "arrival month", "customs", ia),
        ("LTR repaid", f"{int(imp['ltr_months'])} month(s) after the LTR", "ltr_repay", ia - int(imp["ltr_doc_lead"]) + int(imp["ltr_months"])),
    ]
    for name, when, key, idx in steps:
        rows.append({"Step": name, "When": when, **{n: _fmt0(res[n][key][idx]) for n, _ in variants}})
    return pd.DataFrame(rows)


def example_shift(imp: Dict, drv: Dict[str, float], first: Tuple[int, int], value: float = 10_000_000.0) -> pd.DataFrame:
    """The same shipment arriving in month X and one month later: where its cash falls."""
    T = 10
    out = {}
    labels = []
    for k, ia in (("a", 3), ("b", 4)):
        months = next_months((first[0], first[1] - 1) if first[1] > 1 else (first[0] - 1, 12), T)
        sched = pd.DataFrame([{"ref": "e", "source": "Planned", "value": value, "month": months[ia], "note": "", "warning": ""}])
        A = import_schedule(sched, months, imp, drv, 0.0)
        labels.append(cm.month_label(months[ia]))
        out[k] = {"L/C margin": A["margin"], "TT": A["tt"], "Customs (duty + VAT/AT)": A["customs"], "LTR repaid": A["ltr_repay"]}
    rows = []
    for item in out["a"]:
        def when(arr):
            idx = [i for i, v in enumerate(arr) if v > 0]
            return ", ".join(cm.month_label(next_months((first[0], first[1] - 1) if first[1] > 1 else (first[0] - 1, 12), T)[i]) for i in idx) or "—"
        rows.append({"Cash item": item, f"Arrives {labels[0]}": when(out["a"][item]), f"Arrives {labels[1]} (one month later)": when(out["b"][item])})
    return pd.DataFrame(rows)


def example_drivers(final: pd.DataFrame, done_cols: List[str], drv: Dict[str, float], value: float = 10_000_000.0) -> pd.DataFrame:
    """What a 10% higher driver does, on the last closing inventory and on a 10M shipment."""
    opening = float(_row(final, "Closing inventory", done_cols)[-1]) if len(done_cols) else 0.0
    up = 1.10
    net_now, net_up = drv["pct_sold"] * opening, drv["pct_sold"] * up * opening
    rows = [
        ("% of opening inventory sold", f"{drv['pct_sold']:.1%} → {drv['pct_sold'] * up:.1%}", "Net sales next month (on the last closing inventory)", net_now, net_up),
        ("Customer receipts / net sales", f"{drv['collections_pct']:.1%} → {drv['collections_pct'] * up:.1%}", "Customer receipts next month", drv["collections_pct"] * net_now, drv["collections_pct"] * up * net_now),
        ("Pay order / vendor value", f"{drv['pay_order_pct']:.1%} → {drv['pay_order_pct'] * up:.1%}", f"Customs cash on a {value:,.0f} shipment (duty + VAT/AT)",
         (drv["pay_order_pct"] + drv["rebate_pct"]) * value, (drv["pay_order_pct"] * up + drv["rebate_pct"]) * value),
        ("Returns / gross sales", f"{drv['return_rate']:.1%} → {drv['return_rate'] * up:.1%}", "Returns on 15,000,000 of gross sales", drv["return_rate"] * 15e6, drv["return_rate"] * up * 15e6),
    ]
    return pd.DataFrame([{"Driver": d, "10% higher": ch, "Effect on": e, "Now": _fmt0(a), "After": _fmt0(b), "Change": _fmt0(b - a)}
                         for d, ch, e, a, b in rows])


def example_lines(final: pd.DataFrame, done_cols: List[str], label: str = "   Operating expenses") -> pd.DataFrame:
    """The same line under each mode, using its real 3-month average."""
    avg = float(_row(final, label, done_cols)[-AVG_N:].mean()) if len(done_cols) else 0.0
    return pd.DataFrame([
        {"Setting": "avg3, adjust 0%", "Monthly amount": _fmt0(avg)},
        {"Setting": "avg3, adjust −10% (pay a little less)", "Monthly amount": _fmt0(avg * 0.9)},
        {"Setting": "zero", "Monthly amount": _fmt0(0.0)},
        {"Setting": "fixed 2,000,000", "Monthly amount": _fmt0(-2_000_000.0 if avg < 0 else 2_000_000.0)},
    ]).assign(Line=label.strip())[["Line", "Setting", "Monthly amount"]]


def example_known(first: Tuple[int, int]) -> pd.DataFrame:
    m = cm.month_label(next_months(first, 2)[1])
    return pd.DataFrame([
        {"Entry": f"Hospital repayment, in, 1,000,000, {m}, into 'Other inflows'", "Line affected": "Other inflows",
         f"{m} before": "0", f"{m} after": "1,000,000", "Closing cash from then on": "+1,000,000 (before any debt payments it releases)"},
        {"Entry": f"Eid bonus, out, 2,000,000, {m}, into 'Salaries, bonus & staff advances'", "Line affected": "Salaries, bonus & staff advances",
         f"{m} before": "average", f"{m} after": "average − 2,000,000", "Closing cash from then on": "−2,000,000"},
    ])


def _demo_debts(debts: Dict, app_debts: Dict[str, float]) -> Tuple[Dict, Dict[str, float]]:
    d = copy.deepcopy(debts)
    d["deposits"] = []
    return d, debt_opening_balances(d, app_debts or {})


def example_priority(debts: Dict, app_debts: Dict[str, float], cash: float = 1_200_000.0) -> pd.DataFrame:
    """Who is paid when only `cash` is available: the current order vs the first two swapped."""
    d, bal = _demo_debts(debts, app_debts)
    order = [k for k in d.get("priority", []) if k in d["items"]]
    swapped = order[:]
    if len(swapped) >= 2:
        swapped[0], swapped[1] = swapped[1], swapped[0]
    months = next_months((2026, 9), 1)
    res = {}
    for name, pr in (("Current order", order), ("First two swapped", swapped)):
        dd = copy.deepcopy(d)
        dd["priority"] = pr
        # a month in which no instalment is "due" must not hide the rule: use due_month of any lump in this month
        r = run_debts(months, np.array([cash]), np.zeros(1), np.zeros(1), 0.0, bal, dd, 0.0, "cc", "cc")
        res[name] = {k: r["pay"][k][0] for k in dd["items"]}
    rows = [{"Obligation": DEBT_NAMES.get(k, k), "Current order": _fmt0(res["Current order"][k]),
             "First two swapped": _fmt0(res["First two swapped"][k])}
            for k in order if res["Current order"][k] > 0 or res["First two swapped"][k] > 0]
    pos = {k: i + 1 for i, k in enumerate(order)}
    out = pd.DataFrame(rows)
    if not out.empty:
        out.insert(0, "Priority", [pos[next(k for k in order if DEBT_NAMES.get(k, k) == n)] for n in out["Obligation"]])
    return out


def example_headroom(debts: Dict, app_debts: Dict[str, float], request: float = 2_000_000.0) -> pd.DataFrame:
    """A 2,000,000 L/C margin: how much goes on the Dhaka CC vs cash for the current target and a higher one."""
    d, bal = _demo_debts(debts, app_debts)
    months = next_months((2026, 9), 1)
    rows = []
    for name, extra in (("Current target", 0.0), ("Target + 2,000,000", 2_000_000.0)):
        dd = copy.deepcopy(d)
        dd["items"]["dhaka"]["target"] = float(dd["items"]["dhaka"].get("target") or 0.0) + extra
        dd["items"]["dhaka"]["use_limit"] = False
        r = run_debts(months, np.zeros(1), np.array([request]), np.zeros(1), 0.0, bal, dd, 0.0, "cc", "cc")
        rows.append({"Dhaka CC target": f"{name} ({dd['items']['dhaka']['target']:,.0f})", "Balance now": _fmt0(bal.get("dhaka")),
                     "Margin on the CC": _fmt0(r["cc_margin"][0]), "Margin paid in cash": _fmt0(r["cash_margin"][0])})
    return pd.DataFrame(rows)


def example_floor(debts: Dict, app_debts: Dict[str, float], floor: float, cash_before: float = 400_000.0) -> pd.DataFrame:
    """Cash would end the month at `cash_before`: what the CC is drawn for at different floors."""
    d, bal = _demo_debts(debts, app_debts)
    for it in d["items"].values():          # isolate the floor rule from payouts
        it["mode"] = "off"
    months = next_months((2026, 9), 1)
    rows = []
    for f in sorted({0.0, float(floor), float(floor) * 2 or 1_000_000.0}):
        r = run_debts(months, np.array([cash_before]), np.zeros(1), np.zeros(1), 0.0, bal, d, f, "cc", "cc")
        rows.append({"Cash floor": _fmt0(f), "Cash before topping up": _fmt0(cash_before), "Dhaka CC drawn": _fmt0(r["draw"][0]),
                     "Closing cash": _fmt0(r["cash"][0]), "Short of the floor": _fmt0(r["shortfall"][0])})
    return pd.DataFrame(rows)
