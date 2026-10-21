"""Cash Model — pure pandas transforms for Financial Statements → "💵 Cash Model".

Window = Jan (year-1) .. end_month of `year`, the same span as the Monthly
statements. The app's own "Cash & CE" (financial.cash_open_close) is the source
of truth for opening/closing cash; everything here explains the movement
between them from the GL, and ties back to it.

No st.* calls (processing-layer rule).
"""
from __future__ import annotations

import io
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CC_CODES = ["10010002", "10010003", "10010006", "10010007"]   # bank overdraft / CC accounts
PRICE_WINDOW_MONTHS = 3        # trailing months used for the average sales price
PRICE_FALLBACK_MONTHS = 12     # used when an item has no sales in the short window


# ── window / labels ──────────────────────────────────────────────────────────
def window_months(year: int, end_month: int) -> List[Tuple[int, int]]:
    """(year-1, 1..12) followed by (year, 1..end_month)."""
    return [(year - 1, m) for m in range(1, 13)] + [(year, m) for m in range(1, end_month + 1)]


def month_label(ym: Tuple[int, int]) -> str:
    return f"{MONTH_NAMES[ym[1] - 1]}-{str(ym[0])[2:]}"


# ── account / sub-ledger classification ──────────────────────────────────────
def sub_kind(sub_name: str) -> str:
    """Classify a 01050001 (Advance For International Purchase) sub-ledger by its NAME.

    Sub-ledger numbering differs per entity (HMBR 0105000104 = L/C charge, GI
    0105000104 = pay order), so the name — not the code — decides.
    kinds: payorder | lc (margin: payment toward the vendor value) |
           lcchg (L/C charge & commission: a landed cost) | tt | other
    """
    n = (sub_name or "").lower()
    if "pay order" in n or "payorder" in n:
        return "payorder"
    if "margin" in n:
        return "lc"
    if "l/c" in n or "commission" in n:
        return "lcchg"
    if re.search(r"\bt\s?t\b", n) or "advance" in n or "payment by" in n or "deposit for international" in n:
        return "tt"
    return "other"


_SUB_LABEL = {
    "lc": "International: L/C margin & charges",
    "lcchg": "International: L/C margin & charges",
    "tt": "International: TT / advances to vendors",
    "payorder": "International: pay order (customs tax & VAT)",
    "other": "International: other import costs (insurance, C&F, freight, carrying)",
}


def categorize_flow(code: str, sub: Optional[str], direction: str,
                    sub_names: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """Map a counter-account to (group, label). direction is 'in' or 'out'."""
    code = str(code)
    if direction == "in":
        if code == "01030001":
            return "Collections", "Customer receipts"
        if code.startswith("0103"):
            return "Collections", "Recognized agents & other receivables"
        if code in ("08030002", "02030002"):
            return "Additional inflows", "🔁 Rent & property receipts (recurring)"
        if code == "11040001":
            return "Additional inflows", "Security deposit receipts"
        if code == "02050001":
            return "Additional inflows", "Hospital loan repayments (capital inflow)"
        if code.startswith("02050"):
            return "Additional inflows", "Group, director & other loan repayments received"
        if code == "09060001":
            return "Additional inflows", "GI cash passing through HMBR (09060001)"
        if code.startswith("1001"):
            return "Additional inflows", "Bank overdraft (CC) drawn"
        if code.startswith("1002") or code.startswith("1201"):
            return "Additional inflows", "Loans received (directors, group, others)"
        if code.startswith("0904"):
            return "Additional inflows", "Other-party funds received (0904)"
        if code.startswith("0105"):
            return "Additional inflows", "Import advances refunded / VAT rebate received"
        if code.startswith("08"):
            return "Additional inflows", "Other income & interest"
        return "Additional inflows", "Other inflows"
    # outflows
    if code == "01050001":
        name = (sub_names or {}).get(str(sub), "")
        return "Payments", _SUB_LABEL[sub_kind(name)]
    if code == "01050007":
        return "Payments", "VAT / AT rebate claims"
    if code.startswith("0903"):
        return "Payments", "Suppliers (local, internal & international payables)"
    if code == "09010002" or code.startswith(("0104", "0612", "0613", "0614")) or code == "11010003":
        return "Payments", "Salaries, bonus & staff advances"
    if code == "09010006" or code.startswith(("0615", "11020")):
        return "Payments", "Director remuneration & funds"
    if code.startswith("0904"):
        return "Payments", "Other-party loans (0904: Malek Jobeda, Water Lily …)"
    if code in CC_CODES:
        return "Payments", "Bank overdraft (CC) repaid"
    if code.startswith(("1001", "1201")):
        return "Payments", "Bank loans & LTR repaid"
    if code.startswith("1002"):
        return "Payments", "Loans repaid (directors, group, others)"
    if code.startswith("02050"):
        return "Payments", "Loans given (group, directors, others)"
    if code.startswith("0629"):
        return "Payments", "Income tax & VAT"
    if code.startswith(("0630", "0633", "0635")):
        return "Payments", "Interest & bank charges"
    if code.startswith("03"):
        return "Payments", "Capital expenditure (assets)"
    if code == "09060001":
        return "Payments", "GI cash passing through HMBR (09060001)"
    if code.startswith("0905"):
        return "Payments", "C&F and clearing charges"
    if code.startswith("0901"):
        return "Payments", "Accrued expenses paid"
    if code.startswith(("05", "06", "07")):
        return "Payments", "Operating expenses"
    return "Payments", "Other outflows"


# ── cash statement ───────────────────────────────────────────────────────────
def _flow_long(flows: pd.DataFrame, sub_names: Dict[str, str]) -> pd.DataFrame:
    """Flows → one row per (year, month, ac_code, xsub, direction) with a signed amount."""
    if flows is None or flows.empty:
        return pd.DataFrame(columns=["year", "month", "ac_code", "xsub", "group", "label", "amount"])
    f = flows.copy()
    for c in ("inflow", "outflow"):
        f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)
    parts = []
    for direction, col in (("in", "inflow"), ("out", "outflow")):
        d = f[f[col] != 0][["year", "month", "ac_code", "xsub", col]].rename(columns={col: "amount"})
        d["direction"] = direction
        parts.append(d)
    long = pd.concat(parts, ignore_index=True)
    if long.empty:
        return pd.DataFrame(columns=["year", "month", "ac_code", "xsub", "group", "label", "amount"])
    cats = [categorize_flow(c, s, dr, sub_names)
            for c, s, dr in zip(long["ac_code"], long["xsub"], long["direction"])]
    long["group"] = [c[0] for c in cats]
    long["label"] = [c[1] for c in cats]
    return long


def _order_labels(labels: List[str], group: str) -> List[str]:
    """Stable, meaningful ordering: biggest absolute total first would shuffle by data,
    so use rule order = first appearance order in the canonical lists below."""
    canon_in = ["Customer receipts", "Recognized agents & other receivables",
                "🔁 Rent & property receipts (recurring)", "Security deposit receipts",
                "Hospital loan repayments (capital inflow)",
                "Group, director & other loan repayments received",
                "GI cash passing through HMBR (09060001)", "Bank overdraft (CC) drawn",
                "Loans received (directors, group, others)", "Other-party funds received (0904)",
                "Import advances refunded / VAT rebate received", "Other income & interest", "Other inflows"]
    canon_out = [_SUB_LABEL["lc"], _SUB_LABEL["tt"], _SUB_LABEL["payorder"], _SUB_LABEL["other"],
                 "VAT / AT rebate claims", "Suppliers (local, internal & international payables)",
                 "Salaries, bonus & staff advances", "Director remuneration & funds",
                 "Other-party loans (0904: Malek Jobeda, Water Lily …)", "Bank overdraft (CC) repaid",
                 "Bank loans & LTR repaid", "Loans repaid (directors, group, others)",
                 "Loans given (group, directors, others)", "Income tax & VAT", "Interest & bank charges",
                 "Capital expenditure (assets)", "GI cash passing through HMBR (09060001)",
                 "C&F and clearing charges", "Accrued expenses paid", "Operating expenses", "Other outflows"]
    canon = canon_out if group == "Payments" else canon_in
    return sorted(labels, key=lambda l: canon.index(l) if l in canon else len(canon))


def build_cash_statement(flows: pd.DataFrame, cc_payments: pd.DataFrame,
                         app_closing: Dict[Tuple[int, int], float],
                         app_prev_closing: Optional[float],
                         months: List[Tuple[int, int]],
                         sub_names: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Month-column cash statement. Returns columns: Line, kind, then one per month.

    kind: header | line | subtotal | total | check | memo
    Opening cash of every column is the PREVIOUS column's app closing (so the
    columns chain exactly as the app's Cash & CE does); the first column uses
    `app_prev_closing`.
    """
    sub_names = sub_names or {}
    long = _flow_long(flows, sub_names)
    cols = [month_label(m) for m in months]
    rows: List[dict] = []

    def add(line, kind, vals=None):
        r = {"Line": line, "kind": kind}
        for c, m in zip(cols, months):
            r[c] = (vals or {}).get(m, np.nan) if vals is not None else np.nan
        rows.append(r)

    def series(sel: pd.DataFrame) -> Dict[Tuple[int, int], float]:
        if sel.empty:
            return {m: 0.0 for m in months}
        g = sel.groupby(["year", "month"])["amount"].sum()
        return {m: float(g.get(m, 0.0)) for m in months}

    # opening (app)
    opening, prev = {}, app_prev_closing
    for m in months:
        opening[m] = prev if prev is not None else np.nan
        prev = app_closing.get(m, np.nan)
    add("Opening cash & CE (app)", "total", opening)

    totals: Dict[str, Dict[Tuple[int, int], float]] = {}
    for group, title in (("Collections", "COLLECTIONS (sales operations)"),
                         ("Additional inflows", "ADDITIONAL INFLOWS (not from sales operations)"),
                         ("Payments", "PAYMENTS (cash out)")):
        add(title, "header")
        g = long[long["group"] == group]
        for label in _order_labels(list(g["label"].unique()), group):
            add("   " + label, "line", series(g[g["label"] == label]))
        totals[group] = series(g)
        add({"Collections": "Total collections", "Additional inflows": "Total additional inflows",
             "Payments": "TOTAL CASH OUT"}[group], "subtotal", totals[group])
        if group == "Additional inflows":
            add("TOTAL CASH IN", "total",
                {m: totals["Collections"][m] + totals["Additional inflows"][m] for m in months})

    net = {m: totals["Collections"][m] + totals["Additional inflows"][m] + totals["Payments"][m] for m in months}
    add("NET CHANGE IN CASH", "total", net)
    closing_model = {m: opening[m] + net[m] for m in months}
    add("Closing cash — model (opening + net change)", "total", closing_model)
    add("Closing cash & CE — APP (source of truth)", "total", {m: app_closing.get(m, np.nan) for m in months})
    add("Difference (model − app)", "check",
        {m: closing_model[m] - app_closing.get(m, np.nan) for m in months})

    # memo: payments funded straight from the overdraft (no cash moved)
    add("MEMO — paid from bank overdraft (CC), no cash moved", "header")
    if cc_payments is not None and not cc_payments.empty:
        c = cc_payments.copy()
        c["amount"] = pd.to_numeric(c["amount"], errors="coerce").fillna(0.0)
        c["label"] = [categorize_flow(a, s, "out", sub_names)[1] for a, s in zip(c["ac_code"], c["xsub"])]
        for label in _order_labels(list(c["label"].unique()), "Payments"):
            add("   " + label, "memo", series(c[c["label"] == label].assign(amount=lambda d: -d["amount"])))
        add("Total paid from CC overdraft", "subtotal", series(c.assign(amount=-c["amount"])))
    else:
        add("   (none in this window)", "memo", {m: 0.0 for m in months})
    return pd.DataFrame(rows)


def build_account_detail(flows: pd.DataFrame, names: Dict[str, str],
                         sub_names: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Every counter-account behind the cash statement: account, name, group/line, month columns."""
    if flows is None or flows.empty:
        return pd.DataFrame()
    sub_names = sub_names or {}
    long = _flow_long(flows, sub_names)
    long["key"] = long["ac_code"].astype(str) + np.where(long["xsub"].notna(), " / " + long["xsub"].astype(str), "")
    long["ym"] = list(zip(long["year"], long["month"]))
    pv = long.pivot_table(index=["group", "label", "ac_code", "key"], columns="ym",
                          values="amount", aggfunc="sum", fill_value=0.0)
    pv.columns = [month_label(c) for c in pv.columns]
    pv = pv.reset_index()
    pv.insert(3, "Account name", [names.get(str(c), "") for c in pv["ac_code"]])
    pv["Total"] = pv[[c for c in pv.columns if "-" in str(c) and c not in ("Account name",)]].sum(axis=1)
    return pv.drop(columns=["key"]).sort_values(["group", "label", "Total"])


# ── purchases ────────────────────────────────────────────────────────────────
_KF_RE = re.compile(r"KF\s*0*(\d+)\s*/\s*(?:20)?(\d{2})", re.IGNORECASE)


def parse_kf(text: str) -> Optional[Tuple[int, int]]:
    """'MDKF008/25,2025-12-17' or '... L/C KF 008/2025' -> (8, 25)."""
    m = _KF_RE.search(text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def build_shipments(shipments: pd.DataFrame, settlements: pd.DataFrame,
                    sub_names: Dict[str, str]) -> pd.DataFrame:
    """One row per import GRN with landed-cost components found in the L/C settlement vouchers.

    Pay order credits whose remark says "rebate" are transfers of the reclaimable
    VAT/AT to the rebate claim (not cost) and are shown separately; landed cost =
    vendor value + capitalised pay order + other import costs (incl. L/C charges).
    """
    cols = ["Shipment", "PO", "GRN", "GRN date", "Vendor value", "Pay order (capitalised)",
            "VAT/AT rebate transfer", "Other import costs", "Landed cost", "Settlement matched"]
    if shipments is None or shipments.empty:
        return pd.DataFrame(columns=cols)
    comp: Dict[Tuple[int, int], Dict[str, float]] = {}
    if settlements is not None and not settlements.empty:
        s = settlements.copy()
        s["amount"] = pd.to_numeric(s["amount"], errors="coerce").fillna(0.0)
        s["key"] = s["remark"].map(parse_kf)
        kind = s["xsub"].map(lambda x: sub_kind(sub_names.get(str(x), "")))
        rebate = s["remark"].str.contains("rebate", case=False, na=False)
        s["bucket"] = np.where((kind == "payorder") & rebate, "rebate",
                       np.where(kind == "payorder", "payorder",
                       np.where(kind.isin(["lcchg", "other"]), "other", "financing")))
        for (k, b), v in s.dropna(subset=["key"]).groupby(["key", "bucket"])["amount"].sum().items():
            comp.setdefault(k, {})[b] = float(v)
    out = []
    for r in shipments.itertuples(index=False):
        c = comp.get(parse_kf(r.shipment))
        vendor = float(r.vendor_value or 0)
        po = c.get("payorder", 0.0) if c else np.nan
        rb = c.get("rebate", 0.0) if c else np.nan
        other = c.get("other", 0.0) if c else np.nan
        landed = vendor + (po if c else 0.0) + (other if c else 0.0)
        out.append({"Shipment": str(r.shipment).split(",")[0], "PO": r.po, "GRN": r.grn,
                    "GRN date": pd.to_datetime(r.grn_date), "Vendor value": vendor,
                    "Pay order (capitalised)": po, "VAT/AT rebate transfer": rb,
                    "Other import costs": other, "Landed cost": landed, "Settlement matched": bool(c)})
    return pd.DataFrame(out, columns=cols)


def build_purchases(purch: pd.DataFrame, shipments_df: pd.DataFrame, mfg: pd.DataFrame,
                    advance_bal: Dict[Tuple[int, int], float],
                    months: List[Tuple[int, int]]) -> pd.DataFrame:
    """Purchases block (month columns): imports landed, local trading purchases, memo lines."""
    cols = [month_label(m) for m in months]
    rows: List[dict] = []

    def add(line, kind, vals):
        r = {"Line": line, "kind": kind}
        for c, m in zip(cols, months):
            r[c] = vals.get(m, 0.0) if vals is not None else np.nan
        rows.append(r)

    p = purch.copy() if purch is not None else pd.DataFrame(columns=["year", "month", "kind", "sold_item", "value", "grns", "items"])
    for c in ("value", "grns", "items"):
        if c in p:
            p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)

    def agg(sel, col="value"):
        if sel.empty:
            return {}
        g = sel.groupby(["year", "month"])[col].sum()
        return {m: float(g.get(m, 0.0)) for m in months}

    imp = p[p["kind"] == "Import"] if not p.empty else p
    loc_sold = p[(p["kind"] == "Local") & (p["sold_item"].astype(bool))] if not p.empty else p
    loc_ops = p[(p["kind"] == "Local") & (~p["sold_item"].astype(bool))] if not p.empty else p

    add("INTERNATIONAL PURCHASES (import records, IP-- / IGRN)", "header", None)
    add("   Vendor value of shipments received", "line", agg(imp))
    if shipments_df is not None and not shipments_df.empty:
        s = shipments_df.copy()
        s["year"], s["month"] = s["GRN date"].dt.year, s["GRN date"].dt.month

        def sa(col):
            g = s.groupby(["year", "month"])[col].sum(min_count=1)
            return {m: float(g.get(m, 0.0)) if pd.notna(g.get(m, np.nan)) else 0.0 for m in months}
        add("   + Pay order — capitalised customs duty", "line", sa("Pay order (capitalised)"))
        add("   + L/C charges & other import costs (insurance, C&F, freight, carrying)", "line", sa("Other import costs"))
        add("Landed cost of shipments received", "subtotal", sa("Landed cost"))
    if shipments_df is not None and not shipments_df.empty:
        add("   VAT/AT on pay order transferred to rebate claim (reclaimable, not cost)", "memo", sa("VAT/AT rebate transfer"))
    add("   Shipments (GRNs) received", "memo", agg(imp, "grns"))
    add("   International advances outstanding at month-end (01050001, app)", "memo",
        {m: advance_bal.get(m, np.nan) for m in months})

    add("LOCAL PURCHASES (PO-- / GRN-)", "header", None)
    add("   Trading purchases received (items that are sold)", "line", agg(loc_sold))
    add("   Excluded — operational / raw material / packaging (items never sold)", "memo", agg(loc_ops))
    add("   Trading items purchased (distinct)", "memo", agg(loc_sold, "items"))

    if mfg is not None and not mfg.empty:
        m2 = mfg.copy()
        for c in ("qty", "cost"):
            m2[c] = pd.to_numeric(m2[c], errors="coerce").fillna(0.0)
        add("MANUFACTURING (memo)", "header", None)
        add("   Finished goods manufactured — cost", "memo", agg(m2, "cost"))
        add("   Finished goods manufactured — quantity", "memo", agg(m2, "qty"))
    return pd.DataFrame(rows)


# ── inventory & sales ────────────────────────────────────────────────────────
def _ym_list(months: List[Tuple[int, int]]) -> List[int]:
    return [y * 100 + m for y, m in months]


def build_inventory_sales(stock: pd.DataFrame, sales: pd.DataFrame, returns: pd.DataFrame,
                          std_prices: pd.DataFrame, months: List[Tuple[int, int]]
                          ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Inventory ledger (real stock x average sales price) and sales after returns.

    Price for month m = trailing PRICE_WINDOW_MONTHS final sales / units per item
    (fallback: trailing 12 months, then standard price). The SAME price vector
    values that column's opening, movements and closing, so
    closing = opening + revaluation + movements holds exactly; the revaluation
    line carries the difference between last month's closing and this month's
    opening at the new prices.
    """
    cols = [month_label(m) for m in months]
    yms = _ym_list(months)
    info: Dict[str, float] = {}

    # ---- quantities by item x month, with opening balance
    st = stock.copy() if stock is not None else pd.DataFrame(columns=["itemcode", "doctype", "ym", "qty"])
    st["qty"] = pd.to_numeric(st["qty"], errors="coerce").fillna(0.0)
    items = sorted(st["itemcode"].unique())
    idx = {it: i for i, it in enumerate(items)}
    base = np.zeros(len(items))
    for it, q in st[st["ym"] == 0].groupby("itemcode")["qty"].sum().items():
        base[idx[it]] = q
    mv_total = np.zeros((len(items), len(yms)))
    grp_qty: Dict[str, np.ndarray] = {g: np.zeros((len(items), len(yms))) for g in
                                      ("Deliveries (DO--)", "Customer returns (SR--)", "Manufactured (RE--)",
                                       "Goods received & other receipts", "Other issues, adjustments & returns to suppliers")}
    pos = {ym: i for i, ym in enumerate(yms)}
    for r in st[st["ym"] != 0].itertuples(index=False):
        j = pos.get(int(r.ym))
        if j is None:
            continue
        i = idx[r.itemcode]
        mv_total[i, j] += r.qty
        if r.doctype == "DO--":
            g = "Deliveries (DO--)"
        elif r.doctype == "SR--":
            g = "Customer returns (SR--)"
        elif r.doctype == "RE--":
            g = "Manufactured (RE--)"
        else:
            g = "Goods received & other receipts" if r.qty > 0 else "Other issues, adjustments & returns to suppliers"
        grp_qty[g][i, j] += r.qty
    qty_end = base[:, None] + np.cumsum(mv_total, axis=1)
    qty_start = np.hstack([base[:, None], qty_end[:, :-1]])

    # ---- price vectors (trailing average sales price)
    sl = sales.copy() if sales is not None else pd.DataFrame(columns=["itemcode", "ym", "units", "gross", "discount"])
    for c in ("units", "gross", "discount"):
        sl[c] = pd.to_numeric(sl[c], errors="coerce").fillna(0.0)
    sl["final"] = sl["gross"] - sl["discount"]
    all_ym = sorted(set(sl["ym"].unique()) | set(yms))
    # contiguous month index so rolling windows span real months
    def ym_to_n(ym): return (ym // 100) * 12 + (ym % 100)
    n_of = {ym: ym_to_n(ym) for ym in all_ym}
    lookup_u = sl.groupby(["itemcode", "ym"])["units"].sum()
    lookup_f = sl.groupby(["itemcode", "ym"])["final"].sum()
    sp = {r.itemcode: float(r.std_price or 0) for r in std_prices.itertuples(index=False)} if std_prices is not None and not std_prices.empty else {}
    item_sales_n = {}
    for (it, ym), u in lookup_u.items():
        item_sales_n.setdefault(it, {})[n_of.get(ym, ym_to_n(ym))] = (float(u), float(lookup_f[(it, ym)]))

    def price_for(it: str, ym: int) -> Tuple[float, str]:
        n = ym_to_n(ym); d = item_sales_n.get(it, {})
        for span, tag in ((PRICE_WINDOW_MONTHS, "avg"), (PRICE_FALLBACK_MONTHS, "avg12")):
            u = sum(v[0] for k, v in d.items() if n - span < k <= n)
            f = sum(v[1] for k, v in d.items() if n - span < k <= n)
            if u > 0 and f > 0:
                return f / u, tag
        s = sp.get(it, 0.0)
        return (s, "std") if s > 0 else (0.0, "none")

    price = np.zeros((len(items), len(yms))); tags = np.empty((len(items), len(yms)), dtype=object)
    for j, ym in enumerate(yms):
        for it, i in idx.items():
            price[i, j], tags[i, j] = price_for(it, ym)

    # ---- ledger
    rows: List[dict] = []
    def add(line, kind, vals):
        r = {"Line": line, "kind": kind}
        for c, v in zip(cols, vals):
            r[c] = v
        rows.append(r)

    closing = (qty_end * price).sum(axis=0)
    opening_new = (qty_start * price).sum(axis=0)
    reval = np.zeros(len(yms)); opening_prev = np.zeros(len(yms))
    for j in range(len(yms)):
        if j == 0:
            opening_prev[j] = opening_new[j]
        else:
            opening_prev[j] = closing[j - 1]
            reval[j] = opening_new[j] - closing[j - 1]
    add("INVENTORY (real stock x average sales price)", "header", [np.nan] * len(yms))
    add("Opening inventory (prior month closing)", "total", list(opening_prev))
    add("   Revaluation at this month's average prices", "line", list(reval))
    for g, arr in grp_qty.items():
        add("   " + g, "line", list((arr * price).sum(axis=0)))
    add("Closing inventory", "total", list(closing))
    chk = opening_prev + reval + sum((a * price).sum(axis=0) for a in grp_qty.values()) - closing
    add("   check (opening + revaluation + movements − closing)", "check", list(chk))

    # ---- sales after returns (from accounts)
    gross = np.array([float(sl[sl["ym"] == ym]["gross"].sum()) for ym in yms])
    disc = np.array([float(sl[sl["ym"] == ym]["discount"].sum()) for ym in yms])
    rt = np.zeros(len(yms))
    if returns is not None and not returns.empty:
        for r in returns.itertuples(index=False):
            j = pos.get(int(r.year) * 100 + int(r.month))
            if j is not None:
                rt[j] += float(r.amount or 0)
    net_sales = gross - disc - rt
    add("SALES (from accounts, after returns)", "header", [np.nan] * len(yms))
    add("   Gross sales (trade value)", "line", list(gross))
    add("   − Discount", "line", list(-disc))
    add("   − Customer returns (AR credits, SRJV/SRT-)", "line", list(-rt))
    add("Net sales (after returns)", "total", list(net_sales))
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(opening_prev > 0, net_sales / opening_prev, np.nan)
    add("% of opening inventory sold", "pct", list(pct))
    cov = [float((qty_end[:, j].clip(min=0) * price[:, j])[np.isin(tags[:, j], ["avg", "avg12"])].sum()
                 / max((qty_end[:, j].clip(min=0) * price[:, j]).sum(), 1.0)) for j in range(len(yms))]
    add("   Closing stock valued at a sales-based average price (%)", "pct", cov)
    neg = [int((qty_end[:, j] < -0.0001).sum()) for j in range(len(yms))]
    add("   Items with negative book stock (count)", "memo", neg)
    info["items"] = float(len(items))
    return pd.DataFrame(rows), info


# ── export ───────────────────────────────────────────────────────────────────
def to_excel_bytes(sections: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sections.items():
            if df is None or df.empty:
                continue
            out = df.drop(columns=["kind"], errors="ignore")
            out.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = 60 if col[0].column == 1 else 14
            for row in ws.iter_rows(min_row=2, min_col=2):
                for c in row:
                    if isinstance(c.value, (int, float)):
                        c.number_format = "#,##0.00;[Red]-#,##0.00"
    return buf.getvalue()
