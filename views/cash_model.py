# views/cash_model.py
# Financial Statements -> "💵 Cash Model". One entity, monthly columns over the
# same window as the Monthly statements (Jan of year-1 .. chosen month of year):
#   1. Purchases   (import records IP--/IGRN + local trading purchases + memo)
#   2. Inventory & Sales (real stock x average sales price; sales after returns)
#   3. Cash        (collections, additional inflows, payments -> closing cash)
# Opening/closing cash come from the app's own Cash & CE (financial.cash_open_close)
# and the cash statement ties back to it row by row. Pure display; the pandas
# lives in processing/cash_model.py.

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from core.analytics import Analytics
from processing import cash_model as cm
from processing import common, financial

_HIER = ["ac_type", "ac_lv1", "ac_lv2", "ac_lv3", "ac_lv4", "ac_lv5"]


# ── data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=1800)
def _app_cash(zid: str, project, year: int, end_month: int):
    """The app's own Cash & CE per month (source of truth) + 01050001 / 09060001 balances.

    Runs the identical pipeline the Monthly statements use (process_data_month ->
    cash_open_close). A second call for year-1 supplies the closing of
    Dec (year-2), i.e. the opening of the first window column.
    """
    bl = common.load_json("data/labels.json")["balance_sheet_label"]
    label_df = pd.DataFrame(list(bl.items()), columns=["ac_lv4", "Balance Sheet"])
    skip = {"Income", "Expenditure"}

    def closing_map(y, em):
        bs = financial.process_data_month(zid, y, 1, em, "Balance Sheet", label_df, project, skip)
        bs0 = bs.drop(columns=_HIER, errors="ignore")
        coc = financial.cash_open_close(bs0).set_index("ac_name").T
        out = {k: float(v) for k, v in coc["Closing Cash & CE"].items() if isinstance(k, tuple)}
        return out, bs0

    closing, bs0 = closing_map(year, end_month)
    prev, _ = closing_map(year - 1, 12)
    prev_closing = prev.get((year - 2, 12))

    def bal(code):
        r = bs0[bs0["ac_code"] == code]
        if r.empty:
            return {}
        return {c: float(r.iloc[0][c]) for c in r.columns if isinstance(c, tuple)}

    return {"closing": closing, "prev_closing": prev_closing,
            "advance": bal("01050001"), "gi_custody": bal("09060001"), "gi_at_gtc": bal("01020003")}


@st.cache_data(show_spinner=False, ttl=1800)
def _load(zid: str, project, year: int, end_month: int):
    codes = financial._cash_codes_from_json()
    f = {"year": year, "end_month": end_month, "cash_codes": codes, "cc_codes": cm.CC_CODES}

    def get(name):
        d = Analytics(name, zid=zid, project=project, filters=f).data
        return d if d is not None else pd.DataFrame()

    names_df = get("cash_model_account_names")
    names = {r.code: r.name for r in names_df[names_df["kind"] == "acc"].itertuples()} if not names_df.empty else {}
    subs = {r.code: r.name for r in names_df[names_df["kind"] == "sub"].itertuples()} if not names_df.empty else {}
    return {
        "flows": get("cash_model_flows"), "cc": get("cash_model_cc_payments"),
        "ship": get("cash_model_import_shipments"), "settle": get("cash_model_import_settlements"),
        "purch": get("cash_model_purchases_monthly"), "stock": get("cash_model_stock_movement"),
        "sales": get("cash_model_sales_monthly"), "returns": get("cash_model_returns_monthly"),
        "mfg": get("cash_model_mfg_monthly"), "std": get("cash_model_item_prices"),
        "names": names, "subs": subs, "codes": codes,
    }


def build_cash_model(zid: str, project, year: int, end_month: int) -> dict:
    """Load + transform; returns the section DataFrames (used by the UI and by tests/scripts)."""
    months = cm.window_months(year, end_month)
    app = _app_cash(zid, project, year, end_month)
    d = _load(zid, project, year, end_month)
    cash = cm.build_cash_statement(d["flows"], d["cc"], app["closing"], app["prev_closing"], months, d["subs"])
    ships = cm.build_shipments(d["ship"], d["settle"], d["subs"])
    purchases = cm.build_purchases(d["purch"], ships, d["mfg"], app["advance"], months)
    inv, info = cm.build_inventory_sales(d["stock"], d["sales"], d["returns"], d["std"], months)
    detail = cm.build_account_detail(d["flows"], d["names"], d["subs"])
    cols = [cm.month_label(m) for m in months]
    memo = []
    for label, key in (("MEMO — GI's money held in HMBR's accounts (09060001 balance, app; liability to GI)", "gi_custody"),
                       ("MEMO — of which GI's money held at HMBR (01020003 'GI Bank balance In (GTC)', included in GI's cash)", "gi_at_gtc")):
        vals = {c: app[key].get(m, np.nan) for c, m in zip(cols, months)}
        if any(pd.notna(v) and abs(v) > 0.5 for v in vals.values()):
            memo.append({"Line": label, "kind": "memo", **vals})
    gi = pd.DataFrame(memo, columns=["Line", "kind"] + cols)
    return {"months": months, "cols": cols, "cash": cash, "purchases": purchases, "inventory": inv,
            "shipments": ships, "detail": detail, "gi": gi, "app": app, "info": info}


# ── display helpers ──────────────────────────────────────────────────────────

def _style(df: pd.DataFrame, cols):
    kinds = df["kind"].tolist()
    show = df.drop(columns=["kind"]).reset_index(drop=True)
    # Streamlit renders NaN cells of a styled frame as the word "None" (na_rep is ignored),
    # so empty cells (section headers, first-column opening) are held as "" instead.
    show[cols] = show[cols].astype(object).where(show[cols].notna(), "")

    def row_css(row):
        k = kinds[row.name]
        base = {"header": "font-weight:700;background-color:rgba(120,140,200,0.18)",
                "total": "font-weight:700;border-top:1px solid rgba(128,128,128,0.5)",
                "subtotal": "font-weight:600",
                "check": "color:#2e7d32;font-style:italic",
                "memo": "color:rgba(128,128,128,0.95);font-style:italic"}.get(k, "")
        return [base] * len(row)

    def make_fmt(fmt):
        # header rows carry None/NaN in the month cells -> blank, never the word "None"
        def f(v):
            if v is None or isinstance(v, str) or (isinstance(v, float) and np.isnan(v)):
                return ""
            return fmt.format(v)
        return f

    sty = show.style.apply(row_css, axis=1)
    for i, k in enumerate(kinds):
        fmt = "{:,.2f}" if k == "check" else ("{:.1%}" if k == "pct" else "{:,.0f}")
        sty = sty.format(make_fmt(fmt), subset=pd.IndexSlice[[i], cols])
    return sty


def _show(df: pd.DataFrame, cols, height=None):
    st.dataframe(_style(df, cols), use_container_width=True, hide_index=True,
                 height=height or min(35 * (len(df) + 1) + 3, 1500),
                 column_config={"Line": st.column_config.TextColumn("Line", pinned=True, width=430)})


def _tie_banner(cash: pd.DataFrame, cols):
    r = cash[cash["Line"] == "Difference (model − app)"].iloc[0]
    diffs = pd.to_numeric(r[cols], errors="coerce")
    worst = float(diffs.abs().max())
    bad = [c for c, v in diffs.items() if pd.notna(v) and abs(v) >= 1.0]
    if not bad:
        st.success(f"✅ Closing cash ties to the app's Cash & CE in all {len(cols)} months "
                   f"(largest difference {worst:,.2f}).")
    else:
        st.warning(f"⚠️ Model closing cash differs from the app's Cash & CE by ≥ 1 in: {', '.join(bad)} "
                   f"(largest {worst:,.2f}). See the 'Difference' row.")


# ── page ─────────────────────────────────────────────────────────────────────

def display_cash_model(zid):
    st.title("Cash Model — monthly cash, purchases, inventory & sales")
    st.caption("Opening and closing cash are the app's own **Cash & CE** (the source of truth). "
               "Everything below explains the movement between them from the ledger. "
               "The window is the same as the Monthly statements: January of the previous year up to the chosen month.")

    businesses = common.load_json("data/businesses.json").get("businesses", {})
    keys = [k for k in businesses.keys()]
    default = keys.index(str(zid)) if str(zid) in keys else 0
    now = datetime.now()
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        ent = st.selectbox("Business", keys, index=default,
                           format_func=lambda k: f"{k} — {businesses[k].get('zorg', '')}", key="cm_entity")
    with c2:
        year = st.selectbox("End year", [now.year - i for i in range(5)], index=0, key="cm_year")
    max_month = now.month - 1 if year == now.year else 12
    if max_month < 1:
        st.info("No completed month yet in this year — choose the previous year.")
        return
    with c3:
        end_month = st.selectbox("Up to month", list(range(1, max_month + 1)), index=max_month - 1, key="cm_month")
    project = (businesses[ent].get("projects") or [None])[0]

    try:
        with st.spinner("Building the cash model…"):
            m = build_cash_model(str(ent), project, int(year), int(end_month))
    except Exception as e:  # keep the page usable if one query fails
        st.warning(f"Could not build the cash model for {ent}: {e}")
        return

    cols = m["cols"]
    _tie_banner(m["cash"], cols)

    tab1, tab2, tab3, tab4 = st.tabs(["📦 Purchases", "🏬 Inventory & Sales", "💵 Cash", "🔎 Detail by account"])
    with tab1:
        st.caption("International purchases come from the import records (IP-- POs / IGRN receipts) with the "
                   "pay order and other import costs found in the L/C settlement vouchers. Local purchases count as "
                   "**trading** only when the item is sold; raw materials, packaging and consumables are shown as excluded.")
        _show(m["purchases"], cols)
        sh = m["shipments"]
        if not sh.empty:
            st.subheader("Import shipments received in the window")
            disp = sh.copy()
            disp["GRN date"] = disp["GRN date"].dt.strftime("%d-%b-%Y")
            st.dataframe(disp.style.format({c: "{:,.0f}" for c in ["Vendor value", "Pay order (capitalised)", "VAT/AT rebate transfer", "Other import costs", "Landed cost"]},
                                           na_rep="—"), use_container_width=True, hide_index=True)
        else:
            st.info("No import shipments (IP-- receipts) recorded for this business in the window.")
    with tab2:
        st.caption("Inventory is **real book stock** valued at each product's **average sales price** "
                   f"(trailing {cm.PRICE_WINDOW_MONTHS} months of sales; falls back to {cm.PRICE_FALLBACK_MONTHS} months, then standard price). "
                   "Sales are taken from accounts and are **after returns**. The ledger is for the sold-% ratio; cash does not depend on it.")
        _show(m["inventory"], cols)
    with tab3:
        _show(m["cash"], cols)
        st.caption("Payments made from the bank overdraft move no cash (they raise the overdraft); repaying the overdraft "
                   "is a cash outflow, and drawing on it is a cash inflow — both appear above when the money moves.")
        if not m["gi"].empty:
            _show(m["gi"], cols)
    with tab4:
        det = m["detail"]
        if det.empty:
            st.info("No cash movements in this window.")
        else:
            grp = st.selectbox("Group", ["All"] + sorted(det["group"].unique()), key="cm_det_group")
            view = det if grp == "All" else det[det["group"] == grp]
            num = [c for c in view.columns if c not in ("group", "label", "ac_code", "Account name")]
            st.dataframe(view.style.format({c: "{:,.0f}" for c in num}), use_container_width=True, hide_index=True)

    sections = {"Purchases": m["purchases"], "Inventory & Sales": m["inventory"], "Cash": pd.concat([m["cash"], m["gi"]]),
                "Shipments": m["shipments"], "Detail by account": m["detail"]}
    st.download_button("⬇️ Download Excel", data=cm.to_excel_bytes(sections),
                       file_name=f"cash_model_{ent}_{year}_{int(end_month):02d}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
