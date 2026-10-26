# views/cash_model.py
# Financial Statements -> "💵 Cash Model". One entity, monthly columns over the
# same window as the Monthly statements (Jan of year-1 .. chosen month of year):
#   1. Purchases   (import records IP--/IGRN + local trading purchases + memo)
#   2. Inventory & Sales (real stock x average sales price; sales after returns)
#   3. Cash        (collections, additional inflows, payments -> closing cash)
# Opening/closing cash come from the app's own Cash & CE (financial.cash_open_close)
# and the cash statement ties back to it row by row. Pure display; the pandas
# lives in processing/cash_model.py.
# Selecting 100001 also analyses 100009 (Gulshan Packaging) with it: imports are split
# between the two on the same MDKF shipment, 100009 has no cash/advance/sales of its
# own, and transfers between the two are eliminated (processing.cash_model.GROUPS).

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
def _app_cash_one(zid: str, project, year: int, end_month: int):
    """The app's own Cash & CE per month (source of truth) + 01050001 / 09060001 balances.

    Runs the identical pipeline the Monthly statements use (process_data_month ->
    cash_open_close). A second call for year-1 supplies the closing of
    Dec (year-2), i.e. the opening of the first window column.
    """
    bl = common.load_json("data/labels.json")["balance_sheet_label"]
    label_df = pd.DataFrame(list(bl.items()), columns=["ac_lv4", "Balance Sheet"])
    skip = {"Income", "Expenditure"}

    def closing_map(y, em):
        today = datetime.now()
        if y != today.year and em == today.month and today.month != 12:
            em = 12      # process_data_month drops "today's month" even for a past year; pull the full year instead
        bs = financial.process_data_month(zid, y, 1, em, "Balance Sheet", label_df, project, skip)
        bs0 = bs.drop(columns=_HIER, errors="ignore")
        coc = financial.cash_open_close(bs0).set_index("ac_name").T
        out = {k: float(v) for k, v in coc["Closing Cash & CE"].items() if isinstance(k, tuple)}
        return out, bs0

    now = datetime.now()
    is_mtd = (year == now.year and end_month == now.month)          # the app's statements leave out the current month
    done = end_month - 1 if is_mtd else end_month                    # last month the app has closed
    if done >= 1:
        closing, bs0 = closing_map(year, done)
    else:                                                            # January: nothing closed yet this year
        closing, bs0 = {}, pd.DataFrame(columns=["ac_code"])
    prev, _ = closing_map(year - 1, 12)
    prev_closing = prev.get((year - 2, 12))
    if is_mtd:
        # closing cash to date = last closed month's app closing + this month's movement on the cash accounts
        base = closing.get((year, done)) if done >= 1 else prev.get((year - 1, 12))
        mv = Analytics("cash_model_cash_movement", zid=zid, project=project,
                       filters={"year": year, "end_month": end_month, "cash_codes": financial._cash_codes_from_json()}).data
        if base is not None and mv is not None and not mv.empty:
            closing[(year, end_month)] = float(base) + float(mv["net"].iloc[0])

    def bal(code):
        r = bs0[bs0["ac_code"] == code]
        if r.empty:
            return {}
        return {c: float(r.iloc[0][c]) for c in r.columns if isinstance(c, tuple)}

    return {"closing": closing, "prev_closing": prev_closing,
            "advance": bal("01050001"), "gi_custody": bal("09060001"), "gi_at_gtc": bal("01020003"),
            "ltr": bal("10010001"),
            # debt accounts for the projection (liabilities are negative in the app; callers take abs)
            "debts": {code: bal(code) for code in ("10010003", "10010006", "10010007", "09040007", "09040001",
                                                   "10020001", "10020002", "10020016")}}


def _sum_maps(maps):
    out: dict = {}
    for mp in maps:
        for k, v in mp.items():
            out[k] = out.get(k, 0.0) + v
    return out


def _app_cash(members, year: int, end_month: int):
    """Cash & CE of the selected business = sum over the businesses analysed together.

    members = ((zid, project), ...). Gulshan Packaging holds no cash of its own, but
    whatever the app holds counts, so it is summed in.
    """
    parts = [_app_cash_one(z, p, year, end_month) for z, p in members]
    prev = [x["prev_closing"] for x in parts if x["prev_closing"] is not None]
    return {"closing": _sum_maps(x["closing"] for x in parts),
            "prev_closing": sum(prev) if prev else None,
            "advance": _sum_maps(x["advance"] for x in parts),
            "gi_custody": _sum_maps(x["gi_custody"] for x in parts),
            "gi_at_gtc": _sum_maps(x["gi_at_gtc"] for x in parts)}


@st.cache_data(show_spinner=False, ttl=1800)
def _load(members, year: int, end_month: int):
    """members = ((zid, project), ...); the FIRST is the selected business (its sales, advance
    and supplier code define the group; the others are folded in)."""
    primary = members[0][0]
    grouped = len(members) > 1
    codes = financial._cash_codes_from_json()
    out = {k: [] for k in ("flows", "cc", "ship", "settle", "purch", "stock", "mfg", "names", "xcost")}
    for z, project in members:
        is_primary = z == primary
        f = {"year": year, "end_month": end_month, "cash_codes": codes, "cc_codes": cm.CC_CODES,
             "keep_sub": cm.KEEP_SUB_ACCS.get(z) if grouped else ["01050001"],
             "sales_zid": primary, "map_codes": not is_primary,
             "internal_supplier": cm.INTERNAL_SUPPLIER.get(primary) if is_primary and grouped else None}

        def get(name, _f=f, _z=z, _p=project):
            d = Analytics(name, zid=_z, project=_p, filters=_f).data
            return d if d is not None else pd.DataFrame()

        for key, name in (("flows", "cash_model_flows"), ("cc", "cash_model_cc_payments"),
                          ("ship", "cash_model_import_shipments"), ("settle", "cash_model_import_settlements"),
                          ("purch", "cash_model_purchases_monthly"), ("stock", "cash_model_stock_movement"),
                          ("mfg", "cash_model_mfg_monthly"), ("names", "cash_model_account_names"),
                          *((("xcost", "cash_model_internal_transfer_cost"),) if grouped and not is_primary else ())):
            d = get(name)
            if not d.empty and key in ("flows", "cc", "ship", "mfg"):
                d = d.assign(zid=z)
            out[key].append(d)
        if is_primary:
            out["sales"], out["returns"], out["std"] = (get("cash_model_sales_monthly"),
                                                        get("cash_model_returns_monthly"),
                                                        get("cash_model_item_prices"))

    def cat(key, group_cols=None):
        frames = [d for d in out[key] if not d.empty]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        if group_cols:   # the same item/doctype/month can arrive from two businesses -> add up
            val = [c for c in df.columns if c not in group_cols]
            df = df.groupby(group_cols, as_index=False)[val].sum()
        return df

    names, subs = {}, {}
    for nd in reversed(out["names"]):          # primary last so its names win
        if not nd.empty:
            names.update({r.code: r.name for r in nd[nd["kind"] == "acc"].itertuples()})
            subs.update({r.code: r.name for r in nd[nd["kind"] == "sub"].itertuples()})
    return {
        "flows": cat("flows"), "cc": cat("cc"), "ship": cat("ship"), "settle": cat("settle"),
        "purch": cat("purch"), "xcost": cat("xcost"), "stock": cat("stock", ["itemcode", "doctype", "ym"]),
        "sales": out["sales"], "returns": out["returns"], "mfg": cat("mfg"), "std": out["std"],
        "names": names, "subs": subs, "codes": codes,
    }


def build_cash_model(zids, projects, year: int, end_month: int) -> dict:
    """Load + transform; returns the section DataFrames (used by the UI and by tests/scripts).

    zids[0] is the selected business; any further ones are analysed together with it."""
    members = tuple(zip(zids, projects))
    months = cm.window_months(year, end_month)
    app = _app_cash(members, year, end_month)
    d = _load(members, year, end_month)
    cash = cm.build_cash_statement(d["flows"], d["cc"], app["closing"], app["prev_closing"], months, d["subs"])
    ships = cm.build_shipments(d["ship"], d["settle"], d["subs"])
    purchases = cm.build_purchases(d["purch"], ships, d["mfg"], app["advance"], months, d["xcost"])
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
    final = cm.build_final_model(purchases, inv, cash, cols)
    frames = {"cash": cash, "purchases": purchases, "inventory": inv, "detail": detail, "gi": gi, "final": final}
    now = datetime.now()
    mtd_col = None
    if months[-1] == (now.year, now.month):                      # current month = month to date
        mtd_col = cols[-1] + " (MTD)"
        frames = {k: v.rename(columns={cols[-1]: mtd_col}) for k, v in frames.items()}
        cols = cols[:-1] + [mtd_col]
    return {"months": months, "cols": cols, **frames, "shipments": ships, "app": app, "info": info, "mtd_col": mtd_col}


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
    folded = {z for members in cm.GROUPS.values() for z in members[1:]}   # e.g. 100009 -> analysed with 100001
    keys = [k for k in businesses.keys() if k not in folded]
    default = keys.index(str(zid)) if str(zid) in keys else 0
    now = datetime.now()
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        ent = st.selectbox("Business", keys, index=default,
                           format_func=lambda k: f"{k} — {businesses[k].get('zorg', '')}", key="cm_entity")
    with c2:
        year = st.selectbox("End year", [now.year - i for i in range(5)], index=0, key="cm_year")
    max_month = now.month if year == now.year else 12          # the current month is shown month-to-date
    with c3:
        end_month = st.selectbox("Up to month", list(range(1, max_month + 1)), index=max_month - 1, key="cm_month")
    zids = [z for z in cm.group_zids(str(ent)) if z in businesses] or [str(ent)]
    projects = [(businesses[z].get("projects") or [None])[0] for z in zids]
    if len(zids) > 1:
        st.info("📦 **Combined view:** " + " + ".join(f"{z} {businesses[z].get('zorg', '').rstrip('.')}" for z in zids)
                + ". Imports are split between these businesses on the same MDKF shipment, so they are analysed "
                  "together: cash, shipments, purchases and stock cover both, and transfers between them are "
                  "eliminated. Sales are the external sales of " + f"{zids[0]}.")

    try:
        with st.spinner("Building the cash model…"):
            m = build_cash_model(zids, projects, int(year), int(end_month))
    except Exception as e:  # keep the page usable if one query fails
        st.warning(f"Could not build the cash model for {ent}: {e}")
        return

    cols = m["cols"]
    _tie_banner(m["cash"], cols)
    if m.get("mtd_col"):
        st.caption(f"**{m['mtd_col']}** is the current month to date. The app's statements leave out the incomplete month, so its "
                   "'APP' row is the ledger balance of the cash accounts to date (last closed month's app closing + this month's "
                   "movement). Everything below is real entries so far; it changes as the month goes on.")

    tab0, tabp, tab1, tab2, tab3, tab4, tabsp, tabs_ = st.tabs(["📋 Final Model", "📈 Projection", "📦 Purchases",
                                                                "🏬 Inventory & Sales", "💵 Cash", "🔎 Detail by account",
                                                                "🗓 Shipment Plan", "⚙️ Settings"])
    # The Settings tab is the last one, but its widgets must run first so the Projection reads their live values.
    from views import cash_projection
    ctx = cash_projection.prepare(str(zids[0]), zids, projects)
    with tabsp:      # the same page the commercial officer uses in Purchase Analysis -> Shipment Plan
        from views import shipment_plan
        shipment_plan.render_shipment_plan(str(zids[0]))
    with tabs_:
        proj_settings = cash_projection.render_settings(ctx) if ctx else None
    with tabp:
        if ctx:
            cash_projection.render_projection(ctx, proj_settings)
    with tab0:
        st.caption("One fixed-layout table: purchases → inventory & sales → collections → additional inflows → payments → "
                   "net cash. Every figure is taken from the other tabs (they stay as they are); a line with no activity "
                   "shows 0 so the layout never changes. Includes one line beyond the template — the monthly "
                   "revaluation of inventory — so opening plus movements adds up to closing.")
        _show(m["final"], cols, height=min(35 * (len(m["final"]) + 1) + 3, 2600))
    with tab1:
        st.caption("International purchases come from the import records (IP-- POs / IGRN receipts) with the "
                   "pay order and other import costs found in the L/C settlement vouchers. Local purchases count as "
                   "**trading** only when the item is sold; raw materials, packaging and consumables are shown as excluded.")
        _show(m["purchases"], cols)
        sh = m["shipments"]
        if not sh.empty:
            st.subheader("Import shipments received in the window (one row per MDKF shipment)")
            disp = sh.copy()
            disp["GRN date"] = disp["GRN date"].dt.strftime("%d-%b-%Y")
            money = [c for c in disp.columns if c.startswith("Vendor value")] + [
                "Pay order (capitalised)", "VAT/AT rebate transfer", "Other import costs", "Landed cost"]
            st.dataframe(disp.style.format({c: "{:,.0f}" for c in money}, na_rep="—"),
                         use_container_width=True, hide_index=True)
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
            num = [c for c in view.columns if c not in ("group", "label", "ac_code", "Account name", "Business")]
            st.dataframe(view.style.format({c: "{:,.0f}" for c in num}), use_container_width=True, hide_index=True)

    sections = {"Final Model": m["final"], "Purchases": m["purchases"], "Inventory & Sales": m["inventory"], "Cash": pd.concat([m["cash"], m["gi"]]),
                "Shipments": m["shipments"], "Detail by account": m["detail"]}
    st.download_button("⬇️ Download Excel", data=cm.to_excel_bytes(sections),
                       file_name=f"cash_model_{ent}_{year}_{int(end_month):02d}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
