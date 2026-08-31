import calendar

import pandas as pd
import streamlit as st

from core.analytics import Analytics
from processing import manufacturing as mfg
from utils.utils import timed

# ── ZID scope ──────────────────────────────────────────────────────────────────
# Only businesses with manufacturing orders wired into this tool.
_MANUFACTURING_ZIDS = ("100000", "100005", "100009")


# ── Cached loaders ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_mo_header(zid: str) -> pd.DataFrame:
    df = Analytics("mo_header", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_mo_detail(zid: str) -> pd.DataFrame:
    df = Analytics("mo_detail", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_admin_expense(zid: str) -> pd.DataFrame:
    df = Analytics("admin_expense_monthly", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_sales_window(zid: str, years: tuple) -> pd.DataFrame:
    df = Analytics("sales", zid=zid, filters={"year": list(years)}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_returns_window(zid: str, years: tuple) -> pd.DataFrame:
    df = Analytics("return", zid=zid, filters={"year": list(years)}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_opspprc(zid: str) -> pd.DataFrame:
    df = Analytics("opspprc", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_manufacturing_flow(zid: str) -> pd.DataFrame:
    """Raw warehouse-flow movement (RM+FG+Sales warehouses only) for one
    entity — full history, no date filter; the Warehouse Flow view slices
    an arbitrary date range out of this in Python. See
    processing/manufacturing.py::WAREHOUSE_GROUPS/compute_warehouse_flow."""
    warehouses = mfg.all_flow_warehouses(zid)
    if not warehouses:
        return pd.DataFrame()
    df = Analytics("manufacturing_flow_detail", zid=zid, filters={"warehouses": warehouses}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_stock_raw(zid: str) -> pd.DataFrame:
    """Raw imtrn-based 'stock' table (year/month movement buckets) — summed
    with no cutoff in compute_current_stock_from_imtrn() to get the current
    cumulative balance. Deliberately not final_items_view here — see the
    docstring on that function for why.
    """
    df = Analytics("stock", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


# ── Small shared helpers ───────────────────────────────────────────────────────

def _item_picker(label: str, df: pd.DataFrame, code_col: str, name_col: str, key: str):
    """Code+name selectbox; returns the selected code or None."""
    if df is None or df.empty or code_col not in df.columns:
        st.info("No items available for the current window.")
        return None
    opts = df[[code_col, name_col]].dropna(subset=[code_col]).drop_duplicates(code_col).copy()
    opts[name_col] = opts[name_col].fillna("")
    opts = opts.sort_values(name_col)
    labels = [f"{r[name_col]} ({r[code_col]})" for _, r in opts.iterrows()]
    code_by_label = dict(zip(labels, opts[code_col]))
    sel = st.selectbox(label, labels, key=key)
    return code_by_label.get(sel)


def _fmt(df: pd.DataFrame, fmt: dict, na_rep="—"):
    try:
        return df.style.format(fmt, na_rep=na_rep)
    except Exception:
        return df


# ── FG Costing ───────────────────────────────────────────────────────────────────

def _render_fg_costing(zid: str, mo_cost: pd.DataFrame, mo_lines: pd.DataFrame, admin_expense: pd.DataFrame, opspprc_df: pd.DataFrame = None):
    st.subheader("🏭 Finished Good Costing")
    st.caption("📝 Methodology — see Note [1] at the bottom of the page.")

    today = pd.Timestamp.today().normalize()
    cutoff_30d = today - pd.Timedelta(days=30)
    _sales_30d_years = tuple(sorted({cutoff_30d.year, today.year}))
    _sales_raw = _load_sales_window(str(zid), _sales_30d_years)

    n_months = st.slider("Costing window (trailing completed months)", 1, 12, 3, key="mfg_costing_window")
    start, end = mfg.trailing_n_months_window(today, n_months)
    st.markdown(f"**Window:** {start.strftime('%b %Y')} – {end.strftime('%b %Y')}")

    fg_summary = mfg.compute_fg_cost_summary(mo_cost, start, end)
    if fg_summary.empty:
        st.info("No completed manufacturing orders found in this window.")
        return

    col_map = {
        "itemcode": "Item Code", "itemname": "Item Name", "itemgroup": "Item Group",
        "total_qtyprd": "Qty Produced", "n_batches": "Batches",
        "avg_cost_per_unit": "Avg Material Cost/Unit", "latest_cost_per_unit": "Latest Material Cost/Unit",
        "total_material_cost": "Total Material Cost",
    }
    disp = fg_summary.rename(columns=col_map)[[c for c in col_map.values() if c in fg_summary.rename(columns=col_map).columns]]
    st.dataframe(
        _fmt(disp, {
            "Qty Produced": "{:,.0f}", "Batches": "{:,.0f}",
            "Avg Material Cost/Unit": "{:,.2f}", "Latest Material Cost/Unit": "{:,.2f}",
            "Total Material Cost": "{:,.0f}",
        }),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "⬇ Download FG Costing CSV", fg_summary.to_csv(index=False).encode("utf-8"),
        file_name=f"fg_costing_{zid}.csv", mime="text/csv", key="dl_mfg_fg_costing",
    )

    st.markdown("---")
    st.markdown("**🔍 Cost breakdown for one finished good**")
    sel_fg = _item_picker("Select Finished Good", fg_summary, "itemcode", "itemname", "mfg_fg_costing_pick")
    if not sel_fg:
        return

    drivers = mfg.compute_cost_driver_breakdown(mo_lines, sel_fg, start, end)
    alloc_df, alloc_avg_admin = mfg.compute_admin_allocation_for_fg(mo_cost, admin_expense, sel_fg, start, end)

    fg_row = fg_summary[fg_summary["itemcode"] == sel_fg].iloc[0]
    total_cost_per_unit = float(fg_row["avg_cost_per_unit"]) + alloc_avg_admin

    avg_price_30d = None
    if not _sales_raw.empty and "itemcode" in _sales_raw.columns:
        mask = (
            (_sales_raw["itemcode"].astype(str) == str(sel_fg))
            & (pd.to_datetime(_sales_raw["date"]) >= cutoff_30d)
        )
        _item_sales = _sales_raw.loc[mask]
        if not _item_sales.empty:
            _total_qty = float(_item_sales["quantity"].sum())
            if _total_qty > 0:
                avg_price_30d = float(_item_sales["altsales"].sum()) / _total_qty

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Material Cost/Unit (window avg)", f"{fg_row['avg_cost_per_unit']:,.2f}")
    m2.metric("Admin (06) Cost/Unit (allocated)", f"{alloc_avg_admin:,.2f}")
    m3.metric("Total Cost/Unit", f"{total_cost_per_unit:,.2f}")
    m4.metric("Avg Sales Price (last 30d)", f"{avg_price_30d:,.2f}" if avg_price_30d is not None else "—")

    if opspprc_df is not None and not opspprc_df.empty and "item_id" in opspprc_df.columns:
        price_row = opspprc_df[opspprc_df["item_id"].astype(str) == str(sel_fg)]
        if not price_row.empty:
            r = price_row.iloc[0]
            std_price = float(r.get("xstdprice", 0) or 0)
            wh_price  = float(r.get("wh_price",  0) or 0)
            st.markdown("**💰 Selling Price Reference**")
            p1, p2, p3 = st.columns(3)
            p1.metric("Standard Price (caitem)", f"{std_price:,.2f}")
            p2.metric("Wholesale Price (opspprc)", f"{wh_price:,.2f}")
            margin = std_price - total_cost_per_unit if std_price > 0 else None
            p3.metric("Margin vs Total Cost", f"{margin:,.2f}" if margin is not None else "—")

    st.markdown("**Cost driver breakdown (raw materials, by share of material cost)**")
    if drivers.empty:
        st.info("No BOM lines found for this finished good in the window.")
    else:
        d_col_map = {
            "itemcode": "RM Code", "itemname": "RM Name", "total_qty": "Total Qty",
            "avg_rate": "Avg Rate", "total_cost": "Total Cost", "pct_of_total": "% of Total Cost",
        }
        ddisp = drivers.rename(columns=d_col_map)[[c for c in d_col_map.values() if c in drivers.rename(columns=d_col_map).columns]]
        st.dataframe(
            _fmt(ddisp, {"Total Qty": "{:,.2f}", "Avg Rate": "{:,.2f}", "Total Cost": "{:,.0f}", "% of Total Cost": "{:,.1f}%"}),
            width="stretch", hide_index=True,
        )

    st.markdown("**Admin (06) cost allocation, by month**")
    if alloc_df.empty:
        st.info("No GL '06' (Office & Admin) expense or production found for this finished good in the window.")
    else:
        a_col_map = {
            "year": "Year", "month": "Month", "fg_material_cost": "FG Material Cost",
            "fg_qtyprd": "Qty Produced", "total_fg_cost": "Zid Total Material Cost",
            "admin_expense": "Total 06 Expense", "fg_share": "FG Share",
            "allocated_admin": "Allocated Admin", "admin_cost_per_unit": "Admin Cost/Unit",
        }
        adisp = alloc_df.rename(columns=a_col_map)[[c for c in a_col_map.values() if c in alloc_df.rename(columns=a_col_map).columns]]
        st.dataframe(
            _fmt(adisp, {
                "FG Material Cost": "{:,.0f}", "Qty Produced": "{:,.0f}", "Zid Total Material Cost": "{:,.0f}",
                "Total 06 Expense": "{:,.0f}", "FG Share": "{:.2%}", "Allocated Admin": "{:,.0f}", "Admin Cost/Unit": "{:,.2f}",
            }),
            width="stretch", hide_index=True,
        )


# ── FG Cost History ──────────────────────────────────────────────────────────────

def _render_fg_cost_history(zid: str, mo_cost: pd.DataFrame, mo_header: pd.DataFrame):
    st.subheader("📈 Finished Good Cost History")
    st.caption("Tracks how a finished good's material cost per unit has moved month to month, across all available history.")

    sel_fg = _item_picker("Select Finished Good", mo_header, "itemcode", "itemname", "mfg_fg_history_pick")
    if not sel_fg:
        return

    hist = mfg.compute_fg_cost_history(mo_cost, sel_fg)
    if hist.empty:
        st.info("No cost history found for this finished good.")
        return

    hist_disp = hist.copy()
    hist_disp["Period"] = hist_disp.apply(lambda r: f"{calendar.month_abbr[int(r['month'])]} {int(r['year'])}", axis=1)
    st.line_chart(hist_disp.set_index("Period")["avg_cost_per_unit"])

    col_map = {
        "Period": "Period", "total_qtyprd": "Qty Produced", "n_batches": "Batches",
        "total_material_cost": "Total Material Cost", "avg_cost_per_unit": "Avg Material Cost/Unit",
    }
    disp = hist_disp.rename(columns=col_map)[[c for c in col_map.values() if c in hist_disp.rename(columns=col_map).columns]]
    st.dataframe(
        _fmt(disp, {"Qty Produced": "{:,.0f}", "Batches": "{:,.0f}", "Total Material Cost": "{:,.0f}", "Avg Material Cost/Unit": "{:,.2f}"}),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "⬇ Download Cost History CSV", hist.to_csv(index=False).encode("utf-8"),
        file_name=f"fg_cost_history_{sel_fg}_{zid}.csv", mime="text/csv", key="dl_mfg_fg_history",
    )


# ── RM Rate Trend ────────────────────────────────────────────────────────────────

def _render_rm_rate_trend(zid: str, mo_lines: pd.DataFrame, mo_detail: pd.DataFrame):
    st.subheader("📉 Raw Material Rate Trend")
    st.caption("📝 Methodology — see Note [2] at the bottom of the page.")

    today = pd.Timestamp.today().normalize()
    n_months = st.slider("Price-mover window (trailing completed months)", 1, 24, 3, key="mfg_rm_rate_window")
    start, end = mfg.trailing_n_months_window(today, n_months)
    st.markdown(f"**Window:** {start.strftime('%b %Y')} – {end.strftime('%b %Y')}")

    st.markdown("**🔍 Rate history for one raw material**")
    sel_rm = _item_picker("Select Raw Material", mo_detail, "itemcode", "itemname", "mfg_rm_rate_pick")
    if sel_rm:
        detail, monthly = mfg.compute_rm_rate_trend(mo_lines, sel_rm)
        if monthly.empty:
            st.info("No rate history found for this raw material.")
        else:
            m = monthly.copy()
            m["Period"] = m.apply(lambda r: f"{calendar.month_abbr[int(r['month'])]} {int(r['year'])}", axis=1)
            st.line_chart(m.set_index("Period")["avg_rate"])
            st.dataframe(
                _fmt(m.rename(columns={"avg_rate": "Avg Rate", "total_qty": "Total Qty"})[["Period", "Avg Rate", "Total Qty"]],
                     {"Avg Rate": "{:,.2f}", "Total Qty": "{:,.2f}"}),
                width="stretch", hide_index=True,
            )
            st.download_button(
                "⬇ Download Rate History CSV", detail.to_csv(index=False).encode("utf-8"),
                file_name=f"rm_rate_history_{sel_rm}_{zid}.csv", mime="text/csv", key="dl_mfg_rm_rate",
            )

    st.markdown("---")
    st.markdown("**📊 Biggest price movers in the window**")
    movers = mfg.compute_rm_price_movers(mo_lines, start, end)
    if movers.empty:
        st.info("No raw materials with more than one BOM line in this window.")
        return
    mv_col_map = {
        "itemcode": "RM Code", "itemname": "RM Name", "first_date": "First Date", "first_rate": "First Rate",
        "last_date": "Last Date", "last_rate": "Last Rate", "pct_change": "% Change", "n_lines": "BOM Lines",
    }
    mvdisp = movers.rename(columns=mv_col_map)[[c for c in mv_col_map.values() if c in movers.rename(columns=mv_col_map).columns]]
    st.dataframe(
        _fmt(mvdisp, {"First Rate": "{:,.2f}", "Last Rate": "{:,.2f}", "% Change": "{:+,.1f}%", "BOM Lines": "{:,.0f}"}),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "⬇ Download Price Movers CSV", movers.to_csv(index=False).encode("utf-8"),
        file_name=f"rm_price_movers_{zid}.csv", mime="text/csv", key="dl_mfg_rm_movers",
    )


# ── RM Requirement ───────────────────────────────────────────────────────────────

def _render_rm_requirement(zid: str, mo_lines: pd.DataFrame):
    st.subheader("📦 Raw Material Requirement")
    st.caption(
        "Total raw-material quantity and value actually consumed across every completed MO for the "
        "whole business, over a window you choose — also doubles as a 'top RM by spend' ranking."
    )

    today = pd.Timestamp.today().normalize()
    n_months = st.slider("Requirement window (trailing completed months)", 1, 12, 1, key="mfg_rm_req_window")
    start, end = mfg.trailing_n_months_window(today, n_months)
    st.markdown(f"**Window:** {start.strftime('%b %Y')} – {end.strftime('%b %Y')}")

    req = mfg.compute_rm_requirement(mo_lines, start, end)
    if req.empty:
        st.info("No raw material consumption found in this window.")
        return

    sort_choice = st.radio("Sort by", ["Total Value (spend)", "Total Qty"], horizontal=True, key="mfg_rm_req_sort")
    req_sorted = req.sort_values("total_value" if sort_choice == "Total Value (spend)" else "total_qty", ascending=False)

    m1, m2 = st.columns(2)
    m1.metric("Total RM Spend (window)", f"{req['total_value'].sum():,.0f}")
    m2.metric("Distinct RMs Used", f"{len(req):,}")

    col_map = {
        "itemcode": "RM Code", "itemname": "RM Name", "itemgroup": "RM Group", "unit": "Unit",
        "total_qty": "Total Qty", "total_value": "Total Value", "n_lines": "BOM Lines",
    }
    disp = req_sorted.rename(columns=col_map)[[c for c in col_map.values() if c in req_sorted.rename(columns=col_map).columns]]
    st.dataframe(
        _fmt(disp, {"Total Qty": "{:,.2f}", "Total Value": "{:,.0f}", "BOM Lines": "{:,.0f}"}),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "⬇ Download RM Requirement CSV", req_sorted.to_csv(index=False).encode("utf-8"),
        file_name=f"rm_requirement_{zid}.csv", mime="text/csv", key="dl_mfg_rm_req",
    )


# ── RM Stock Coverage ────────────────────────────────────────────────────────────

def _render_rm_stock_coverage(zid: str, mo_header: pd.DataFrame, mo_lines: pd.DataFrame, stock_raw: pd.DataFrame):
    st.subheader("⚠️ Raw Material Stock Coverage")
    st.caption("📝 Methodology — see Note [3] at the bottom of the page.")

    today = pd.Timestamp.today().normalize()
    bom_start, bom_end = mfg.trailing_n_months_window(today, 3)
    sales_years = tuple(sorted({int(bom_start.year), int(bom_end.year)}))

    with st.spinner("Loading sales history…"):
        sales_3mo = _load_sales_window(str(zid), sales_years)
        returns_3mo = _load_returns_window(str(zid), sales_years)

    bom_ratio = mfg.compute_bom_ratio(mo_header, mo_lines, bom_start, bom_end)
    avg_sales = mfg.compute_avg_monthly_fg_sales(sales_3mo, returns_3mo, bom_start, bom_end)
    current_stock = mfg.compute_current_stock_from_imtrn(stock_raw)

    if bom_ratio.empty:
        st.info("No BOM/production data found in the last 3 completed months to derive RM requirements.")
        return

    coverage = mfg.compute_rm_stock_coverage(bom_ratio, avg_sales, current_stock, threshold_months=1.0)
    if coverage.empty:
        st.info("No raw materials found.")
        return

    n_short = int(coverage["is_short"].sum())
    m1, m2 = st.columns(2)
    m1.metric("Raw Materials Tracked", f"{len(coverage):,}")
    m2.metric("🔴 Below 1 Month Coverage", f"{n_short:,}")

    col_map = {
        "itemcode": "RM Code", "itemname": "RM Name", "projected_monthly_need": "Projected Monthly Need",
        "current_stock": "Current Stock", "coverage_months": "Coverage (Months)",
    }
    disp = coverage.rename(columns=col_map)[[c for c in col_map.values() if c in coverage.rename(columns=col_map).columns]]

    def _row_style(row):
        if row.get("Coverage (Months)") is not None and pd.notna(row.get("Coverage (Months)")) and row["Coverage (Months)"] < 1.0:
            return ["background-color: #F8D7DA; color: #721C24"] * len(row)
        return [""] * len(row)

    try:
        styled = disp.style.apply(_row_style, axis=1).format(
            {"Projected Monthly Need": "{:,.2f}", "Current Stock": "{:,.2f}", "Coverage (Months)": "{:,.2f}"},
            na_rep="—",
        )
        st.dataframe(styled, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(disp, width="stretch", hide_index=True)

    st.download_button(
        "⬇ Download Stock Coverage CSV", coverage.to_csv(index=False).encode("utf-8"),
        file_name=f"rm_stock_coverage_{zid}.csv", mime="text/csv", key="dl_mfg_rm_coverage",
    )


# ── BOM Variance / Wastage ───────────────────────────────────────────────────────

def _render_bom_variance(zid: str, mo_lines: pd.DataFrame):
    st.subheader("🔍 BOM Variance / Wastage")
    st.caption(
        "Compares actual raw-material qty issued (xqty) against the standard BOM qty (xqtyord) — "
        "raw materials running persistently above standard are highlighted as a wastage signal."
    )

    today = pd.Timestamp.today().normalize()
    n_months = st.slider("Variance window (trailing completed months)", 1, 12, 3, key="mfg_variance_window")
    start, end = mfg.trailing_n_months_window(today, n_months)
    st.markdown(f"**Window:** {start.strftime('%b %Y')} – {end.strftime('%b %Y')}")

    variance = mfg.compute_bom_variance(mo_lines, start, end)
    if variance.empty:
        st.info("No BOM lines found in this window.")
        return

    n_over = int(variance["over_consumption"].sum())
    m1, m2 = st.columns(2)
    m1.metric("Raw Materials Tracked", f"{len(variance):,}")
    m2.metric("🔴 Over Standard Consumption", f"{n_over:,}")

    col_map = {
        "itemcode": "RM Code", "itemname": "RM Name", "total_qty": "Actual Qty",
        "total_qtyord": "Standard Qty", "variance_qty": "Variance Qty", "variance_pct": "Variance %",
    }
    disp = variance.rename(columns=col_map)[[c for c in col_map.values() if c in variance.rename(columns=col_map).columns]]

    def _row_style(row):
        if row.get("Variance Qty", 0) > 0:
            return ["background-color: #FFF3CD; color: #856404"] * len(row)
        return [""] * len(row)

    try:
        styled = disp.style.apply(_row_style, axis=1).format(
            {"Actual Qty": "{:,.2f}", "Standard Qty": "{:,.2f}", "Variance Qty": "{:+,.2f}", "Variance %": "{:+,.1f}%"},
            na_rep="—",
        )
        st.dataframe(styled, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(disp, width="stretch", hide_index=True)

    st.download_button(
        "⬇ Download BOM Variance CSV", variance.to_csv(index=False).encode("utf-8"),
        file_name=f"bom_variance_{zid}.csv", mime="text/csv", key="dl_mfg_variance",
    )


# ── MO Detail ────────────────────────────────────────────────────────────────────

def _render_mo_detail(zid: str, mo_header: pd.DataFrame, mo_lines: pd.DataFrame):
    st.subheader("🔍 MO Detail")
    st.caption("Pick a finished good to see all MOs for it in the last 4 months, then drill into one MO's raw-material lines.")

    today = pd.Timestamp.today()
    month_start = pd.Timestamp(today.year, today.month, 1)
    window_start = month_start - pd.DateOffset(months=3)
    st.caption(f"Window: {window_start.strftime('%b %Y')} – {today.strftime('%b %Y')}")

    mh = mo_header.copy()
    mh["date"] = pd.to_datetime(mh["date"], errors="coerce")
    mh_window = mh[(mh["date"] >= window_start) & (mh["date"] <= today)].copy()

    if mh_window.empty:
        st.info("No completed MOs in the last 4 months.")
        return

    # Product picker
    sel_fg = _item_picker("Select Finished Good", mh_window, "itemcode", "itemname", "mfg_mo_detail_prod")
    if not sel_fg:
        return

    # MO picker for this product
    prod_mos = mh_window[mh_window["itemcode"] == sel_fg].sort_values("date", ascending=False)
    if prod_mos.empty:
        st.info("No MOs found for this product in the window.")
        return

    mo_labels = []
    mo_code_by_label = {}
    for _, r in prod_mos.iterrows():
        date_str = r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else "?"
        qty_str  = f"{float(r.get('qtyprd', 0)):,.0f}"
        lbl = f"{r['monumber']}  —  {date_str}  —  Qty: {qty_str}"
        mo_labels.append(lbl)
        mo_code_by_label[lbl] = str(r["monumber"])

    sel_mo_label = st.selectbox("Select Manufacturing Order", mo_labels, key="mfg_mo_detail_mo")
    sel_mo = mo_code_by_label.get(sel_mo_label)
    if not sel_mo:
        return

    mo_row = prod_mos[prod_mos["monumber"] == sel_mo].iloc[0]
    qty_produced = float(mo_row.get("qtyprd", 0) or 0)
    mo_date      = mo_row["date"]
    unit_fg      = str(mo_row.get("unit", "") or "")

    c1, c2, c3 = st.columns(3)
    c1.metric("MO Number", sel_mo)
    c2.metric("Date", mo_date.strftime("%Y-%m-%d") if pd.notna(mo_date) else "—")
    c3.metric("Qty Produced", f"{qty_produced:,.0f} {unit_fg}".strip())

    # RM lines for this MO (from mo_lines which already has line_cost)
    mo_lines_sel = mo_lines[mo_lines["monumber"] == sel_mo].copy()
    if mo_lines_sel.empty:
        st.info("No raw material lines found for this MO.")
        return

    mo_lines_sel["qty"]       = pd.to_numeric(mo_lines_sel["qty"],       errors="coerce").fillna(0.0)
    mo_lines_sel["qtyord"]    = pd.to_numeric(mo_lines_sel.get("qtyord"), errors="coerce").fillna(0.0)
    mo_lines_sel["rate"]      = pd.to_numeric(mo_lines_sel["rate"],       errors="coerce").fillna(0.0)
    mo_lines_sel["line_cost"] = pd.to_numeric(mo_lines_sel["line_cost"],  errors="coerce").fillna(0.0)

    total_material_cost = float(mo_lines_sel["line_cost"].sum())
    unit_cost = total_material_cost / qty_produced if qty_produced > 0 else 0.0

    cm1, cm2 = st.columns(2)
    cm1.metric("Total Material Cost", f"{total_material_cost:,.0f}")
    cm2.metric("Material Cost / Unit", f"{unit_cost:,.2f}")

    col_map = {
        "itemcode_rm": "RM Code",
        "itemname_rm": "RM Name",
        "unit_rm":     "Unit",
        "qty":         "Qty Issued",
        "qtyord":      "Standard Qty",
        "rate":        "Rate",
        "line_cost":   "Line Cost",
    }
    disp_cols = [c for c in col_map if c in mo_lines_sel.columns]
    disp = mo_lines_sel[disp_cols].rename(columns=col_map).reset_index(drop=True)
    st.dataframe(
        _fmt(disp, {
            "Qty Issued":   "{:,.3f}",
            "Standard Qty": "{:,.3f}",
            "Rate":         "{:,.2f}",
            "Line Cost":    "{:,.0f}",
        }),
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "⬇ Download MO Detail CSV",
        mo_lines_sel.to_csv(index=False).encode("utf-8"),
        file_name=f"mo_detail_{sel_mo}_{zid}.csv",
        mime="text/csv",
        key="dl_mfg_mo_detail",
    )


# ── Warehouse Flow ───────────────────────────────────────────────────────────────

def _render_warehouse_flow(zid: str):
    st.subheader("🔄 Warehouse Flow")
    st.caption(
        "Per-product flow of goods: Raw Material → (MO) → Finished Goods warehouse → (transfer) → "
        "Sales Store → (DO) → market. Raw Material warehouse itself is excluded from the flow "
        "table(s) below (per what was asked) but its BDT value still shows alongside Finished Goods "
        "value in each mode."
    )

    with st.spinner("Loading warehouse movement history…"):
        flow_raw = _load_manufacturing_flow(str(zid))

    if flow_raw.empty:
        st.info("No warehouse movement data found for this business.")
        return

    groups = mfg.WAREHOUSE_GROUPS.get(str(zid), {})
    same_wh = set(groups.get("fg", [])) == set(groups.get("sales", []))
    if same_wh:
        st.info(
            "This business has no separate Sales Store warehouse — Delivery Orders are issued "
            "directly out of the Finished Goods warehouse, so Transferred is always 0 and both "
            "warehouses' opening/closing figures below are identical by construction, not an error."
        )

    flow_mode = st.radio(
        "Mode",
        ["📅 Choice Timeline", "📦 7-Day Stock Target"],
        horizontal=True, key="mfg_flow_mode",
    )
    if flow_mode == "📅 Choice Timeline":
        _render_flow_choice_timeline(zid, flow_raw)
    else:
        _render_flow_seven_day_target(zid, flow_raw)


def _render_flow_choice_timeline(zid: str, flow_raw: pd.DataFrame):
    today = pd.Timestamp.today().normalize()
    default_start = today.replace(day=1)
    date_range = st.date_input(
        "Timeline", value=(default_start.date(), today.date()), key="mfg_flow_range",
    )
    if not (isinstance(date_range, tuple) and len(date_range) == 2):
        st.info("Pick a full date range (start and end).")
        return
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    if start > end:
        st.warning("Start date must be on or before end date.")
        return
    st.caption(f"Window: **{start.date()}** to **{end.date()}**.")

    r = mfg.compute_warehouse_flow(flow_raw, str(zid), start, end)

    st.markdown("**Finished Goods Warehouse → Sales Store flow — per product**")
    flow_by_product = mfg.compute_warehouse_flow_by_product(flow_raw, str(zid), start, end)
    if flow_by_product.empty:
        st.info("No product movement found in this window.")
    else:
        col_map = {
            "itemcode": "Item Code", "itemname": "Item Name", "itemgroup": "Item Group",
            "fg_opening_qty": "FG Opening", "fg_mo_added_qty": "FG: MO Added",
            "fg_transferred_out_qty": "FG: Transferred Out", "fg_other_qty": "FG: Other",
            "fg_closing_qty": "FG Closing",
            "sales_opening_qty": "Sales Opening", "sales_transferred_in_qty": "Sales: Transferred In",
            "sales_returns_qty": "Sales: Returns", "sales_do_sold_qty": "Sales: Sold (DO)",
            "sales_other_qty": "Sales: Other", "sales_closing_qty": "Sales Closing",
        }
        flow_disp = flow_by_product.rename(columns=col_map)[list(col_map.values())]

        search = st.text_input("Search item code or name", "", key="mfg_flow_search")
        if search:
            mask = (
                flow_disp["Item Code"].str.contains(search, case=False, na=False)
                | flow_disp["Item Name"].str.contains(search, case=False, na=False)
            )
            flow_disp = flow_disp[mask]

        qty_cols = [c for c in col_map.values() if c not in ("Item Code", "Item Name", "Item Group")]
        st.caption(f"{len(flow_disp):,} product(s) with any opening balance or activity in this window.")

        # TOTAL row -- summed directly from the (possibly search-filtered)
        # displayed rows, so it can never silently disagree with what's shown.
        if not flow_disp.empty:
            total_row = {c: "" for c in flow_disp.columns}
            total_row["Item Code"] = "─── TOTAL ───"
            for c in qty_cols:
                total_row[c] = flow_disp[c].sum()
            flow_with_total = pd.concat([flow_disp, pd.DataFrame([total_row])], ignore_index=True)
        else:
            flow_with_total = flow_disp

        st.dataframe(
            _fmt(flow_with_total, {c: "{:,.2f}" for c in qty_cols}),
            width="stretch", hide_index=True,
        )
        st.caption(
            "**Transferred Out** (FG side) and **Transferred In** (Sales side) are measured "
            "independently, not derived from one another — the two legs of a transfer voucher don't "
            "always post within the same window (confirmed on real data: a 3-month window where "
            "473,174 units left the FG side but only 238,572 had arrived at Sales by the window's end "
            "across the whole business — a real timing lag, not a data error). **Sales: Returns** "
            "(`SR--`) is broken out on its own — confirmed against real Postgres this doctype is "
            "always a positive (inventory-increasing) movement across all three entities, i.e. "
            "genuinely a return-driven stock increase. **Other** covers every remaining doctype (e.g. "
            "issues, adjustments) — included so Opening + inflows − outflows + Other reconciles "
            "exactly to Closing, verified per-item against real Postgres with zero residual for all "
            "three entities."
        )
        st.download_button(
            "⬇ Download Warehouse Flow CSV",
            flow_disp.to_csv(index=False).encode("utf-8"),
            file_name=f"warehouse_flow_{zid}_{start.date()}_{end.date()}.csv",
            mime="text/csv",
            key="dl_mfg_warehouse_flow",
        )

    st.markdown("---")
    st.markdown("**💰 Inventory Value (BDT)**")
    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("Raw Material")
        st.metric("Start of Period", f"{r['rm_value_start']:,.0f}")
        st.metric("End of Period", f"{r['rm_value_end']:,.0f}")
    with v2:
        st.markdown("Finished Goods Warehouse")
        st.metric("Start of Period", f"{r['fg_value_start']:,.0f}")
        st.metric("End of Period", f"{r['fg_value_end']:,.0f}")
    with v3:
        st.markdown("Sales Store")
        st.metric("Start of Period", f"{r['sales_value_start']:,.0f}")
        st.metric("End of Period", f"{r['sales_value_end']:,.0f}")

    st.metric("Total Sold in Period (at cost)", f"{r['total_sold_value']:,.0f}")
    st.caption(
        "All values are inventory-cost basis (`imtrn.xval`), not sales revenue — consistent with "
        "how the Raw Material / Finished Goods values above are computed. 'Total Sold' is the COGS "
        "value of everything that left via DO in this window, not the amount billed to customers."
    )


def _render_flow_seven_day_target(zid: str, flow_raw: pd.DataFrame):
    st.caption(
        "Trailing 3 months, split into non-overlapping 7-day segments (walking backward from "
        "today); each metric is averaged across those segments — a 'typical week' for this product. "
        "**Stock Target (Qty)** = average 7-day Sold (DO) — the amount that should sit in the Sales "
        "Store at all times to cover one week of average demand."
    )

    prod, summary = mfg.compute_seven_day_stock_target(flow_raw, str(zid), months_back=3)
    if prod.empty:
        st.info("Not enough history (need at least 7 days in the trailing 3 months) to compute this.")
        return

    st.caption(
        f"Window: **{summary['window_start'].date()}** to **{summary['window_end'].date()}** "
        f"({summary['n_segments']} × 7-day segments)."
    )

    col_map = {
        "itemcode": "Item Code", "itemname": "Item Name", "itemgroup": "Item Group",
        "fg_opening_qty_avg": "FG Opening (avg/7d)", "fg_mo_added_qty_avg": "FG: MO Added (avg/7d)",
        "sales_returns_qty_avg": "Sales: Returns (avg/7d)", "sales_do_sold_qty_avg": "Sales: Sold DO (avg/7d)",
        "sales_other_qty_avg": "Sales: Other (avg/7d)", "sales_closing_qty_avg": "Sales Closing (avg/7d)",
        "stock_target_qty": "Stock Target (Qty)", "unit_cost": "Est. Unit Cost", "target_value": "Target Value (BDT)",
    }
    disp = prod.rename(columns=col_map)[list(col_map.values())]

    search = st.text_input("Search item code or name", "", key="mfg_7d_search")
    if search:
        mask = (
            disp["Item Code"].str.contains(search, case=False, na=False)
            | disp["Item Name"].str.contains(search, case=False, na=False)
        )
        disp = disp[mask]

    qty_cols = [c for c in col_map.values() if c not in ("Item Code", "Item Name", "Item Group")]
    st.caption(f"{len(disp):,} product(s) with any activity in the window.")

    if not disp.empty:
        total_row = {c: "" for c in disp.columns}
        total_row["Item Code"] = "─── TOTAL ───"
        for c in qty_cols:
            if c not in ("Est. Unit Cost",):  # a summed "average unit cost" isn't meaningful
                total_row[c] = disp[c].sum(skipna=True)
        disp_with_total = pd.concat([disp, pd.DataFrame([total_row])], ignore_index=True)
    else:
        disp_with_total = disp

    fmt = {c: "{:,.2f}" for c in qty_cols}
    fmt["Est. Unit Cost"] = "{:,.2f}"
    fmt["Target Value (BDT)"] = "{:,.0f}"
    st.dataframe(_fmt(disp_with_total, fmt), width="stretch", hide_index=True)
    n_no_cost = summary["n_items_no_cost_basis"]
    if n_no_cost:
        st.caption(
            f"⚠️ {n_no_cost:,} product(s) had no MO receipt in this window, so no cost basis is "
            f"available — they still show a real Stock Target (Qty) above, just no BDT value."
        )
    st.download_button(
        "⬇ Download 7-Day Stock Target CSV",
        disp.to_csv(index=False).encode("utf-8"),
        file_name=f"seven_day_stock_target_{zid}.csv",
        mime="text/csv",
        key="dl_mfg_7d_target",
    )

    st.markdown("---")
    st.markdown("**💰 Current vs. Target (BDT, 7-day basis)**")
    st.caption(
        "The 'New/Target' figure is the SAME number in both comparisons below — one week of average "
        "demand, valued at each product's own MO receipt cost, summed across every product with a "
        "cost basis. It answers two different questions from the same target: how much should be "
        "*produced* per week to keep up with demand, and how much should be *standing in the Sales "
        "Store* at any given moment as a buffer — under a constant-7-day-buffer assumption, those are "
        "the same quantity."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Production (MO) value, per 7 days**")
        st.metric("Current (avg actual)", f"{summary['current_mo_value_7d']:,.0f}")
        st.metric("New (target)", f"{summary['new_mo_value_7d']:,.0f}")
        st.metric("Difference", f"{summary['new_mo_value_7d'] - summary['current_mo_value_7d']:+,.0f}")
    with c2:
        st.markdown("**Sales Store FG value, standing**")
        st.metric("Current (avg actual)", f"{summary['current_fg_value_7d']:,.0f}")
        st.metric("New (target)", f"{summary['new_fg_value_7d']:,.0f}")
        st.metric("Difference", f"{summary['new_fg_value_7d'] - summary['current_fg_value_7d']:+,.0f}")
    st.caption(
        "All values are inventory-cost basis (`imtrn.xval`), not sales revenue, consistent with the "
        "rest of this page. 'Current' is the average actually observed per 7-day segment over the "
        "trailing 3 months; 'New/Target' is the steady-state figure needed to always hold 7 days of "
        "stock. A positive Difference means the target is higher than what's currently happening."
    )


# ── Main entry point ─────────────────────────────────────────────────────────────

@timed
def display_manufacturing_analysis_page(current_page, zid: str):
    st.title("🏭 Manufacturing Analysis")

    view_mode = st.radio(
        "View",
        ["🏭 FG Costing", "📈 FG Cost History", "📉 RM Rate Trend", "📦 RM Requirement",
         "⚠️ RM Stock Coverage", "🔍 BOM Variance / Wastage", "📋 MO Detail", "🔄 Warehouse Flow"],
        horizontal=True, key="mfg_view_mode",
    )

    if str(zid) not in _MANUFACTURING_ZIDS:
        st.warning(
            "Manufacturing Analysis is only available for GI Corporation (100000), "
            "Zepto Chemicals (100005), and Gulshan Packaging (100009). "
            "Switch ZID in the sidebar to use this page."
        )
        return

    if view_mode == "🔄 Warehouse Flow":
        # Independent of MO header/detail (only reads warehouse movement
        # history), so this runs before the MO-data early-return below.
        _render_warehouse_flow(zid)
        return

    with st.spinner("Loading manufacturing order history…"):
        mo_header = _load_mo_header(str(zid))
        mo_detail = _load_mo_detail(str(zid))
        admin_expense = _load_admin_expense(str(zid))
        stock_raw = _load_stock_raw(str(zid))

    if mo_header.empty or mo_detail.empty:
        st.info("No completed manufacturing orders found for this business.")
        return

    mo_lines = mfg.merge_mo_lines(mo_header, mo_detail)
    mo_cost = mfg.compute_mo_cost(mo_lines)
    opspprc_df = _load_opspprc(str(zid))

    if view_mode == "🏭 FG Costing":
        _render_fg_costing(zid, mo_cost, mo_lines, admin_expense, opspprc_df)
    elif view_mode == "📈 FG Cost History":
        _render_fg_cost_history(zid, mo_cost, mo_header)
    elif view_mode == "📉 RM Rate Trend":
        _render_rm_rate_trend(zid, mo_lines, mo_detail)
    elif view_mode == "📦 RM Requirement":
        _render_rm_requirement(zid, mo_lines)
    elif view_mode == "⚠️ RM Stock Coverage":
        _render_rm_stock_coverage(zid, mo_header, mo_lines, stock_raw)
    elif view_mode == "🔍 BOM Variance / Wastage":
        _render_bom_variance(zid, mo_lines)
    elif view_mode == "📋 MO Detail":
        _render_mo_detail(zid, mo_header, mo_lines)

    st.markdown("---")
    st.subheader("📝 Notes")
    st.markdown(
        "**[1] FG Costing** — Material cost per unit comes from actual BOM consumption "
        "(qty × rate) on completed MOs only, divided by qty produced; there is no labor/overhead "
        "in moord/moodt. Admin (06) cost per unit allocates each month's GL '06' (Office & "
        "Administrative) expense across every finished good produced that month, by that FG's "
        "share of the zid's total material production cost, then divides by that FG's qty "
        "produced that month — e.g. if 06 expense is 100 for the month and this FG's material "
        "cost is 5 of the zid's 50 total, it gets 10 allocated, spread across its units produced."
    )
    st.markdown(
        "**[2] RM Rate Trend** — Price movers compare each raw material's first vs last BOM-line "
        "rate within the selected window (chronological), ranked by absolute % change. Raw "
        "materials with only one BOM line in the window are excluded — there's no real 'change' to report."
    )
    st.markdown(
        "**[3] RM Stock Coverage** — For each raw material: projected monthly need = sum across "
        "every finished good that uses it of (that FG's average monthly net sales qty over the "
        "trailing 3 months × the RM qty actually required per unit of that FG, from the last 3 "
        "months of BOM history) — compared against current stock (summed from imtrn's full "
        "movement history, not final_items_view, which was found to under-report stock for some "
        "of these items). Flagged red when coverage is below 1 month of projected need."
    )
