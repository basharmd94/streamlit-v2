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

def _render_fg_costing(zid: str, mo_cost: pd.DataFrame, mo_lines: pd.DataFrame, admin_expense: pd.DataFrame):
    st.subheader("🏭 Finished Good Costing")
    st.caption("📝 Methodology — see Note [1] at the bottom of the page.")

    today = pd.Timestamp.today().normalize()
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
        use_container_width=True, hide_index=True,
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

    m1, m2, m3 = st.columns(3)
    m1.metric("Material Cost/Unit (window avg)", f"{fg_row['avg_cost_per_unit']:,.2f}")
    m2.metric("Admin (06) Cost/Unit (allocated)", f"{alloc_avg_admin:,.2f}")
    m3.metric("Total Cost/Unit", f"{total_cost_per_unit:,.2f}")

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
            use_container_width=True, hide_index=True,
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
            use_container_width=True, hide_index=True,
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
        use_container_width=True, hide_index=True,
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
                use_container_width=True, hide_index=True,
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
        use_container_width=True, hide_index=True,
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
        use_container_width=True, hide_index=True,
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
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)

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
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(disp, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download BOM Variance CSV", variance.to_csv(index=False).encode("utf-8"),
        file_name=f"bom_variance_{zid}.csv", mime="text/csv", key="dl_mfg_variance",
    )


# ── Main entry point ─────────────────────────────────────────────────────────────

@timed
def display_manufacturing_analysis_page(current_page, zid: str):
    st.title("🏭 Manufacturing Analysis")

    if str(zid) not in _MANUFACTURING_ZIDS:
        st.warning(
            "Manufacturing Analysis is only available for GI Corporation (100000), "
            "Zepto Chemicals (100005), and Gulshan Packaging (100009). "
            "Switch ZID in the sidebar to use this page."
        )
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

    view_mode = st.radio(
        "View",
        ["🏭 FG Costing", "📈 FG Cost History", "📉 RM Rate Trend", "📦 RM Requirement",
         "⚠️ RM Stock Coverage", "🔍 BOM Variance / Wastage"],
        horizontal=True, key="mfg_view_mode",
    )

    if view_mode == "🏭 FG Costing":
        _render_fg_costing(zid, mo_cost, mo_lines, admin_expense)
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
