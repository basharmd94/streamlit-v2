from __future__ import annotations

import calendar
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from processing import (
    common, target_management as tm, buying_pattern as bp, daily_sales as ds,
    salesman_score as ssc,
)
from utils.utils import timed

from views._tm_shared import (
    _get_holidays, _prune_targets, _prune_holidays, _toggle_holiday,
    _count_working_days,
    _get_target, _save_target,
    _current_month_label,
    _sp_opts, _cus_opts, _item_opts, _codes, _filter_code,
    _format_unquoted_dict,
    _render_table, _render_not_ordered_table,
    _load_opmob_pending, _load_opmob_all,
    _load_final_items, _load_opspprc, _load_cacus_directory,
    _render_inventory_coverage,
    _render_buying_pattern,
)
from views.salesman_score import _render_salesman_score
from views.next_month_target import _render_next_month_target
from views.field_tracking import _render_field_tracking
from views.glpmt_shared import render_glpmt_panel as _render_glpmt_panel
from views.returns_registry import _render_returns_registry


# ── Metric cards ───────────────────────────────────────────────────────────────

def _render_metric_cards(
    sp_sales: pd.DataFrame,
    opmob_df: pd.DataFrame,
    sel_spid: str,
    zid,
    sp_returns: pd.DataFrame = None,
    collection_df: pd.DataFrame = None,
):
    """Render the performance metric cards for the selected salesman."""
    today = pd.Timestamp.today().normalize()
    holidays = _get_holidays()

    # ── Base data prep ────────────────────────────────────────────────────────
    sp = sp_sales.copy()
    has_date = "date" in sp.columns
    if has_date:
        sp["_dt"] = pd.to_datetime(sp["date"], errors="coerce")
        sp["_d"]  = sp["_dt"].dt.date

    # ── Last 3 complete months ────────────────────────────────────────────────
    mo_start_cur = pd.Timestamp(today.year, today.month, 1)
    mo_start_3mo = mo_start_cur - pd.DateOffset(months=3)
    end_3mo      = mo_start_cur - pd.Timedelta(days=1)

    total_3mo = 0.0
    daily_avg_3mo = 0.0
    wd_3mo = 0
    last3 = pd.DataFrame()
    if has_date:
        last3 = sp[(sp["_dt"] >= mo_start_3mo) & (sp["_dt"] <= end_3mo)].copy()
        total_3mo = float(last3["final_sales"].sum())
        wd_3mo = _count_working_days(mo_start_3mo.date(), end_3mo.date(), holidays)
        daily_avg_3mo = total_3mo / wd_3mo if wd_3mo > 0 else 0.0

    monthly_avg_3mo = total_3mo / 3

    # ── MTD sales + returns ───────────────────────────────────────────────────
    mtd_df = pd.DataFrame()
    mtd_sales = 0.0
    mtd_unique_products = 0
    mtd_unique_customers = 0
    if has_date:
        mtd_df = sp[sp["_dt"] >= mo_start_cur]
        mtd_sales = float(mtd_df["final_sales"].sum())
        mtd_unique_products = int(mtd_df["itemcode"].nunique()) if "itemcode" in mtd_df.columns else 0
        mtd_unique_customers = int(mtd_df["cusid"].nunique()) if "cusid" in mtd_df.columns else 0

    mtd_return = 0.0
    if sp_returns is not None and not sp_returns.empty and "treturnamt" in sp_returns.columns:
        _r = sp_returns.copy()
        _r["_dt"] = pd.to_datetime(_r["date"], errors="coerce")
        mtd_return = float(_r[_r["_dt"] >= mo_start_cur]["treturnamt"].sum())

    net_sales = mtd_sales - mtd_return

    # ── MTD collection for this salesman, current calendar month ─────────────
    mtd_collection = 0.0
    if (
        collection_df is not None and not collection_df.empty
        and "value" in collection_df.columns and "spid" in collection_df.columns
    ):
        _c = collection_df.copy()
        _c["spid"]  = _c["spid"].astype(str)
        _c["year"]  = pd.to_numeric(_c["year"],  errors="coerce")
        _c["month"] = pd.to_numeric(_c["month"], errors="coerce")
        _c_mtd = _c[
            (_c["spid"] == str(sel_spid)) & (_c["year"] == today.year) & (_c["month"] == today.month)
        ]
        mtd_collection = float(_c_mtd["value"].sum())

    import calendar as _cal
    last_day_num = _cal.monthrange(today.year, today.month)[1]
    month_end    = date(today.year, today.month, last_day_num)
    tomorrow     = (today + pd.Timedelta(days=1)).date()
    remaining_wd = _count_working_days(tomorrow, month_end, holidays) if tomorrow <= month_end else 0

    # ── Month options for target selector ─────────────────────────────────────
    # Show full current year: Jan <current_year> → Dec <current_year>
    month_options = []
    current_year = today.year
    for m in range(1, 13):
        label = f"{calendar.month_abbr[m]} {current_year}"
        month_options.append((label, current_year, m))

    # ── Layout ────────────────────────────────────────────────────────────────
    st.markdown("---")

    # ── Target controls — fully horizontal ───────────────────────────────────
    st.markdown("**🎯 Monthly Target**")
    t_cols = st.columns([1.5, 1.5, 0.7, 1.3, 1.3, 1.3, 1.5])

    with t_cols[0]:
        _default_mo_idx = next(
            (i for i, m in enumerate(month_options) if m[1] == today.year and m[2] == today.month),
            0,
        )
        sel_mo_label = st.selectbox(
            "Month",
            [m[0] for m in month_options],
            index=_default_mo_idx,
            key=f"tm_target_month_{sel_spid}",
        )
    sel_mo_year, sel_mo_month = next(
        (m[1], m[2]) for m in month_options if m[0] == sel_mo_label
    )
    is_current_month = (sel_mo_year == today.year and sel_mo_month == today.month)
    saved_target = _get_target(zid, sel_spid, sel_mo_year, sel_mo_month)

    with t_cols[1]:
        target_val = st.number_input(
            "Target",
            min_value=0.0,
            value=float(saved_target) if saved_target is not None else 0.0,
            step=1000.0,
            format="%.0f",
            key=f"tm_target_{sel_spid}_{sel_mo_year}_{sel_mo_month}",
        )

    with t_cols[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save", key=f"tm_save_{sel_spid}"):
            _save_target(zid, sel_spid, sel_mo_year, sel_mo_month, target_val)
            st.toast(f"Target saved for {sel_mo_label}!", icon="✅")

    if is_current_month:
        with t_cols[3]:
            st.metric("MTD Sales", f"{mtd_sales:,.0f}")
        with t_cols[4]:
            st.metric("MTD Return", f"{mtd_return:,.0f}")
        with t_cols[5]:
            st.metric("Net Sales", f"{net_sales:,.0f}")
        with t_cols[6]:
            if target_val > 0:
                gap = target_val - mtd_sales
                if remaining_wd > 0:
                    daily_req = gap / remaining_wd
                    icon = (
                        "🔴" if daily_req > daily_avg_3mo * 1.2
                        else "🟡" if daily_req > daily_avg_3mo
                        else "🟢"
                    )
                    st.metric(
                        "Daily Required",
                        f"{icon} {daily_req:,.0f}",
                        delta=f"{remaining_wd} days left",
                        delta_color="off",
                    )
                else:
                    pct = ((mtd_sales - mtd_return) - target_val) / target_val * 100
                    label = "Above target 🟢" if pct >= 0 else "Below target 🔴"
                    st.metric("vs Target", f"{pct:+.1f}%", delta=label, delta_color="off")
    else:
        with t_cols[3]:
            st.caption("MTD & daily required shown for current month only.")

    # ── Row 2: MTD performance — collection, activity, % vs target, 3M avg ───
    if is_current_month:
        pct_mtd_vs_target = round(net_sales / target_val * 100, 1) if target_val > 0 else None
        pct_coll_vs_target = round(mtd_collection / target_val * 100, 1) if target_val > 0 else None

        p_cols = st.columns(6)
        with p_cols[0]:
            st.metric("💰 Collection (MTD)", f"{mtd_collection:,.0f}")
        with p_cols[1]:
            st.metric("📦 Products Sold (MTD)", f"{mtd_unique_products:,}")
        with p_cols[2]:
            st.metric("👥 Customers Visited (MTD)", f"{mtd_unique_customers:,}")
        with p_cols[3]:
            st.metric("🎯 % MTD Sales vs Target", f"{pct_mtd_vs_target:.1f}%" if pct_mtd_vs_target is not None else "—")
        with p_cols[4]:
            st.metric("💧 % Collection vs Target", f"{pct_coll_vs_target:.1f}%" if pct_coll_vs_target is not None else "—")
        with p_cols[5]:
            st.metric("📈 Monthly Avg Sales (3M)", f"{monthly_avg_3mo:,.0f}", delta="last 3 months", delta_color="off")
        st.caption(
            "ℹ️ **% Collection vs Target** = MTD Collection ÷ Monthly Target "
            "(previously MTD Collection ÷ (MTD Sales × 1.02)). "
            "All Salesmen Overview retains the old formula (÷ MTD Sales × 1.02)."
        )

    st.markdown("---")


# ── All-Salesmen Overview ──────────────────────────────────────────────────────

def _render_overview(sales_df: pd.DataFrame, returns_df: pd.DataFrame, opmob_all: pd.DataFrame, zid, collection_df: pd.DataFrame = None):
    """
    Two-table overview for the currently selected ZID:
      Table 1 — one row per salesman: target/MTD/daily-required/3-month metrics
      Table 2 — one row per salesman × date × area for the current month:
                 sales, unique customers, unique products, pending opmob total
    """
    today        = pd.Timestamp.today().normalize()
    holidays     = _get_holidays()
    cur_year     = today.year
    cur_month    = today.month
    mo_start_cur = pd.Timestamp(cur_year, cur_month, 1)
    mo_start_3mo = mo_start_cur - pd.DateOffset(months=3)
    end_3mo      = mo_start_cur - pd.Timedelta(days=1)
    month_end    = pd.Timestamp(cur_year, cur_month,
                                calendar.monthrange(cur_year, cur_month)[1])

    remaining_wd   = _count_working_days(today.date(), month_end.date(), holidays)
    wd_elapsed     = max(1, _count_working_days(mo_start_cur.date(), today.date(), holidays))
    total_wd_month = _count_working_days(mo_start_cur.date(), month_end.date(), holidays)

    if "date" not in sales_df.columns or "final_sales" not in sales_df.columns:
        st.warning("Required columns missing.")
        return

    df = sales_df.copy()
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["_d"]  = df["_dt"].dt.date

    last3   = df[(df["_dt"] >= mo_start_3mo) & (df["_dt"] <= end_3mo)]
    mtd_all = df[(df["_dt"] >= mo_start_cur) & (df["_dt"] <= today)]

    # Warn if historical data is missing — Monthly Avg (3M) will show as 0 without it
    if last3.empty:
        st.warning(
            f"⚠️ No data found for the 3-month lookback window "
            f"({mo_start_3mo.strftime('%b %Y')} – {end_3mo.strftime('%b %Y')}). "
            "**Monthly Avg (3M)** will show 0. "
            "Please load at least 3 prior months in the sidebar filters."
        )

    # ── MTD returns per salesman ───────────────────────────────────────────────
    mtd_ret_by_sp: dict = {}   # spid -> mtd treturnamt
    if returns_df is not None and not returns_df.empty and "treturnamt" in returns_df.columns:
        _r = returns_df.copy()
        _r["_dt"] = pd.to_datetime(_r["date"], errors="coerce")
        _r_mtd = _r[(_r["_dt"] >= mo_start_cur) & (_r["_dt"] <= today)]
        if "spid" in _r_mtd.columns:
            mtd_ret_by_sp = _r_mtd.groupby(_r_mtd["spid"].astype(str))["treturnamt"].sum().astype(float).to_dict()

    # ── Collection per salesman × year × month ────────────────────────────────
    coll_by_sp: dict = {}  # (spid, year, month) -> collection value
    if collection_df is not None and not collection_df.empty and "value" in collection_df.columns:
        _c = collection_df.copy()
        _c["spid"]  = _c["spid"].astype(str)
        _c["year"]  = pd.to_numeric(_c["year"],  errors="coerce")
        _c["month"] = pd.to_numeric(_c["month"], errors="coerce")
        coll_by_sp = _c.groupby(["spid", "year", "month"])["value"].sum().astype(float).to_dict()

    # ── opmob pending per salesman × area ────────────────────────────────────
    pend_sp_area: dict = {}   # (spid, area) -> total pending
    if not opmob_all.empty:
        cusid_area = (
            df[["cusid", "area"]].dropna().drop_duplicates("cusid")
            .set_index("cusid")["area"].to_dict()
        )
        ob = opmob_all.copy()
        ob["area"] = ob["cusid"].astype(str).map(cusid_area)
        if "spid" in ob.columns and "linetotal" in ob.columns:
            for (sp, ar), grp in ob.dropna(subset=["area"]).groupby(["spid", "area"]):
                pend_sp_area[(str(sp), str(ar))] = float(grp["linetotal"].sum())

    # ── Table 1: salesman summary ─────────────────────────────────────────────
    st.subheader("📋 Salesman Summary — Current Month")
    sp_list = (
        df[["spid", "spname"]].dropna().drop_duplicates()
        .sort_values("spname")
    )
    rows1 = []
    for _, sp_row in sp_list.iterrows():
        spid   = str(sp_row["spid"])
        spname = sp_row["spname"]

        sp3   = last3[last3["spid"].astype(str) == spid]
        sp_mtd = mtd_all[mtd_all["spid"].astype(str) == spid]

        total_3mo   = float(sp3["final_sales"].sum())
        monthly_avg = round(total_3mo / 3, 0)
        mtd_sales   = float(sp_mtd["final_sales"].sum())

        mtd_ret   = round(mtd_ret_by_sp.get(spid, 0.0), 0)
        net_sales = round(mtd_sales - mtd_ret, 0)

        target    = _get_target(zid, spid, cur_year, cur_month) or 0.0
        gap       = target - mtd_sales
        daily_req = round(gap / remaining_wd, 0) if remaining_wd > 0 and target > 0 else 0.0
        pct_tgt   = round(net_sales / target * 100, 1) if target > 0 else None

        mtd_coll  = round(float(coll_by_sp.get((spid, cur_year, cur_month), 0.0)), 0)
        pct_coll  = round(mtd_coll / (1.02 * mtd_sales) * 100, 1) if mtd_sales > 0 else None

        mtd_up = int(sp_mtd["itemcode"].nunique()) if "itemcode" in sp_mtd.columns else 0
        mtd_uc = int(sp_mtd["cusid"].nunique())    if "cusid"    in sp_mtd.columns else 0

        rows1.append({
            "Salesman":                 spname,
            "Target":                   target,
            "MTD Sales":                round(mtd_sales, 0),
            "MTD Return":               mtd_ret,
            "Net Sales":                net_sales,
            "% vs Target":              pct_tgt,
            "MTD Collection":           mtd_coll,
            "% Collection":             pct_coll,
            "Products Sold (MTD)":      mtd_up,
            "Customers Visited (MTD)":  mtd_uc,
            "Days Left":                remaining_wd,
            "Daily Required":           daily_req,
            "Monthly Avg (3M)":         monthly_avg,
            "Daily Avg (MTD)":          round(mtd_sales / wd_elapsed, 0),
            "Predicted Total":          round((mtd_sales / wd_elapsed) * total_wd_month, 0),
            "Predicted vs Target":      round((mtd_sales / wd_elapsed) * total_wd_month - target, 0) if target > 0 else None,
            "Collection Gap":           round(target * 1.02 - mtd_coll, 0) if target > 0 else None,
        })

    if rows1:
        t1 = pd.DataFrame(rows1).sort_values("MTD Sales", ascending=False).reset_index(drop=True)

        def _style_t1(df):
            styled = df.style.format({
                "Target":                   "{:,.0f}",
                "MTD Sales":                "{:,.0f}",
                "MTD Return":               "{:,.0f}",
                "Net Sales":                "{:,.0f}",
                "% vs Target":              lambda v: f"{v:.1f}%" if v is not None else "—",
                "MTD Collection":           "{:,.0f}",
                "% Collection":             lambda v: f"{v:.1f}%" if v is not None else "—",
                "Products Sold (MTD)":      "{:,.0f}",
                "Customers Visited (MTD)":  "{:,.0f}",
                "Days Left":                "{:,.0f}",
                "Daily Required":           "{:,.0f}",
                "Monthly Avg (3M)":         "{:,.0f}",
                "Daily Avg (MTD)":          "{:,.0f}",
                "Predicted Total":          "{:,.0f}",
                "Predicted vs Target":      lambda v: f"{v:+,.0f}" if v is not None else "—",
                "Collection Gap":           lambda v: f"{v:,.0f}" if v is not None else "—",
            }, na_rep="—")

            def _col_pct(col):
                out = []
                for v in col:
                    if v is None:
                        out.append("")
                    elif v >= 100:
                        out.append("background-color: #D4EDDA; color: #155724")
                    elif v >= 75:
                        out.append("background-color: #FFF3CD; color: #856404")
                    else:
                        out.append("background-color: #F8D7DA; color: #721C24")
                return out

            if "% vs Target" in df.columns:
                styled = styled.apply(_col_pct, subset=["% vs Target"])
            if "% Collection" in df.columns:
                styled = styled.apply(_col_pct, subset=["% Collection"])
            return styled

        try:
            st.dataframe(_style_t1(t1), width="stretch", hide_index=True)
        except Exception:
            st.dataframe(t1, width="stretch", hide_index=True)

        _3m_period = f"{mo_start_3mo.strftime('%b %Y')} – {end_3mo.strftime('%b %Y')}"
        st.caption(
            f"**Monthly Avg (3M)** = total sales in prior 3 months ÷ 3 &nbsp;|&nbsp; "
            f"3M window: {_3m_period} &nbsp;|&nbsp; "
            f"see the **3 Month Averages** tab for the full 3-month breakdown per salesman.",
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download Summary CSV",
            t1.to_csv(index=False).encode("utf-8"),
            file_name=f"summary_{zid}_{cur_year}_{cur_month:02d}.csv",
            mime="text/csv",
            key="dl_ov_summary",
        )

        # ── Raw data export — pick a salesman, get their row as copy/paste text ──
        with st.expander("📋 Raw Data Export (copy/paste)", expanded=False):
            sel_export_sp = st.selectbox(
                "Select Salesman", t1["Salesman"].tolist(), key="tm_ov_export_sp"
            )
            if sel_export_sp:
                export_row = t1[t1["Salesman"] == sel_export_sp].iloc[0].to_dict()
                st.code(_format_unquoted_dict(export_row), language=None)

    # ── Prior 3 months — one expander each ───────────────────────────────────
    st.markdown("---")
    st.subheader("📆 Previous 3 Months")
    for _i in range(1, 4):
        _prior = mo_start_cur - pd.DateOffset(months=_i)
        _render_prior_month_section(
            df, returns_df, zid, int(_prior.year), int(_prior.month), holidays, collection_df
        )


# ── Prior-month salesman performance section ─────────────────────────────────

def _render_prior_month_section(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    zid,
    year: int,
    month: int,
    holidays: set,
    collection_df: pd.DataFrame = None,
):
    """
    Render one prior month's salesman performance inside an expander.
    Columns match the current-month summary table, with Days Left / Daily Required
    fixed at 0 and Sales showing the full-month total.
    Monthly Avg (3M) = average of the 3 months immediately before this month.
    """
    import calendar as _cal

    mo_start  = pd.Timestamp(year, month, 1)
    last_day  = _cal.monthrange(year, month)[1]
    mo_end    = pd.Timestamp(year, month, last_day)
    mo_label  = f"{_cal.month_abbr[month]} {year}"

    # Working days in this month
    wd_month = _count_working_days(mo_start.date(), mo_end.date(), holidays)

    # 3-month lookback window (months immediately before this month)
    m3_end   = mo_start - pd.Timedelta(days=1)
    m3_start = mo_start - pd.DateOffset(months=3)

    df = sales_df.copy()
    if "_dt" not in df.columns:
        df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    if "_d" not in df.columns:
        df["_d"] = df["_dt"].dt.date

    mo_data = df[(df["_dt"] >= mo_start) & (df["_dt"] <= mo_end)]
    m3_data = df[(df["_dt"] >= m3_start) & (df["_dt"] < mo_start)]

    # ── Returns for this month per salesman ───────────────────────────────────
    ret_by_sp: dict = {}
    if returns_df is not None and not returns_df.empty and "treturnamt" in returns_df.columns:
        _r = returns_df.copy()
        _r["_dt"] = pd.to_datetime(_r["date"], errors="coerce")
        _r_mo = _r[(_r["_dt"] >= mo_start) & (_r["_dt"] <= mo_end)]
        if "spid" in _r_mo.columns:
            ret_by_sp = _r_mo.groupby(_r_mo["spid"].astype(str))["treturnamt"].sum().astype(float).to_dict()

    # ── Collection per salesman for this month ────────────────────────────────
    prior_coll_by_sp: dict = {}
    if collection_df is not None and not collection_df.empty and "value" in collection_df.columns:
        _c = collection_df.copy()
        _c["spid"]  = _c["spid"].astype(str)
        _c["year"]  = pd.to_numeric(_c["year"],  errors="coerce")
        _c["month"] = pd.to_numeric(_c["month"], errors="coerce")
        _mo_c = _c[(_c["year"] == year) & (_c["month"] == month)]
        prior_coll_by_sp = _mo_c.groupby("spid")["value"].sum().astype(float).to_dict()

    sp_list = (
        df[["spid", "spname"]].dropna().drop_duplicates()
        .sort_values("spname")
    )

    rows = []
    for _, sp_row in sp_list.iterrows():
        spid   = str(sp_row["spid"])
        spname = sp_row["spname"]

        sp_mo = mo_data[mo_data["spid"].astype(str) == spid]
        sp_m3 = m3_data[m3_data["spid"].astype(str) == spid]

        sales      = float(sp_mo["final_sales"].sum())
        ret        = round(ret_by_sp.get(spid, 0.0), 0)
        net_sales  = round(sales - ret, 0)
        target     = float(_get_target(zid, spid, year, month) or 0.0)
        pct_tgt    = round(net_sales / target * 100, 1) if target > 0 else 0.0
        daily_avg  = round(sales / wd_month, 0) if wd_month > 0 else 0.0
        monthly_avg_3m = round(float(sp_m3["final_sales"].sum()) / 3, 0)

        coll       = round(float(prior_coll_by_sp.get(spid, 0.0)), 0)
        pct_coll   = round(coll / (1.02 * sales) * 100, 1) if sales > 0 else 0.0

        uc = int(sp_mo["cusid"].nunique())    if "cusid"    in sp_mo.columns else 0
        up = int(sp_mo["itemcode"].nunique()) if "itemcode" in sp_mo.columns else 0

        rows.append({
            "Salesman":                spname,
            "Target":                  target,
            "Sales":                   round(sales, 0),
            "Return":                  ret,
            "Net Sales":               net_sales,
            "% vs Target":             pct_tgt,
            "Collection":              coll,
            "% Collection":            pct_coll,
            "Products Sold":           up,
            "Customers Visited":       uc,
            "Days Left":               0,
            "Daily Required":          0,
            "Daily Avg":               daily_avg,
            "Monthly Avg (3M)":        monthly_avg_3m,
        })

    with st.expander(f"📅 {mo_label}", expanded=False):
        if not rows:
            st.info("No data available for this month.")
            return

        t = (
            pd.DataFrame(rows)
            .sort_values("Sales", ascending=False)
            .reset_index(drop=True)
        )

        def _style_prior(df_inner):
            styled = df_inner.style.format(
                {
                    "Target":              "{:,.0f}",
                    "Sales":               "{:,.0f}",
                    "Return":              "{:,.0f}",
                    "Net Sales":           "{:,.0f}",
                    "% vs Target":         "{:.1f}%",
                    "Collection":          "{:,.0f}",
                    "% Collection":        "{:.1f}%",
                    "Products Sold":       "{:,.0f}",
                    "Customers Visited":   "{:,.0f}",
                    "Days Left":           "{:,.0f}",
                    "Daily Required":      "{:,.0f}",
                    "Daily Avg":           "{:,.0f}",
                    "Monthly Avg (3M)":    "{:,.0f}",
                },
                na_rep="—",
            )

            def _col_pct(col):
                out = []
                for v in col:
                    if not v:
                        out.append("")
                    elif v >= 100:
                        out.append("background-color: #D4EDDA; color: #155724")
                    elif v >= 75:
                        out.append("background-color: #FFF3CD; color: #856404")
                    else:
                        out.append("background-color: #F8D7DA; color: #721C24")
                return out

            if "% vs Target" in df_inner.columns:
                styled = styled.apply(_col_pct, subset=["% vs Target"])
            if "% Collection" in df_inner.columns:
                styled = styled.apply(_col_pct, subset=["% Collection"])
            return styled

        try:
            st.dataframe(_style_prior(t), width="stretch", hide_index=True)
        except Exception:
            st.dataframe(t, width="stretch", hide_index=True)

        st.download_button(
            f"⬇ Download {mo_label} CSV",
            t.to_csv(index=False).encode("utf-8"),
            file_name=f"summary_{zid}_{year}_{month:02d}.csv",
            mime="text/csv",
            key=f"dl_prior_{year}_{month:02d}",
        )


# ── 3 Month Averages ──────────────────────────────────────────────────────────

def _render_three_month_averages(sales_df: pd.DataFrame, returns_df: pd.DataFrame, zid):
    """
    Per-salesman 3-month averages for the last 3 calendar months INCLUDING the
    current month (this month to-date + the 2 complete months before it):
    Daily/Monthly Avg Sales, Unique Customers/Products + their daily averages,
    and ZID-wide Unique Products for comparison.

    Also hosts Moving Average Analysis underneath (folded in from the former
    standalone "Moving Average" tab).
    """
    st.subheader("📊 3 Month Averages")
    today = pd.Timestamp.today().normalize()
    holidays = _get_holidays()

    mo_start_cur = pd.Timestamp(today.year, today.month, 1)
    mo_start_3mo = mo_start_cur - pd.DateOffset(months=2)
    end_3mo      = today
    wd_3mo       = _count_working_days(mo_start_3mo.date(), end_3mo.date(), holidays)

    if "date" not in sales_df.columns or "final_sales" not in sales_df.columns:
        st.warning("Required columns missing.")
        return

    df = sales_df.copy()
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["_d"]  = df["_dt"].dt.date

    last3 = df[(df["_dt"] >= mo_start_3mo) & (df["_dt"] <= end_3mo)]

    if last3.empty:
        st.warning(
            f"⚠️ No data found for the 3-month window "
            f"({mo_start_3mo.strftime('%b %Y')} – {end_3mo.strftime('%b %Y')}). "
            "Please load at least 3 months (incl. this one) in the sidebar filters."
        )
        return

    zid_up = int(last3["itemcode"].nunique()) if "itemcode" in last3.columns else 0

    sp_list = df[["spid", "spname"]].dropna().drop_duplicates().sort_values("spname")

    rows = []
    for _, sp_row in sp_list.iterrows():
        spid   = str(sp_row["spid"])
        spname = sp_row["spname"]
        sp3 = last3[last3["spid"].astype(str) == spid]

        total_3mo   = float(sp3["final_sales"].sum())
        daily_avg   = round(total_3mo / wd_3mo, 0) if wd_3mo > 0 else 0.0
        monthly_avg = round(total_3mo / 3, 0)

        uc_3mo   = int(sp3["cusid"].nunique())    if "cusid"    in sp3.columns else 0
        up_3mo   = int(sp3["itemcode"].nunique()) if "itemcode" in sp3.columns else 0
        daily_uc = round(float(sp3.groupby("_d")["cusid"].nunique().mean()),    1) if not sp3.empty and "cusid"    in sp3.columns else 0.0
        daily_up = round(float(sp3.groupby("_d")["itemcode"].nunique().mean()), 1) if not sp3.empty and "itemcode" in sp3.columns else 0.0

        rows.append({
            "Salesman ID":              spid,
            "Salesman":                 spname,
            "Daily Avg Sales (3M)":     daily_avg,
            "Monthly Avg Sales (3M)":   monthly_avg,
            "Unique Customers (3M)":    uc_3mo,
            "Avg Daily Customers (3M)": daily_uc,
            "Unique Products (3M)":     up_3mo,
            "Avg Daily Products (3M)":  daily_up,
            "ZID Unique Products (3M)": zid_up,
        })

    if not rows:
        st.info("No salesmen found for this selection.")
        return

    t = pd.DataFrame(rows).sort_values("Monthly Avg Sales (3M)", ascending=False).reset_index(drop=True)

    _3m_period = (f"{mo_start_3mo.strftime('%b %Y')} – {end_3mo.strftime('%b %Y')}"
                  f" ({wd_3mo} working days)")
    st.caption(f"3M window: {_3m_period}")

    try:
        styled = t.style.format({
            "Daily Avg Sales (3M)":     "{:,.0f}",
            "Monthly Avg Sales (3M)":   "{:,.0f}",
            "Unique Customers (3M)":    "{:,.0f}",
            "Avg Daily Customers (3M)": "{:.1f}",
            "Unique Products (3M)":     "{:,.0f}",
            "Avg Daily Products (3M)":  "{:.1f}",
            "ZID Unique Products (3M)": "{:,.0f}",
        }, na_rep="—")
        st.dataframe(styled, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(t, width="stretch", hide_index=True)

    st.download_button(
        "⬇ Download 3 Month Averages CSV",
        t.to_csv(index=False).encode("utf-8"),
        file_name=f"three_month_averages_{zid}.csv",
        mime="text/csv",
        key="dl_3m_avg",
    )

    # ── Moving Average Analysis (folded in from the former standalone tab) ────
    st.markdown("---")
    st.subheader("📈 Moving Average Analysis")

    from datetime import date as _date
    ma_col1, ma_col2, ma_col3 = st.columns(3)
    with ma_col1:
        ma_entity = st.selectbox(
            "Entity", ["Salesman", "Product", "Product Group"], key="tm_ma_entity"
        )
    with ma_col2:
        ma_metric = st.selectbox(
            "Metric", ["Net Sales", "Net Returns"], key="tm_ma_metric"
        )
    with ma_col3:
        ma_end_date = st.date_input(
            "End Date", value=_date.today(), key="tm_ma_end_date"
        )

    try:
        ma_df = ds.compute_moving_avg_table(
            sales_df=sales_df,
            returns_df=returns_df,
            entity=ma_entity,
            metric=ma_metric,
            end_date=ma_end_date,
            collection_df=None,
        )
        if ma_df is not None and not ma_df.empty:
            st.dataframe(ma_df, width="stretch")
        else:
            st.info("No moving average data available for the selected filters.")
    except Exception as _ma_err:
        st.warning("Unable to compute moving average.")
        st.caption(f"Details: {_ma_err}")


# ── Salesman daily breakdown (current month) ──────────────────────────────────

def _render_sp_daily_breakdown(
    sp_sales: pd.DataFrame,
    opmob_df: pd.DataFrame,
    sel_spid: str,
    zid,
):
    """
    Daily breakdown for the selected salesman for the current month:
    one row per date × area with Sales, Pending opmob, Uniq Cust, Uniq Prods.
    """
    today        = pd.Timestamp.today().normalize()
    cur_year     = today.year
    cur_month    = today.month
    mo_start_cur = pd.Timestamp(cur_year, cur_month, 1)

    if "date" not in sp_sales.columns or sp_sales.empty:
        st.info("No sales data available for the daily breakdown.")
        return

    df = sp_sales.copy()
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["_d"]  = df["_dt"].dt.date
    mtd = df[(df["_dt"] >= mo_start_cur) & (df["_dt"] <= today)]

    if mtd.empty or "area" not in mtd.columns:
        st.info("No current-month sales data available for this salesman.")
        return

    # cusid → area lookup for pending opmob mapping
    cusid_area_map: dict = {}
    if "cusid" in df.columns and "area" in df.columns:
        cusid_area_map = (
            df[["cusid", "area"]].dropna()
            .drop_duplicates("cusid")
            .set_index("cusid")["area"]
            .to_dict()
        )

    # pending opmob by area for this salesman
    pend_area: dict = {}
    if not opmob_df.empty and "cusid" in opmob_df.columns and "linetotal" in opmob_df.columns:
        ob = opmob_df.copy()
        ob["area"] = ob["cusid"].astype(str).map(cusid_area_map)
        for ar, grp in ob.dropna(subset=["area"]).groupby("area"):
            pend_area[str(ar)] = float(grp["linetotal"].sum())

    grp = (
        mtd.dropna(subset=["area"])
        .groupby(["_d", "area"])
        .agg(
            Sales       =("final_sales", "sum"),
            uniq_cust   =("cusid",       pd.Series.nunique),
            uniq_prods  =("itemcode",    pd.Series.nunique),
        )
        .reset_index()
        .rename(columns={"_d": "Date", "area": "Area",
                         "uniq_cust": "Uniq Cust", "uniq_prods": "Uniq Prods"})
        .sort_values(["Date", "Area"], ascending=[False, True])
    )

    grp["Pending"] = grp["Area"].apply(lambda a: pend_area.get(str(a), 0.0))

    t = grp[["Date", "Area", "Sales", "Pending", "Uniq Cust", "Uniq Prods"]].reset_index(drop=True)

    try:
        st.dataframe(
            t.style.format({
                "Sales":      "{:,.0f}",
                "Pending":    "{:,.0f}",
                "Uniq Cust":  "{:,.0f}",
                "Uniq Prods": "{:,.0f}",
            }, na_rep="—"),
            width="stretch",
            hide_index=True,
            height=min(35 * len(t) + 60, 520),
        )
    except Exception:
        st.dataframe(t, width="stretch", hide_index=True)

    st.download_button(
        "⬇ Download Daily Breakdown CSV",
        t.to_csv(index=False).encode("utf-8"),
        file_name=f"daily_{sel_spid}_{cur_year}_{cur_month:02d}.csv",
        mime="text/csv",
        key="dl_sp_daily",
    )


# ── Collection Details tab — salesman + month picker, total shown outside ─────

def _render_collection_details_tab(
    sales_df: pd.DataFrame,
    zid,
    collection_df: pd.DataFrame,
    return_df: pd.DataFrame,
):
    """
    Voucher-level collection rows (RCT/CRCT/BRCT/STJV/JV--/ADJV) for one
    salesman and one month, plus a unified day-book table beneath it showing
    all transaction types (DOs, returns, collections) in separate amount columns.
    """
    st.subheader("🧾 SR Trn")
    today = pd.Timestamp.today().normalize()

    month_opts = ssc.month_choices(today)
    sel_label = st.selectbox(
        "Month", [m[0] for m in month_opts], index=0, key="tm_coll_details_month"
    )
    sel_year, sel_month = next((y, m) for (lbl, y, m) in month_opts if lbl == sel_label)

    sp_opts = _sp_opts(sales_df)
    sel_sp_raw = st.selectbox(
        "Salesman",
        [None] + sp_opts,
        format_func=lambda x: "— select a salesman —" if x is None else x,
        key="tm_coll_details_sp",
    )
    if not sel_sp_raw:
        st.info("👆 Select a salesman to view collection details.")
        return
    sel_spid = _codes([sel_sp_raw])[0]

    # ── Helper: normalise a dataframe's spid/year/month columns ──────────────
    def _prep(df):
        df = df.copy()
        df["spid"]  = df["spid"].astype(str)
        df["year"]  = pd.to_numeric(df["year"],  errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        return df

    def _sp_month_filter(df):
        return df[(df["spid"] == str(sel_spid)) & (df["year"] == sel_year) & (df["month"] == sel_month)]

    # ── Collections ───────────────────────────────────────────────────────────
    if collection_df is None or collection_df.empty or not {"spid","year","month","glvoucher","date","cusid","value"}.issubset(collection_df.columns):
        st.info("No collection data loaded.")
        coll_detail = pd.DataFrame()
    else:
        c = _prep(collection_df)
        coll_detail = _sp_month_filter(c).copy()

    total = float(coll_detail["value"].sum()) if not coll_detail.empty else 0.0
    st.metric(f"💰 Total Collection — {sel_label}", f"{total:,.0f}")

    if not coll_detail.empty:
        coll_detail["date"] = pd.to_datetime(coll_detail["date"], errors="coerce")
        coll_cols = ["date", "glvoucher", "cusid"] + (["cusname"] if "cusname" in coll_detail.columns else []) + ["value"]
        t = (
            coll_detail[coll_cols]
            .rename(columns={"date": "Date", "glvoucher": "Transaction Code",
                              "cusid": "Customer Code", "cusname": "Customer", "value": "Amount"})
            .sort_values("Date", ascending=False)
            .reset_index(drop=True)
        )
        st.caption(f"{len(t):,} collection transaction(s)")
        try:
            st.dataframe(
                t.style.format({"Date": "{:%Y-%m-%d}", "Amount": "{:,.0f}"}, na_rep="—"),
                width="stretch", hide_index=True,
            )
        except Exception:
            st.dataframe(t, width="stretch", hide_index=True)

        st.download_button(
            "⬇ Download Collection CSV",
            t.to_csv(index=False).encode("utf-8"),
            file_name=f"collection_detail_{sel_spid}_{sel_year}_{sel_month:02d}.csv",
            mime="text/csv",
            key="dl_collection_details_tab",
        )
    else:
        st.info("No collection transactions for this salesman in this month.")

    # ── Unified day-book: Sales (DO) + Returns + Collections + Mobile Orders ──
    st.markdown("---")
    st.markdown("#### 📋 All Transactions")

    rows = []

    _COLS = ["Date","Voucher","Cust Code","Customer","Sales","Return","Collection","Mobile Order"]

    # Sales (DOs) — one row per voucher per customer
    if sales_df is not None and not sales_df.empty and {"spid","year","month","voucher","date","cusid","cusname","altsales"}.issubset(sales_df.columns):
        s = _prep(sales_df)
        s_filt = _sp_month_filter(s)
        if not s_filt.empty:
            s_grp = (
                s_filt.groupby(["date","voucher","cusid","cusname"], as_index=False)
                      .agg(Sales=("altsales","sum"))
            )
            s_grp["date"] = pd.to_datetime(s_grp["date"], errors="coerce")
            s_grp = s_grp.rename(columns={"voucher":"Voucher","date":"Date","cusid":"Cust Code","cusname":"Customer"})
            s_grp["Return"]       = None
            s_grp["Collection"]   = None
            s_grp["Mobile Order"] = None
            rows.append(s_grp[_COLS])

    # Returns — one row per return voucher per customer
    if return_df is not None and not return_df.empty and {"spid","year","month","revoucher","date","cusid","cusname","treturnamt"}.issubset(return_df.columns):
        r = _prep(return_df)
        r_filt = _sp_month_filter(r)
        if not r_filt.empty:
            r_grp = (
                r_filt.groupby(["date","revoucher","cusid","cusname"], as_index=False)
                      .agg(Return=("treturnamt","sum"))
            )
            r_grp["date"] = pd.to_datetime(r_grp["date"], errors="coerce")
            r_grp = r_grp.rename(columns={"revoucher":"Voucher","date":"Date","cusid":"Cust Code","cusname":"Customer"})
            r_grp["Sales"]        = None
            r_grp["Collection"]   = None
            r_grp["Mobile Order"] = None
            rows.append(r_grp[_COLS])

    # Collections
    if not coll_detail.empty:
        has_cusname = "cusname" in coll_detail.columns
        c_cols = ["date","glvoucher","cusid"] + (["cusname"] if has_cusname else [])
        c_txn = coll_detail[c_cols + ["value"]].copy()
        if not has_cusname:
            c_txn["cusname"] = None
        c_txn = c_txn.rename(columns={"date":"Date","glvoucher":"Voucher","cusid":"Cust Code","cusname":"Customer","value":"Collection"})
        c_txn["Sales"]        = None
        c_txn["Return"]       = None
        c_txn["Mobile Order"] = None
        rows.append(c_txn[_COLS])

    # Mobile orders (opmob) — one row per order number per customer
    mob_df = _load_opmob_all(str(zid))
    if not mob_df.empty and {"spid","year","month","mob_voucher","date","cusid","cusname","mob_total"}.issubset(mob_df.columns):
        m = _prep(mob_df)
        m_filt = _sp_month_filter(m)
        if not m_filt.empty:
            m_filt = m_filt.copy()
            m_filt["date"] = pd.to_datetime(m_filt["date"], errors="coerce")
            m_filt = m_filt.rename(columns={"mob_voucher":"Voucher","date":"Date","cusid":"Cust Code","cusname":"Customer","mob_total":"Mobile Order"})
            m_filt["Sales"]      = None
            m_filt["Return"]     = None
            m_filt["Collection"] = None
            rows.append(m_filt[_COLS])

    if not rows:
        st.info("No transactions found for this salesman and month.")
        return

    txn = pd.concat(rows, ignore_index=True)
    txn["Date"] = pd.to_datetime(txn["Date"], errors="coerce")
    txn = txn.sort_values(["Date","Voucher"]).reset_index(drop=True)

    st.caption(f"{len(txn):,} transaction row(s) — DOs, Returns, Collections, Mobile Orders")
    try:
        st.dataframe(
            txn.style.format(
                {"Date": "{:%Y-%m-%d}", "Sales": "{:,.0f}", "Return": "{:,.0f}",
                 "Collection": "{:,.0f}", "Mobile Order": "{:,.0f}"},
                na_rep="—",
            ),
            width="stretch",
            hide_index=True,
        )
    except Exception:
        st.dataframe(txn, width="stretch", hide_index=True)

    st.download_button(
        "⬇ Download All Transactions CSV",
        txn.to_csv(index=False).encode("utf-8"),
        file_name=f"all_txn_{sel_spid}_{sel_year}_{sel_month:02d}.csv",
        mime="text/csv",
        key="dl_all_txn_tab",
    )


# ── Main view ─────────────────────────────────────────────────────────────────

@timed
def display_target_management_page(current_page, zid, data_dict):
    st.title("Target Management")

    # ── Maintenance: prune stale JSON entries on every load ───────────────────
    _prune_targets()
    _prune_holidays()

    # ── Holiday warning: prompt if no holidays entered for current year ────────
    _cur_year = pd.Timestamp.today().year
    _cur_year_holidays = [h for h in _get_holidays() if h.startswith(str(_cur_year))]
    if not _cur_year_holidays:
        st.warning(
            f"⚠️ No public holidays have been entered for **{_cur_year}**. "
            "Working-day calculations (daily averages, daily required, days left) "
            "will not account for public holidays until you add them. "
            "Please open the **🗓 Manage Public Holidays** panel above and add this year's holidays."
        )

    raw_sales   = data_dict.get("sales",  pd.DataFrame())
    raw_returns = data_dict.get("return", pd.DataFrame())

    if raw_sales.empty:
        st.warning("No sales data available for the selected filters.")
        return

    sales_df, returns_df = common.data_copy_add_columns(raw_sales, raw_returns)

    if "final_sales" not in sales_df.columns:
        st.error("Could not compute net sales — 'final_sales' column missing.")
        return

    current_col = _current_month_label()

    # ── View mode radio ───────────────────────────────────────────────────────
    _view_mode = st.radio(
        "View",
        ["👤 Individual Salesman", "📊 All Salesmen Overview", "🎯 Salesman Score",
         "📊 3 Month Averages", "🧾 SR Trn",
         "📦 Current Stock", "🔮 Next Month Target", "🗺️ Field Tracking",
         "📲 App Collections", "↩️ Returns Registry"],
        horizontal=True,
        key="tm_view_mode",
    )

    # ── Public holidays management (always accessible) ───────────────────────
    with st.expander("🗓 Manage Public Holidays", expanded=False):
        all_holidays = sorted(_get_holidays())

        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            hol_range = st.date_input(
                "Select a day or drag to pick a range (any month, any year)",
                value=(),
                key="tm_new_hol",
            )
        with h_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add", key="tm_add_hol"):
                if hol_range:
                    if isinstance(hol_range, (list, tuple)) and len(hol_range) == 2:
                        cur_h, end_h = hol_range
                        while cur_h <= end_h:
                            _toggle_holiday(str(cur_h), add=True)
                            cur_h += timedelta(days=1)
                    else:
                        single = hol_range[0] if isinstance(hol_range, (list, tuple)) else hol_range
                        _toggle_holiday(str(single), add=True)
                    st.rerun()

        if all_holidays:
            from itertools import groupby
            def _ym(d): return d[:7]  # "YYYY-MM"
            st.write(f"**{len(all_holidays)} holiday(s) saved across all months:**")
            for ym, group in groupby(all_holidays, key=_ym):
                yr, mo = int(ym[:4]), int(ym[5:])
                st.markdown(f"*{calendar.month_abbr[mo]} {yr}*")
                for h in group:
                    hc1, hc2 = st.columns([4, 1])
                    with hc1:
                        st.write(h)
                    with hc2:
                        if st.button("✖", key=f"rm_hol_{h}"):
                            _toggle_holiday(h, add=False)
                            st.rerun()
        else:
            st.info("No public holidays saved yet.")

    if _view_mode == "📊 All Salesmen Overview":
        opmob_all = _load_opmob_pending(str(zid))
        _render_overview(sales_df, returns_df, opmob_all, zid,
                         collection_df=data_dict.get("collection", pd.DataFrame()))
        return

    if _view_mode == "🎯 Salesman Score":
        _render_salesman_score(sales_df, returns_df, zid,
                                collection_df=data_dict.get("collection", pd.DataFrame()))
        return

    if _view_mode == "📊 3 Month Averages":
        _render_three_month_averages(sales_df, returns_df, zid)
        return

    if _view_mode == "🧾 SR Trn":
        _render_collection_details_tab(
            sales_df,
            zid,
            data_dict.get("collection", pd.DataFrame()),
            returns_df if returns_df is not None else pd.DataFrame(),
        )
        return

    if _view_mode == "📲 App Collections":
        _render_glpmt_panel(str(zid), key_suffix="_tm")
        return

    if _view_mode == "↩️ Returns Registry":
        _render_returns_registry(str(zid))
        return

    if _view_mode == "📦 Current Stock":
        st.subheader("📦 Current Stock")
        with st.spinner("Loading stock data…"):
            stock_df  = _load_final_items(str(zid))
            wh_df     = _load_opspprc(str(zid))

        if stock_df.empty:
            st.warning("No stock data available from final_items_view for this entity.")
        else:
            # Merge wholesale price info if available
            if not wh_df.empty:
                wh_df = wh_df[["item_id", "wh_qty", "wh_price"]].copy()
                wh_df["item_id"] = wh_df["item_id"].astype(str)
                stock_df["item_id"] = stock_df["item_id"].astype(str)
                stock_df = stock_df.merge(wh_df, on="item_id", how="left")

            _col_map = {
                "item_id":    "Item ID",
                "item_name":  "Item Name",
                "item_group": "Item Group",
                "stock":      "Stock",
                "wh_qty":     "WH Qty",
                "wh_price":   "WH Price",
            }
            disp = (
                stock_df
                .rename(columns=_col_map)
                [[c for c in _col_map.values() if c in stock_df.rename(columns=_col_map).columns]]
                .reset_index(drop=True)
            )

            # Search filter
            _search = st.text_input("🔍 Search by Item Name or Group", key="tm_stock_search")
            if _search:
                _mask = (
                    disp["Item Name"].str.contains(_search, case=False, na=False) |
                    disp["Item Group"].str.contains(_search, case=False, na=False)
                )
                disp = disp[_mask].reset_index(drop=True)

            st.caption(f"{len(disp):,} items  —  WH Qty = minimum qty for wholesale price; WH Price = std price minus discount")
            fmt = {"Stock": "{:,.0f}"}
            if "WH Qty" in disp.columns:
                fmt["WH Qty"]   = "{:,.0f}"
                fmt["WH Price"] = "{:,.2f}"
            st.dataframe(
                disp.style.format(fmt, na_rep="—"),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "⬇ Download CSV",
                disp.to_csv(index=False).encode("utf-8"),
                file_name=f"current_stock_{zid}.csv",
                mime="text/csv",
                key="dl_current_stock",
            )
        return

    if _view_mode == "🔮 Next Month Target":
        _render_next_month_target(zid)
        return

    if _view_mode == "🗺️ Field Tracking":
        _render_field_tracking(zid)
        return

    # ── Filters: salesman (single), customer, area (cascading) ────────────────
    fcols = st.columns(3)

    with fcols[0]:
        sp_opts = _sp_opts(sales_df)
        sel_sp_raw = st.selectbox(
            "Salesman *(required)",
            [None] + sp_opts,
            format_func=lambda x: "— select a salesman —" if x is None else x,
            key="tm_sp",
        )

    if not sel_sp_raw:
        st.info("👆 Select a salesman to view reports.")
        return

    sel_spid  = _codes([sel_sp_raw])[0]
    sel_spids = [sel_spid]

    # Filter by salesman
    f_sp     = _filter_code(sales_df,   "spid", sel_spids)
    f_sp_ret = _filter_code(returns_df, "spid", sel_spids)

    # Load pending opmob orders and filter to this salesman
    opmob_df = _load_opmob_pending(str(zid))
    if not opmob_df.empty:
        opmob_df = opmob_df[opmob_df["spid"].astype(str).isin([str(s) for s in sel_spids])].copy()
    pending_cusids = set(opmob_df["cusid"].astype(str).unique()) if not opmob_df.empty else set()

    # Customer options cascade from salesman selection
    with fcols[1]:
        sel_cus_raw = st.multiselect("Customer", _cus_opts(f_sp), key="tm_cus")

    sel_cusids = _codes(sel_cus_raw)
    f_sp_cus   = _filter_code(f_sp,     "cusid", sel_cusids) if sel_cusids else f_sp
    f_sp_cus_r = _filter_code(f_sp_ret, "cusid", sel_cusids) if sel_cusids else f_sp_ret

    # Area options cascade from salesman + customer selection
    with fcols[2]:
        area_opts = sorted(f_sp_cus["area"].dropna().unique().tolist())
        sel_area  = st.multiselect("Area", area_opts, key="tm_area")

    f_final   = f_sp_cus[f_sp_cus["area"].isin(sel_area)]     if sel_area else f_sp_cus
    f_final_r = f_sp_cus_r[f_sp_cus_r["area"].isin(sel_area)] if sel_area and "area" in f_sp_cus_r.columns else f_sp_cus_r

    # ── Metric cards (uses full salesman data, not customer/area filtered) ─────
    collection_df_all = data_dict.get("collection", pd.DataFrame())
    _render_metric_cards(
        f_sp, opmob_df, sel_spid, zid, sp_returns=f_sp_ret,
        collection_df=collection_df_all,
    )

    # ── Customer-wise pivot ───────────────────────────────────────────────────
    try:
        pivot = tm.build_customer_wise_monthly(f_final, f_final_r)
    except Exception as e:
        st.warning("Unable to build report. Please adjust your filters.")
        st.caption(f"Details: {e}")
        return

    if pivot.empty:
        st.warning("No data for the current selection.")
        return

    id_raw      = ["spid", "spname", "cusid", "cusname", "cusmobile", "whatsapp", "area"]
    month_col_list = [c for c in pivot.columns if c not in id_raw and c != "Total"]

    rename_map = {
        "spname":    "Salesman",
        "cusid":     "Cust. Code",
        "cusname":   "Customer",
        "cusmobile": "Mobile",
        "whatsapp":  "WhatsApp Number",
        "area":      "Area",
    }
    display_pivot = (
        pivot
        .drop(columns=["spid"], errors="ignore")
        .rename(columns=rename_map)
    )
    id_cols_display = [rename_map.get(c, c) for c in id_raw if c != "spid"]

    # Split on running month
    if current_col in display_pivot.columns:
        not_ordered = display_pivot[display_pivot[current_col] == 0].copy()
        ordered     = display_pivot[display_pivot[current_col]  > 0].copy()
    else:
        not_ordered = display_pivot.copy()
        ordered     = pd.DataFrame(columns=display_pivot.columns)

    st.subheader(f"🔴 Not Ordered — {current_col}")

    # Add pending order indicator column
    if pending_cusids and not not_ordered.empty:
        not_ordered["Pending Order"] = not_ordered["Cust. Code"].astype(str).apply(
            lambda x: "✓" if x in pending_cusids else ""
        )
        not_ordered_id_cols = id_cols_display + ["Pending Order"]
    else:
        not_ordered_id_cols = id_cols_display

    _render_not_ordered_table(
        not_ordered, not_ordered_id_cols, current_col, pending_cusids,
        "not_ordered", f"not_ordered_{current_col}.csv",
    )

    # Expander: pending order product breakdown
    if pending_cusids and not not_ordered.empty:
        not_ordered_cusids = set(not_ordered["Cust. Code"].astype(str).unique())
        pending_in_not_ordered = pending_cusids & not_ordered_cusids
        if pending_in_not_ordered and not opmob_df.empty:
            pending_detail = opmob_df[
                opmob_df["cusid"].astype(str).isin(pending_in_not_ordered)
            ].copy()
            with st.expander(
                f"📋 Pending opmob Orders — {len(pending_in_not_ordered)} customer(s)", expanded=False
            ):
                detail_display = (
                    pending_detail[["cusname", "cusid", "itemcode", "itemname", "linetotal"]]
                    .rename(columns={
                        "cusname":   "Customer",
                        "cusid":     "Cust. Code",
                        "itemcode":  "Item Code",
                        "itemname":  "Item",
                        "linetotal": "Line Total",
                    })
                    .reset_index(drop=True)
                )
                try:
                    st.dataframe(
                        detail_display.style.format({"Line Total": "{:,.2f}"}, na_rep="-"),
                        width="stretch",
                    )
                except Exception:
                    st.dataframe(detail_display, width="stretch")
                st.download_button(
                    label=f"⬇ Download CSV ({len(detail_display):,} rows)",
                    data=detail_display.to_csv(index=False).encode("utf-8"),
                    file_name=f"pending_orders_{current_col}.csv",
                    mime="text/csv",
                    key="dl_pending_orders",
                )

    st.markdown(" ")
    st.subheader(f"✅ Ordered — {current_col}")
    _render_table(ordered, id_cols_display, current_col, "ordered", f"ordered_{current_col}.csv")

    st.markdown("---")

    # ── Customer-Product sub-section ──────────────────────────────────────────
    st.subheader("📦 Customer-Product Breakdown")
    st.caption("Scoped to salesman / customer / area selections above.")

    scols = st.columns(2)
    with scols[0]:
        sel_sec_cus_raw  = st.multiselect("Customer", _cus_opts(f_final), key="tm_sec_cus")

    sel_sec_cusids = _codes(sel_sec_cus_raw)
    fs2 = _filter_code(f_final,   "cusid", sel_sec_cusids) if sel_sec_cusids else f_final
    fr2 = _filter_code(f_final_r, "cusid", sel_sec_cusids) if sel_sec_cusids else f_final_r

    with scols[1]:
        sel_sec_item_raw = st.multiselect("Product", _item_opts(fs2), key="tm_sec_item")

    sel_sec_itemcodes = _codes(sel_sec_item_raw)
    fs2 = _filter_code(fs2, "itemcode", sel_sec_itemcodes) if sel_sec_itemcodes else fs2
    fr2 = _filter_code(fr2, "itemcode", sel_sec_itemcodes) if sel_sec_itemcodes else fr2

    try:
        prod_pivot = tm.build_customer_product_monthly(fs2, fr2)
    except Exception as e:
        st.warning("Unable to build product breakdown.")
        st.caption(f"Details: {e}")
        prod_pivot = pd.DataFrame()

    if not prod_pivot.empty:
        prod_rename = {
            "spname":    "Salesman",
            "cusid":     "Cust. Code",
            "cusname":   "Customer",
            "cusmobile": "Mobile",
            "whatsapp":  "WhatsApp Number",
            "area":      "Area",
            "itemcode":  "Item Code",
            "itemname":  "Item",
        }
        prod_id_raw = ["spname", "cusid", "cusname", "cusmobile", "whatsapp", "area", "itemcode", "itemname"]
        prod_display = (
            prod_pivot
            .drop(columns=["spid"], errors="ignore")
            .rename(columns=prod_rename)
        )
        prod_id_cols = [prod_rename.get(c, c) for c in prod_id_raw]
        _render_table(prod_display, prod_id_cols, current_col, "prod_breakdown", "customer_product_breakdown.csv")
    else:
        st.info("No product-level data for the current selection.")

    st.markdown("---")

    # ── No-sales customers from cacus ─────────────────────────────────────────
    st.subheader("🚫 Customers with No Sales")
    st.caption("Select an area to see customers in the directory with zero sales in the loaded period.")

    cacus_df = _load_cacus_directory(str(zid))

    if cacus_df.empty:
        st.warning("Customer directory not available.")
    else:
        # Use the same area options as the top filter (salesman's territory)
        sel_cacus_area = st.multiselect("Filter by Area", area_opts, key="tm_cacus_area")

        if not sel_cacus_area:
            st.info("Select one or more areas above to see customers with no sales.")
        else:
            cacus_filtered = cacus_df[cacus_df["area"].isin(sel_cacus_area)].copy()
            sold_cusids    = set(sales_df["cusid"].astype(str).unique())
            no_sales_df    = cacus_filtered[
                ~cacus_filtered["cusid"].astype(str).isin(sold_cusids)
            ].copy()

            no_sales_display = (
                no_sales_df
                .rename(columns={"cusname": "Customer", "cusmobile": "Mobile",
                                  "whatsapp": "WhatsApp Number", "area": "Area"})
                [["Customer", "Mobile", "WhatsApp Number", "Area"]]
                .reset_index(drop=True)
            )
            st.caption(f"{len(no_sales_display):,} customers with no sales in the selected area(s)")

            if no_sales_display.empty:
                st.success("All customers in the selected area(s) have sales in this period.")
            else:
                st.dataframe(no_sales_display, width="stretch")
                st.download_button(
                    label=f"⬇ Download CSV ({len(no_sales_display):,} rows)",
                    data=no_sales_display.to_csv(index=False).encode("utf-8"),
                    file_name="no_sales_customers.csv",
                    mime="text/csv",
                    key="dl_no_sales",
                )

    # ── Daily Breakdown for selected salesman (collapsed by default) ──────────
    st.markdown("---")
    with st.expander("📅 Daily Breakdown — Current Month", expanded=False):
        _render_sp_daily_breakdown(f_sp, opmob_df, sel_spid, zid)

    # ── Buying Pattern Analysis — commented out for now ───────────────────────
    # try:
    #     bp_df = bp.compute_buying_pattern(
    #         pivot_df   = pivot,
    #         sales_df   = f_final,
    #         id_cols    = [c for c in id_raw if c in pivot.columns],
    #         month_cols = month_col_list,
    #     )
    # except Exception as e:
    #     bp_df = pd.DataFrame()
    #     st.caption(f"Buying pattern error: {e}")
    #
    # _render_buying_pattern(bp_df, is_any_filter=bool(sel_spid))

    # ── Inventory Coverage (collapsed; loads only on demand — can be slow) ─────
    st.markdown("---")
    with st.expander("🗂️ Inventory Coverage — This Month vs Prior-Month Stock", expanded=False):
        gen_key = f"tm_inv_cov_ready_{sel_spid}"
        if st.button("▶ Generate Inventory Coverage", key=f"tm_gen_inv_cov_{sel_spid}"):
            st.session_state[gen_key] = True
        if st.session_state.get(gen_key):
            _render_inventory_coverage(f_sp, str(zid))
        else:
            st.info("Click **Generate Inventory Coverage** above to load this section — it can take a while.")
