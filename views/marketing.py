import base64
import html as _html
import json
import re
import shutil
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core import whatsfly
from core import direct_whatsapp
from core import whatsapp_webhook_db

from views.call_log_shared import render_call_log_panel as _render_call_log_panel
from views.lead_call_log_shared import (
    render_lead_call_log_panel as _render_lead_call_log_panel,
    load_all_lead_call_logs as _load_all_lead_call_logs,
    bust_lead_call_log_cache as _bust_lead_call_log_cache,
)
from processing.marketing import (
    build_customer_marketing_table,
    build_area_campaign_top_customers,
    build_area_top_products,
    build_stock_gap,
    build_inactive_customers,
)
from processing.marketing_leads import (
    parse_leads_upload,
    build_manual_lead_row,
    build_lead_summary_table,
    build_lead_call_log_table,
    build_leads_upload_template,
)
from processing.common import normalize_phone_cols, customer_whatsapp_numbers
from core.analytics import Analytics


# ---------------------------------------------------------------------------
# dual-ZID constants (100001 + 100000 share the same field sales team)
# ---------------------------------------------------------------------------
_DUAL_ZIDS = frozenset({"100001", "100000"})
_OTHER_ZID  = {"100001": "100000", "100000": "100001"}

# ---------------------------------------------------------------------------
# column display config — customer scoring table
# ---------------------------------------------------------------------------

_DISPLAY_LABELS = {
    "cusid":                        "Customer ID",
    "cusname":                      "Customer Name",
    "cusmobile":                    "Mobile",
    "area":                         "Area",
    "spname":                       "Salesman",
    "total_sales":                  "Total Sales",
    "total_collection":             "Total Collection",
    "yoy_sales_growth_pct":         "Sales YoY Growth %",
    "yoy_collection_growth_pct":    "Collection YoY Growth %",
    "avg_days_to_collection":       "Avg Days to Collection",
    "avg_days_between_collections": "Avg Days Between Collections",
    "avg_order_interval_days":      "Avg Order Interval (days)",
    "monthly_activity_rate":        "Monthly Activity Rate %",
    "current_balance":              "Current Balance",
    "composite_score":              "Score",
}

_CURRENCY_COLS = {"total_sales", "total_collection", "current_balance"}
_PCT_COLS      = {"yoy_sales_growth_pct", "yoy_collection_growth_pct", "monthly_activity_rate"}
_DAYS_COLS     = {
    "avg_days_to_collection",
    "avg_days_between_collections",
    "avg_order_interval_days",
}
_HELPER_COLS = {"order_count", "coll_event_count"}

_NOTES = """
### Column Reference

| Column | Formula / Source | Notes |
|---|---|---|
| **Total Sales** | `SUM(altsales)` from `mv_sales_line_items` for selected year(s) + filters | Gross sales before discount; consistent with IS Revenue |
| **Total Collection** | `SUM(value)` from `mv_collection_vouchers` for selected year(s) + filters | Includes RCT, CRCT, BRCT, JV, STJV, ADJV voucher types |
| **Sales YoY Growth %** | **Sequential QoQ**: the selected period is treated as one continuous time series of quarters (Q1'24, Q2'24, … Q2'25). For each consecutive quarter pair, compute *(Qn − Qn-1) / Qn-1 × 100*. Average all those changes. Silent quarters (zero sales) are included in the grid, so going quiet produces a real -100% that is counted. Recovering from zero is skipped (undefined %). | **"New ↑"** = customer had no prior-quarter base across the entire window (entirely new). N/A when only 1 year selected |
| **Collection YoY Growth %** | Same sequential QoQ logic applied to collection amounts | Same "New ↑" logic applies |
| **Avg Days to Collection** | For each collection event: days elapsed since that customer's most recent invoice date. Averaged across all events in the selected period | Customers with no collection events are excluded |
| **Avg Days Between Collections** | Mean gap in days between consecutive collection vouchers per customer | Requires ≥ 2 collection events. Shows **"1 collection"** when only 1 event exists in the period |
| **Avg Order Interval (days)** | Mean gap in days between consecutive distinct order dates per customer | Requires ≥ 2 distinct order dates. Shows **"1 order"** when only 1 date exists in the period |
| **Monthly Activity Rate %** | Active months with ≥ 1 order ÷ total calendar months in selected period × 100 | 2 years selected → denominator is 24; a customer ordering in 7 of those 24 months scores 29.2% |
| **Current Balance** | `SUM(xprime)` across *all* AR ledger history (`mv_ar_transactions`) | **Not year-filtered** — reflects the live outstanding balance. Positive = customer owes; negative = customer is in credit |
| **Score** | Weighted composite of 7 metrics, each min-max scaled 0–100 | Weights: Total Sales 25%, Monthly Activity Rate 20%, Sales YoY Growth 15%, Avg Days to Collection 15% (inverted), Total Collection 10%, Avg Order Interval 10% (inverted), Collection YoY Growth 5%. "New ↑" growth (∞) is capped at the 90th percentile of finite values. Higher = better customer. |

### Year Aggregation
When multiple years are selected, sales and collection columns are summed across the full period. Growth metrics compare year-by-year within the selection. Interval and frequency metrics use all transaction dates in the period as a single continuous window.

### Filter Logic
The sidebar Salesman and Area filters restrict which customers appear by matching against the sales data. The Current Balance is always computed from the full AR ledger (no year restriction) so it reflects the customer's actual live balance regardless of the selected period.

### Why might a cell be blank?
- **Growth % blank**: both years had zero sales/collection (e.g. truly inactive).
- **"New ↑"**: customer first appeared in the later year — growth from zero is undefined as a %.
- **Interval / Between-collection blank → "1 order" / "1 collection"**: only one event in the period, so no gap can be measured.
"""


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _load_ar_balance(zid: str, project: str) -> pd.DataFrame:
    df = Analytics("ar_due_ledger", zid=zid, project=project, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_cacus(zid: str) -> pd.DataFrame:
    df = Analytics("cacus_directory", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_final_items(zid: str) -> pd.DataFrame:
    """Load final_items_view for the given ZID."""
    df = Analytics("final_items_view", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_sales_alltime(zid: str, proj: str) -> pd.DataFrame:
    """All-time sales (no year filter) — used for inactive outreach last-order dates."""
    df = Analytics("sales", zid=zid, project=proj, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_sales_daily_alltime(zid: str, proj: str) -> pd.DataFrame:
    """sales_daily_item (daily item aggregates) — full history, no date filter.
    Used for trailing-12-month velocity in High Stock Marketing."""
    df = Analytics("sales_daily_item", zid=zid, project=proj, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_inv_overview(zid: str) -> pd.DataFrame:
    """inventory_overview for a given ZID — stock + std_price from caitem + opspprc."""
    df = Analytics("inventory_overview", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


def _resolve_packcode(df: pd.DataFrame) -> pd.DataFrame:
    """Add resolved_code column: packcode wins unless blank / NO / KH-prefix."""
    if df.empty:
        return df
    d = df.copy()
    d["resolved_code"] = d.apply(
        lambda r: (
            r["packcode"]
            if (
                r.get("packcode", "")
                and r["packcode"] not in ("", "NO")
                and not str(r["packcode"]).upper().startswith("KH")
            )
            else r["item_id"]
        ),
        axis=1,
    )
    return d


# ---------------------------------------------------------------------------
# formatting helpers — customer scoring table
# ---------------------------------------------------------------------------

def _fmt_currency(val) -> str:
    try:
        v = float(val)
        if abs(v) >= 1_000_000:
            return f"{v/1_000_000:.2f}M"
        if abs(v) >= 1_000:
            return f"{v/1_000:.1f}K"
        return f"{v:,.0f}"
    except Exception:
        return "—"


def _fmt_pct(v) -> str:
    if pd.isna(v):
        return ""
    if np.isinf(v):
        return "New ↑" if v > 0 else "New ↓"
    return f"{v:+.1f}%"


def _fmt_days(v, count_val, single_label: str) -> str:
    if pd.notna(v):
        return f"{v:.1f}"
    try:
        if pd.notna(count_val) and int(count_val) <= 1:
            return single_label
    except (TypeError, ValueError):
        pass
    return ""


# ---------------------------------------------------------------------------
# customer scoring sub-view
# ---------------------------------------------------------------------------

def _show_customer_scoring(result: pd.DataFrame):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers", f"{len(result):,}")
    m2.metric("Total Sales",       _fmt_currency(result.get("total_sales",       pd.Series(dtype=float)).sum()))
    m3.metric("Total Collection",  _fmt_currency(result.get("total_collection",  pd.Series(dtype=float)).sum()))
    bal_series = result.get("current_balance", pd.Series(dtype=float))
    if not isinstance(bal_series, pd.Series):
        bal_series = pd.Series(dtype=float)
    m4.metric("Outstanding Balance", _fmt_currency(bal_series.sum()))

    st.markdown("---")

    display_df = result.copy()

    for raw in _CURRENCY_COLS:
        lbl = _DISPLAY_LABELS[raw]
        if raw in display_df.columns:
            display_df[lbl] = display_df[raw].apply(
                lambda v: f"{v:,.0f}" if pd.notna(v) else ""
            )

    for raw in _PCT_COLS:
        lbl = _DISPLAY_LABELS[raw]
        if raw in display_df.columns:
            display_df[lbl] = display_df[raw].apply(_fmt_pct)

    _interval_map = {
        "avg_order_interval_days":      ("order_count",      "1 order"),
        "avg_days_between_collections": ("coll_event_count", "1 collection"),
        "avg_days_to_collection":       (None,               ""),
    }
    for raw, (count_raw, single_lbl) in _interval_map.items():
        lbl = _DISPLAY_LABELS[raw]
        if raw not in display_df.columns:
            continue
        if count_raw and count_raw in display_df.columns:
            display_df[lbl] = display_df.apply(
                lambda row, r=raw, cl=count_raw, sl=single_lbl:
                    _fmt_days(row[r], row[cl], sl),
                axis=1,
            )
        else:
            display_df[lbl] = display_df[raw].apply(
                lambda v: f"{v:.1f}" if pd.notna(v) else ""
            )

    if "composite_score" in display_df.columns:
        display_df[_DISPLAY_LABELS["composite_score"]] = display_df["composite_score"].apply(
            lambda v: f"{v:.1f}" if pd.notna(v) else ""
        )

    already_formatted = set(_DISPLAY_LABELS.keys())
    for raw, lbl in _DISPLAY_LABELS.items():
        if raw in display_df.columns and lbl not in display_df.columns:
            display_df = display_df.rename(columns={raw: lbl})

    cols_to_drop = (already_formatted | _HELPER_COLS) - set(_DISPLAY_LABELS.values())
    display_df = display_df.drop(columns=[c for c in cols_to_drop if c in display_df.columns])

    visible = [v for v in _DISPLAY_LABELS.values() if v in display_df.columns]
    display_df = display_df[visible]

    search = st.text_input("Search customer name or ID", "")
    if search:
        cname_lbl = _DISPLAY_LABELS["cusname"]
        cid_lbl   = _DISPLAY_LABELS["cusid"]
        mask = (
            display_df.get(cname_lbl, pd.Series(dtype=str)).astype(str).str.contains(search, case=False, na=False)
            | display_df.get(cid_lbl, pd.Series(dtype=str)).astype(str).str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    display_df = normalize_phone_cols(display_df)

    total_rows = len(display_df)
    cap = 50_000
    if total_rows > cap:
        st.info(f"Showing first {cap:,} of {total_rows:,} rows. Use Download for full data.")
        display_df = display_df.head(cap)

    st.dataframe(display_df, width="stretch")

    dl_df = result.drop(columns=[c for c in _HELPER_COLS if c in result.columns])
    for col in dl_df.select_dtypes(include=[float]).columns:
        dl_df[col] = dl_df[col].replace([np.inf, -np.inf], np.nan)
    dl_df = normalize_phone_cols(dl_df)
    csv = dl_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV",
        data=csv,
        file_name="marketing_analysis.csv",
        mime="text/csv",
    )

    with st.expander("📋 Column Definitions & Calculation Notes", expanded=False):
        st.markdown(_NOTES)


# ---------------------------------------------------------------------------
# area campaign planner sub-view
# ---------------------------------------------------------------------------

def _show_campaign_planner(
    result: pd.DataFrame,
    sales_df: pd.DataFrame,
    zid: str,
) -> None:
    # Salesman + area are already pre-filtered at the view level; pass None so the
    # processing helpers operate on the full (already-filtered) input DataFrames.

    # ── Section A: Top 10 customers by composite score ───────────────────────
    st.markdown("#### 📋 Top Customers to Contact")
    top_cus = build_area_campaign_top_customers(result)
    if top_cus.empty:
        st.info("No customers with a composite score for the current filters.")
    else:
        disp_cus = normalize_phone_cols(top_cus.copy())
        disp_cus["composite_score"] = disp_cus["composite_score"].apply(
            lambda v: f"{v:.1f}" if pd.notna(v) else ""
        )
        for c in ["total_sales", "total_collection"]:
            if c in disp_cus.columns:
                disp_cus[c] = disp_cus[c].apply(
                    lambda v: f"{v:,.0f}" if pd.notna(v) else ""
                )
        disp_cus = disp_cus.rename(columns={
            "cusid": "Customer ID", "cusname": "Customer Name",
            "cusmobile": "Mobile", "area": "Area", "spname": "Salesman",
            "composite_score": "Score", "total_sales": "Total Sales",
            "total_collection": "Total Collection",
        })
        st.dataframe(disp_cus, width="stretch")

    # ── Section B: Top 10 products by sales value ────────────────────────────
    st.markdown("#### 📦 Top Products (by Sales Value)")
    top_prod = build_area_top_products(sales_df)
    if top_prod.empty:
        st.info("No sales data for the current filters.")
    else:
        disp_prod = top_prod.copy()
        disp_prod["total_sales"] = disp_prod["total_sales"].apply(
            lambda v: f"{v:,.0f}" if pd.notna(v) else ""
        )
        disp_prod = disp_prod.rename(columns={
            "itemcode": "Item Code", "itemname": "Item Name", "itemgroup": "Group",
            "total_sales": "Total Sales", "transaction_count": "# Lines",
        })
        st.dataframe(disp_prod, width="stretch")

    # ── Combined download ────────────────────────────────────────────────────
    st.markdown("---")
    frames = []
    if not top_cus.empty:
        tc = normalize_phone_cols(top_cus.copy())
        tc.insert(0, "section", "Top Customers")
        frames.append(tc)
    if not top_prod.empty:
        tp = top_prod.copy()
        tp.insert(0, "section", "Top Products")
        frames.append(tp)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        csv = combined.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Campaign Report",
            data=csv,
            file_name="campaign_report.csv",
            mime="text/csv",
        )

    with st.expander("📋 How to use this report", expanded=False):
        st.markdown("""
**Section A — Top Customers to Contact**
Ranked by composite Score (0–100). These are the most valuable, most active customers in the
current salesman/area filter. Prioritise them for WhatsApp/phone calls promoting the products
in Section B.

**Section B — Top Products**
The products that have driven the most revenue in the current filter over the selected period.
Use these as the focus of your campaign message — they have proven demand here.
        """)


# ---------------------------------------------------------------------------
# inactive outreach sub-view
# ---------------------------------------------------------------------------

def _show_inactive_outreach(zid: str, proj: str, sales_raw: pd.DataFrame) -> None:
    # sales_raw is year-filtered (from data_dict) — same source as Customer Scoring
    # and Area Campaign Planner, so the salesman/area filter options match exactly.
    # All-time sales are loaded separately below for the inactive computation itself.

    if sales_raw.empty:
        st.warning("No sales data available for the selected filters.")
        return

    # ── Salesman + area filters — built from year-filtered sales_raw ──────────
    sp_opts = sorted(sales_raw["spname"].dropna().astype(str).unique().tolist())
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        sp_sel = st.selectbox("Salesman", ["All Salesmen"] + sp_opts, key="outreach_sp")
    area_pool = (
        sorted(sales_raw["area"].dropna().astype(str).unique().tolist())
        if sp_sel == "All Salesmen"
        else sorted(
            sales_raw[sales_raw["spname"].astype(str) == sp_sel]["area"]
            .dropna().astype(str).unique().tolist()
        )
    )
    with f_col2:
        area_sel = st.multiselect("Area", area_pool, default=area_pool, key="outreach_area")

    # ── Months slider ─────────────────────────────────────────────────────────
    months = st.slider(
        "Inactive for more than (months)", min_value=1, max_value=12, value=6,
        key="outreach_months",
    )

    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=months)
    st.caption(
        f"Customers with **no orders since {cutoff.strftime('%d %b %Y')}** "
        f"({months} month{'s' if months != 1 else ''} ago)"
    )

    # ── All-time sales: both ZIDs when 100001/100000 (shared sales team) ────
    _is_dual = zid in _DUAL_ZIDS
    with st.spinner("Loading all-time sales…"):
        sales_all = _load_sales_alltime(zid, proj)
        if "zid" not in sales_all.columns:
            sales_all = sales_all.copy()
            sales_all["zid"] = zid
        if _is_dual:
            _other_zid = _OTHER_ZID[zid]
            _other_all = _load_sales_alltime(_other_zid, proj)
            if not _other_all.empty:
                if "zid" not in _other_all.columns:
                    _other_all = _other_all.copy()
                    _other_all["zid"] = _other_zid
                sales_all = pd.concat([sales_all, _other_all], ignore_index=True)

    if sales_all.empty:
        st.warning("No sales data available.")
        return

    # Scope customers by area (the salesman's territory) but check inactivity
    # against company-wide purchases — any order from any salesman within
    # the window removes the customer, regardless of which ZID took the order.
    if area_sel:
        _area_cusids = set(sales_all[sales_all["area"].isin(area_sel)]["cusid"].unique())
        sales_for_inactive = sales_all[sales_all["cusid"].isin(_area_cusids)]
    else:
        sales_for_inactive = sales_all.copy()

    cacus_df = _load_cacus(zid)
    if _is_dual:
        _other_cacus = _load_cacus(_OTHER_ZID[zid])
        if not _other_cacus.empty:
            cacus_df = pd.concat([cacus_df, _other_cacus], ignore_index=True)

    inactive = build_inactive_customers(sales_for_inactive, cacus_df=cacus_df, months=months)

    # Pin the salesman column to the selected name so every row reads as
    # the calling salesman's responsibility.
    if sp_sel != "All Salesmen" and not inactive.empty and "spname" in inactive.columns:
        inactive["spname"] = sp_sel

    if inactive.empty:
        st.success(f"No customers inactive for more than {months} months — great retention!")
        return

    m1, m2 = st.columns(2)
    m1.metric("Inactive Customers", f"{len(inactive):,}")
    m2.metric(
        "Their Lifetime Sales",
        _fmt_currency(inactive.get("total_lifetime_sales", pd.Series(dtype=float)).sum()),
    )

    # Format display copy
    disp = inactive.copy()
    disp = normalize_phone_cols(disp)
    if "last_order_date" in disp.columns:
        disp["last_order_date"] = pd.to_datetime(disp["last_order_date"]).dt.strftime("%Y-%m-%d")
    if "total_lifetime_sales" in disp.columns:
        disp["total_lifetime_sales"] = disp["total_lifetime_sales"].apply(
            lambda v: f"{v:,.0f}" if pd.notna(v) else ""
        )

    rename_map = {
        "cusid":               "Customer ID",
        "zid":                 "ZID",
        "cusname":             "Customer Name",
        "cusmobile":           "Mobile",
        "whatsapp":            "WhatsApp",
        "area":                "Area",
        "spname":              "Salesman",
        "last_order_date":     "Last Order Date",
        "total_lifetime_sales":"Lifetime Sales",
    }
    disp = disp.rename(columns={k: v for k, v in rename_map.items() if k in disp.columns})
    visible = [v for v in rename_map.values() if v in disp.columns]
    disp = disp[visible]

    cap = 50_000
    if len(disp) > cap:
        st.info(f"Showing first {cap:,} of {len(disp):,} rows. Use Download for full list.")
        disp = disp.head(cap)

    st.dataframe(disp, width="stretch")

    # Download — mobile numbers as strings (comma check applied by normalize_phone_cols)
    dl = normalize_phone_cols(inactive.copy())
    if "last_order_date" in dl.columns:
        dl["last_order_date"] = pd.to_datetime(dl["last_order_date"]).dt.strftime("%Y-%m-%d")
    csv = dl.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Outreach List",
        data=csv,
        file_name="inactive_customers_outreach.csv",
        mime="text/csv",
    )

    with st.expander("📋 How to use this list", expanded=False):
        st.markdown(f"""
**What this list shows**
Customers who have made at least one purchase historically but have placed **no orders in the
last {months} months** (since {cutoff.strftime('%d %b %Y')}). Sorted by Last Order Date
descending — the most *recently* lapsed customers appear first, as they are the warmest leads
for re-engagement.

**Recommended outreach approach**
1. Filter by Salesman or Area to assign follow-up ownership.
2. Use the Mobile and WhatsApp columns to send a personalised message.
3. Reference their Lifetime Sales to tailor the tone — high-value lapsed customers deserve
   a personal call, not just a broadcast.
4. Download the list and share with your marketing team for bulk WhatsApp outreach.

**Tip**: Start with 3 months to catch recently-lapsed customers, then widen to 6–9 months
for a broader reactivation push.
        """)

    # ── Shared customer selector — drives both product history and call log ──
    st.markdown("---")
    _cus_opts_df = inactive[["cusid", "cusname"]].drop_duplicates("cusid")
    _cus_opts = {
        f"{row['cusname']} ({row['cusid']})": row["cusid"]
        for _, row in _cus_opts_df.iterrows()
    }
    _cus_sel = st.selectbox(
        "Select Customer",
        ["— pick a customer —"] + list(_cus_opts.keys()),
        key="outreach_cus_sel",
    )

    if _cus_sel and _cus_sel != "— pick a customer —":
        _sel_cusid = _cus_opts[_cus_sel]
        _sel_name  = _cus_sel.split(" (")[0]

        # ── Product purchase history ─────────────────────────────────────
        with st.expander(f"📦 Purchase History — {_sel_name}", expanded=True):
            _cus_sales = sales_all[sales_all["cusid"].astype(str) == str(_sel_cusid)].copy()
            if _cus_sales.empty:
                st.info("No purchase history found for this customer.")
            else:
                # Resolve qty and revenue column names
                _qty_col = "xqty" if "xqty" in _cus_sales.columns else (
                    "quantity" if "quantity" in _cus_sales.columns else None
                )
                _rev_col = "altsales" if "altsales" in _cus_sales.columns else (
                    "totalsales" if "totalsales" in _cus_sales.columns else None
                )
                _name_col = "itemname" if "itemname" in _cus_sales.columns else (
                    "xdesc" if "xdesc" in _cus_sales.columns else None
                )

                _grp_cols = ["itemcode"]
                if _name_col:
                    _grp_cols.append(_name_col)

                _agg: dict = {}
                if _qty_col:
                    _agg[_qty_col] = "sum"
                if _rev_col:
                    _agg[_rev_col] = "sum"

                if _agg:
                    _prod_df = (
                        _cus_sales.groupby(_grp_cols, as_index=False)
                        .agg(_agg)
                        .sort_values(_rev_col if _rev_col else list(_agg.keys())[0], ascending=False)
                        .reset_index(drop=True)
                    )

                    _prod_rename = {"itemcode": "Item Code"}
                    if _name_col:
                        _prod_rename[_name_col] = "Item Name"
                    if _qty_col:
                        _prod_rename[_qty_col] = "Qty"
                    if _rev_col:
                        _prod_rename[_rev_col] = "Total Value"

                    _prod_disp = _prod_df.rename(columns=_prod_rename)
                    if "Total Value" in _prod_disp.columns:
                        _prod_disp["Total Value"] = _prod_disp["Total Value"].apply(
                            lambda v: f"{v:,.0f}" if pd.notna(v) else ""
                        )
                    if "Qty" in _prod_disp.columns:
                        _prod_disp["Qty"] = _prod_disp["Qty"].apply(
                            lambda v: f"{v:,.0f}" if pd.notna(v) else ""
                        )

                    st.dataframe(_prod_disp, width="stretch", hide_index=True)
                else:
                    st.info("Sales value columns not available in this dataset.")

        # ── Call log panel — right below the purchase history ────────────
        st.markdown("#### 📞 Log a Call")
        st.caption(
            "Call logs are shared with Customer Support. "
            "Every entry records who placed the call (your login username)."
        )
        _render_call_log_panel(
            cusid=_sel_cusid,
            zid=zid,
            customer_name=_sel_name,
            key_suffix="_outreach",
        )


# ---------------------------------------------------------------------------
# media library — file helpers
# ---------------------------------------------------------------------------

_IMG_BASE  = Path("data/product_images")
_IMG_EXTS  = {".jpg", ".jpeg", ".png", ".webp"}
_GALLERY_N = 3   # images per row


def _img_folder(code: str) -> Path:
    return _IMG_BASE / code


def _img_sort_key(p: Path, code: str) -> int:
    """Primary image (no numeric suffix) sorts first; _2, _3 … follow in order."""
    stem = p.stem
    if stem == code:
        return 0
    m = re.match(rf"^{re.escape(code)}_(\d+)$", stem)
    return int(m.group(1)) if m else 9999


def _scan_images(folder: Path, code: str) -> list:
    if not folder.exists():
        return []
    imgs = [f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in _IMG_EXTS]
    return sorted(imgs, key=lambda p: _img_sort_key(p, code))


def _next_img_path(folder: Path, code: str, ext: str) -> Path:
    """Return the next available path for a new image, respecting the naming convention."""
    primary_exists = any((folder / f"{code}{e}").exists() for e in _IMG_EXTS)
    if not primary_exists:
        return folder / f"{code}{ext}"
    n = 2
    while any((folder / f"{code}_{n}{e}").exists() for e in _IMG_EXTS):
        n += 1
    return folder / f"{code}_{n}{ext}"


# ---------------------------------------------------------------------------
# media library sub-view
# ---------------------------------------------------------------------------

def _show_media_library(zid: str) -> None:
    st.markdown("#### 🖼️ Product Media Library")

    with st.spinner("Loading product list…"):
        items_df = _load_final_items(zid)

    if items_df.empty:
        st.warning("No product data available.")
        return

    # ── Product list in collapsible expander ─────────────────────────────
    with st.expander("📋 Full Product List", expanded=False):
        tbl = items_df.copy().rename(columns={
            "item_id": "Item Code", "item_name": "Item Name",
            "item_group": "Group",  "stock": "Stock",
        })
        tbl["Images"] = tbl["Item Code"].apply(
            lambda c: "✓" if _scan_images(_img_folder(str(c)), str(c)) else "—"
        )
        show_cols = [c for c in ["Item Code", "Item Name", "Group", "Stock", "Images"]
                     if c in tbl.columns]
        st.dataframe(tbl[show_cols], width="stretch", hide_index=True)

    # ── Product selector ─────────────────────────────────────────────────
    item_opts = (
        items_df.apply(lambda r: f"{r['item_id']} — {r['item_name']}", axis=1).tolist()
    )
    sel_label = st.selectbox(
        "Select product",
        ["— pick a product —"] + item_opts,
        key="ml_product_sel",
    )
    if not sel_label or sel_label == "— pick a product —":
        return

    sel_code = sel_label.split(" — ")[0]
    sel_name = items_df.loc[items_df["item_id"] == sel_code, "item_name"].iloc[0]
    folder   = _img_folder(sel_code)
    images   = _scan_images(folder, sel_code)

    st.markdown(f"**{sel_name}** &nbsp;·&nbsp; `{sel_code}`", unsafe_allow_html=True)
    st.markdown("---")

    # ── Gallery ──────────────────────────────────────────────────────────
    if not images:
        st.markdown(
            '<div style="display:inline-flex;width:220px;height:220px;'
            'background:#F2F3F4;border:2px dashed #AEB6BF;border-radius:8px;'
            'align-items:center;justify-content:center;color:#AEB6BF;font-size:13px;">'
            'No images yet</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
    else:
        cols = st.columns(_GALLERY_N)
        for i, img_path in enumerate(images):
            label = sel_code if i == 0 else f"{sel_code}_{i + 1}"
            with cols[i % _GALLERY_N]:
                with open(img_path, "rb") as fh:
                    st.image(fh.read(), caption=label, width="stretch")

    # ── Upload new images ─────────────────────────────────────────────────
    st.markdown("#### ⬆️ Upload Images")
    uploaded = st.file_uploader(
        "Drag and drop or browse — JPG, PNG, WebP",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key=f"ml_upload_{sel_code}",
        label_visibility="visible",
    )
    if uploaded:
        if st.button("💾 Save to Library", key=f"ml_save_{sel_code}"):
            folder.mkdir(parents=True, exist_ok=True)
            saved, skipped = [], []
            for uf in uploaded:
                ext = Path(uf.name).suffix.lower()
                if ext not in _IMG_EXTS:
                    skipped.append(uf.name)
                    continue
                dest = _next_img_path(folder, sel_code, ext)
                dest.write_bytes(uf.getvalue())
                saved.append(dest.name)
            if saved:
                st.success(f"Saved: {', '.join(saved)}")
            if skipped:
                st.warning(f"Skipped (unsupported format): {', '.join(skipped)}")
            if saved:
                st.rerun()

    # ── Series copy (only if this item already has images) ───────────────
    if images:
        st.markdown("#### 🔗 Copy to Related Series")
        st.caption(
            "Same product, different sizes or variants? "
            "Enter the related item codes — the images above will be copied into "
            "each code's folder with the matching file-naming convention."
        )
        series_input = st.text_input(
            "Related item codes (comma-separated)",
            placeholder="e.g. ITEM002, ITEM003, ITEM004",
            key=f"ml_series_{sel_code}",
        )
        overwrite = st.checkbox(
            "Overwrite existing images in target folders",
            value=False,
            key=f"ml_overwrite_{sel_code}",
        )
        if st.button("📋 Copy Images to Series", key=f"ml_copy_{sel_code}"):
            if not series_input.strip():
                st.warning("Enter at least one related item code.")
            else:
                dest_codes = [c.strip() for c in series_input.split(",") if c.strip()]
                for dest_code in dest_codes:
                    dest_folder = _img_folder(dest_code)
                    dest_folder.mkdir(parents=True, exist_ok=True)
                    n_copied = n_skipped = 0
                    for i, src_img in enumerate(images):
                        dest_name = (
                            f"{dest_code}{src_img.suffix}"
                            if i == 0
                            else f"{dest_code}_{i + 1}{src_img.suffix}"
                        )
                        dest_path = dest_folder / dest_name
                        if dest_path.exists() and not overwrite:
                            n_skipped += 1
                        else:
                            shutil.copy2(src_img, dest_path)
                            n_copied += 1
                    msg = f"**{dest_code}**: {n_copied} image(s) copied"
                    if n_skipped:
                        msg += f", {n_skipped} skipped (already exist)"
                    st.success(msg)


# ---------------------------------------------------------------------------
# high stock marketing sub-view
# ---------------------------------------------------------------------------

def _show_high_stock_marketing(zid: str, proj: str) -> None:
    """All items with positive stock — avg monthly sales, days to clear, std price.
    For ZID 100001: combines 100001 + 100009 stock via packcode cross-ZID merge.
    For other ZIDs: loads inventory directly for that entity."""

    with st.spinner("Loading inventory and sales velocity…"):
        if zid == "100001":
            inv_001 = _resolve_packcode(_load_inv_overview("100001").copy())
            inv_009 = _resolve_packcode(_load_inv_overview("100009").copy())

            combined = pd.concat([inv_001, inv_009], ignore_index=True)
            stock_agg = combined.groupby("resolved_code", as_index=False)["stock"].sum()

            _meta_cols = ["item_name", "item_group", "std_price"]
            m1 = (
                inv_001[["resolved_code"] + [c for c in _meta_cols if c in inv_001.columns]]
                .drop_duplicates("resolved_code")
                if not inv_001.empty else pd.DataFrame()
            )
            m2 = (
                inv_009[["resolved_code"] + [c for c in _meta_cols if c in inv_009.columns]]
                .drop_duplicates("resolved_code")
                if not inv_009.empty else pd.DataFrame()
            )
            if not m1.empty and not m2.empty:
                meta = pd.concat([m1, m2]).drop_duplicates("resolved_code", keep="first")
            elif not m1.empty:
                meta = m1
            else:
                meta = m2

            inv_df = stock_agg.merge(meta, on="resolved_code", how="left").rename(
                columns={"resolved_code": "item_id"}
            )

            # Map sales itemcode → resolved_code for velocity merge
            code_map: dict = {}
            if not inv_001.empty:
                for _, row in inv_001[["item_id", "resolved_code"]].drop_duplicates().iterrows():
                    code_map[row["item_id"]] = row["resolved_code"]
        else:
            inv_df = _load_inv_overview(zid).copy()
            code_map = {}

        sales_daily = _load_sales_daily_alltime(zid, proj)

    if inv_df.empty:
        st.warning("No inventory data available.")
        return

    # ── Filter to positive stock ──────────────────────────────────────────
    inv_df = inv_df[inv_df["stock"] > 0].copy()

    # ── Trailing 12-month avg monthly sales ──────────────────────────────
    _ROLLING_MONTHS = 12
    num_months = 0

    if not sales_daily.empty and "itemcode" in sales_daily.columns and "quantity" in sales_daily.columns:
        sd = sales_daily.copy()
        sd["date"] = pd.to_datetime(sd["date"], errors="coerce")
        sd = sd.dropna(subset=["date"])
        cutoff = pd.Timestamp.today() - pd.DateOffset(months=_ROLLING_MONTHS)
        sd = sd[sd["date"] >= cutoff]

        sd["_key"] = sd["itemcode"].map(code_map).fillna(sd["itemcode"]) if code_map else sd["itemcode"]

        num_months = max(sd["date"].dt.to_period("M").nunique(), 1)
        vel = (
            sd.groupby("_key", as_index=False)["quantity"].sum()
            .assign(avg_monthly_sales=lambda d: d["quantity"] / num_months)
            .rename(columns={"_key": "item_id"})
            [["item_id", "avg_monthly_sales"]]
        )
        inv_df = inv_df.merge(vel, on="item_id", how="left")
    else:
        inv_df["avg_monthly_sales"] = 0.0

    inv_df["avg_monthly_sales"] = inv_df["avg_monthly_sales"].fillna(0.0)

    # ── Days to clear ─────────────────────────────────────────────────────
    inv_df["days_to_clear"] = inv_df.apply(
        lambda r: round(r["stock"] / r["avg_monthly_sales"] * 30, 1)
        if r["avg_monthly_sales"] > 0 else None,
        axis=1,
    )

    inv_df = inv_df.sort_values("days_to_clear", ascending=False, na_position="last").reset_index(drop=True)

    # ── Caption ───────────────────────────────────────────────────────────
    entity_label = "100001 + 100009 (cross-ZID)" if zid == "100001" else zid
    vel_note = (
        f"trailing {num_months} month(s) with data (≤ 12)"
        if num_months > 0 else "no sales data in trailing 12 months"
    )
    st.caption(
        f"**{len(inv_df)}** items with positive stock — {entity_label}. "
        f"Avg Monthly Sales = {vel_note}. "
        f"Days to Clear = Stock ÷ Avg Monthly Sales × 30 (— = no recent sales)."
    )

    # ── Search + table ────────────────────────────────────────────────────
    search = st.text_input("🔍 Search item", "", key="hs_search")
    disp_df = inv_df.copy()
    if search:
        mask = (
            disp_df.get("item_name", pd.Series(dtype=str)).astype(str).str.contains(search, case=False, na=False)
            | disp_df.get("item_id", pd.Series(dtype=str)).astype(str).str.contains(search, case=False, na=False)
            | disp_df.get("item_group", pd.Series(dtype=str)).astype(str).str.contains(search, case=False, na=False)
        )
        disp_df = disp_df[mask]

    disp_inv = disp_df[
        [c for c in ["item_id", "item_name", "item_group", "stock",
                      "avg_monthly_sales", "days_to_clear", "std_price"]
         if c in disp_df.columns]
    ].rename(columns={
        "item_id": "Item Code", "item_name": "Item Name", "item_group": "Group",
        "stock": "Stock", "avg_monthly_sales": "Avg Monthly Sales",
        "days_to_clear": "Days to Clear", "std_price": "Std Price",
    })

    st.dataframe(
        disp_inv.style.format({
            "Stock":             "{:,.0f}",
            "Avg Monthly Sales": "{:,.1f}",
            "Days to Clear":     "{:,.1f}",
            "Std Price":         "{:,.2f}",
        }, na_rep="—"),
        width="stretch",
        hide_index=True,
    )

    # ── Product drill-down ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Customer / Area Breakdown")

    # Product selector uses the full (pre-search) list
    product_opts = (
        inv_df.apply(
            lambda r: f"{r['item_id']} — {r.get('item_name', r['item_id'])}",
            axis=1,
        ).tolist()
    )
    sel_label = st.selectbox(
        "Select a product to see who bought it",
        ["— pick a product —"] + product_opts,
        key="hs_product_sel",
    )
    if not sel_label or sel_label == "— pick a product —":
        return

    sel_code = sel_label.split(" — ")[0]
    sel_name = (
        inv_df.loc[inv_df["item_id"] == sel_code, "item_name"].iloc[0]
        if "item_name" in inv_df.columns and not inv_df.loc[inv_df["item_id"] == sel_code].empty
        else sel_code
    )

    with st.spinner("Loading customer history…"):
        sales_all = _load_sales_alltime(zid, proj)

    if sales_all.empty:
        st.info("No sales history available.")
        return

    sales_all["date"] = pd.to_datetime(sales_all["date"], errors="coerce")
    sales_all["altsales"] = pd.to_numeric(sales_all["altsales"], errors="coerce").fillna(0)

    item_sales = sales_all[sales_all["itemcode"].astype(str) == str(sel_code)].copy()

    if item_sales.empty:
        st.info(f"No historical sales found for **{sel_name}**.")
        return

    today = pd.Timestamp.today().normalize()

    last_any = (
        sales_all.dropna(subset=["date"])
        .groupby("cusid")["date"].max()
        .reset_index()
        .rename(columns={"date": "_last_any"})
    )
    last_any["Months Since Last Order (Any)"] = (
        (today - last_any["_last_any"]).dt.days / 30.44
    ).round(1)

    view_mode = st.radio(
        "View by",
        ["👤 Customer", "📍 Area"],
        horizontal=True,
        key="hs_view_mode",
    )

    if view_mode == "📍 Area":
        area_agg = (
            item_sales.groupby("area", as_index=False)["altsales"]
            .sum()
            .rename(columns={"area": "Area", "altsales": "Net Sales"})
            .sort_values("Net Sales", ascending=False)
            .reset_index(drop=True)
        )
        st.caption(f"**{len(area_agg)}** area(s) with sales of **{sel_name}**")
        st.dataframe(
            area_agg.style.format({"Net Sales": "{:,.0f}"}),
            width="stretch",
            hide_index=True,
        )

    else:
        cust_agg = (
            item_sales.dropna(subset=["date"])
            .groupby("cusid", as_index=False).agg(
                cusname        =("cusname",   "first"),
                cusmobile      =("cusmobile", "first"),
                area           =("area",      "first"),
                net_sales      =("altsales",  "sum"),
                last_item_date =("date",      "max"),
            )
        )
        cust_agg["Months Since Last Purchase (This Product)"] = (
            (today - cust_agg["last_item_date"]).dt.days / 30.44
        ).round(1)

        cust_agg = cust_agg.merge(
            last_any[["cusid", "Months Since Last Order (Any)"]],
            on="cusid", how="left",
        ).sort_values("net_sales", ascending=False).reset_index(drop=True)

        disp_cust = normalize_phone_cols(cust_agg.copy()).rename(columns={
            "cusid":     "Cust Code",
            "cusname":   "Customer",
            "cusmobile": "Mobile",
            "area":      "Area",
            "net_sales": "Net Sales",
        })

        show_cols = [c for c in [
            "Cust Code", "Customer", "Mobile", "Area", "Net Sales",
            "Months Since Last Purchase (This Product)",
            "Months Since Last Order (Any)",
        ] if c in disp_cust.columns]

        st.caption(f"**{len(disp_cust)}** customer(s) previously bought **{sel_name}**")
        st.dataframe(
            disp_cust[show_cols].style.format({
                "Net Sales": "{:,.0f}",
                "Months Since Last Purchase (This Product)": "{:.1f}",
                "Months Since Last Order (Any)": "{:.1f}",
            }, na_rep="—"),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "⬇ Download Customer List",
            data=normalize_phone_cols(cust_agg.copy()).drop(
                columns=["last_item_date", "_last_any"], errors="ignore"
            ).to_csv(index=False).encode("utf-8"),
            file_name=f"high_stock_customers_{sel_code}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# leads sub-view — CRM: full access. Sales: read-only Table 1 only.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def _load_marketing_leads(zid: str) -> pd.DataFrame:
    from core.queries import get_marketing_leads
    from core.db import get_data
    sql, params = get_marketing_leads(zid)
    records, cols = get_data(sql, *params)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records, columns=cols)


@st.cache_data(show_spinner=False, ttl=3600)
def _load_cacus_lead_links(zid: str) -> pd.DataFrame:
    df = Analytics("cacus_lead_links", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


def _bulk_insert_leads(parsed_df: pd.DataFrame, zid: str, uploaded_by: str) -> int:
    """Insert a parsed upload batch. Returns the number of NEW leads inserted.

    Dedup happens here in Python, not via ON CONFLICT DO NOTHING — that clause
    needs Postgres 9.5+ and this server predates it (confirmed: it errors with
    'syntax error at or near "ON"'). So existing fb_lead_ids for this ZID are
    fetched first and matching rows are dropped before the insert; a
    re-uploaded export still ends up a safe no-op, just decided here instead
    of at the DB.
    """
    from core.db import get_data, execute_values_insert
    from core.queries import get_existing_lead_fb_ids, insert_marketing_leads_sql

    if parsed_df.empty:
        return 0

    sql, params = get_existing_lead_fb_ids(zid)
    records, _ = get_data(sql, *params)
    if records is None:
        return -1  # DB error fetching existing ids — surface as a failure, not "0 new"
    existing_ids = {str(r[0]) for r in records}

    df = parsed_df[~parsed_df["fb_lead_id"].astype(str).isin(existing_ids)].copy()
    if df.empty:
        return 0

    # Native Python datetime — safest for the psycopg2 adapter (avoid passing
    # pandas Timestamp objects straight through execute_values).
    df["created_time"] = df["created_time"].apply(
        lambda v: v.to_pydatetime() if isinstance(v, pd.Timestamp) and pd.notna(v) else None
    )

    rows = [
        (zid,) + tuple(r) + (uploaded_by,)
        for r in df.itertuples(index=False, name=None)
    ]
    # Do NOT clamp to 0 here — execute_values_insert returns -1 on a DB error,
    # and callers rely on that negative sentinel to show "Upload failed" instead
    # of silently reporting "0 new leads saved" for a real failure (e.g. the
    # tables not existing yet on this server).
    return execute_values_insert(insert_marketing_leads_sql(), rows)


def _show_lead_upload(zid: str) -> None:
    st.caption(
        "Facebook Lead Ads CSV/Excel export. The `id` column is required — it's the "
        "join key used to detect when a lead converts to a customer (staff paste it "
        "into the customer's URL field in the ERP once that happens)."
    )

    with st.expander("📄 Need a blank template instead?", expanded=False):
        st.caption(
            "A clean, English-only column set to fill in by hand — use this instead of "
            "a raw platform export if you're compiling leads from another source. **`id`** "
            "must be unique per lead (use the lead's phone number, a Facebook lead id if "
            "you have one, or just make one up e.g. `LEAD-0001`, `LEAD-0002`, ...). Every "
            "other column can be left blank if you don't know it — just keep the column "
            "names exactly as they are. Don't add, rename, or translate any column; an "
            "unrecognized column (e.g. a form question in Bengali) gets tucked away as "
            "extra metadata instead of showing up in the leads table."
        )
        st.download_button(
            "⬇ Download Sample Template (CSV)",
            data=build_leads_upload_template().to_csv(index=False).encode("utf-8"),
            file_name="marketing_leads_upload_template.csv",
            mime="text/csv",
            key="leads_template_dl",
        )

    uploaded = st.file_uploader(
        "Drag and drop or browse — CSV or Excel",
        type=["csv", "xlsx", "xls"],
        key="leads_upload",
    )
    if uploaded is None:
        return

    try:
        # dtype=str for every column -- otherwise pandas infers numeric-looking
        # columns (work_phone_number, id) as int64 and silently drops leading
        # zeros, corrupting every Bangladeshi phone number ("01711234567" ->
        # "1711234567"). Blank cells still read as real NaN with dtype=str, so
        # this doesn't change how missing values are handled downstream.
        if uploaded.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded, dtype=str)
        else:
            raw_df = pd.read_excel(uploaded, dtype=str)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return

    try:
        parsed_df = parse_leads_upload(raw_df)
    except ValueError as e:
        st.error(str(e))
        return

    if parsed_df.empty:
        st.warning("No valid lead rows found in this file.")
        return

    st.info(f"**{len(parsed_df):,}** lead row(s) found in the file.")
    with st.expander("Preview first 5 rows", expanded=False):
        st.dataframe(parsed_df.head(5), width="stretch", hide_index=True)

    if st.button("💾 Save to Leads Table", key="leads_upload_save"):
        n_new = _bulk_insert_leads(parsed_df, zid, st.session_state.get("username", ""))
        if n_new < 0:
            st.error(
                "Upload failed — no rows were saved. Check the server logs for an "
                "'execute_values_insert error' line (usually means the marketing_leads "
                "tables haven't been created on this DB yet — see "
                "db/sql_scripts/create_marketing_leads_tables.sql)."
            )
        else:
            n_dupe = len(parsed_df) - n_new
            msg = f"**{n_new:,}** new lead(s) saved."
            if n_dupe:
                msg += f" **{n_dupe:,}** already existed and were skipped."
            st.success(msg)
            _load_marketing_leads.clear()
            st.rerun()


_LEAD_STAGES = ["New", "Contacted", "Qualified", "Follow-up", "Converted", "Not Interested"]

# Column order here MUST match core/queries.py::update_marketing_lead_sql's
# SET list exactly -- _update_lead builds its params tuple positionally off
# this list, not by name.
_LEAD_UPDATE_COLS = [
    "full_name", "company_name", "work_phone_number", "job_title",
    "street_address", "area", "lead_stage",
    "ad_id", "ad_name", "adset_id", "adset_name",
    "campaign_id", "campaign_name", "form_id", "form_name",
    "is_organic", "platform", "inbox_url", "lead_status",
    "lead_cost", "created_time",
]


def _render_lead_fields(prefix: str, defaults: dict | None = None, show_stage: bool = False) -> dict:
    """Render the full marketing_leads field set as form inputs -- shared by
    _show_manual_lead_entry (defaults=None, blank form) and _show_edit_lead
    (defaults=the lead's current row, show_stage=True). Must be called
    inside an st.form(...) block; returns the raw widget values, still
    strings/labels at this point -- pass through _parse_lead_fields before
    using them.
    """
    d = defaults or {}

    def _s(col: str) -> str:
        v = d.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v)

    c1, c2 = st.columns(2)
    full_name    = c1.text_input("Full Name*", value=_s("full_name"), key=f"{prefix}_full_name")
    company_name = c2.text_input("Company Name", value=_s("company_name"), key=f"{prefix}_company_name")
    c3, c4 = st.columns(2)
    phone     = c3.text_input("Phone Number*", value=_s("work_phone_number"), key=f"{prefix}_phone")
    job_title = c4.text_input("Job Title", value=_s("job_title"), key=f"{prefix}_job_title")
    c5, c6 = st.columns(2)
    address = c5.text_input("Address", value=_s("street_address"), key=f"{prefix}_address")
    area    = c6.text_input("Area", value=_s("area"), placeholder="e.g. Dhanmondi, Dhaka", key=f"{prefix}_area")
    c7, c8 = st.columns(2)
    platform    = c7.text_input("Platform", value=_s("platform") or "Manual", key=f"{prefix}_platform")
    lead_status = c8.text_input("Lead Status", value=_s("lead_status"), key=f"{prefix}_lead_status")
    c9, c10 = st.columns(2)
    _organic_opts = ["— Unknown —", "Yes", "No"]
    _organic_default = {True: "Yes", False: "No"}.get(d.get("is_organic"), "— Unknown —")
    is_organic_label = c9.selectbox(
        "Organic Lead?", _organic_opts, index=_organic_opts.index(_organic_default), key=f"{prefix}_is_organic",
    )
    lead_cost_str = c10.text_input(
        "Lead Cost", value=_s("lead_cost"), placeholder="e.g. 150", key=f"{prefix}_lead_cost",
    )

    raw_created = d.get("created_time")
    created_default = None
    if raw_created is not None and pd.notna(raw_created):
        created_default = pd.to_datetime(raw_created).date()
    created_date = st.date_input(
        "Created Date", value=created_default,
        help="Leave blank to use the current date/time.", key=f"{prefix}_created_date",
    )

    stage = None
    if show_stage:
        current_stage = d.get("lead_stage") or "New"
        stage_opts = _LEAD_STAGES + ([current_stage] if current_stage not in _LEAD_STAGES else [])
        stage = st.selectbox(
            "Lead Stage", stage_opts, index=stage_opts.index(current_stage), key=f"{prefix}_stage",
        )

    st.markdown("**Campaign / Ad Info** *(optional — usually only set for platform-sourced leads)*")
    c11, c12 = st.columns(2)
    ad_id   = c11.text_input("Ad ID", value=_s("ad_id"), key=f"{prefix}_ad_id")
    ad_name = c12.text_input("Ad Name", value=_s("ad_name"), key=f"{prefix}_ad_name")
    c13, c14 = st.columns(2)
    adset_id   = c13.text_input("Adset ID", value=_s("adset_id"), key=f"{prefix}_adset_id")
    adset_name = c14.text_input("Adset Name", value=_s("adset_name"), key=f"{prefix}_adset_name")
    c15, c16 = st.columns(2)
    campaign_id   = c15.text_input("Campaign ID", value=_s("campaign_id"), key=f"{prefix}_campaign_id")
    campaign_name = c16.text_input("Campaign Name", value=_s("campaign_name"), key=f"{prefix}_campaign_name")
    c17, c18 = st.columns(2)
    form_id   = c17.text_input("Form ID", value=_s("form_id"), key=f"{prefix}_form_id")
    form_name = c18.text_input("Form Name", value=_s("form_name"), key=f"{prefix}_form_name")
    inbox_url = st.text_input("Inbox URL", value=_s("inbox_url"), key=f"{prefix}_inbox_url")

    return {
        "full_name": full_name, "company_name": company_name, "phone": phone,
        "job_title": job_title, "address": address, "area": area,
        "platform": platform, "lead_status": lead_status,
        "is_organic_label": is_organic_label, "lead_cost_str": lead_cost_str,
        "created_date": created_date, "stage": stage,
        "ad_id": ad_id, "ad_name": ad_name, "adset_id": adset_id, "adset_name": adset_name,
        "campaign_id": campaign_id, "campaign_name": campaign_name,
        "form_id": form_id, "form_name": form_name, "inbox_url": inbox_url,
    }


def _parse_lead_fields(raw: dict) -> dict:
    """Convert _render_lead_fields' raw widget output into typed values ready
    for build_manual_lead_row / _update_lead. Raises ValueError with a
    user-facing message if Lead Cost isn't a valid number."""
    out = {k: (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
    out["is_organic"] = {"Yes": True, "No": False}.get(raw["is_organic_label"])

    lead_cost = None
    if raw["lead_cost_str"].strip():
        try:
            lead_cost = float(raw["lead_cost_str"].strip())
        except ValueError:
            raise ValueError("Lead Cost must be a number (e.g. 150 or 150.50).")
    out["lead_cost"] = lead_cost

    created_date = raw.get("created_date")
    out["created_time"] = pd.Timestamp(created_date, tz="UTC") if created_date else None
    return out


def _show_manual_lead_entry(zid: str) -> None:
    st.caption(
        "For leads that come in by phone or walk-in rather than a platform export. "
        "Only Full Name and Phone Number are required — every other field mirrors "
        "the marketing_leads table and can be left blank. A generated lead id is "
        "used the same way as a Facebook lead id — paste it into the customer's "
        "URL field in the ERP if you want conversion tracked."
    )
    with st.form("manual_lead_form", clear_on_submit=True):
        fields = _render_lead_fields("new_lead", defaults=None, show_stage=False)
        notes = st.text_area("Notes", placeholder="Any additional context about this lead")
        submitted = st.form_submit_button("💾 Save Lead")

    if not submitted:
        return

    if not fields["full_name"].strip() or not fields["phone"].strip():
        st.error("Full Name and Phone Number are required.")
        return

    try:
        parsed = _parse_lead_fields(fields)
    except ValueError as e:
        st.error(str(e))
        return

    row_df = build_manual_lead_row(
        full_name=parsed["full_name"], work_phone_number=parsed["phone"],
        company_name=parsed["company_name"], job_title=parsed["job_title"],
        street_address=parsed["address"], area=parsed["area"], notes=notes,
        lead_cost=parsed["lead_cost"], created_time=parsed["created_time"],
        ad_id=parsed["ad_id"], ad_name=parsed["ad_name"],
        adset_id=parsed["adset_id"], adset_name=parsed["adset_name"],
        campaign_id=parsed["campaign_id"], campaign_name=parsed["campaign_name"],
        form_id=parsed["form_id"], form_name=parsed["form_name"],
        is_organic=parsed["is_organic"], platform=parsed["platform"],
        inbox_url=parsed["inbox_url"], lead_status=parsed["lead_status"],
    )
    n_new = _bulk_insert_leads(row_df, zid, st.session_state.get("username", ""))
    if n_new == 1:
        st.success(f"Lead saved: **{parsed['full_name']}**.")
        _load_marketing_leads.clear()
        st.rerun()
    else:
        st.error(
            "Failed to save — check the server logs for an 'execute_values_insert error' "
            "line (usually means the marketing_leads tables haven't been created on this "
            "DB yet — see db/sql_scripts/create_marketing_leads_tables.sql)."
        )


# Optional text columns where a blank field should store NULL, not "" --
# matches build_manual_lead_row's _blank_to_none convention for the INSERT
# path, so clearing a field in Edit behaves the same as leaving it blank
# when creating a lead.
_LEAD_BLANK_TO_NONE_COLS = [
    "company_name", "street_address", "area", "job_title", "inbox_url",
    "ad_id", "ad_name", "adset_id", "adset_name",
    "campaign_id", "campaign_name", "form_id", "form_name",
]


def _update_lead(lead_id: int, zid: str, parsed: dict) -> bool:
    from core.db import execute_write
    from core.queries import update_marketing_lead_sql
    values = {
        **parsed,
        "work_phone_number": parsed["phone"],
        "street_address": parsed["address"],
        "lead_stage": parsed["stage"],
    }
    for col in _LEAD_BLANK_TO_NONE_COLS:
        if not values.get(col):
            values[col] = None
    # platform/lead_status fall back to "manual" rather than NULL, same as
    # build_manual_lead_row -- there's no meaningful "unset" state for these
    # beyond that default.
    values["platform"] = values.get("platform") or "manual"
    values["lead_status"] = values.get("lead_status") or "manual"

    params = tuple(values[c] for c in _LEAD_UPDATE_COLS) + (lead_id, zid)
    return execute_write(update_marketing_lead_sql(), params)


def _show_edit_lead(zid: str) -> None:
    st.caption(
        "Edit a lead's details after it's been saved — from a bulk upload or the "
        "single-lead form. The Lead ID (and Facebook Lead ID, where one exists) "
        "never changes here — it's the join key used for conversion tracking and "
        "call-log history."
    )
    leads_df = _load_marketing_leads(zid)
    if leads_df.empty:
        st.info("No leads yet — switch to **Bulk Upload** or **Single Lead** to add one.")
        return

    lead_opts = {
        f"{r['full_name']} — {r['company_name']} (#{r['id']})": int(r["id"])
        for _, r in leads_df[["id", "full_name", "company_name"]].fillna("").iterrows()
    }
    sel_label = st.selectbox(
        "Select lead to edit",
        ["— pick a lead —"] + list(lead_opts.keys()),
        key="leads_edit_sel",
    )
    if not sel_label or sel_label == "— pick a lead —":
        return

    sel_id = lead_opts[sel_label]
    row = leads_df[leads_df["id"] == sel_id].iloc[0]

    st.caption(f"Lead ID: **#{sel_id}** (fixed) · FB Lead ID: `{row.get('fb_lead_id') or '—'}` (fixed)")

    with st.form(f"edit_lead_form_{sel_id}"):
        fields = _render_lead_fields(f"edit_lead_{sel_id}", defaults=row.to_dict(), show_stage=True)
        submitted = st.form_submit_button("💾 Save Changes")

    if not submitted:
        return
    if not fields["full_name"].strip() or not fields["phone"].strip():
        st.error("Full Name and Phone Number are required.")
        return

    try:
        parsed = _parse_lead_fields(fields)
    except ValueError as e:
        st.error(str(e))
        return

    ok = _update_lead(sel_id, zid, parsed)
    if ok:
        st.success(f"Lead #{sel_id} updated.")
        _load_marketing_leads.clear()
        st.rerun()
    else:
        st.error(
            "Failed to save changes — check the server logs for an 'execute_write "
            "error' line."
        )


def _render_leads_table(zid: str, leads_df: pd.DataFrame, links_df: pd.DataFrame,
                         call_logs_df: pd.DataFrame, is_crm: bool) -> None:
    """Table 1 — individual lead + latest call info. Shared by the sales
    read-only path and the CRM 'Call Log' tab (leads table shown first there)."""
    if leads_df.empty:
        st.info("No leads uploaded yet.")
        return

    st.markdown("#### 📋 Leads")
    summary = build_lead_summary_table(leads_df, links_df, call_logs_df)

    search = st.text_input("Search name / company / phone / area", "", key="leads_search")
    disp = summary.copy()
    if search:
        mask = (
            disp.get("full_name", pd.Series(dtype=str)).astype(str).str.contains(search, case=False, na=False)
            | disp.get("company_name", pd.Series(dtype=str)).astype(str).str.contains(search, case=False, na=False)
            | disp.get("work_phone_number", pd.Series(dtype=str)).astype(str).str.contains(search, case=False, na=False)
            | disp.get("area", pd.Series(dtype=str)).astype(str).str.contains(search, case=False, na=False)
        )
        disp = disp[mask]

    disp["converted"] = disp["cusid"].apply(lambda v: "✅ Converted" if pd.notna(v) and str(v).strip() else "")

    _rename = {
        "id": "Lead ID", "created_time": "Created", "full_name": "Name",
        "company_name": "Company", "work_phone_number": "Phone", "area": "Area",
        "job_title": "Job Title", "campaign_name": "Campaign",
        "lead_status": "FB Status", "converted": "Status",
        "cusid": "Cus Code", "cusname": "Cus Name",
        "last_called": "Last Called", "last_outcome": "Last Outcome",
        "next_visit_date": "Next Follow Up", "last_notes": "Last Notes",
    }
    show_cols = [c for c in _rename if c in disp.columns]
    disp = disp[show_cols].rename(columns=_rename)

    if "Created" in disp.columns:
        disp["Created"] = pd.to_datetime(disp["Created"], errors="coerce").dt.strftime("%Y-%m-%d")

    st.caption(f"**{len(disp):,}** lead(s)")
    st.dataframe(
        disp,
        column_config={
            "Last Called":     st.column_config.DateColumn("Last Called",     format="YYYY-MM-DD"),
            "Next Follow Up":  st.column_config.DateColumn("Next Follow Up",  format="YYYY-MM-DD"),
        },
        width="stretch",
        hide_index=True,
    )

    if is_crm:
        st.download_button(
            "⬇ Download Leads CSV",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name=f"marketing_leads_{zid}.csv",
            mime="text/csv",
            key="leads_dl",
        )


def _render_lead_call_log_entry(zid: str, leads_df: pd.DataFrame) -> None:
    """Call log entry — same panel style as Customer Support."""
    st.markdown("#### 📞 Log a Call")
    if leads_df.empty:
        st.info("No leads yet — switch to **➕ Add Leads** to get started.")
        return

    lead_opts_df = leads_df[["id", "full_name", "company_name"]].copy()
    lead_opts = {
        f"{r['full_name']} — {r['company_name']} (#{r['id']})": int(r["id"])
        for _, r in lead_opts_df.iterrows()
    }
    lead_sel = st.selectbox(
        "Select lead",
        ["— pick a lead —"] + list(lead_opts.keys()),
        key="leads_call_sel",
    )
    if lead_sel and lead_sel != "— pick a lead —":
        sel_id = lead_opts[lead_sel]
        sel_name = lead_sel.split(" — ")[0]
        _render_lead_call_log_panel(sel_id, zid, sel_name, key_suffix="_leads")


def _render_all_lead_call_logs(zid: str, call_logs_df: pd.DataFrame) -> None:
    """Table 2 — all call logs, filterable by date called / outcome / area /
    next follow up, all four filters in a single row."""
    st.markdown("#### 📒 All Call Logs")
    log_tbl = build_lead_call_log_table(call_logs_df)
    if log_tbl.empty:
        st.info("No calls logged yet.")
        return

    lf1, lf2, lf3, lf4 = st.columns(4)

    # "Date called" is NOT NULL on every row, so defaulting the range to
    # the full min/max is a true no-op — matches the date-range convention
    # used elsewhere (collection.py, margin.py).
    called_dates = log_tbl["called_at"].dt.date.dropna()
    called_range = None
    if not called_dates.empty:
        with lf1:
            called_range = st.date_input(
                "Date called (range)",
                value=(called_dates.min(), called_dates.max()),
                key="leads_log_called_range",
            )

    with lf2:
        outcome_opts = sorted(log_tbl["outcome"].dropna().unique().tolist())
        outcome_sel = st.multiselect("Outcome", outcome_opts, key="leads_log_outcome")

    with lf3:
        area_opts = sorted(log_tbl["area"].dropna().unique().tolist()) if "area" in log_tbl.columns else []
        area_sel = st.multiselect("Area", area_opts, key="leads_log_area")

    # Next Follow Up is usually NULL (most calls don't set one). A plain
    # multiselect of the distinct dates that ARE scheduled -- rather than a
    # date-range input gated behind a checkbox -- means an empty selection
    # naturally shows everything, with no toggle needed to opt in first.
    with lf4:
        nvd_opts = sorted(log_tbl["next_visit_date"].dt.date.dropna().unique().tolist())
        nvd_sel = st.multiselect(
            "Next Follow Up", nvd_opts,
            format_func=lambda d: d.strftime("%Y-%m-%d"),
            key="leads_log_nvd_sel",
        )

    filt = log_tbl.copy()
    if isinstance(called_range, tuple) and len(called_range) == 2:
        start, end = called_range
        filt = filt[
            (filt["called_at"].dt.date >= start) & (filt["called_at"].dt.date <= end)
        ]
    if outcome_sel:
        filt = filt[filt["outcome"].isin(outcome_sel)]
    if area_sel:
        filt = filt[filt["area"].isin(area_sel)]
    if nvd_sel:
        filt = filt[filt["next_visit_date"].dt.date.isin(nvd_sel)]

    log_rename = {
        "lead_id": "Lead ID", "full_name": "Name", "company_name": "Company",
        "work_phone_number": "Phone", "area": "Area", "called_at": "Called At",
        "called_by": "Called By", "outcome": "Outcome",
        "next_visit_date": "Next Follow Up", "notes": "Notes",
    }
    log_cols = [c for c in log_rename if c in filt.columns]
    log_disp = filt[log_cols].rename(columns=log_rename)

    st.caption(f"**{len(log_disp):,}** call(s)")
    st.dataframe(
        log_disp,
        column_config={
            "Called At":       st.column_config.DatetimeColumn("Called At",       format="YYYY-MM-DD HH:mm"),
            "Next Follow Up":  st.column_config.DateColumn("Next Follow Up",      format="YYYY-MM-DD"),
        },
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "⬇ Download Call Log CSV",
        data=log_disp.to_csv(index=False).encode("utf-8"),
        file_name=f"lead_call_logs_{zid}.csv",
        mime="text/csv",
        key="leads_log_dl",
    )


def _show_leads(zid: str) -> None:
    role = st.session_state.get("user_role")
    is_crm = role in ("crm", "admin")

    with st.spinner("Loading leads…"):
        leads_df = _load_marketing_leads(zid)
        links_df = _load_cacus_lead_links(zid)
        call_logs_df = _load_all_lead_call_logs(zid)

    if not is_crm:
        # Sales: Table 1 only, read-only — no radio, no upload, no call log.
        _render_leads_table(zid, leads_df, links_df, call_logs_df, is_crm=False)
        return

    sub_mode = st.radio(
        "Leads",
        ["➕ Add Leads", "📞 Call Log"],
        horizontal=True,
        key="leads_top_mode",
    )
    st.markdown("---")

    if sub_mode == "➕ Add Leads":
        tab_bulk, tab_single, tab_edit = st.tabs(
            ["📤 Bulk Upload", "➕ Single Lead", "✏️ Edit Lead"]
        )
        with tab_bulk:
            _show_lead_upload(zid)
        with tab_single:
            _show_manual_lead_entry(zid)
        with tab_edit:
            _show_edit_lead(zid)
        return

    # ── 📞 Call Log: leads table first, then log-a-call, then all call logs ───
    _render_leads_table(zid, leads_df, links_df, call_logs_df, is_crm=True)

    if leads_df.empty:
        return

    st.markdown("---")
    _render_lead_call_log_entry(zid, leads_df)

    st.markdown("---")
    _render_all_lead_call_logs(zid, call_logs_df)


# ---------------------------------------------------------------------------
# 💬 WhatsFly Messaging — single-message test panel
#
# Build-phase scope only, per Whatsfly_Integration_docs/whatsfly-integration-guide.md:
# send ONE message to ONE number chosen by hand and look at the raw feedback.
# No receive-side / webhook handling here — that's a separate always-on
# FastAPI service, a later phase, and explicitly not this Streamlit app.
# Account-wide (one WhatsApp Business number), so this doesn't take zid.
# ---------------------------------------------------------------------------

# Confirmed real convention across this account's templates so far (both real
# dashboard examples used exactly this pair, in this order) — positional
# default for {{1}}/{{2}}'s variable NAME. A 3rd+ variable has no confirmed
# name yet, so it stays blank rather than guessing further.
_WF_DEFAULT_VAR_NAMES = ["CUSNAME", "CUSCODE"]


def _wf_guess(d: dict, keys: tuple) -> str | None:
    for k in keys:
        v = d.get(k) if isinstance(d, dict) else None
        if v:
            return str(v)
    return None


def _wf_normalize_templates(raw) -> list:
    """WhatsFly's real shape (confirmed against the live account):
    {"status": "1", "message": [ {id, template_id, template_name, ...}, ... ]}
    — the template list sits under "message", not "data"/"templates"/etc.
    (a genuinely surprising key name — "message" doubling as the payload
    array, not an error string). Every other common wrapper key is still
    tried too, plus a bare-list / single-template fallback, since this is
    all reverse-engineered from one account's response, not documented."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("message", "data", "templates", "result", "results", "list"):
            val = raw.get(key)
            if isinstance(val, list):
                return val
        if any(k in raw for k in ("template_name", "name", "elementName")):
            return [raw]
    return []


def _wf_template_label(t: dict, i: int) -> str:
    if not isinstance(t, dict):
        return f"Template {i + 1}"
    tid = _wf_guess(t, ("id",))
    name = _wf_guess(t, ("template_name", "name", "elementName")) or f"Template {i + 1}"
    category = _wf_guess(t, ("template_category", "category"))
    lang = _wf_guess(t, ("locale", "language", "language_code", "lang"))
    label = f"{name} (id {tid})" if tid else name
    if category:
        label += f" — {category}"
    if lang:
        label += f" ({lang})"
    return label


def _wf_templates_table(templates: list) -> pd.DataFrame:
    """The 'what can we actually access' check. Confirmed real fields on the
    live account: id (WhatsFly's own internal row id), template_id (the
    long numeric Meta-style id), template_name, template_type,
    template_category — language/status kept as a guessed fallback in case
    a template on this account ever carries them, since the API's Get
    Template docs describe an approval `status` this particular response
    didn't happen to include."""
    rows = [
        {
            "ID": _wf_guess(t, ("id",)) or "—",
            "Template ID": _wf_guess(t, ("template_id", "wa_template_id", "uuid")) or "—",
            "Name": _wf_guess(t, ("template_name", "name", "elementName")) or "—",
            "Category": _wf_guess(t, ("template_category", "category")) or "—",
            "Type": _wf_guess(t, ("template_type", "type")) or "—",
            "Language": _wf_guess(t, ("locale", "language", "language_code", "lang")) or "—",
            "Status": _wf_guess(t, ("status", "template_status")) or "—",
        }
        for t in templates
    ]
    return pd.DataFrame(rows)


def _wf_extract_components(t: dict) -> dict:
    """Pull header/body/footer text from a template entry for the preview.

    CONFIRMED against the live account (fetched and inspected directly,
    not guessed): WhatsFly's own template-list response is NOT the
    Meta-style `components: [{type, text}]` shape at the top level — that
    only exists nested inside a SEPARATE stringified `template_json` field.
    The top-level object instead carries `body_content`/`header_content`/
    `footer_content` — plain, already-unicode-decoded text, ready to
    display as-is. This is the real bug behind "preview shows the coded
    message": the old fallback here (scanning every string for '{{') found
    none in body_content (WhatsFly's own placeholders are '#NAME#'-style,
    not '{{n}}' — see _wf_extract_variable_map) and instead matched
    `template_json` itself (a giant string that DOES contain literal
    '{{1}}' inside its nested, still-JSON-escaped text) and rendered that
    whole raw blob as if it were the body.

    Old Meta Cloud API `components`-list shape is kept as a fallback below
    for safety, in case a template not managed through WhatsFly's own UI
    ever shows up with that shape instead — not needed for any of this
    account's real templates, all 8 of which use body_content."""
    out = {"header": None, "body": "", "footer": None, "style": "meta"}
    if not isinstance(t, dict):
        return out

    if "body_content" in t or "header_type" in t:
        return {
            "header": t.get("header_content") or None,
            "body": t.get("body_content") or "",
            "footer": t.get("footer_content") or None,
            "style": "whatsfly",
        }

    for key in ("body", "body_text", "message", "text", "template_text"):
        v = t.get(key)
        if isinstance(v, str) and v.strip():
            out["body"] = v
            break

    components = t.get("components")
    if isinstance(components, list):
        for comp in components:
            if not isinstance(comp, dict):
                continue
            ctype = str(comp.get("type", "")).upper()
            text = comp.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            if ctype == "HEADER" and not out["header"]:
                out["header"] = text
            elif ctype == "BODY" and not out["body"]:
                out["body"] = text
            elif ctype == "FOOTER" and not out["footer"]:
                out["footer"] = text

    if not out["body"]:
        def _scan(node):
            if isinstance(node, str) and "{{" in node:
                return node
            if isinstance(node, dict):
                for v in node.values():
                    found = _scan(v)
                    if found:
                        return found
            if isinstance(node, list):
                for v in node:
                    found = _scan(v)
                    if found:
                        return found
            return None

        out["body"] = _scan(t) or ""

    return out


_VAR_TOKEN_RE = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")

# WhatsFly's OWN placeholder syntax in body_content/header_content/
# footer_content — confirmed live, and confirmed INCONSISTENT even within
# one template: system_abandoned_cart_reminder_new's own variable_map has
# both "#LEAD_USER_FIRST_NAME#" (bare, no '!') and "#!system-cart-product-
# list!#" (with '!') side by side for its own variables 1 and 2. This
# matches both forms; names may contain hyphens (seen live: "system-cart-
# total-price").
_WF_PLACEHOLDER_RE = re.compile(r"#!?([A-Za-z0-9_-]+)!?#")


def _wf_extract_variable_tokens(body_text: str) -> list:
    """Ordered, de-duplicated `{{...}}`-style variable tokens — the Meta
    Cloud API convention, used only as a fallback when a template has no
    `variable_map` of its own (see _wf_extract_variable_map, the primary
    path for this account's real WhatsFly-managed templates)."""
    seen = []
    for m in _VAR_TOKEN_RE.finditer(body_text or ""):
        tok = m.group(1)
        if tok not in seen:
            seen.append(tok)
    return seen


def _wf_extract_variable_map(t: dict) -> list:
    """Ordered [(position, clean_name), ...] straight from WhatsFly's own
    `variable_map` field — e.g. {"header":[],"body":{"1":"#!CUSNAME!#",
    "2":"#!CUSCODE!#"},"button":[]} on the live account, confirmed against
    a real fetch. This is ground truth for what each positional variable
    is actually called — no more guessing via _WF_DEFAULT_VAR_NAMES for
    any template that has one (i.e. every real template on this account
    today). `body` can also come back as an empty list `[]` (a template
    with zero variables, e.g. eid_shuveccha) rather than a dict — treated
    the same as absent."""
    raw_map = t.get("variable_map") if isinstance(t, dict) else None
    if not raw_map:
        return []
    try:
        parsed = json.loads(raw_map) if isinstance(raw_map, str) else raw_map
    except (TypeError, ValueError):
        return []
    body_map = parsed.get("body") if isinstance(parsed, dict) else None
    if not isinstance(body_map, dict) or not body_map:
        return []

    def _sort_key(k):
        return int(k) if str(k).isdigit() else 0

    out = []
    for pos in sorted(body_map, key=_sort_key):
        raw_name = str(body_map[pos]).strip()
        m = _WF_PLACEHOLDER_RE.fullmatch(raw_name)
        out.append((pos, m.group(1) if m else raw_name))
    return out


def _wf_substitute_positional_preview(body_html: str, values: list) -> str:
    """Replaces each '#NAME#'/'#!NAME!#'-style placeholder occurrence in
    the (already markup-formatted + escaped) body HTML with the value at
    the SAME appearance position, left to right — matches WhatsFly's own
    variable_map ordering ("1", "2", ...) regardless of what the
    placeholder text itself says, since that text is confirmed
    inconsistent (see _WF_PLACEHOLDER_RE). Highlighted where filled in,
    dimmed as a `[n]` hint where still blank. Separate from
    _wf_substitute_preview (the `{{...}}`-matching one) which Direct
    WhatsApp also uses as-is — this one is WhatsFly-panel-only."""
    counter = {"i": 0}

    def _sub(m):
        i = counter["i"]
        counter["i"] += 1
        val = values[i].strip() if i < len(values) and values[i] else ""
        if val:
            return (
                '<span style="background:#FFF3B0;border-radius:3px;padding:0 3px;">'
                f'{_html.escape(val)}</span>'
            )
        return f'<span style="color:#7a8a99;font-style:italic;">[{i + 1}]</span>'

    return _WF_PLACEHOLDER_RE.sub(_sub, body_html)


def _wf_extract_media_ref(resp) -> tuple:
    """Best-effort media id/url extraction from the upload/media response —
    shape unconfirmed against the live account (first real use of this
    endpoint), same defensive multi-key-guess stance as templates. Unwraps a
    `message`/`data`/`result` wrapper dict first, since this account already
    confirmed "message" as its payload-wrapper key for the template list."""
    if not isinstance(resp, dict):
        return None, None
    node = resp
    for key in ("message", "data", "result"):
        v = resp.get(key)
        if isinstance(v, dict):
            node = v
            break
    media_id = _wf_guess(node, ("id", "media_id", "mediaId", "media_hash"))
    media_url = _wf_guess(node, ("url", "media_url", "link", "file_url"))
    return media_id, media_url


def _wf_format_whatsapp_markup(text: str) -> str:
    """WhatsApp's own lightweight markup (*bold*, _italic_, ~strike~,
    ```mono```) turned into HTML for the beautified preview bubble. Escapes
    the source text first so template copy can never inject arbitrary HTML."""
    escaped = _html.escape(text or "")
    escaped = re.sub(r"\*(.+?)\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"_(.+?)_", r"<i>\1</i>", escaped)
    escaped = re.sub(r"~(.+?)~", r"<s>\1</s>", escaped)
    escaped = re.sub(r"```(.+?)```", r"<code>\1</code>", escaped, flags=re.DOTALL)
    return escaped.replace("\n", "<br>")


def _wf_substitute_preview(body_html: str, tokens: list, variables: list) -> str:
    """Drops the entered variable values into the already-formatted body
    HTML in place of each `{{token}}` — highlighted where filled in, dimmed
    as a `[token]` hint where still blank — so the bubble below updates live
    as inputs are typed into. `tokens`/`variables` are parallel lists (same
    order as _wf_extract_variable_tokens found them) — matched by TOKEN TEXT
    here, not by casting to int, so this works for both positional (`{{1}}`)
    and named (`{{cusname}}`) templates alike. `{{`/`}}` survive
    _wf_format_whatsapp_markup's html.escape untouched, so this regex still
    matches after that pass."""
    value_by_token = {tok: (variables[i].strip() if i < len(variables) and variables[i] else "")
                       for i, tok in enumerate(tokens)}

    def _sub(m):
        tok = m.group(1)
        val = value_by_token.get(tok, "")
        if val:
            return (
                '<span style="background:#FFF3B0;border-radius:3px;padding:0 3px;">'
                f'{_html.escape(val)}</span>'
            )
        return f'<span style="color:#7a8a99;font-style:italic;">[{tok}]</span>'

    return _VAR_TOKEN_RE.sub(_sub, body_html)


def _wf_render_bubble(header: str | None, body_html: str, footer: str | None) -> None:
    parts = []
    if header:
        parts.append(
            f'<div style="font-weight:700;margin-bottom:6px;">'
            f'{_wf_format_whatsapp_markup(header)}</div>'
        )
    parts.append(f'<div>{body_html}</div>')
    if footer:
        parts.append(
            f'<div style="color:#5B7083;font-size:12px;margin-top:8px;">'
            f'{_wf_format_whatsapp_markup(footer)}</div>'
        )
    st.markdown(
        '<div style="background:#DCF8C6;border:1px solid #B4E2A0;border-radius:10px;'
        'padding:12px 16px;color:#111;max-width:560px;font-size:15px;line-height:1.45;">'
        + "".join(parts) + '</div>',
        unsafe_allow_html=True,
    )


def _wf_render_phone_preview(header: str | None, body_html: str, footer: str | None, image_src: str | None = None) -> None:
    """Deliberately plain 'phone screen' frame around the message bubble —
    not a literal phone graphic, just a bordered, WhatsApp-chat-colored
    panel so the preview reads as 'this is what shows up on their phone'.
    WhatsFly-panel-only (Direct WhatsApp keeps using _wf_render_bubble
    as-is, unchanged, since it's shared code — see that function's own
    call sites). Built as ONE html string / one st.markdown call so the
    image + bubble actually nest inside the frame — two separate
    st.markdown calls would render as siblings, not nested."""
    img_html = (
        f'<img src="{_html.escape(image_src, quote=True)}" '
        'style="width:100%;max-height:220px;object-fit:cover;border-radius:10px 10px 0 0;display:block;">'
    ) if image_src else ""
    text_parts = []
    if header:
        text_parts.append(f'<div style="font-weight:700;margin-bottom:6px;">{_wf_format_whatsapp_markup(header)}</div>')
    text_parts.append(
        f'<div>{body_html}</div>' if body_html
        else '<div style="color:#7a8a99;">No body text found for this template.</div>'
    )
    if footer:
        text_parts.append(f'<div style="color:#5B7083;font-size:12px;margin-top:8px;">{_wf_format_whatsapp_markup(footer)}</div>')
    bubble_html = (
        '<div style="background:#DCF8C6;border:1px solid #B4E2A0;border-radius:10px;overflow:hidden;'
        'color:#111;font-size:15px;line-height:1.45;">'
        + img_html
        + f'<div style="padding:12px 16px;">{"".join(text_parts)}</div>'
        + '</div>'
    )
    st.markdown(
        '<div style="background:#ECE5DD;border:1px solid #ccc;border-radius:16px;'
        'padding:20px 14px;min-height:180px;">' + bubble_html + '</div>',
        unsafe_allow_html=True,
    )


def _wf_build_template_payload(
    template_id: str, named_variables: list, header_media_url: str | None,
    use_meta_style: bool = False, template_name: str = "", language_code: str = "en",
    header_image_param: dict | None = None,
) -> dict:
    """WhatsFly's confirmed-working flat send shape by default; the
    documented-but-unconfirmed Meta Cloud API nested shape as an
    admin-only fallback (see the WhatsFly Messaging notes in CLAUDE.md for
    how each key here was confirmed against real dashboard examples).
    Pure function of the current widget state — called fresh on every
    rerun, never a separate editable copy that can go stale, so there's no
    'rebuild' step before sending."""
    if use_meta_style:
        components = []
        if header_image_param:
            components.append({"type": "header", "parameters": [{"type": "image", "image": header_image_param}]})
        body_values = [v for _, v in named_variables]
        if body_values:
            components.append({"type": "body", "parameters": [{"type": "text", "text": v} for v in body_values]})
        return {"template": {"name": template_name, "language": {"code": language_code}, "components": components}}
    payload = {"template_id": template_id}
    for i, (vname, vval) in enumerate(named_variables):
        if vname.strip():
            payload[f"templateVariable-{vname.strip()}-{i + 1}"] = vval
    if header_media_url:
        payload["template_header_media_url"] = header_media_url
    return payload


def _render_wf_response(resp) -> None:
    st.markdown("---")
    st.markdown(f"**HTTP status:** `{resp.status_code}`")
    try:
        body = resp.json()
    except ValueError:
        st.code(resp.text or "(empty response body)")
        return

    # response envelope inconsistency per the build guide: most endpoints
    # return status as the STRING "1"/"0", catalog endpoints return a
    # boolean — handle both rather than assuming one.
    status_val = body.get("status") if isinstance(body, dict) else None
    is_ok = status_val in ("1", 1, True) or (resp.ok and status_val is None)
    if is_ok:
        st.success("Sent — see raw response below for the details WhatsFly returned.")
    else:
        st.error("WhatsFly reported an error — see raw response below.")
        err_msg = str(body.get("message", "")) if isinstance(body, dict) else ""
        if "does not exist" in err_msg and "graph-api" in err_msg.lower():
            st.info(
                "This is a **Meta Graph API-level** error, not a request-shape problem — "
                "the request reached Meta's backend and Meta itself rejected "
                "`phone_number_id`. Nothing left to fix in this panel; check WhatsFly's "
                "dashboard (is the number still connected?) or Meta Business Manager "
                "(permissions on the System User / app for this WABA number), or ask "
                "WhatsFly support directly, quoting this exact message."
            )
    st.json(body)


def _render_wf_text_send(phone_number: str) -> None:
    """Plain session text — an actual chat box (st.chat_input), not a
    separate text area + button, per explicit ask. Sends immediately on
    Enter, right below the conversation history above it."""
    message = st.chat_input("Type a message and press Enter to send…", key="wf_chat_input")
    if not message:
        return
    if not phone_number.strip():
        st.error("Enter a recipient phone number first.")
        return
    with st.spinner("Sending…"):
        try:
            resp = whatsfly.send_text(phone_number.strip(), message)
        except Exception as e:
            st.error(f"Send failed: {e}")
            return
    _render_wf_response(resp)


def _render_wf_template_send(phone_number: str, customer_name: str | None = None, customer_code: str | None = None) -> None:
    """Dummy-phone layout: type on the left, live preview on the right —
    everything on the right always reflects exactly what's on the left,
    no separate 'rebuild' step. Image attachment sits at the TOP of the
    left column so it's already in the preview by the time you reach the
    variables. Raw payload JSON / template list dump only ever shows for
    admin users (Advanced / Debug), never in the day-to-day flow.

    `customer_name`/`customer_code` are only ever populated when the
    recipient was picked from the customer list (not typed manually) —
    when set, a CUSNAME/CUSCODE variable is auto-filled from them instead
    of asking for it."""
    if st.button("🔄 Refresh Templates", key="wf_refresh_templates_btn"):
        st.session_state.pop("_wf_templates_raw", None)

    if "_wf_templates_raw" not in st.session_state:
        with st.spinner("Fetching templates…"):
            try:
                st.session_state["_wf_templates_raw"] = whatsfly.get_templates()
            except Exception as e:
                st.error(f"Couldn't fetch templates: {e}")
                return

    is_admin = st.session_state.get("user_role") == "admin"
    raw = st.session_state["_wf_templates_raw"]
    templates = _wf_normalize_templates(raw)
    if not templates:
        st.warning("No templates found.")
        if is_admin:
            with st.expander("⚙️ Advanced / Debug (admin only)"):
                st.json(raw)
        return

    labels = [_wf_template_label(t, i) for i, t in enumerate(templates)]
    idx = st.selectbox(
        f"📋 Template ({len(templates)} available)", range(len(templates)),
        format_func=lambda i: labels[i], key="wf_template_idx",
    )
    template = templates[idx]

    comps = _wf_extract_components(template)
    is_native = comps.get("style") == "whatsfly"
    body_text = comps["body"]
    body_html = _wf_format_whatsapp_markup(body_text)

    # Ground truth for this account's real templates: WhatsFly's own
    # variable_map gives the actual name per position (see
    # _wf_extract_variable_map's docstring) — no guessing needed. Only
    # falls back to the old {{token}} scan + _WF_DEFAULT_VAR_NAMES guess
    # for a template that somehow has no variable_map of its own.
    var_map = _wf_extract_variable_map(template) if is_native else []
    tokens = []
    if var_map:
        var_entries = var_map
    else:
        tokens = _wf_extract_variable_tokens(body_text)
        var_entries = [
            (str(i + 1), tok if not tok.isdigit() else (
                _WF_DEFAULT_VAR_NAMES[i] if i < len(_WF_DEFAULT_VAR_NAMES) else f"var{i + 1}"
            ))
            for i, tok in enumerate(tokens)
        ]

    left, right = st.columns([1, 1])

    with left:
        # Auto-detected from the template's own header_type/header_subtype
        # (confirmed live: "media"/"image") for a WhatsFly-native template
        # — no more manual checkbox. A non-native (fallback-shape)
        # template has no such field, so it still asks.
        if is_native:
            is_image_header = template.get("header_type") == "media" and template.get("header_subtype") == "image"
            if is_image_header:
                st.markdown("**🖼️ Header Image** _(auto-detected)_")
        else:
            st.markdown("**🖼️ Header Image (optional)**")
            is_image_header = st.checkbox(
                "This template's header is an image",
                key=f"wf_has_img_header_{idx}",
                help="Not auto-detected for this template — check manually. Meta requires JPEG/PNG, 5MB max.",
            )

        header_media_id = header_media_url = None
        image_preview_src = None
        if is_image_header:
            uploaded_file = st.file_uploader(
                "Attach image", type=["jpg", "jpeg", "png"], key=f"wf_header_img_{idx}", label_visibility="collapsed",
            )
            manual_url = st.text_input("…or paste a hosted image URL", key=f"wf_header_img_url_{idx}")

            if uploaded_file is not None:
                file_sig = (uploaded_file.name, uploaded_file.size)
                upload_cache_key = f"wf_header_upload_{idx}"
                cached = st.session_state.get(upload_cache_key)
                if not cached or cached.get("sig") != file_sig:
                    with st.spinner("Uploading…"):
                        try:
                            raw_up = whatsfly.upload_media(
                                uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type or "image/jpeg"
                            )
                            mid, murl = _wf_extract_media_ref(raw_up)
                            st.session_state[upload_cache_key] = {
                                "sig": file_sig, "raw": raw_up, "media_id": mid, "media_url": murl, "error": None,
                            }
                        except Exception as e:
                            st.session_state[upload_cache_key] = {"sig": file_sig, "raw": None, "error": str(e)}

                cached = st.session_state.get(upload_cache_key)
                if cached and cached.get("error"):
                    st.error(f"Upload failed: {cached['error']}")
                elif cached and cached.get("raw") is not None:
                    if cached.get("media_id") or cached.get("media_url"):
                        st.success("Uploaded — see preview on the right.")
                        header_media_id = cached.get("media_id")
                        header_media_url = cached.get("media_url")
                    else:
                        st.warning("Uploaded, but no id/url found for it (see Advanced / Debug).")
                # Preview the actual file itself (not just whatever url came
                # back from the upload) so the right-hand preview is exactly
                # what was picked, even before/without a successful upload.
                image_preview_src = f"data:{uploaded_file.type or 'image/jpeg'};base64," + base64.b64encode(uploaded_file.getvalue()).decode()

            if not header_media_id and not header_media_url and manual_url.strip():
                header_media_url = manual_url.strip()
                if not image_preview_src:
                    image_preview_src = header_media_url

        header_image_param = (
            {"id": header_media_id} if header_media_id
            else {"link": header_media_url} if header_media_url
            else None
        )

        # WhatsFly's send contract always keys a variable by NAME
        # (`templateVariable-<name>-<n>`, confirmed via a real dashboard
        # example). CUSNAME/CUSCODE — this account's own convention for
        # "the picked customer's name/code" — auto-fill from the selected
        # customer and skip the input entirely; every other variable
        # (there's no data source for e.g. "cart total") still asks.
        named_variables = []
        if var_entries:
            st.markdown(f"**✏️ Fill in {len(var_entries)} variable(s)**")
            for pos, name in var_entries:
                upper = name.strip().upper()
                if upper == "CUSNAME" and customer_name:
                    st.caption(f"**{name}** → {customer_name} _(from selected customer)_")
                    named_variables.append((name, customer_name))
                elif upper == "CUSCODE" and customer_code:
                    st.caption(f"**{name}** → {customer_code} _(from selected customer)_")
                    named_variables.append((name, customer_code))
                else:
                    vval = st.text_input(name, key=f"wf_var_{idx}_{pos}")
                    named_variables.append((name, vval))
        else:
            st.caption("This template has no variables to fill in.")

    with right:
        st.markdown("**📱 Preview**")
        if body_text:
            values = [v for _, v in named_variables]
            preview_body = (
                _wf_substitute_positional_preview(body_html, values) if is_native
                else _wf_substitute_preview(body_html, tokens, values)
            )
        else:
            preview_body = ""
        _wf_render_phone_preview(comps["header"], preview_body, comps["footer"], image_preview_src)

    # Everything below builds the send payload fresh from the widget state
    # above — no separate editable/persisted copy, so it can never go stale.
    template_id_val = _wf_guess(template, ("id", "template_id", "wa_template_id", "uuid")) or ""
    endpoint = "/whatsapp/send/template"
    use_meta_style = False
    template_name_override = _wf_guess(template, ("template_name", "name", "elementName")) or ""
    language_code_override = _wf_guess(template, ("locale", "language", "language_code", "lang")) or "en"

    if is_admin:
        with st.expander("⚙️ Advanced / Debug (admin only)"):
            # Confirmed via real dashboard-generated examples: endpoint is
            # POST /whatsapp/send/template, flat params. `template_id` is a
            # naming trap — it wants WhatsFly's short internal `id`, NOT
            # the longer `template_id` field the list response returns.
            template_id_val = st.text_input(
                "Template ID (WhatsFly's short `id`, not the list's `template_id`)",
                value=template_id_val, key=f"wf_template_id_{idx}",
            )
            endpoint = st.text_input("Send endpoint", value=endpoint, key="wf_endpoint")
            use_meta_style = st.checkbox(
                "Try Meta Cloud API style instead",
                key=f"wf_use_meta_style_{idx}",
                help="Documented fallback only — the flat shape above is the confirmed-working one; "
                     "this failed with \"Message template not found\" on a real attempt.",
            )
            if use_meta_style:
                template_name_override = st.text_input("Template name", value=template_name_override, key=f"wf_template_name_{idx}")
                language_code_override = st.text_input("Language code", value=language_code_override, key=f"wf_lang_{idx}")

            payload = _wf_build_template_payload(
                template_id_val, named_variables, header_media_url,
                use_meta_style, template_name_override, language_code_override, header_image_param,
            )
            st.markdown("**Payload that will be sent** _(builds live from the fields above — no rebuild needed)_")
            st.json(payload)
            st.markdown("**Raw template data**")
            st.json(raw, expanded=False)
            st.json(template, expanded=False)
            st.dataframe(_wf_templates_table(templates), width="stretch", hide_index=True)
    else:
        payload = _wf_build_template_payload(
            template_id_val, named_variables, header_media_url,
            use_meta_style, template_name_override, language_code_override, header_image_param,
        )

    if st.button("📤 Send Template Message", key="wf_send_template_btn", type="primary"):
        if not phone_number.strip():
            st.error("Enter a recipient phone number first.")
            return
        with st.spinner("Sending…"):
            try:
                resp = whatsfly.send_template(phone_number.strip(), endpoint.strip(), payload)
            except Exception as e:
                st.error(f"Send failed: {e}")
                return
        _render_wf_response(resp)


_WF_STATUS_LABEL = {"sent": "✓ Sent", "delivered": "✓✓ Delivered", "read": "✓✓ Read", "failed": "⚠️ Failed"}


def _wf_load_conversation_history(phone_numbers: list) -> list:
    """Loads a customer's WhatsApp thread, matched purely by phone number —
    the whatsapp_webhooks database has no concept of a customer code at
    all, so this is the only way to relate a cacus customer to their
    conversation. `phone_numbers` is a list (not one number) since a
    customer can have two WhatsApp-eligible numbers (primary/secondary)
    and either could be the one they've actually messaged from.

    Returns [] on any lookup problem (DB not configured, connection error,
    etc.) rather than raising — a config/infra gap here is not something a
    person sending a message should ever see the internals of; it just
    reads as 'no history yet'."""
    try:
        return whatsapp_webhook_db.get_messages_for_contact(phone_numbers)
    except Exception:
        return []


def _wf_last_inbound_timestamp(messages: list):
    """Most recent INBOUND message's timestamp, or None if the customer has
    never messaged — this, not our own outbound sends, is what starts/
    extends WhatsApp's 24h free-text session window."""
    latest = None
    for m in messages:
        if m.get("direction") != "inbound":
            continue
        ts = m.get("message_timestamp") or m.get("created_at")
        if ts and (latest is None or ts > latest):
            latest = ts
    return latest


def _wf_session_window_status(last_inbound_ts) -> tuple:
    """(is_open, caption) — WhatsApp's real rule: a plain session message
    only works within 24h of the CUSTOMER's own last message; outside that
    window (or if they've never messaged at all), only an approved
    template can be sent. Timestamps without tzinfo are treated as UTC —
    matches this account's own confirmed convention (WhatsFly's
    status_time is a naive string, confirmed UTC — see
    whatsapp_webhook/whatsfly_handlers.py)."""
    if not last_inbound_ts:
        return False, "🔒 No active session — the customer hasn't messaged you yet, so only an approved template can be sent."
    ts = last_inbound_ts if last_inbound_ts.tzinfo else last_inbound_ts.replace(tzinfo=timezone.utc)
    remaining = timedelta(hours=24) - (datetime.now(timezone.utc) - ts)
    if remaining.total_seconds() <= 0:
        hours_ago = int((datetime.now(timezone.utc) - ts).total_seconds() // 3600)
        return False, f"🔒 Session expired — their last message was {hours_ago}h ago. Only an approved template can be sent now."
    hrs, rem = divmod(int(remaining.total_seconds()), 3600)
    mins = rem // 60
    return True, f"🟢 Session open — {hrs}h {mins}m left to reply with plain text (from their last message)."


def _wf_render_chat_history(messages: list) -> None:
    """Renders an already-loaded message thread as chat bubbles."""
    if not messages:
        st.caption("No conversation history yet.")
        return

    for m in messages:
        is_inbound = m.get("direction") == "inbound"
        with st.chat_message("user" if is_inbound else "assistant", avatar="🧑" if is_inbound else "🏢"):
            content = m.get("content")
            body = content.get("body") if isinstance(content, dict) else None
            st.markdown(body or f"_{m.get('message_type') or 'message'} (no text)_")
            ts = m.get("message_timestamp") or m.get("created_at")
            meta = ts.strftime("%b %d, %Y %H:%M") if ts else ""
            if not is_inbound and m.get("current_status"):
                status_label = _WF_STATUS_LABEL.get(m["current_status"], m["current_status"])
                meta = f"{meta}  ·  {status_label}" if meta else status_label
            if meta:
                st.caption(meta)


def _show_whatsfly_messaging(zid: str) -> None:
    st.subheader("💬 WhatsFly — Send Single Message")
    st.caption("Send one message to one customer, with their conversation history right here.")

    try:
        whatsfly.get_credentials()
    except whatsfly.WhatsFlyConfigError as e:
        st.warning(str(e))
        return

    phone_number = ""
    primary = secondary = None
    customer_name = customer_code = None

    with st.container(border=True):
        st.markdown("**👤 Recipient**")
        recipient_mode = st.radio(
            "Recipient", ["Pick a customer", "Enter manually"],
            horizontal=True, label_visibility="collapsed", key="wf_recipient_mode",
        )

        if recipient_mode == "Pick a customer":
            cacus_df = _load_cacus(str(zid))
            if cacus_df.empty:
                st.warning("No customer data available for this ZID.")
            else:
                labels = ["— Select a customer —"] + [
                    f"{r.cusid} - {str(r.cusname).strip() or '(no name)'}" for r in cacus_df.itertuples()
                ]
                choice = st.selectbox("Customer", labels, key="wf_customer_pick")
                if choice != labels[0]:
                    cusid = choice.split(" - ", 1)[0]
                    row = cacus_df.loc[cacus_df["cusid"] == cusid].iloc[0]
                    customer_code = cusid
                    customer_name = str(row.get("cusname") or "").strip() or None
                    primary, secondary = customer_whatsapp_numbers(row.get("cusmobile"), row.get("whatsapp"))
                    if not primary:
                        st.warning("No phone number on file for this customer — switch to manual entry instead.")
                    else:
                        phone_number = primary
                        if secondary and st.checkbox(
                            f"Use secondary number ({secondary}) instead of primary ({primary})",
                            key="wf_use_secondary",
                        ):
                            phone_number = secondary
                        st.caption(f"Sending to: `{phone_number}`")
        else:
            phone_number = st.text_input(
                "Recipient phone number",
                key="wf_phone_number",
                help="Country code + digits only — no '+', no spaces, e.g. 8801XXXXXXXXX.",
            )

    # Sanity check, not a hard block — a standard BD WhatsApp number is
    # 880 + 10 digits = 13 digits total. Real customer-master data has
    # genuine entry errors (e.g. a double leading zero) that normalize to
    # something the wrong length; flag it rather than silently sending to
    # a number that's likely wrong, or guessing at a "corrected" one.
    if phone_number and (not phone_number.isdigit() or len(phone_number) != 13):
        st.caption(
            f"⚠️ `{phone_number}` doesn't look like a standard 880-format number "
            "(expected 13 digits) — double-check before sending."
        )

    history_numbers = [n for n in {phone_number, primary, secondary} if n]
    messages = []
    if history_numbers:
        messages = _wf_load_conversation_history(history_numbers)
        session_open, window_caption = _wf_session_window_status(_wf_last_inbound_timestamp(messages))
        with st.container(border=True):
            st.markdown("**💬 Conversation History**")
            _wf_render_chat_history(messages)
            st.caption(window_caption)
            # Reply lives right here, in the same panel as the thread it's
            # replying to — only offered while the 24h session is actually
            # open, since WhatsApp itself rejects plain text outside it.
            if session_open:
                _render_wf_text_send(phone_number)

    # Templates work regardless of session state — that's what they're
    # for — so this section is always available, no toggle needed; plain
    # text lives entirely in the Conversation History panel above now.
    st.markdown("---")
    st.markdown("**✉️ Send a Template**")
    _render_wf_template_send(phone_number, customer_name, customer_code)


# ---------------------------------------------------------------------------
# 📢 WhatsFly Bulk Messaging — campaign sends, one template at a time.
#
# Deliberately built template-FIRST rather than as one generic bulk-send
# form: each real campaign (audience, what gets filled into each variable,
# any extra filtering) is its own thing, and trying to guess a one-size-
# shape now, before more than one campaign exists, would just mean
# rebuilding it anyway. So the skeleton here is: pick a template, dispatch
# to that template's own view function via _WF_BULK_TEMPLATE_VIEWS below,
# with an explicit "not built yet" placeholder for everything unregistered.
# Once several real campaigns exist side by side, look for what they share
# and fold the common parts into one generic flow — not before.
# ---------------------------------------------------------------------------


def _wf_bulk_default_view(zid: str, template: dict) -> None:
    """Shown for any template with no campaign view registered yet in
    _WF_BULK_TEMPLATE_VIEWS — just a preview, so there's something concrete
    on screen while each campaign gets built out one at a time."""
    comps = _wf_extract_components(template)
    body_html = _wf_format_whatsapp_markup(comps["body"])
    st.info(
        "This template doesn't have a bulk-send campaign built for it yet. "
        "Describe who it should go to and what fills each variable, and "
        "it'll get its own view registered in _WF_BULK_TEMPLATE_VIEWS."
    )
    _wf_render_phone_preview(comps["header"], body_html, comps["footer"])


# {template_name: handler(zid, template) -> None} — add one entry per
# campaign as it gets defined. Falls back to _wf_bulk_default_view for
# every template not listed here yet.
_WF_BULK_TEMPLATE_VIEWS = {}


# ---------------------------------------------------------------------------
# Audience filter builder — free-form "add a filter", any order, each one
# narrowing whatever candidates survived the ones before it (AND across
# filter types, confirmed). See processing/wf_bulk_audience.py for the
# actual query/pandas logic; everything here is orchestration + widgets.
# ---------------------------------------------------------------------------

_BULK_FILTER_CATALOG = [
    ("area", "Area"),
    ("salesman", "Salesman"),
    ("product_count_band", "Unique Products Bought"),
    ("net_sales", "Net Sales (window)"),
    ("total_returns", "Total Returns (window)"),
    ("days_since_last_sale", "Days Since Last Sale"),
    ("customer_score", "Customer Score"),
    ("current_balance", "Current Balance"),
    ("product", "Product"),
    ("inactive", "Inactive (own timeline)"),
    ("order_date", "Ordered On (exact date)"),
    ("collection_date", "Collection Received On (exact date)"),
    ("avg_collection_days", "Avg Collection Days (CP)"),
]
_BULK_FILTER_LABELS = dict(_BULK_FILTER_CATALOG)


def _wfb_available_to_add(active_types: set) -> list:
    """Which filter types can still be added, given what's already active.
    Two real dependency rules: Salesman needs Area first (they come off
    the same opdor rows — Salesman's own options are meaningless without
    an Area to scope them to), and nothing can be added once Avg
    Collection Days is active — it's the one expensive per-customer
    computation, so it always runs last, over whatever's already been
    narrowed down."""
    if "avg_collection_days" in active_types:
        return []
    out = []
    for key, label in _BULK_FILTER_CATALOG:
        if key in active_types:
            continue
        if key == "salesman" and "area" not in active_types:
            continue
        out.append((key, label))
    return out


def _wfb_describe_filter(f: dict) -> str:
    t = f["type"]
    if t == "area":
        return ", ".join(f["value"])
    if t == "salesman":
        labels = f.get("labels") or [str(v) for v in f["value"]]
        return f"{', '.join(labels)} (in {', '.join(f['area'])})"
    if t == "avg_collection_days":
        return f"{f['min']:g}–{f['max']:g} days (own {f.get('window_months', '?')}-month window)"
    if t in ("product_count_band", "days_since_last_sale"):
        return f"{f['min']:g}–{f['max']:g}"
    if t == "customer_score":
        return f"{f['min']:g}–{f['max']:g}"
    if t in ("current_balance", "net_sales", "total_returns"):
        return f"{f['min']:,.0f}–{f['max']:,.0f}"
    if t == "product":
        return ", ".join(f.get("labels") or f["value"])
    if t == "inactive":
        return f"no purchase (own {f.get('window_months', '?')}-month window)"
    if t in ("order_date", "collection_date"):
        return str(f["value"])
    return ""


@st.cache_data(show_spinner=False, ttl=3600)
def _wfb_customer_metrics(zid: str, proj: str, _sales_raw: pd.DataFrame, _coll_df: pd.DataFrame, selected_years: tuple) -> pd.DataFrame:
    """cusid + composite_score + current_balance, via the exact same
    build_customer_marketing_table used by Marketing -> Customer Scoring —
    "whatever score/balance is currently showing", not re-scoped to this
    feature's own window slider. One computation feeds both the Customer
    Score and Current Balance filters, plus the two columns of the same
    name always shown in the final audience table. Leading underscore on
    the DataFrame params tells st.cache_data to hash them by identity/
    cheaply rather than full content (they're already cached upstream by
    Analytics)."""
    sales_raw = _sales_raw if isinstance(_sales_raw, pd.DataFrame) else pd.DataFrame()
    if sales_raw.empty:
        return pd.DataFrame(columns=["cusid", "composite_score", "current_balance"])
    coll_df = _coll_df if isinstance(_coll_df, pd.DataFrame) else pd.DataFrame()
    ar_df = _load_ar_balance(str(zid), proj)
    cacus_df = _load_cacus(str(zid))
    result = build_customer_marketing_table(
        sales_df=sales_raw,
        collection_df=coll_df,
        ar_df=ar_df,
        selected_years=selected_years,
        cacus_df=cacus_df if not cacus_df.empty else None,
    )
    if result is None or result.empty:
        return pd.DataFrame(columns=["cusid", "composite_score", "current_balance"])
    keep = [c for c in ("cusid", "composite_score", "current_balance") if c in result.columns]
    out = result[keep].copy()
    out["cusid"] = out["cusid"].astype(str)
    return out


def _wfb_map_from_metrics(metrics: pd.DataFrame, col: str) -> dict:
    """cusid -> col value, dropping rows where col is NaN — shared by the
    Customer Score and Current Balance filters/table columns."""
    if metrics is None or metrics.empty or col not in metrics.columns:
        return {}
    sub = metrics.dropna(subset=[col])
    return dict(zip(sub["cusid"], sub[col]))


def _wfb_compute_candidates(zid: str, window_months: int, filters: list, upto: int = None, score_map: dict = None, balance_map: dict = None) -> set:
    """Replays `filters` (in add-order) from scratch, returning the
    candidate cusid set after applying filters[:upto] (all of them if
    upto is None). Recomputed fresh each call — simplest correct approach;
    revisit with incremental caching only if this proves slow once real
    campaigns are actually using it, not before."""
    from processing import wf_bulk_audience as wfb

    start_date, end_date = wfb.window_dates(window_months)
    active = filters if upto is None else filters[:upto]

    def _all_customers():
        cdf = _load_cacus(str(zid))
        return set(cdf["cusid"].dropna().astype(str)) if not cdf.empty else set()

    candidates = None
    area_pool = None

    for f in active:
        ftype = f["type"]
        if ftype == "area":
            if area_pool is None:
                area_pool = wfb.load_area_salesman_pool(zid, start_date, end_date)
            qualifying = wfb.apply_area(area_pool, f["value"])
        elif ftype == "salesman":
            if area_pool is None:
                area_pool = wfb.load_area_salesman_pool(zid, start_date, end_date)
            qualifying = wfb.apply_salesman(area_pool, f["area"], f["value"])
        elif ftype == "product_count_band":
            lines = wfb.load_sales_lines(zid, start_date, end_date, cusids=candidates)
            qualifying = wfb.apply_product_count_band(lines, f["min"], f["max"])
        elif ftype == "days_since_last_sale":
            last_sale = wfb.load_last_sale_dates(zid, cusids=candidates)
            qualifying = wfb.apply_days_since_last_sale(last_sale, f["min"], f["max"])
        elif ftype == "customer_score":
            qualifying = {cid for cid, v in (score_map or {}).items() if f["min"] <= v <= f["max"]}
        elif ftype == "current_balance":
            qualifying = {cid for cid, v in (balance_map or {}).items() if f["min"] <= v <= f["max"]}
        elif ftype == "net_sales":
            lines = wfb.load_sales_lines(zid, start_date, end_date, cusids=candidates)
            ret_lines = wfb.load_returns_lines(zid, start_date, end_date, cusids=candidates)
            qualifying = wfb.apply_net_sales_band(lines, ret_lines, f["min"], f["max"])
        elif ftype == "total_returns":
            ret_lines = wfb.load_returns_lines(zid, start_date, end_date, cusids=candidates)
            qualifying = wfb.apply_total_returns_band(ret_lines, f["min"], f["max"])
        elif ftype == "product":
            lines = wfb.load_sales_lines(zid, start_date, end_date, cusids=candidates)
            qualifying = wfb.apply_product_filter(lines, f["value"])
        elif ftype == "inactive":
            base = candidates if candidates is not None else _all_customers()
            # Own independent window, NOT the shared slider -- per explicit
            # follow-up ask: "how long since anything happened" is a
            # different question from what the shared window otherwise
            # scopes, same reasoning as Avg Collection Days above.
            inact_start, inact_end = wfb.window_dates(f.get("window_months", wfb.DEFAULT_WINDOW_MONTHS))
            lines = wfb.load_sales_lines(zid, inact_start, inact_end, cusids=base)
            qualifying = wfb.apply_inactive(base, lines)
        elif ftype == "order_date":
            qualifying = wfb.apply_order_date(zid, f["value"])
        elif ftype == "collection_date":
            qualifying = wfb.apply_collection_date(zid, f["value"])
        elif ftype == "avg_collection_days":
            base = candidates if candidates is not None else _all_customers()
            # Own independent window, NOT the shared slider — per explicit
            # ask, since it's a genuinely different question ("how long a
            # history to compute this specific average over") from "how
            # recently did they buy/sell" that the shared window answers.
            acd_start, acd_end = wfb.window_dates(f.get("window_months", wfb.DEFAULT_WINDOW_MONTHS))
            avg_df = wfb.compute_avg_collection_days(zid, acd_start, acd_end, base)
            qualifying = wfb.apply_avg_collection_days(avg_df, f["min"], f["max"])
        else:
            qualifying = candidates if candidates is not None else set()

        candidates = qualifying if candidates is None else (candidates & qualifying)

    return candidates if candidates is not None else _all_customers()


def _wfb_render_new_filter_input(ftype: str, zid: str, window_months: int, filters: list, candidates_so_far, score_map: dict, balance_map: dict = None) -> dict:
    """Renders the input widget(s) for a filter type being added, scoped
    against `candidates_so_far` (whoever survived the filters already
    active) — options only show what's actually still reachable, per
    explicit ask, not the full unfiltered universe. Returns the filter
    dict once there's something valid to add, else None."""
    from processing import wf_bulk_audience as wfb

    start_date, end_date = wfb.window_dates(window_months)
    gen = st.session_state["_wfb_add_gen"]

    if ftype == "area":
        pool = wfb.load_area_salesman_pool(zid, start_date, end_date)
        if candidates_so_far is not None:
            pool = pool[pool["cusid"].astype(str).isin(candidates_so_far)]
        opts = wfb.area_options(pool)
        if not opts:
            st.info("No areas found for the current candidates in this window.")
            return None
        vals = st.multiselect(
            "Areas — a customer qualifies if sold to in ANY of the selected areas", opts,
            key=f"wfb_new_area_{gen}",
        )
        if not vals:
            st.caption("Select at least one area above to enable Add.")
            return None
        return {"type": "area", "value": vals}

    if ftype == "salesman":
        area_f = next((f for f in filters if f["type"] == "area"), None)
        if area_f is None:
            return None
        pool = wfb.load_area_salesman_pool(zid, start_date, end_date)
        opts = wfb.salesman_options(pool, area_f["value"])
        if not opts:
            st.info("No salesman sold in ANY of the selected areas within this window.")
            return None
        coverage = wfb.salesman_area_coverage(pool, area_f["value"])
        labels = [f"{spid} - {spname} ({', '.join(coverage.get(spid, []))})" for spid, spname in opts]
        st.caption(
            "Shown here if active in ANY of the selected areas — the area(s) in parentheses are "
            "which of your selected areas they actually cover. Picking one only pulls their "
            "customers from the areas you selected, even if they also sell elsewhere."
        )
        idxs = st.multiselect(
            "Salesmen — a customer qualifies if sold to by ANY of the selected salesmen", range(len(opts)),
            format_func=lambda i: labels[i], key=f"wfb_new_sp_{gen}",
        )
        if not idxs:
            st.caption("Select at least one salesman above to enable Add.")
            return None
        return {
            "type": "salesman",
            "area": area_f["value"],
            "value": [opts[i][0] for i in idxs],
            "labels": [labels[i] for i in idxs],
        }

    if ftype == "product_count_band":
        lines = wfb.load_sales_lines(zid, start_date, end_date, cusids=candidates_so_far)
        counts = wfb.unique_product_counts(lines)
        if counts.empty:
            st.info("No purchase data for the current candidates in this window.")
            return None
        lo, hi = int(counts.min()), int(counts.max())
        c1, c2 = st.columns(2)
        mn = c1.number_input("Min unique products", min_value=0, value=lo, key=f"wfb_new_pcb_min_{gen}")
        mx = c2.number_input("Max unique products", min_value=0, value=hi, key=f"wfb_new_pcb_max_{gen}")
        return {"type": "product_count_band", "min": mn, "max": mx}

    if ftype == "days_since_last_sale":
        last_sale = wfb.load_last_sale_dates(zid, cusids=candidates_so_far)
        if last_sale.empty:
            st.info("No sales history for the current candidates.")
            return None
        lo, hi = int(last_sale["days_since"].min()), int(last_sale["days_since"].max())
        c1, c2 = st.columns(2)
        mn = c1.number_input("Min days since last sale", min_value=0, value=lo, key=f"wfb_new_dsls_min_{gen}")
        mx = c2.number_input("Max days since last sale", min_value=0, value=hi, key=f"wfb_new_dsls_max_{gen}")
        return {"type": "days_since_last_sale", "min": mn, "max": mx}

    if ftype == "customer_score":
        vals = [v for cid, v in (score_map or {}).items() if candidates_so_far is None or cid in candidates_so_far]
        if not vals:
            st.info("No scored customers among the current candidates.")
            return None
        lo, hi = float(min(vals)), float(max(vals))
        c1, c2 = st.columns(2)
        mn = c1.number_input("Min score", value=lo, key=f"wfb_new_score_min_{gen}")
        mx = c2.number_input("Max score", value=hi, key=f"wfb_new_score_max_{gen}")
        return {"type": "customer_score", "min": mn, "max": mx}

    if ftype == "current_balance":
        vals = [v for cid, v in (balance_map or {}).items() if candidates_so_far is None or cid in candidates_so_far]
        if not vals:
            st.info("No balance data for the current candidates.")
            return None
        lo, hi = float(min(vals)), float(max(vals))
        st.caption("Current AR balance — debit-positive means the customer owes money; not scoped to the time window.")
        c1, c2 = st.columns(2)
        mn = c1.number_input("Min balance", value=lo, key=f"wfb_new_bal_min_{gen}")
        mx = c2.number_input("Max balance", value=hi, key=f"wfb_new_bal_max_{gen}")
        return {"type": "current_balance", "min": mn, "max": mx}

    if ftype == "net_sales":
        lines = wfb.load_sales_lines(zid, start_date, end_date, cusids=candidates_so_far)
        ret_lines = wfb.load_returns_lines(zid, start_date, end_date, cusids=candidates_so_far)
        net = wfb.net_sales_by_customer(lines, ret_lines)
        if net.empty:
            st.info("No sales data for the current candidates in this window.")
            return None
        lo, hi = float(net.min()), float(net.max())
        st.caption(
            f"Net of returns (sales − returns) between {start_date} and {end_date} — "
            "same window as the slider above."
        )
        c1, c2 = st.columns(2)
        mn = c1.number_input("Min net sales", value=lo, key=f"wfb_new_ns_min_{gen}")
        mx = c2.number_input("Max net sales", value=hi, key=f"wfb_new_ns_max_{gen}")
        return {"type": "net_sales", "min": mn, "max": mx}

    if ftype == "total_returns":
        ret_lines = wfb.load_returns_lines(zid, start_date, end_date, cusids=candidates_so_far)
        totals = wfb.total_returns_by_customer(ret_lines)
        if totals.empty:
            st.info("No returns for the current candidates in this window.")
            return None
        lo, hi = float(totals.min()), float(totals.max())
        st.caption(f"Total returns between {start_date} and {end_date} — same window as the slider above.")
        c1, c2 = st.columns(2)
        mn = c1.number_input("Min total returns", value=lo, key=f"wfb_new_tr_min_{gen}")
        mx = c2.number_input("Max total returns", value=hi, key=f"wfb_new_tr_max_{gen}")
        return {"type": "total_returns", "min": mn, "max": mx}

    if ftype == "product":
        lines = wfb.load_sales_lines(zid, start_date, end_date, cusids=candidates_so_far)
        opts = wfb.product_options(lines)
        n_candidates = len(candidates_so_far) if candidates_so_far is not None else "all"
        st.caption(
            f"{len(opts)} product(s) sold to the {n_candidates} current candidate(s) between {start_date} "
            f"and {end_date}. A product with zero sales to them in this window won't be listed below."
        )

        with st.expander("🔍 Check whether a specific product was sold in this window at all"):
            query = st.text_input("Search by product code or name", key=f"wfb_prod_check_{gen}")
            if query.strip():
                all_lines = wfb.load_sales_lines(zid, start_date, end_date)  # unscoped by candidates
                q = query.strip().lower()
                hits = all_lines[
                    all_lines["itemcode"].astype(str).str.lower().str.contains(q, na=False)
                    | all_lines["itemname"].astype(str).str.lower().str.contains(q, na=False)
                ]
                if hits.empty:
                    st.warning(f"No sales at all for “{query}” in this window ({start_date} to {end_date}).")
                else:
                    total_buyers = hits["cusid"].nunique()
                    among_candidates = (
                        hits[hits["cusid"].astype(str).isin(candidates_so_far)]["cusid"].nunique()
                        if candidates_so_far is not None else total_buyers
                    )
                    matches = sorted(set(f"{r.itemcode} - {r.itemname}" for r in hits.itertuples()))
                    does_or_not = "does" if among_candidates else "doesn't"
                    st.success(
                        f"Sold to {total_buyers} customer(s) total in this window — {among_candidates} of them "
                        f"among your current candidates. That's why it {does_or_not} show up in the list below."
                    )
                    shown = ", ".join(matches[:10]) + (f" (+{len(matches) - 10} more)" if len(matches) > 10 else "")
                    st.caption(f"Matched: {shown}")

        if not opts:
            st.info("No products found for the current candidates in this window.")
            return None
        labels = [f"{code} - {name}" for code, name in opts]
        idxs = st.multiselect(
            "Products — must have bought EVERY selected one at least once", range(len(opts)),
            format_func=lambda i: labels[i], key=f"wfb_new_prod_{gen}",
        )
        if not idxs:
            st.caption("Select at least one product above to enable Add.")
            return None
        return {"type": "product", "value": [opts[i][0] for i in idxs], "labels": [labels[i] for i in idxs]}

    if ftype == "inactive":
        inact_window = st.slider(
            "Time window for this calculation (months) — independent of the main slider above",
            min_value=1, max_value=24, value=window_months, key=f"wfb_new_inactive_window_{gen}",
            help="How far back to check for ANY purchase. A customer with no purchase in this "
                 "window qualifies as inactive, regardless of what the shared window above is set to.",
        )
        inact_start, inact_end = wfb.window_dates(inact_window)
        st.caption(f"No purchase at all between {inact_start} and {inact_end}.")
        return {"type": "inactive", "window_months": inact_window}

    if ftype == "order_date":
        val = st.date_input("Ordered on", key=f"wfb_new_odate_{gen}")
        return {"type": "order_date", "value": val}

    if ftype == "collection_date":
        val = st.date_input("Collection received on", key=f"wfb_new_cdate_{gen}")
        return {"type": "collection_date", "value": val}

    if ftype == "avg_collection_days":
        st.caption(
            "Computed only for the current candidates — this is why it's always added last, "
            "so the per-customer calculation stays cheap. This is the average number of days "
            "it took each customer to pay AFTER their last sale (not the average gap between "
            "one collection and the next)."
        )
        acd_window = st.slider(
            "Time window for this calculation (months) — independent of the main slider above",
            min_value=1, max_value=36, value=window_months, key=f"wfb_new_acd_window_{gen}",
            help="How far back sales/returns/collections are pulled just for this one filter. "
                 "A sale that happened before this window started won't be seen, so a collection "
                 "near the start of the window can look like it has no matching sale — widen this "
                 "if that's throwing off the average.",
        )
        c1, c2 = st.columns(2)
        mn = c1.number_input("Min avg days to collection", min_value=0.0, value=0.0, key=f"wfb_new_acd_min_{gen}")
        mx = c2.number_input("Max avg days to collection", min_value=0.0, value=100.0, key=f"wfb_new_acd_max_{gen}")
        return {"type": "avg_collection_days", "min": mn, "max": mx, "window_months": acd_window}

    return None


def _show_wf_bulk_messaging(zid: str, proj: str, data_dict: dict, selected_years: list) -> None:
    st.subheader("📢 WhatsFly — Bulk Messaging")
    st.caption("Build an audience with filters, then pick what to send them.")

    try:
        whatsfly.get_credentials()
    except whatsfly.WhatsFlyConfigError as e:
        st.warning(str(e))
        return

    from processing import wf_bulk_audience as wfb

    st.session_state.setdefault("_wfb_filters", [])
    st.session_state.setdefault("_wfb_add_gen", 0)
    filters = st.session_state["_wfb_filters"]
    active_types = {f["type"] for f in filters}

    window_months = st.slider(
        "Time window (months)", min_value=1, max_value=24, value=wfb.DEFAULT_WINDOW_MONTHS, step=1,
        key="wfb_window_months",
        help="Governs Area/Salesman, Unique Products Bought, Net Sales, Total Returns, and Product. Days "
             "Since Last Sale, Customer Score, Current Balance, Inactive, Avg Collection Days, and the "
             "two exact-date filters each have their own independent scope.",
    )
    start_date, end_date = wfb.window_dates(window_months)
    st.caption(f"Window: {start_date} → {end_date}")

    _sr = data_dict.get("sales")
    sales_raw = _sr if isinstance(_sr, pd.DataFrame) else pd.DataFrame()
    _cd = data_dict.get("collection")
    coll_df = _cd if isinstance(_cd, pd.DataFrame) else pd.DataFrame()
    metrics = _wfb_customer_metrics(
        str(zid), proj, sales_raw, coll_df, tuple(int(y) for y in selected_years) if selected_years else tuple(),
    )
    score_map = _wfb_map_from_metrics(metrics, "composite_score")
    balance_map = _wfb_map_from_metrics(metrics, "current_balance")

    st.markdown("**🧰 Audience Filters**")

    for i, f in enumerate(filters):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"**{_BULK_FILTER_LABELS[f['type']]}** — {_wfb_describe_filter(f)}")
            with c2:
                if st.button("✕ Remove", key=f"wfb_remove_{i}"):
                    st.session_state["_wfb_filters"] = filters[:i]
                    st.rerun()

    candidates_so_far = _wfb_compute_candidates(str(zid), window_months, filters, score_map=score_map, balance_map=balance_map)
    st.metric("Customers matching so far", f"{len(candidates_so_far):,}")

    available = _wfb_available_to_add(active_types)
    if available:
        gen = st.session_state["_wfb_add_gen"]
        options = ["— choose a filter to add —"] + [label for _, label in available]
        choice_label = st.selectbox("➕ Add a filter", options, key=f"wfb_add_selector_{gen}")
        if choice_label != options[0]:
            chosen_type = next(k for k, lbl in available if lbl == choice_label)
            with st.container(border=True):
                new_filter = _wfb_render_new_filter_input(
                    chosen_type, str(zid), window_months, filters, candidates_so_far, score_map, balance_map,
                )
                if new_filter is not None and st.button("✅ Add this filter", key=f"wfb_confirm_add_{gen}"):
                    st.session_state["_wfb_filters"] = filters + [new_filter]
                    st.session_state["_wfb_add_gen"] = gen + 1
                    st.rerun()
    elif "avg_collection_days" in active_types:
        st.caption("Avg Collection Days is active — remove it to add any further filters.")
    else:
        st.caption("Every filter type is active.")

    if filters and st.button("🗑️ Clear all filters", key="wfb_clear_all"):
        st.session_state["_wfb_filters"] = []
        st.session_state["_wfb_excluded_cusids"] = set()
        st.rerun()

    st.session_state.setdefault("_wfb_excluded_cusids", set())

    st.markdown("---")
    st.markdown(f"**👥 Audience — {len(candidates_so_far):,} customers matched by filters**")
    if candidates_so_far:
        cacus_df = _load_cacus(str(zid))
        audience_df = cacus_df[cacus_df["cusid"].astype(str).isin(candidates_so_far)].copy()
        audience_df["cusid"] = audience_df["cusid"].astype(str)

        # Always shown, regardless of which filters are active — Net Sales
        # uses the SAME shared window as the rest of the feature (not
        # build_customer_marketing_table's own sidebar-year scope, and not
        # any per-filter independent window like Avg Collection Days/
        # Inactive have); Current Balance/Score are the same "currently
        # showing" snapshot the Customer Score/Current Balance filters use.
        # Net of returns, not gross — same "Net Sales" meaning as the
        # Net Sales filter and everywhere else in this app (Common
        # Pitfall #1 in CLAUDE.md).
        sales_window_lines = wfb.load_sales_lines(str(zid), start_date, end_date, cusids=candidates_so_far)
        returns_window_lines = wfb.load_returns_lines(str(zid), start_date, end_date, cusids=candidates_so_far)
        net_sales_map = wfb.net_sales_by_customer(sales_window_lines, returns_window_lines).to_dict()
        audience_df["Net Sales (window)"] = audience_df["cusid"].map(net_sales_map).fillna(0.0)
        audience_df["Current Balance"] = audience_df["cusid"].map(balance_map)
        audience_df["Current Score"] = audience_df["cusid"].map(score_map)

        show_cols = [c for c in ["cusid", "cusname", "cusmobile", "whatsapp", "area"] if c in audience_df.columns]
        show_cols += ["Net Sales (window)", "Current Balance", "Current Score"]

        # Contactable-only gate — must have BOTH Mobile and WhatsApp on
        # file, not just one (deliberately stricter than the "either
        # works" fallback the single-message WhatsFly panel uses
        # elsewhere) — always applied, per explicit ask, not an optional
        # filter in the builder above. Count always shown, even when 0,
        # so the check is visibly running rather than silently no-op.
        contactable_df, n_dropped_phone = wfb.apply_contactable_only(audience_df[show_cols])
        st.caption(f"📵 {n_dropped_phone:,} customer(s) excluded — missing Mobile or WhatsApp number on file.")

        if contactable_df.empty:
            st.warning("No customers with both a Mobile and WhatsApp number on file match the current filters.")
        else:
            # Multiselect-to-exclude instead of an editable checkbox grid —
            # st.data_editor's canvas-rendered grid proved unreliable to
            # drive/verify (a real, known rough edge for that widget), so
            # this swaps to a plain multiselect: pick who to REMOVE, and
            # the final table/CSV below is just everyone else.
            contactable_df = contactable_df.reset_index(drop=True)
            current_cusids = set(contactable_df["cusid"])
            label_for_cusid = {
                row.cusid: f"{row.cusid} - {row.cusname}" for row in contactable_df.itertuples()
            }
            cusid_for_label = {v: k for k, v in label_for_cusid.items()}

            persisted_excluded = st.session_state["_wfb_excluded_cusids"]
            default_labels = [label_for_cusid[c] for c in persisted_excluded if c in current_cusids]
            # Fingerprint the widget key to the current row set, same
            # reasoning as elsewhere in this feature — a filter/window
            # change swaps in a different candidate set, and this forces a
            # fresh widget instance instead of Streamlit trying to
            # validate a stale `default` against options that no longer
            # contain it. The persisted exclusion set (by cusid, synced
            # below) is what actually survives across that change, not
            # the widget's own state.
            _fingerprint = hash(tuple(sorted(current_cusids)))
            st.markdown("**✏️ Final review** — pick any customers below to remove them from the final list, even ones the filters matched.")
            selected_labels = st.multiselect(
                "Remove from final list",
                options=list(label_for_cusid.values()),
                default=default_labels,
                key=f"wfb_exclude_ms_{_fingerprint}",
            )
            selected_cusids = {cusid_for_label[l] for l in selected_labels}
            # Cusids currently out of view keep whatever exclusion state
            # they already had; only the currently-visible set is
            # reconciled against what the widget just returned.
            st.session_state["_wfb_excluded_cusids"] = (persisted_excluded - current_cusids) | selected_cusids

            final_df = contactable_df[~contactable_df["cusid"].isin(st.session_state["_wfb_excluded_cusids"])]
            st.metric("Final list (after phone check + manual review)", f"{len(final_df):,}")
            st.dataframe(final_df, width="stretch", hide_index=True)
            st.download_button(
                "📥 Download audience (CSV)",
                final_df.to_csv(index=False).encode("utf-8"),
                file_name=f"bulk_audience_{zid}.csv", mime="text/csv", key="wfb_download_csv",
            )

    st.markdown("---")
    st.markdown("**✉️ Template**")
    st.caption("Pick a template — the view below is built per campaign as each one gets defined.")

    if st.button("🔄 Refresh Templates", key="wf_bulk_refresh_templates_btn"):
        st.session_state.pop("_wf_templates_raw", None)

    if "_wf_templates_raw" not in st.session_state:
        with st.spinner("Fetching templates…"):
            try:
                st.session_state["_wf_templates_raw"] = whatsfly.get_templates()
            except Exception as e:
                st.error(f"Couldn't fetch templates: {e}")
                return

    templates = _wf_normalize_templates(st.session_state["_wf_templates_raw"])
    if not templates:
        st.warning("No templates found.")
        return

    labels = [_wf_template_label(t, i) for i, t in enumerate(templates)]
    idx = st.selectbox(
        f"📋 Template ({len(templates)} available)", range(len(templates)),
        format_func=lambda i: labels[i], key="wf_bulk_template_idx",
    )
    template = templates[idx]

    st.markdown("---")
    handler = _WF_BULK_TEMPLATE_VIEWS.get(template.get("template_name"), _wf_bulk_default_view)
    handler(zid, template)


# ---------------------------------------------------------------------------
# 📨 Direct WhatsApp — single-message test panel, straight to Meta's own
# WhatsApp Cloud API (graph.facebook.com), no WhatsFly in between.
#
# Same test-phase scope as WhatsFly Messaging above: send ONE message to ONE
# number, pull templates, fill in variables, see the raw response — against
# a separate Meta test WABA + test number (config/direct_whatsapp.ini), not
# the real WhatsFly-routed production number. Reuses the generic
# markup/preview helpers defined above (_wf_format_whatsapp_markup,
# _wf_render_bubble, _wf_substitute_preview, _wf_extract_variable_tokens,
# _wf_extract_components) — those are plain WhatsApp-template rendering
# helpers, not WhatsFly-specific, and Meta's own template shape
# (top-level `components: [{type, text}]`) is exactly what
# _wf_extract_components already parses.
#
# Unlike the WhatsFly panel, there's no per-variable NAME *field* or
# payload-shape guessing here — Meta's Cloud API contract is officially
# documented (not reverse-engineered) and there's exactly one real request
# shape. But Meta templates DO come in two placeholder formats, chosen at
# template-creation time (never mixed within one template): positional
# (`{{1}}`, `{{2}}`) or named (`{{cusname}}`, `{{cuscode}}`). For a named
# template, each body parameter sent to Meta must carry a `parameter_name`
# matching the token — taken straight from the body text itself, not typed
# in by hand, since Meta (unlike WhatsFly) ties the name to the approved
# template, not to metadata chosen at send time.
# ---------------------------------------------------------------------------


def _dwa_normalize_templates(raw) -> list:
    """Meta's documented shape: {"data": [...], "paging": {...}}."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        val = raw.get("data")
        if isinstance(val, list):
            return val
    return []


def _dwa_template_label(t: dict, i: int) -> str:
    if not isinstance(t, dict):
        return f"Template {i + 1}"
    name = t.get("name") or f"Template {i + 1}"
    category = t.get("category")
    lang = t.get("language")
    label = name
    if category:
        label += f" — {category}"
    if lang:
        label += f" ({lang})"
    return label


def _dwa_templates_table(templates: list) -> pd.DataFrame:
    rows = [
        {
            "ID": t.get("id", "—"),
            "Name": t.get("name", "—"),
            "Category": t.get("category", "—"),
            "Language": t.get("language", "—"),
            "Status": t.get("status", "—"),
        }
        for t in templates
    ]
    return pd.DataFrame(rows)


def _render_dwa_response(resp) -> None:
    st.markdown("---")
    st.markdown(f"**HTTP status:** `{resp.status_code}`")
    try:
        body = resp.json()
    except ValueError:
        st.code(resp.text or "(empty response body)")
        return

    # Meta's own contract: success carries "messages": [{"id": "wamid...."}]
    # and no "error" key; failure carries a nested "error": {message, type,
    # code, error_subcode, fbtrace_id} and a non-2xx status.
    is_ok = resp.ok and isinstance(body, dict) and "error" not in body
    if is_ok:
        st.success("Sent — see raw response below (look for the `wamid...` message id).")
    else:
        st.error("Meta rejected the request — see raw response below.")
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict) and err.get("message"):
            st.info(
                f"**{err.get('type', 'Error')} (code {err.get('code', '—')})**: {err['message']}"
                + (f" — {err['error_data']['details']}" if isinstance(err.get("error_data"), dict) and err["error_data"].get("details") else "")
            )
    st.json(body)


def _render_dwa_text_send(phone_number: str) -> None:
    st.caption(
        "Session message — only works within 24h of the recipient last "
        "messaging the test number. A template message (below) is required "
        "to start a new conversation."
    )
    message = st.text_area("Message", key="dwa_text_message", height=100)
    if st.button("📤 Send Text Message", key="dwa_send_text_btn"):
        if not phone_number.strip():
            st.error("Enter a recipient phone number first.")
            return
        if not message.strip():
            st.error("Message is empty.")
            return
        with st.spinner("Sending…"):
            try:
                resp = direct_whatsapp.send_text(phone_number.strip(), message)
            except Exception as e:
                st.error(f"Send failed: {e}")
                return
        _render_dwa_response(resp)


def _render_dwa_template_send(phone_number: str) -> None:
    if st.button("🔄 Load / Refresh Templates", key="dwa_refresh_templates_btn"):
        st.session_state.pop("_dwa_templates_raw", None)

    if "_dwa_templates_raw" not in st.session_state:
        with st.spinner("Fetching templates…"):
            try:
                st.session_state["_dwa_templates_raw"] = direct_whatsapp.get_templates()
            except Exception as e:
                st.error(f"Couldn't fetch templates: {e}")
                return

    raw = st.session_state["_dwa_templates_raw"]
    with st.expander("🔍 Raw template list response"):
        st.json(raw)

    templates = _dwa_normalize_templates(raw)
    if not templates:
        st.warning("No templates found in the response above — expand it to see the actual shape returned.")
        return

    st.markdown(f"**{len(templates)} template(s) available on this test WABA:**")
    st.dataframe(_dwa_templates_table(templates), width="stretch", hide_index=True)

    labels = [_dwa_template_label(t, i) for i, t in enumerate(templates)]
    idx = st.selectbox(
        "Select a template to send", range(len(templates)), format_func=lambda i: labels[i], key="dwa_template_idx"
    )
    template = templates[idx]

    with st.expander("🔍 Selected template (raw)"):
        st.json(template)

    comps = _wf_extract_components(template)
    body_text = comps["body"]
    tokens = _wf_extract_variable_tokens(body_text)
    is_named_format = bool(tokens) and not tokens[0].isdigit()
    body_html = _wf_format_whatsapp_markup(body_text)

    st.markdown("**Template Preview**")
    if body_text:
        _wf_render_bubble(comps["header"], body_html, comps["footer"])
    else:
        st.caption("No body text found for this template — check the raw JSON above to see the actual shape returned.")

    variable_values = []  # parallel to `tokens` — Meta matches each entry to
    # its own {{token}} by POSITION for a positional template (`{{1}}`,
    # `{{2}}`, ...), or by the token text itself (sent as `parameter_name`)
    # for a named template (`{{cusname}}`, `{{cuscode}}`, ...); no free-form
    # naming here, unlike WhatsFly's send contract, since Meta ties the name
    # to the approved template itself.
    if tokens:
        st.markdown(f"**Fill in {len(tokens)} variable(s)** — the preview below updates as you type:")
        if is_named_format:
            st.caption("Named-parameter template — each value below is sent tagged with its own `{{name}}`.")
        for tok in tokens:
            vval = st.text_input(f"Value for {{{{{tok}}}}}", key=f"dwa_var_{idx}_{tok}")
            variable_values.append(vval)

        st.markdown("**Message Preview (with your edits)**")
        _wf_render_bubble(comps["header"], _wf_substitute_preview(body_html, tokens, variable_values), comps["footer"])
    else:
        st.caption("No {{...}} variables detected in this template's body.")

    has_image_header = st.checkbox(
        "🖼️ This template's header is an image",
        key=f"dwa_has_img_header_{idx}",
        help="Meta requires JPEG/PNG, 5MB max, for image headers.",
    )
    header_media_id = None
    header_media_url = None
    if has_image_header:
        uploaded_file = st.file_uploader(
            "Attach header image", type=["jpg", "jpeg", "png"], key=f"dwa_header_img_{idx}"
        )
        manual_url = st.text_input(
            "…or paste an already-hosted image URL instead",
            key=f"dwa_header_img_url_{idx}",
        )
        st.caption(
            "Uploading goes through Meta's own `/media` endpoint and yields a "
            "`media_id`, used as `image: {id: ...}` in the request — Meta also "
            "accepts a plain hosted `image: {link: ...}` URL as a fallback, "
            "which the manual-URL box below feeds instead."
        )

        if uploaded_file is not None:
            st.image(uploaded_file, width=200)
            file_sig = (uploaded_file.name, uploaded_file.size)
            upload_cache_key = f"dwa_header_upload_{idx}"
            cached = st.session_state.get(upload_cache_key)
            if not cached or cached.get("sig") != file_sig:
                with st.spinner("Uploading image to Meta…"):
                    try:
                        raw_upload = direct_whatsapp.upload_media(
                            uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type or "image/jpeg"
                        )
                        mid = raw_upload.get("id") if isinstance(raw_upload, dict) else None
                        st.session_state[upload_cache_key] = {
                            "sig": file_sig, "raw": raw_upload, "media_id": mid, "error": None,
                        }
                    except Exception as e:
                        st.session_state[upload_cache_key] = {"sig": file_sig, "raw": None, "error": str(e)}

            cached = st.session_state.get(upload_cache_key)
            if cached and cached.get("error"):
                st.error(f"Upload failed: {cached['error']}")
            elif cached and cached.get("raw") is not None:
                with st.expander("🔍 Raw upload response"):
                    st.json(cached["raw"])
                if cached.get("media_id"):
                    st.success(f"Uploaded — media_id: `{cached['media_id']}`")
                    header_media_id = cached["media_id"]
                else:
                    st.warning("Uploaded, but no `id` found in the response — check the raw JSON above.")

        if not header_media_id and manual_url.strip():
            header_media_url = manual_url.strip()

    components = []
    if header_media_id:
        components.append({"type": "header", "parameters": [{"type": "image", "image": {"id": header_media_id}}]})
    elif header_media_url:
        components.append({"type": "header", "parameters": [{"type": "image", "image": {"link": header_media_url}}]})
    if variable_values:
        if is_named_format:
            # Named-parameter template — each parameter must carry the
            # token as `parameter_name`, matched by name rather than by
            # position (Meta's requirement, confirmed against the docs).
            body_params = [
                {"type": "text", "parameter_name": tok, "text": v} for tok, v in zip(tokens, variable_values)
            ]
        else:
            body_params = [{"type": "text", "text": v} for v in variable_values]
        components.append({"type": "body", "parameters": body_params})

    with st.expander("⚙️ Send request details"):
        st.caption(
            "Meta's documented shape: `POST /{phone_number_id}/messages` with a "
            "nested `template: {name, language: {code}, components: [...]}` body."
        )
        template_name_val = st.text_input(
            "Template name", value=template.get("name", ""), key=f"dwa_template_name_{idx}",
        )
        language_code_val = st.text_input(
            "Language code", value=template.get("language", "en_US") or "en_US", key=f"dwa_lang_{idx}",
        )

        components_key = f"dwa_components_json_{idx}"
        if components_key not in st.session_state:
            st.session_state[components_key] = json.dumps(components, indent=2)

        if st.button("↻ Rebuild components from fields above", key=f"dwa_rebuild_components_{idx}"):
            st.session_state[components_key] = json.dumps(components, indent=2)
            st.rerun()

        st.caption("`template.components` that will be sent — edit directly if needed.")
        components_text = st.text_area("Components JSON", key=components_key, height=140)

    if st.button("📤 Send Template Message", key="dwa_send_template_btn"):
        if not phone_number.strip():
            st.error("Enter a recipient phone number first.")
            return
        if not template_name_val.strip():
            st.error("Template name is empty.")
            return
        try:
            components_val = json.loads(components_text)
        except json.JSONDecodeError as e:
            st.error(f"Components isn't valid JSON: {e}")
            return
        with st.spinner("Sending…"):
            try:
                resp = direct_whatsapp.send_template(
                    phone_number.strip(), template_name_val.strip(), language_code_val.strip(), components_val
                )
            except Exception as e:
                st.error(f"Send failed: {e}")
                return
        _render_dwa_response(resp)


def _show_direct_whatsapp_messaging() -> None:
    st.subheader("📨 Direct WhatsApp — Send Test Message")
    st.caption(
        "Sends straight to Meta's WhatsApp Cloud API (graph.facebook.com) — no "
        "WhatsFly in between. Same single-message test flow as WhatsFly "
        "Messaging above, against a separate Meta test WABA + test number "
        "(config/direct_whatsapp.ini), so nothing here touches the real "
        "WhatsFly-routed production number."
    )

    try:
        direct_whatsapp.get_credentials()
    except direct_whatsapp.DirectWhatsAppConfigError as e:
        st.warning(str(e))
        return

    msg_type = st.radio(
        "Message type",
        ["Approved Template", "Plain Text (session message)"],
        horizontal=True,
        key="dwa_msg_type",
    )
    phone_number = st.text_input(
        "Recipient phone number",
        key="dwa_phone_number",
        help="Country code + digits only — no '+', no spaces, e.g. 8801XXXXXXXXX. "
             "A Meta test number can only message numbers added to its recipient "
             "list in the Meta App Dashboard.",
    )

    st.markdown("---")

    if msg_type.startswith("Plain Text"):
        _render_dwa_text_send(phone_number)
    else:
        _render_dwa_template_send(phone_number)


# ---------------------------------------------------------------------------
# 📥 WhatsApp Message Log — read-only viewer into the whatsapp_webhook
# service's own database (whatsapp_webhooks), so a message sent via the
# panels above (or a real customer reply) can be verified end-to-end without
# leaving Streamlit. This app never writes to that database — see
# core/whatsapp_webhook_db.py.
# ---------------------------------------------------------------------------

def _preview_wa_content(content) -> str:
    if not isinstance(content, dict):
        return ""
    return str(content.get("body") or content.get("caption") or "")[:120]


def _show_whatsapp_message_log() -> None:
    st.subheader("📥 WhatsApp Message Log")
    st.caption(
        "Read-only view into the whatsapp_webhook service's own database — "
        "confirms a send (WhatsFly / Direct WhatsApp above) actually reached "
        "Meta, and shows inbound replies as they arrive. This app never "
        "writes here."
    )

    try:
        counts = whatsapp_webhook_db.get_counts()
    except whatsapp_webhook_db.WhatsAppWebhookDBConfigError as e:
        st.warning(str(e))
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Webhook Events Logged", counts["webhook_events"])
    c2.metric("Messages", counts["messages"])
    c3.metric("Known Contacts", counts["contacts"])

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col_b:
        limit = st.number_input("Rows to show", min_value=10, max_value=1000, value=100, step=10)

    try:
        rows = whatsapp_webhook_db.get_recent_messages(int(limit))
    except whatsapp_webhook_db.WhatsAppWebhookDBConfigError as e:
        st.warning(str(e))
        return
    except Exception as e:
        st.error(f"Could not load messages: {e}")
        return

    if not rows:
        st.info("No messages recorded yet.")
        return

    df = pd.DataFrame(rows)
    df["preview"] = df["content"].apply(_preview_wa_content)
    display_df = df.rename(columns={
        "wamid": "Message ID", "direction": "Direction", "contact_phone": "Phone",
        "contact_name": "Contact", "message_type": "Type", "template_name": "Template",
        "current_status": "Status", "message_timestamp": "Sent/Received At",
        "created_at": "Logged At", "preview": "Preview",
    })[["Direction", "Phone", "Contact", "Type", "Template", "Status",
        "Preview", "Sent/Received At", "Logged At", "Message ID"]]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Drill down on one message** — status history + raw payload")
    wamid_options = df["wamid"].tolist()
    if wamid_options:
        chosen = st.selectbox("Message ID (wamid)", wamid_options)
        chosen_row = df[df["wamid"] == chosen].iloc[0]
        st.json(chosen_row["content"] if isinstance(chosen_row["content"], dict) else {})
        try:
            history = whatsapp_webhook_db.get_status_history(chosen)
        except Exception as e:
            st.error(f"Could not load status history: {e}")
            history = []
        if history:
            st.table(pd.DataFrame(history))
        else:
            st.caption(
                "No status events yet for this message — normal for a fresh "
                "inbound message, or an outbound one still in flight."
            )


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

_PRODUCT_ONLY_MODES = {
    "📈 High Stock Marketing", "🖼️ Media Library", "📱 Inactive Outreach", "🎣 Leads",
    "💬 WhatsFly Messaging", "📢 Bulk Messaging", "📨 Direct WhatsApp", "📥 WhatsApp Message Log",
}


def display_marketing_analysis(zid: str, proj: str, data_dict: dict, selected_years: list):
    st.title("Marketing Analysis")

    # ── Mode radio FIRST so product-only modes skip the salesman/area filters ──
    mode = st.radio(
        "View",
        [
            "📊 Customer Scoring",
            "🎯 Area Campaign Planner",
            "📱 Inactive Outreach",
            "📈 High Stock Marketing",
            "🖼️ Media Library",
            "🎣 Leads",
            "💬 WhatsFly Messaging",
            "📢 Bulk Messaging",
            # "📨 Direct WhatsApp" — shut off, not deleted. WhatsFly is the
            # path being developed now (see CLAUDE.md); the Direct WhatsApp
            # code (core/direct_whatsapp.py, _show_direct_whatsapp_messaging,
            # and its dispatch/_PRODUCT_ONLY_MODES entries below) is left in
            # place, just unreachable via this radio — re-add the string
            # here to bring it back.
            # "📥 WhatsApp Message Log" — shut off, not deleted, same as
            # Direct WhatsApp above. Conversation history now shows inline
            # per-customer in WhatsFly Messaging itself, making this
            # standalone browse-everything view redundant for day-to-day
            # use; the code (_show_whatsapp_message_log, dispatch/
            # _PRODUCT_ONLY_MODES entries below) is untouched — re-add the
            # string here to bring it back.
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Product-only modes need neither salesman/area filters nor the heavy
    # customer scoring build — dispatch immediately and return.
    _sr = data_dict.get("sales")
    sales_raw = _sr if isinstance(_sr, pd.DataFrame) else pd.DataFrame()
    coll_df   = data_dict.get("collection")

    if mode in _PRODUCT_ONLY_MODES:
        if mode == "🖼️ Media Library":
            _show_media_library(str(zid))
        elif mode == "📱 Inactive Outreach":
            _show_inactive_outreach(str(zid), proj, sales_raw)
        elif mode == "🎣 Leads":
            _show_leads(str(zid))
        elif mode == "💬 WhatsFly Messaging":
            _show_whatsfly_messaging(str(zid))
        elif mode == "📢 Bulk Messaging":
            _show_wf_bulk_messaging(str(zid), proj, data_dict, selected_years)
        elif mode == "📨 Direct WhatsApp":
            _show_direct_whatsapp_messaging()
        elif mode == "📥 WhatsApp Message Log":
            _show_whatsapp_message_log()
        else:
            _show_high_stock_marketing(str(zid), proj)
        return

    if not isinstance(sales_raw, pd.DataFrame) or sales_raw.empty:
        st.info("No sales data available for the selected filters.")
        return

    # ── Inline salesman + area filters ───────────────────────────────────────
    sp_opts = sorted(sales_raw["spname"].dropna().astype(str).unique().tolist())

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        _sp_default_idx = 1 if sp_opts else 0
        sp_sel = st.selectbox(
            "Salesman",
            ["All Salesmen"] + sp_opts,
            index=_sp_default_idx,
            key="mkt_inline_sp",
        )

    # Area options cascade from selected salesman
    if sp_sel == "All Salesmen":
        area_pool = sorted(sales_raw["area"].dropna().astype(str).unique().tolist())
    else:
        area_pool = sorted(
            sales_raw[sales_raw["spname"].astype(str) == sp_sel]["area"]
            .dropna().astype(str).unique().tolist()
        )

    with f_col2:
        area_sel = st.multiselect(
            "Area",
            area_pool,
            default=area_pool,
            key="mkt_inline_area",
        )

    # Apply filters — empty area_sel means all areas for the salesman
    sales_df = sales_raw.copy()
    if sp_sel != "All Salesmen":
        sales_df = sales_df[sales_df["spname"].astype(str) == sp_sel]
    if area_sel:
        sales_df = sales_df[sales_df["area"].isin(area_sel)]

    if sales_df.empty:
        st.info("No sales data for the selected salesman / area combination.")
        return

    st.markdown("---")

    # ── Load supporting data ─────────────────────────────────────────────────
    with st.spinner("Loading supporting data…"):
        ar_df    = _load_ar_balance(str(zid), proj)
        cacus_df = _load_cacus(str(zid))

    with st.spinner("Building customer performance table…"):
        result = build_customer_marketing_table(
            sales_df=sales_df,
            collection_df=coll_df if coll_df is not None else pd.DataFrame(),
            ar_df=ar_df,
            selected_years=tuple(int(y) for y in selected_years),
            cacus_df=cacus_df if not cacus_df.empty else None,
        )

    if result.empty:
        st.warning("No results to display for the selected filters.")
        return

    # build_customer_marketing_table uses an outer merge so collection/AR data
    # can re-introduce customers outside the salesman/area filter.  Restrict the
    # result to only customers who actually appear in the filtered sales data.
    filtered_cusids = set(sales_df["cusid"].dropna().astype(str).unique())
    result = result[result["cusid"].astype(str).isin(filtered_cusids)].reset_index(drop=True)

    if mode == "📊 Customer Scoring":
        _show_customer_scoring(result)
    else:
        _show_campaign_planner(result, sales_df, str(zid))
