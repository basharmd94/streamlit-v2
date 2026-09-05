import html as _html
import json
import re
import shutil
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from core import whatsfly
from core import direct_whatsapp

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
from processing.common import normalize_phone_cols
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
    lang = _wf_guess(t, ("language", "language_code", "lang"))
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
            "Language": _wf_guess(t, ("language", "language_code", "lang")) or "—",
            "Status": _wf_guess(t, ("status", "template_status")) or "—",
        }
        for t in templates
    ]
    return pd.DataFrame(rows)


def _wf_extract_components(t: dict) -> dict:
    """Best-effort pull of header/body/footer text from a template entry so
    the send panel can render a beautified preview. Real key names aren't
    confirmed against a live WhatsFly response, so this checks common flat
    keys first, then Meta Cloud API's own
    `components: [{type: HEADER/BODY/FOOTER, text: ...}]` shape, then a
    recursive fallback scan for any string containing '{{' (for the body)."""
    out = {"header": None, "body": "", "footer": None}
    if not isinstance(t, dict):
        return out

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


def _wf_extract_variable_tokens(body_text: str) -> list:
    """Ordered, de-duplicated variable tokens found in a template body.
    Meta templates support two mutually-exclusive placeholder formats,
    chosen when the template is created — **positional** (`{{1}}`, `{{2}}`,
    ...) or **named** (`{{cusname}}`, `{{cuscode}}`, ...) — never mixed
    within one template. Each token here is the raw text inside `{{ }}`,
    either a digit string or a name, in first-appearance order."""
    seen = []
    for m in _VAR_TOKEN_RE.finditer(body_text or ""):
        tok = m.group(1)
        if tok not in seen:
            seen.append(tok)
    return seen


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
    st.caption(
        "Session message — only works within 24h of the recipient last "
        "messaging the business number. Use this after the rep messages "
        "in first (the guide's suggested first-test shortcut)."
    )
    message = st.text_area("Message", key="wf_text_message", height=100)
    if st.button("📤 Send Text Message", key="wf_send_text_btn"):
        if not phone_number.strip():
            st.error("Enter a recipient phone number first.")
            return
        if not message.strip():
            st.error("Message is empty.")
            return
        with st.spinner("Sending…"):
            try:
                resp = whatsfly.send_text(phone_number.strip(), message)
            except Exception as e:
                st.error(f"Send failed: {e}")
                return
        _render_wf_response(resp)


def _render_wf_template_send(phone_number: str) -> None:
    if st.button("🔄 Load / Refresh Templates", key="wf_refresh_templates_btn"):
        st.session_state.pop("_wf_templates_raw", None)

    if "_wf_templates_raw" not in st.session_state:
        with st.spinner("Fetching templates…"):
            try:
                st.session_state["_wf_templates_raw"] = whatsfly.get_templates()
            except Exception as e:
                st.error(f"Couldn't fetch templates: {e}")
                return

    raw = st.session_state["_wf_templates_raw"]
    with st.expander("🔍 Raw template list response"):
        st.json(raw)

    templates = _wf_normalize_templates(raw)
    if not templates:
        st.warning("No templates found in the response above — expand it to see the actual shape returned.")
        return

    st.markdown(f"**{len(templates)} template(s) available on this account:**")
    st.dataframe(_wf_templates_table(templates), width="stretch", hide_index=True)

    labels = [_wf_template_label(t, i) for i, t in enumerate(templates)]
    idx = st.selectbox(
        "Select a template to send", range(len(templates)), format_func=lambda i: labels[i], key="wf_template_idx"
    )
    template = templates[idx]

    with st.expander("🔍 Selected template (raw)"):
        st.json(template)

    comps = _wf_extract_components(template)
    body_text = comps["body"]
    tokens = _wf_extract_variable_tokens(body_text)
    body_html = _wf_format_whatsapp_markup(body_text)

    st.markdown("**Template Preview**")
    if body_text:
        _wf_render_bubble(comps["header"], body_html, comps["footer"])
    else:
        st.caption("No body text found for this template — check the raw JSON above to see the actual shape returned.")

    named_variables = []  # list of (name, value) — WhatsFly's SEND contract always
    # keys each variable by a NAME (`templateVariable-<name>-<n>`, confirmed via a
    # real dashboard example), regardless of which placeholder format the template
    # body itself uses. Meta templates support two formats, chosen at template-
    # creation time (never mixed within one template): positional (`{{1}}`,
    # `{{2}}`) or named (`{{cusname}}`, `{{cuscode}}`). When the body already uses
    # named tokens, that token IS the name WhatsFly needs — defaulted straight
    # from it. For positional bodies there's no name in the text itself, so
    # `{{1}}`/`{{2}}` fall back to this account's confirmed convention
    # (`CUSNAME`/`CUSCODE`, both real examples used exactly this pair); a 3rd+
    # positional variable has no confirmed name and stays blank.
    if tokens:
        st.markdown(f"**Fill in {len(tokens)} variable(s)** — the preview below updates as you type:")
        st.caption(
            "Name defaults to the template's own `{{name}}` token if it uses named "
            "placeholders, else this account's confirmed convention (`CUSNAME`/`CUSCODE` "
            "for `{{1}}`/`{{2}}`) — override if needed. Sent as `templateVariable-<name>-<n>` "
            "per variable, per a real dashboard example."
        )
        for i, tok in enumerate(tokens):
            c1, c2 = st.columns([1, 2])
            with c1:
                if not tok.isdigit():
                    default_name = tok
                else:
                    default_name = _WF_DEFAULT_VAR_NAMES[i] if i < len(_WF_DEFAULT_VAR_NAMES) else ""
                vname = st.text_input(
                    f"Name for {{{{{tok}}}}}", value=default_name, key=f"wf_varname_{idx}_{i}"
                )
            with c2:
                vval = st.text_input(f"Value for {{{{{tok}}}}}", key=f"wf_var_{idx}_{i}")
            named_variables.append((vname, vval))

        st.markdown("**Message Preview (with your edits)**")
        _wf_render_bubble(
            comps["header"], _wf_substitute_preview(body_html, tokens, [v for _, v in named_variables]), comps["footer"]
        )
    else:
        st.caption("No {{...}} variables detected in this template's body.")

    has_image_header = st.checkbox(
        "🖼️ This template's header is an image",
        key=f"wf_has_img_header_{idx}",
        help=(
            "Not auto-detected — this account's template-list response doesn't "
            "expose header format, so check it manually against the template "
            "and flip this on. Meta requires JPEG/PNG, 5MB max, for image headers."
        ),
    )
    header_media_id = None
    header_media_url = None
    if has_image_header:
        uploaded_file = st.file_uploader(
            "Attach header image", type=["jpg", "jpeg", "png"], key=f"wf_header_img_{idx}"
        )
        manual_url = st.text_input(
            "…or paste an already-hosted image URL instead",
            key=f"wf_header_img_url_{idx}",
        )
        st.caption(
            "**Confirmed via a real dashboard example** (image-header template): the "
            "flat shape wants a plain **`template_header_media_url`** field — a hosted "
            "URL, not a media id and not `media_type` (both of those were wrong guesses, "
            "now removed from the default). The uploaded image's own hosted URL feeds "
            "this automatically; the manual-URL box below is the same field if you'd "
            "rather skip uploading."
        )

        if uploaded_file is not None:
            st.image(uploaded_file, width=200)
            file_sig = (uploaded_file.name, uploaded_file.size)
            upload_cache_key = f"wf_header_upload_{idx}"
            cached = st.session_state.get(upload_cache_key)
            if not cached or cached.get("sig") != file_sig:
                with st.spinner("Uploading image to WhatsFly…"):
                    try:
                        raw = whatsfly.upload_media(
                            uploaded_file.getvalue(), uploaded_file.name, uploaded_file.type or "image/jpeg"
                        )
                        mid, murl = _wf_extract_media_ref(raw)
                        st.session_state[upload_cache_key] = {
                            "sig": file_sig, "raw": raw, "media_id": mid, "media_url": murl, "error": None,
                        }
                    except Exception as e:
                        st.session_state[upload_cache_key] = {"sig": file_sig, "raw": None, "error": str(e)}

            cached = st.session_state.get(upload_cache_key)
            if cached and cached.get("error"):
                st.error(f"Upload failed: {cached['error']}")
            elif cached and cached.get("raw") is not None:
                with st.expander("🔍 Raw upload response"):
                    st.json(cached["raw"])
                if cached.get("media_id") or cached.get("media_url"):
                    st.success(
                        f"Uploaded — media_id: `{cached.get('media_id') or '—'}`, "
                        f"url: `{cached.get('media_url') or '—'}`"
                    )
                    header_media_id = cached.get("media_id")
                    header_media_url = cached.get("media_url")
                else:
                    st.warning("Uploaded, but couldn't find an id/url in the response — check the raw JSON above.")

        if not header_media_id and not header_media_url and manual_url.strip():
            header_media_url = manual_url.strip()

    with st.expander("⚙️ Send request details"):
        st.caption(
            "**Confirmed via a real dashboard-generated example**: endpoint is "
            "`POST /whatsapp/send/template` with flat params — but its `template_id` "
            "param is a naming trap: it wants WhatsFly's short internal **`id`** "
            "(e.g. `435966`), NOT the longer `template_id` field the template-list "
            "response itself returns for the same template. Defaulted below accordingly."
        )
        template_id_val = st.text_input(
            "Template ID for send (WhatsFly's short `id` — confirmed, not the list's `template_id`)",
            value=_wf_guess(template, ("id", "template_id", "wa_template_id", "uuid")) or "",
            key=f"wf_template_id_{idx}",
        )
        endpoint = st.text_input(
            "Send endpoint",
            value="/whatsapp/send/template",
            key="wf_endpoint",
            help="Confirmed via a real dashboard-generated curl/GET example — this is the actual path.",
        )

        payload_shape = st.radio(
            "Payload shape to try",
            ["Flat (template_id + templateVariable-<name>-<n>)", "Meta Cloud API style (nested template/components)"],
            horizontal=True,
            key=f"wf_payload_shape_{idx}",
            help=(
                "Flat is confirmed via TWO real dashboard-generated examples now "
                "(text-only, and this image-header one) — templateVariable-<name>-<n> "
                "per variable, template_header_media_url for the header image, no "
                "template_name/language_code in either real example. Meta Cloud API "
                "style is kept only as a fallback; the send endpoint appears to key "
                "its own template lookup off the flat template_id specifically, so "
                "this shape (no flat template_id) got \"Message template not found\" "
                "on a real attempt."
            ),
        )
        header_image_param = (
            {"id": header_media_id} if header_media_id
            else {"link": header_media_url} if header_media_url
            else None
        )

        if payload_shape.startswith("Meta"):
            template_name = st.text_input(
                "Template name (`template.name` — required for this shape only)",
                value=_wf_guess(template, ("template_name", "name", "elementName")) or "",
                key=f"wf_template_name_{idx}",
            )
            language_code = st.text_input(
                "Language code",
                value=_wf_guess(template, ("language", "language_code", "lang")) or "en",
                key=f"wf_lang_{idx}",
            )
            components = []
            if header_image_param:
                components.append({"type": "header", "parameters": [{"type": "image", "image": header_image_param}]})
            body_values = [v for _, v in named_variables]
            if body_values:
                components.append({"type": "body", "parameters": [{"type": "text", "text": v} for v in body_values]})
            default_payload = {
                "template": {
                    "name": template_name,
                    "language": {"code": language_code},
                    "components": components,
                },
            }
            shape_key = "meta"
        else:
            # Confirmed via a real dashboard example against an image-header,
            # named-variable template — templateVariable-<name>-<n> per variable
            # (NOT a numbered/generic array), template_header_media_url as a plain
            # hosted URL (NOT media_id/media_url/media_type, all wrong guesses from
            # the first attempt). template_name/language_code weren't present in
            # either real example, so dropped from the default entirely.
            default_payload = {"template_id": template_id_val}
            for i, (vname, vval) in enumerate(named_variables):
                if vname.strip():
                    default_payload[f"templateVariable-{vname.strip()}-{i + 1}"] = vval
            if header_media_url:
                default_payload["template_header_media_url"] = header_media_url
            shape_key = "flat"

        payload_key = f"wf_payload_json_{idx}_{shape_key}"
        if payload_key not in st.session_state:
            st.session_state[payload_key] = json.dumps(default_payload, indent=2)

        if st.button("↻ Rebuild payload from fields above", key=f"wf_rebuild_payload_{idx}_{shape_key}"):
            st.session_state[payload_key] = json.dumps(default_payload, indent=2)
            st.rerun()

        st.caption(
            "Request body that will be sent (merged with apiToken/phone_number_id/phone_number) "
            "— edit directly if the real API expects something different still."
        )
        payload_text = st.text_area("Payload JSON", key=payload_key, height=140)

    if st.button("📤 Send Template Message", key="wf_send_template_btn"):
        if not phone_number.strip():
            st.error("Enter a recipient phone number first.")
            return
        try:
            extra_params = json.loads(payload_text)
        except json.JSONDecodeError as e:
            st.error(f"Payload isn't valid JSON: {e}")
            return
        with st.spinner("Sending…"):
            try:
                resp = whatsfly.send_template(phone_number.strip(), endpoint.strip(), extra_params)
            except Exception as e:
                st.error(f"Send failed: {e}")
                return
        _render_wf_response(resp)


def _show_whatsfly_messaging() -> None:
    st.subheader("💬 WhatsFly — Send Test Message")
    st.caption(
        "Single-message test phase: send one message to one number and see what "
        "comes back. No reply/webhook handling here yet — that's a separate "
        "FastAPI service, a later phase (see Whatsfly_Integration_docs/)."
    )

    try:
        whatsfly.get_credentials()
    except whatsfly.WhatsFlyConfigError as e:
        st.warning(str(e))
        return

    msg_type = st.radio(
        "Message type",
        ["Approved Template", "Plain Text (session message)"],
        horizontal=True,
        key="wf_msg_type",
    )
    phone_number = st.text_input(
        "Recipient phone number",
        key="wf_phone_number",
        help="Country code + digits only — no '+', no spaces, e.g. 8801XXXXXXXXX.",
    )

    st.markdown("---")

    if msg_type.startswith("Plain Text"):
        _render_wf_text_send(phone_number)
    else:
        _render_wf_template_send(phone_number)


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
# public entry point
# ---------------------------------------------------------------------------

_PRODUCT_ONLY_MODES = {
    "📈 High Stock Marketing", "🖼️ Media Library", "📱 Inactive Outreach", "🎣 Leads",
    "💬 WhatsFly Messaging", "📨 Direct WhatsApp",
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
            "📨 Direct WhatsApp",
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
            _show_whatsfly_messaging()
        elif mode == "📨 Direct WhatsApp":
            _show_direct_whatsapp_messaging()
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
