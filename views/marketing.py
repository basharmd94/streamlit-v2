import streamlit as st
import pandas as pd
import numpy as np

from processing.marketing import (
    build_customer_marketing_table,
    build_area_campaign_top_customers,
    build_area_top_products,
    build_stock_gap,
    build_inactive_customers,
)
from processing.common import normalize_phone_cols
from core.analytics import Analytics


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
def _load_stock_for_gap(zid: str) -> pd.DataFrame:
    """Load stock movements for 100001+100009 (cross-ZID) or the given ZID."""
    zids = ["100001", "100009"] if zid == "100001" else [zid]
    frames = []
    for z in zids:
        df = Analytics("stock", zid=z, filters={}).data
        if df is not None and not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_sales_alltime(zid: str, proj: str) -> pd.DataFrame:
    """All-time sales (no year filter) — used for inactive outreach last-order dates."""
    df = Analytics("sales", zid=zid, project=proj, filters={}).data
    return df if df is not None else pd.DataFrame()


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

    total_rows = len(display_df)
    cap = 50_000
    if total_rows > cap:
        st.info(f"Showing first {cap:,} of {total_rows:,} rows. Use Download for full data.")
        display_df = display_df.head(cap)

    st.dataframe(display_df, use_container_width=True)

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
        st.dataframe(disp_cus, use_container_width=True)

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
        st.dataframe(disp_prod, use_container_width=True)

    # ── Section C: Stock gap opportunities ───────────────────────────────────
    st.markdown("#### 🔍 Stock Gap Opportunities")
    st.caption(
        "Products **in stock** whose **product group** has sold in the current filter, "
        "but this specific product has **not yet sold** here."
    )

    with st.spinner("Loading stock data…"):
        stock_df = _load_stock_for_gap(zid)

    gap_df = pd.DataFrame()
    if stock_df.empty:
        st.info("No stock data available.")
    else:
        warehouses = sorted(stock_df["warehouse"].dropna().unique().tolist())
        selected_wh = st.multiselect(
            "Warehouses", warehouses, default=warehouses, key="camp_wh"
        )
        gap_df = build_stock_gap(
            sales_df, stock_df,
            warehouses=selected_wh if selected_wh else None,
        )
        if gap_df.empty:
            st.success("No gap found — all in-stock items for matching groups have already sold here.")
        else:
            st.info(f"**{len(gap_df)}** products in stock not yet sold here, but their group has sold here.")
            disp_gap = gap_df.copy()
            disp_gap["stock_qty"] = disp_gap["stock_qty"].apply(
                lambda v: f"{v:,.2f}" if pd.notna(v) else ""
            )
            disp_gap = disp_gap.rename(columns={
                "itemcode": "Item Code", "itemname": "Item Name", "itemgroup": "Group",
                "warehouse": "Warehouse", "stock_qty": "Stock Qty",
            })
            st.dataframe(disp_gap, use_container_width=True)

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
    if not gap_df.empty:
        gf = gap_df.copy()
        gf.insert(0, "section", "Stock Gap")
        frames.append(gf)

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

**Section C — Stock Gap Opportunities**
Products sitting in your warehouse whose *product group* sells in this area, but this specific
SKU has never been ordered by anyone in the filter. These are expansion opportunities:
- **For 100001/100009**: actual on-hand stock shown by warehouse — you have the goods ready.
- **For 100000/100005**: the product group is known to sell here; push these SKUs to your agents.

**Download → Campaign Report CSV** bundles all three sections into one file.
Each row has a `section` column so you can filter/sort in Excel.
        """)


# ---------------------------------------------------------------------------
# inactive outreach sub-view
# ---------------------------------------------------------------------------

def _show_inactive_outreach(
    zid: str,
    proj: str,
    cacus_df: pd.DataFrame,
    sp_filter: str = None,
    area_filter: list = None,
) -> None:
    months = st.slider(
        "Inactive for more than (months)", min_value=1, max_value=12, value=6,
        key="outreach_months",
    )

    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=months)
    st.caption(
        f"Customers with **no orders since {cutoff.strftime('%d %b %Y')}** "
        f"({months} month{'s' if months != 1 else ''} ago)"
    )

    with st.spinner("Loading all-time sales…"):
        sales_all = _load_sales_alltime(zid, proj)

    if sales_all.empty:
        st.warning("No sales data available.")
        return

    # Apply inline salesman/area filter to all-time data
    if sp_filter:
        sales_all = sales_all[sales_all["spname"].fillna("") == sp_filter]
    if area_filter:
        sales_all = sales_all[sales_all["area"].isin(area_filter)]

    inactive = build_inactive_customers(sales_all, cacus_df=cacus_df, months=months)

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

    st.dataframe(disp, use_container_width=True)

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


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def display_marketing_analysis(zid: str, proj: str, data_dict: dict, selected_years: list):
    st.title("Marketing Analysis")

    sales_raw = data_dict.get("sales")
    coll_df   = data_dict.get("collection")

    if sales_raw is None or (isinstance(sales_raw, pd.DataFrame) and sales_raw.empty):
        st.info("No sales data available for the selected filters.")
        return

    # ── Inline salesman + area filters ───────────────────────────────────────
    sp_opts = sorted(sales_raw["spname"].dropna().astype(str).unique().tolist())

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        sp_sel = st.selectbox(
            "Salesman",
            ["All Salesmen"] + sp_opts,
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

    # ── View radio ───────────────────────────────────────────────────────────
    mode = st.radio(
        "View",
        ["📊 Customer Scoring", "🎯 Area Campaign Planner", "📱 Inactive Outreach"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    sp_filter   = None if sp_sel == "All Salesmen" else sp_sel
    area_filter = area_sel if area_sel else None

    if mode == "📊 Customer Scoring":
        _show_customer_scoring(result)
    elif mode == "🎯 Area Campaign Planner":
        _show_campaign_planner(result, sales_df, str(zid))
    else:
        _show_inactive_outreach(str(zid), proj, cacus_df, sp_filter=sp_filter, area_filter=area_filter)
