import streamlit as st
import pandas as pd
import numpy as np

from processing.marketing import build_customer_marketing_table
from core.analytics import Analytics


# ---------------------------------------------------------------------------
# column display config
# ---------------------------------------------------------------------------

_DISPLAY_LABELS = {
    "cusid":                        "Customer ID",
    "cusname":                      "Customer Name",
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
}

_CURRENCY_COLS = {"total_sales", "total_collection", "current_balance"}
_PCT_COLS = {"yoy_sales_growth_pct", "yoy_collection_growth_pct", "monthly_activity_rate"}
_DAYS_COLS = {
    "avg_days_to_collection",
    "avg_days_between_collections",
    "avg_order_interval_days",
}

_NOTES = """
### Column Reference

| Column | Formula / Source | Notes |
|---|---|---|
| **Total Sales** | `SUM(altsales)` from `mv_sales_line_items` for selected year(s) + filters | Gross sales before discount; consistent with IS Revenue |
| **Total Collection** | `SUM(value)` from `mv_collection_vouchers` for selected year(s) + filters | Includes RCT, CRCT, BRCT, JV, STJV, ADJV voucher types |
| **Sales YoY Growth %** | *(Year₂ − Year₁) / Year₁ × 100*. For 3+ years: average of successive annual growth rates | N/A when only 1 year is selected |
| **Collection YoY Growth %** | Same formula applied to Total Collection by year | N/A when only 1 year is selected |
| **Avg Days to Collection** | For each collection event: days since that customer's most recent invoice date. Averaged across all such events in the selected period | Customers with zero collection events are excluded |
| **Avg Days Between Collections** | Mean gap in days between consecutive collection vouchers per customer | Requires ≥ 2 collection events; customers with fewer are shown as blank |
| **Avg Order Interval (days)** | Mean gap in days between consecutive distinct order dates per customer | Requires ≥ 2 orders; measures how often the customer reorders |
| **Monthly Activity Rate %** | Active months with ≥ 1 order ÷ total calendar months in selected period × 100 | 2 years selected → denominator is 24; a customer who ordered in 7 of those 24 months scores 29.2% |
| **Current Balance** | `SUM(xprime)` across *all* AR ledger history (`mv_ar_transactions`) | Not year-filtered — reflects the live outstanding balance. Positive = customer owes; negative = customer is in credit |

### Year Aggregation
When multiple years are selected, sales and collection columns are summed across the full period. Growth metrics compare year-by-year within the selection. Interval and frequency metrics use all transaction dates in the period as a single continuous window (not per-year averages).

### Filter Logic
The sidebar Salesman and Area filters restrict which customers appear by matching against the sales data. The Current Balance is always computed from the full AR ledger (no year restriction) so it reflects the customer's actual live balance regardless of the selected period.
"""


# ---------------------------------------------------------------------------
# load AR balance (not year-filtered)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _load_ar_balance(zid: str) -> pd.DataFrame:
    df = Analytics("ar_due_ledger", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def display_marketing_analysis(zid: str, data_dict: dict, selected_years: list):
    st.title("Marketing Analysis")

    sales_df = data_dict.get("sales")
    coll_df = data_dict.get("collection")

    if sales_df is None or (isinstance(sales_df, pd.DataFrame) and sales_df.empty):
        st.info("No sales data available for the selected filters.")
        return

    ar_df = _load_ar_balance(str(zid))

    with st.spinner("Building customer performance table…"):
        result = build_customer_marketing_table(
            sales_df=sales_df,
            collection_df=coll_df if coll_df is not None else pd.DataFrame(),
            ar_df=ar_df,
            selected_years=tuple(int(y) for y in selected_years),
        )

    if result.empty:
        st.warning("No results to display for the selected filters.")
        return

    # ── summary metrics row ──────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers", f"{len(result):,}")
    m2.metric("Total Sales", _fmt_currency(result.get("total_sales", pd.Series()).sum()))
    m3.metric(
        "Total Collection",
        _fmt_currency(result.get("total_collection", pd.Series()).sum()),
    )
    balance_total = result.get("current_balance", pd.Series(dtype=float))
    if not isinstance(balance_total, pd.Series):
        balance_total = pd.Series(dtype=float)
    m4.metric("Outstanding Balance", _fmt_currency(balance_total.sum()))

    st.markdown("---")

    # ── rename + format ──────────────────────────────────────────────────────
    display_df = result.copy()
    display_df = display_df.rename(
        columns={k: v for k, v in _DISPLAY_LABELS.items() if k in display_df.columns}
    )

    # Format columns for display
    for raw_col, label in _DISPLAY_LABELS.items():
        if label not in display_df.columns:
            continue
        if raw_col in _CURRENCY_COLS:
            display_df[label] = display_df[label].apply(
                lambda v: f"{v:,.0f}" if pd.notna(v) else ""
            )
        elif raw_col in _PCT_COLS:
            display_df[label] = display_df[label].apply(
                lambda v: f"{v:+.1f}%" if pd.notna(v) and str(v) != "nan" else ""
            )
        elif raw_col in _DAYS_COLS:
            display_df[label] = display_df[label].apply(
                lambda v: f"{v:.1f}" if pd.notna(v) else ""
            )

    # ── search + filter ──────────────────────────────────────────────────────
    search = st.text_input("Search customer name or ID", "")
    if search:
        cname_col = _DISPLAY_LABELS.get("cusname", "Customer Name")
        cid_col = _DISPLAY_LABELS.get("cusid", "Customer ID")
        mask = (
            display_df.get(cname_col, pd.Series(dtype=str))
            .astype(str)
            .str.contains(search, case=False, na=False)
            | display_df.get(cid_col, pd.Series(dtype=str))
            .astype(str)
            .str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    total_rows = len(display_df)
    cap = 50_000
    if total_rows > cap:
        st.info(f"Showing first {cap:,} of {total_rows:,} rows. Use Download for full data.")
        display_df = display_df.head(cap)

    st.dataframe(display_df, use_container_width=True)

    # ── download ─────────────────────────────────────────────────────────────
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV",
        data=csv,
        file_name="marketing_analysis.csv",
        mime="text/csv",
    )

    # ── notes expander ───────────────────────────────────────────────────────
    with st.expander("📋 Column Definitions & Calculation Notes", expanded=False):
        st.markdown(_NOTES)


# ---------------------------------------------------------------------------
# helpers
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
