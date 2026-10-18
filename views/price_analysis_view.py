# views/price_analysis_view.py
# UI for Accounting Analysis -> "💲 Price Analysis". Two sections:
#   1. Total operating-expense change (06xx SG&A + 07xx S&D, Level 0) over a
#      chosen timeline -- the Financial Statements "Expense Changes" rows,
#      totalled.
#   2. A per-product price table (stock, wholesale tier, realized price).
# Pure display; the pandas lives in processing/price_analysis.py.

from __future__ import annotations

import calendar

import pandas as pd
import streamlit as st

from core.analytics import Analytics
from processing import common, financial
from processing import price_analysis as pa

_TIMELINES = ["Monthly", "Quarterly", "Yearly"]
_MAX_PERIODS = {"Monthly": 12, "Quarterly": 8, "Yearly": 5}
_YEARLY_SPAN = 5          # years loaded for the Yearly timeline


# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_final_items(zid: str) -> pd.DataFrame:
    df = Analytics("final_items_view", zid=zid, filters={}).data
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_wh_tiers(zid: str) -> pd.DataFrame:
    df = Analytics("opspprc", zid=zid, filters={}).data
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_price_window(zid: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = Analytics(
        "item_price_window", zid=zid,
        filters={"start_date": start_date, "end_date": end_date},
    ).data
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


# ── Level 0 expense frame per timeline ───────────────────────────────────────

def _income_label_df() -> pd.DataFrame:
    labels = common.load_json("data/labels.json")["income_statement_label"]
    return pd.DataFrame(list(labels.items()), columns=["ac_lv4", "Income Statement"])


def _last_complete_month(today: pd.Timestamp) -> tuple[int, int]:
    prev = today.replace(day=1) - pd.Timedelta(days=1)
    return prev.year, prev.month


def _load_expense_frame(zid: str, project, timeline: str, end_year: int, end_month: int):
    """Level 0 income-statement frame (account rows x period columns) for
    the chosen timeline, built with the exact Financial Statements builders
    (process_data_month / collapse_monthly_to_quarterly / process_data), so
    the totals tie to its Level 0 "Expense Changes" table. Returns
    (frame, last_month_included) -- last_month_included is only meaningful
    for Quarterly (flags a partial final quarter)."""
    label_df = _income_label_df()
    excluded = {"Asset", "Liability"}

    if timeline in ("Monthly", "Quarterly"):
        pl = financial.process_data_month(
            zid, end_year, 1, end_month, "Income Statement", label_df, project, excluded
        )
        # Same trick the Quarterly Financial Statements view uses: a second
        # call for the prior year contributes the year-2 columns too.
        try:
            prev = financial.process_data_month(
                zid, end_year - 1, 1, 12, "Income Statement", label_df, project, excluded
            )
            yr2 = [c for c in prev.columns if isinstance(c, tuple) and c[0] == end_year - 2]
            if yr2:
                pl = pl.merge(prev[["ac_code"] + yr2], on="ac_code", how="left")
        except Exception:
            pass
        monthly_cols = pa.period_columns(pl)
        last_month = int(monthly_cols[-1][1]) if monthly_cols else 0
        if timeline == "Quarterly":
            pl, _ = financial.collapse_monthly_to_quarterly(pl, pl)
        return pl, last_month

    today = pd.Timestamp.today()
    years = [end_year - i for i in range(_YEARLY_SPAN - 1, -1, -1)]
    full_years = [y for y in years if y < today.year]
    pl = None
    if full_years:
        pl = financial.process_data(
            zid, full_years, 1, 12, "Income Statement", label_df, project, excluded
        )
    if end_year == today.year:
        ytd = financial.process_data(
            zid, [today.year], 1, today.month - 1, "Income Statement", label_df, project, excluded
        )
        pl = ytd if pl is None else pl.merge(ytd[["ac_code", today.year]], on="ac_code", how="left")
    return pl, 0


# ── Section 1: total expense change ──────────────────────────────────────────

def _fmt_money(v: float) -> str:
    return f"{v:,.0f}"


def _render_expense_change(zid: str) -> None:
    st.markdown("### Operating expense change")
    st.caption(
        "Level 0 income statement, 06xx SG&A + 07xx S&D accounts — the same rows as "
        "Financial Statements → Level 0 → Expense Changes, added up. "
        "**+** = costs went up · **−** = costs went down."
    )

    today = pd.Timestamp.today().normalize()
    lc_year, lc_month = _last_complete_month(today)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        timeline = st.selectbox("Timeline", _TIMELINES, key="pa_timeline")
    with c2:
        end_year = st.number_input(
            "End Year", min_value=2018, max_value=today.year,
            value=lc_year, step=1, key="pa_end_year",
        )
    with c3:
        end_month = st.number_input(
            "End Month", min_value=1, max_value=12, value=lc_month, step=1,
            key="pa_end_month", disabled=(timeline == "Yearly"),
        )
    with c4:
        max_p = _MAX_PERIODS[timeline]
        n_periods = st.number_input(
            "Periods to compare", min_value=2, max_value=max_p,
            value=2, step=1, key=f"pa_n_periods_{timeline}",
        )

    end_year, end_month = int(end_year), int(end_month)
    if timeline != "Yearly" and (end_year, end_month) > (lc_year, lc_month):
        end_year, end_month = lc_year, lc_month
        st.caption(
            f"Latest completed month is {calendar.month_abbr[lc_month]} {lc_year} "
            f"— using that as the end period."
        )
    if timeline == "Yearly" and end_year == today.year and today.month == 1:
        st.info(f"No completed month in {today.year} yet — pick {today.year - 1} or earlier.")
        return

    project = st.session_state.get("proj")
    try:
        with st.spinner("Loading Level 0 income statement…"):
            pl, last_month = _load_expense_frame(str(zid), project, timeline, end_year, end_month)
    except ValueError as e:
        st.warning(f"Nothing to load for that period: {e}")
        return
    except Exception as e:
        st.warning(f"Could not load the Level 0 income statement: {e}")
        return

    if pl is None or pl.empty:
        st.info("No income-statement data for this entity and period.")
        return

    totals = pa.build_expense_totals(pl, timeline, int(n_periods))
    summary = pa.summarize_expense_change(totals)
    if summary is None:
        st.info("Not enough periods (or no 06xx / 07xx rows) to compute a change.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Total expenses — {summary['first_label']}", _fmt_money(summary["first_total"]))
    m2.metric(f"Total expenses — {summary['last_label']}", _fmt_money(summary["last_total"]))
    pct = summary["pct"]
    m3.metric(
        "Total change",
        f"{summary['change']:+,.0f}",
        delta=f"{pct:+.1f}%" if pct is not None else None,
        delta_color="inverse",
    )

    if len(totals) < int(n_periods):
        st.caption(f"Only {len(totals)} periods available — showing all of them.")
    if timeline == "Quarterly" and last_month % 3 != 0:
        st.caption(
            f"⚠️ {summary['last_label']} is a partial quarter (data through "
            f"{calendar.month_abbr[last_month]}); earlier quarters are full."
        )
    if timeline == "Yearly" and end_year == today.year:
        st.caption(
            f"⚠️ {end_year} is year-to-date (through {calendar.month_abbr[lc_month]}) "
            "— compared against full prior years, so the change is understated."
        )

    if len(totals) > 2:
        st.dataframe(
            totals.style.format(
                {"Total Expenses": "{:,.0f}", "Change": "{:+,.0f}", "% Change": "{:+.1f}%"},
            ),
            hide_index=True,
            width="stretch",
        )


# ── Section 2: product price table ───────────────────────────────────────────

def _render_product_table(zid: str) -> None:
    st.markdown("### Product price table")

    c1, c2 = st.columns([1, 3])
    with c1:
        window_months = st.number_input(
            "Avg Price window (months)", min_value=1, max_value=12, value=3, step=1,
            key="pa_price_window",
        )
    with c2:
        search = st.text_input("Search code / name", key="pa_product_search")

    items = _load_final_items(str(zid))
    if items.empty:
        st.info("No stock-view data for this entity.")
        return

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(months=int(window_months))
    prices = _load_price_window(str(zid), start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    wh = _load_wh_tiers(str(zid))

    table = pa.build_product_price_table(items, wh, prices)

    if search.strip():
        q = search.strip().lower()
        hit = (
            table["Product Code"].astype(str).str.lower().str.contains(q, regex=False)
            | table["Name"].astype(str).str.lower().str.contains(q, regex=False)
        )
        table = table[hit]

    st.caption(
        f"{len(table):,} products. **Avg Monthly Sale** = the stock view's own trailing-3-month "
        f"average (units). **Avg Price** = realized sales amount ÷ units sold over the last "
        f"{int(window_months)} month(s) (amount = sales − discount, before returns). "
        "**WH Qty / WH Price** = the item's lowest wholesale tier; None = no tier on file; Avg Price None = nothing sold in the window."
    )
    st.dataframe(
        table.style.format(
            {
                "Avg Monthly Sale": "{:,.1f}", "Current Stock": "{:,.0f}",
                "Std Price": "{:,.2f}", "WH Qty": "{:,.0f}",
                "WH Price": "{:,.2f}", "Avg Price": "{:,.2f}",
            },
        ),
        hide_index=True,
        width="stretch",
        height=520,
    )
    st.download_button(
        "⬇ Download CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="price_analysis_products.csv",
        mime="text/csv",
        key="pa_product_csv",
    )


def render_price_analysis(zid) -> None:
    _render_expense_change(str(zid))
    st.divider()
    _render_product_table(str(zid))
