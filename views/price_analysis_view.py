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

# Yearly names mirror Financial Statements' own perspective names.
_YEARLY_FULL = "Yearly - Full Year vs YTD"
_YEARLY_CUSTOM = "Yearly - Custom Range"
_TIMELINES = ["Monthly", "Quarterly", _YEARLY_FULL, _YEARLY_CUSTOM]
_MAX_PERIODS = {"Monthly": 12, "Quarterly": 8, _YEARLY_FULL: 5, _YEARLY_CUSTOM: 5}
_YEARLY_SPAN = 5          # years loaded for either Yearly timeline


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


def _load_expense_frame(
    zid: str, project, timeline: str, end_year: int, end_month: int, start_month: int = 1
):
    """Level 0 income-statement frame (account rows x period columns) for
    the chosen timeline, built with the exact Financial Statements builders
    (process_data_month / collapse_monthly_to_quarterly / process_data), so
    the totals tie to its Level 0 "Expense Changes" table. Returns
    (frame, last_month_included) -- last_month_included is only meaningful
    for Quarterly (flags a partial final quarter). `start_month` only
    matters for Yearly - Custom Range."""
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
    if timeline == _YEARLY_CUSTOM:
        # Same call Financial Statements' "Yearly - Custom Range" makes: every
        # year summed over the SAME start..end month window (YTD vs YTD).
        pl = financial.process_data(
            zid, years, start_month, end_month, "Income Statement", label_df, project, excluded
        )
        return pl, 0
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


def _render_expense_change(zid: str) -> dict | None:
    st.markdown("### Operating expense change")
    st.caption(
        "Level 0 income statement, 06xx SG&A + 07xx S&D accounts — the same rows as "
        "Financial Statements → Level 0 → Expense Changes, added up. "
        "**+** = costs went up · **−** = costs went down."
    )

    today = pd.Timestamp.today().normalize()
    lc_year, lc_month = _last_complete_month(today)

    tl_col, _ = st.columns(2)
    with tl_col:
        timeline = st.selectbox("Timeline", _TIMELINES, key="pa_timeline")
    custom = timeline == _YEARLY_CUSTOM
    yearly = timeline in (_YEARLY_FULL, _YEARLY_CUSTOM)

    cols = st.columns(4 if custom else 3)
    with cols[0]:
        end_year = st.number_input(
            "End Year", min_value=2018, max_value=today.year,
            value=lc_year, step=1, key="pa_end_year",
        )
    start_month = 1
    if custom:
        with cols[1]:
            start_month = st.number_input(
                "Start Month", min_value=1, max_value=12, value=1, step=1,
                key="pa_start_month",
            )
    with cols[-2]:
        end_month = st.number_input(
            "End Month", min_value=1, max_value=12, value=lc_month, step=1,
            key="pa_end_month", disabled=(timeline == _YEARLY_FULL),
        )
    with cols[-1]:
        n_periods = st.number_input(
            "Periods to compare", min_value=2, max_value=_MAX_PERIODS[timeline],
            value=2, step=1, key=f"pa_n_periods_{timeline}",
        )

    end_year, end_month, start_month = int(end_year), int(end_month), int(start_month)
    if not yearly and (end_year, end_month) > (lc_year, lc_month):
        end_year, end_month = lc_year, lc_month
        st.caption(
            f"Latest completed month is {calendar.month_abbr[lc_month]} {lc_year} "
            f"— using that as the end period."
        )
    if yearly and end_year == today.year and today.month == 1:
        st.info(f"No completed month in {today.year} yet — pick {today.year - 1} or earlier.")
        return
    if custom:
        if end_year == today.year and end_month > lc_month:
            end_month = lc_month
            st.caption(
                f"{today.year} only has data through {calendar.month_abbr[lc_month]} — "
                "using that as the end month so every year covers the same months."
            )
        if start_month > end_month:
            st.warning("Start Month is after End Month — pick a start on or before the end.")
            return

    project = st.session_state.get("proj")
    try:
        with st.spinner("Loading Level 0 income statement…"):
            pl, last_month = _load_expense_frame(
                str(zid), project, timeline, end_year, end_month, start_month
            )
    except ValueError as e:
        st.warning(f"Nothing to load for that period: {e}")
        return
    except Exception as e:
        st.warning(f"Could not load the Level 0 income statement: {e}")
        return

    if pl is None or pl.empty:
        st.info("No income-statement data for this entity and period.")
        return

    suffix = (
        f" ({calendar.month_abbr[start_month]}–{calendar.month_abbr[end_month]})" if custom else ""
    )
    totals = pa.build_expense_totals(pl, timeline, int(n_periods), suffix)
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
    if custom:
        st.caption(
            f"Each year covers {calendar.month_abbr[start_month]}–{calendar.month_abbr[end_month]}, "
            "so years are compared like-for-like."
        )
    if timeline == _YEARLY_FULL and end_year == today.year:
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

    return {**summary, "period_months": pa.months_in_period(timeline, start_month, end_month)}


# ── Section 2: price increase needed ─────────────────────────────────────────

def _sales_window(months: int) -> tuple[str, str]:
    """The last `months` COMPLETE calendar months (so today's partial month
    never dilutes a monthly average), as (start, end) ISO dates."""
    today = pd.Timestamp.today().normalize()
    end = today.replace(day=1) - pd.Timedelta(days=1)
    start = end.replace(day=1) - pd.DateOffset(months=int(months) - 1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _render_price_target(zid: str, expense: dict | None) -> int:
    """Sizes the uniform price rise needed to recover the expense change
    above. Returns the sales window (months) so the product table below
    uses the same basis."""
    st.markdown("### Price increase needed")

    r1 = st.columns(3)
    with r1[0]:
        window_months = int(st.number_input(
            "Sales window (months)", min_value=1, max_value=12, value=3, step=1,
            key="pa_price_window",
            help="Last N complete calendar months. Sets monthly revenue/margin here and Avg Price in the table below.",
        ))

    if expense is None:
        st.info("Pick a period with an expense change above to size the price increase.")
        return window_months

    total = float(expense["change"])
    if total <= 0:
        st.success(
            f"Expenses fell by {abs(total):,.0f} ({expense['first_label']} → "
            f"{expense['last_label']}) — nothing to recover."
        )
        return window_months

    months = int(expense["period_months"])
    with r1[1]:
        horizon = int(st.number_input(
            "Recovery horizon (months)", min_value=1, max_value=60, value=months, step=1,
            key=f"pa_horizon_{months}",
            help="Recover the past total over this many months. Defaults to the length of one period.",
        ))
    with r1[2]:
        share = float(st.number_input(
            "Raise the top items covering (% of revenue)", min_value=5, max_value=100,
            value=80, step=5, key="pa_cover_share",
        ))
    r2 = st.columns(3)
    with r2[0]:
        loss_pct = float(st.number_input(
            "Volume loss (%)", min_value=0.0, max_value=50.0, value=4.0, step=0.5,
            key="pa_vol_loss",
        ))
    with r2[1]:
        ref_pct = float(st.number_input(
            "…when prices rise by (%)", min_value=0.5, max_value=50.0, value=3.0, step=0.5,
            key="pa_vol_ref",
        ))
    elasticity = loss_pct / ref_pct

    start, end = _sales_window(window_months)
    base = pa.build_item_base(_load_price_window(str(zid), start, end), window_months)
    if base.empty:
        st.info(f"No sales found between {start} and {end}.")
        return window_months

    subset = pa.select_top_coverage(base, share / 100)
    rev, mar = float(subset["rev_m"].sum()), float(subset["mar_m"].sum())
    gap = total / horizon
    res = pa.required_price_rise(gap, rev, mar, elasticity)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Past total to recover", _fmt_money(total))
    m2.metric(f"Per month (over {horizon} mo)", _fmt_money(gap))
    if res["reachable"]:
        m3.metric("Price rise needed", f"{res['x'] * 100:.2f}%")
        m4.metric("Volume loss assumed", f"{res['volume_loss'] * 100:.2f}%")
    else:
        m3.metric("Price rise needed", "n/a")
        m4.metric("Max per month", _fmt_money(res["max_gain"]))
        st.warning(
            f"A uniform rise on these {len(subset)} items can recover at most "
            f"{res['max_gain']:,.0f} a month — the volume loss outruns the price gain beyond that. "
            "Widen the revenue coverage or lengthen the horizon."
        )

    st.caption(
        f"{len(subset):,} items = {share:.0f}% of revenue · monthly revenue {rev:,.0f}, gross margin "
        f"{mar:,.0f} ({(mar / rev * 100) if rev else 0:.1f}%) over {start} → {end}. "
        f"Volume loss scales with the rise: {loss_pct:.1f}% at {ref_pct:.1f}% means each 1% of price "
        f"loses {elasticity:.2f}% of units. Lost units cost their margin, not their full price. "
        "This is a single uniform rise as a benchmark; per-item eligibility comes with the per-item columns."
    )

    scen = pa.coverage_scenarios(base, gap, elasticity)
    st.dataframe(
        scen.style.format({
            "Monthly Revenue": "{:,.0f}", "Monthly Margin": "{:,.0f}",
            "Price Rise Needed": lambda v: "—" if pd.isna(v) else f"{v:.2f}%",
            "Volume Loss": lambda v: "—" if pd.isna(v) else f"{v:.2f}%",
        }),
        hide_index=True,
        width="stretch",
    )
    return window_months


# ── Section 3: product price table ───────────────────────────────────────────

def _render_product_table(zid: str, window_months: int) -> None:
    st.markdown("### Product price table")

    search = st.text_input("Search code / name", key="pa_product_search")

    items = _load_final_items(str(zid))
    if items.empty:
        st.info("No stock-view data for this entity.")
        return

    start, end = _sales_window(window_months)
    prices = _load_price_window(str(zid), start, end)
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
        f"{int(window_months)} complete month(s), {start} → {end} (amount = sales − discount, before returns). "
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
    expense = _render_expense_change(str(zid))
    st.divider()
    window_months = _render_price_target(str(zid), expense)
    st.divider()
    _render_product_table(str(zid), window_months)
