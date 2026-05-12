"""views/financial_dashboard.py
Analysis Dashboard — rendered as the 📈 Analysis Dashboard panel at Level S.
All data drawn from pl_s / bs_s / cfs_s DataFrames computed upstream.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Colour palette ─────────────────────────────────────────────────────────────
_BLUE   = "#378ADD"
_GREEN  = "#1D9E75"
_RED    = "#E24B4A"
_AMBER  = "#BA7517"
_PURPLE = "#534AB7"
_GREY   = "rgba(150,150,150,0.85)"

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(gridcolor="rgba(0,0,0,0.06)", linecolor="rgba(0,0,0,0.15)"),
    yaxis=dict(gridcolor="rgba(0,0,0,0.06)", linecolor="rgba(0,0,0,0.15)"),
)

OPERATIONAL_HISTORY_NOTES = """
## Group Structure
- **4 entities under common ownership:** HMBR Tools & Chemicals Ltd. (100001, parent importer),
  GI Corporation (100000, manufacturing subsidiary), Gulshan Packaging Co. (100009, internal
  captive packaging), Zepto Chemicals (100005, independent consumer brand).
- All entities share back-office functions: Management, Accounts & Finance, HR, IT, Administration.
- HMBR and GI share the same field sales team. Zepto has its own dedicated sales team.
  Gulshan Packaging has no sales team — all transactions are internal.

## Revenue & Entity History
- **Pre-2024:** GI Corporation's I&H (Industrial & Household) product segment was sold through
  Trading (HMBR). From 2024, GI was given its own sales and accounting entity. The Level T
  adjustment removes this segment from Trading's historical revenue to show the continuing
  business on a like-for-like basis.
- **HMBR (Trading) revenue peak:** ৳41.1 Cr in 2016. Declined to ৳11.8 Cr by 2025 due to
  (a) the GI segment separation and (b) bank debt-driven inventory liquidation at net losses
  in order to service loan obligations (see Debt & Bank section below).
- **GI Corporation growth:** Revenue ৳4.3 Cr (2016, within Trading) → ৳11.7 Cr (2025,
  standalone). Growth driven by strategic investment in new fabricated product lines:
  ladders, paint brushes, steel wire brushes, aluminium and steel-fabricated goods, plus
  organic growth in existing adhesive and filler product lines.
- **Zepto Chemicals:** Started operations ~2018. Revenue ৳4.63 Cr (2025), up 41% YoY.
  Gross margin ~55% — highest in the group. Operates independently with its own sales force
  and brand identity across consumer cleaning and chemical products.
- **Gulshan Packaging:** Captive packaging unit. Purchases goods imported by HMBR, packages
  them, and sells back to HMBR at cost-plus. All AR is intercompany (owed by HMBR).
  All AP is external (raw materials and consumable suppliers). No third-party customers.

## Debt, Bank Loans & Interest Burden
- The group carried significant bank debt across 2016–2024, with total interest charges
  peaking at ৳4.26 Cr in 2019 — representing 136% of that year's EBITDA (৳3.11 Cr).
- Interest charges exceeded EBITDA in multiple years (2017, 2018, 2021, 2022, 2023),
  making net profitability structurally impossible regardless of operational performance.
- The high debt load forced HMBR to liquidate inventory at significant net losses in order
  to generate cash for loan repayments, which is the primary driver of HMBR's revenue
  and margin decline across 2017–2024.
- By 2025, total interest has been reduced to ৳1.18 Cr, giving an interest coverage ratio
  of 2.7× — the first healthy reading since the group was formed.
- Long-term bank loan fully retired by 2024 (from ৳9.13 Cr in 2016 to ৳0).
- Short-term bank loan reduced from ৳13.4 Cr (2023) to ৳6.3 Cr (2025).

## Hospital Investment & Asset Recovery
- Approximately ৳17.6 Cr was invested in a hospital project over multiple years, recorded
  as "Loan to Hospital (Staff Salary)" in Other Assets on the Balance Sheet.
- The hospital did not achieve profitability targets. In 2025, the asset was sold.
- Sale proceeds: ৳9 Cr recovered (against ৳17.6 Cr invested — net loss of ৳8.6 Cr).
- Of the ৳9 Cr proceeds: ৳6 Cr was used directly to retire short-term and long-term
  bank debt; ৳3 Cr went into working capital.
- A further ৳3.5 Cr remains receivable from the hospital sale (outstanding as of 2025).
  Recommended allocation: ৳2 Cr to retire remaining short-term bank debt, ৳1 Cr to
  build cash reserve, ৳0.5 Cr to GI working capital for inventory build.

## Intercompany Balances
- **GI AR / Trading Internal AP (৳670 lakh, 2025):** When GI was carved out as a standalone
  entity in 2024, all assets built and paid for by Trading were transferred to GI's books.
  The corresponding payable sits in Trading's internal AP. This is a capital transfer account
  reflecting historical investment by Trading into GI — not a commercial collection issue.
  Expected to unwind over time as GI retains profits (GI net income 2025: ৳189 lakh).
- **Packaging AR / Trading Internal AP (৳48.6 lakh, 2025):** Live running balance for current
  packaging services rendered by Gulshan Packaging to HMBR. Normal operational balance.
- **Zepto Internal AP to Group (৳35 lakh):** Fixed intercompany balance, unchanged since 2023.
  Effectively a permanent internal funding balance.
- **International AP — frozen (৳176.74 lakh):** This balance has been unchanged across 2023,
  2024, and 2025. Likely an unresolved LC or import payable. Requires active reconciliation
  or write-off decision.

## Working Capital Trends
- Group net working capital (excl. cash) improved from ৳18.97 Cr (2016) to ৳5.09 Cr (2023)
  before rising slightly to ৳6.70 Cr (2025) due to GI inventory build for new product lines.
- Cash conversion cycle improved from 183 days (2016) to 48 days (2023), then widened to
  90 days (2025) due to inventory days rising from 91 to 117.
- Trading DPO of 384 days (2025) reflects accumulated unpaid supplier obligations from the
  debt-repayment period — not a negotiated credit term. Supplier relationships require
  active management as cash position recovers.
- **Zepto CCC:** Best-managed working capital in the group. CCC improved from 303 days (2021)
  to 19 days (2025). DIO halved while revenue grew 41% YoY. DPO extended to 103 days as
  supplier credibility was established.
- Recommended minimum cash buffer for a ৳28 Cr revenue business of this operating model:
  ৳2.0 Cr (floor) to ৳3.5 Cr (comfort). Current cash (2025): ৳0.69 Cr — critically thin.

## Cash Flow
- Operating cash flow turned positive in 2024 for the first time after multiple years of
  negative OCF driven by working capital absorption and debt costs.
- 2025 saw a large investing outflow (৳8.3 Cr) funded by a financing inflow (৳7.4 Cr —
  debt raised) and the hospital sale proceeds. Closing cash: ৳0.69 Cr.
- Free cash flow (OCF + investing) has been negative in most years; 2025 is the
  transitional year where operational improvement and asset monetisation converge.
"""


# ── Data helpers ───────────────────────────────────────────────────────────────

def get_val(df: pd.DataFrame, account_name: str, year, name_col: str = "ac_name") -> float:
    """Safely retrieve a single cell from a Level S DataFrame."""
    if name_col not in df.columns or year not in df.columns:
        return 0.0
    mask = df[name_col].astype(str) == str(account_name)
    if not mask.any():
        return 0.0
    v = df.loc[mask, year].iloc[0]
    return float(v) if pd.notna(v) else 0.0


def fmt_cr(val: float) -> str:
    return f"৳{val / 1e7:.2f} Cr"


def fmt_lakh(val: float) -> str:
    return f"৳{val / 1e5:.2f} L"


def _to_cr(val: float) -> float:
    return val / 1e7


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def _delta_pct(curr: float, prev: float):
    if prev == 0:
        return None
    return (curr / prev - 1) * 100


def _apply_layout(fig: go.Figure, height: int = 340, title: str = "", extra: dict = None) -> go.Figure:
    layout = dict(**CHART_LAYOUT, height=height, title=dict(text=title, font=dict(size=13)))
    if extra:
        layout.update(extra)
    fig.update_layout(**layout)
    return fig


def _wc_for_year(is_df: pd.DataFrame, bs_df: pd.DataFrame, year, months: int = 12) -> dict:
    """Compute working capital metrics for a single year.

    Pass months < 12 for a partial (running) year so WC days are scaled to
    the actual period elapsed rather than a full 365-day year.
    """
    adj_rev  = abs(get_val(is_df, "Adjusted Revenue (Pending)", year))
    cogs     = abs(get_val(is_df, "COGS", year))
    ar       = get_val(bs_df, "Accounts Receivable", year)
    stock    = get_val(bs_df, "0106-Stock in Hand", year)
    advance  = get_val(bs_df, "0105-Advance Accounts", year)
    ap_loc   = abs(get_val(bs_df, "Accounts Payable (Local)", year))
    ap_int   = abs(get_val(bs_df, "Accounts Payable (International)", year))
    ap_intr  = abs(get_val(bs_df, "Accounts Payable (Internal)", year))
    ar_intr  = abs(get_val(bs_df, "Accounts Receivable (Internal)", year))
    ar_loc   = abs(get_val(bs_df, "Accounts Receivable (Local)", year))
    total_ca = get_val(bs_df, "01-Total Current Assets (A)", year)
    total_cl = abs(get_val(bs_df, "Current Liability (A)", year))
    m_agent  = abs(get_val(bs_df, "0904-Money Agent Liability", year))

    days = 365 * months / 12
    dso = _safe_div(ar * days, adj_rev)
    dio = _safe_div(stock * days, cogs)
    dpo = _safe_div((ap_loc + ap_int) * days, cogs)
    ccc = dso + dio - dpo

    adj_ca = total_ca - ar_intr - ar_loc
    adj_cl = total_cl - ap_intr - m_agent
    curr_r = _safe_div(adj_ca, adj_cl)
    quick_r = _safe_div(adj_ca - stock, adj_cl)
    nwc = ar + stock + advance - ap_loc - ap_int

    return {
        "DSO": dso, "DIO": dio, "DPO": dpo, "CCC": ccc,
        "Current Ratio": curr_r, "Quick Ratio": quick_r, "NWC": nwc,
        "AR": ar, "Stock": stock, "Advance": advance,
        "AP": ap_loc + ap_int, "Revenue": adj_rev,
    }


# ── Tab 1 — Group P&L & Margins ────────────────────────────────────────────────

def _tab1_pl(is_df: pd.DataFrame, bs_df: pd.DataFrame, years: list, entity_zid: str,
             partial_year_months: int = 12):
    if not years:
        st.info("No years available for the selected range.")
        return

    max_year = max(years)
    end_y    = years[-1]
    prev_y   = years[-2] if len(years) >= 2 else None
    is_partial = (end_y == max_year and partial_year_months < 12)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    rev       = get_val(is_df, "Adjusted Revenue (Pending)", end_y)
    gp        = get_val(is_df, "Gross Profit", end_y)
    ebitda    = get_val(is_df, "EBITDA", end_y)
    net_inc   = get_val(is_df, "Net Income", end_y)
    gp_m      = _safe_div(gp, abs(rev)) * 100
    ebitda_m  = _safe_div(ebitda, abs(rev)) * 100

    p_rev     = get_val(is_df, "Adjusted Revenue (Pending)", prev_y) if prev_y else 0
    p_gp      = get_val(is_df, "Gross Profit", prev_y) if prev_y else 0
    p_ebitda  = get_val(is_df, "EBITDA", prev_y) if prev_y else 0
    p_net     = get_val(is_df, "Net Income", prev_y) if prev_y else 0
    p_gp_m    = _safe_div(p_gp, abs(p_rev)) * 100 if p_rev else 0
    p_ebitda_m = _safe_div(p_ebitda, abs(p_rev)) * 100 if p_rev else 0

    # For a partial running year, scale prior-year absolute values to the same
    # number of months so the delta is like-for-like (e.g. 3M 2026 vs 3M est. 2025).
    # Margin ratios (%) are period-neutral and need no scaling.
    if is_partial:
        scale = partial_year_months / 12
        p_rev_cmp    = p_rev    * scale
        p_gp_cmp     = p_gp     * scale
        p_ebitda_cmp = p_ebitda * scale
        p_net_cmp    = p_net    * scale
        _mo = partial_year_months
        _cmp_label = f"vs {_mo}M est. {prev_y}"
    else:
        p_rev_cmp, p_gp_cmp, p_ebitda_cmp, p_net_cmp = p_rev, p_gp, p_ebitda, p_net
        _cmp_label = f"vs {prev_y}"

    def _d_pct(c, p):
        v = _delta_pct(c, p)
        return f"{v:+.1f}% {_cmp_label}" if v is not None else ""

    def _d_pp(c, p):
        return f"{c - p:+.1f} pp vs {prev_y}" if prev_y else ""

    if is_partial:
        st.caption(
            f"⚡ Running year ({end_y}, {partial_year_months} months) — "
            f"absolute KPI deltas compare against estimated {partial_year_months}-month "
            f"equivalent from {prev_y}. Margin % deltas compare full-year {prev_y}."
        )

    k = st.columns(6)
    k[0].metric("Revenue",        fmt_cr(rev),          _d_pct(rev, p_rev_cmp))
    k[1].metric("Gross Profit",   fmt_cr(gp),           _d_pct(gp, p_gp_cmp))
    k[2].metric("Gross Margin",   f"{gp_m:.1f}%",       _d_pp(gp_m, p_gp_m))
    k[3].metric("EBITDA",         fmt_cr(ebitda),       _d_pct(ebitda, p_ebitda_cmp))
    k[4].metric("EBITDA Margin",  f"{ebitda_m:.1f}%",   _d_pp(ebitda_m, p_ebitda_m))
    k[5].metric("Net Income",     fmt_cr(net_inc),      _d_pct(net_inc, p_net_cmp))

    st.markdown(" ")

    # ── Chart A — Revenue Trend ────────────────────────────────────────────────
    try:
        x_yrs  = [str(y) for y in years]
        rev_v  = [_to_cr(get_val(is_df, "Revenue", y)) for y in years]
        adj_v  = [_to_cr(get_val(is_df, "Adjusted Revenue (Pending)", y)) for y in years]
        gp_v   = [_to_cr(get_val(is_df, "Gross Profit", y)) for y in years]
        ebd_v  = [_to_cr(get_val(is_df, "EBITDA", y)) for y in years]

        fig_a = go.Figure()
        # Only show Revenue bar if it differs from Adjusted Revenue
        if any(abs(r - a) > 0.01 for r, a in zip(rev_v, adj_v)):
            fig_a.add_trace(go.Bar(
                x=x_yrs, y=rev_v, name="Revenue",
                marker_color=_BLUE, opacity=0.5,
            ))
        fig_a.add_trace(go.Bar(
            x=x_yrs, y=adj_v, name="Adjusted Revenue",
            marker_color=_BLUE, opacity=1.0,
        ))
        fig_a.add_trace(go.Scatter(
            x=x_yrs, y=gp_v, name="Gross Profit",
            mode="lines+markers", line=dict(color=_GREEN, width=2),
        ))
        fig_a.add_trace(go.Scatter(
            x=x_yrs, y=ebd_v, name="EBITDA",
            mode="lines+markers", line=dict(color=_AMBER, width=2, dash="dash"),
        ))
        _apply_layout(fig_a, 340, "Revenue, Gross Profit & EBITDA (৳ Cr)",
                      extra=dict(barmode="group",
                                 yaxis=dict(title="৳ Crore", gridcolor="rgba(0,0,0,0.06)")))
        st.plotly_chart(fig_a, use_container_width=True)
        with st.expander("ℹ️ How to read this chart", expanded=False):
            st.caption(
                "**Revenue, Gross Profit & EBITDA** — Bars show annual top-line revenue "
                "(adjusted for pending sales). Lines track Gross Profit (revenue minus direct cost of goods) "
                "and EBITDA (operating profit before interest, tax, depreciation & amortisation). "
                "When GP and EBITDA lines track revenue upwards together, margins are holding. "
                "A widening gap between revenue and GP indicates rising input costs; a gap between "
                "GP and EBITDA reflects growing overhead (SG&A / S&D expenses)."
            )
    except Exception as e:
        st.warning(f"Chart A error: {e}")

    # ── Chart B — Margin Evolution ─────────────────────────────────────────────
    try:
        gp_pct  = [_safe_div(get_val(is_df, "Gross Profit", y),
                             abs(get_val(is_df, "Adjusted Revenue (Pending)", y))) * 100 for y in years]
        ebd_pct = [_safe_div(get_val(is_df, "EBITDA", y),
                             abs(get_val(is_df, "Adjusted Revenue (Pending)", y))) * 100 for y in years]
        ni_pct  = [_safe_div(get_val(is_df, "Net Income", y),
                             abs(get_val(is_df, "Adjusted Revenue (Pending)", y))) * 100 for y in years]

        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(
            x=x_yrs, y=gp_pct, name="Gross Margin %", mode="lines+markers",
            line=dict(color=_GREEN, width=2),
            fill="tozeroy", fillcolor=f"rgba(29,158,117,0.10)",
        ))
        fig_b.add_trace(go.Scatter(
            x=x_yrs, y=ebd_pct, name="EBITDA Margin %", mode="lines+markers",
            line=dict(color=_AMBER, width=2, dash="dash"),
        ))
        ni_colors = [_GREEN if v >= 0 else _RED for v in ni_pct]
        fig_b.add_trace(go.Scatter(
            x=x_yrs, y=ni_pct, name="Net Income Margin %", mode="lines+markers",
            line=dict(color=_RED, width=1.5),
            marker=dict(color=ni_colors),
        ))
        _apply_layout(fig_b, 220, "Margin Evolution (%)",
                      extra=dict(yaxis=dict(ticksuffix="%", gridcolor="rgba(0,0,0,0.06)")))
        st.plotly_chart(fig_b, use_container_width=True)
        with st.expander("ℹ️ How to read this chart", expanded=False):
            st.caption(
                "**Margin Evolution** — Shows three profitability rates as a percentage of revenue over time. "
                "Gross Margin % (green) is revenue minus COGS — this measures pricing power and procurement efficiency. "
                "EBITDA Margin % (amber) deducts overhead costs; the gap between it and gross margin reveals "
                "how much overhead consumes each taka earned. Net Income Margin % (red/green dots) is the "
                "bottom-line return after interest, tax and other charges — points above zero are profitable years. "
                "Converging lines indicate that overheads or interest are consuming an increasing share of gross profit."
            )
    except Exception as e:
        st.warning(f"Chart B error: {e}")

    # ── Chart C — Cost Structure ───────────────────────────────────────────────
    try:
        def _seg(row, y): return abs(_to_cr(get_val(is_df, row, y)))

        fig_c = go.Figure()
        segs = [
            ("COGS",              "COGS",                                   _RED,    1.0),
            ("Total SG&A",        "Total SG&A",                             _PURPLE, 1.0),
            ("Total S&D",         "Total Sales & Distribution",             _BLUE,   0.65),
            ("Interest",          "Total Interest & Charges",               _AMBER,  1.0),
            ("VAT & Tax",         "0629-VAT & Tax Total (A+B+C)",           _GREY,   1.0),
        ]
        for label, row, colour, opacity in segs:
            vals = [_seg(row, y) for y in years]
            fig_c.add_trace(go.Bar(
                y=x_yrs, x=vals, name=label, orientation="h",
                marker_color=colour, opacity=opacity,
            ))
        # Net Income segment — green if positive, red if negative
        ni_vals = [_to_cr(get_val(is_df, "Net Income", y)) for y in years]
        ni_col  = [_GREEN if v >= 0 else _RED for v in ni_vals]
        fig_c.add_trace(go.Bar(
            y=x_yrs, x=ni_vals, name="Net Income", orientation="h",
            marker_color=ni_col,
        ))
        _apply_layout(fig_c, 220, "Revenue Decomposition by Cost Layer",
                      extra=dict(barmode="stack",
                                 xaxis=dict(title="৳ Crore", gridcolor="rgba(0,0,0,0.06)")))
        st.plotly_chart(fig_c, use_container_width=True)
        with st.expander("ℹ️ How to read this chart", expanded=False):
            st.caption(
                "**Revenue Decomposition** — Each horizontal bar represents one year's revenue split into "
                "its cost layers from left to right: COGS (direct product cost), SG&A (staff and admin overhead), "
                "Sales & Distribution (field sales, discounts, logistics), Interest & Charges, VAT & Tax, "
                "and the remaining Net Income slice. The total bar width equals total revenue. "
                "A shrinking COGS segment means improving gross margins; a shrinking Interest segment "
                "reflects debt reduction. A green Net Income segment means the year was profitable; red means a loss. "
                "Use this to quickly see which cost layer is consuming the most revenue in any given year."
            )
    except Exception as e:
        st.warning(f"Chart C error: {e}")

    # ── Year Drill ─────────────────────────────────────────────────────────────
    drill_y = st.selectbox(
        "Drill into a specific year",
        options=years, index=len(years) - 1,
        key="dash_year_drill",
    )
    try:
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"**Income Statement — {drill_y}** *(৳ Lakh)*")
            # Build single-year IS table
            nc = "ac_name"
            is_year = is_df[[nc, drill_y]].copy() if drill_y in is_df.columns else pd.DataFrame()
            if not is_year.empty:
                is_year = is_year.rename(columns={nc: "Account", drill_y: "৳ Lakh"})
                is_year["৳ Lakh"] = is_year["৳ Lakh"].apply(
                    lambda v: round(float(v) / 1e5, 2) if pd.notna(v) else None
                )
                highlight_rows = {"Gross Profit", "EBITDA", "Net Income"}

                def _hl_is(row):
                    if row["Account"] in highlight_rows:
                        return ["background-color: #D4EDDA; font-weight: bold"] * 2
                    return [""] * 2

                try:
                    st.dataframe(
                        is_year.style.apply(_hl_is, axis=1).format({"৳ Lakh": "{:.2f}"}),
                        use_container_width=True, hide_index=True,
                    )
                except Exception:
                    st.dataframe(is_year, use_container_width=True, hide_index=True)

        with d2:
            st.markdown(f"**Cost Composition — {drill_y}**")
            segs_d = [
                ("COGS", _RED),
                ("Total SG&A", _PURPLE),
                ("Total Sales & Distribution", _BLUE),
                ("Total Interest & Charges", _AMBER),
                ("0629-VAT & Tax Total (A+B+C)", _GREY),
            ]
            lbls  = [s[0].replace("Total ", "").replace("0629-", "") for s in segs_d]
            vals  = [abs(_to_cr(get_val(is_df, s[0], drill_y))) for s in segs_d]
            clrs  = [s[1] for s in segs_d]
            ni_d  = _to_cr(get_val(is_df, "Net Income", drill_y))
            lbls.append("Net Income")
            vals.append(abs(ni_d))
            clrs.append(_GREEN if ni_d >= 0 else _RED)
            fig_d = go.Figure(go.Pie(
                labels=lbls, values=vals, hole=0.45,
                marker=dict(colors=clrs),
            ))
            _apply_layout(fig_d, 340, f"Cost Composition {drill_y}")
            st.plotly_chart(fig_d, use_container_width=True)
            with st.expander("ℹ️ How to read this chart", expanded=False):
                st.caption(
                    f"**Cost Composition — {drill_y}** — Donut chart showing what share of revenue "
                    "each cost category consumed in the selected year. Each slice is proportional to "
                    "its value in ৳ Crore. The Net Income slice (green = profit, red = loss) is the "
                    "residual after all costs. A large COGS slice is expected in a trading business; "
                    "a large Interest slice indicates significant debt burden relative to revenue. "
                    "Hover over a slice to see the exact value."
                )
    except Exception as e:
        st.warning(f"Year drill error: {e}")


# ── Tab 2 — Working Capital & Cash Cycle ───────────────────────────────────────

def _tab2_wc(is_df: pd.DataFrame, bs_df: pd.DataFrame, cfs_df: pd.DataFrame,
             years: list, entity_zid: str, partial_year_months: int = 12):
    if not years:
        st.info("No years available for the selected range.")
        return

    if entity_zid == "100000":
        st.info(
            "GI Corporation was established as a standalone accounting entity in 2024. "
            "Pre-2024 balance sheet data is zero for this entity — working capital metrics "
            "for earlier years will show as zero."
        )

    max_year = max(years)
    end_y  = years[-1]
    prev_y = years[-2] if len(years) >= 2 else None

    def _months_for(y):
        return partial_year_months if y == max_year else 12

    wc_end  = _wc_for_year(is_df, bs_df, end_y,  _months_for(end_y))
    wc_prev = _wc_for_year(is_df, bs_df, prev_y, _months_for(prev_y)) if prev_y else {}

    def _day_delta(key):
        if not wc_prev:
            return ""
        d = wc_end[key] - wc_prev.get(key, 0)
        return f"{d:+.0f} days vs {prev_y}"

    # ── KPI cards ─────────────────────────────────────────────────────────────
    k = st.columns(4)
    k[0].metric("DSO",  f"{wc_end['DSO']:.0f} days",  _day_delta("DSO"))
    k[1].metric("DIO",  f"{wc_end['DIO']:.0f} days",  _day_delta("DIO"))
    k[2].metric("DPO",  f"{wc_end['DPO']:.0f} days",  _day_delta("DPO"))
    k[3].metric("CCC",  f"{wc_end['CCC']:.0f} days",  _day_delta("CCC"))

    st.markdown(" ")
    x_yrs = [str(y) for y in years]

    # ── Chart D — Cash Conversion Cycle ───────────────────────────────────────
    try:
        wc_all = {y: _wc_for_year(is_df, bs_df, y, _months_for(y)) for y in years}
        dio_v = [wc_all[y]["DIO"] for y in years]
        dso_v = [wc_all[y]["DSO"] for y in years]
        dpo_v = [wc_all[y]["DPO"] for y in years]
        ccc_v = [wc_all[y]["CCC"] for y in years]

        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(
            x=x_yrs, y=dio_v, name="DIO (days)", marker_color=_PURPLE, opacity=0.8,
        ))
        fig_d.add_trace(go.Bar(
            x=x_yrs, y=dso_v, name="DSO (days)", marker_color=_BLUE, opacity=0.8,
        ))
        fig_d.add_trace(go.Bar(
            x=x_yrs, y=[-v for v in dpo_v], name="DPO (days)",
            marker_color=_RED, opacity=0.6,
        ))
        fig_d.add_trace(go.Scatter(
            x=x_yrs, y=ccc_v, name="Net CCC", mode="lines+markers",
            line=dict(color=_AMBER, width=2),
        ))
        fig_d.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.25)")
        _apply_layout(fig_d, 340, "Cash Conversion Cycle — DSO + DIO − DPO (days)",
                      extra=dict(barmode="relative",
                                 yaxis=dict(title="Days", gridcolor="rgba(0,0,0,0.06)")))
        st.plotly_chart(fig_d, use_container_width=True)
        with st.expander("ℹ️ How to read this chart", expanded=False):
            st.caption(
                "**Cash Conversion Cycle (CCC)** — Measures how many days it takes to convert "
                "inventory purchases into cash collected from customers. "
                "DIO (purple, above zero) = days inventory is held before sale. "
                "DSO (blue, above zero) = days taken to collect payment after a sale. "
                "DPO (red, below zero) = days the business takes to pay its own suppliers — "
                "this offsets the cycle because supplier credit delays your cash outflow. "
                "The amber line is the net CCC = DIO + DSO − DPO. "
                "A shorter CCC means cash recycles faster; a negative CCC (rare) means suppliers "
                "are effectively financing your operations. Trend downward is a positive signal."
            )
    except Exception as e:
        st.warning(f"Chart D error: {e}")

    # ── Chart E — Working Capital Composition ─────────────────────────────────
    try:
        ar_v  = [_to_cr(get_val(bs_df, "Accounts Receivable", y)) for y in years]
        stk_v = [_to_cr(get_val(bs_df, "0106-Stock in Hand", y)) for y in years]
        adv_v = [_to_cr(get_val(bs_df, "0105-Advance Accounts", y)) for y in years]
        ap_v  = [-_to_cr(abs(get_val(bs_df, "Accounts Payable (Local)", y)) +
                         abs(get_val(bs_df, "Accounts Payable (International)", y))) for y in years]
        rev_v = [abs(get_val(is_df, "Adjusted Revenue (Pending)", y)) for y in years]
        nwc_v = [_to_cr(wc_all[y]["NWC"]) for y in years]
        nwc_pct = [_safe_div(nwc_v[i] * 1e7, rev_v[i]) * 100 for i in range(len(years))]

        fig_e = go.Figure()
        fig_e.add_trace(go.Bar(x=x_yrs, y=stk_v, name="Stock in Hand",
                               marker_color=_PURPLE))
        fig_e.add_trace(go.Bar(x=x_yrs, y=ar_v,  name="AR (External)",
                               marker_color=_BLUE))
        fig_e.add_trace(go.Bar(x=x_yrs, y=adv_v, name="Advance Accounts",
                               marker_color=_AMBER))
        fig_e.add_trace(go.Bar(x=x_yrs, y=ap_v,  name="AP (Local + Intl)",
                               marker_color=_RED, opacity=0.6))
        fig_e.add_trace(go.Scatter(
            x=x_yrs, y=nwc_pct, name="NWC % of Revenue",
            mode="lines+markers", line=dict(color=_GREEN, width=2, dash="dash"),
            yaxis="y2",
        ))
        _apply_layout(fig_e, 220, "Net Working Capital Composition (৳ Cr)", extra=dict(
            barmode="relative",
            yaxis=dict(title="৳ Crore", gridcolor="rgba(0,0,0,0.06)"),
            yaxis2=dict(title="NWC % Revenue", overlaying="y", side="right",
                        showgrid=False, ticksuffix="%"),
        ))
        st.plotly_chart(fig_e, use_container_width=True)
        with st.expander("ℹ️ How to read this chart", expanded=False):
            st.caption(
                "**Net Working Capital Composition** — Stacked bars show the balance sheet components "
                "that make up working capital. Positive bars (Stock, AR, Advances) are assets that "
                "tie up cash; the negative bar (AP) is a liability that frees cash — it represents "
                "financing provided by suppliers. The net height of the stacked bar is Net Working Capital (NWC). "
                "The dashed line on the right axis shows NWC as a % of revenue: a rising trend means "
                "the business needs more capital to support each taka of sales, which can strain cash flow. "
                "A declining trend indicates improving capital efficiency."
            )
    except Exception as e:
        st.warning(f"Chart E error: {e}")

    # ── Chart F — Cash Flow Summary ────────────────────────────────────────────
    try:
        def _cfs(row, y): return _to_cr(get_val(cfs_df, row, y))
        cfo_v = [_cfs("Cash from Operations", y) for y in years]
        cfi_v = [_cfs("Cash from Investing",  y) for y in years]
        cff_v = [_cfs("Cash from Financing",  y) for y in years]
        cls_v = [_cfs("Closing Cash & CE",    y) for y in years]

        fig_f = go.Figure()
        fig_f.add_trace(go.Bar(
            x=x_yrs, y=cfo_v, name="Cash from Operations",
            marker_color=[_GREEN if v >= 0 else _RED for v in cfo_v],
        ))
        fig_f.add_trace(go.Bar(
            x=x_yrs, y=cfi_v, name="Cash from Investing", marker_color=_BLUE,
        ))
        fig_f.add_trace(go.Bar(
            x=x_yrs, y=cff_v, name="Cash from Financing", marker_color=_AMBER,
        ))
        fig_f.add_trace(go.Scatter(
            x=x_yrs, y=cls_v, name="Closing Cash & CE", mode="lines+markers",
            line=dict(color=_GREY, width=2), yaxis="y2",
        ))
        _apply_layout(fig_f, 220, "Cash Flow Summary (৳ Cr)", extra=dict(
            barmode="group",
            yaxis=dict(title="৳ Crore", gridcolor="rgba(0,0,0,0.06)"),
            yaxis2=dict(title="Closing Cash (৳ Cr)", overlaying="y", side="right",
                        showgrid=False),
        ))
        st.plotly_chart(fig_f, use_container_width=True)
        with st.expander("ℹ️ How to read this chart", expanded=False):
            st.caption(
                "**Cash Flow Summary** — Three grouped bars per year show where cash came from and went. "
                "Cash from Operations (green = positive, red = negative) is the core cash the business "
                "generated from trading — persistent negative CFO means the business cannot self-fund. "
                "Cash from Investing (blue) reflects capital expenditure and asset sales — typically negative "
                "in growth years. Cash from Financing (amber) includes loan drawdowns (positive) and "
                "repayments (negative). The grey line on the right axis tracks Closing Cash & equivalents. "
                "A healthy pattern: positive CFO funds investing activities with modest financing support. "
                "Reliance on financing to cover negative CFO signals operational cash stress."
            )
    except Exception as e:
        st.warning(f"Chart F error: {e}")

    # ── Year Drill — Working Capital ───────────────────────────────────────────
    wc_year = st.selectbox(
        "Working capital detail for year",
        options=years, index=len(years) - 1,
        key="dash_wc_year",
    )
    try:
        wc_d = _wc_for_year(is_df, bs_df, wc_year, _months_for(wc_year))
        m1, m2, m3 = st.columns(3)
        m1.metric("Net Working Capital",
                  fmt_cr(wc_d["NWC"]),
                  f"AR + Stock + Adv − AP")
        m2.metric("Current Ratio",    f"{wc_d['Current Ratio']:.2f}×")
        m3.metric("Quick Ratio",      f"{wc_d['Quick Ratio']:.2f}×")

        components = [
            ("AR (External)",       wc_d["AR"],       _BLUE),
            ("Stock in Hand",       wc_d["Stock"],    _PURPLE),
            ("Advance Accounts",    wc_d["Advance"],  _AMBER),
            ("AP (Local + Intl)",  -wc_d["AP"],       _RED),
        ]
        lbls_w = [c[0] for c in components]
        vals_w = [c[1] / 1e5 for c in components]
        clrs_w = [c[2] for c in components]
        fig_wc = go.Figure(go.Bar(
            y=lbls_w, x=vals_w, orientation="h",
            marker_color=clrs_w,
        ))
        _apply_layout(fig_wc, 220, f"Working Capital Breakdown — {wc_year} (৳ Lakh)",
                      extra=dict(xaxis=dict(title="৳ Lakh", gridcolor="rgba(0,0,0,0.06)")))
        st.plotly_chart(fig_wc, use_container_width=True)
        with st.expander("ℹ️ How to read this chart", expanded=False):
            st.caption(
                f"**Working Capital Breakdown — {wc_year}** — Horizontal bars show the four components "
                "of net working capital in ৳ Lakh. AR (blue), Stock (purple) and Advance Accounts (amber) "
                "are positive — they represent cash tied up in the business cycle. AP (red) is shown as a "
                "negative bar because it is supplier-funded capital that offsets your cash requirement. "
                "Net Working Capital = AR + Stock + Advances − AP. A large Stock bar relative to AR "
                "suggests inventory is the main cash trap; a large AP bar means suppliers are absorbing "
                "a significant portion of working capital needs."
            )
    except Exception as e:
        st.warning(f"WC drill error: {e}")


# ── Tab 3 — Balance Sheet & Leverage ───────────────────────────────────────────

def _tab3_bs(is_df: pd.DataFrame, bs_df: pd.DataFrame, cfs_df: pd.DataFrame,
             years: list, entity_zid: str):
    if not years:
        st.info("No years available for the selected range.")
        return

    end_y  = years[-1]
    prev_y = years[-2] if len(years) >= 2 else None
    x_yrs  = [str(y) for y in years]

    # ── KPI cards ─────────────────────────────────────────────────────────────
    total_assets = get_val(bs_df, "Total Assets (A+B+C)", end_y)
    total_eq     = abs(get_val(bs_df, "Total Equity", end_y))
    stb_loan     = abs(get_val(bs_df, "1001-Short Term Bank Loan (B)", end_y))
    st_loan      = abs(get_val(bs_df, "1001-Short Term Loan (Related Parties)", end_y))
    st_debt      = stb_loan + st_loan

    ebitda   = get_val(is_df, "EBITDA", end_y)
    interest = abs(get_val(is_df, "Total Interest & Charges", end_y))
    int_cov  = _safe_div(ebitda, interest)

    total_stl = abs(get_val(bs_df, "Total Short Term Liability (C)", end_y))
    lt_loan   = abs(get_val(bs_df, "1201-Long Term Bank Loan (E)", end_y))
    de2       = _safe_div(total_stl + lt_loan, total_eq)

    k = st.columns(5)
    k[0].metric("Total Assets",      fmt_cr(total_assets))
    k[1].metric("Total Equity",      fmt_cr(total_eq))
    k[2].metric("Short-term Debt",   fmt_cr(st_debt))
    k[3].metric("D/E Ratio",         f"{de2:.2f}×")
    k[4].metric("Interest Coverage", f"{int_cov:.2f}×")

    st.markdown(" ")

    # ── Chart G — Asset & Liability Structure ─────────────────────────────────
    try:
        ca_v  = [_to_cr(get_val(bs_df, "01-Total Current Assets (A)", y)) for y in years]
        oa_v  = [_to_cr(abs(get_val(bs_df, "Total Other Assets (B)", y))) for y in years]
        fa_v  = [_to_cr(abs(get_val(bs_df, "Total Fixed Assets (C)", y))) for y in years]
        cl_v  = [-_to_cr(abs(get_val(bs_df, "Current Liability (A)", y))) for y in years]
        stl_v = [-_to_cr(abs(get_val(bs_df, "Total Short Term Liability (C)", y))) for y in years]
        lt_v  = [-_to_cr(abs(get_val(bs_df, "1201-Long Term Bank Loan (E)", y))) for y in years]
        eq_v  = [-_to_cr(abs(get_val(bs_df, "Total Equity", y))) for y in years]

        fig_g = go.Figure()
        fig_g.add_trace(go.Bar(x=x_yrs, y=ca_v, name="Current Assets",
                               marker_color=_BLUE))
        fig_g.add_trace(go.Bar(x=x_yrs, y=oa_v, name="Other Assets",
                               marker_color=_BLUE, opacity=0.5))
        fig_g.add_trace(go.Bar(x=x_yrs, y=fa_v, name="Fixed Assets",
                               marker_color=_PURPLE))
        fig_g.add_trace(go.Bar(x=x_yrs, y=cl_v, name="Current Liability",
                               marker_color=_RED))
        fig_g.add_trace(go.Bar(x=x_yrs, y=stl_v, name="ST Liability",
                               marker_color=_RED, opacity=0.7))
        fig_g.add_trace(go.Bar(x=x_yrs, y=lt_v, name="LT Bank Loan",
                               marker_color=_RED, opacity=0.5))
        fig_g.add_trace(go.Bar(x=x_yrs, y=eq_v, name="Total Equity",
                               marker_color=_GREEN))
        fig_g.add_annotation(
            text="Assets shown positive · Liabilities & Equity shown negative for visual balance",
            xref="paper", yref="paper", x=0.5, y=-0.12,
            showarrow=False, font=dict(size=10, color="rgba(0,0,0,0.45)"),
        )
        _apply_layout(fig_g, 340, "Balance Sheet Structure — Assets vs Funding (৳ Cr)",
                      extra=dict(barmode="relative",
                                 yaxis=dict(title="৳ Crore", gridcolor="rgba(0,0,0,0.06)")))
        st.plotly_chart(fig_g, use_container_width=True)
        with st.expander("ℹ️ How to read this chart", expanded=False):
            st.caption(
                "**Balance Sheet Structure** — A mirrored chart where assets (positive, above zero) "
                "are matched against their funding sources (negative, below zero). "
                "Assets are split into Current (blue), Other non-current (blue, faded), and Fixed (purple). "
                "Funding is split into Current Liabilities (red), Short-term Loans (red, faded), "
                "Long-term Bank Loan (red, light), and Equity (green). "
                "The chart should balance — the total positive bar height equals the total negative bar height. "
                "A shrinking red funding block and growing green equity block indicate deleveraging and "
                "accumulated retained earnings. A growing fixed asset bar without matching equity growth "
                "signals debt-funded capital investment."
            )
    except Exception as e:
        st.warning(f"Chart G error: {e}")

    # ── Chart H — Debt Profile & Interest Burden ──────────────────────────────
    try:
        stb_v  = [_to_cr(abs(get_val(bs_df, "1001-Short Term Bank Loan (B)", y))) for y in years]
        ltb_v  = [_to_cr(abs(get_val(bs_df, "1201-Long Term Bank Loan (E)", y))) for y in years]
        str_v  = [_to_cr(abs(get_val(bs_df, "1001-Short Term Loan (Related Parties)", y))) for y in years]
        int_v  = [_to_cr(abs(get_val(is_df, "Total Interest & Charges", y))) for y in years]

        fig_h = go.Figure()
        fig_h.add_trace(go.Bar(x=x_yrs, y=stb_v, name="ST Bank Loan",
                               marker_color=_RED))
        fig_h.add_trace(go.Bar(x=x_yrs, y=ltb_v, name="LT Bank Loan",
                               marker_color=_RED, opacity=0.5))
        fig_h.add_trace(go.Bar(x=x_yrs, y=str_v, name="ST Related Party",
                               marker_color=_AMBER))
        fig_h.add_trace(go.Scatter(
            x=x_yrs, y=int_v, name="Total Interest & Charges",
            mode="lines+markers", line=dict(color="rgba(30,30,30,0.85)", width=2),
            yaxis="y2",
        ))
        _apply_layout(fig_h, 220, "Debt Profile & Interest Burden (৳ Cr)", extra=dict(
            barmode="group",
            yaxis=dict(title="৳ Crore (Debt)", gridcolor="rgba(0,0,0,0.06)"),
            yaxis2=dict(title="৳ Crore (Interest)", overlaying="y", side="right",
                        showgrid=False),
        ))
        st.plotly_chart(fig_h, use_container_width=True)
        with st.expander("ℹ️ How to read this chart", expanded=False):
            st.caption(
                "**Debt Profile & Interest Burden** — Grouped bars show outstanding debt broken into "
                "Short-term Bank Loan (dark red), Long-term Bank Loan (light red), and Related-party "
                "Short-term Loans (amber) for each year. The black line on the right axis shows the "
                "total interest and finance charge incurred that year — it is a flow (cost), not a stock. "
                "Falling bars indicate debt is being repaid. A falling interest line that lags behind "
                "falling debt is typical when fixed-rate loans are repaid mid-year. "
                "Compare interest charges against EBITDA (from Tab 1) to assess debt serviceability — "
                "interest coverage below 1.5× is generally a concern."
            )
    except Exception as e:
        st.warning(f"Chart H error: {e}")

    # ── Chart I — Equity & Net Income ─────────────────────────────────────────
    try:
        eq_walk = [_to_cr(abs(get_val(bs_df, "Total Equity", y))) for y in years]
        ni_walk = [_to_cr(get_val(is_df, "Net Income", y)) for y in years]
        eq_grow = [eq_walk[i] >= (eq_walk[i - 1] if i > 0 else eq_walk[i]) for i in range(len(years))]

        fig_i = go.Figure()
        fig_i.add_trace(go.Bar(
            x=x_yrs, y=eq_walk, name="Total Equity",
            marker_color=[_GREEN if g else _AMBER for g in eq_grow],
        ))
        fig_i.add_trace(go.Scatter(
            x=x_yrs, y=ni_walk, name="Net Income", mode="lines+markers",
            line=dict(color=_GREEN, width=2),
            marker=dict(color=[_GREEN if v >= 0 else _RED for v in ni_walk]),
            yaxis="y2",
        ))
        _apply_layout(fig_i, 220, "Equity & Net Income (৳ Cr)", extra=dict(
            yaxis=dict(title="৳ Crore (Equity)", gridcolor="rgba(0,0,0,0.06)"),
            yaxis2=dict(title="৳ Crore (Net Income)", overlaying="y", side="right",
                        showgrid=False),
        ))
        st.plotly_chart(fig_i, use_container_width=True)
        with st.expander("ℹ️ How to read this chart", expanded=False):
            st.caption(
                "**Equity & Net Income** — Bars show total equity at year-end (green = grew vs prior year, "
                "amber = shrank). The line on the right axis shows net income for the year. "
                "Equity grows when the business retains profits and shrinks when it posts losses or "
                "distributes more than it earns. A year with positive net income but falling equity "
                "indicates that owner drawings or prior-year loss write-offs exceeded current earnings. "
                "Consistently rising equity alongside positive net income is the signature of a "
                "compounding, self-financing business."
            )
    except Exception as e:
        st.warning(f"Chart I error: {e}")

    # ── Year Drill — Balance Sheet ─────────────────────────────────────────────
    bs_year = st.selectbox(
        "Balance sheet detail for year",
        options=years, index=len(years) - 1,
        key="dash_bs_year",
    )
    try:
        nc = "ac_name"
        if bs_year in bs_df.columns:
            bs_yr = bs_df[[nc, bs_year]].copy().rename(
                columns={nc: "Account", bs_year: "৳ Lakh"}
            )
            bs_yr["৳ Lakh"] = bs_yr["৳ Lakh"].apply(
                lambda v: round(float(v) / 1e5, 2) if pd.notna(v) else None
            )
            # Split at Total Assets row
            split_idx = bs_yr[bs_yr["Account"] == "Total Assets (A+B+C)"].index
            if len(split_idx):
                cut = split_idx[0] + 1
                asset_df = bs_yr.iloc[:cut].reset_index(drop=True)
                liab_df  = bs_yr.iloc[cut:].reset_index(drop=True)
            else:
                asset_df = bs_yr
                liab_df  = pd.DataFrame(columns=bs_yr.columns)

            subtotal_rows = {
                "01-Total Current Assets (A)", "Total Other Assets (B)",
                "Total Fixed Assets (C)", "Total Assets (A+B+C)",
                "Current Liability (A)", "Total Short Term Liability (C)",
                "Total Reserve & Funds (D)", "Total Liabilities (A+B+C+D+E)",
                "Total Equity", "Total Liabilities & Equity", "Balance Check",
            }

            def _hl_bs(row):
                if row["Account"] in subtotal_rows:
                    return ["font-weight: bold"] * 2
                return [""] * 2

            b1, b2 = st.columns(2)
            with b1:
                st.markdown(f"**Assets — {bs_year}** *(৳ Lakh)*")
                try:
                    st.dataframe(
                        asset_df.style.apply(_hl_bs, axis=1).format({"৳ Lakh": "{:.2f}"}),
                        use_container_width=True, hide_index=True,
                    )
                except Exception:
                    st.dataframe(asset_df, use_container_width=True, hide_index=True)
            with b2:
                st.markdown(f"**Liabilities & Equity — {bs_year}** *(৳ Lakh)*")
                try:
                    st.dataframe(
                        liab_df.style.apply(_hl_bs, axis=1).format({"৳ Lakh": "{:.2f}"}),
                        use_container_width=True, hide_index=True,
                    )
                except Exception:
                    st.dataframe(liab_df, use_container_width=True, hide_index=True)

            # Asset mix donut
            ca_d  = abs(get_val(bs_df, "01-Total Current Assets (A)", bs_year))
            oa_d  = abs(get_val(bs_df, "Total Other Assets (B)", bs_year))
            fa_d  = abs(get_val(bs_df, "Total Fixed Assets (C)", bs_year))
            if ca_d + oa_d + fa_d > 0:
                fig_am = go.Figure(go.Pie(
                    labels=["Current Assets", "Other Assets", "Fixed Assets"],
                    values=[ca_d, oa_d, fa_d], hole=0.45,
                    marker=dict(colors=[_BLUE, f"rgba(55,138,221,0.45)", _PURPLE]),
                ))
                _apply_layout(fig_am, 260, f"Asset Mix — {bs_year}")
                st.plotly_chart(fig_am, use_container_width=True)
                with st.expander("ℹ️ How to read this chart", expanded=False):
                    st.caption(
                        f"**Asset Mix — {bs_year}** — Donut showing how total assets are distributed "
                        "between Current Assets (blue, liquid — cash, receivables, inventory), "
                        "Other Assets (faded blue — long-term receivables, investments, hospital loan etc.), "
                        "and Fixed Assets (purple — property, plant & equipment). "
                        "A trading business typically holds a high proportion of current assets. "
                        "A large Other Assets slice may indicate tied-up capital in non-core investments. "
                        "A growing Fixed Assets share suggests capital expenditure on productive capacity."
                    )
    except Exception as e:
        st.warning(f"BS drill error: {e}")


# ── Main entry point ───────────────────────────────────────────────────────────

def render_analysis_dashboard(
    is_df: pd.DataFrame,
    bs_df: pd.DataFrame,
    cfs_df: pd.DataFrame,
    ratio_df: pd.DataFrame,
    entity_label: str,
    available_years: list,
    entity_zid: str = "consolidated",
    partial_year_months: int = 12,
):
    if not available_years:
        st.warning("No year data available for the Analysis Dashboard.")
        return

    # ── Global controls ────────────────────────────────────────────────────────
    gc1, gc2 = st.columns(2)
    with gc1:
        start_year = st.selectbox(
            "From Year", options=available_years, index=0, key="dash_start_year"
        )
    with gc2:
        end_year = st.selectbox(
            "To Year", options=available_years,
            index=len(available_years) - 1, key="dash_end_year"
        )

    years_in_range = [y for y in available_years if start_year <= y <= end_year]
    if not years_in_range:
        st.warning("Start year must be before or equal to end year.")
        return

    st.info(f"Viewing: **{entity_label}** · Level S · {start_year}–{end_year}")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "Group P&L & Margins",
        "Working Capital & Cash Cycle",
        "Balance Sheet & Leverage",
    ])

    with tab1:
        _tab1_pl(is_df, bs_df, years_in_range, entity_zid, partial_year_months)

    with tab2:
        _tab2_wc(is_df, bs_df, cfs_df, years_in_range, entity_zid, partial_year_months)

    with tab3:
        _tab3_bs(is_df, bs_df, cfs_df, years_in_range, entity_zid)

    # ── Operational history notes (always visible below tabs) ──────────────────
    with st.expander("📋 Operational & Financial History Notes", expanded=False):
        if str(entity_zid) != "consolidated":
            st.caption(
                "Showing entity-specific history. "
                "Group-level notes are also included for context."
            )
        st.markdown(OPERATIONAL_HISTORY_NOTES)
