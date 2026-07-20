import json
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
from processing import common, financial
from processing import consolidation as _consol
from utils.utils import timed
from core import queries
from core.db import get_dataframe
from views.financial_dashboard import render_analysis_dashboard

_LP_PROJ_FILE = Path(__file__).resolve().parent.parent / "data" / "level_p_projections.json"
_LP_PROJ_MAX  = 20  # max snapshots to keep


def _load_lp_projections() -> list:
    try:
        if _LP_PROJ_FILE.exists():
            return json.loads(_LP_PROJ_FILE.read_text(encoding="utf-8")).get("snapshots", [])
    except Exception:
        pass
    return []


def _save_lp_projection(snapshot: dict) -> None:
    snaps = _load_lp_projections()
    snaps.append(snapshot)
    snaps = snaps[-_LP_PROJ_MAX:]
    _LP_PROJ_FILE.write_text(
        json.dumps({"snapshots": snaps}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _fmt(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Return a Styler that formats every numeric column with comma separators."""
    num_cols = df.select_dtypes("number").columns
    fmt = {col: "{:,.1f}" for col in num_cols}
    return df.style.format(fmt, na_rep="")


def _extract_row(df: pd.DataFrame, label: str, name_col: str = "ac_name") -> pd.Series:
    """Return the numeric cells of the first row matching label, or zeros."""
    match = df.loc[df[name_col].astype(str) == label]
    if match.empty:
        num = df.select_dtypes("number").columns
        return pd.Series(0.0, index=num)
    return match.select_dtypes("number").iloc[0]


def _monthly_to_ytd(series: pd.Series) -> pd.Series:
    """
    Convert a per-month net-income Series to year-to-date cumulative.

    Level 0 BS uses net_profit_m (YTD cumsum via groupby), so Level S BS
    must receive the same kind of value — otherwise only January balances.

    Handles column labels that are tuples (2024, 1), strings "('2024','1')",
    or any other format recognised by financial._period_key.
    """
    pk = financial._period_key           # reuse the shared period-key helper

    # Sort columns chronologically
    sorted_idx = sorted(series.index, key=pk)

    # Extract the year component of each column (first element of period key)
    years = [str(pk(c)[0]) for c in sorted_idx]

    # cumsum within each fiscal year, then restore original index order
    ytd = series.reindex(sorted_idx).groupby(years).cumsum()
    ytd.index = sorted_idx
    return ytd


def _merge_ytd(df_prior: pd.DataFrame, df_curr: pd.DataFrame) -> pd.DataFrame:
    """
    Merge prior-years full-year data with current-year YTD data.

    df_prior has columns: meta_cols + [yr-4, yr-3, yr-2, yr-1]
    df_curr  has columns: meta_cols + [yr]

    Result keeps all meta from df_prior and appends the current year column.
    """
    meta = {'ac_code', 'ac_name', 'ac_type', 'ac_lv1', 'ac_lv2', 'ac_lv3', 'ac_lv4', 'ac_lv5'}
    curr_year_cols = [c for c in df_curr.columns if c not in meta]
    merged = df_prior.merge(df_curr[['ac_code'] + curr_year_cols], on='ac_code', how='left')
    num_cols = merged.select_dtypes('number').columns
    merged[num_cols] = merged[num_cols].fillna(0)
    return merged


def _render_ls_notes(key_suffix: str = "") -> None:
    """
    Render the Level S account notes expander, loaded from ls_account_notes.json.
    Displays two tabs inside the expander:
      • Account Notes  — description + contextual notes for every IS/BS/CFS row
      • Ratio Notes    — formula (using exact account names) + interpretation for every ratio
    """
    import json
    from pathlib import Path

    _notes_path = Path(__file__).parent.parent / "data" / "ls_account_notes.json"
    try:
        with open(_notes_path, encoding="utf-8") as _fh:
            _all: dict = json.load(_fh)
    except Exception as _e:
        st.warning(f"Could not load account notes: {_e}")
        return

    # Separate account entries (plain string keys) from the ratios list (_ratios key)
    _acct_notes  = {k: v for k, v in _all.items() if k != "_ratios" and isinstance(v, dict)}
    _ratio_notes = _all.get("_ratios", [])

    with st.expander("📋 Account Notes & Descriptions", expanded=False):
        _tab_acct, _tab_ratio = st.tabs(["Account Notes", "Ratio Notes"])

        # ── Account Notes tab ─────────────────────────────────────────────────
        with _tab_acct:
            for _account, _info in _acct_notes.items():
                _desc = _info.get("description", "").strip()
                _note = _info.get("notes", "").strip()
                if not _desc and not _note:
                    continue
                st.markdown(f"**{_account}**")
                if _desc:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;{_desc}", unsafe_allow_html=True)
                if _note:
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;📌 *{_note}*",
                        unsafe_allow_html=True,
                    )
                st.markdown("---")

        # ── Ratio Notes tab ───────────────────────────────────────────────────
        with _tab_ratio:
            for _r in _ratio_notes:
                _rname  = _r.get("name", "")
                _rform  = _r.get("formula", "")
                _rdesc  = _r.get("description", "").strip()
                _rnote  = _r.get("notes", "").strip()
                st.markdown(f"**{_rname}**")
                if _rform:
                    st.code(_rform, language=None)
                if _rdesc:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;{_rdesc}", unsafe_allow_html=True)
                if _rnote:
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;📌 *{_rnote}*",
                        unsafe_allow_html=True,
                    )
                st.markdown("---")


def _render_mtd_dashboard(pl_s: pd.DataFrame, mtd: dict) -> None:
    """
    Render the MTD Income Statement dashboard inside the Level S Monthly view.

    Parameters
    ----------
    pl_s : Level S IS DataFrame (Monthly perspective, period tuple columns).
    mtd  : dict returned by financial.compute_mtd_is().
    """
    import calendar as _cal

    year      = mtd["year"]
    month     = mtd["month"]
    avg_cols  = mtd["avg_cols"]
    actuals   = mtd["actuals"]
    avgs      = mtd["avgs"]

    month_name = _cal.month_name[month]

    # Describe the 3 months being averaged
    avg_labels = []
    for c in avg_cols:
        y, m = financial._period_key(c)
        avg_labels.append(f"{_cal.month_abbr[m]} {y}")
    avg_period_str = ", ".join(avg_labels) if avg_labels else "N/A (no prior months in range)"

    st.subheader(f"📊 MTD Income Statement — {month_name} {year}")
    st.caption(
        f"**MTD Actual** = transactions posted in {month_name} {year} up to today &nbsp;|&nbsp; "
        f"**3M Avg** = average of: {avg_period_str}",
        unsafe_allow_html=True,
    )

    use_avg = st.toggle(
        "Use 3M Averages for SG&A, Interest & Tax in Net Income",
        value=False,
        key="mtd_use_avg_toggle",
    )

    # Lines whose calculated aggregates are replaced by 3M avg when toggle is ON
    # (Discount Paid and S&D always stay as MTD actual per spec)
    _AVG_KEYS = {
        "Total SG&A",
        "Total Interest & Charges",
        "Net VAT Expenses Cash (B)",
        "0629-Income Tax Expenses (C)",
        "0629-VAT & Tax Total (A+B+C)",
    }

    def _eff(label: str) -> float:
        """Return the effective value: avg when toggle is ON and label is controlled."""
        if use_avg and label in _AVG_KEYS:
            return avgs.get(label, 0.0)
        return actuals.get(label, 0.0)

    # Recalculate EBITDA and Net Income with effective values
    eff_gross_profit  = actuals["Gross Profit"]
    eff_total_sga     = _eff("Total SG&A")
    eff_total_sd      = actuals["Total Sales & Distribution"]   # always actual
    eff_others_direct = actuals["0501-Others Direct Expenses"]  # always actual
    eff_ebitda        = eff_gross_profit + eff_total_sga + eff_total_sd + eff_others_direct
    eff_total_int     = _eff("Total Interest & Charges")
    eff_vat_total     = _eff("0629-VAT & Tax Total (A+B+C)")
    eff_net_income    = eff_ebitda + eff_total_int + eff_vat_total

    # Rows marked as calculated totals (bold in table)
    _CALC_ROWS = {
        "Adjusted Revenue (Pending)", "Gross Profit", "Total SG&A",
        "Total Sales & Distribution", "EBITDA",
        "Total Interest & Charges", "0629-VAT & Tax Total (A+B+C)", "Net Income",
    }

    # Display order
    _DISPLAY = [
        "Revenue",
        "COGS",
        "MRP Discount",
        "Adjusted Revenue (Pending)",
        "Gross Profit",
        "SG&A",
        "0612-Salary Expenses",
        "0613-Employee Bonus",
        "0614-Overtime",
        "0615-Director Remuneration",
        "Total SG&A",
        "0708-Discount Paid",
        "Sales & Distribution Expenses",
        "Total Sales & Distribution",
        "0501-Others Direct Expenses",
        "EBITDA",
        "0630-Bank Interest & Charges",
        "0633-Interest-Loan",
        "Total Interest & Charges",
        "Net VAT Expenses Cash (B)",
        "0629-Income Tax Expenses (C)",
        "0629-VAT & Tax Total (A+B+C)",
        "Net Income",
    ]

    mtd_col_label = "MTD Actual" if not use_avg else "MTD / Avg-Adj ★"

    rows = []
    for lbl in _DISPLAY:
        a = actuals.get(lbl, 0.0)
        v = avgs.get(lbl, 0.0)

        # Override calculated totals with effective values when toggle is ON
        if lbl == "EBITDA":
            a = eff_ebitda
        elif lbl == "Net Income":
            a = eff_net_income
        elif use_avg and lbl in _AVG_KEYS:
            a = avgs.get(lbl, 0.0)

        delta     = a - v
        delta_pct = (delta / abs(v) * 100.0) if v != 0.0 else 0.0

        rows.append({
            "Line Item":      lbl,
            mtd_col_label:    a,
            "3M Average":     v,
            "Δ":              delta,
            "Δ %":            delta_pct,
            "_calc":          lbl in _CALC_ROWS,
        })

    df_disp = pd.DataFrame(rows)

    # ── Styling ───────────────────────────────────────────────────────────
    # Revenue / profit rows: higher MTD vs avg = good (green)
    # Cost / expense rows  : lower MTD vs avg (i.e. Δ > 0 because both negative) = good
    # In both cases Δ > 0 with Level S sign = better, so colour logic is unified:
    #   Δ > 0  →  green  (revenue up, or expense less negative = spending down)
    #   Δ < 0  →  red
    # Note: _calc is dropped before the styler runs, so use row.name (= "Line Item" index)

    df_show = df_disp.drop(columns=["_calc"])

    def _style_row(row):
        # row.name is the "Line Item" string because df_show is indexed by "Line Item"
        is_calc = row.name in _CALC_ROWS
        delta   = row["Δ"]
        n       = len(row)
        base_bg = "background-color: rgba(55,138,221,0.07); font-weight: 600" if is_calc else ""

        delta_colour = (
            "color: #1D9E75; font-weight: 600" if delta > 0
            else "color: #E24B4A; font-weight: 600" if delta < 0
            else "color: inherit"
        )
        if base_bg:
            delta_colour += "; background-color: rgba(55,138,221,0.07)"

        styles = [base_bg] * n
        try:
            d_idx = list(row.index).index("Δ")
            p_idx = list(row.index).index("Δ %")
            styles[d_idx] = delta_colour
            styles[p_idx] = delta_colour
        except ValueError:
            pass
        return styles
    fmt = {
        mtd_col_label: "{:,.1f}",
        "3M Average":  "{:,.1f}",
        "Δ":           "{:+,.1f}",
        "Δ %":         "{:+.1f}%",
    }
    styler = (
        df_show
        .set_index("Line Item")
        .style
        .format(fmt, na_rep="—")
        .apply(_style_row, axis=1)
    )

    if use_avg:
        st.caption(
            "★ **Toggle ON**: Rows marked with ★ use 3M Average values. "
            "EBITDA and Net Income recalculate using averaged SG&A, Interest & Tax. "
            "Discount Paid and S&D always reflect MTD actuals."
        )

    st.dataframe(styler, use_container_width=True)

    # ── Breakdown expanders ────────────────────────────────────────────────
    def _render_breakdown(title: str, df: pd.DataFrame,
                          total_actual: float, total_avg: float) -> None:
        with st.expander(f"📋 {title} — Account Breakdown", expanded=False):
            if df.empty:
                st.info(f"No GL activity found for {title} this month.")
                return
            total_delta = total_actual - total_avg
            st.caption(
                f"**Total MTD**: `{total_actual:,.1f}` &nbsp;|&nbsp; "
                f"**3M Avg**: `{total_avg:,.1f}` &nbsp;|&nbsp; "
                f"**Δ**: `{total_delta:+,.1f}`",
                unsafe_allow_html=True,
            )
            fmt_bd = {"MTD Actual": "{:,.1f}"}
            st.dataframe(
                df.set_index("ac_code")
                  .style.format(fmt_bd, na_rep="—"),
                use_container_width=True,
            )

    _render_breakdown(
        "SG&A",
        mtd["breakdown"]["sga"],
        actuals["Total SG&A"],
        avgs["Total SG&A"],
    )
    _render_breakdown(
        "Sales & Distribution",
        mtd["breakdown"]["sd"],
        actuals["Total Sales & Distribution"],
        avgs["Total Sales & Distribution"],
    )
    _render_breakdown(
        "Interest & Charges",
        mtd["breakdown"]["interest"],
        actuals["Total Interest & Charges"],
        avgs["Total Interest & Charges"],
    )


def _build_ls_ratios(
    pl_s: pd.DataFrame,
    bs_s: pd.DataFrame,
    cfs_s: pd.DataFrame,
    perspective: str = "Yearly",
    partial_year_months: int = 12,
) -> pd.DataFrame:
    """
    Compute Level S financial ratios from IS, BS and CFS DataFrames.

    Returns a formatted string DataFrame with ratio names as rows and
    period columns as columns — ready for st.dataframe display.
    """
    name_col = "ac_name"

    def _g(df: pd.DataFrame, label: str) -> pd.Series:
        """Return first matching row's numeric values, or zeros aligned to df columns."""
        num = df.select_dtypes("number").columns
        mask = df[name_col].astype(str) == label
        if not mask.any():
            return pd.Series(0.0, index=num)
        return df.loc[mask].select_dtypes("number").iloc[0].reindex(num, fill_value=0.0)

    # Period columns from IS (IS and BS share the same period columns)
    _num_cols = pd.Index(
        sorted(
            [c for c in pl_s.columns
             if c != name_col and pd.api.types.is_numeric_dtype(pl_s[c])],
            key=financial._period_key,
        )
    )

    def _align(s: pd.Series) -> pd.Series:
        return s.reindex(_num_cols, fill_value=float("nan"))

    # ── IS rows ────────────────────────────────────────────────────────────────
    adj_rev     = _align(_g(pl_s, "Adjusted Revenue (Pending)"))
    others_rev  = _align(_g(pl_s, "Others Revenue"))
    cogs        = _align(_g(pl_s, "COGS"))
    gp          = _align(_g(pl_s, "Gross Profit"))
    salary      = _align(_g(pl_s, "0612-Salary Expenses"))
    bonus       = _align(_g(pl_s, "0613-Employee Bonus"))
    overtime    = _align(_g(pl_s, "0614-Overtime"))
    total_sga   = _align(_g(pl_s, "Total SG&A"))
    disc_paid   = _align(_g(pl_s, "0708-Discount Paid"))
    sd_exp      = _align(_g(pl_s, "Sales & Distribution Expenses"))
    total_sd    = _align(_g(pl_s, "Total Sales & Distribution"))
    ebitda      = _align(_g(pl_s, "EBITDA"))
    total_int   = _align(_g(pl_s, "Total Interest & Charges"))
    vat_cash    = _align(_g(pl_s, "Net VAT Expenses Cash (B)"))
    vat_tax_tot = _align(_g(pl_s, "0629-VAT & Tax Total (A+B+C)"))

    # ── BS rows ────────────────────────────────────────────────────────────────
    ar_ext      = _align(_g(bs_s, "Accounts Receivable"))
    ar_internal = _align(_g(bs_s, "Accounts Receivable (Internal)"))
    ar_local    = _align(_g(bs_s, "Accounts Receivable (Local)"))
    stock       = _align(_g(bs_s, "0106-Stock in Hand"))
    total_ca    = _align(_g(bs_s, "01-Total Current Assets (A)"))
    ap_local    = _align(_g(bs_s, "Accounts Payable (Local)"))
    ap_intl     = _align(_g(bs_s, "Accounts Payable (International)"))
    ap_internal = _align(_g(bs_s, "Accounts Payable (Internal)"))
    money_agent = _align(_g(bs_s, "0904-Money Agent Liability"))
    total_cl    = _align(_g(bs_s, "Current Liability (A)"))
    stb_loan    = _align(_g(bs_s, "1001-Short Term Bank Loan (B)"))
    total_stl   = _align(_g(bs_s, "Total Short Term Liability (C)"))
    lt_loan     = _align(_g(bs_s, "1201-Long Term Bank Loan (E)"))
    total_eq    = _align(_g(bs_s, "Total Equity"))

    # ── CFS rows (CFS has delta cols only — align to IS/BS col set) ────────────
    _cfs_num = pd.Index(
        sorted(
            [c for c in cfs_s.columns
             if c != name_col and pd.api.types.is_numeric_dtype(cfs_s[c])],
            key=financial._period_key,
        )
    )
    def _gc(label: str) -> pd.Series:
        mask = cfs_s[name_col].astype(str) == label
        if not mask.any():
            return pd.Series(float("nan"), index=_num_cols)
        raw = cfs_s.loc[mask].select_dtypes("number").iloc[0]
        return raw.reindex(_cfs_num, fill_value=float("nan")).reindex(_num_cols, fill_value=float("nan"))

    cfo = _gc("Cash from Operations")
    cfi = _gc("Cash from Investing")

    # ── Denominators (safe against divide-by-zero) ─────────────────────────────
    _safe = lambda s: s.where(s != 0, other=float("nan"))
    adj_plus_oth  = _safe(adj_rev + others_rev)   # > 0 (revenue base for %)
    safe_cogs_abs = _safe(cogs.abs())
    safe_adj      = _safe(adj_rev.abs())
    safe_eq       = _safe(total_eq.abs())
    safe_cl       = _safe(total_cl.abs())

    # Days factor: 365 for yearly, 30 as monthly approximation.
    # For the running (partial) year, scale by months elapsed so WC days
    # are comparable to full-year prior periods.
    days = 365 if perspective == "Yearly" else 30
    _max_int_col = max((c for c in _num_cols if isinstance(c, int)), default=None)
    _days_s = pd.Series(
        [days * partial_year_months / 12 if c == _max_int_col else days for c in _num_cols],
        index=_num_cols,
    )

    # ── IS / Profitability ─────────────────────────────────────────────────────
    markup       = (adj_rev / safe_cogs_abs - 1) * 100
    gp_margin    = gp / adj_plus_oth * 100
    staff_pct    = (salary + bonus + overtime).abs() / adj_plus_oth * 100
    sga_pct      = total_sga.abs() / adj_plus_oth * 100
    disc_pct     = disc_paid.abs() / adj_plus_oth * 100
    dist_pct     = sd_exp.abs() / adj_plus_oth * 100
    tsd_pct      = total_sd.abs() / adj_plus_oth * 100
    ebitda_pct   = ebitda / adj_plus_oth * 100
    int_cov      = ebitda / _safe(total_int.abs())
    vat_pct      = vat_cash.abs() / adj_plus_oth * 100
    tax_pct      = vat_tax_tot.abs() / adj_plus_oth * 100

    # ── Liquidity ──────────────────────────────────────────────────────────────
    adj_ca    = total_ca - ar_internal.abs() - ar_local.abs()
    adj_cl_lq = _safe(total_cl.abs() - ap_internal.abs() - money_agent.abs())
    curr_rat  = adj_ca / adj_cl_lq
    quick_rat = (adj_ca - stock) / adj_cl_lq

    # ── Efficiency / Working Capital Days ─────────────────────────────────────
    dso = ar_ext * _days_s / safe_adj
    dio = stock  * _days_s / safe_cogs_abs
    dpo = (ap_local + ap_intl).abs() * _days_s / safe_cogs_abs
    ccc = dio + dso - dpo

    # ── Leverage ───────────────────────────────────────────────────────────────
    de1 = (stb_loan + money_agent + lt_loan).abs() / safe_eq
    de2 = (total_stl + lt_loan).abs() / safe_eq

    # ── CFS ────────────────────────────────────────────────────────────────────
    ocf_cov = cfo / safe_cl
    fcf     = cfo + cfi

    # ── Format helper ──────────────────────────────────────────────────────────
    def _fmt_row(label: str, series: pd.Series, fmt: str) -> dict:
        row = {"Ratio": label}
        for col in _num_cols:
            val = series.get(col, float("nan"))
            if pd.isna(val) or (fmt and val == 0.0):
                row[col] = ""
            else:
                try:
                    row[col] = fmt.format(val)
                except Exception:
                    row[col] = ""
        return row

    def _hdr(label: str) -> dict:
        return {"Ratio": label, **{col: "" for col in _num_cols}}

    rows = [
        _hdr("── Profitability ───────────────────────────"),
        _fmt_row("Markup on COGS (%)",          markup,     "{:.1f}%"),
        _fmt_row("Gross Profit Margin (%)",      gp_margin,  "{:.1f}%"),
        _fmt_row("Staff Cost / Revenue (%)",     staff_pct,  "{:.1f}%"),
        _fmt_row("Total SG&A / Revenue (%)",     sga_pct,    "{:.1f}%"),
        _fmt_row("Discount Paid / Revenue (%)",  disc_pct,   "{:.1f}%"),
        _fmt_row("Distribution / Revenue (%)",   dist_pct,   "{:.1f}%"),
        _fmt_row("Total S&D / Revenue (%)",      tsd_pct,    "{:.1f}%"),
        _fmt_row("EBITDA Margin (%)",            ebitda_pct, "{:.1f}%"),
        _fmt_row("Interest Coverage (×)",        int_cov,    "{:.2f}×"),
        _fmt_row("VAT Expense / Revenue (%)",    vat_pct,    "{:.1f}%"),
        _fmt_row("Total Tax / Revenue (%)",      tax_pct,    "{:.1f}%"),
        _hdr("── Liquidity ────────────────────────────────"),
        _fmt_row("Current Ratio (×)",            curr_rat,   "{:.2f}×"),
        _fmt_row("Quick Ratio (×)",              quick_rat,  "{:.2f}×"),
        _hdr("── Working Capital Days ─────────────────────"),
        _fmt_row("DSO — Receivable Days",        dso,        "{:.0f} days"),
        _fmt_row("DIO — Inventory Days",         dio,        "{:.0f} days"),
        _fmt_row("DPO — Payable Days",           dpo,        "{:.0f} days"),
        _fmt_row("Cash Conversion Cycle",        ccc,        "{:.0f} days"),
        _hdr("── Leverage ────────────────────────────────"),
        _fmt_row("D/E 1 — Bank + MA + LT (×)",  de1,        "{:.2f}×"),
        _fmt_row("D/E 2 — STL + LT (×)",        de2,        "{:.2f}×"),
        _hdr("── Cash Flow ────────────────────────────────"),
        _fmt_row("OCF Coverage (×)",             ocf_cov,    "{:.2f}×"),
        _fmt_row("Free Cash Flow",               fcf,        "{:,.0f}"),
    ]
    return pd.DataFrame(rows)


def _render_zid_contribution_breakdown(
    pl_s: pd.DataFrame,
    bs_s: pd.DataFrame,
    zid_frames_pl: dict,
    zid_frames_bs: dict,
    perspective: str,
    key_suffix: str,
) -> None:
    """
    Render a ZID contribution breakdown panel below the Level S balance check.

    Lets the user pick any Level S IS or BS account name and a period, then
    shows a per-ZID table and a Plotly pie chart of absolute contributions.
    Only intended for the consolidated view.
    """
    import json
    import plotly.express as px
    from pathlib import Path

    # ── ZID display names from businesses.json ────────────────────────────────
    _biz_path = Path(__file__).parent.parent / "data" / "businesses.json"
    try:
        with open(_biz_path, encoding="utf-8") as _fh:
            _biz_data = json.load(_fh)
        _zid_name_map = {
            int(k): v.get("zorg", f"ZID {k}")
            for k, v in _biz_data.get("businesses", {}).items()
        }
    except Exception:
        _zid_name_map = {}

    def _zid_label(zid: int) -> str:
        name = _zid_name_map.get(zid, f"ZID {zid}")
        return f"{zid} — {name}"

    # ── Build Level S IS and BS for each ZID (pure pandas, no DB calls) ───────
    _zid_pl_ls: dict = {}
    _zid_bs_ls: dict = {}

    for _bz, _bz_pl in zid_frames_pl.items():
        _bz_bs = zid_frames_bs.get(_bz, pd.DataFrame())
        if _bz_pl is None or _bz_pl.empty:
            continue
        try:
            _bz_pl_s = financial.build_pl_level_s(_bz_pl, perspective)
            _bz_ni   = _extract_row(_bz_pl_s, "Net Income")
            # BS needs YTD cumulative NI for monthly to balance correctly
            _bz_ni_bs = _monthly_to_ytd(_bz_ni) if perspective == "Monthly" else _bz_ni
            _bz_bs_s  = financial.build_bs_level_s(_bz_bs, _bz_ni_bs, zid=_bz)
            _zid_pl_ls[_bz] = _bz_pl_s
            _zid_bs_ls[_bz] = _bz_bs_s
        except Exception:
            pass

    if not _zid_pl_ls:
        return

    # ── Collect all Level S account names (IS first, then BS, no duplicates) ──
    _name_col = "ac_name"
    _seen: set = set()
    _all_names: list = []
    for _nm in list(pl_s[_name_col]) + list(bs_s[_name_col]):
        if isinstance(_nm, str) and _nm and _nm not in _seen:
            _seen.add(_nm)
            _all_names.append(_nm)

    # ── Period columns from consolidated pl_s ────────────────────────────────
    _period_cols = sorted(
        [c for c in pl_s.columns
         if c != _name_col and pd.api.types.is_numeric_dtype(pl_s[c])],
        key=financial._period_key,
    )
    if not _period_cols:
        return

    # ── UI ────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("ZID Contribution Breakdown")
    _c1, _c2 = st.columns(2)
    with _c1:
        _sel_name = st.selectbox(
            "Account",
            _all_names,
            key=f"zid_brkdn_account_{key_suffix}",
        )
    with _c2:
        _sel_period = st.selectbox(
            "Period",
            _period_cols,
            index=len(_period_cols) - 1,   # default: most recent period
            key=f"zid_brkdn_period_{key_suffix}",
        )

    # ── Per-ZID lookup ────────────────────────────────────────────────────────
    _rows = []
    for _bz in sorted(_zid_pl_ls.keys()):
        _val = 0.0
        # Search IS DataFrame first, then BS
        for _src_df in (_zid_pl_ls.get(_bz), _zid_bs_ls.get(_bz)):
            if _src_df is None or _src_df.empty:
                continue
            _mask = _src_df[_name_col].astype(str) == str(_sel_name)
            if not _mask.any():
                continue
            if _sel_period in _src_df.columns:
                # Use chained indexing so tuple period columns (monthly)
                # are treated as a single column name, not a multi-index key.
                _val = float(_src_df.loc[_mask][_sel_period].iloc[0])
            break   # found in this source — don't also search BS
        _rows.append({"ZID": _zid_label(_bz), "Value": _val})

    _brkdn_df = pd.DataFrame(_rows)
    # Drop ZIDs with zero contribution to keep the view clean
    _brkdn_df = _brkdn_df[_brkdn_df["Value"] != 0.0].reset_index(drop=True)

    if _brkdn_df.empty:
        st.info(f"All ZIDs show zero for '{_sel_name}' in period {_sel_period}.")
        return

    # ── Table ─────────────────────────────────────────────────────────────────
    st.dataframe(
        _brkdn_df.style.format({"Value": "{:,.1f}"}),
        use_container_width=True,
    )

    # ── Pie chart (sized by absolute value; note in title if values are mixed) ─
    _pie_df = _brkdn_df.copy()
    _pie_df["Abs"] = _pie_df["Value"].abs()
    _pie_df = _pie_df[_pie_df["Abs"] > 0]
    if not _pie_df.empty:
        _has_mixed = (_brkdn_df["Value"] > 0).any() and (_brkdn_df["Value"] < 0).any()
        _title = (
            f"{_sel_name}  ·  {_sel_period}"
            + ("  (absolute — mixed signs present)" if _has_mixed else "")
        )
        _fig = px.pie(
            _pie_df,
            names="ZID",
            values="Abs",
            title=_title,
            hole=0.3,
        )
        _fig.update_traces(textposition="inside", textinfo="percent+label")
        _fig.update_layout(margin=dict(t=60, b=20, l=20, r=20))
        st.plotly_chart(_fig, use_container_width=True)


def _sanity_checks(
    pl_sorted, pl_lv1, pl_lv2, pl_s,
    bs_lv0, bs_lv1, bs_lv2, bs_s,
    summary_df, summary_df1, summary_df2, summary_s,
):
    """Render a cross-level sanity check table at the bottom of Level S."""
    st.markdown("---")
    st.subheader("Cross-Level Sanity Checks")
    st.caption(
        "Net Profit/Loss, BS Balance Check, and CFS Cash-flow Check across "
        "all levels. All values should be internally consistent. "
        "BS Balance Check and CFS Cash-flow Check should be near zero."
    )

    def _build_check(label_0, label_s, dfs_and_labels, name_col="ac_name"):
        """Build a check DataFrame with one row per level.

        Normalises all period-column labels to str so that integer columns
        (from Level S select_dtypes) and string columns (from Level 0/1/2)
        collapse into the same column instead of creating duplicates.
        """
        rows = []
        for lvl_name, df, label in dfs_and_labels:
            s = _extract_row(df, label, name_col)
            # Normalise index to str to avoid int/str column duplication
            s.index = s.index.astype(str)
            rows.append(pd.Series({**{"Level": lvl_name}, **s.to_dict()}))
        return pd.DataFrame(rows).set_index("Level")

    # Net Profit / Net Income
    # Level S IS uses management-view sign (positive=profit, negative=loss).
    # Level 0/1/2 use accounting sign (positive=loss, negative=profit).
    # Negate Level S Net Income so all rows share the same sign convention
    # in this comparison table.
    _ni_s = _extract_row(pl_s, "Net Income")
    _ni_s_negated = -_ni_s
    _pl_s_for_check = pl_s.copy()
    name_col_s = "ac_name"
    _pl_s_for_check.loc[
        _pl_s_for_check[name_col_s] == "Net Income", _ni_s.index
    ] = _ni_s_negated.values

    np_check = _build_check(
        "Net Profit/Loss", "Net Income",
        [
            ("Level 0", pl_sorted,        "Net Profit/Loss"),
            ("Level 1", pl_lv1,           "Net Profit/Loss"),
            ("Level 2", pl_lv2,           "Net Profit/Loss"),
            ("Level S", _pl_s_for_check,  "Net Income"),
        ]
    )

    # BS Balance Check
    bs_check = _build_check(
        "Balance Check", "Balance Check",
        [
            ("Level 0", bs_lv0, "Balance Check"),
            ("Level 1", bs_lv1, "Balance Check"),
            ("Level 2", bs_lv2, "Balance Check"),
            ("Level S", bs_s,   "Balance Check"),
        ]
    )

    # CFS Cash-flow Check
    cfs_check = _build_check(
        "Cash-flow Check", "Cash-flow Check",
        [
            ("Level 0", summary_df,  "Cash-flow Check"),
            ("Level 1", summary_df1, "Cash-flow Check"),
            ("Level 2", summary_df2, "Cash-flow Check"),
            ("Level S", summary_s,   "Cash-flow Check"),
        ]
    )

    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Net Profit / Net Income**")
        st.dataframe(np_check, use_container_width=True)
    with cols[1]:
        st.markdown("**BS Balance Check** *(should be ~0)*")
        st.dataframe(bs_check, use_container_width=True)
    with cols[2]:
        st.markdown("**CFS Cash-flow Check** *(should be ~0)*")
        st.dataframe(cfs_check, use_container_width=True)


def _level_p_sanity_checks(pl_p: pd.DataFrame, pl_s: pd.DataFrame, name_col: str = "ac_name"):
    """
    Render Level P arithmetic checks and cross-level NI comparison.
    Shows: (1) IS formula verification and (2) Level P NI vs Level S NI.
    """
    st.markdown("---")
    st.subheader("Level P Sanity Checks")
    st.caption(
        "Arithmetic verification of Level P IS rows. "
        "GP = Adj Revenue + COGS, EBITDA = GP + SG&A + S&D + Others Direct, "
        "NI = EBITDA + Interest + VAT. "
        "Level P NI should equal Level S NI (Others Revenue redistributed equally into SG&A and S&D)."
    )

    num_p = financial._ls_num_cols(pl_p)
    num_s = financial._ls_num_cols(pl_s)

    def _gp(df, label, num):
        r = df.loc[df[name_col].astype(str) == label]
        if r.empty:
            return pd.Series(0.0, index=num)
        return r.select_dtypes("number").iloc[0].reindex(num, fill_value=0.0)

    _adj_rev  = _gp(pl_p, "Adjusted Revenue (Pending)", num_p)
    _cogs     = _gp(pl_p, "COGS", num_p)
    _gross_p  = _gp(pl_p, "Gross Profit", num_p)
    _sga      = _gp(pl_p, "Total SG&A", num_p)
    _sd       = _gp(pl_p, "Total Sales & Distribution", num_p)
    _od       = _gp(pl_p, "0501-Others Direct Expenses", num_p)
    _ebitda   = _gp(pl_p, "EBITDA", num_p)
    _int      = _gp(pl_p, "Total Interest & Charges", num_p)
    _vat      = _gp(pl_p, "0629-VAT & Tax Total (A+B+C)", num_p)
    _ni_p     = _gp(pl_p, "Net Income", num_p)
    _ni_s     = _gp(pl_s, "Net Income", num_s)

    _gp_computed     = _adj_rev + _cogs
    _ebitda_computed = _gross_p + _sga + _sd + _od
    _ni_computed     = _ebitda + _int + _vat

    _lbl = [str(common._period_col_label(c)) for c in num_p]
    _lbl_s = [str(common._period_col_label(c)) for c in num_s]

    def _fmt_val(v):
        try:
            f = float(v)
            return f"{f:,.0f}" if abs(f) >= 1 else f"{f:.2f}"
        except Exception:
            return str(v)

    def _check_row(check_name, reported, computed, col_labels):
        diff = (reported - computed).fillna(0.0)
        status = "✅ Pass" if diff.abs().max() < 1 else "⚠️ Diff"
        row = {"Check": check_name, "Status": status}
        row.update({c: _fmt_val(v) for c, v in zip(col_labels, diff.values)})
        return row

    arith_rows = [
        _check_row("GP = Adj Rev + COGS",              _gross_p, _gp_computed,     _lbl),
        _check_row("EBITDA = GP + SG&A + S&D + OthDir", _ebitda, _ebitda_computed, _lbl),
        _check_row("NI = EBITDA + Interest + VAT",      _ni_p,   _ni_computed,     _lbl),
    ]
    arith_df = pd.DataFrame(arith_rows).set_index("Check")
    st.markdown("**IS Arithmetic Verification** *(difference shown; should be 0)*")
    st.dataframe(arith_df, use_container_width=True)

    # Cross-level NI comparison (Level P NI vs Level S NI)
    ni_p_row = {"Level": "Level P"};  ni_p_row.update({c: _fmt_val(v) for c, v in zip(_lbl, _ni_p.values)})
    ni_s_row = {"Level": "Level S"};  ni_s_row.update({c: _fmt_val(v) for c, v in zip(_lbl_s, _ni_s.values)})
    ni_cross_df = pd.DataFrame([ni_p_row, ni_s_row]).set_index("Level")
    st.markdown("**Level P vs Level S — Net Income** *(should be identical)*")
    st.dataframe(ni_cross_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Config Editor
# ─────────────────────────────────────────────────────────────────────────────

def _render_config_editor():
    import json
    from pathlib import Path
    st.title("⚙️ Financial Config Editor")
    _base = Path(__file__).resolve().parent.parent / "data"
    _files = {
        "hierarchy.json": str(_base / "hierarchy.json"),
        "labels.json": str(_base / "labels.json"),
        "level_s_mapping.json": str(_base / "level_s_mapping.json"),
    }
    _sel = st.radio("Select file", list(_files.keys()), horizontal=True, key="cfg_file_radio")
    _path = _files[_sel]
    try:
        with open(_path, "r", encoding="utf-8") as _fh:
            _raw = json.load(_fh)
        _txt = st.text_area(
            f"Edit `{_sel}` (valid JSON)",
            value=json.dumps(_raw, indent=2),
            height=600,
            key=f"cfg_ta_{_sel}",
        )
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("💾 Save", key="cfg_save_btn"):
                try:
                    _parsed = json.loads(_txt)
                    with open(_path, "w", encoding="utf-8") as _fh:
                        json.dump(_parsed, _fh, indent=2, ensure_ascii=False)
                    st.success(f"✅ `{_sel}` saved.")
                    st.cache_data.clear()
                except json.JSONDecodeError as _e:
                    st.error(f"Invalid JSON: {_e}")
    except FileNotFoundError:
        st.warning(f"File not found: `{_path}`")


# ─────────────────────────────────────────────────────────────────────────────
# Quarterly view
# ─────────────────────────────────────────────────────────────────────────────

def _render_quarterly_view(businesses, income_label_df, balance_label_df,
                            level_options, selected_year, end_month, global_zid):
    """
    Quarterly financial statements.
    Loads: 2 prior years (full) + current year Jan-end_month (monthly).
    e.g. year=2026 month=6 → Q1 2024 … Q2 2026.
    Collapses to quarterly, then reuses Monthly view builders.
    """
    import calendar as _cal

    st.title("Financial Statement Analysis — Quarterly")

    # ── Build quarterly DataFrames for every business ─────────────────────────
    # process_data_month(year) already returns (year, m) AND (year-1, m) columns.
    # A second call with (year-1) adds (year-2, m) columns for the extra prior year.
    main_data_dict_pl_q: dict = {}
    main_data_dict_bs_q: dict = {}

    for zid_k, details in businesses.items():
        for project in details.get('projects', [None]):
            try:
                # Primary: current year + 1 prior year
                pl_monthly = financial.process_data_month(
                    zid_k, selected_year, 1, end_month, 'Income Statement', income_label_df,
                    project, {'Asset', 'Liability'},
                )
                bs_monthly = financial.process_data_month(
                    zid_k, selected_year, 1, end_month, 'Balance Sheet', balance_label_df,
                    project, {'Income', 'Expenditure'},
                )

                # Secondary: year-1 full year → extracts year-2 columns
                yr2 = selected_year - 2
                try:
                    pl_prev = financial.process_data_month(
                        zid_k, selected_year - 1, 1, 12, 'Income Statement', income_label_df,
                        project, {'Asset', 'Liability'},
                    )
                    bs_prev = financial.process_data_month(
                        zid_k, selected_year - 1, 1, 12, 'Balance Sheet', balance_label_df,
                        project, {'Income', 'Expenditure'},
                    )
                    pl_yr2 = [c for c in pl_prev.columns if isinstance(c, tuple) and c[0] == yr2]
                    bs_yr2 = [c for c in bs_prev.columns if isinstance(c, tuple) and c[0] == yr2]
                    if pl_yr2:
                        pl_monthly = pl_monthly.merge(
                            pl_prev[['ac_code'] + pl_yr2], on='ac_code', how='left'
                        )
                    if bs_yr2:
                        bs_monthly = bs_monthly.merge(
                            bs_prev[['ac_code'] + bs_yr2], on='ac_code', how='left'
                        )
                except Exception:
                    pass  # year-2 data unavailable — show what we have

                pl_q, bs_q = financial.collapse_monthly_to_quarterly(pl_monthly, bs_monthly)
                main_data_dict_pl_q[(zid_k, project)] = pl_q
                main_data_dict_bs_q[(zid_k, project)] = bs_q
            except Exception as _e:
                st.warning(f"Could not load quarterly data for ZID {zid_k}: {_e}")

    # ── Business / Level selectors ────────────────────────────────────────────
    cols = st.columns(2)
    with cols[0]:
        _CONSOL_KEY = ("consolidated", "All Businesses - Consolidated")
        biz_options  = [_CONSOL_KEY] + [k for k in main_data_dict_pl_q.keys()]
        default_idx  = next(
            (i for i, k in enumerate(biz_options) if str(k[0]) == str(global_zid)), 0
        )
        analyse_zid = st.selectbox("View Statements For", biz_options, index=default_idx,
                                    key="qtr_biz_sel")
    _is_consolidated = str(analyse_zid[0]) == "consolidated"
    with cols[1]:
        if _is_consolidated:
            _level_opts_q = [
                "Level C - Raw Consolidation",
                "Level C2 - Consolidated Detail",
                "Level 1 - Moderate Detail",
                "Level 2 - Least Detail",
                "Level S - Customised Detail",
            ]
        else:
            _level_opts_q = level_options
        selected_level = st.selectbox("Select Level", _level_opts_q, key="qtr_level_sel")

    # ── Raw DataFrames ────────────────────────────────────────────────────────
    _hier_cols_q = ['ac_type', 'ac_lv1', 'ac_lv2', 'ac_lv3', 'ac_lv4', 'ac_lv5']
    from processing import consolidation as _consol_q
    level_c_is = level_c_bs = c2_is = c2_bs = None

    if _is_consolidated:
        _zid_frames_pl_q = {
            int(k[0]): v.drop(columns=_hier_cols_q, errors='ignore')
            for k, v in main_data_dict_pl_q.items()
        }
        _zid_frames_bs_q = {
            int(k[0]): v.drop(columns=_hier_cols_q, errors='ignore')
            for k, v in main_data_dict_bs_q.items()
        }
        level_c_is = _consol_q.build_level_c(_zid_frames_pl_q, kind="is")
        level_c_bs = _consol_q.build_level_c(_zid_frames_bs_q, kind="bs")
        c2_is = _consol_q.build_level_c2_is(level_c_is)
        c2_bs = _consol_q.build_level_c2_bs(level_c_bs)
        pl_raw = c2_is
        bs_raw = c2_bs
    else:
        if analyse_zid not in main_data_dict_pl_q:
            st.warning("No quarterly data available for this entity.")
            return
        pl_raw = main_data_dict_pl_q[analyse_zid]
        bs_raw = main_data_dict_bs_q[analyse_zid]

    pl_lv0 = pl_raw.drop(columns=_hier_cols_q, errors='ignore')
    bs_lv0 = bs_raw.drop(columns=_hier_cols_q, errors='ignore')

    # ── Level 0 builders ──────────────────────────────────────────────────────
    pl_sorted, net_profit, net_profit_m, dep_row = financial.sort_pl_level0(
        pl_lv0, selected_perspective='Monthly'
    )
    bs_lv0 = financial.append_net_profit_to_bs_level0(bs_lv0, net_profit_m)
    coc_lv0 = financial.cash_open_close(bs_lv0)

    _cfs_qtr_available = True
    try:
        cfs_df, summary_df = financial.make_cashflow_statement_level0(
            pl_lv0, bs_lv0, coc_lv0, selected_perspective='Monthly'
        )
    except Exception:
        _cfs_qtr_available = False

    pl_lv1, pl_lv2 = financial.level_builder(pl_sorted, "IS")
    bs_lv1, bs_lv2 = financial.level_builder(bs_lv0, "BS")

    pl_lv1, bs_lv1, net_profitlv1, dep_rowlv1 = financial.add_np_and_balance_lv1(
        pl_lv1, bs_lv1, selected_perspective='Monthly'
    )
    if _cfs_qtr_available:
        cfs_lv1 = financial.consolidate_cfs(cfs_df, level=1, debug=True)
        cfs_lv1, summary_df1 = financial.build_cfs_level1_summary_df(
            cfs_lv1, net_profitlv1, dep_rowlv1, coc_lv0
        )

    pl_lv2, bs_lv2, net_profitlv2 = financial.add_np_and_balance_lv2(
        pl_lv2, bs_lv2, selected_perspective='Monthly'
    )
    if _cfs_qtr_available:
        cfs_lv2 = financial.consolidate_cfs(cfs_df, level=2, debug=True)
        cfs_lv2, summary_df2 = financial.build_cfs_level2_summary(
            cfs_lv2, net_profitlv2, dep_rowlv1, coc_lv0
        )

    # ── Helper to rename (year, quarter) columns for display ─────────────────
    def _fmt_qtr(df: pd.DataFrame):
        num = df.select_dtypes("number").columns
        rename_map = {}
        for c in num:
            if isinstance(c, tuple) and len(c) == 2:
                rename_map[c] = f"Q{c[1]} {c[0]}"
        df2 = df.rename(columns=rename_map)
        new_num = [rename_map.get(c, c) for c in num]
        fmt = {col: "{:,.1f}" for col in new_num}
        return df2.style.format(fmt, na_rep="")

    # ─��� Level S ───────────────────────────────────────────────────────────────
    _az  = str(analyse_zid[0]) if analyse_zid else None
    _ap  = analyse_zid[1] if analyse_zid and len(analyse_zid) > 1 else None
    _num_cols_q = pl_raw.select_dtypes("number").columns
    _years_in_data_q = sorted({c[0] for c in _num_cols_q if isinstance(c, tuple)})

    _vat_rows_q = {}
    if _az and not _is_consolidated:
        try:
            _sql_vat_q, _p_vat_q = queries.get_vat_breakdown_gl(
                zid=_az, project=_ap, year_list=_years_in_data_q, smonth=1, emonth=12,
            )
            _gl_vat_q = get_dataframe(_sql_vat_q, _p_vat_q)
            _vat_rows_q = financial.compute_vat_is_rows(
                _gl_vat_q, _num_cols_q, selected_perspective='Monthly'
            )
        except Exception:
            pass

    pl_s  = financial.build_pl_level_s(
        pl_raw, selected_perspective='Monthly', vat_rows=_vat_rows_q
    )
    net_income_s_q    = _extract_row(pl_s, "Net Income")
    net_income_s_ytd_q = _monthly_to_ytd(net_income_s_q)
    bs_s  = financial.build_bs_level_s(bs_raw, net_income_s_ytd_q, zid=_az)
    cfs_s, summary_s = financial.build_cfs_level_s(
        pl_raw, bs_raw, coc_lv0, net_income_s_q, zid=_az
    )

    # ── Dispatch to selected level ────────────────────────────────────────────
    def _expanders(pl_df, bs_df, cfs_df_=None, summary_df_=None, fmt_fn=_fmt_qtr):
        with st.expander("Income Statement", expanded=True):
            st.dataframe(fmt_fn(pl_df), use_container_width=True)
        with st.expander("Balance Sheet", expanded=True):
            st.dataframe(fmt_fn(bs_df), use_container_width=True)
        if cfs_df_ is not None:
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(fmt_fn(cfs_df_), use_container_width=True)
        if summary_df_ is not None:
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(fmt_fn(summary_df_), use_container_width=True)

    def _qtr_to_yearly(pl_q, bs_q, cfs_q=None):
        """Collapse quarterly (year,q) tuple columns to integer year columns.
        IS/CFS: sum quarters within year.  BS: last quarter of each year."""
        qcols_pl = [c for c in pl_q.columns if isinstance(c, tuple) and len(c) == 2]
        years    = sorted({c[0] for c in qcols_pl})
        meta_pl  = [c for c in pl_q.columns  if c not in qcols_pl]
        meta_bs  = [c for c in bs_q.columns  if c not in [c2 for c2 in bs_q.columns if isinstance(c2, tuple)]]
        pl_yr = pl_q[meta_pl].copy()
        bs_yr = bs_q[meta_bs].copy()
        for y in years:
            ycols = sorted([c for c in qcols_pl if c[0] == y])
            pl_yr[y] = pl_q[ycols].sum(axis=1)
            bs_yr[y] = bs_q[ycols[-1]]
        if cfs_q is not None and not cfs_q.empty:
            qcols_cf = [c for c in cfs_q.columns if isinstance(c, tuple) and len(c) == 2]
            meta_cf  = [c for c in cfs_q.columns if c not in qcols_cf]
            cfs_yr = cfs_q[meta_cf].copy()
            for y in years:
                ycols_cf = sorted([c for c in qcols_cf if c[0] == y])
                if ycols_cf:
                    cfs_yr[y] = cfs_q[ycols_cf].sum(axis=1)
        else:
            cfs_yr = pd.DataFrame()
        return pl_yr, bs_yr, cfs_yr

    if selected_level == "Level S - Customised Detail":
        _expanders(pl_s, bs_s, cfs_s, summary_s)
        _dl = common.create_combined_ls_download_link(
            pl_s=pl_s, bs_s=bs_s, cfs_s=cfs_s,
            filename="LevelS_Financial_Statements_Quarterly.xlsx",
        )
        st.markdown(_dl, unsafe_allow_html=True)

        # ── Financial Ratios (computed on yearly-aggregated data) ─────────────
        _pl_s_yr_q, _bs_s_yr_q, _cfs_s_yr_q = _qtr_to_yearly(pl_s, bs_s, cfs_s if _cfs_qtr_available else None)
        _avail_yrs_q = sorted({c[0] for c in pl_s.select_dtypes("number").columns if isinstance(c, tuple)})
        _entity_label_q = "Consolidated Group" if _is_consolidated else f"ZID {_az}"
        try:
            _ratio_df_q = _build_ls_ratios(
                _pl_s_yr_q, _bs_s_yr_q, _cfs_s_yr_q,
                perspective="Yearly",
                partial_year_months=end_month,
            )
        except Exception:
            _ratio_df_q = None
        with st.expander("📊 Financial Ratios", expanded=False):
            if _ratio_df_q is not None:
                st.dataframe(_ratio_df_q.set_index("Ratio"), use_container_width=True)
            else:
                st.warning("Could not compute ratios.")

        # ── Cross-Level Sanity Checks ─────────────────────────────────────────
        if _cfs_qtr_available:
            _sanity_checks(
                pl_sorted, pl_lv1, pl_lv2, pl_s,
                bs_lv0, bs_lv1, bs_lv2, bs_s,
                summary_df, summary_df1, summary_df2, summary_s,
            )

        # ── Notes & Context / ZID Breakdown / Analysis Dashboard ─────────────
        if _is_consolidated:
            _panel_q = st.radio(
                "View",
                ["📋 Notes & Context", "📊 ZID Contribution Breakdown", "📈 Analysis Dashboard"],
                horizontal=True,
                key="ls_panel_q",
            )
            if _panel_q == "📋 Notes & Context":
                _render_ls_notes(key_suffix="q")
            elif _panel_q == "📊 ZID Contribution Breakdown":
                _render_zid_contribution_breakdown(
                    pl_s, bs_s,
                    _zid_frames_pl_q, _zid_frames_bs_q,
                    perspective="Monthly",
                    key_suffix="q",
                )
            else:
                render_analysis_dashboard(
                    _pl_s_yr_q, _bs_s_yr_q, _cfs_s_yr_q, _ratio_df_q,
                    _entity_label_q, _avail_yrs_q,
                    entity_zid="consolidated",
                    partial_year_months=end_month,
                )
        else:
            _panel_q_single = st.radio(
                "View",
                ["📋 Notes & Context", "📈 Analysis Dashboard"],
                horizontal=True,
                key="ls_panel_q_single",
            )
            if _panel_q_single == "📋 Notes & Context":
                _render_ls_notes(key_suffix="q_single")
            else:
                render_analysis_dashboard(
                    _pl_s_yr_q, _bs_s_yr_q, _cfs_s_yr_q, _ratio_df_q,
                    _entity_label_q, _avail_yrs_q,
                    entity_zid=str(_az),
                    partial_year_months=end_month,
                )

    elif selected_level == "Level 0 - Most Detail":
        _expanders(pl_sorted, bs_lv0,
                   cfs_df if _cfs_qtr_available else None,
                   summary_df if _cfs_qtr_available else None)
    elif selected_level == "Level 1 - Moderate Detail":
        _expanders(pl_lv1, bs_lv1,
                   cfs_lv1 if _cfs_qtr_available else None,
                   summary_df1 if _cfs_qtr_available else None)
    elif selected_level == "Level 2 - Least Detail":
        _expanders(pl_lv2, bs_lv2,
                   cfs_lv2 if _cfs_qtr_available else None,
                   summary_df2 if _cfs_qtr_available else None)
    elif selected_level == "Level P - Projection View":
        pl_p_q, bs_p_q = financial.build_condensed_view(pl_s, bs_s)
        _expanders(pl_p_q, bs_p_q, summary_s if _cfs_qtr_available else None, fmt_fn=_fmt_qtr)
        st.markdown(
            common.create_combined_ls_download_link(
                pl_s=pl_p_q, bs_s=bs_p_q,
                cfs_s=summary_s if _cfs_qtr_available else pd.DataFrame(),
                filename="LevelP_Financial_Statements_Quarterly.xlsx",
                link_label="⬇ Download Level P Financial Statements (Excel)",
            ),
            unsafe_allow_html=True,
        )
        _level_p_sanity_checks(pl_p_q, pl_s)

    elif selected_level in ("Level C - Raw Consolidation", "Level C2 - Consolidated Detail"):
        st.info("Consolidated quarterly view uses Level S. Select 'Level S - Customised Detail'.")
        _expanders(pl_s, bs_s, cfs_s, summary_s)

    if not _cfs_qtr_available:
        st.info("Cash flow statement not available — requires ≥ 2 common quarterly periods.")


# ─────────────────────────────────────────────────────────────────────────────
# Daily view
# ─────────────────────────────────────────────────────────────────────────────

def _render_daily_view(income_label_df, balance_label_df, end_year, end_month, global_zid):
    """
    Daily financial statements — Level S only, single ZID.
    3-month consecutive window ending at end_month.
    IS: per-day GL movements (non-cumulative).
    BS: cumulative daily balance (opening + daily movements).
    """
    import calendar as _cal

    # ── ZID selector ─────────────────────────────────────────────────────────
    data = common.load_json('data/businesses.json')
    businesses_d = data.get('businesses', {})
    biz_opts = [(str(z), det.get('zorg', str(z))) for z, det in businesses_d.items()]
    if not biz_opts:
        st.warning("No businesses configured.")
        return

    _zid_labels  = [f"{name} ({z})" for z, name in biz_opts]
    _zid_keys    = [z for z, _ in biz_opts]
    _default_idx = next((i for i, z in enumerate(_zid_keys) if z == str(global_zid)), 0)
    _sel_label   = st.selectbox("View Statements For", _zid_labels, index=_default_idx,
                                 key="daily_zid_sel")
    _sel_zid     = _zid_keys[_zid_labels.index(_sel_label)]

    # ── Load daily data (3-month window) ─────────────────────────────────────
    _start_m = end_month - 2
    _start_y = end_year
    while _start_m <= 0:
        _start_m += 12
        _start_y -= 1
    _title_range = (
        f"{_cal.month_name[_start_m][:3]} {_start_y} – "
        f"{_cal.month_name[end_month][:3]} {end_year}"
    )
    st.title(f"Financial Statements — Daily · {_title_range}")

    with st.spinner(f"Loading daily data for {_title_range}…"):
        try:
            pl_daily, bs_daily, months, prior_is_df = financial.process_data_daily(
                _sel_zid, end_year, end_month, income_label_df, balance_label_df
            )
        except Exception as _e:
            st.error(f"Failed to load daily data: {_e}")
            return

    if pl_daily is None or pl_daily.empty:
        st.info("No GL data found for this period.")
        return

    _hier_d = ['ac_type', 'ac_lv1', 'ac_lv2', 'ac_lv3', 'ac_lv4', 'ac_lv5']
    bs_lv0_d = bs_daily.drop(columns=_hier_d, errors='ignore')

    # ── Level S IS ───────────────────────────────────────────────────────────
    pl_s_d   = financial.build_pl_level_s(pl_daily, selected_perspective='Monthly', vat_rows=None)
    net_income_s_d     = _extract_row(pl_s_d, "Net Income")
    net_income_s_ytd_d = _monthly_to_ytd(net_income_s_d)

    # ── Prior-period NP offset (Jan 1 → month before window start) ───────────
    # The daily BS uses opening BS GL accounts which exclude current-year IS.
    # Without this offset, total_equity is missing Jan→prior_month NP and the
    # balance check shows a constant non-zero error equal to that amount.
    _open_col = (months[0][0], months[0][1], 0)
    prior_np_scalar = 0.0
    if prior_is_df is not None and not prior_is_df.empty:
        _prior_pl_s = financial.build_pl_level_s(
            prior_is_df, selected_perspective='Monthly', vat_rows=None
        )
        _prior_ni = _extract_row(_prior_pl_s, "Net Income")
        if _open_col in _prior_ni.index:
            prior_np_scalar = float(_prior_ni[_open_col])
    # Shift all daily cumulative NP values by the prior-period offset, then
    # prepend the opening column. Use pd.concat with an explicit object-dtype
    # Index to avoid "Too many indexers" — _monthly_to_ytd returns a Series
    # with a 3-level MultiIndex when columns are 3-tuples, and __setitem__
    # with a tuple key fails on MultiIndex Series.
    net_income_s_ytd_d = net_income_s_ytd_d + prior_np_scalar
    _ni_idx = pd.Index([_open_col] + list(net_income_s_ytd_d.index), dtype=object)
    _ni_vals = [prior_np_scalar] + list(net_income_s_ytd_d.values)
    net_income_s_ytd_d = pd.Series(_ni_vals, index=_ni_idx)

    # ── Level S BS ───────────────────────────────────────────────────────────
    bs_s_d = financial.build_bs_level_s(bs_daily, net_income_s_ytd_d, zid=_sel_zid)

    # ── Level S CFS ──────────────────────────────────────────────────────────
    coc_d = financial.cash_open_close(bs_lv0_d)
    _cfs_d_available = True
    try:
        cfs_s_d, summary_s_d = financial.build_cfs_level_s(
            pl_daily, bs_daily, coc_d, net_income_s_d, zid=_sel_zid
        )
    except Exception:
        _cfs_d_available = False

    # ── Format helper ─────────────────────────────────────────────────────────
    def _fmt_daily(df: pd.DataFrame):
        num = df.select_dtypes("number").columns
        rename_map = {}
        for c in num:
            if isinstance(c, tuple) and len(c) == 3:
                yr, mo, dy = c
                rename_map[c] = "Opening" if dy == 0 else f"{_cal.month_abbr[mo]} {dy:02d}"
        df2 = df.rename(columns=rename_map)
        new_num = [rename_map.get(c, str(c)) for c in num]
        return df2.style.format({col: "{:,.1f}" for col in new_num}, na_rep="")

    # ── Data range banner ─────────────────────────────────────────────────────
    _day_cols = sorted(
        [c for c in pl_s_d.select_dtypes("number").columns
         if isinstance(c, tuple) and len(c) == 3 and c[2] > 0],
        key=financial._period_key,
    )
    if _day_cols:
        _first_d, _last_d = _day_cols[0], _day_cols[-1]
        st.caption(
            f"📅 {_cal.month_abbr[_first_d[1]]} {_first_d[2]:02d}, {_first_d[0]} – "
            f"{_cal.month_abbr[_last_d[1]]} {_last_d[2]:02d}, {_last_d[0]} "
            f"({len(_day_cols)} days)"
        )

    # ── Main statements ───────────────────────────────────────────────────────
    with st.expander("Income Statement (per-day movements)", expanded=True):
        st.dataframe(_fmt_daily(pl_s_d), use_container_width=True)

    with st.expander("Balance Sheet (daily running balance)", expanded=True):
        st.dataframe(_fmt_daily(bs_s_d), use_container_width=True)

    if _cfs_d_available:
        with st.expander("Cash Flow Statement (day-on-day changes)", expanded=True):
            st.dataframe(_fmt_daily(cfs_s_d), use_container_width=True)
        with st.expander("Cash Flow Summary", expanded=False):
            st.dataframe(_fmt_daily(summary_s_d), use_container_width=True)
    else:
        st.info("Cash flow statement requires ≥ 2 days of data.")

    # ── Balance / integrity checks ────────────────────────────────────────────
    with st.expander("🔍 Balance & Integrity Checks", expanded=False):
        if _day_cols:
            # 1. BS balance check for each day (Assets + Liabilities should ≈ 0)
            _bs_bal_row = _extract_row(bs_s_d, "Balance Check")
            if not _bs_bal_row.empty and _bs_bal_row.abs().max() < 1e6:
                _bs_chk = pd.DataFrame(
                    [_bs_bal_row.rename(index={c: ("Opening" if c[2]==0 else f"{_cal.month_abbr[c[1]]} {c[2]:02d}") for c in _bs_bal_row.index if isinstance(c, tuple) and len(c)==3})],
                    index=["Balance Check (≈ 0)"],
                )
                st.markdown("**BS Balance Check** (should be ~0 each day)")
                st.dataframe(_fmt(
                    _bs_chk.rename(
                        columns={c: ("Opening" if c[2]==0 else f"{_cal.month_abbr[c[1]]} {c[2]:02d}")
                                 for c in _bs_bal_row.index if isinstance(c, tuple) and len(c)==3}
                    )
                ), use_container_width=True)

            # 2. Month-end IS total vs prior monthly view
            _last_col = _day_cols[-1]
            _last_yr, _last_mo = _last_col[0], _last_col[1]
            _ni_last_day = float(net_income_s_d.get(_last_col, 0))
            st.markdown(
                f"**IS Last-Day Net Income** ({_cal.month_name[_last_mo]} {_last_yr}): "
                f"`{_ni_last_day:,.1f}` — should equal Monthly Level S Net Income for that month."
            )

            # 3. Simple per-month IS total check
            _month_ni: dict = {}
            for c in _day_cols:
                mk = (c[0], c[1])
                _month_ni[mk] = _month_ni.get(mk, 0.0) + float(net_income_s_d.get(c, 0))
            _chk_rows = [{"Month": f"{_cal.month_name[m]} {y}", "Daily IS Net Income Sum": v}
                         for (y, m), v in sorted(_month_ni.items())]
            st.markdown("**Per-Month IS Sum** (each row = sum of daily NI for that month)")
            st.dataframe(pd.DataFrame(_chk_rows), use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    _dl_d = common.create_combined_ls_download_link(
        pl_s=pl_s_d, bs_s=bs_s_d,
        cfs_s=cfs_s_d if _cfs_d_available else pd.DataFrame(),
        filename=f"Daily_Financial_Statements_{_cal.month_name[end_month]}_{end_year}.xlsx",
    )
    st.markdown(_dl_d, unsafe_allow_html=True)


@timed
def display_financial_statements(current_page, zid):
    st.sidebar.title("Financial Statements")
    selected_perspective = st.sidebar.selectbox(
        "Timeframe",
        [
            'Yearly - Custom Range',
            'Yearly - Full Year vs YTD',
            'Monthly',
            'Quarterly',
            'Daily',
            'Lifetime',
            '⚙️ Config Editor',
        ],
        index=0,
    )

    # ── Config Editor: early return, no data loading needed ──────────────────
    if selected_perspective == '⚙️ Config Editor':
        _render_config_editor()
        return

    main_data_dict_pl = {}
    main_data_dict_bs = {}
    current_year = datetime.now().year
    month_list   = list(range(1, 13))

    if selected_perspective == 'Lifetime':
        # Lifetime: all years from user-chosen earliest up to current year.
        # Prior years use full-year data; current year uses a selectable end month.
        _lt_earliest_opts = list(range(2010, current_year))
        lifetime_earliest = st.sidebar.selectbox(
            "Earliest Year",
            _lt_earliest_opts,
            index=0,                      # default = 2010 (full history)
        )
        ytd_month   = st.sidebar.selectbox(
            "Current Year Up to Month", month_list, index=len(month_list) - 1
        )
        year_list   = list(range(lifetime_earliest, current_year + 1))
        start_month = 1
        end_month   = 12
        selected_year = current_year

    elif selected_perspective == 'Yearly - Full Year vs YTD':
        options      = [current_year - i for i in range(10)]
        selected_year = st.sidebar.selectbox("Select End Year", options, index=0)
        year_list    = [selected_year - i for i in range(4, -1, -1)]
        ytd_month    = st.sidebar.selectbox(
            "Up to Month (Current Year)", month_list, index=len(month_list) - 1
        )
        start_month, end_month = 1, 12

    elif selected_perspective == 'Quarterly':
        import calendar as _cal
        options       = [current_year - i for i in range(5)]
        selected_year = st.sidebar.selectbox("Select Year", options, index=0)
        year_list     = [selected_year]
        end_month     = st.sidebar.selectbox(
            "Up to Month", month_list,
            index=datetime.now().month - 1,
            key="qtr_end_month",
        )
        start_month = 1
        ytd_month   = end_month

    elif selected_perspective == 'Daily':
        import calendar as _cal
        _now_d = datetime.now()
        _daily_year_opts = [_now_d.year - i for i in range(3)]
        _daily_year = st.sidebar.selectbox(
            "End Year", _daily_year_opts, index=0, key="daily_year_sel"
        )
        _max_month = _now_d.month - 1 if _daily_year == _now_d.year else 12
        if _max_month < 1:
            _max_month = 12
            _daily_year = _daily_year_opts[1] if len(_daily_year_opts) > 1 else _now_d.year - 1
        _month_opts = list(range(1, _max_month + 1))
        _daily_month = st.sidebar.selectbox(
            "End Month", _month_opts, index=len(_month_opts) - 1, key="daily_month_sel"
        )
        # Dummy values to satisfy existing validation
        selected_year = _daily_year
        year_list     = [_daily_year]
        start_month   = _daily_month
        end_month     = _daily_month
        ytd_month     = _daily_month

    else:
        # Yearly - Custom Range  or  Monthly
        options      = [current_year - i for i in range(10)]
        selected_year = st.sidebar.selectbox("Select End Year", options, index=0)
        year_list    = [selected_year - i for i in range(4, -1, -1)]
        start_month  = st.sidebar.selectbox("Select Start Month", month_list)
        end_month    = st.sidebar.selectbox(
            "Select End Month", month_list, index=len(month_list) - 1
        )
        ytd_month    = end_month          # unused in non-YTD modes

    if not (1 <= start_month <= 12 and 1 <= end_month <= 12):
        st.warning("Month must be between 1 and 12.")
        st.stop()

    data = common.load_json('data/businesses.json')
    businesses = data.get('businesses', {})

    income_label = common.load_json('data/labels.json')['income_statement_label']
    income_label_df = pd.DataFrame(list(income_label.items()), columns=['ac_lv4', 'Income Statement'])

    balance_label = common.load_json('data/labels.json')['balance_sheet_label']
    balance_label_df = pd.DataFrame(list(balance_label.items()), columns=['ac_lv4', 'Balance Sheet'])

    level_options = [
        "Level 0 - Most Detail",
        "Level 1 - Moderate Detail",
        "Level 2 - Least Detail",
        "Level S - Customised Detail",
        "Level P - Projection View",
    ]

    # ── Quarterly view ────────────────────────────────────────────────────────
    if selected_perspective == 'Quarterly':
        _render_quarterly_view(
            businesses, income_label_df, balance_label_df, level_options,
            selected_year, end_month,
            st.session_state.get("zid", None),
        )
        return

    # ── Daily view ────────────────────────────────────────────────────────────
    if selected_perspective == 'Daily':
        _render_daily_view(
            income_label_df, balance_label_df,
            _daily_year, _daily_month,   # end_year, end_month
            st.session_state.get("zid", None),
        )
        return

    _is_lt_persp = selected_perspective in ('Yearly - Full Year vs YTD', 'Lifetime')

    if selected_perspective in ('Yearly - Custom Range', 'Yearly - Full Year vs YTD', 'Lifetime'):
        if selected_perspective == 'Yearly - Custom Range':
            for zid, details in businesses.items():
                for project in details.get('projects', [None]):
                    main_data_dict_pl[(zid, project)] = financial.process_data(zid, year_list, start_month, end_month, 'Income Statement', income_label_df, project, {'Asset', 'Liability'})
            for zid, details in businesses.items():
                for project in details.get('projects', [None]):
                    main_data_dict_bs[(zid, project)] = financial.process_data(zid, year_list, start_month, end_month, 'Balance Sheet', balance_label_df, project, {'Income', 'Expenditure'})

        elif selected_perspective == 'Yearly - Full Year vs YTD':
            prior_years = year_list[:-1]
            curr_year   = [year_list[-1]]
            for zid, details in businesses.items():
                for project in details.get('projects', [None]):
                    pl_prior = financial.process_data(zid, prior_years, 1, 12, 'Income Statement', income_label_df, project, {'Asset', 'Liability'})
                    pl_curr  = financial.process_data(zid, curr_year,   1, ytd_month, 'Income Statement', income_label_df, project, {'Asset', 'Liability'})
                    main_data_dict_pl[(zid, project)] = _merge_ytd(pl_prior, pl_curr)
            for zid, details in businesses.items():
                for project in details.get('projects', [None]):
                    bs_prior = financial.process_data(zid, prior_years, 1, 12, 'Balance Sheet', balance_label_df, project, {'Income', 'Expenditure'})
                    bs_curr  = financial.process_data(zid, curr_year,   1, ytd_month, 'Balance Sheet', balance_label_df, project, {'Income', 'Expenditure'})
                    main_data_dict_bs[(zid, project)] = _merge_ytd(bs_prior, bs_curr)

        else:  # Lifetime — all prior years full-year, current year up to ytd_month
            _lt_prior = year_list[:-1]
            _lt_curr  = [year_list[-1]]    # always current_year
            for zid, details in businesses.items():
                for project in details.get('projects', [None]):
                    pl_prior = financial.process_data(zid, _lt_prior, 1, 12, 'Income Statement', income_label_df, project, {'Asset', 'Liability'})
                    pl_curr  = financial.process_data(zid, _lt_curr,  1, ytd_month, 'Income Statement', income_label_df, project, {'Asset', 'Liability'})
                    main_data_dict_pl[(zid, project)] = _merge_ytd(pl_prior, pl_curr)
            for zid, details in businesses.items():
                for project in details.get('projects', [None]):
                    bs_prior = financial.process_data(zid, _lt_prior, 1, 12, 'Balance Sheet', balance_label_df, project, {'Income', 'Expenditure'})
                    bs_curr  = financial.process_data(zid, _lt_curr,  1, ytd_month, 'Balance Sheet', balance_label_df, project, {'Income', 'Expenditure'})
                    main_data_dict_bs[(zid, project)] = _merge_ytd(bs_prior, bs_curr)

        if selected_perspective == 'Lifetime':
            st.title("Financial Statement Analysis — Lifetime")
        else:
            st.title("Financial Statement Analysis Yearly")
        cols = st.columns(2)
        with cols[0]:
            _CONSOL_KEY = ("consolidated", "All Businesses - Consolidated")
            biz_options = [_CONSOL_KEY] + [k for k in main_data_dict_pl.keys()]
            global_zid = st.session_state.get("zid", None)
            default_idx = next(
                (i for i, k in enumerate(biz_options) if str(k[0]) == str(global_zid)),
                0
            )
            analyse_zid = st.selectbox("View Statements For", biz_options, index=default_idx)
        _is_consolidated = str(analyse_zid[0]) == "consolidated"
        with cols[1]:
            # Level T variants are only available in Full Year vs YTD and Lifetime,
            # and not available for the consolidated view.
            if _is_consolidated:
                _level_opts = [
                    "Level C - Raw Consolidation",
                    "Level C2 - Consolidated Detail",
                    "Level 1 - Moderate Detail",
                    "Level 2 - Least Detail",
                    "Level S - Customised Detail",
                    "Level P - Projection View",
                ]
            else:
                _level_opts = level_options + (
                    ["Level T - Trading Adjustments"]
                    if str(analyse_zid[0]) == '100001' and _is_lt_persp else []
                ) + (
                    ["Level T - GI Adjustments"]
                    if str(analyse_zid[0]) == '100000' and _is_lt_persp else []
                )
            selected_level = st.selectbox("Select Level", _level_opts)

        # ── Raw data: consolidated or single-business ─────────────────────────
        _all_pl_map: dict = {}
        _all_bs_map: dict = {}
        _hier_cols = ['ac_type','ac_lv1','ac_lv2','ac_lv3','ac_lv4','ac_lv5']
        level_c_is = level_c_bs = c2_is = c2_bs = None
        if _is_consolidated:
            _all_pl_map = {str(k[0]): v for k, v in main_data_dict_pl.items()}
            _all_bs_map = {str(k[0]): v for k, v in main_data_dict_bs.items()}
            # Strip hierarchy columns; key by int ZID for the consolidation engine
            _zid_frames_pl = {
                int(k[0]): v.drop(columns=_hier_cols, errors='ignore')
                for k, v in main_data_dict_pl.items()
            }
            _zid_frames_bs = {
                int(k[0]): v.drop(columns=_hier_cols, errors='ignore')
                for k, v in main_data_dict_bs.items()
            }
            level_c_is = _consol.build_level_c(_zid_frames_pl, kind="is")
            level_c_bs = _consol.build_level_c(_zid_frames_bs, kind="bs")
            c2_is = _consol.build_level_c2_is(level_c_is)
            c2_bs = _consol.build_level_c2_bs(level_c_bs)
            # C2 serves as the "Level 0" raw frames for all downstream levels
            pl_raw = c2_is
            bs_raw = c2_bs
        else:
            pl_raw = main_data_dict_pl[analyse_zid]
            bs_raw = main_data_dict_bs[analyse_zid]

        drop_cols = _hier_cols
        pl_lv0 = pl_raw.drop(columns=drop_cols, errors='ignore')
        bs_lv0 = bs_raw.drop(columns=drop_cols, errors='ignore')

        pl_sorted, net_profit, net_profit_m, dep_row = financial.sort_pl_level0(pl_lv0, selected_perspective='Yearly')
        bs_lv0 = financial.append_net_profit_to_bs_level0(bs_lv0, net_profit)
        coc_lv0 = financial.cash_open_close(bs_lv0)
        cfs_df, summary_df = financial.make_cashflow_statement_level0(pl_lv0, bs_lv0, coc_lv0, selected_perspective='Yearly')

        pl_lv1, pl_lv2 = financial.level_builder(pl_sorted, "IS")
        bs_lv1, bs_lv2 = financial.level_builder(bs_lv0, "BS")

        pl_lv1, bs_lv1, net_profitlv1, dep_rowlv1 = financial.add_np_and_balance_lv1(pl_lv1, bs_lv1, selected_perspective='Yearly')
        cfs_lv1 = financial.consolidate_cfs(cfs_df, level=1, debug=True)
        cfs_lv1, summary_df1 = financial.build_cfs_level1_summary_df(cfs_lv1, net_profitlv1, dep_rowlv1, coc_lv0)

        pl_lv2, bs_lv2, net_profitlv2 = financial.add_np_and_balance_lv2(pl_lv2, bs_lv2, selected_perspective='Yearly')
        cfs_lv2 = financial.consolidate_cfs(cfs_df, level=2, debug=True)
        cfs_lv2, summary_df2 = financial.build_cfs_level2_summary(cfs_lv2, net_profitlv2, dep_rowlv1,coc_lv0)

        if selected_level == "Level C - Raw Consolidation":
            # Level C: simple concat of all ZID frames — ZID column retained.
            # Cast zid to str so the numeric formatter ignores it.
            _lc_is_disp = level_c_is.assign(zid=level_c_is["zid"].astype(str))
            _lc_bs_disp = level_c_bs.assign(zid=level_c_bs["zid"].astype(str))

            # ── IS: append Net Profit/Loss row (sum of all numeric cols) ──────
            _lc_per_cols = [c for c in _lc_is_disp.columns
                            if c not in {"zid", "ac_code", "ac_name"}
                            and pd.api.types.is_numeric_dtype(_lc_is_disp[c])]
            _lc_np_series = _lc_is_disp[_lc_per_cols].sum()
            _lc_np_row_is = {"zid": "", "ac_code": "", "ac_name": "Net Profit/Loss"}
            _lc_np_row_is.update(_lc_np_series.to_dict())
            _lc_is_disp = pd.concat(
                [_lc_is_disp, pd.DataFrame([_lc_np_row_is])], ignore_index=True
            )

            # ── BS: append Net Profit/Loss row then Balance Check ─────────────
            _lc_bs_per_cols = [c for c in _lc_bs_disp.columns
                               if c not in {"zid", "ac_code", "ac_name"}
                               and pd.api.types.is_numeric_dtype(_lc_bs_disp[c])]
            # align NP series to BS period columns (some periods may not overlap)
            _lc_np_for_bs = _lc_np_series.reindex(_lc_bs_per_cols, fill_value=0)
            _lc_np_row_bs = {"zid": "", "ac_code": "", "ac_name": "Net Profit/Loss"}
            _lc_np_row_bs.update(_lc_np_for_bs.to_dict())
            _lc_bs_plus_np = pd.concat(
                [_lc_bs_disp, pd.DataFrame([_lc_np_row_bs])], ignore_index=True
            )
            _lc_bal_series = _lc_bs_plus_np[_lc_bs_per_cols].sum()
            _lc_bal_row = {"zid": "", "ac_code": "", "ac_name": "Balance Check"}
            _lc_bal_row.update(_lc_bal_series.to_dict())
            _lc_bs_disp = pd.concat(
                [_lc_bs_plus_np, pd.DataFrame([_lc_bal_row])], ignore_index=True
            )

            _lc_cfs_ok = True
            try:
                _lc_cfs, _lc_summary = _consol.build_level_c_cfs(
                    level_c_bs, level_c_is, 'Yearly'
                )
            except Exception:
                _lc_cfs_ok = False
                _lc_cfs = _lc_summary = pd.DataFrame()
            with st.expander("Income Statement (Level C)", expanded=True):
                st.dataframe(_fmt(_lc_is_disp), use_container_width=True)
            with st.expander("Balance Sheet (Level C)", expanded=True):
                st.dataframe(_fmt(_lc_bs_disp), use_container_width=True)
            if _lc_cfs_ok:
                with st.expander("Cash Flow Statement (Level C)", expanded=True):
                    st.dataframe(_fmt(_lc_cfs), use_container_width=True)
                with st.expander("Cash Flow Summary", expanded=False):
                    st.dataframe(_fmt(_lc_summary), use_container_width=True)
            st.markdown(
                common.create_combined_ls_download_link(
                    pl_s=_lc_is_disp, bs_s=_lc_bs_disp,
                    cfs_s=_lc_cfs if _lc_cfs_ok else pd.DataFrame(),
                    filename="LevelC_Consolidated_Financial_Statements.xlsx",
                    link_label="⬇ Download Level C Financial Statements (Excel)",
                ),
                unsafe_allow_html=True,
            )

        elif selected_level in ("Level 0 - Most Detail", "Level C2 - Consolidated Detail"):
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_sorted), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_lv0), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_df), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_df), use_container_width=True)
            st.markdown(
                common.create_combined_ls_download_link(
                    pl_s=pl_sorted, bs_s=bs_lv0, cfs_s=cfs_df,
                    filename="LevelC2_Consolidated_Financial_Statements.xlsx"
                    if _is_consolidated else f"Level0_{analyse_zid[0]}_Financial_Statements.xlsx",
                    link_label="⬇ Download Level C2 Financial Statements (Excel)"
                    if _is_consolidated else "⬇ Download Level 0 Financial Statements (Excel)",
                ),
                unsafe_allow_html=True,
            )

        elif selected_level == "Level 1 - Moderate Detail":
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_lv1), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_lv1), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_lv1), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_df1), use_container_width=True)
            st.markdown(
                common.create_combined_ls_download_link(
                    pl_s=pl_lv1, bs_s=bs_lv1, cfs_s=cfs_lv1,
                    filename=f"Level1_{analyse_zid[0]}_Financial_Statements.xlsx",
                    link_label="⬇ Download Level 1 Financial Statements (Excel)",
                ),
                unsafe_allow_html=True,
            )

        elif selected_level == "Level 2 - Least Detail":
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_lv2), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_lv2), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_lv2), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_df2), use_container_width=True)
            st.markdown(
                common.create_combined_ls_download_link(
                    pl_s=pl_lv2, bs_s=bs_lv2, cfs_s=cfs_lv2,
                    filename=f"Level2_{analyse_zid[0]}_Financial_Statements.xlsx",
                    link_label="⬇ Download Level 2 Financial Statements (Excel)",
                ),
                unsafe_allow_html=True,
            )

        elif selected_level == "Level S - Customised Detail":
            # Level S requires raw data (ac_code still present)
            _az, _ap = analyse_zid
            _num_cols = financial._ls_num_cols(pl_raw)

            if _is_consolidated:
                # Consolidated: fetch VAT GL for every ZID and sum them.
                # The 4 VAT rows are informational references — no interaction with
                # the rest of the statement, same as for individual entities.
                _vat_gl_parts = []
                _all_zids_vat = _consol.load_consolidation_rules().get("all_zids", [])
                for _vat_zid in _all_zids_vat:
                    try:
                        _sql_vat_z, _params_vat_z = queries.get_vat_breakdown_gl(
                            zid=_vat_zid, year_list=year_list, smonth=1, emonth=12,
                        )
                        _gl_vat_z = get_dataframe(_sql_vat_z, _params_vat_z)
                        if _gl_vat_z is not None and not _gl_vat_z.empty:
                            _vat_gl_parts.append(_gl_vat_z)
                    except Exception:
                        pass
                _gl_vat_all = (pd.concat(_vat_gl_parts, ignore_index=True)
                               if _vat_gl_parts else pd.DataFrame())
                _vat_rows = financial.compute_vat_is_rows(
                    _gl_vat_all, _num_cols, selected_perspective='Yearly'
                )
            else:
                # For Full Year vs YTD: fetch all months (1-12) so prior years are complete;
                # current year data already bounded by ytd_month via process_data.
                # For Custom Range: use the selected start/end months.
                _vat_smonth = 1 if selected_perspective == 'Yearly - Full Year vs YTD' else start_month
                _vat_emonth = 12 if selected_perspective == 'Yearly - Full Year vs YTD' else end_month
                _sql_vat, _params_vat = queries.get_vat_breakdown_gl(
                    zid=_az, project=_ap, year_list=year_list,
                    smonth=_vat_smonth, emonth=_vat_emonth,
                )
                _gl_vat   = get_dataframe(_sql_vat, _params_vat)
                _vat_rows = financial.compute_vat_is_rows(
                    _gl_vat, _num_cols, selected_perspective='Yearly'
                )

            pl_s  = financial.build_pl_level_s(
                pl_raw, selected_perspective='Yearly', vat_rows=_vat_rows
            )
            net_income_s = _extract_row(pl_s, "Net Income")
            bs_s  = financial.build_bs_level_s(bs_raw, net_income_s, zid=_az)
            cfs_s, summary_s = financial.build_cfs_level_s(
                pl_raw, bs_raw, coc_lv0, net_income_s, zid=_az
            )

            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_s), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_s), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_s), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_s), use_container_width=True)

            # Compute ratios once — reused by dashboard and expander
            _partial_months = ytd_month if _is_lt_persp else 12
            try:
                _ratio_df = _build_ls_ratios(
                    pl_s, bs_s, cfs_s,
                    perspective="Yearly",
                    partial_year_months=_partial_months,
                )
            except Exception as _re:
                _ratio_df = None

            _num_year_cols = [c for c in pl_s.columns if isinstance(c, int)]
            _available_years = sorted(_num_year_cols)
            _entity_label = (
                "Consolidated Group"
                if _is_consolidated
                else f"ZID {_az}"
            )

            # ── Financial Ratios ──────────────────────────────────────────────
            with st.expander("📊 Financial Ratios", expanded=False):
                if _ratio_df is not None:
                    st.dataframe(_ratio_df.set_index("Ratio"), use_container_width=True)
                else:
                    st.warning("Could not compute ratios.")

            _dl_link = common.create_combined_ls_download_link(
                pl_s=pl_s, bs_s=bs_s, cfs_s=cfs_s,
                filename="LevelS_Consolidated_Financial_Statements.xlsx"
                if _is_consolidated else "LevelS_Financial_Statements.xlsx",
            )
            st.markdown(_dl_link, unsafe_allow_html=True)

            _sanity_checks(
                pl_sorted, pl_lv1, pl_lv2, pl_s,
                bs_lv0, bs_lv1, bs_lv2, bs_s,
                summary_df, summary_df1, summary_df2, summary_s,
            )

            # ── Notes / ZID Breakdown / Analysis Dashboard panel ─────────────
            if _is_consolidated:
                _panel_y = st.radio(
                    "View",
                    ["📋 Notes & Context", "📊 ZID Contribution Breakdown", "📈 Analysis Dashboard"],
                    horizontal=True,
                    key="ls_panel_y",
                )
                if _panel_y == "📋 Notes & Context":
                    _render_ls_notes(key_suffix="y")
                elif _panel_y == "📊 ZID Contribution Breakdown":
                    _render_zid_contribution_breakdown(
                        pl_s, bs_s,
                        _zid_frames_pl, _zid_frames_bs,
                        perspective="Yearly",
                        key_suffix="y",
                    )
                else:
                    render_analysis_dashboard(
                        pl_s, bs_s, cfs_s, _ratio_df,
                        _entity_label, _available_years,
                        entity_zid="consolidated",
                        partial_year_months=_partial_months,
                    )
            else:
                _panel_y_single = st.radio(
                    "View",
                    ["📋 Notes & Context", "📈 Analysis Dashboard"],
                    horizontal=True,
                    key="ls_panel_y_single",
                )
                if _panel_y_single == "📋 Notes & Context":
                    _render_ls_notes(key_suffix="y_single")
                else:
                    render_analysis_dashboard(
                        pl_s, bs_s, cfs_s, _ratio_df,
                        _entity_label, _available_years,
                        entity_zid=str(_az),
                        partial_year_months=_partial_months,
                    )

        elif selected_level == "Level P - Projection View":
            st.info("ℹ️ Level P — Projection View (condensed from Level S).")
            _az, _ap = analyse_zid
            _num_cols_p = financial._ls_num_cols(pl_raw)
            _years_in_data_p = sorted(set(
                int(financial._period_key(c)[0]) for c in _num_cols_p
            ))
            if _is_consolidated:
                _vat_gl_parts_p = []
                for _vat_zid in _consol.load_consolidation_rules().get("all_zids", []):
                    try:
                        _sql_vat_z, _params_vat_z = queries.get_vat_breakdown_gl(
                            zid=_vat_zid, year_list=_years_in_data_p, smonth=1, emonth=12,
                        )
                        _gl_vat_z = get_dataframe(_sql_vat_z, _params_vat_z)
                        if _gl_vat_z is not None and not _gl_vat_z.empty:
                            _vat_gl_parts_p.append(_gl_vat_z)
                    except Exception:
                        pass
                _gl_vat_all_p = (
                    pd.concat(_vat_gl_parts_p, ignore_index=True)
                    if _vat_gl_parts_p else pd.DataFrame()
                )
                _vat_rows_p = financial.compute_vat_is_rows(
                    _gl_vat_all_p, _num_cols_p, selected_perspective='Yearly'
                )
            else:
                _vat_smonth_p = 1  if _is_lt_persp else start_month
                _vat_emonth_p = 12 if _is_lt_persp else end_month
                _sql_vat_p, _params_vat_p = queries.get_vat_breakdown_gl(
                    zid=_az, project=_ap, year_list=_years_in_data_p,
                    smonth=_vat_smonth_p, emonth=_vat_emonth_p,
                )
                _gl_vat_p = get_dataframe(_sql_vat_p, _params_vat_p)
                _vat_rows_p = financial.compute_vat_is_rows(
                    _gl_vat_p, _num_cols_p, selected_perspective='Yearly'
                )
            pl_s_p = financial.build_pl_level_s(
                pl_raw, selected_perspective='Yearly', vat_rows=_vat_rows_p
            )
            net_income_s_p = _extract_row(pl_s_p, "Net Income")
            bs_s_p = financial.build_bs_level_s(bs_raw, net_income_s_p, zid=_az)
            _, summary_s_p = financial.build_cfs_level_s(
                pl_raw, bs_raw, coc_lv0, net_income_s_p, zid=_az
            )
            pl_p_y, bs_p_y = financial.build_condensed_view(pl_s_p, bs_s_p)

            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_p_y), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_p_y), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=True):
                st.dataframe(_fmt(summary_s_p), use_container_width=True)
            st.markdown(
                common.create_combined_ls_download_link(
                    pl_s=pl_p_y, bs_s=bs_p_y, cfs_s=summary_s_p,
                    filename="LevelP_Financial_Statements_Yearly.xlsx",
                    link_label="⬇ Download Level P Financial Statements (Excel)",
                ),
                unsafe_allow_html=True,
            )
            _level_p_sanity_checks(pl_p_y, pl_s_p)

            # ── FY Projection (only for Full Year vs YTD) ────────────────────
            if selected_perspective == 'Yearly - Full Year vs YTD' and ytd_month > 0:
                _ytd_int_col = year_list[-1]
                _num_p_y     = financial._ls_num_cols(pl_p_y)
                if _ytd_int_col in _num_p_y:
                    st.markdown("---")
                    st.subheader(f"📈 Full-Year Projection ({_ytd_int_col})")

                    _proj_mode = st.radio(
                        "Projection method",
                        ["📅 Straight-line (month average)", "✏️ Manual % adjustment"],
                        horizontal=True,
                        key="lp_proj_mode_y",
                    )

                    # Annualised baseline for every IS row
                    def _annualise(row_label):
                        r = pl_p_y.loc[pl_p_y["ac_name"].astype(str) == row_label, _ytd_int_col]
                        return float(r.iloc[0]) if not r.empty else 0.0

                    # pl_p_y already has GP(P)=Adj Rev+COGS and adjusted SGA/SD
                    # (Others Revenue split equally into SGA and SD via build_condensed_view).
                    _ann_rev        = _annualise("Adjusted Revenue (Pending)")    * 12 / ytd_month
                    _ann_cogs       = _annualise("COGS")                          * 12 / ytd_month
                    _ann_gp_p       = _annualise("Gross Profit")                  * 12 / ytd_month
                    _ann_sga        = _annualise("Total SG&A")                    * 12 / ytd_month
                    _ann_sd         = _annualise("Total Sales & Distribution")    * 12 / ytd_month
                    _ann_others_dir = _annualise("0501-Others Direct Expenses")   * 12 / ytd_month
                    _ann_int        = _annualise("Total Interest & Charges")      * 12 / ytd_month
                    _ann_vat        = _annualise("0629-VAT & Tax Total (A+B+C)")  * 12 / ytd_month
                    _ann_ebit       = _ann_gp_p + _ann_sga + _ann_sd + _ann_others_dir
                    _ann_ni         = _ann_ebit + _ann_int + _ann_vat

                    if _proj_mode == "📅 Straight-line (month average)":
                        _p_rev        = _ann_rev
                        _p_cogs       = _ann_cogs
                        _p_gp         = _ann_gp_p
                        _p_sga        = _ann_sga
                        _p_sd         = _ann_sd
                        _p_others_dir = _ann_others_dir
                        _p_ebit       = _ann_ebit
                        _p_int        = _ann_int
                        _p_vat        = _ann_vat
                        _p_ni         = _ann_ni
                        st.caption(
                            f"Annualising {ytd_month} months of actuals × 12/{ytd_month}. "
                            "Switch to Manual % adjustment to override individual lines."
                        )
                    else:
                        st.caption(
                            "Adjust each line relative to the straight-line annualised base. "
                            "COGS is expressed as % of projected revenue; all others as % of annualised. "
                            "SG&A and S&D already reflect the Others Revenue offset."
                        )
                        # Level S sign convention: COGS/SG&A/S&D/Interest/VAT are negative.
                        # GP(P) = Adj Revenue + COGS; GP% = GP(P) / Revenue.
                        _gm_default   = round(_ann_gp_p / _ann_rev * 100, 1) if _ann_rev > 0 else 35.0
                        _cogs_default = max(0.0, round(-_ann_cogs / _ann_rev * 100, 1)) if _ann_rev > 0 else 65.0
                        _c1, _c2, _c3 = st.columns(3)
                        with _c1:
                            _pct_rev = st.number_input(
                                "Revenue % of annualised", min_value=0.0, value=100.0,
                                step=5.0, key="lp_pct_rev",
                            )
                            _pct_sga = st.number_input(
                                "Total SG&A % of annualised", min_value=0.0, value=100.0,
                                step=5.0, key="lp_pct_sga",
                            )
                        with _c2:
                            _pct_cogs_rev = st.number_input(
                                "COGS % of projected revenue", min_value=0.0, value=_cogs_default,
                                step=1.0, key="lp_pct_cogs",
                                help=f"Current gross margin ≈ {_gm_default:.1f}% "
                                     f"(COGS ≈ {_cogs_default:.1f}% of revenue). "
                                     "Lower COGS% to widen the margin.",
                            )
                            _pct_sd = st.number_input(
                                "Total S&D % of annualised", min_value=0.0, value=100.0,
                                step=5.0, key="lp_pct_sd",
                            )
                        with _c3:
                            _pct_int = st.number_input(
                                "Interest % of annualised", min_value=0.0, value=100.0,
                                step=5.0, key="lp_pct_int",
                            )
                            _pct_vat = st.number_input(
                                "VAT/Tax % of annualised", min_value=0.0, value=100.0,
                                step=5.0, key="lp_pct_vat",
                            )
                        # Keep Level S sign convention throughout: costs stay negative.
                        _p_rev        = _ann_rev * _pct_rev / 100
                        _p_cogs       = -(_p_rev * _pct_cogs_rev / 100)       # negate → negative
                        _p_gp         = _p_rev + _p_cogs                       # GP(P) = Adj Rev + COGS
                        _p_sga        = _ann_sga * _pct_sga / 100             # already negative, adjusted
                        _p_sd         = _ann_sd  * _pct_sd  / 100             # already negative, adjusted
                        _p_others_dir = _ann_others_dir                       # keep same as annualised
                        _p_ebit       = _p_gp + _p_sga + _p_sd + _p_others_dir
                        _p_int        = _ann_int * _pct_int / 100             # already negative
                        _p_vat        = _ann_vat * _pct_vat / 100             # already negative
                        _p_ni         = _p_ebit + _p_int + _p_vat

                    _proj_col_lbl = f"Proj FY {_ytd_int_col}"
                    _proj_rows = [
                        ("Adjusted Revenue (Pending)",   _p_rev),
                        ("COGS",                          _p_cogs),
                        ("Gross Profit",                  _p_gp),
                        ("Total SG&A",                    _p_sga),
                        ("Total Sales & Distribution",    _p_sd),
                        ("0501-Others Direct Expenses",   _p_others_dir),
                        ("EBITDA",                        _p_ebit),
                        ("Total Interest & Charges",      _p_int),
                        ("0629-VAT & Tax Total (A+B+C)",  _p_vat),
                        ("Net Income",                    _p_ni),
                    ]
                    # Build side-by-side: YTD actual | annualised | projected
                    _ytd_vals = {
                        r: float(pl_p_y.loc[pl_p_y["ac_name"].astype(str) == r, _ytd_int_col].iloc[0])
                        if not pl_p_y.loc[pl_p_y["ac_name"].astype(str) == r].empty else 0.0
                        for r, _ in _proj_rows
                    }
                    _proj_df = pd.DataFrame([
                        {
                            "Row": r,
                            f"YTD {_ytd_int_col} ({ytd_month}m)": _ytd_vals[r],
                            "Annualised (×12/m)": _ytd_vals[r] * 12 / ytd_month,
                            _proj_col_lbl: v,
                        }
                        for r, v in _proj_rows
                    ]).set_index("Row")

                    def _proj_fmt(val):
                        try:
                            f = float(val)
                            return f"{f:,.0f}" if abs(f) >= 1 else f"{f:.2f}"
                        except Exception:
                            return str(val)

                    st.dataframe(
                        _proj_df.style.format(_proj_fmt),
                        use_container_width=True,
                    )

                    # ── Save Projection Snapshot ──────────────────────────────
                    st.markdown("---")
                    st.subheader("💾 Save Projection Snapshot")
                    _snap_note = st.text_area(
                        "Projection note / explanation",
                        placeholder=(
                            "e.g. Conservative estimate based on H1 momentum — "
                            "assuming 10% revenue growth and stable margins..."
                        ),
                        key=f"lp_snap_note_{_ytd_int_col}_{_az}",
                        height=90,
                    )
                    if st.button("💾 Save Snapshot", key=f"lp_snap_save_{_ytd_int_col}_{_az}"):
                        _snap_inputs = {}
                        if _proj_mode == "✏️ Manual % adjustment":
                            _snap_inputs = {
                                "Revenue % of annualised": _pct_rev,
                                "COGS % of projected revenue": _pct_cogs_rev,
                                "Total SG&A % of annualised": _pct_sga,
                                "Total S&D % of annualised": _pct_sd,
                                "Interest % of annualised": _pct_int,
                                "VAT/Tax % of annualised": _pct_vat,
                            }
                        _snap = {
                            "saved_at": datetime.now().isoformat(timespec="seconds"),
                            "label": _proj_col_lbl,
                            "year": int(_ytd_int_col),
                            "ytd_month": int(ytd_month),
                            "zid": str(_az),
                            "mode": _proj_mode,
                            "inputs": _snap_inputs,
                            "projection": {r: round(v, 2) for r, v in _proj_rows},
                            "note": _snap_note.strip(),
                        }
                        _save_lp_projection(_snap)
                        st.success(f"Snapshot saved — {_snap['saved_at']}")

                    # ── Last Saved Snapshot ───────────────────────────────────
                    _all_snaps = _load_lp_projections()
                    if _all_snaps:
                        _latest = _all_snaps[-1]
                        _snap_header = (
                            f"📋 Last Saved Snapshot — {_latest.get('label', '?')}  |  "
                            f"ZID {_latest.get('zid', '?')}  |  "
                            f"saved {_latest.get('saved_at', '?')}"
                        )
                        with st.expander(_snap_header, expanded=False):
                            if _latest.get("note"):
                                st.info(_latest["note"])
                            _meta_rows = [
                                ("Year", _latest.get("year")),
                                ("YTD Months", _latest.get("ytd_month")),
                                ("ZID", _latest.get("zid")),
                                ("Projection Mode", _latest.get("mode")),
                                ("Saved At", _latest.get("saved_at")),
                            ]
                            st.table(
                                pd.DataFrame(_meta_rows, columns=["Field", "Value"]).set_index("Field")
                            )
                            if _latest.get("inputs"):
                                st.markdown("**Adjustment Inputs**")
                                _inp_df = pd.DataFrame(
                                    [{"Input": k, "%": v}
                                     for k, v in _latest["inputs"].items()]
                                ).set_index("Input")
                                st.dataframe(_inp_df, use_container_width=True)
                            if _latest.get("projection"):
                                st.markdown("**Projected Income Statement**")
                                _snap_proj_df = pd.DataFrame(
                                    [{"Row": r, "Projected Value": v}
                                     for r, v in _latest["projection"].items()]
                                ).set_index("Row")
                                st.dataframe(
                                    _snap_proj_df.style.format(
                                        {"Projected Value": lambda v: f"{v:,.0f}"}
                                    ),
                                    use_container_width=True,
                                )
                            if len(_all_snaps) > 1:
                                st.caption(
                                    f"Showing latest of {len(_all_snaps)} saved snapshots. "
                                    "Older snapshots are stored in `data/level_p_projections.json`."
                                )

        elif selected_level == "Level T - Trading Adjustments":
            st.info("📈 Analysis Dashboard is available at Level S only.")
            # ── Step 1: Build Level S base (identical to Level S block) ──────
            _az, _ap = analyse_zid
            _vat_smonth = 1  if _is_lt_persp else start_month
            _vat_emonth = 12 if _is_lt_persp else end_month
            _sql_vat, _params_vat = queries.get_vat_breakdown_gl(
                zid=_az, project=_ap, year_list=year_list,
                smonth=_vat_smonth, emonth=_vat_emonth,
            )
            _gl_vat   = get_dataframe(_sql_vat, _params_vat)
            _num_cols = financial._ls_num_cols(pl_raw)
            _vat_rows = financial.compute_vat_is_rows(
                _gl_vat, _num_cols, selected_perspective='Yearly'
            )
            pl_s = financial.build_pl_level_s(
                pl_raw, selected_perspective='Yearly', vat_rows=_vat_rows
            )
            net_income_s = _extract_row(pl_s, "Net Income")
            bs_s  = financial.build_bs_level_s(bs_raw, net_income_s, zid=_az)
            cfs_s, summary_s = financial.build_cfs_level_s(
                pl_raw, bs_raw, coc_lv0, net_income_s, zid=_az
            )

            # ── Step 2: Build adj_data — I&H net sales & net cost (years <= 2024)
            _t_num_cols    = financial._ls_num_cols(pl_s)
            _ind_hh_series = pd.Series(0.0, index=_t_num_cols)
            _cogs_adj_series = pd.Series(0.0, index=_t_num_cols)
            _t_years = [y for y in year_list if int(y) <= 2024]
            if _t_years:
                _t_smonth = 1  if _is_lt_persp else start_month
                _t_emonth = 12 if _is_lt_persp else end_month

                # Net sales (revenue adjustment)
                _sql_t, _params_t = queries.get_ind_hh_net_sales(
                    zid=_az, year_list=_t_years,
                    smonth=_t_smonth, emonth=_t_emonth,
                )
                _inh_df = get_dataframe(_sql_t, _params_t)
                if _inh_df is not None and not _inh_df.empty:
                    for _col in _t_num_cols:
                        _yr = financial._period_key(_col)[0]
                        if _yr <= 2024:
                            _m = _inh_df.loc[_inh_df['year'] == _yr, 'net_sales']
                            if not _m.empty:
                                _ind_hh_series[_col] = float(_m.iloc[0])

                # Net cost (COGS adjustment)
                _sql_c, _params_c = queries.get_ind_hh_net_cost(
                    zid=_az, year_list=_t_years,
                    smonth=_t_smonth, emonth=_t_emonth,
                )
                _inh_cost_df = get_dataframe(_sql_c, _params_c)
                if _inh_cost_df is not None and not _inh_cost_df.empty:
                    for _col in _t_num_cols:
                        _yr = financial._period_key(_col)[0]
                        if _yr <= 2024:
                            _m = _inh_cost_df.loc[_inh_cost_df['year'] == _yr, 'net_cost']
                            if not _m.empty:
                                _cogs_adj_series[_col] = float(_m.iloc[0])

            _adj_data = {
                'ind_hh_net_sales': _ind_hh_series,
                'cogs_adj':         _cogs_adj_series,
                # Level S revenue (pre-T) needed for the AR scaling ratio in BS adjust
                'level_s_revenue':  _extract_row(pl_s, "Revenue"),
            }

            # ── Step 3: Apply Level T adjustments ─────────────────────────────
            pl_t = financial.adjust_pl_level_t(pl_s, _adj_data)

            # Build BS with updated net income, then apply all BS adjustments
            # (stock, AR, cash/bank, equity balancing).  The adjusted BS is
            # fully balanced (Balance Check = 0), so the CFS built from it
            # will also check to zero.
            net_income_t     = _extract_row(pl_t, "Net Income")
            bs_t             = financial.build_bs_level_s(bs_raw, net_income_t, zid=_az)
            bs_t             = financial.adjust_bs_level_t(bs_t, _adj_data)
            cfs_t, summary_t = financial.build_cfs_from_bs_t(bs_t, net_income_t)

            # ── Step 4: Display ───────────────────────────────────────────────
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_t), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_t), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_t), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_t), use_container_width=True)

            _dl_link_t = common.create_combined_ls_download_link(
                pl_s=pl_t, bs_s=bs_t, cfs_s=cfs_t,
                filename="LevelT_Financial_Statements.xlsx",
            )
            st.markdown(_dl_link_t, unsafe_allow_html=True)

            # ── Adjustment summary panel ───────────────────────────────────────
            _vat_prior_series   = financial.get_level_t_vat_prior(_t_num_cols)
            _stock_inh_series   = financial.get_level_t_stock_inh(_t_num_cols)
            _level_s_rev_series = _adj_data['level_s_revenue'].reindex(_t_num_cols, fill_value=0.0)
            _yr_labels = [str(financial._period_key(c)[0]) for c in _t_num_cols]

            # Compute I&H share ratio (same formula used inside adjust_bs_level_t)
            _safe_rev  = _level_s_rev_series.where(_level_s_rev_series != 0, other=float('nan'))
            _ih_ratio  = (_ind_hh_series / _safe_rev).fillna(0.0).clip(0.0, 1.0)

            # AR removal = AR Total (Level S) * I&H ratio
            _ar_total_s = _extract_row(bs_s, "0103-Accounts Receivable (Total)")
            _ar_total_s = _ar_total_s.reindex(_t_num_cols, fill_value=0.0)
            _ar_removed = _ar_total_s * _ih_ratio

            # Cash & Bank removal = (Cash + Bank) (Level S) * I&H ratio
            _cash_s     = _extract_row(bs_s, "0101-Cash & Cash Equivalent")
            _bank_s     = _extract_row(bs_s, "0102-Bank Balance")
            _cash_s     = _cash_s.reindex(_t_num_cols, fill_value=0.0)
            _bank_s     = _bank_s.reindex(_t_num_cols, fill_value=0.0)
            _cash_removed = (_cash_s + _bank_s) * _ih_ratio

            _summary_data = {
                "I&H Revenue (Net Sales)": [
                    _ind_hh_series.get(c, 0.0) for c in _t_num_cols
                ],
                "I&H Net Cost (COGS Removed)": [
                    _cogs_adj_series.get(c, 0.0) for c in _t_num_cols
                ],
                "8% of I&H Revenue (Expense Reduction)": [
                    0.08 * _ind_hh_series.get(c, 0.0) for c in _t_num_cols
                ],
                "Pre-2022 VAT Rebate Reclassified from COGS": [
                    _vat_prior_series.get(c, 0.0) for c in _t_num_cols
                ],
                "I&H Stock Removed from BS": [
                    _stock_inh_series.get(c, 0.0) for c in _t_num_cols
                ],
                "I&H AR Removed from BS (Estimated)": [
                    float(_ar_removed.get(c, 0.0)) for c in _t_num_cols
                ],
                "I&H Cash & Bank Removed from BS (Estimated)": [
                    float(_cash_removed.get(c, 0.0)) for c in _t_num_cols
                ],
            }
            _summary_df = pd.DataFrame(_summary_data, index=_yr_labels).T
            _summary_df.index.name = "Adjustment"
            with st.expander("Level T — Adjustment Breakdown", expanded=True):
                st.dataframe(_fmt(_summary_df.reset_index()), use_container_width=True)

            # ── Notes ─────────────────────────────────────────────────────────
            st.info(
                "**Revenue Adjustment**\n\n"
                "Prior to 2025, GI Corporation operated as a consolidated entity. "
                "Following the business split, Industrial & Household (I&H) product revenue "
                "has been separated. This adjustment removes I&H net sales from reported "
                "revenue to reflect only the continuing business.\n\n"
                "**COGS Adjustment**\n\n"
                "Consistent with the revenue adjustment, the cost of goods sold attributable "
                "to I&H products (net of returns) has been removed from COGS for the same periods.\n\n"
                "**Expense Adjustment**\n\n"
                "Based on operational experience, approximately 8% of I&H revenue represents "
                "the proportional share of SG&A and Sales & Distribution expenses attributed "
                "to that segment. This amount is redistributed proportionally across all "
                "sub-line items within those two categories. Bank interest and tax lines are "
                "not adjusted.\n\n"
                "**Pre-2022 VAT Note**\n\n"
                "Prior to 2022, VAT Rebate expenses were not recorded as a separate line in "
                "the system — they were embedded directly within COGS. For years 2021 and "
                "earlier, this adjustment reclassifies the estimated VAT Rebate amount out of "
                "COGS and into the VAT Expenses from Rebate (A) line, matching the treatment "
                "used from 2022 onwards. Net Income is unchanged by this reclassification. "
                "Placeholder values are used — update data/level_t_vat_prior.json when "
                "exact figures are available from physical records.\n\n"
                "**Balance Sheet — Stock in Hand**\n\n"
                "The ending I&H inventory balance has been manually calculated for 2016–2023 "
                "and is subtracted from Stock in Hand. These values are stored in "
                "data/level_t_stock_inh.json and should be updated if the calculations change.\n\n"
                "**Balance Sheet — Accounts Receivable**\n\n"
                "Accounts Receivable (all sub-rows and the total) is scaled down by the I&H "
                "share of total revenue (I&H Net Sales ÷ Level S Revenue) for each year, "
                "reflecting the estimated portion of AR attributable to the split entity.\n\n"
                "**Balance Sheet — Cash & Bank**\n\n"
                "Cash & Cash Equivalent and Bank Balance are each scaled by the same non-I&H "
                "revenue ratio used for Accounts Receivable, removing the estimated portion "
                "of the cash float that supported I&H operations.\n\n"
                "**Balance Sheet — Equity Balancing**\n\n"
                "The asset reductions above (Stock, AR, Cash & Bank) reduce Total Assets "
                "without a corresponding liability adjustment. To keep the Balance Sheet "
                "in balance, the full gap is absorbed into the '1302-Error Adjustment For "
                "Retained Earning' row. Total Equity and Total Liabilities & Equity are "
                "recalculated accordingly. Balance Check = 0 for all years.\n\n"
                "**Cash Flow Statement**\n\n"
                "The Level T CFS is rebuilt entirely from the adjusted Balance Sheet. "
                "Year-over-year changes in every BS row (including the adjusted cash, AR, "
                "stock, and the new equity balancing entry) feed directly into the CFS, "
                "so Cash-flow Check = 0 by construction."
            )

        elif selected_level == "Level T - GI Adjustments":
            # ══════════════════════════════════════════════════════════════════
            # Level T — GI Adjustments  (ZID 100000)
            #
            # Step 1  Build GI (100000) Level S base statements.
            # Step 2  Re-compute the 100001 Trading Adjustments breakdown for
            #         the same year range, so analysts can compare side-by-side.
            # Step 3  (TODO) Apply GI-specific adjustments once logic is defined.
            # ══════════════════════════════════════════════════════════════════

            # ── Step 1: Build GI Level S ──────────────────────────────────────
            _gi_az, _gi_ap = analyse_zid
            _sql_gi_vat, _params_gi_vat = queries.get_vat_breakdown_gl(
                zid=_gi_az, project=_gi_ap, year_list=year_list,
                smonth=1, emonth=12,
            )
            _gl_gi_vat   = get_dataframe(_sql_gi_vat, _params_gi_vat)
            _gi_num_cols = financial._ls_num_cols(pl_raw)
            _gi_vat_rows = financial.compute_vat_is_rows(
                _gl_gi_vat, _gi_num_cols, selected_perspective='Yearly'
            )
            pl_gi_s = financial.build_pl_level_s(
                pl_raw, selected_perspective='Yearly', vat_rows=_gi_vat_rows
            )
            net_income_gi_s = _extract_row(pl_gi_s, "Net Income")
            bs_gi_s         = financial.build_bs_level_s(bs_raw, net_income_gi_s, zid=_gi_az)
            cfs_gi_s, summary_gi_s = financial.build_cfs_level_s(
                pl_raw, bs_raw, coc_lv0, net_income_gi_s, zid=_gi_az
            )

            # Level S kept in memory for the breakdown table; not displayed
            # (Level T adjusted statements are shown in Step 3 below)

            # ── Step 2: 100001 Adjustment Breakdown (reference / future input) ─
            # The breakdown figures from 100001 are re-queried for whatever years
            # are currently in view, so the table stays aligned to the GI columns.
            _gi_t_num_cols = financial._ls_num_cols(pl_gi_s)
            _gi_yr_labels  = [str(financial._period_key(c)[0]) for c in _gi_t_num_cols]

            _key_101 = next((k for k in main_data_dict_pl if str(k[0]) == '100001'), None)
            if _key_101 is not None:
                _pl_raw_101 = main_data_dict_pl[_key_101]
                _bs_raw_101 = main_data_dict_bs[_key_101]
                _az_101, _ap_101 = _key_101

                # Build 100001 Level S (needed for revenue and AR/Cash ratios)
                _sql_101_vat, _params_101_vat = queries.get_vat_breakdown_gl(
                    zid=_az_101, project=_ap_101, year_list=year_list,
                    smonth=1, emonth=12,
                )
                _gl_101_vat   = get_dataframe(_sql_101_vat, _params_101_vat)
                _num_cols_101 = financial._ls_num_cols(_pl_raw_101)
                _vat_rows_101 = financial.compute_vat_is_rows(
                    _gl_101_vat, _num_cols_101, selected_perspective='Yearly'
                )
                pl_s_101      = financial.build_pl_level_s(
                    _pl_raw_101, selected_perspective='Yearly', vat_rows=_vat_rows_101
                )
                net_income_101 = _extract_row(pl_s_101, "Net Income")
                bs_s_101       = financial.build_bs_level_s(_bs_raw_101, net_income_101, zid=_az_101)

                # Query I&H sales/cost for 100001 — restricted to years <= 2024
                _gi_t_years   = [y for y in year_list if int(y) <= 2024]
                _ind_hh_101   = pd.Series(0.0, index=_gi_t_num_cols)
                _cogs_adj_101 = pd.Series(0.0, index=_gi_t_num_cols)

                if _gi_t_years:
                    _sql_inh101, _params_inh101 = queries.get_ind_hh_net_sales(
                        zid=_az_101, year_list=_gi_t_years, smonth=1, emonth=12,
                    )
                    _inh101_df = get_dataframe(_sql_inh101, _params_inh101)
                    if _inh101_df is not None and not _inh101_df.empty:
                        for _col in _gi_t_num_cols:
                            _yr = financial._period_key(_col)[0]
                            if _yr <= 2024:
                                _m = _inh101_df.loc[_inh101_df['year'] == _yr, 'net_sales']
                                if not _m.empty:
                                    _ind_hh_101[_col] = float(_m.iloc[0])

                    _sql_cost101, _params_cost101 = queries.get_ind_hh_net_cost(
                        zid=_az_101, year_list=_gi_t_years, smonth=1, emonth=12,
                    )
                    _cost101_df = get_dataframe(_sql_cost101, _params_cost101)
                    if _cost101_df is not None and not _cost101_df.empty:
                        for _col in _gi_t_num_cols:
                            _yr = financial._period_key(_col)[0]
                            if _yr <= 2024:
                                _m = _cost101_df.loc[_cost101_df['year'] == _yr, 'net_cost']
                                if not _m.empty:
                                    _cogs_adj_101[_col] = float(_m.iloc[0])

                # Compute I&H share ratio (using 100001 Level S revenue)
                _rev_101      = _extract_row(pl_s_101, "Revenue").reindex(_gi_t_num_cols, fill_value=0.0)
                _safe_rev_101 = _rev_101.where(_rev_101 != 0, other=float('nan'))
                _ih_ratio_101 = (_ind_hh_101 / _safe_rev_101).fillna(0.0).clip(0.0, 1.0)

                # JSON-sourced adjustments (same data as 100001 Level T)
                _vat_prior_101  = financial.get_level_t_vat_prior(_gi_t_num_cols)
                _stock_inh_101  = financial.get_level_t_stock_inh(_gi_t_num_cols)

                # AR and Cash/Bank from 100001 BS (positive asset values × ih_ratio)
                # Full AR total → placed into AR (Local) in GI BS for years ≤ 2024
                # Cash and Bank → each placed into their own rows for years ≤ 2024
                _ar_tot_101 = _extract_row(bs_s_101, "0103-Accounts Receivable (Total)").reindex(_gi_t_num_cols, fill_value=0.0)
                _cash_101   = _extract_row(bs_s_101, "0101-Cash & Cash Equivalent").reindex(_gi_t_num_cols, fill_value=0.0)
                _bank_101   = _extract_row(bs_s_101, "0102-Bank Balance").reindex(_gi_t_num_cols, fill_value=0.0)

                # Full 100001 AR total × ih_ratio — placed into AR (Local) in GI BS
                _ar_tot_add_101   = _ar_tot_101   * _ih_ratio_101
                _cash_add_101     = _cash_101      * _ih_ratio_101
                _bank_add_101     = _bank_101      * _ih_ratio_101
                _cb_removed_101   = _cash_add_101 + _bank_add_101  # kept for breakdown table

                # Build breakdown data dict (display deferred to after statements)
                _gi_bkdn_data = {
                    "I&H Revenue (Net Sales)": [
                        _ind_hh_101.get(c, 0.0) for c in _gi_t_num_cols
                    ],
                    "I&H Net Cost (COGS Removed)": [
                        _cogs_adj_101.get(c, 0.0) for c in _gi_t_num_cols
                    ],
                    "8% of I&H Revenue (Expense Reduction)": [
                        0.08 * _ind_hh_101.get(c, 0.0) for c in _gi_t_num_cols
                    ],
                    "Pre-2022 VAT Rebate Reclassified from COGS": [
                        _vat_prior_101.get(c, 0.0) for c in _gi_t_num_cols
                    ],
                    "I&H Stock Added to GI BS (100001 JSON)": [
                        _stock_inh_101.get(c, 0.0) for c in _gi_t_num_cols
                    ],
                    "I&H AR Added to GI BS (100001, Estimated)": [
                        float(_ar_tot_add_101.get(c, 0.0)) for c in _gi_t_num_cols
                    ],
                    "I&H Cash & Bank Added to GI BS (100001, Estimated)": [
                        float(_cb_removed_101.get(c, 0.0)) for c in _gi_t_num_cols
                    ],
                }

                _adj_data_gi = {
                    'ind_hh_net_sales': _ind_hh_101,
                    'cogs_adj':         _cogs_adj_101,
                    'level_s_revenue':  _rev_101,
                    'stock_inh':        _stock_inh_101,
                    'cash_add':         _cash_add_101,
                    'bank_add':         _bank_add_101,
                    # Full 100001 AR total → placed into AR (Local) in GI BS
                    'ar_local_add':     _ar_tot_add_101,
                }
            else:
                st.warning(
                    "ZID 100001 data not available for the selected year range. "
                    "Ensure 100001 is included in the businesses configuration."
                )
                _adj_data_gi = {}
                _gi_bkdn_data = {}

            # ── Step 2b: Sum COGS from subsidiary entities 100002/3/4/7/8 ────
            _cogs_sub_zids  = {'100002', '100003', '100004', '100007', '100008'}
            _cogs_codes     = ["04010020","04010002","04010004","04010008","04010011"]
            _gi_cogs_sum    = pd.Series(0.0, index=_gi_t_num_cols)
            for _ck in main_data_dict_pl:
                if str(_ck[0]) in _cogs_sub_zids:
                    _cogs_k = financial._ls_sum(
                        main_data_dict_pl[_ck], _cogs_codes, "ac_code"
                    ).reindex(_gi_t_num_cols, fill_value=0.0)
                    _gi_cogs_sum = _gi_cogs_sum + _cogs_k
            _adj_data_gi['cogs_from_subsidiaries'] = _gi_cogs_sum

            # ── Step 2c: Sum SG&A and S&D from subsidiary entities 100002/3/4/7/8
            _sga_codes = [
                "06010001","06010002","06010003","06010004","06010005",
                "06010006","06010007","06010008","06010009","06020001","06030001","06030002",
                "06030003","06030004","06030005","06030006","06030007",
                "06030008","06030009","06030010","06030011","06030012",
                "06030013","06030014","06040001","06040002","06040003",
                "06040004","06040005","06040006","06040007","06040008",
                "06060001","06060002","06070001","06070002","06070003",
                "06100001","06160001","06160002","06160003","06160004",
                "06160005","06160007","06160008","06160009","06160010",
                "06170001","06180001","06180002","06180003","06180004",
                "06190001","06190002","06190003","06190004","06190005",
                "06200001","06210001","06210002","06210003","06220001",
                "06220002","06220003","06220004","06220005","06220006",
                "06220007","06220008","06220009","06220010","06220011",
                "06220012","06220013","06220014","06230001","06230002","06230003","06230004",
                "06240001","06240002","06250001","06250002","06250003",
                "06260001","06260002","06260003","06270001","06280001",
                "06310001","06310002","06310003","06310004","06310005",
                "06310006","06310007","06320001","06320002","06340001",
                "06360001","06370001","06380001","06390001","06290003",
                "06290004","05010002","06160006",
                "06120001","06120002",  # salary
                "06130001","06130002",  # bonus
                "06140001","06140002",  # overtime
                "06150001","06150002","06150003",  # director remuneration
            ]
            _sd_codes = [
                "07080001",  # discount paid
                "07010001","07010002","07010003","07010004","07010005",
                "07010006","07010007","07010008","07010009","07020001","07020002","07020003","07020004",
                "07030001","07030002","07030003","07040001","07040002",
                "07040003","07040005","07050001","07050002","07050003","07060001","07060002",
                "07060003","07060004","07060005","07090001","07100001",
                "07100002","07110001","07110002","07110003","07110004",
                "07120001","07120002","07120003","07120004","07120005",
                "07130001","07130002","07130003","07140001",
            ]
            _gi_sga_sum = pd.Series(0.0, index=_gi_t_num_cols)
            _gi_sd_sum  = pd.Series(0.0, index=_gi_t_num_cols)
            for _sk in main_data_dict_pl:
                if str(_sk[0]) in _cogs_sub_zids:
                    _gi_sga_sum = _gi_sga_sum + financial._ls_sum(
                        main_data_dict_pl[_sk], _sga_codes, "ac_code"
                    ).reindex(_gi_t_num_cols, fill_value=0.0)
                    _gi_sd_sum = _gi_sd_sum + financial._ls_sum(
                        main_data_dict_pl[_sk], _sd_codes, "ac_code"
                    ).reindex(_gi_t_num_cols, fill_value=0.0)
            _adj_data_gi['sga_from_subsidiaries'] = _gi_sga_sum
            _adj_data_gi['sd_from_subsidiaries']  = _gi_sd_sum

            # ── Step 2d: Collect raw BS DataFrames for subsidiaries 100002/3/4/7/8 ──
            # Passed to adjust_bs_level_t_gi for consolidated BS adjustments.
            # Raw Level 0 DataFrames — _ls_sum_raw preserves Level 0 signs:
            #   assets positive, liabilities/equity negative.
            _sub_bs_raws = []
            for _bk in main_data_dict_bs:
                if str(_bk[0]) in _cogs_sub_zids:
                    _sub_bs_raws.append(main_data_dict_bs[_bk])
            _adj_data_gi['sub_bs_raws'] = _sub_bs_raws

            # ── Step 2e: Sum Net Profit from Level S IS of subsidiaries 100002/3/4/7/8
            # Build Level S P&L for each sub (no VAT rows needed) and extract Net Income.
            _gi_np_sum = pd.Series(0.0, index=_gi_t_num_cols)
            for _nk in main_data_dict_pl:
                if str(_nk[0]) in _cogs_sub_zids:
                    _sub_pl_s = financial.build_pl_level_s(
                        main_data_dict_pl[_nk], selected_perspective='Yearly'
                    )
                    _ni_k = _extract_row(_sub_pl_s, "Net Income").reindex(
                        _gi_t_num_cols, fill_value=0.0
                    )
                    _gi_np_sum = _gi_np_sum + _ni_k

            # ── Step 3: Apply GI Level T adjustments ─────────────────────────
            pl_gi_t         = financial.adjust_pl_level_t_gi(pl_gi_s, _adj_data_gi)
            net_income_gi_t = _extract_row(pl_gi_t, "Net Income")
            bs_gi_t         = financial.build_bs_level_s(bs_raw, net_income_gi_t, zid=_gi_az)
            bs_gi_t         = financial.adjust_bs_level_t_gi(bs_gi_t, _adj_data_gi)
            cfs_gi_t, summary_gi_t = financial.build_cfs_from_bs_t_gi(bs_gi_t, net_income_gi_t)

            # ── Step 4: Display statements ────────────────────────────────────
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_gi_t), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_gi_t), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_gi_t), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_gi_t), use_container_width=True)

            _dl_link_gi = common.create_combined_ls_download_link(
                pl_s=pl_gi_t, bs_s=bs_gi_t, cfs_s=cfs_gi_t,
                filename="GI_LevelT_Financial_Statements.xlsx",
            )
            st.markdown(_dl_link_gi, unsafe_allow_html=True)

            # ── Step 5: Adjustment breakdown table (at bottom) ────────────────
            if _gi_bkdn_data:
                # Add GI COGS row to the breakdown
                _gi_bkdn_data["GI COGS (Sum of Subsidiaries 100002/3/4/7/8)"] = [
                    float(_gi_cogs_sum.get(c, 0.0)) for c in _gi_t_num_cols
                ]
                _gi_bkdn_data["GI Total SG&A (Sum of Subsidiaries 100002/3/4/7/8)"] = [
                    float(_gi_sga_sum.get(c, 0.0)) for c in _gi_t_num_cols
                ]
                _gi_bkdn_data["GI Total S&D (Sum of Subsidiaries 100002/3/4/7/8)"] = [
                    float(_gi_sd_sum.get(c, 0.0)) for c in _gi_t_num_cols
                ]
                _gi_bkdn_data["Net Profit (Sum of Subsidiaries 100002/3/4/7/8)"] = [
                    float(_gi_np_sum.get(c, 0.0)) for c in _gi_t_num_cols
                ]
                _gi_bkdn_df = pd.DataFrame(_gi_bkdn_data, index=_gi_yr_labels).T
                _gi_bkdn_df.index.name = "Adjustment"
                with st.expander(
                    "GI Level T — Adjustment Breakdown",
                    expanded=True,
                ):
                    st.dataframe(_fmt(_gi_bkdn_df.reset_index()), use_container_width=True)

            # ── Notes ─────────────────────────────────────────────────────────
            st.info(
                "**Revenue Adjustment**\n\n"
                "GI Corporation (100000) operated as the intercompany entity through which "
                "I&H product sales were conducted. The revenue adjustment is applied as follows:\n\n"
                "- **2023 and earlier**: GI's Level S Revenue is replaced entirely with the "
                "I&H Net Sales figure recorded by ZID 100001 for the same year, reflecting "
                "that GI's historical revenue was entirely pass-through I&H sales.\n\n"
                "- **2024**: The I&H Net Sales from 100001 is added on top of GI's existing "
                "Level S Revenue, reflecting a partial-year consolidation where both GI's "
                "own revenue and the I&H contribution are recognised.\n\n"
                "- **2025 and beyond**: GI's Level S Revenue is used as-is with no adjustment.\n\n"
                "Adjusted Revenue (Pending), Gross Profit, EBITDA, and Net Income are "
                "recalculated to reflect the revised revenue in all affected years.\n\n"
                "**COGS Adjustment**\n\n"
                "COGS for GI (100000) is sourced from the five operating subsidiaries "
                "(ZIDs 100002, 100003, 100004, 100007, 100008) using the same Level S "
                "account codes (04010020, 04010002, 04010004, 04010008, 04010011).\n\n"
                "- **Up to and including 2023**: GI's own Level S COGS is replaced "
                "entirely with the sum of COGS from the subsidiaries.\n\n"
                "- **2024**: The subsidiaries COGS sum is added on top of GI's existing "
                "Level S COGS (partial-year consolidation).\n\n"
                "- **2025 and beyond**: GI's Level S COGS is used as-is with no adjustment, "
                "as GI adjustments are already reflected in the source data.\n\n"
                "Gross Profit, EBITDA, and Net Income are recalculated accordingly.\n\n"
                "**SG&A Adjustment**\n\n"
                "Total SG&A for GI (100000) is sourced from the five operating subsidiaries "
                "(ZIDs 100002, 100003, 100004, 100007, 100008) using the same Level S account "
                "codes as Total SG&A (SG&A lines + salary, bonus, overtime, director remuneration).\n\n"
                "- **Up to and including 2023**: GI's Level S Total SG&A is replaced entirely "
                "with the sum from subsidiaries.\n\n"
                "- **2024**: The subsidiaries SG&A sum is added on top of GI's existing Level S "
                "Total SG&A.\n\n"
                "- **2025 and beyond**: GI's Level S Total SG&A is used as-is. "
                "Further adjustments to SG&A may be applied in a later step.\n\n"
                "**S&D Adjustment**\n\n"
                "Total Sales & Distribution for GI (100000) follows the same approach as SG&A, "
                "sourced from subsidiaries 100002/3/4/7/8 using the Level S S&D account codes "
                "(discount paid + S&D expenses).\n\n"
                "- **Up to and including 2023**: GI's Level S Total S&D is replaced entirely "
                "with the sum from subsidiaries.\n\n"
                "- **2024**: The subsidiaries S&D sum is added on top of GI's existing Level S "
                "Total S&D.\n\n"
                "- **2025 and beyond**: GI's Level S Total S&D is used as-is.\n\n"
                "EBITDA and Net Income are recalculated to reflect all SG&A and S&D adjustments.\n\n"
                "**Balance Sheet Adjustments (Consolidation)**\n\n"
                "Sign convention (inherited from Level 0): assets → positive; "
                "liabilities & equity → negative.\n\n"
                "**Current Assets (from 100001 trading adjustments, added for years ≤ 2024):**\n\n"
                "- **0101 Cash CE / 0102 Bank Balance**: Each component of 100001's cash "
                "and bank is multiplied by ih_ratio (= I&H Net Sales / 100001 Revenue) and "
                "added as a positive value. Level T = Level S for > 2024.\n\n"
                "- **AR rows (main, agent, local, total)**: Same approach — 100001's "
                "AR sub-rows × ih_ratio added as positive values. 0103-AR Total and "
                "01-Total Current Assets (A) recalculated. Level T = Level S for > 2024.\n\n"
                "- **0106-Stock in Hand**: I&H stock (from JSON, 2016–2023, positive) "
                "added. Additionally, Level S stock from subs 100002/3/4/7/8 is summed "
                "and added for years < 2024 only. Level T = Level S for ≥ 2024.\n\n"
                "**Other Assets (B) — added from subs 100002/3/4/7/8, years < 2024 only:**\n\n"
                "- 0205-Loan to Others Concern, 0206-Other Investment → Total Other Assets (B) recalculated.\n\n"
                "**Fixed Assets (C) — added from subs 100002/3/4/7/8, years < 2024 only:**\n\n"
                "- 0301–0308 (Office Equipment through Land & Building) → "
                "Total Fixed Assets (C) and Total Assets (A+B+C) recalculated.\n\n"
                "**Current Liabilities (A) — added from subs, years < 2024 only:**\n\n"
                "- 0901 Accrued Expenses, AP sub-rows + 0903-AP Total, "
                "0904 Money Agent / Recon, 0905 C&F, 0906 Others → Current Liability (A) recalculated.\n\n"
                "**Short / Long-Term Loans — added from subs, years < 2024 only:**\n\n"
                "- 1001 STB + ST Loan → Total Short Term Liability (C). "
                "1201 Long Term Bank Loan (E).\n\n"
                "**Reserve & Funds (D) — added from subs, years < 2024 only:**\n\n"
                "- 1101 Employee Fund, 1102 Directors Award / Office Rent Tax, "
                "1103 Educational Fund, 1104 Security Deposit → Total RF and "
                "Total Liabilities (A+B+C+D+E) recalculated.\n\n"
                "**Equity — added from subs, years < 2024 only:**\n\n"
                "- 1301-Share Capital, 1301A-Non-Cash Capital (Retained Earning).\n\n"
                "**Balancing (step 12):** After all rows are set, the Error Adjustment "
                "for Retained Earnings absorbs the full imbalance so that Balance Check = 0. "
                "Total Equity and Total Liabilities & Equity are recalculated last."
            )

    else: # Monthly
        year = year_list[-1]

        # Process Profit and Loss Data
        for zid, details in businesses.items():
            for project in details.get('projects', [None]):
                main_data_dict_pl[(zid, project)] = financial.process_data_month(zid, year, start_month, end_month, 'Income Statement', income_label_df, project, {'Asset', 'Liability'})

        # Process Balance Sheet Data
        for zid, details in businesses.items():
            for project in details.get('projects', [None]):
                main_data_dict_bs[(zid, project)] = financial.process_data_month(zid, year, start_month, end_month, 'Balance Sheet', balance_label_df, project, {'Income', 'Expenditure'})

        st.title("Financial Statement Analysis Monthly")
        cols = st.columns(2)
        with cols[0]:
            _CONSOL_KEY = ("consolidated", "All Businesses - Consolidated")
            biz_options = [_CONSOL_KEY] + [k for k in main_data_dict_pl.keys()]
            global_zid = st.session_state.get("zid", None)
            default_idx = next(
                (i for i, k in enumerate(biz_options) if str(k[0]) == str(global_zid)),
                0
            )
            analyse_zid = st.selectbox("View Statements For", biz_options, index=default_idx)
        _is_consolidated = str(analyse_zid[0]) == "consolidated"
        with cols[1]:
            if _is_consolidated:
                _level_opts_m = [
                    "Level C - Raw Consolidation",
                    "Level C2 - Consolidated Detail",
                    "Level 1 - Moderate Detail",
                    "Level 2 - Least Detail",
                    "Level S - Customised Detail",
                    "Level P - Projection View",
                ]
            else:
                _level_opts_m = level_options
            selected_level = st.selectbox("Select Level", _level_opts_m)

        # ── Raw data: consolidated or single-business ─────────────────────────
        _all_pl_map: dict = {}
        _all_bs_map: dict = {}
        _hier_cols_m = ['ac_type','ac_lv1','ac_lv2','ac_lv3','ac_lv4','ac_lv5']
        level_c_is = level_c_bs = c2_is = c2_bs = None
        if _is_consolidated:
            _all_pl_map = {str(k[0]): v for k, v in main_data_dict_pl.items()}
            _all_bs_map = {str(k[0]): v for k, v in main_data_dict_bs.items()}
            _zid_frames_pl_m = {
                int(k[0]): v.drop(columns=_hier_cols_m, errors='ignore')
                for k, v in main_data_dict_pl.items()
            }
            _zid_frames_bs_m = {
                int(k[0]): v.drop(columns=_hier_cols_m, errors='ignore')
                for k, v in main_data_dict_bs.items()
            }
            level_c_is = _consol.build_level_c(_zid_frames_pl_m, kind="is")
            level_c_bs = _consol.build_level_c(_zid_frames_bs_m, kind="bs")
            c2_is = _consol.build_level_c2_is(level_c_is)
            c2_bs = _consol.build_level_c2_bs(level_c_bs)
            pl_raw = c2_is
            bs_raw = c2_bs
        else:
            pl_raw = main_data_dict_pl[analyse_zid]
            bs_raw = main_data_dict_bs[analyse_zid]

        drop_cols = _hier_cols_m
        pl_lv0 = pl_raw.drop(columns=drop_cols, errors='ignore')
        bs_lv0 = bs_raw.drop(columns=drop_cols, errors='ignore')

        pl_sorted, net_profit, net_profit_m, dep_row = financial.sort_pl_level0(pl_lv0, selected_perspective='Monthly')
        bs_lv0 = financial.append_net_profit_to_bs_level0(bs_lv0, net_profit_m)
        coc_lv0 = financial.cash_open_close(bs_lv0)

        # CFS requires ≥2 common monthly periods between IS and BS.
        # For newer entities with limited history this may not be available —
        # gracefully degrade to IS + BS only rather than crashing the page.
        _cfs_monthly_available = True
        try:
            cfs_df, summary_df = financial.make_cashflow_statement_level0(
                pl_lv0, bs_lv0, coc_lv0, selected_perspective='Monthly'
            )
        except (ValueError, Exception):
            _cfs_monthly_available = False
            cfs_df    = pd.DataFrame()
            summary_df = pd.DataFrame()

        pl_lv1, pl_lv2 = financial.level_builder(pl_sorted, "IS")
        bs_lv1, bs_lv2 = financial.level_builder(bs_lv0, "BS")

        pl_lv1, bs_lv1, net_profitlv1, dep_rowlv1 = financial.add_np_and_balance_lv1(pl_lv1, bs_lv1, selected_perspective='Monthly')
        if _cfs_monthly_available:
            cfs_lv1 = financial.consolidate_cfs(cfs_df, level=1, debug=True)
            cfs_lv1, summary_df1 = financial.build_cfs_level1_summary_df(cfs_lv1, net_profitlv1, dep_rowlv1, coc_lv0)
        else:
            cfs_lv1 = pd.DataFrame()
            summary_df1 = pd.DataFrame()

        pl_lv2, bs_lv2, net_profitlv2 = financial.add_np_and_balance_lv2(pl_lv2, bs_lv2, selected_perspective='Monthly')
        if _cfs_monthly_available:
            cfs_lv2 = financial.consolidate_cfs(cfs_df, level=2, debug=True)
            cfs_lv2, summary_df2 = financial.build_cfs_level2_summary(cfs_lv2, net_profitlv2, dep_rowlv1, coc_lv0)
        else:
            cfs_lv2 = pd.DataFrame()
            summary_df2 = pd.DataFrame()

        if not _cfs_monthly_available:
            st.warning(
                "Cash Flow Statement is not available for this business/period — "
                "at least two months of overlapping data across both the Income "
                "Statement and Balance Sheet are required. "
                "Income Statement and Balance Sheet are shown below."
            )

        if selected_level == "Level C - Raw Consolidation":
            # Level C: simple concat of all ZID frames — ZID column retained.
            _lc_is_disp_m = level_c_is.assign(zid=level_c_is["zid"].astype(str))
            _lc_bs_disp_m = level_c_bs.assign(zid=level_c_bs["zid"].astype(str))

            # ── IS: append Net Profit/Loss row ────────────────────────────────
            _lc_per_cols_m = [c for c in _lc_is_disp_m.columns
                              if c not in {"zid", "ac_code", "ac_name"}
                              and pd.api.types.is_numeric_dtype(_lc_is_disp_m[c])]
            _lc_np_series_m = _lc_is_disp_m[_lc_per_cols_m].sum()
            _lc_np_row_is_m = {"zid": "", "ac_code": "", "ac_name": "Net Profit/Loss"}
            _lc_np_row_is_m.update(_lc_np_series_m.to_dict())
            _lc_is_disp_m = pd.concat(
                [_lc_is_disp_m, pd.DataFrame([_lc_np_row_is_m])], ignore_index=True
            )

            # ── BS: append Net Profit/Loss row then Balance Check ─────────────
            _lc_bs_per_cols_m = [c for c in _lc_bs_disp_m.columns
                                 if c not in {"zid", "ac_code", "ac_name"}
                                 and pd.api.types.is_numeric_dtype(_lc_bs_disp_m[c])]
            _lc_np_for_bs_m = _lc_np_series_m.reindex(_lc_bs_per_cols_m, fill_value=0)
            _lc_np_row_bs_m = {"zid": "", "ac_code": "", "ac_name": "Net Profit/Loss"}
            _lc_np_row_bs_m.update(_lc_np_for_bs_m.to_dict())
            _lc_bs_plus_np_m = pd.concat(
                [_lc_bs_disp_m, pd.DataFrame([_lc_np_row_bs_m])], ignore_index=True
            )
            _lc_bal_series_m = _lc_bs_plus_np_m[_lc_bs_per_cols_m].sum()
            _lc_bal_row_m = {"zid": "", "ac_code": "", "ac_name": "Balance Check"}
            _lc_bal_row_m.update(_lc_bal_series_m.to_dict())
            _lc_bs_disp_m = pd.concat(
                [_lc_bs_plus_np_m, pd.DataFrame([_lc_bal_row_m])], ignore_index=True
            )

            _lc_cfs_ok_m = True
            try:
                _lc_cfs_m, _lc_summary_m = _consol.build_level_c_cfs(
                    level_c_bs, level_c_is, 'Monthly'
                )
            except Exception:
                _lc_cfs_ok_m = False
                _lc_cfs_m = _lc_summary_m = pd.DataFrame()
            with st.expander("Income Statement (Level C)", expanded=True):
                st.dataframe(_fmt(_lc_is_disp_m), use_container_width=True)
            with st.expander("Balance Sheet (Level C)", expanded=True):
                st.dataframe(_fmt(_lc_bs_disp_m), use_container_width=True)
            if _lc_cfs_ok_m:
                with st.expander("Cash Flow Statement (Level C)", expanded=True):
                    st.dataframe(_fmt(_lc_cfs_m), use_container_width=True)
                with st.expander("Cash Flow Summary", expanded=False):
                    st.dataframe(_fmt(_lc_summary_m), use_container_width=True)
            st.markdown(
                common.create_combined_ls_download_link(
                    pl_s=_lc_is_disp_m, bs_s=_lc_bs_disp_m,
                    cfs_s=_lc_cfs_m if _lc_cfs_ok_m else pd.DataFrame(),
                    filename="LevelC_Consolidated_Financial_Statements_Monthly.xlsx",
                    link_label="⬇ Download Level C Financial Statements (Excel)",
                ),
                unsafe_allow_html=True,
            )

        elif selected_level in ("Level 0 - Most Detail", "Level C2 - Consolidated Detail"):
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_sorted), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_lv0), use_container_width=True)
            if _cfs_monthly_available:
                with st.expander("Cash Flow Statement", expanded=True):
                    st.dataframe(_fmt(cfs_df), use_container_width=True)
                with st.expander("Cash Flow Summary", expanded=False):
                    st.dataframe(_fmt(summary_df), use_container_width=True)
            st.markdown(
                common.create_combined_ls_download_link(
                    pl_s=pl_sorted, bs_s=bs_lv0,
                    cfs_s=cfs_df if _cfs_monthly_available else pd.DataFrame(),
                    filename="LevelC2_Consolidated_Financial_Statements_Monthly.xlsx"
                    if _is_consolidated else f"Level0_{analyse_zid[0]}_Financial_Statements.xlsx",
                    link_label="⬇ Download Level C2 Financial Statements (Excel)"
                    if _is_consolidated else "⬇ Download Level 0 Financial Statements (Excel)",
                ),
                unsafe_allow_html=True,
            )

        elif selected_level == "Level 1 - Moderate Detail":
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_lv1), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_lv1), use_container_width=True)
            if _cfs_monthly_available:
                with st.expander("Cash Flow Statement", expanded=True):
                    st.dataframe(_fmt(cfs_lv1), use_container_width=True)
                with st.expander("Cash Flow Summary", expanded=False):
                    st.dataframe(_fmt(summary_df1), use_container_width=True)
            st.markdown(
                common.create_combined_ls_download_link(
                    pl_s=pl_lv1, bs_s=bs_lv1,
                    cfs_s=cfs_lv1 if _cfs_monthly_available else pd.DataFrame(),
                    filename=f"Level1_{analyse_zid[0]}_Financial_Statements.xlsx",
                    link_label="⬇ Download Level 1 Financial Statements (Excel)",
                ),
                unsafe_allow_html=True,
            )

        elif selected_level == "Level 2 - Least Detail":
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_lv2), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_lv2), use_container_width=True)
            if _cfs_monthly_available:
                with st.expander("Cash Flow Statement", expanded=True):
                    st.dataframe(_fmt(cfs_lv2), use_container_width=True)
                with st.expander("Cash Flow Summary", expanded=False):
                    st.dataframe(_fmt(summary_df2), use_container_width=True)
            st.markdown(
                common.create_combined_ls_download_link(
                    pl_s=pl_lv2, bs_s=bs_lv2,
                    cfs_s=cfs_lv2 if _cfs_monthly_available else pd.DataFrame(),
                    filename=f"Level2_{analyse_zid[0]}_Financial_Statements.xlsx",
                    link_label="⬇ Download Level 2 Financial Statements (Excel)",
                ),
                unsafe_allow_html=True,
            )

        elif selected_level == "Level S - Customised Detail":
            # Level S requires raw data (ac_code still present)
            _az, _ap = analyse_zid
            _num_cols = financial._ls_num_cols(pl_raw)

            # Period columns span two calendar years (prior year full + current
            # year up to end_month). Derive all years present so the VAT query
            # covers every month in the displayed IS.
            _years_in_data = sorted(set(
                int(financial._period_key(c)[0]) for c in _num_cols
            ))
            if _is_consolidated:
                # Consolidated: fetch VAT GL for every ZID and sum them.
                # The 4 VAT rows are informational references — no interaction with
                # the rest of the statement, same as for individual entities.
                _vat_gl_parts = []
                _all_zids_vat = _consol.load_consolidation_rules().get("all_zids", [])
                for _vat_zid in _all_zids_vat:
                    try:
                        _sql_vat_z, _params_vat_z = queries.get_vat_breakdown_gl(
                            zid=_vat_zid, year_list=_years_in_data, smonth=1, emonth=12,
                        )
                        _gl_vat_z = get_dataframe(_sql_vat_z, _params_vat_z)
                        if _gl_vat_z is not None and not _gl_vat_z.empty:
                            _vat_gl_parts.append(_gl_vat_z)
                    except Exception:
                        pass
                _gl_vat_all = (pd.concat(_vat_gl_parts, ignore_index=True)
                               if _vat_gl_parts else pd.DataFrame())
                _vat_rows = financial.compute_vat_is_rows(
                    _gl_vat_all, _num_cols, selected_perspective='Monthly'
                )
            else:
                _sql_vat, _params_vat = queries.get_vat_breakdown_gl(
                    zid=_az, project=_ap, year_list=_years_in_data,
                    smonth=1, emonth=12,
                )
                _gl_vat   = get_dataframe(_sql_vat, _params_vat)
                _vat_rows = financial.compute_vat_is_rows(
                    _gl_vat, _num_cols, selected_perspective='Monthly'
                )

            pl_s  = financial.build_pl_level_s(
                pl_raw, selected_perspective='Monthly', vat_rows=_vat_rows
            )
            net_income_s = _extract_row(pl_s, "Net Income")
            # BS needs YTD cumulative NI (mirrors Level 0's net_profit_m).
            # Monthly raw NI only equals YTD in January, so all other months
            # fail the balance check without this conversion.
            net_income_s_ytd = _monthly_to_ytd(net_income_s)
            bs_s  = financial.build_bs_level_s(bs_raw, net_income_s_ytd, zid=_az)
            # CFS uses the raw monthly NI (not YTD) — same as Level 0 CFS.
            cfs_s, summary_s = financial.build_cfs_level_s(
                pl_raw, bs_raw, coc_lv0, net_income_s, zid=_az
            )

            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_s), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_s), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_s), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_s), use_container_width=True)

            # ── Financial Ratios ──────────────────────────────────────────────
            with st.expander("📊 Financial Ratios", expanded=False):
                try:
                    _ratio_df_m = _build_ls_ratios(pl_s, bs_s, cfs_s, perspective="Monthly")
                    st.caption("Monthly working-capital days are approximated using 30 days/period.")
                    st.dataframe(_ratio_df_m.set_index("Ratio"), use_container_width=True)
                except Exception as _re:
                    st.warning(f"Could not compute ratios: {_re}")

            _dl_link = common.create_combined_ls_download_link(
                pl_s=pl_s, bs_s=bs_s, cfs_s=cfs_s,
                filename="LevelS_Consolidated_Financial_Statements_Monthly.xlsx"
                if _is_consolidated else "LevelS_Financial_Statements_Monthly.xlsx",
            )
            st.markdown(_dl_link, unsafe_allow_html=True)

            _sanity_checks(
                pl_sorted, pl_lv1, pl_lv2, pl_s,
                bs_lv0, bs_lv1, bs_lv2, bs_s,
                summary_df, summary_df1, summary_df2, summary_s,
            )

            # ── Notes / ZID Breakdown / MTD Dashboard panel ──────────────────
            # Determine ZID list and current month for MTD query
            _now           = datetime.now()
            _mtd_year      = _now.year
            _mtd_month     = _now.month
            if _is_consolidated:
                _mtd_zids = _consol.load_consolidation_rules().get("all_zids", [])
            else:
                _mtd_zids = [_az]

            if _is_consolidated:
                _panel_m = st.radio(
                    "View",
                    ["📋 Notes & Context", "📊 ZID Contribution Breakdown", "📊 MTD Dashboard"],
                    horizontal=True,
                    key="ls_panel_m",
                )
                if _panel_m == "📋 Notes & Context":
                    _render_ls_notes(key_suffix="m")
                elif _panel_m == "📊 ZID Contribution Breakdown":
                    _render_zid_contribution_breakdown(
                        pl_s, bs_s,
                        _zid_frames_pl_m, _zid_frames_bs_m,
                        perspective="Monthly",
                        key_suffix="m",
                    )
                else:
                    try:
                        _mtd = financial.compute_mtd_is(_mtd_zids, _mtd_year, _mtd_month, pl_s)
                        _render_mtd_dashboard(pl_s, _mtd)
                    except Exception as _mtd_err:
                        st.error(f"MTD Dashboard error: {_mtd_err}")
            else:
                _panel_m_single = st.radio(
                    "View",
                    ["📋 Notes & Context", "📊 MTD Dashboard"],
                    horizontal=True,
                    key="ls_panel_m_single",
                )
                if _panel_m_single == "📋 Notes & Context":
                    _render_ls_notes(key_suffix="m_single")
                else:
                    try:
                        _mtd = financial.compute_mtd_is(_mtd_zids, _mtd_year, _mtd_month, pl_s)
                        _render_mtd_dashboard(pl_s, _mtd)
                    except Exception as _mtd_err:
                        st.error(f"MTD Dashboard error: {_mtd_err}")

        elif selected_level == "Level P - Projection View":
            st.info("ℹ️ Level P — Projection View (condensed from Level S).")
            _az, _ap = analyse_zid
            _num_cols_pm = financial._ls_num_cols(pl_raw)
            _years_in_data_pm = sorted(set(
                int(financial._period_key(c)[0]) for c in _num_cols_pm
            ))
            if _is_consolidated:
                _vat_gl_parts_pm = []
                for _vat_zid in _consol.load_consolidation_rules().get("all_zids", []):
                    try:
                        _sql_vat_z, _params_vat_z = queries.get_vat_breakdown_gl(
                            zid=_vat_zid, year_list=_years_in_data_pm, smonth=1, emonth=12,
                        )
                        _gl_vat_z = get_dataframe(_sql_vat_z, _params_vat_z)
                        if _gl_vat_z is not None and not _gl_vat_z.empty:
                            _vat_gl_parts_pm.append(_gl_vat_z)
                    except Exception:
                        pass
                _gl_vat_all_pm = (
                    pd.concat(_vat_gl_parts_pm, ignore_index=True)
                    if _vat_gl_parts_pm else pd.DataFrame()
                )
                _vat_rows_pm = financial.compute_vat_is_rows(
                    _gl_vat_all_pm, _num_cols_pm, selected_perspective='Monthly'
                )
            else:
                _sql_vat_pm, _params_vat_pm = queries.get_vat_breakdown_gl(
                    zid=_az, project=_ap, year_list=_years_in_data_pm,
                    smonth=1, emonth=12,
                )
                _gl_vat_pm = get_dataframe(_sql_vat_pm, _params_vat_pm)
                _vat_rows_pm = financial.compute_vat_is_rows(
                    _gl_vat_pm, _num_cols_pm, selected_perspective='Monthly'
                )
            pl_s_pm = financial.build_pl_level_s(
                pl_raw, selected_perspective='Monthly', vat_rows=_vat_rows_pm
            )
            net_income_s_pm = _extract_row(pl_s_pm, "Net Income")
            net_income_s_ytd_pm = _monthly_to_ytd(net_income_s_pm)
            bs_s_pm = financial.build_bs_level_s(bs_raw, net_income_s_ytd_pm, zid=_az)
            _, summary_s_pm = financial.build_cfs_level_s(
                pl_raw, bs_raw, coc_lv0, net_income_s_pm, zid=_az
            )
            pl_p_m, bs_p_m = financial.build_condensed_view(pl_s_pm, bs_s_pm)
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_p_m), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_p_m), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=True):
                st.dataframe(_fmt(summary_s_pm), use_container_width=True)
            st.markdown(
                common.create_combined_ls_download_link(
                    pl_s=pl_p_m, bs_s=bs_p_m, cfs_s=summary_s_pm,
                    filename="LevelP_Financial_Statements_Monthly.xlsx",
                    link_label="⬇ Download Level P Financial Statements (Excel)",
                ),
                unsafe_allow_html=True,
            )
            _level_p_sanity_checks(pl_p_m, pl_s_pm)

            # ── Margin Sensitivity Analysis ───────────────────────────────────
            with st.expander("📐 Margin Sensitivity Analysis", expanded=False):
                try:
                    _msa_cols = financial._ls_num_cols(pl_p_m)
                    _name_col = "ac_name"

                    def _msa_row(label):
                        r = pl_p_m.loc[pl_p_m[_name_col].astype(str) == label]
                        if r.empty:
                            return pd.Series(0.0, index=_msa_cols)
                        return r[_msa_cols].iloc[0].fillna(0.0)

                    _rev   = _msa_row("Adjusted Revenue (Pending)")
                    _cogs  = _msa_row("COGS")
                    _gp    = _msa_row("Gross Profit")
                    _sga   = _msa_row("Total SG&A")
                    _sd    = _msa_row("Total Sales & Distribution")
                    _int   = _msa_row("Total Interest & Charges")
                    _vat   = _msa_row("0629-VAT & Tax Total (A+B+C)")
                    _ni    = _msa_row("Net Income")

                    # Level S sign: SGA/SD/Int/VAT are negative; negate to positive for MSA.
                    _gm_pct         = _gp / _rev.replace(0, float('nan'))  # GP(P)/Rev, positive
                    _fixed_costs_abs = -(_sga + _sd + _int + _vat)         # positive absolute costs
                    # Break-even revenue = Fixed Costs / Gross Margin%
                    _be_rev      = _fixed_costs_abs / _gm_pct.replace(0, float('nan'))
                    _safety_mar  = _rev - _be_rev
                    _safety_pct  = (_safety_mar / _rev.replace(0, float('nan'))) * 100
                    # Max tolerable fixed cost at current revenue to achieve break even
                    _max_fixed   = _rev * _gm_pct

                    _lbl = [common._period_col_label(c) for c in _msa_cols]

                    def _pct_fmt(s):
                        return {c: f"{v:.1f}%" if not pd.isna(v) else "—"
                                for c, v in zip(_lbl, s.values)}

                    def _amt_fmt(s):
                        return {c: f"{v:,.0f}" if not pd.isna(v) else "—"
                                for c, v in zip(_lbl, s.values)}

                    _be_df = pd.DataFrame([
                        {"Metric": "Revenue (Actual)", **_amt_fmt(_rev)},
                        {"Metric": "Gross Margin %",   **_pct_fmt(_gm_pct * 100)},
                        {"Metric": "Fixed Costs (SG&A + S&D + Interest + Tax)",
                                                        **_amt_fmt(_fixed_costs_abs)},
                        {"Metric": "Break-even Revenue Required",
                                                        **_amt_fmt(_be_rev)},
                        {"Metric": "Revenue Surplus / (Deficit) vs Break-even",
                                                        **_amt_fmt(_safety_mar)},
                        {"Metric": "Safety Margin %",   **_pct_fmt(_safety_pct)},
                        {"Metric": "Max Tolerable Fixed Cost at Current Revenue",
                                                        **_amt_fmt(_max_fixed)},
                    ]).set_index("Metric")

                    st.markdown("**Break-even Summary (per month)**")
                    st.caption(
                        "Fixed Costs = SG&A + S&D + Interest + VAT/Tax  |  "
                        "Break-even Revenue = Fixed Costs ÷ Gross Margin %  |  "
                        "Safety Margin = Actual Revenue − Break-even Revenue"
                    )
                    st.dataframe(_be_df, use_container_width=True)

                    # ── Revenue sensitivity grid ──────────────────────────────
                    st.markdown("**Revenue Sensitivity — Net Income at Various Revenue Levels**")
                    st.caption(
                        "Assumes COGS scales proportionally with revenue (constant gross margin %). "
                        "Fixed costs (SG&A, S&D, Interest, VAT/Tax) held constant."
                    )
                    # NI = GP(P) - |FixedCosts| = Rev*gm% - fixed_costs_abs
                    _scenarios = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
                    _sens_rows2 = []
                    for _pct in _scenarios:
                        _sc_rev = _rev * _pct
                        _sc_ni  = _sc_rev * _gm_pct - _fixed_costs_abs
                        row = {"Revenue Level": f"{int(_pct*100)}% of Actual"}
                        row.update({c: (f"{v:,.0f}" if not pd.isna(v) else "—")
                                    for c, v in zip(_lbl, _sc_ni.values)})
                        _sens_rows2.append(row)
                    _sens_df = pd.DataFrame(_sens_rows2).set_index("Revenue Level")
                    st.dataframe(_sens_df, use_container_width=True)

                    # ── Cost sensitivity: max tolerable fixed cost for NI targets ─
                    st.markdown("**Max Tolerable Fixed Cost — at Current Revenue for Target Net Margin**")
                    st.caption(
                        "Shows the maximum fixed cost (SG&A + S&D + Interest + Tax) "
                        "the business can sustain at current revenue to achieve each Net Income margin target."
                    )
                    _ni_targets = [0.0, 0.02, 0.05, 0.08, 0.10]
                    _cost_rows = []
                    for _tgt in _ni_targets:
                        _max_fc = _rev * (_gm_pct - _tgt)
                        row = {"Target Net Margin": f"{int(_tgt*100)}%"}
                        row.update({c: (f"{v:,.0f}" if not pd.isna(v) else "—")
                                    for c, v in zip(_lbl, _max_fc.values)})
                        _cost_rows.append(row)
                    _cost_df = pd.DataFrame(_cost_rows).set_index("Target Net Margin")
                    st.dataframe(_cost_df, use_container_width=True)

                except Exception as _msa_err:
                    st.warning(f"Could not compute sensitivity analysis: {_msa_err}")
