import streamlit as st
import pandas as pd
from datetime import datetime
from processing import common, financial
from processing import consolidation as _consol
from utils.utils import timed
from core import queries
from core.db import get_dataframe


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


@timed
def display_financial_statements(current_page, zid):
    st.sidebar.title("Financial Statements")
    selected_perspective = st.sidebar.selectbox(
        "Timeframe",
        [
            'Yearly - Custom Range',
            'Yearly - Full Year vs YTD',
            'Monthly',
            'Lifetime',
        ],
        index=0,
    )

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
    ]

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
                # Consolidated: no per-business VAT breakdown — skip VAT query
                _vat_rows = None
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
                pl_raw, bs_raw, coc_lv0, net_income_s
            )

            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_s), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_s), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_s), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_s), use_container_width=True)

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

        elif selected_level == "Level T - Trading Adjustments":
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
                pl_raw, bs_raw, coc_lv0, net_income_s
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
                pl_raw, bs_raw, coc_lv0, net_income_gi_s
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

            if _is_consolidated:
                # Consolidated: no per-business VAT breakdown — skip VAT query
                _vat_rows = None
            else:
                # Period columns span two calendar years (prior year full + current
                # year up to end_month). Derive all years present so the VAT query
                # covers every month in the displayed IS.
                _years_in_data = sorted(set(
                    int(financial._period_key(c)[0]) for c in _num_cols
                ))
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
                pl_raw, bs_raw, coc_lv0, net_income_s
            )

            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_s), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_s), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_s), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_s), use_container_width=True)

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
