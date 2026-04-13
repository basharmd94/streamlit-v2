import streamlit as st
import pandas as pd
from datetime import datetime
from processing import common, financial
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
        ['Yearly - Custom Range', 'Yearly - Full Year vs YTD', 'Monthly'],
        index=0,
    )

    main_data_dict_pl = {}
    main_data_dict_bs = {}
    current_year = datetime.now().year
    options = [current_year - i for i in range(10)]
    selected_year = st.sidebar.selectbox("Select End Year", options, index=0)
    year_list = [selected_year - i for i in range(4, -1, -1)]

    month_list = list(range(1, 13))

    if selected_perspective == 'Yearly - Full Year vs YTD':
        ytd_month  = st.sidebar.selectbox("Up to Month (Current Year)", month_list, index=len(month_list) - 1)
        start_month, end_month = 1, 12   # prior years always full year
    else:
        start_month = st.sidebar.selectbox("Select Start Month", month_list)
        end_month   = st.sidebar.selectbox("Select End Month", month_list, index=len(month_list) - 1)
        ytd_month   = end_month           # unused in non-YTD modes

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

    if selected_perspective in ('Yearly - Custom Range', 'Yearly - Full Year vs YTD'):
        if selected_perspective == 'Yearly - Custom Range':
            for zid, details in businesses.items():
                for project in details.get('projects', [None]):
                    main_data_dict_pl[(zid, project)] = financial.process_data(zid, year_list, start_month, end_month, 'Income Statement', income_label_df, project, {'Asset', 'Liability'})
            for zid, details in businesses.items():
                for project in details.get('projects', [None]):
                    main_data_dict_bs[(zid, project)] = financial.process_data(zid, year_list, start_month, end_month, 'Balance Sheet', balance_label_df, project, {'Income', 'Expenditure'})

        else:  # Yearly - Full Year vs YTD
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

        st.title("Financial Statement Analysis Yearly")
        cols = st.columns(2)
        with cols[0]:
            biz_options = [k for k in main_data_dict_pl.keys()]
            global_zid = st.session_state.get("zid", None)
            default_idx = next(
                (i for i, k in enumerate(biz_options) if str(k[0]) == str(global_zid)),
                0
            )
            analyse_zid = st.selectbox("View Statements For", biz_options, index=default_idx)
        with cols[1]:
            selected_level = st.selectbox("Select Level", level_options)

        pl_raw = main_data_dict_pl[analyse_zid]
        bs_raw = main_data_dict_bs[analyse_zid]

        drop_cols = ['ac_type','ac_lv1','ac_lv2','ac_lv3','ac_lv4','ac_lv5']
        pl_lv0 = pl_raw.drop(columns=drop_cols)
        bs_lv0 = bs_raw.drop(columns=drop_cols)

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

        if selected_level == "Level 0 - Most Detail":
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_sorted), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_lv0), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_df), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_df), use_container_width=True)

        elif selected_level == "Level 1 - Moderate Detail":
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_lv1), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_lv1), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_lv1), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_df1), use_container_width=True)

        elif selected_level == "Level 2 - Least Detail":
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_lv2), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_lv2), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_lv2), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_df2), use_container_width=True)

        elif selected_level == "Level S - Customised Detail":
            # Level S requires raw data (ac_code still present)
            _az, _ap = analyse_zid
            # For Full Year vs YTD: fetch all months (1-12) so prior years are complete;
            # current year data already bounded by ytd_month via process_data.
            # For Custom Range: use the selected start/end months.
            _vat_smonth = 1 if selected_perspective == 'Yearly - Full Year vs YTD' else start_month
            _vat_emonth = 12 if selected_perspective == 'Yearly - Full Year vs YTD' else end_month
            _sql_vat, _params_vat = queries.get_vat_breakdown_gl(
                zid=_az, project=_ap, year_list=year_list,
                smonth=_vat_smonth, emonth=_vat_emonth,
            )
            _gl_vat = get_dataframe(_sql_vat, _params_vat)
            _num_cols = financial._ls_num_cols(pl_raw)
            _vat_rows = financial.compute_vat_is_rows(
                _gl_vat, _num_cols, selected_perspective='Yearly'
            )
            pl_s  = financial.build_pl_level_s(
                pl_raw, selected_perspective='Yearly', vat_rows=_vat_rows
            )
            net_income_s = _extract_row(pl_s, "Net Income")
            bs_s  = financial.build_bs_level_s(bs_raw, net_income_s)
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
                filename="LevelS_Financial_Statements.xlsx",
            )
            st.markdown(_dl_link, unsafe_allow_html=True)

            _sanity_checks(
                pl_sorted, pl_lv1, pl_lv2, pl_s,
                bs_lv0, bs_lv1, bs_lv2, bs_s,
                summary_df, summary_df1, summary_df2, summary_s,
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
            biz_options = [k for k in main_data_dict_pl.keys()]
            global_zid = st.session_state.get("zid", None)
            default_idx = next(
                (i for i, k in enumerate(biz_options) if str(k[0]) == str(global_zid)),
                0
            )
            analyse_zid = st.selectbox("View Statements For", biz_options, index=default_idx)
        with cols[1]:
            selected_level = st.selectbox("Select Level", level_options)

        pl_raw = main_data_dict_pl[analyse_zid]
        bs_raw = main_data_dict_bs[analyse_zid]

        drop_cols = ['ac_type','ac_lv1','ac_lv2','ac_lv3','ac_lv4','ac_lv5']
        pl_lv0 = pl_raw.drop(columns=drop_cols)
        bs_lv0 = bs_raw.drop(columns=drop_cols)

        pl_sorted, net_profit, net_profit_m, dep_row = financial.sort_pl_level0(pl_lv0, selected_perspective='Monthly')
        bs_lv0 = financial.append_net_profit_to_bs_level0(bs_lv0, net_profit_m)
        coc_lv0 = financial.cash_open_close(bs_lv0)
        cfs_df, summary_df = financial.make_cashflow_statement_level0(pl_lv0, bs_lv0, coc_lv0, selected_perspective='Monthly')

        pl_lv1, pl_lv2 = financial.level_builder(pl_sorted, "IS")
        bs_lv1, bs_lv2 = financial.level_builder(bs_lv0, "BS")

        pl_lv1, bs_lv1, net_profitlv1, dep_rowlv1 = financial.add_np_and_balance_lv1(pl_lv1, bs_lv1, selected_perspective='Monthly')
        cfs_lv1 = financial.consolidate_cfs(cfs_df, level=1, debug=True)
        cfs_lv1, summary_df1 = financial.build_cfs_level1_summary_df(cfs_lv1, net_profitlv1, dep_rowlv1,coc_lv0)

        pl_lv2, bs_lv2, net_profitlv2 = financial.add_np_and_balance_lv2(pl_lv2, bs_lv2, selected_perspective='Monthly')
        cfs_lv2 = financial.consolidate_cfs(cfs_df, level=2, debug=True)
        cfs_lv2, summary_df2 = financial.build_cfs_level2_summary(cfs_lv2, net_profitlv2, dep_rowlv1,coc_lv0)

        if selected_level == "Level 0 - Most Detail":
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_sorted), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_lv0), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_df), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_df), use_container_width=True)

        elif selected_level == "Level 1 - Moderate Detail":
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_lv1), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_lv1), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_lv1), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_df1), use_container_width=True)

        elif selected_level == "Level 2 - Least Detail":
            with st.expander("Income Statement", expanded=True):
                st.dataframe(_fmt(pl_lv2), use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.dataframe(_fmt(bs_lv2), use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.dataframe(_fmt(cfs_lv2), use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.dataframe(_fmt(summary_df2), use_container_width=True)

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
            _sql_vat, _params_vat = queries.get_vat_breakdown_gl(
                zid=_az, project=_ap, year_list=_years_in_data,
                smonth=1, emonth=12,
            )
            _gl_vat = get_dataframe(_sql_vat, _params_vat)
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
            bs_s  = financial.build_bs_level_s(bs_raw, net_income_s_ytd)
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
                filename="LevelS_Financial_Statements_Monthly.xlsx",
            )
            st.markdown(_dl_link, unsafe_allow_html=True)

            _sanity_checks(
                pl_sorted, pl_lv1, pl_lv2, pl_s,
                bs_lv0, bs_lv1, bs_lv2, bs_s,
                summary_df, summary_df1, summary_df2, summary_s,
            )
