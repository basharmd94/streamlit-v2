import streamlit as st
import pandas as pd
from datetime import datetime
from processing import common, financial
from utils.utils import timed


def _extract_row(df: pd.DataFrame, label: str, name_col: str = "ac_name") -> pd.Series:
    """Return the numeric cells of the first row matching label, or zeros."""
    match = df.loc[df[name_col].astype(str) == label]
    if match.empty:
        num = df.select_dtypes("number").columns
        return pd.Series(0.0, index=num)
    return match.select_dtypes("number").iloc[0]


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
        """Build a check DataFrame with one row per level."""
        rows = []
        for lvl_name, df, label in dfs_and_labels:
            s = _extract_row(df, label, name_col)
            rows.append(pd.Series({**{"Level": lvl_name}, **s.to_dict()}))
        return pd.DataFrame(rows).set_index("Level")

    # Net Profit / Net Income
    np_check = _build_check(
        "Net Profit/Loss", "Net Income",
        [
            ("Level 0", pl_sorted,  "Net Profit/Loss"),
            ("Level 1", pl_lv1,     "Net Profit/Loss"),
            ("Level 2", pl_lv2,     "Net Profit/Loss"),
            ("Level S", pl_s,       "Net Income"),
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
    selected_perspective = st.sidebar.selectbox("Timeframe",['Yearly','Monthly'],index=0)

    main_data_dict_pl = {}
    main_data_dict_bs = {}
    # Get the current year
    current_year = datetime.now().year
    # Generate a list of the 10 years starting from the most recent year
    options = [current_year - i for i in range(10)]
    default_option = current_year
    default_index = options.index(default_option)
    selected_year = st.sidebar.selectbox("Select End Year",options,index= default_index)
    year_list = [selected_year - i for i in range(4, -1, -1)]

    month_list = [i for i in range(1, 13)]

    start_month = st.sidebar.selectbox("Select Start Month",month_list)
    end_month = st.sidebar.selectbox("Select End Month",month_list,index=len(month_list)-1)

    if start_month < 1 or start_month > 12 or end_month < 1 or end_month > 12:
        st.markedown("Month should be an integer between 1 and 12")
        raise ValueError("Month should be an integer between 1 and 12")

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

    if selected_perspective == 'Yearly':
         # Process Profit and Loss Data
        for zid, details in businesses.items():
            for project in details.get('projects', [None]):
                main_data_dict_pl[(zid, project)] = financial.process_data(zid, year_list, start_month, end_month, 'Income Statement', income_label_df, project, {'Asset', 'Liability'})

        # Process Balance Sheet Data
        for zid, details in businesses.items():
            for project in details.get('projects', [None]):
                main_data_dict_bs[(zid, project)] = financial.process_data(zid, year_list, start_month, end_month, 'Balance Sheet', balance_label_df, project, {'Income', 'Expenditure'})

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
                st.write(pl_sorted, use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.write(bs_lv0, use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.write(cfs_df, use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.write(summary_df, use_container_width=True)

        elif selected_level == "Level 1 - Moderate Detail":
            with st.expander("Income Statement", expanded=True):
                st.write(pl_lv1, use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.write(bs_lv1, use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.write(cfs_lv1, use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.write(summary_df1, use_container_width=True)

        elif selected_level == "Level 2 - Least Detail":
            with st.expander("Income Statement", expanded=True):
                st.write(pl_lv2, use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.write(bs_lv2, use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.write(cfs_lv2, use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.write(summary_df2, use_container_width=True)

        elif selected_level == "Level S - Customised Detail":
            # Level S requires raw data (ac_code still present)
            pl_s  = financial.build_pl_level_s(pl_raw, selected_perspective='Yearly')
            net_income_s = _extract_row(pl_s, "Net Income")
            bs_s  = financial.build_bs_level_s(bs_raw, net_income_s)
            cfs_s, summary_s = financial.build_cfs_level_s(
                pl_raw, bs_raw, coc_lv0, net_income_s
            )

            with st.expander("Income Statement", expanded=True):
                st.write(pl_s, use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.write(bs_s, use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.write(cfs_s, use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.write(summary_s, use_container_width=True)

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
                st.write(pl_sorted, use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.write(bs_lv0, use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.write(cfs_df, use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.write(summary_df, use_container_width=True)

        elif selected_level == "Level 1 - Moderate Detail":
            with st.expander("Income Statement", expanded=True):
                st.write(pl_lv1, use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.write(bs_lv1, use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.write(cfs_lv1, use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.write(summary_df1, use_container_width=True)

        elif selected_level == "Level 2 - Least Detail":
            with st.expander("Income Statement", expanded=True):
                st.write(pl_lv2, use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.write(bs_lv2, use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.write(cfs_lv2, use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.write(summary_df2, use_container_width=True)

        elif selected_level == "Level S - Customised Detail":
            # Level S requires raw data (ac_code still present)
            pl_s  = financial.build_pl_level_s(pl_raw, selected_perspective='Monthly')
            net_income_s = _extract_row(pl_s, "Net Income")
            bs_s  = financial.build_bs_level_s(bs_raw, net_income_s)
            cfs_s, summary_s = financial.build_cfs_level_s(
                pl_raw, bs_raw, coc_lv0, net_income_s
            )

            with st.expander("Income Statement", expanded=True):
                st.write(pl_s, use_container_width=True)
            with st.expander("Balance Sheet", expanded=True):
                st.write(bs_s, use_container_width=True)
            with st.expander("Cash Flow Statement", expanded=True):
                st.write(cfs_s, use_container_width=True)
            with st.expander("Cash Flow Summary", expanded=False):
                st.write(summary_s, use_container_width=True)

            _sanity_checks(
                pl_sorted, pl_lv1, pl_lv2, pl_s,
                bs_lv0, bs_lv1, bs_lv2, bs_s,
                summary_df, summary_df1, summary_df2, summary_s,
            )
