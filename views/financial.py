import streamlit as st
import pandas as pd
from datetime import datetime
from processing import common, financial
from utils.utils import timed


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
        # choice of streamlit
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
            analyse_type = st.selectbox("Select Analysis Type",['Single Business Unit','Consolidated'])

        if analyse_type == 'Single Business Unit':
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

            with st.expander("Level 0 - Most Detailed"):
                for tbl in (pl_sorted, bs_lv0,cfs_df, summary_df):
                    st.write(tbl, use_container_width=True)

            with st.expander("Level 1 - Cosolidated"):
                for tbl in (pl_lv1, bs_lv1, cfs_lv1, summary_df1):
                    st.write(tbl, use_container_width=True)

            with st.expander("Level 2 - Most Consolidated"):
                for tbl in (pl_lv2, bs_lv2, cfs_lv2, summary_df2):
                    st.write(tbl, use_container_width=True)

        else:
            pass
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
        # choice of streamlit
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
            analyse_type = st.selectbox("Select Analysis Type",['Single Business Unit','Consolidated'])

        if analyse_type == 'Single Business Unit':
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

            with st.expander("Level 0 - Most Detailed"):
                for tbl in (pl_sorted, bs_lv0, cfs_df, summary_df):
                    st.write(tbl, use_container_width=True)

            with st.expander("Level 1 - Cosolidated"):
                for tbl in (pl_lv1, bs_lv1, cfs_lv1, summary_df1):
                    st.write(tbl, use_container_width=True)

            with st.expander("Level 2 - Most Consolidated"):
                for tbl in (pl_lv2, bs_lv2, cfs_lv2, summary_df2):
                    st.write(tbl, use_container_width=True)

        else:
            pass
