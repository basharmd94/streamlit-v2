import streamlit as st
import pandas as pd,numpy as np
from .analytics import Analytics, basket_prepare
from modules.data_process_files import common,overall_sales, overall_margin, collection, purchase, financial
from modules.visualization_files import common_v
from datetime import datetime
pd.set_option('display.float_format', '{:.2f}'.format)
from io import BytesIO
import calendar
from utils.utils import timed
from typing import Dict, List, Tuple




@timed
def render_top_download_buttons(data_dict: dict):
    """
    Render right-aligned, side-by-side download buttons for each non-empty table in data_dict.
    """
    # Filter only non-empty DataFrames
    valid_items = [(name, df) for name, df in data_dict.items() if df is not None and not df.empty]

    if not valid_items:
        return  # Nothing to render
    total_buttons = len(valid_items)

    # Add empty space columns for right alignment
    cols = st.columns([8.0] + [1.0] * total_buttons)

    # Place buttons on the right side
    for i, (table_name, df) in enumerate(valid_items):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer) as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        buffer.seek(0)

        with cols[i + 1]:  # First column is spacer, skip it
            st.download_button(
                label=f"📥 {table_name.capitalize()}",
                data=buffer,
                file_name=f"{table_name}_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_{table_name}"
            )

@timed
def safe_month_to_num(x):
    try:
        return int(x)
    except:
        try:
            return list(calendar.month_name).index(x) if x in calendar.month_name else None
        except:
            return None

@timed
def display_overall_sales_analysis_page(current_page, zid, data_dict):
    st.title("Overall Sales Analysis")
    filtered_data, filtered_data_r = common.data_copy_add_columns(data_dict['sales'], data_dict['return'])
    analysis_mode = st.radio("Choose Analysis Mode:",["Overview", "Comparison", "Distributions", "Descriptive Stats"],horizontal=True)

    if analysis_mode == "Overview":
        st.subheader("📈 Select Plot Type")
        plot_options = [
            'Net Sales',
            'Total Returns',
            'Total Discounts',
            'Number of Orders',
            'Number of Returns',
            'Number of Discounts',
            'Number of Customers',
            'Number of Customer Returns',
            'Number of Products',
            'Number of Product Returns'
        ]

        selected_plot = st.selectbox("Choose a plot to display:", plot_options)
        
        # Display selected plot
        if selected_plot == 'Net Sales':
            overall_sales.plot_net(filtered_data, filtered_data_r, ['year', 'month'], 'final_sales', 'treturnamt', 'Net Sales (filtered)', current_page)
        elif selected_plot == 'Number of Orders':
            overall_sales.plot_number_of_orders(filtered_data, current_page)
        elif selected_plot == 'Number of Returns':
            overall_sales.plot_number_of_returns(filtered_data_r, current_page)
        elif selected_plot == 'Number of Customers':
            overall_sales.plot_number_of_customers(filtered_data, current_page)
        elif selected_plot == 'Number of Customer Returns':
            overall_sales.plot_number_of_customer_returns(filtered_data_r, current_page)
        elif selected_plot == 'Number of Products':
            overall_sales.plot_number_of_products(filtered_data, current_page)
        elif selected_plot == 'Number of Product Returns':
            overall_sales.plot_number_of_product_returns(filtered_data_r, current_page)
        elif selected_plot == 'Total Returns':
            overall_sales.plot_total_returns(filtered_data_r, current_page)
        elif selected_plot == 'Total Discounts':
            overall_sales.plot_total_discounts(filtered_data, current_page)
        elif selected_plot == 'Number of Discounts':
            overall_sales.plot_number_of_discounts(filtered_data, current_page)
        
        # Summary ratios
        st.subheader("📊 Sales Performance Ratios")
        performance_ratios = overall_sales.prepare_sales_performance_ratios(
            filtered_data, filtered_data_r
        )
        st.write(performance_ratios, use_container_width=True)

        # Summary stats
        summary_stats = overall_sales.calculate_summary_statistics(filtered_data, filtered_data_r)
        overall_sales.display_summary_statistics(summary_stats)
        
        # Expandable section for pivot tables
        overall_sales.display_entity_metric_pivot(filtered_data, filtered_data_r, current_page)

        overall_sales.display_cross_relation_pivot(filtered_data, filtered_data_r, current_page)
            
    elif analysis_mode == "Comparison":
        st.subheader("🧭 Compare Multiple Entities")

        # load years from main list

        all_years = sorted(filtered_data["year"].dropna().unique().astype(int).tolist())
        # Step 2: Metric and Compare By — in single line
       
        
        compare_type = st.selectbox("Compare Across", ["Year-over-Year (YOY)", "Month vs Month"])
        if compare_type == "Year-over-Year (YOY)":
            st.subheader("📅 Year-over-Year (YOY) Comparison")
            selected_years = st.multiselect("Select Years", all_years, default=all_years)  
            granularity = st.selectbox("Group By", ["Monthly", "Daily", "Day of Week", "Day of Month"])
            if granularity == "Monthly":
                all_months = sorted(filtered_data["month"].dropna().unique().astype(int).tolist())
                month_map = {i: calendar.month_abbr[i] for i in all_months}
                selected_months = st.multiselect("Select Months", [month_map[m] for m in all_months], default=[month_map[m] for m in all_months])
            elif granularity == "Daily":
                latest_year = max(all_years) if selected_years else datetime.now().year
                default_start = datetime(latest_year, 1, 1)
                default_end = datetime(latest_year, 1, 30)
                start_date, end_date = st.date_input("Select Date Range", value=[default_start, default_end])
            elif granularity == "Day of Week":
                average_or_total_YOY_DOW = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
            elif granularity == "Day of Month":
                all_months = sorted(filtered_data["month"].dropna().unique().astype(int).tolist())
                month_map = {i: calendar.month_abbr[i] for i in all_months}
                selected_months = st.multiselect("Select Months", [month_map[m] for m in all_months], default=[month_map[m] for m in all_months])
                average_or_total_YOY_DOM = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
        elif compare_type == "Month vs Month":
            st.subheader("📅 Month vs Month Comparison")
            granularity = st.selectbox("Group By", ["Monthly", "Day of Week", "Day of Month"])
            filtered_data["month_label"] = (filtered_data["month"].astype(int).apply(lambda x: f"{x:02d}")+ "-" + filtered_data["year"].astype(str))
            month_options = sorted(filtered_data["month_label"].dropna().unique().tolist())
            selected_months = st.multiselect("Select Months to Compare", options=month_options, default=month_options[:3])
            if granularity == 'Day of Week':
                average_or_total_MOM_DOW = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
            elif granularity == 'Day of Month':
                average_or_total_MOM_DOM = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
                day_options = list(range(1, 32))  # 1 to 31
                selected_dom_days = st.multiselect("Select Days (leave empty to include all days)",options=day_options)

        compare_by = st.selectbox("Compare By", ["Product", "Product Group", "Salesman", "Customer", "Area"])
        metric = st.selectbox("Metric", [
                "Net Sales", "Total Sales", "Number of Orders", "Total Returns", 
                "Number of Returns", "Number of Products", "Total Product Discounts", 
                "Number of Product Discounts", "Number of Customers"
        ])

        # Step 3: Build entity list
        dimension_column_map = {
            "Product": ("itemcode", "itemname"),
            "Product Group": ("itemgroup", None),
            "Customer": ("cusid", "cusname"),
            "Salesman": ("spid", "spname"),
            "Area": ("area", None)
        }
        code_col, name_col = dimension_column_map[compare_by]

        # Only single item selection
        if name_col:
            filtered_sub = filtered_data[[code_col, name_col]].dropna().drop_duplicates()
            filtered_sub["combined"] = filtered_sub[code_col].astype(str) + " - " + filtered_sub[name_col].astype(str)
            display_options = sorted(filtered_sub["combined"].tolist())
        else:
            display_options = sorted(filtered_data[code_col].dropna().unique().tolist())

        

        # Plot if metric + compare selected
        if compare_type == "Year-over-Year (YOY)" and granularity == "Monthly":
            selected_display = st.selectbox(f"Select {compare_by} to Filter (optional)", options=["(All)"] + display_options)
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []
            overall_sales.plot_yoy_monthly_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                selected_month_names=selected_months
            )
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Daily":
            selected_display = st.selectbox(f"Select {compare_by} to Filter (optional)", options=["(All)"] + display_options)
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []
            overall_sales.plot_yoy_daily_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                start_date=start_date,
                end_date=end_date
            )
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Day of Week":
            selected_display = st.selectbox(f"Select {compare_by} to Filter (optional)", options=["(All)"] + display_options)
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []
            overall_sales.plot_yoy_dow_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                average_or_total=average_or_total_YOY_DOW
            )
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Day of Month":
            selected_display = st.selectbox(f"Select {compare_by} to Filter (optional)", options=["(All)"] + display_options)
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []
            overall_sales.plot_yoy_dom_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                selected_month_names=selected_months,
                average_or_total=average_or_total_YOY_DOM
            )
        elif compare_type == "Month vs Month" and granularity == "Monthly":
            selected_display = st.multiselect(f"Select {compare_by}(s) to Compare", options=display_options, default=display_options[:3], max_selections=7)
            if name_col:
                selected_codes = [x.split(" - ")[0] for x in selected_display]
            else:
                selected_codes = selected_display
            
            overall_sales.plot_month_vs_month_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                name_col=name_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_months=selected_months
            )
        elif compare_type == "Month vs Month" and granularity == "Day of Week":
            selected_display = st.multiselect(f"Select {compare_by}(s) to Compare", options=display_options, default=display_options[:3], max_selections=7)
            if name_col:
                selected_codes = [x.split(" - ")[0] for x in selected_display]
            else:
                selected_codes = selected_display
            
            overall_sales.plot_month_vs_month_dow_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                name_col=name_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_months=selected_months,
                aggregation_type=average_or_total_MOM_DOW
            )
        elif compare_type == "Month vs Month" and granularity == "Day of Month":
            selected_display = st.multiselect(f"Select {compare_by}(s) to Compare", options=display_options, default=display_options[:3], max_selections=7)
            if name_col:
                selected_codes = [x.split(" - ")[0] for x in selected_display]
            else:
                selected_codes = selected_display
            
            overall_sales.plot_month_vs_month_dom_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                name_col=name_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_months=selected_months,
                aggregation_type=average_or_total_MOM_DOM,
                selected_days=selected_dom_days  # Optional
            )
        else:
            st.warning("Please select at least one year and month.")

    elif analysis_mode == "Distributions":
        st.subheader("📊 Distribution Analysis")

           
        # Metric, Group, and Date Range
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_metric = st.selectbox("Metric", [
                "Net Sales", "Total Sales", "Total Returns", "Product Discounts",
                "Number of Orders", "Number of Returns", "Number of Customers", "Number of Products"
            ])
        with col2:
            selected_group = st.selectbox("Group By", ["Customer", "Product", "Salesman", "Area", "Product Group","Day of Month", "Day of Week"])
        with col3:
            min_date = filtered_data["date"].min()
            max_date = filtered_data["date"].max()
            start_date, end_date = st.date_input("Date Range", value=(min_date, max_date))

        # Value Filters and Bin Selection
        col_min, col_max, col_bins, col_button = st.columns(4)
        with col_min:
            value_min = st.number_input("Min Value (optional)", value=None, placeholder="e.g. 1000")
        with col_max:
            value_max = st.number_input("Max Value (optional)", value=None, placeholder="e.g. 50000")
        with col_bins:
            nbins = st.number_input("Number of Bins", min_value=1, max_value=500, value=100)
        with col_button:
            st.markdown("<br>", unsafe_allow_html=True)
            run_distribution = st.button("Generate Distribution")

        if run_distribution:
            filtered_data_d = filtered_data[
                (filtered_data["date"] >= pd.to_datetime(start_date)) &
                (filtered_data["date"] <= pd.to_datetime(end_date))
            ]
            filtered_data_r_d = filtered_data_r[
                (filtered_data_r["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_r["date"] <= pd.to_datetime(end_date))
            ]

            overall_sales.plot_distribution_analysis(
                filtered_data=filtered_data_d,
                filtered_data_r=filtered_data_r_d,
                metric=selected_metric,
                group_by=selected_group,
                value_min=value_min,
                value_max=value_max,
                nbins=nbins
            )

    elif analysis_mode == "Descriptive Stats":
        st.subheader("📐 Summary Statistics")
        
        # Date Range Selector
        min_date = filtered_data["date"].min()
        max_date = filtered_data["date"].max()
        start_date, end_date = st.date_input("Select Date Range", value=(min_date, max_date))

        group_by = st.selectbox("Group By", [
            "Customer", "Product", "Salesman", "Area", "Product Group",
            "Month", "Year", "Day of Month", "Day of Week"
        ])

        if st.button("Generate Summary Statistics"):
            # Filter both datasets
            filtered_data_d = filtered_data[
                (filtered_data["date"] >= pd.to_datetime(start_date)) &
                (filtered_data["date"] <= pd.to_datetime(end_date))
            ]
            filtered_data_r_d = filtered_data_r[
                (filtered_data_r["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_r["date"] <= pd.to_datetime(end_date))
            ]

            # Compute and display
            stats_df = overall_sales.generate_descriptive_statistics(filtered_data_d, filtered_data_r_d, group_by)
            st.markdown("### 📋 Summary Table")
            st.dataframe(stats_df, use_container_width=True)

@timed
def display_margin_analysis_page(current_page, zid, data_dict):
    st.sidebar.title("Overall Margin Analysis")
    filtered_data,filtered_data_r = common.data_copy_add_columns(data_dict['sales'], data_dict['return'])
    analysis_mode = st.radio("Choose Analysis Mode:",["Overview","Comparison","Distributions","Descriptive Stats","Metric Comparison"],horizontal=True)

    if analysis_mode == "Overview":
        st.subheader("📈 Select Plot Type")
        plot_options = [
            'Net Sales',
            'Net Returns',
            'Net Discounts',
            'Net Margin',
        ]

        selected_plot = st.selectbox("Choose a plot to display:", plot_options)
        
        # Display selected plot
        if selected_plot == 'Net Sales':
            overall_margin.plot_net(filtered_data, filtered_data_r, ['year', 'month'], 'final_sales', 'treturnamt', 'Net Sales (filtered)', current_page)
        elif selected_plot == 'Net Returns':
            overall_margin.plot_total_returns(filtered_data_r, current_page)
        elif selected_plot == 'Net Discounts':
            overall_margin.plot_total_discounts(filtered_data, current_page)
        elif selected_plot == 'Net Margin':
            overall_margin.plot_net_margin(filtered_data, filtered_data_r, ['year', 'month'], 'gross_margin', 'treturnamt', 'Net Margin (filtered)', current_page)
        
        # Summary stats
        summary_stats = overall_margin.calculate_summary_statistics(filtered_data, filtered_data_r)
        overall_margin.display_summary_statistics(summary_stats)

        # Expandable section for pivot tables
        overall_margin.display_entity_metric_pivot(filtered_data, filtered_data_r, current_page)

        overall_margin.display_cross_relation_pivot(filtered_data, filtered_data_r, current_page)
            
    elif analysis_mode == "Comparison":
        st.subheader("🧭 Compare Multiple Entities")

        # load years from main list

        all_years = sorted(filtered_data["year"].dropna().unique().astype(int).tolist())
        # Step 2: Metric and Compare By — in single line
       
        
        compare_type = st.selectbox("Compare Across", ["Year-over-Year (YOY)", "Month vs Month"])
        if compare_type == "Year-over-Year (YOY)":
            st.subheader("📅 Year-over-Year (YOY) Comparison")
            selected_years = st.multiselect("Select Years", all_years, default=all_years)  
            granularity = st.selectbox("Group By", ["Monthly", "Daily", "Day of Week", "Day of Month"])
            if granularity == "Monthly":
                all_months = sorted(filtered_data["month"].dropna().unique().astype(int).tolist())
                month_map = {i: calendar.month_abbr[i] for i in all_months}
                selected_months = st.multiselect("Select Months", [month_map[m] for m in all_months], default=[month_map[m] for m in all_months])
            elif granularity == "Daily":
                latest_year = max(all_years) if selected_years else datetime.now().year
                default_start = datetime(latest_year, 1, 1)
                default_end = datetime(latest_year, 1, 30)
                start_date, end_date = st.date_input("Select Date Range", value=[default_start, default_end])
            elif granularity == "Day of Week":
                average_or_total_YOY_DOW = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
            elif granularity == "Day of Month":
                all_months = sorted(filtered_data["month"].dropna().unique().astype(int).tolist())
                month_map = {i: calendar.month_abbr[i] for i in all_months}
                selected_months = st.multiselect("Select Months", [month_map[m] for m in all_months], default=[month_map[m] for m in all_months])
                average_or_total_YOY_DOM = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
        elif compare_type == "Month vs Month":
            st.subheader("📅 Month vs Month Comparison")
            granularity = st.selectbox("Group By", ["Monthly", "Day of Week", "Day of Month"])
            filtered_data["month_label"] = (filtered_data["month"].astype(int).apply(lambda x: f"{x:02d}")+ "-" + filtered_data["year"].astype(str))
            month_options = sorted(filtered_data["month_label"].dropna().unique().tolist())
            selected_months = st.multiselect("Select Months to Compare", options=month_options, default=month_options[:3])
            if granularity == 'Day of Week':
                average_or_total_MOM_DOW = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
            elif granularity == 'Day of Month':
                average_or_total_MOM_DOM = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
                day_options = list(range(1, 32))  # 1 to 31
                selected_dom_days = st.multiselect("Select Days (leave empty to include all days)",options=day_options)

        compare_by = st.selectbox("Compare By", ["Product", "Product Group", "Salesman", "Customer", "Area"])
        metric = st.selectbox("Metric", [
                "Net Sales", 
                "Total Returns", 
                "Total Discounts", 
                "Net Margin"
        ])

        # Step 3: Build entity list
        dimension_column_map = {
            "Product": ("itemcode", "itemname"),
            "Product Group": ("itemgroup", None),
            "Customer": ("cusid", "cusname"),
            "Salesman": ("spid", "spname"),
            "Area": ("area", None)
        }
        
        code_col, name_col = dimension_column_map[compare_by]

        # Only single item selection
        if name_col:
            filtered_sub = filtered_data[[code_col, name_col]].dropna().drop_duplicates()
            filtered_sub["combined"] = filtered_sub[code_col].astype(str) + " - " + filtered_sub[name_col].astype(str)
            display_options = sorted(filtered_sub["combined"].tolist())
        else:
            display_options = sorted(filtered_data[code_col].dropna().unique().tolist())

        

        # Plot if metric + compare selected
        if compare_type == "Year-over-Year (YOY)" and granularity == "Monthly":
            selected_display = st.selectbox(f"Select {compare_by} to Filter (optional)", options=["(All)"] + display_options)
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []
            overall_margin.plot_yoy_monthly_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                selected_month_names=selected_months
            )
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Daily":
            selected_display = st.selectbox(f"Select {compare_by} to Filter (optional)", options=["(All)"] + display_options)
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []
            overall_margin.plot_yoy_daily_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                start_date=start_date,
                end_date=end_date
            )
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Day of Week":
            selected_display = st.selectbox(f"Select {compare_by} to Filter (optional)", options=["(All)"] + display_options)
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []
            overall_margin.plot_yoy_dow_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                average_or_total=average_or_total_YOY_DOW
            )
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Day of Month":
            selected_display = st.selectbox(f"Select {compare_by} to Filter (optional)", options=["(All)"] + display_options)
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []
            overall_margin.plot_yoy_dom_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                selected_month_names=selected_months,
                average_or_total=average_or_total_YOY_DOM
            )
        elif compare_type == "Month vs Month" and granularity == "Monthly":
            selected_display = st.multiselect(f"Select {compare_by}(s) to Compare", options=display_options, default=display_options[:3], max_selections=7)
            if name_col:
                selected_codes = [x.split(" - ")[0] for x in selected_display]
            else:
                selected_codes = selected_display
            
            overall_margin.plot_month_vs_month_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                name_col=name_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_months=selected_months
            )
        elif compare_type == "Month vs Month" and granularity == "Day of Week":
            selected_display = st.multiselect(f"Select {compare_by}(s) to Compare", options=display_options, default=display_options[:3], max_selections=7)
            if name_col:
                selected_codes = [x.split(" - ")[0] for x in selected_display]
            else:
                selected_codes = selected_display
            
            overall_margin.plot_month_vs_month_dow_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                name_col=name_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_months=selected_months,
                aggregation_type=average_or_total_MOM_DOW
            )
        elif compare_type == "Month vs Month" and granularity == "Day of Month":
            selected_display = st.multiselect(f"Select {compare_by}(s) to Compare", options=display_options, default=display_options[:3], max_selections=7)
            if name_col:
                selected_codes = [x.split(" - ")[0] for x in selected_display]
            else:
                selected_codes = selected_display
            
            overall_margin.plot_month_vs_month_dom_comparison(
                filtered_data=filtered_data,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                name_col=name_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_months=selected_months,
                aggregation_type=average_or_total_MOM_DOM,
                selected_days=selected_dom_days  # Optional
            )
        else:
            st.warning("Please select at least one year and month.")

    elif analysis_mode == "Distributions":
        st.subheader("📊 Distribution Analysis")

           
        # Metric, Group, and Date Range
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_metric = st.selectbox("Metric", [
                "Net Sales", 
                "Total Returns",
                "Total Discounts",
                "Net Margin"
            ])
        with col2:
            selected_group = st.selectbox("Group By", ["Customer", "Product", "Salesman", "Area", "Product Group","Day of Month", "Day of Week"])
        with col3:
            min_date = filtered_data["date"].min()
            max_date = filtered_data["date"].max()
            start_date, end_date = st.date_input("Date Range", value=(min_date, max_date))

        # Value Filters and Bin Selection
        col_min, col_max, col_bins, col_button = st.columns(4)
        with col_min:
            value_min = st.number_input("Min Value (optional)", value=None, placeholder="e.g. 1000")
        with col_max:
            value_max = st.number_input("Max Value (optional)", value=None, placeholder="e.g. 50000")
        with col_bins:
            nbins = st.number_input("Number of Bins", min_value=1, max_value=500, value=100)
        with col_button:
            st.markdown("<br>", unsafe_allow_html=True)
            run_distribution = st.button("Generate Distribution")

        if run_distribution:
            filtered_data_d = filtered_data[
                (filtered_data["date"] >= pd.to_datetime(start_date)) &
                (filtered_data["date"] <= pd.to_datetime(end_date))
            ]
            filtered_data_r_d = filtered_data_r[
                (filtered_data_r["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_r["date"] <= pd.to_datetime(end_date))
            ]

            overall_margin.plot_distribution_analysis(
                filtered_data=filtered_data_d,
                filtered_data_r=filtered_data_r_d,
                metric=selected_metric,
                group_by=selected_group,
                value_min=value_min,
                value_max=value_max,
                nbins=nbins
            )

    elif analysis_mode == "Descriptive Stats":
        st.subheader("📐 Summary Statistics")
        
        # Date Range Selector
        min_date = filtered_data["date"].min()
        max_date = filtered_data["date"].max()
        start_date, end_date = st.date_input("Select Date Range", value=(min_date, max_date))

        group_by = st.selectbox("Group By", [
            "Customer", "Product", "Salesman", "Area", "Product Group",
            "Month", "Year", "Day of Month", "Day of Week"
        ])

        if st.button("Generate Summary Statistics"):
            # Filter both datasets
            filtered_data_d = filtered_data[
                (filtered_data["date"] >= pd.to_datetime(start_date)) &
                (filtered_data["date"] <= pd.to_datetime(end_date))
            ]
            filtered_data_r_d = filtered_data_r[
                (filtered_data_r["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_r["date"] <= pd.to_datetime(end_date))
            ]

            # Compute and display
            stats_df = overall_margin.generate_descriptive_statistics(filtered_data_d, filtered_data_r_d, group_by)
            st.markdown("### 📋 Summary Table")
            st.dataframe(stats_df, use_container_width=True)

    elif analysis_mode == "Metric Comparison":
        st.subheader("Compare Metrics")
        
        compare_by = st.selectbox("Compare By", ["Product", "Product Group", "Salesman", "Customer", "Area"])

        metric_choices = [
            "Net Sales", "Total Returns", "Total Discounts", "Net Margin"
        ]

        col1, col2 = st.columns(2)
        with col1:
            metric_x = st.selectbox("Metric A (Base)", metric_choices, index=0)
        with col2:
            metric_y = st.selectbox("Metric B (Compared Against A)", metric_choices, index=3)

        all_years = sorted(filtered_data["year"].dropna().unique().astype(int).tolist())
        all_months = sorted(filtered_data["month"].dropna().unique().astype(int).tolist())
        month_map = {i: calendar.month_abbr[i] for i in all_months}

        selected_years = st.multiselect("Select Years", all_years, default=all_years)
        selected_months = st.multiselect("Select Months", [month_map[m] for m in all_months], default=[month_map[m] for m in all_months])

        dimension_column_map = {
            "Product": ("itemcode", "itemname"),
            "Product Group": ("itemgroup", None),
            "Customer": ("cusid", "cusname"),
            "Salesman": ("spid", "spname"),
            "Area": ("area", None)
        }
        
        code_col, name_col = dimension_column_map[compare_by]

        # Only single item for focused view
        if name_col:
            filtered_sub = filtered_data[[code_col, name_col]].dropna().drop_duplicates()
            filtered_sub["combined"] = filtered_sub[code_col].astype(str) + " - " + filtered_sub[name_col].astype(str)
            display_options = sorted(filtered_sub["combined"].tolist())
        else:
            display_options = sorted(filtered_data[code_col].dropna().unique().tolist())

        selected_display = st.selectbox(f"Select {compare_by} to View", options=["(All)"] + display_options)
        selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []

        overall_margin.plot_metric_comparison_monthly(
            filtered_data=filtered_data,
            filtered_data_r=filtered_data_r,
            code_col=code_col,
            selected_codes=selected_codes,
            metric_x=metric_x,
            metric_y=metric_y,
            selected_years=selected_years,
            selected_month_names=selected_months
        )

@timed
def display_collection_analysis_page(current_page, zid, project, data_dict):
    st.sidebar.title("Overall Collection Analysis")

    #collection using sales, returns and collection separately
    filtered_data_c, filtered_data_s,filtered_data_r = common.data_copy_add_columns(data_dict['collection'],data_dict['sales'], data_dict['return'])
    filtered_data_c = common.enrich_collection_with_sales_info(filtered_data_c, filtered_data_s)
    analysis_mode = st.radio("Choose Analysis Mode:",["Overview","Comparison","Distributions","Descriptive Stats","Metric Comparison","CP","CPA","Customer Ledger"],horizontal=True)

    #collection using glheader and details. 
    filtered_data_ar = data_dict['ar']
    filtered_data_ar = filtered_data_ar[filtered_data_ar['project'] == project]
    filtered_data_ar = common.add_voucher_type_ar(filtered_data_ar)
    # st.write(common.create_download_link(filtered_data_ar,"filtered_data_ar.xlsx"), unsafe_allow_html=True)
    

    if analysis_mode == "Overview":
        st.subheader("📈 Select Plot Type")
        plot_options = [
            'Net Sales',
            'Net Returns',
            'Net Discounts',
            'Collections'
        ]

        selected_plot = st.selectbox("Choose a plot to display:", plot_options)
        
        # Display selected plot
        if selected_plot == 'Net Sales':
            collection.plot_net(filtered_data_s, filtered_data_r, ['year', 'month'], 'final_sales', 'treturnamt', 'Net Sales (filtered)', current_page)
        elif selected_plot == 'Net Returns':
            collection.plot_total_returns(filtered_data_r, current_page)
        elif selected_plot == 'Net Discounts':
            collection.plot_total_discounts(filtered_data_s, current_page)
        elif selected_plot == 'Collections':
            collection.plot_collection(filtered_data_c, current_page)
        
        # Summary stats
        summary_stats = collection.calculate_summary_statistics(filtered_data_c,filtered_data_s, filtered_data_r)
        collection.display_summary_statistics(summary_stats)

        # Expandable section for pivot tables
        collection.display_entity_metric_pivot(filtered_data_c, filtered_data_s, filtered_data_r, current_page)

        collection.display_cross_relation_pivot(filtered_data_c, filtered_data_s, filtered_data_r, current_page)

    elif analysis_mode == "Comparison":
        st.subheader("🧭 Compare Multiple Entities")

        all_years = sorted(filtered_data_c["year"].dropna().unique().astype(int).tolist())
        # Step 2: Metric and Compare By — in single line
        compare_type = st.selectbox("Compare Across", ["Year-over-Year (YOY)", "Month vs Month"])
        if compare_type == "Year-over-Year (YOY)":
            st.subheader("📅 Year-over-Year (YOY) Comparison")
            selected_years = st.multiselect("Select Years", all_years, default=all_years)  
            granularity = st.selectbox("Group By", ["Monthly", "Daily", "Day of Week", "Day of Month"])
            if granularity == "Monthly":
                # filtered_data_c["month_numeric"] = filtered_data_c["month"].apply(safe_month_to_num)
                all_months = sorted(filtered_data_c["month"].dropna().unique().astype(int).tolist())
                month_map = {i: calendar.month_abbr[i] for i in all_months}
                selected_months = st.multiselect("Select Months", [month_map[m] for m in all_months], default=[month_map[m] for m in all_months])
            elif granularity == "Daily":
                latest_year = max(all_years) if selected_years else datetime.now().year
                default_start = datetime(latest_year, 1, 1)
                default_end = datetime(latest_year, 1, 30)
                start_date, end_date = st.date_input("Select Date Range", value=[default_start, default_end])
            elif granularity == "Day of Week":
                average_or_total_YOY_DOW = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
            elif granularity == "Day of Month":
                # filtered_data_c["month_numeric"] = filtered_data_c["month"].apply(safe_month_to_num)
                all_months = sorted(filtered_data_c["month"].dropna().unique().astype(int).tolist())
                month_map = {i: calendar.month_abbr[i] for i in all_months}
                selected_months = st.multiselect("Select Months", [month_map[m] for m in all_months], default=[month_map[m] for m in all_months])
                average_or_total_YOY_DOM = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
        elif compare_type == "Month vs Month":
            st.subheader("📅 Month vs Month Comparison")
            granularity = st.selectbox("Group By", ["Monthly", "Day of Week", "Day of Month"])
            filtered_data_c["month_label"] = (filtered_data_c["month"].astype(int).apply(lambda x: f"{x:02d}")+ "-" + filtered_data_c["year"].astype(str))
            month_options = sorted(filtered_data_c["month_label"].dropna().unique().tolist())
            selected_months = st.multiselect("Select Months to Compare", options=month_options, default=month_options[:3])
            if granularity == 'Day of Week':
                average_or_total_MOM_DOW = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
            elif granularity == 'Day of Month':
                average_or_total_MOM_DOM = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
                day_options = list(range(1, 32))  # 1 to 31
                selected_dom_days = st.multiselect("Select Days (leave empty to include all days)",options=day_options)

        # 4. Compare By
        compare_by = st.selectbox("Compare By", ["Product", "Product Group", "Salesman", "Customer", "Area"])

        # 5. Metric Choices (dynamic based on compare_by)
        full_metric_options = [
            "Net Sales", 
            "Total Returns", 
            "Total Discounts", 
            "Collection"
        ]

        if compare_by in ["Product", "Product Group"]:
            metric_options = [
                "Net Sales", 
                "Total Returns", 
                "Total Discounts", 
            ]
        else:
            metric_options = full_metric_options

        metric = st.selectbox("Metric", metric_options)

        # 6. Build Entity List (codes and names)
        dimension_column_map = {
            "Product": ("itemcode", "itemname"),
            "Product Group": ("itemgroup", None),
            "Customer": ("cusid", "cusname"),
            "Salesman": ("spid", "spname"),
            "Area": ("area", None)
        }
        code_col, name_col = dimension_column_map[compare_by]

        if name_col:
            filtered_sub = filtered_data_s[[code_col, name_col]].dropna().drop_duplicates()
            filtered_sub["combined"] = filtered_sub[code_col].astype(str) + " - " + filtered_sub[name_col].astype(str)
            display_options = sorted(filtered_sub["combined"].tolist())
        else:
            display_options = sorted(filtered_data_s[code_col].dropna().unique().tolist())

        # Plot if metric + compare selected
        if compare_type == "Year-over-Year (YOY)" and granularity == "Monthly":
            selected_display = st.selectbox(f"Select {compare_by} to Filter (optional)", options=["(All)"] + display_options)
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []
            collection.plot_yoy_monthly_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                selected_month_names=selected_months
            )
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Daily":
            selected_display = st.selectbox(f"Select {compare_by} to Filter (optional)", options=["(All)"] + display_options)
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []
            collection.plot_yoy_daily_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                start_date=start_date,
                end_date=end_date
            )
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Day of Week":
            selected_display = st.selectbox(f"Select {compare_by} to Filter (optional)", options=["(All)"] + display_options)
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []
            collection.plot_yoy_dow_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                average_or_total=average_or_total_YOY_DOW
            )
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Day of Month":
            selected_display = st.selectbox(f"Select {compare_by} to Filter (optional)", options=["(All)"] + display_options)
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []
            collection.plot_yoy_dom_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                selected_month_names=selected_months,
                average_or_total=average_or_total_YOY_DOM
            )
        elif compare_type == "Month vs Month" and granularity == "Monthly":
            selected_display = st.multiselect(f"Select {compare_by}(s) to Compare", options=display_options, default=display_options[:3], max_selections=7)
            if name_col:
                selected_codes = [x.split(" - ")[0] for x in selected_display]
            else:
                selected_codes = selected_display
            
            collection.plot_month_vs_month_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                name_col=name_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_months=selected_months
            )
        elif compare_type == "Month vs Month" and granularity == "Day of Week":
            selected_display = st.multiselect(f"Select {compare_by}(s) to Compare", options=display_options, default=display_options[:3], max_selections=7)
            if name_col:
                selected_codes = [x.split(" - ")[0] for x in selected_display]
            else:
                selected_codes = selected_display
            
            collection.plot_month_vs_month_dow_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                name_col=name_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_months=selected_months,
                aggregation_type=average_or_total_MOM_DOW
            )
        elif compare_type == "Month vs Month" and granularity == "Day of Month":
            selected_display = st.multiselect(f"Select {compare_by}(s) to Compare", options=display_options, default=display_options[:3], max_selections=7)
            if name_col:
                selected_codes = [x.split(" - ")[0] for x in selected_display]
            else:
                selected_codes = selected_display
            
            collection.plot_month_vs_month_dom_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                name_col=name_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_months=selected_months,
                aggregation_type=average_or_total_MOM_DOM,
                selected_days=selected_dom_days  # Optional
            )
        else:
            st.warning("Please select at least one year and month.")
    
    elif analysis_mode == "Distributions":
        st.subheader("📊 Distribution Analysis")

        # Step 1: Metric and Group Choice
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_metric = st.selectbox("Metric", [
                "Net Sales", 
                "Total Returns",
                "Total Discounts",
                "Collection"
            ], key="dist_metric")

        with col2:
            # Dynamically adjust group options
            if selected_metric == "Collection":
                group_options = ["Salesman", "Customer", "Area", "Day of Month", "Day of Week"]
            else:
                group_options = ["Customer", "Product", "Salesman", "Area", "Product Group", "Day of Month", "Day of Week"]

            selected_group = st.selectbox("Group By", group_options, key="dist_group")

        with col3:
            min_date = filtered_data_c["date"].min()
            max_date = filtered_data_c["date"].max()
            start_date, end_date = st.date_input("Date Range", value=(min_date, max_date))

        # Step 2: Value Range and Bins
        col_min, col_max, col_bins, col_button = st.columns(4)
        with col_min:
            value_min = st.number_input("Min Value (optional)", value=None, placeholder="e.g. 1000", key="dist_min")
        with col_max:
            value_max = st.number_input("Max Value (optional)", value=None, placeholder="e.g. 50000", key="dist_max")
        with col_bins:
            nbins = st.number_input("Number of Bins", min_value=1, max_value=500, value=100, key="dist_bins")
        with col_button:
            st.markdown("<br>", unsafe_allow_html=True)
            run_distribution = st.button("Generate Distribution", key="dist_button")

        # Step 3: Run Distribution on Button Click
        if run_distribution:
            filtered_data_d = filtered_data_s[
                (filtered_data_s["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_s["date"] <= pd.to_datetime(end_date))
            ]
            filtered_data_r_d = filtered_data_r[
                (filtered_data_r["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_r["date"] <= pd.to_datetime(end_date))
            ]
            filtered_data_c_d = filtered_data_c[
                (filtered_data_c["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_c["date"] <= pd.to_datetime(end_date))
            ]

            collection.plot_distribution_analysis(
                filtered_data_s=filtered_data_d,
                filtered_data_r=filtered_data_r_d,
                filtered_data_c=filtered_data_c_d,  # ✅ Pass collection data also
                metric=selected_metric,
                group_by=selected_group,
                value_min=value_min,
                value_max=value_max,
                nbins=nbins
            )

    elif analysis_mode == "Descriptive Stats":
        st.subheader("📐 Summary Statistics")

        # Date Range Selector
        min_date = filtered_data_c["date"].min()
        max_date = filtered_data_c["date"].max()
        start_date, end_date = st.date_input("Select Date Range", value=(min_date, max_date))

        group_by = st.selectbox("Group By", [
            "Customer", "Product", "Salesman", "Area", "Product Group",
            "Month", "Year", "Day of Month", "Day of Week"
        ])

        if st.button("Generate Summary Statistics"):
            # Filter datasets by Date
            filtered_data_d = filtered_data_s[
                (filtered_data_s["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_s["date"] <= pd.to_datetime(end_date))
            ]
            filtered_data_r_d = filtered_data_r[
                (filtered_data_r["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_r["date"] <= pd.to_datetime(end_date))
            ]
            filtered_data_c_d = filtered_data_c[
                (filtered_data_c["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_c["date"] <= pd.to_datetime(end_date))
            ]

            # Compute and display
            stats_df = collection.generate_descriptive_statistics(filtered_data_d, filtered_data_r_d, filtered_data_c_d, group_by)
            st.markdown("### 📋 Summary Table")
            st.dataframe(stats_df, use_container_width=True)

    elif analysis_mode == "Metric Comparison":
        st.subheader("Compare Metrics")

        # 1. Compare By Entity
        compare_by = st.selectbox("Compare By", ["Product", "Product Group", "Salesman", "Customer", "Area"])

        # 2. Metric Choices based on Compare By
        full_metric_choices = [
            "Net Sales", "Total Returns", "Total Discounts", "Collection"
        ]

        # Restrict Collection to only allowed entities
        if compare_by in ["Product", "Product Group"]:
            metric_choices = [
                "Net Sales", "Total Returns", "Total Discounts"
            ]
        else:
            metric_choices = full_metric_choices

        # 3. Metric A and B Selection
        col1, col2 = st.columns(2)
        with col1:
            metric_x = st.selectbox("Metric A (Base)", metric_choices, index=0)
        with col2:
            metric_y = st.selectbox("Metric B (Compared Against A)", metric_choices, index=3)

        # 4. Year and Month Filters
        all_years = sorted(filtered_data_s["year"].dropna().unique().astype(int).tolist())
        all_months = sorted(filtered_data_s["month"].dropna().unique().astype(int).tolist())
        month_map = {i: calendar.month_abbr[i] for i in all_months}

        selected_years = st.multiselect("Select Years", all_years, default=all_years)
        selected_months = st.multiselect("Select Months", [month_map[m] for m in all_months], default=[month_map[m] for m in all_months])

        # 5. Entity Code Selection
        dimension_column_map = {
            "Product": ("itemcode", "itemname"),
            "Product Group": ("itemgroup", None),
            "Customer": ("cusid", "cusname"),
            "Salesman": ("spid", "spname"),
            "Area": ("area", None)
        }
        code_col, name_col = dimension_column_map[compare_by]

        if name_col:
            filtered_sub = filtered_data_s[[code_col, name_col]].dropna().drop_duplicates()
            filtered_sub["combined"] = filtered_sub[code_col].astype(str) + " - " + filtered_sub[name_col].astype(str)
            display_options = sorted(filtered_sub["combined"].tolist())
        else:
            display_options = sorted(filtered_data_s[code_col].dropna().unique().tolist())

        selected_display = st.selectbox(f"Select {compare_by} to View", options=["(All)"] + display_options)
        selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []

        # 6. 🔥 Call the Plot Function
        collection.plot_metric_comparison_monthly(
            filtered_data_c=filtered_data_c,
            filtered_data_s=filtered_data_s,
            filtered_data_r=filtered_data_r,
            code_col=code_col,
            selected_codes=selected_codes,
            metric_x=metric_x,
            metric_y=metric_y,
            selected_years=selected_years,
            selected_month_names=selected_months
        )

    elif analysis_mode == "CP":
        st.subheader("Collection Performance")
        sales_df = filtered_data_s
        returns_df = filtered_data_r
        collection_df = filtered_data_c

        #filtered options for collection, - shows the filtering options for sales, returns and collections and outputs the dataset after the filter
        sales_df, returns_df, collection_df = collection.filtered_options_for_collection(sales_df, returns_df, collection_df)

        sales_df = sales_df.groupby(['date', 'year', 'month', 'cusid', 'cusname', 'DOM', 'DOW']).final_sales.sum().reset_index()
        returns_df = returns_df.groupby(['date', 'year', 'month', 'cusid', 'cusname', 'DOM', 'DOW']).treturnamt.sum().reset_index()
        collection_df = collection_df.groupby(['date', 'year', 'month', 'cusid', 'cusname', 'DOM', 'DOW']).value.sum().reset_index()
        # Compute average days and other metrics
        avg_days, pivot_df, avg_days_between, combined_df = collection.average_days_to_collection(sales_df, returns_df, collection_df)
        summary_df = collection.customer_segmentation_by_collection_days(avg_days)

        # Display metrics
        for title, df in {
            "Average Days to Collection": avg_days,
            "Customer Segmentation by Days to Collection": summary_df,
            "Collection Days by Year/Month": pivot_df,
            "Average Days to Collection": avg_days_between
        }.items():
            st.markdown(title)
            st.write(df)

        # Step 1: Pull unique customer list from collection_df
        combined_df = combined_df[['year','month','cusid','cusname','date','final_sales','treturnamt','value']]
        customer_options = (
            collection_df[['cusid', 'cusname']]
            .drop_duplicates()
            .sort_values(by='cusname')
        )
        customer_options["combined"] = customer_options["cusid"].astype(str) + " - " + customer_options["cusname"]

        default_customer = customer_options["combined"].iloc[0]
        selected_customer = st.selectbox(
            "Select Customer", 
            options=customer_options["combined"].tolist(),
            index=customer_options["combined"].tolist().index(default_customer)  # Ensure default is selected
        )

        selected_cusid = selected_customer.split(" - ")[0]
        ledger_df = combined_df[combined_df['cusid'] == selected_cusid]
        ledger_df = ledger_df.sort_values(by='date')

        columns_to_hide = ['cusid', 'cusname']  # or any other internal columns you want to omit
        display_df = ledger_df.drop(columns=columns_to_hide)

        st.markdown("Customer Ledger")
        st.write(display_df)

        # Analysis and Metric Selection
        timeframe = st.selectbox('Select Time Range', ['Daily', 'Monthly', 'Yearly'], index=2)
        grouped_df, grouped_df_DOM, grouped_df_DOW = collection.get_grouped_df_collection(sales_df, returns_df, collection_df, timeframe)

        # Display and visualize data
        for title, (df, x_axis) in {
                f"Comparison of Sales/Collection {timeframe}": (grouped_df, 'timeframe'),
                "Comparison of Sales/Collection DOM": (grouped_df_DOM, 'DOM'),
                "Comparison of Sales/Collection DOW": (grouped_df_DOW, 'DOW')
        }.items():
            st.markdown(title)
            df = df.rename(columns={'value':'value_collection'})
            st.write(df)
            # long_df = df.melt(id_vars=[x_axis], value_vars=['final_sales', 'treturnamt', 'value_collection'], var_name='category', value_name='value')
            # common_v.plot_bar_chart(data=long_df, x_axis=x_axis, y_axis='value', color='category', title=f'{x_axis} Collection Analysis')
    
    elif analysis_mode == "CPA":
        st.subheader("Collection Performance Advanced")

        performance_analysis_type = st.radio("Choose Analysis Type:",["Order Timeliness Metrics","Payment Timeliness","Composite Scoring"],horizontal=True)

        if performance_analysis_type == "Order Timeliness Metrics":
            order_df, avg_df, std_df = collection.compute_order_frequency_metrics(filtered_data_ar)

            st.subheader("Order Frequency (Count per Year)")
            st.dataframe(order_df)

            st.subheader("Average Interval Between Orders (Days)")
            st.write("Average number of days between consecutive orders.")
            st.dataframe(avg_df)

            st.subheader("Std. Dev. of Interval Between Orders (Days)")
            st.write("Standard deviation of inter‐order intervals (low sd ⇒ regular orders).")
            st.dataframe(std_df)
        
        elif performance_analysis_type == "Payment Timeliness":
            collection.display_payment_timeliness_page(filtered_data_ar)

        elif performance_analysis_type == "Composite Scoring":
            collection.display_composite_scoring_page(filtered_data_ar)

    elif analysis_mode == "Customer Ledger":
        st.subheader("Customer Ledger")

        cust_df = (filtered_data_ar.loc[:,['cusid','cusname']].drop_duplicates().sort_values('cusname').assign(option=lambda df: df['cusid'].astype(str) + " - " + df['cusname']))
        selected = st.selectbox("Select customer",cust_df['option'].tolist())
        selected_cusid = selected.split(" - ")[0]
        ledger_df = (filtered_data_ar[filtered_data_ar['cusid'] == selected_cusid].sort_values('date'))
        st.dataframe(ledger_df)

@timed
def display_purchase_analysis_page(current_page, zid, data_dict):

    mode = st.radio(
        "Purchase View",
        ["Purchase Cohort & Requisition", "Batch Profitability & Capital Engine"],
        horizontal=True,
        index=0
    )

    # -----------------------------
    # MODE 1: Existing cohort logic (unchanged)
    # -----------------------------
    if mode == "Purchase Cohort & Requisition":
        options = [i for i in range(10)]
        default_option = 2
        default_index = options.index(default_option)
        selected_time = st.selectbox("Select Time Frame", options, index=default_index)

        sales_df, purchase_df, year_ago = common.time_filtered_data_purchase(
            data_dict['sales'], data_dict['purchase'], selected_time
        )
        cohort_df = purchase.main_purchase_product_cohort_process(sales_df, purchase_df)

        selected_products = st.multiselect("Select Product", common.update_pair_options(cohort_df,'itemcode','itemname'))
        if selected_products:
            selected_itemcodes = [x.split(" - ")[0] for x in selected_products]
            cohort_df = cohort_df[cohort_df['itemcode'].isin(selected_itemcodes)]

        cohort_df = cohort_df.applymap(common.handle_infinity_and_round).fillna(0)
        st.markdown("Product-Based Purchase Cohort")
        st.write(cohort_df, use_container_width=True)
        st.write(common.create_download_link(cohort_df, "purchase_cohort.xlsx"), unsafe_allow_html=True)

        if st.button("Generate Purchase Requirement"):
            result_df = purchase.generate_cohort(
                data_dict['purchase'],
                year_ago,
                data_dict['stock_movement'],
                sales_df,
                cohort_df
            )
            st.markdown("Generated Purchase Requisition")
            st.write(result_df, use_container_width=True)
            st.write(common.create_download_link(result_df, "purchase_requisition.xlsx"), unsafe_allow_html=True)

        return
        
    else:
        # -----------------------------
        # MODE 2: Batch Profitability & Capital Engine
        # -----------------------------
        st.subheader("Batch Profitability & Capital Engine")

        purchase_df = data_dict.get("purchase", pd.DataFrame())
        if purchase_df is None or purchase_df.empty:
            st.warning("No purchase data loaded.")
            return

             # ============================================================
        # SECTION RADIO (ONLY ONE SECTION COMPUTES PER RERUN)
        # Make Accounts Explorer the default landing section
        # ============================================================
        engine_section = st.radio(
            "Engine Section",
            [
                "Accounts Explorer (Overhead)",
                "Inventory Check",
                "Warehouse Snapshot",
                "Batch Profitability",
                "SKU Simulator",
            ],
            horizontal=True,
            index=0,
            key="purchase_engine_section",
        )

        # --------- Shipment selector ----------
        ship_df = purchase_df[["zid", "shipmentname", "povoucher", "grnvoucher", "combinedate"]].copy()
        ship_df["shipmentname"] = ship_df["shipmentname"].astype(str).fillna("").str.strip()
        ship_df = ship_df[ship_df["shipmentname"] != ""].copy()

        shipment_options = (
            ship_df[["shipmentname"]]
            .drop_duplicates()
            .sort_values("shipmentname")["shipmentname"]
            .tolist()
        )

        if not shipment_options:
            st.warning("No shipmentname found in purchase data.")
            return

        selected_shipment = st.selectbox(
            "Select Shipment (bridges 100001 + 100009)",
            shipment_options,
            index=0,
            key="purchase_selected_shipment",
            disabled=(engine_section != "Accounts Explorer (Overhead)"),
        )

        # Combinedate resolve
        _sel_rows = ship_df[ship_df["shipmentname"] == selected_shipment].copy()
        selected_combinedate = pd.to_datetime(_sel_rows["combinedate"], errors="coerce").min()

        if pd.isna(selected_combinedate):
            st.error("Could not resolve combinedate.")
            return

        selected_combinedate = selected_combinedate.normalize()

        # Display IP/GRN transparency
        ship_pick = ship_df[ship_df["shipmentname"] == selected_shipment].copy()

        info_cols = st.columns(2)
        with info_cols[0]:
            st.caption("100001 (Trading)")
            st.write({
                "IP": ship_pick[ship_pick["zid"].astype(str) == "100001"]["povoucher"].dropna().unique().tolist(),
                "GRN": ship_pick[ship_pick["zid"].astype(str) == "100001"]["grnvoucher"].dropna().unique().tolist()
            })

        with info_cols[1]:
            st.caption("100009 (Packaging)")
            st.write({
                "IP": ship_pick[ship_pick["zid"].astype(str) == "100009"]["povoucher"].dropna().unique().tolist(),
                "GRN": ship_pick[ship_pick["zid"].astype(str) == "100009"]["grnvoucher"].dropna().unique().tolist()
            })

        st.divider()

        # ---------------------------------------------------
        # Warehouse Selection (Shared across all sections)
        # ---------------------------------------------------
        wh_opts = purchase.get_all_warehouse_options(data_dict["stock_movement"])

        sel_wh_100001 = st.multiselect(
            "Warehouses (100001)",
            options=wh_opts.get("100001", []),
            default=wh_opts.get("100001", []),
        )

        sel_wh_100009 = st.multiselect(
            "Warehouses (100009)",
            options=wh_opts.get("100009", []),
            default=wh_opts.get("100009", []),
        )

        override_wh = {
            "100001": sel_wh_100001,
            "100009": sel_wh_100009,
        }
        # ============================================================
        # 1️⃣ INVENTORY CHECK
        # ============================================================

        if engine_section == "Inventory Check":

            tables = purchase.build_shipment_inventory_tables(
                purchase_df=data_dict["purchase"],
                stock_movement_df=data_dict["stock_movement"],
                sales_df=data_dict["sales"],
                returns_df=data_dict["return"],
                shipmentname=selected_shipment,
                project=st.session_state.proj,
                zid_deplete="100001",
            )

            # -----------------------------
            # Save audit results for reuse in Batch Profitability
            # -----------------------------
            st.session_state["invcheck_tables"] = tables
            st.session_state["invcheck_reconcile"] = tables.get("reconcile_sales_vs_stock", pd.DataFrame())


            with st.expander("Inventory Check Tables", expanded=True):
                st.subheader("Arrival Check — 100001")
                st.dataframe(tables["arrival_check_100001_only"], use_container_width=True)

                st.subheader("Arrival Check — 100009 Items")
                st.dataframe(tables["arrival_check_100009_items"], use_container_width=True)

                st.subheader("Sales vs Stock Reconciliation")
                st.dataframe(tables["reconcile_sales_vs_stock"], use_container_width=True)

                # st.subheader("Warehouse Breakdown")
                # st.dataframe(tables["warehouse_breakdown"], use_container_width=True)

            return


        # ============================================================
        # 2️⃣ ACCOUNTS EXPLORER
        # ============================================================

        if engine_section == "Accounts Explorer (Overhead)":

            opts = purchase.build_accounts_selector_options(
                glmst_df=data_dict["glmst_simple"],
                hierarchy_path="hierarchy.json",
            )

            level_choice = st.radio(
                "Selection level",
                ["Level 0", "Level 1", "Level 2"],
                horizontal=True,
                index=2,
            )

            level_map = {"Level 0": 0, "Level 1": 1, "Level 2": 2}
            level = level_map[level_choice]

            if level_choice == "Level 0":
                selections = st.multiselect(
                    "Select ac_code(s)",
                    opts["level0_options"],
                    default=opts["level0_options"],
                )
                selections = [s.split(" ", 1)[0].strip() for s in selections]

            elif level_choice == "Level 1":
                selections = st.multiselect(
                    "Select Level 1 head(s)",
                    opts["level1_options"],
                    default=opts["level1_options"],
                )

            else:
                selections = st.multiselect(
                    "Select Level 2 head(s)",
                    opts["level2_options"],
                    default=opts["level2_options"],
                )

            show_details = st.checkbox("Show daily ratio diagnostics")

            overhead_out = purchase.build_accounts_overhead_summary(
                purchase_df=data_dict["purchase"],
                stock_movement_df=data_dict["stock_movement"],
                glheader_df=data_dict["glheader_simple"],
                gldetail_df=data_dict["gldetail_simple"],
                glmst_df=data_dict["glmst_simple"],
                hierarchy_path="hierarchy.json",
                shipmentname=selected_shipment,
                level=level,
                selections=selections,
                include_details=show_details,
                zids_inventory=["100001", "100009"],
                warehouse_filters=override_wh,                 # NEW
                warehouse_json_path="warehouse_filters.json",  # NEW
            )

            db_overhead = float(overhead_out["totals"].get("overhead_for_shipment_sum", 0.0))

            st.markdown("### Overhead Add-ons (optional)")

            c1, c2 = st.columns(2)
            with c1:
                use_vat = st.checkbox("Add VAT overhead (%) on sales", value=st.session_state.get("use_vat_overhead", False))
                vat_pct = st.number_input("VAT %", min_value=0.0, max_value=50.0, value=float(st.session_state.get("vat_pct", 0.0)), step=0.5)
            with c2:
                use_manual = st.checkbox("Add manual overhead (BDT)", value=st.session_state.get("use_manual_overhead", False))
                manual_overhead_value = st.number_input("Manual overhead value", min_value=0.0, value=float(st.session_state.get("manual_overhead_value", 0.0)), step=100.0)

            st.session_state["use_vat_overhead"] = use_vat
            st.session_state["vat_pct"] = vat_pct if use_vat else 0.0
            st.session_state["use_manual_overhead"] = use_manual
            st.session_state["manual_overhead_value"] = manual_overhead_value if use_manual else 0.0

            # We show VAT estimate using last computed revenue if available (exact VAT will be applied in engine)
            last_rev = float(st.session_state.get("last_shipment_realized_revenue", 0.0))
            vat_est = (st.session_state["vat_pct"] / 100.0) * max(0.0, last_rev)
            manual_val = float(st.session_state.get("manual_overhead_value", 0.0))

            st.write({
                "DB overhead (allocated)": round(db_overhead, 2),
                "VAT overhead (estimate; exact in engine)": round(vat_est, 2),
                "Manual overhead": round(manual_val, 2),
                "Total overhead passed to engine (estimate)": round(db_overhead + vat_est + manual_val, 2),
            })

            st.session_state["shipment_overhead_total"] = float(
                overhead_out["totals"]["overhead_for_shipment_sum"]
            )

            st.dataframe(overhead_out["summary_df"], use_container_width=True)

            st.metric("Shipment Overhead Allocated",
                    round(overhead_out["totals"]["overhead_for_shipment_sum"], 2))

            if show_details and overhead_out["details_df"] is not None:
                with st.expander("Daily Diagnostics", expanded=False):
                    st.dataframe(overhead_out["details_df"], use_container_width=True)

            return


        # ============================================================
        # 3️⃣ BATCH PROFITABILITY
        # ============================================================

        if engine_section == "Batch Profitability":

            if "shipment_overhead_total" not in st.session_state:
                st.warning("Please run Accounts Explorer first to compute shipment overhead.")
                return

            shipment_overhead_total = float(st.session_state.get("shipment_overhead_total", 0.0))
            vat_pct = float(st.session_state.get("vat_pct", 0.0))
            manual_overhead_value = float(st.session_state.get("manual_overhead_value", 0.0))

            result_df = purchase.run_batch_profitability_engine(
            purchase_df=data_dict["purchase"],
            sales_df=data_dict["sales"],
            returns_df=data_dict["return"],
            stock_movement_df=data_dict["stock_movement"],
            glheader_df=data_dict["glheader_simple"],
            gldetail_df=data_dict["gldetail_simple"],
            glmst_df=data_dict["glmst_simple"],
            hierarchy_path="hierarchy.json",
            shipmentname=selected_shipment,
            discount_pct=0.0,
            zid_deplete="100001",
            shipment_overhead_total=shipment_overhead_total,
            vat_pct=vat_pct,
            manual_overhead_value=manual_overhead_value,
            inventory_tables=st.session_state.get("invcheck_tables"),   # ✅ add this
            )

            st.session_state["last_batch_df"] = result_df.copy()
            st.session_state["last_shipment_realized_revenue"] = float(result_df["sold_revenue"].sum()) if not result_df.empty else 0.0

            st.dataframe(result_df, use_container_width=True)

            if result_df is not None and not result_df.empty:
                sum_cols = [
                    "sold_revenue", "realized_cogs", "realized_gm",
                    "overhead_realized", "net_profit_realized",
                    "remaining_cost_value", "proj_remaining_revenue", "proj_remaining_gm",
                    "overhead_projected", "Proj_remaining_profit", "proj_final_profit",
                ]
                totals = {c: float(result_df[c].sum()) for c in sum_cols if c in result_df.columns}

                # averages (simple + weighted)
                vel = result_df["velocity"].replace([np.inf, -np.inf], np.nan)
                dtc = result_df["days_to_clear"].replace([np.inf, -np.inf], np.nan)

                simple_avg_velocity = float(vel.mean(skipna=True))
                simple_avg_days_to_clear = float(dtc.mean(skipna=True))

                # Weighted: velocity weighted by sold_qty; days_to_clear weighted by remaining_qty
                sold_w = result_df["sold_qty"].clip(lower=0.0)
                rem_w = result_df["remaining_qty"].clip(lower=0.0)

                w_avg_velocity = float((vel.fillna(0.0) * sold_w).sum() / sold_w.sum()) if sold_w.sum() > 0 else 0.0
                w_avg_days_to_clear = float((dtc.fillna(0.0) * rem_w).sum() / rem_w.sum()) if rem_w.sum() > 0 else 0.0

                st.markdown("### Shipment Totals & Averages")
                st.write({k: round(v, 2) for k, v in totals.items()})

                st.write({
                    "Avg velocity (simple)": round(simple_avg_velocity, 4),
                    "Avg days_to_clear (simple)": round(simple_avg_days_to_clear, 2),
                    "Avg velocity (weighted by sold_qty)": round(w_avg_velocity, 4),
                    "Avg days_to_clear (weighted by remaining_qty)": round(w_avg_days_to_clear, 2),
                })

            return
        # ============================================================
        # 4️⃣ WAREHOUSE SNAPSHOT
        # ============================================================

        if engine_section == "Warehouse Snapshot":

            warehouse_basis = st.radio(
                "Warehouse valuation date",
                ["Shipment arrival (combinedate)", "Today"],
                horizontal=True,
            )

            as_of_dt = selected_combinedate if warehouse_basis == "Shipment arrival (combinedate)" else pd.Timestamp.today().normalize()

            wh_value_df = purchase.build_warehouse_total_value_table(
                stock_movement_df=data_dict["stock_movement"],
                as_of_date=as_of_dt,
                zids=["100001", "100009"],
                warehouse_filters=override_wh,
                warehouse_json_path="warehouse_filters.json",
            )

            st.dataframe(wh_value_df, use_container_width=True)

            st.metric("Total Inventory Value",
                    round(wh_value_df["totalvalue"].sum(), 2))

            return

        # ============================================================
        # 3️⃣ SKU Simulator
        # ============================================================

        if engine_section == "SKU Simulator":
            st.subheader("SKU Simulator (no recompute)")

            base_df = st.session_state.get("last_batch_df")
            if base_df is None or base_df.empty:
                st.warning("Run 'Batch Profitability' once first so I can cache the results for simulation.")
                return

            base_df = base_df.copy()
            base_df["sku_label"] = base_df["itemcode"].astype(str) + " - " + base_df["itemname"].astype(str)

            sku = st.selectbox("Select SKU", base_df["sku_label"].tolist(), index=0)
            row = base_df[base_df["sku_label"] == sku].iloc[0]

            c1, c2 = st.columns(2)
            with c1:
                new_price = st.number_input("Scenario price", min_value=0.0, value=float(row["scenario_price"]), step=1.0)
            with c2:
                new_days_to_clear = st.number_input("Target days_to_clear", min_value=0.0, value=float(row["days_to_clear"]) if pd.notna(row["days_to_clear"]) else 0.0, step=1.0)

            # Recompute ONLY this SKU using cached realized numbers
            remaining_qty = float(row["remaining_qty"])
            unit_cost = float(row["unit_cost"])
            remaining_cost_value = float(row["remaining_cost_value"])
            overhead_realized = float(row["overhead_realized"])
            days_active = max(1, int(row["days_active"]))

            proj_remaining_revenue = remaining_qty * float(new_price)
            proj_remaining_gm = proj_remaining_revenue - remaining_cost_value

            # -----------------------------------------
            # Match Batch Profitability projected logic
            # -----------------------------------------
            sim_days_to_clear = min(max(float(new_days_to_clear), 0.0), 730.0)

            # Shipment-level D0 = total overhead pool / max(days_active)
            max_days_active = max(1, int(base_df["days_active"].max()))
            total_overhead_pool = float(base_df["overhead_realized"].sum())
            D0 = total_overhead_pool / max_days_active

            # Recompute projected remaining revenue for all SKUs:
            # selected SKU uses new scenario price, others keep current projected remaining revenue
            sim_proj_rev_all = base_df[["itemcode", "proj_remaining_revenue", "remaining_cost_value"]].copy()
            sim_proj_rev_all["sim_proj_remaining_revenue"] = sim_proj_rev_all["proj_remaining_revenue"].astype(float)

            sim_proj_rev_all.loc[
                sim_proj_rev_all["itemcode"].astype(str) == str(row["itemcode"]),
                "sim_proj_remaining_revenue"
            ] = proj_remaining_revenue

            total_proj_remaining_revenue = float(sim_proj_rev_all["sim_proj_remaining_revenue"].sum())

            if total_proj_remaining_revenue > 0:
                remaining_share = proj_remaining_revenue / total_proj_remaining_revenue
            else:
                total_remaining_cost = float(sim_proj_rev_all["remaining_cost_value"].astype(float).sum())
                remaining_share = (remaining_cost_value / total_remaining_cost) if total_remaining_cost > 0 else 0.0

            decay_factor = 0.97 ** (sim_days_to_clear / 60.0)

            overhead_projected = D0 * decay_factor * sim_days_to_clear * remaining_share

            proj_remaining_profit = proj_remaining_gm - overhead_projected
            proj_final_profit = float(row["net_profit_realized"]) + proj_remaining_profit

            st.markdown("### Simulator Output (this SKU only)")
            st.write({
                "Realized sold_revenue": round(float(row["sold_revenue"]), 2),
                "Realized GM": round(float(row["realized_gm"]), 2),
                "Realized overhead": round(overhead_realized, 2),
                "Net profit realized": round(float(row["net_profit_realized"]), 2),
                "Projected remaining revenue": round(proj_remaining_revenue, 2),
                "Projected remaining GM": round(proj_remaining_gm, 2),
                "Projected overhead": round(overhead_projected, 2),
                "Projected remaining profit": round(proj_remaining_profit, 2),
                "Projected final profit": round(proj_final_profit, 2),
            })

            return

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
    
    data = common.load_json('modules/businesses.json')
    businesses = data.get('businesses', {})
    
    income_label = common.load_json('modules/labels.json')['income_statement_label']
    income_label_df = pd.DataFrame(list(income_label.items()), columns=['ac_lv4', 'Income Statement'])
    
    balance_label = common.load_json('modules/labels.json')['balance_sheet_label']
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
            try:
                default_idx = biz_options.index(zid)
            except ValueError:
                default_idx = 0
            analyse_zid = st.selectbox("Select Business",biz_options,index=default_idx)
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

            # 5️⃣ histogram picker
            state = pd.concat(
                [pl_sorted.drop(columns="", errors="ignore"), bs_lv0],
                axis=0,
                ignore_index=True
            )

            cols = st.columns(4)
            with cols[0]:
                selected_index = st.selectbox("Select Account",state['ac_name'].to_list(),index=1)

            # Select the row you want to convert into a dictionary
            selected_row = state[state['ac_name'] == selected_index]

            result_dict = selected_row.to_dict(orient='list') 
            first_key = list(result_dict.keys())[0] 
            second_key = list(result_dict.keys())[1]  # Get the first key
            result_dict.pop(first_key)
            result_dict.pop(second_key)

            common_v.plot_histogram(result_dict, selected_index)
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
            try:
                default_idx = biz_options.index(zid)
            except ValueError:
                default_idx = 0
            analyse_zid = st.selectbox("Select Business",biz_options,index=default_idx)
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

            # 5️⃣ histogram picker
            state = pd.concat(
                [pl_sorted.drop(columns="", errors="ignore"), bs_lv0],
                axis=0,
                ignore_index=True
            )

            cols = st.columns(4)
            with cols[0]:
                selected_index = st.selectbox("Select Account",state['ac_name'].to_list(),index=1)

            # Select the row you want to convert into a dictionary
            selected_row = state[state['ac_name'] == selected_index]

            result_dict = selected_row.to_dict(orient='list') 
            first_key = list(result_dict.keys())[0] 
            second_key = list(result_dict.keys())[1]  # Get the first key
            result_dict.pop(first_key)
            result_dict.pop(second_key)

            common_v.plot_histogram(result_dict, selected_index)
        else:
            pass

# ---------- Cached loaders (simple table pulls) ----------
@st.cache_data(show_spinner=False)
def _load_cacus(zid: str) -> pd.DataFrame:
    df = Analytics("cacus_simple", zid=zid, filters={"zid": (str(zid),)}).data
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_gldetail(zid: str) -> pd.DataFrame:
    project = st.session_state.proj
    filters = {"zid": (str(zid),)}
    df = Analytics("gldetail_simple", zid=zid, project=project, filters=filters).data
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_glheader(zid: str) -> pd.DataFrame:
    df = Analytics("glheader_simple", zid=zid, filters={"zid": (str(zid),)}).data
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_glmst(zid: str) -> pd.DataFrame:
    df = Analytics("glmst_simple", zid=zid, filters={"zid": (str(zid),)}).data
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_casup(zid: str) -> pd.DataFrame:
    df = Analytics("casup_simple", zid=zid, filters={"zid": (str(zid),)}).data
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _compute_ar_balances(zid: str, year: int, month: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cacus = _load_cacus(zid)
    gld   = _load_gldetail(zid)
    glh   = _load_glheader(zid)
    glm   = _load_glmst(zid)

    if any(df.empty for df in (cacus, gld, glh, glm)):
        return pd.DataFrame(), pd.DataFrame()

    # Normalize types for safe merges/filters
    for df, col in ((gld, "voucher"), (glh, "voucher"),
                    (gld, "ac_code"), (glm, "ac_code"),
                    (gld, "ac_sub"), (cacus, "cusid")):
        df[col] = df[col].astype(str)

    # Enrich detail with date parts and account info
    lines = gld.merge(glh[["zid","voucher","date","year","month"]], on=["zid","voucher"], how="left")
    lines = lines.merge(glm[["zid","ac_code","ac_type","ac_name"]], on=["zid","ac_code"], how="left")

    # Keep AR-related lines (Asset + linked to a customer)
    lines = lines[lines["ac_code"] == "01030001"]

    # IGNORE all OB vouchers globally (closing, period, trail)
    vup = lines["voucher"].astype(str).str.upper()
    lines = lines[~vup.str.startswith("OB--", na=False)].copy()

    # Attach customer names (inner join = keep only valid customers in lines)
    lines = lines.merge(cacus[["zid","cusid","cusname"]],
                        left_on=["zid","ac_sub"], right_on=["zid","cusid"], how="inner")

    # Build masks and helper columns once (faster than lambda per-group)
    yr = int(year); mo = int(month)
    y  = lines["year"].astype(int)
    m  = lines["month"].astype(int)
    val = lines["value"].astype(float)

    asof_cond   = (y < yr) | ((y == yr) & (m <= mo))
    period_cond = (y == yr) & (m == mo)

    lines["val_asof"]       = np.where(asof_cond,   val, 0.0)
    lines["val_period"]     = np.where(period_cond, val, 0.0)
    lines["tx_in_period"]   = np.where(period_cond, 1,   0)

    # Aggregate by customer
    agg = (lines.groupby(["cusid","cusname"], as_index=False)
                 .agg(closing_balance=("val_asof","sum"),
                      month_movement=("val_period","sum"),
                      tx_count_in_month=("tx_in_period","sum")))
    agg["had_tx_in_month"] = agg["tx_count_in_month"] > 0
    agg = agg.drop(columns=["tx_count_in_month"])

    # Include all customers (even if no lines after OB removal)
    all_customers = cacus[["cusid","cusname"]].drop_duplicates()
    out = (all_customers.merge(agg, on=["cusid","cusname"], how="left")
                      .fillna({"closing_balance":0.0, "month_movement":0.0, "had_tx_in_month":False}))

    # Sort by magnitude of closing balance
    out = out.sort_values("closing_balance", key=lambda s: s.abs(), ascending=False)

    # Trail up to as-of (also without OB lines)
    trail_asof = lines[asof_cond].copy().sort_values(["date","voucher","ac_code"])

    return out, trail_asof

def _compute_ap_balances(zid: str, year: int, month: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    casup = _load_casup(zid)
    gld   = _load_gldetail(zid)    # existing helper
    glh   = _load_glheader(zid)    # existing helper
    glm   = _load_glmst(zid)       # existing helper

    if any(df.empty for df in (casup, gld, glh, glm)):
        return pd.DataFrame(), pd.DataFrame()

    # Normalize types for safe merges/filters
    for df, col in ((gld, "voucher"), (glh, "voucher"),
                    (gld, "ac_code"), (glm, "ac_code"),
                    (gld, "ac_sub"), (casup, "supid")):
        df[col] = df[col].astype(str)

    # Enrich detail
    lines = gld.merge(glh[["zid","voucher","date","year","month"]], on=["zid","voucher"], how="left")
    lines = lines.merge(glm[["zid","ac_code","ac_type","ac_name"]], on=["zid","ac_code"], how="left")

    # Keep AP-related lines: Liability accounts + supplier-linked subledger
    lines = lines[lines["ac_code"].isin(["09030001", "09030004"])]

    # Ignore all OB vouchers
    vup = lines["voucher"].str.upper()
    lines = lines[~vup.str.startswith("OB--", na=False)].copy()

    # Attach supplier names (inner join = only valid suppliers)
    lines = lines.merge(casup[["zid","supid","supname"]],
                        left_on=["zid","ac_sub"], right_on=["zid","supid"], how="inner")

    yr, mo = int(year), int(month)
    y = lines["year"].astype(int)
    m = lines["month"].astype(int)
    val = lines["value"].astype(float)

    asof_cond   = (y < yr) | ((y == yr) & (m <= mo))
    period_cond = (y == yr) & (m == mo)

    lines["val_asof"]     = np.where(asof_cond,   val, 0.0)
    lines["val_period"]   = np.where(period_cond, val, 0.0)
    lines["tx_in_period"] = np.where(period_cond, 1,   0)

    # Aggregate by supplier
    agg = (lines.groupby(["supid","supname"], as_index=False)
                 .agg(closing_balance=("val_asof","sum"),
                      month_movement=("val_period","sum"),
                      tx_count_in_month=("tx_in_period","sum")))
    agg["had_tx_in_month"] = agg["tx_count_in_month"] > 0
    agg = agg.drop(columns=["tx_count_in_month"])

    # Include all suppliers (even with no lines after OB removal)
    all_sup = casup[["supid","supname"]].drop_duplicates()
    out = (all_sup.merge(agg, on=["supid","supname"], how="left")
                 .fillna({"closing_balance":0.0, "month_movement":0.0, "had_tx_in_month":False}))

    # Sort by magnitude of closing balance
    out = out.sort_values("closing_balance", key=lambda s: s.abs(), ascending=False)

    # Trail up to as-of (OB ignored)
    trail_asof = lines[asof_cond].copy().sort_values(["date","voucher","ac_code"])

    return out, trail_asof

@st.cache_data(show_spinner=False)
def _ledger_accounts_by_type(zid: str, ac_type: str) -> pd.DataFrame:
    glm = _load_glmst(zid)
    if glm.empty:
        return pd.DataFrame()
    df = glm[glm["ac_type"] == ac_type].copy()
    df["label"] = df["ac_code"].astype(str) + " — " + df["ac_name"].astype(str)
    return df.sort_values("ac_code")

@st.cache_data(show_spinner=False)
def _compute_ledger(zid: str, ac_type: str, ac_codes: list[str], year: int, month: int,
                    mode: str, ignore_ob: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (voucher_summary, line_table)
    - voucher_summary columns: date, voucher, amount, ob_amount, lines
    - line_table columns: date, voucher, ac_code, ac_name, ac_lv1, ac_lv2, amount, ob_amount, is_ob
    """
    gld = _load_gldetail(zid)
    glh = _load_glheader(zid)
    glm = _load_glmst(zid)

    if any(df.empty for df in (gld, glh, glm)):
        return pd.DataFrame(), pd.DataFrame()

    # normalize dtypes
    for df, col in ((gld,"voucher"), (glh,"voucher"), (gld,"ac_code"), (glm,"ac_code")):
        df[col] = df[col].astype(str)

    # enrich lines with date parts and account meta
    lines = gld.merge(glh[["zid","voucher","date","year","month"]],
                      on=["zid","voucher"], how="left")
    lines = lines.merge(glm[["zid","ac_code","ac_name","ac_type","ac_lv1","ac_lv2"]],
                        on=["zid","ac_code"], how="left")

    # filter by type and selected accounts
    lines = lines[lines["ac_type"] == ac_type]
    if ac_codes:
        lines = lines[lines["ac_code"].isin([str(c) for c in ac_codes])]

    if lines.empty:
        return pd.DataFrame(), pd.DataFrame()

    # compute period masks
    yr, mo = int(year), int(month)
    y  = lines["year"].astype(int)
    m  = lines["month"].astype(int)
    v  = lines["value"].astype(float)

    if mode == "Income (this month)":
        mask = (y == yr) & (m == mo)
    else:  # "Balance (as-of)"
        mask = (y < yr) | ((y == yr) & (m <= mo))

    # OB identification (case-insensitive)
    is_ob = lines["voucher"].astype(str).str.upper().str.startswith("OB--", na=False)
    lines = lines[mask].copy()
    lines["is_ob"] = is_ob[mask].values
    lines["ob_amount"] = np.where(lines["is_ob"], v[mask].values, 0.0)

    # amount column respects ignore_ob toggle; OB always shown in ob_amount
    if ignore_ob:
        lines["amount"] = np.where(lines["is_ob"], 0.0, v[mask].values)
    else:
        lines["amount"] = v[mask].values

    # build outputs (narration intentionally omitted)
    line_table = (lines[["date","voucher","ac_code","ac_name","ac_lv1","ac_lv2","amount","ob_amount","is_ob"]]
                        .sort_values(["date","voucher","ac_code"])
                        .reset_index(drop=True))

    return line_table

@timed
def display_accounting_analysis_main(current_page, zid: str):
    st.title("Accounting Analysis")

    tab_ar, tab_ap, tab_ledger = st.tabs([
        "🧾 AR Analysis",
        "📄 AP Analysis",
        "📘 Ledger Entries"
    ])

    # ───────────────────────────── AR Analysis ────────────────────────────────
    with tab_ar:
        c1, c2, _ = st.columns([1,1,2])
        with c1:
            year = st.number_input("Year", min_value=2018, max_value=2035, value=pd.Timestamp.today().year, step=1)
        with c2:
            month = st.number_input("Month", min_value=1, max_value=12, value=pd.Timestamp.today().month, step=1)

        if st.button("Load AR", type="primary"):
            st.session_state["_ar_loaded"] = True

        if st.session_state.get("_ar_loaded"):
            summary, trail_asof = _compute_ar_balances(zid, int(year), int(month))
            if summary.empty:
                st.info("No AR balances found for the selected period.")
            else:
                st.caption("AR balances as of selected month-end (OB included automatically)")
                total_ar = float(pd.to_numeric(summary["closing_balance"], errors="coerce").sum())
                st.caption(
                    f"AR balances as of selected month-end — OB vouchers (OB--) are ignored in all calculations and trails. "
                    f"Total closing: **{total_ar:,.2f}**"
                )
                st.dataframe(summary, use_container_width=True, height=440)
                st.write(common.create_download_link(summary,"ar_balances.xlsx"), unsafe_allow_html=True)

                # Drill-down for a single customer (up to month-end)
                st.markdown("### Customer Trail (up to selected month)")
                pick_id = st.selectbox(
                    "Choose a Customer ID:",
                    options=["—"] + summary["cusid"].astype(str).tolist(),
                    index=0
                )
                if pick_id and pick_id != "—":
                    cust_trail = trail_asof[trail_asof["cusid"].astype(str) == str(pick_id)]
                    if cust_trail.empty:
                        st.info("No GL lines up to the selected month for this customer.")
                    else:
                        st.dataframe(
                            cust_trail[["date","voucher","ac_code","ac_name","value"]]
                            .reset_index(drop=True),
                            use_container_width=True, height=420
                        )
                        st.write(common.create_download_link(cust_trail,"ar_trail.xlsx"), unsafe_allow_html=True)

    # ───────────────────────────── AP Analysis (placeholder) ──────────────────
    with tab_ap:
        c1, c2, _ = st.columns([1,1,2])
        with c1:
            ap_year = st.number_input("Year", min_value=2018, max_value=2035,
                                    value=pd.Timestamp.today().year, step=1, key="ap_year")
        with c2:
            ap_month = st.number_input("Month", min_value=1, max_value=12,
                                    value=pd.Timestamp.today().month, step=1, key="ap_month")

        if st.button("Load AP", type="primary"):
            st.session_state["_ap_loaded"] = True

        if st.session_state.get("_ap_loaded"):
            ap_summary, ap_trail_asof = _compute_ap_balances(zid, int(ap_year), int(ap_month))
            if ap_summary.empty:
                st.info("No AP balances found for the selected period.")
            else:
                st.caption("AP balances as of selected month-end — OB vouchers (OB--) are ignored in all calculations and trails.")
                total_ap = float(pd.to_numeric(ap_summary["closing_balance"], errors="coerce").sum())
                st.caption(
                    f"AP balances as of selected month-end — OB vouchers (OB--) are ignored in all calculations and trails. "
                    f"Total closing: **{total_ap:,.2f}**"
                )
                st.dataframe(ap_summary, use_container_width=True, height=440)
                st.write(common.create_download_link(ap_summary,"ap_balances.xlsx"), unsafe_allow_html=True)

                st.markdown("### Supplier Trail (up to selected month)")
                pick_sup = st.selectbox(
                    "Choose a Supplier ID:",
                    options=["—"] + ap_summary["supid"].astype(str).tolist(),
                    index=0
                )
                if pick_sup and pick_sup != "—":
                    sup_trail = ap_trail_asof[ap_trail_asof["supid"].astype(str) == str(pick_sup)]
                    if sup_trail.empty:
                        st.info("No GL lines up to the selected month for this supplier.")
                    else:
                        st.dataframe(
                            sup_trail[["date","voucher","ac_code","ac_name","value"]]
                            .reset_index(drop=True),
                            use_container_width=True, height=420
                        )
                        st.write(common.create_download_link(sup_trail,"ap_trail.xlsx"), unsafe_allow_html=True)

    # ───────────────────────── Ledger Entries (placeholder) ───────────────────
    with tab_ledger:
        colL, colR = st.columns([2,3], gap="large")
        with colL:
            st.subheader("Filters")

            mode = st.radio(
                "Computation",
                options=["Income (this month)", "Balance (as-of)"],
                index=0,
                help="Income: only the picked month. Balance: up to & including the picked month."
            )

            ac_choices = ["Income", "Expenditure"] if mode == "Income (this month)" else ["Asset", "Liability"]

            if st.session_state.get("ledger_prev_mode") != mode:
                st.session_state["ledger_prev_mode"] = mode
                st.session_state.pop("ledger_ac_type", None)
                st.session_state.pop("ledger_accounts", None)  # if you keyed your multiselect

            c1, c2 = st.columns(2)
            with c1:
                year = st.number_input("Year", min_value=2018, max_value=2035,
                                        value=pd.Timestamp.today().year, step=1, key="led_year")
            with c2:
                month = st.number_input("Month", min_value=1, max_value=12,
                                        value=pd.Timestamp.today().month, step=1, key="led_month")

            ignore_ob = st.toggle("Ignore OB (OB--)", value=True,
                                    help="If ON, OB amounts are excluded from totals. OB values still appear in a separate column.")

            # Account type → account picker
            ac_type = st.selectbox("Account Type", ac_choices, index=0, key="ledger_ac_type")
            acc_df  = _ledger_accounts_by_type(zid, ac_type)
            acc_labels = acc_df["label"].tolist()
            label2code = dict(zip(acc_df["label"], acc_df["ac_code"]))

            picked_labels = st.multiselect("Accounts (ac_code — ac_name)", options=acc_labels, placeholder="Pick one or more (empty = all in type)")
            ac_codes = [label2code[l] for l in picked_labels]

            go = st.button("Load ledger", type="primary")

        with colR:
            if go:
                lines = _compute_ledger(zid, ac_type, ac_codes, int(year), int(month), mode, bool(ignore_ob))
                if lines.empty:
                    st.info("No postings for the selected filters.")
                else:
                    st.caption("Line entries (no narration)")
                    st.dataframe(lines, use_container_width=True, height=420)
                    st.write(common.create_download_link(lines,"ledger_lines.xlsx"), unsafe_allow_html=True)

# ---------- Inventory Analysis (updated for single-zid SQL + 100001→+100009 merge) ----------


def _effective_zids(primary_zid: str) -> list[str]:
    """
    If 100001 is chosen, auto-include 100009 (packaging) as well.
    Otherwise, just use the primary zid.
    """
    p = str(primary_zid)
    return [p, "100009"] if p == "100001" else [p]

@st.cache_data(show_spinner=False)
def _load_stock_flow(zid: str) -> pd.DataFrame:
    zids = _effective_zids(zid)  # keep your 100001 → also 100009 rule
    frames = []
    for z in zids:
        try:
            df = Analytics("stock_flow", zid=z, filters={"zid": (str(z),)}).data
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df.assign(_src_zid=str(z)))
        except Exception as e:
            st.error(f"Error loading stock_flow for zid={z}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_product_inventory(zid: str) -> pd.DataFrame:
    """
    Loads monthly product-level inventory transactions from `stock` joined to `caitem`.
    Your SQL now uses: WHERE stock.zid = (%s)
    We therefore call once per effective zid and concatenate.
    The SQL already returns itemcode with packcode CASE logic applied.
    """
    zids = _effective_zids(zid)
    frames = []
    for z in zids:
        try:
            # IMPORTANT: Single parameter tuple (z,) to match "= (%s)"
            df = Analytics("stock", zid=z, filters={"zid": (str(z),)}).data
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df.assign(_src_zid=str(z)))
        except Exception as e:
            st.error(f"Error loading product inventory for zid={z}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_inventory_value(zid: str) -> pd.DataFrame:
    """
    Loads warehouse-level monthly stock value snapshots from `stock_value`.
    The SQL uses: WHERE zid = (%s)
    We will mirror the product logic and include 100009 when the primary zid is 100001,
    so that packaging warehouses are reflected in warehouse snapshot/value too.
    """
    zids = _effective_zids(zid)
    frames = []
    for z in zids:
        try:
            df = Analytics("stock_value", zid=z, filters={"zid": (str(z),)}).data
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df.assign(_src_zid=str(z)))
        except Exception as e:
            st.error(f"Error loading stock_value for zid={z}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@timed
def display_inventory_analysis_main(current_page, zid: str):
    st.title("Inventory Analysis")

    inv_df = _load_product_inventory(zid)
    val_df = _load_inventory_value(zid)

    if inv_df is None or inv_df.empty:
        st.info("No product inventory data found for the selected company (zid).")
        return

    # --- Normalize dtypes ---
    if "zid" in inv_df.columns:
        inv_df["zid"] = inv_df["zid"].astype(str)
    for col in ["year", "month"]:
        if col in inv_df.columns:
            inv_df[col] = pd.to_numeric(inv_df[col], errors="coerce").astype("Int64")
    for c in ["warehouse", "itemgroup", "itemcode", "itemname"]:
        if c in inv_df.columns:
            inv_df[c] = inv_df[c].astype(str)
    if "stockqty" in inv_df.columns:
        inv_df["stockqty"] = pd.to_numeric(inv_df["stockqty"], errors="coerce").fillna(0.0)
    if "stockvalue" in inv_df.columns:
        inv_df["stockvalue"] = pd.to_numeric(inv_df["stockvalue"], errors="coerce").fillna(0.0)

    # Helper year-month key for comparisons
    inv_df["ym"] = inv_df["year"].fillna(0).astype(int) * 100 + inv_df["month"].fillna(0).astype(int)
    inv_df["year_month"] = inv_df.apply(
        lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}"
        if pd.notna(r['year']) and pd.notna(r['month']) else None, axis=1
    )

    # --- Build filter options from data ---
    years = sorted(inv_df["year"].dropna().astype(int).unique().tolist())
    months = list(range(1, 13))
    warehouses = sorted(inv_df["warehouse"].dropna().unique().tolist())
    itemgroups = sorted(inv_df["itemgroup"].dropna().unique().tolist())

    # Product selector label: "code — name"
    inv_df["product_label"] = inv_df.apply(
        lambda r: f"{r['itemcode']} — {r['itemname']}" if pd.notna(r.get("itemname")) else str(r['itemcode']),
        axis=1
    )
    products = (
        inv_df[["itemcode", "product_label"]]
        .drop_duplicates()
        .sort_values("itemcode")["product_label"]
        .tolist()
    )

    # --- UI: Filters ---
    st.subheader("Filters")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])

    with c1:
        year_sel = st.selectbox("Year (cutoff)", years, index=len(years) - 1 if years else 0, key="inv_year")
    with c2:
        month_sel = st.selectbox(
            "Month (cutoff)", months,
            index=(pd.Timestamp.today().month - 1),
            format_func=lambda m: calendar.month_abbr[m],
            key="inv_month"
        )
    with c3:
        use_latest = st.toggle(
            "Use Max Year–Month", value=False,
            help="If ON, uses the latest available period in data (considering auto-included packaging zid when applicable)."
        )
    with c4:
        st.caption("🔎 Item code already applies packcode CASE logic in SQL (no extra toggle needed).")

    wh_sel = st.multiselect("Warehouse(s)", options=warehouses, default=warehouses)
    ig_sel = st.multiselect("Item Group(s)", options=itemgroups, default=itemgroups)

    prod_sel_labels = st.multiselect("Product(s)", options=products, default=[],
                                     placeholder="Type to search code/name…")
    label_to_code = dict(inv_df[["product_label", "itemcode"]].drop_duplicates().values.tolist())
    prod_codes = [label_to_code.get(lbl) for lbl in prod_sel_labels]

    # Determine cutoff based on toggle (on merged dataset)
    if use_latest:
        latest_row = inv_df.loc[inv_df["ym"].idxmax()]
        cutoff_year, cutoff_month = int(latest_row["year"]), int(latest_row["month"])
    else:
        cutoff_year, cutoff_month = int(year_sel), int(month_sel)
    cutoff_ym = cutoff_year * 100 + cutoff_month
    st.markdown(f"**Cutoff:** {cutoff_year}-{cutoff_month:02d}")

    # Apply filters
    df_f = inv_df.copy()
    if wh_sel:
        df_f = df_f[df_f["warehouse"].isin(wh_sel)]
    if ig_sel:
        df_f = df_f[df_f["itemgroup"].isin(ig_sel)]
    if prod_codes:
        df_f = df_f[df_f["itemcode"].isin(prod_codes)]
    df_f = df_f[df_f["ym"] <= cutoff_ym]  # up to & including cutoff

    if df_f.empty:
        st.warning("No rows match the current filters.")
        return

    # ------ Report 1: Product Inventory Ledger (Monthly Qty) ------
    st.subheader("1) Product Inventory Ledger (Monthly Qty)")
    ledger_cols = ["year", "month", "warehouse", "itemcode", "itemname", "itemgroup", "stockqty"]
    # Keep zid in the ledger to trace packaging vs primary if desired; not grouped by zid when running totals
    if "zid" in df_f.columns and "zid" not in ledger_cols:
        ledger_cols = ["zid"] + ledger_cols

    ledger = (
        df_f[ledger_cols]
        .sort_values(["itemcode", "warehouse", "year", "month"])
        .reset_index(drop=True)
    )
    # Running cumulative qty per itemcode×warehouse across combined zids
    ledger["running_qty"] = ledger.groupby(["itemcode", "warehouse"])["stockqty"].cumsum()

    st.dataframe(ledger, use_container_width=True, height=420)
    st.write(common.create_download_link(ledger, "inventory_ledger.xlsx"), unsafe_allow_html=True)

    # ------ Report 2: Final Stock by Product & Warehouse (Qty & Value as-of cutoff) ------
    st.subheader("2) Final Stock — Qty & Value (as of cutoff)")
    final = (
        df_f
        .groupby(["warehouse", "itemcode", "itemname", "itemgroup"], as_index=False)
        .agg(final_qty=("stockqty", "sum"),
             final_value=("stockvalue", "sum"))
        .sort_values(["warehouse", "itemcode"])
        .reset_index(drop=True)
    )
    st.dataframe(final, use_container_width=True, height=420)
    st.write(common.create_download_link(final, "final_stock_by_product.xlsx"), unsafe_allow_html=True)

    # ------ Report 3: Warehouse Value Transactions per Month (flow up to cutoff) ------
    st.subheader("3) Warehouse Value Transactions (Monthly, up to cutoff)")
    flow = (
        df_f
        .groupby(["year", "month", "warehouse"], as_index=False)
        .agg(value_txn=("stockvalue", "sum"))
        .sort_values(["year", "month", "warehouse"])
    )
    flow["year_month"] = flow.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}", axis=1)
    flow = flow[["year_month", "warehouse", "value_txn"]]
    st.dataframe(flow, use_container_width=True, height=360)
    st.write(common.create_download_link(flow, "warehouse_value_flow.xlsx"), unsafe_allow_html=True)

   # ------ Report 4: Warehouse Ending Stock Value (as of cutoff) ------
    st.subheader("4) Warehouse Ending Stock Value (as of cutoff)")

    # df_f already contains all rows up to the cutoff (and only selected warehouses/products/groups)
    warehouse_ending = (
        df_f.groupby(["warehouse"], as_index=False)
            .agg(ending_value=("stockvalue", "sum"))
            .sort_values("warehouse")
            .reset_index(drop=True)
    )

    st.dataframe(warehouse_ending, use_container_width=True, height=300)
    st.write(common.create_download_link(warehouse_ending, "warehouse_ending_stock_value.xlsx"),
            unsafe_allow_html=True)

   # ------ Report 5: Movement Analysis (Fast / Slow / Stagnant) ------
    st.subheader("5) Movement Analysis (Fast / Slow / Stagnant)")

    flow_df = _load_stock_flow(zid)

    if flow_df is None or flow_df.empty:
        st.info("No stock_flow data found for the effective zids.")
    else:
        import numpy as np

        # ---- 1) Normalize types ----
        for col in ["year", "month"]:
            if col in flow_df.columns:
                flow_df[col] = pd.to_numeric(flow_df[col], errors="coerce")
        for c in ["warehouse", "itemcode"]:
            if c in flow_df.columns:
                flow_df[c] = flow_df[c].astype(str)
        for c in ["qty_in","qty_out","net_qty","val_in","val_out","net_val"]:
            if c in flow_df.columns:
                flow_df[c] = pd.to_numeric(flow_df[c], errors="coerce").fillna(0.0)

        # ---- 2) Make stock_flow pack-aware so keys align with Report 2 ----
        # We mirror the same CASE logic used in your stock SQL.
        # Load caitem for effective zids and build a raw->display mapping.
        try:
            zids_eff = _effective_zids(zid)
            caitem_frames = []
            for z in zids_eff:
                try:
                    ci = Analytics("caitem", zid=z, filters={"zid": (str(z),)}).data
                    if isinstance(ci, pd.DataFrame) and not ci.empty:
                        caitem_frames.append(ci[["itemcode", "itemname", "itemgroup", "packcode"]].copy())
                except Exception as _e:
                    pass
            if caitem_frames:
                caitem_map = pd.concat(caitem_frames, ignore_index=True).drop_duplicates()
                # compute display_code = packcode (when valid) else raw itemcode
                def _choose_code(row):
                    pk = str(row.get("packcode") or "").strip()
                    if pk and pk.upper() != "NO" and not pk.startswith("KH"):
                        return pk
                    return str(row.get("itemcode"))
                caitem_map["display_code"] = caitem_map.apply(_choose_code, axis=1)
                # Raw->display mapping
                raw_to_display = dict(zip(caitem_map["itemcode"].astype(str), caitem_map["display_code"]))
                # Apply mapping to flow_df item codes
                flow_df["itemcode"] = flow_df["itemcode"].map(raw_to_display).fillna(flow_df["itemcode"]).astype(str)
                # Build a lookup of display_code -> (name, group)
                disp_lookup = (
                    caitem_map[["display_code","itemname","itemgroup"]]
                    .drop_duplicates()
                    .rename(columns={"display_code":"itemcode"})
                )
            else:
                disp_lookup = inv_df[["itemcode","itemname","itemgroup"]].drop_duplicates()
        except Exception:
            # Fallback: use metadata from inv_df if caitem load fails
            disp_lookup = inv_df[["itemcode","itemname","itemgroup"]].drop_duplicates()

        # Attach itemname/itemgroup after mapping
        flow_df = flow_df.merge(disp_lookup, on="itemcode", how="left")

        # ---- 3) Apply the same user filters (warehouse / itemgroup / product) ----
        if wh_sel:
            flow_df = flow_df[flow_df["warehouse"].isin(wh_sel)]
        if ig_sel:
            flow_df = flow_df[flow_df["itemgroup"].isin(ig_sel)]
        if prod_codes:
            flow_df = flow_df[flow_df["itemcode"].isin(prod_codes)]

        # If nothing remains, still show balances with zeros
        # Compute balances base from df_f (already filtered and <= cutoff)
        if "final" not in locals():
            final = (
                df_f.groupby(["warehouse","itemcode","itemname","itemgroup"], as_index=False)
                    .agg(final_qty=("stockqty","sum"),
                        final_value=("stockvalue","sum"))
            )
        base = final.rename(columns={"final_qty":"ending_qty","final_value":"ending_value"})

        if flow_df.empty:
            # No movement rows after filters: show base with zero movement
            movement = base.copy()
            for c in ["qty_in_K","qty_out_K","abs_qty_K","active_months_K","months_since_move","f2s_qty_K"]:
                movement[c] = 0.0
            movement["movement_class"] = np.where(movement["ending_qty"] > 0, "STAGNANT", "NORMAL")

            out_cols = [
                "warehouse","itemgroup","itemcode","itemname",
                "abs_qty_K","qty_in_K","qty_out_K","active_months_K","months_since_move",
                "ending_qty","ending_value","f2s_qty_K","movement_class"
            ]
            movement = movement[out_cols].sort_values(
                ["warehouse","movement_class","abs_qty_K"], ascending=[True, True, False]
            )
            st.dataframe(movement, use_container_width=True, height=420)
            st.write(common.create_download_link(movement, "movement_analysis_0m.xlsx"), unsafe_allow_html=True)
        else:
            # ---- 4) Timing model: history to cutoff + trailing window K months ----
            y = pd.to_numeric(flow_df["year"], errors="coerce").fillna(0).astype("int64")
            m = pd.to_numeric(flow_df["month"], errors="coerce").fillna(0).astype("int64")
            flow_df["mi"] = y * 12 + m

            cutoff_mi = int(cutoff_year) * 12 + int(cutoff_month)
            flow_all = flow_df[flow_df["mi"] <= cutoff_mi]

            # Trailing window size (months)
            K = st.slider("Trailing window (months) for movement ranking", 3, 12, 6,
                        help="Used for movement intensity (in/out) metrics only.")
            window = flow_all[flow_all["mi"] >= (cutoff_mi - (K - 1))]

            # ---- 5) Aggregate movement over window (INTENSITY) ----
            grp = ["warehouse","itemcode","itemname","itemgroup"]
            window_agg = (
                window.groupby(grp, as_index=False)
                    .agg(qty_in_K=("qty_in","sum"),
                        qty_out_K=("qty_out","sum"),
                        active_months_K=("net_qty", lambda s: (s != 0).sum()))
            )
            window_agg["abs_qty_K"] = window_agg["qty_in_K"] + window_agg["qty_out_K"]

            # ---- 6) Last movement across full history to cutoff (not just window) ----
            moved = flow_all.loc[(flow_all["qty_in"] > 0) | (flow_all["qty_out"] > 0)]
            last_move = (
                moved.groupby(grp, as_index=False)["mi"].max()
                    .rename(columns={"mi":"last_move_mi"})
            )

            # ---- 7) LEFT-JOIN movement onto balances base ----
            agg = (base
                .merge(window_agg, on=grp, how="left")
                .merge(last_move,  on=grp, how="left"))

            # Fill zeros for missing movement; ∞ months for never-moved
            for c in ["qty_in_K","qty_out_K","abs_qty_K","active_months_K"]:
                if c in agg.columns:
                    agg[c] = agg[c].fillna(0.0)
            agg["months_since_move"] = cutoff_mi - agg["last_move_mi"]
            agg["months_since_move"] = agg["months_since_move"].where(agg["last_move_mi"].notna(), np.inf)

            # ---- 8) Ratios & classification ----
            agg["f2s_qty_K"] = (agg["abs_qty_K"] / agg["ending_qty"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Percentiles per (warehouse,itemgroup)
            ref_75 = agg.groupby(["warehouse","itemgroup"])["abs_qty_K"].transform(lambda s: s.quantile(0.75) if len(s) else 0.0)
            low_25 = agg.groupby(["warehouse","itemgroup"])["abs_qty_K"].transform(lambda s: s.quantile(0.25) if len(s) else 0.0)

            agg["movement_class"] = np.where(
                (agg["abs_qty_K"] >= ref_75) & (agg["months_since_move"] <= 1), "FAST",
                np.where(
                    (agg["active_months_K"] == 0) & (agg["ending_qty"] > 0), "STAGNANT",
                    np.where(
                        (agg["abs_qty_K"] <= low_25) | (agg["months_since_move"] >= 3), "SLOW",
                        "NORMAL"
                    )
                )
            )

            # ---- 9) Output ----
            out_cols = [
                "warehouse","itemgroup","itemcode","itemname",
                "abs_qty_K","qty_in_K","qty_out_K","active_months_K","months_since_move",
                "ending_qty","ending_value","f2s_qty_K","movement_class"
            ]
            movement = agg[out_cols].sort_values(
                ["warehouse","movement_class","abs_qty_K"], ascending=[True, True, False]
            )
            st.dataframe(movement, use_container_width=True, height=420)
            st.write(common.create_download_link(movement, f"movement_analysis_{K}m.xlsx"),
                    unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def _build_customer_data_view_options(zid: int,sales_shape: tuple,sales_cols: tuple,sales_df: pd.DataFrame):
    """
    Cached builder for dropdown options to avoid repeated drop_duplicates()
    on every Streamlit rerun.
    """
    def build_code_name(df, code_col, name_col):
        tmp = df[[code_col, name_col]].dropna().drop_duplicates()
        tmp["label"] = tmp[code_col].astype(str) + " - " + tmp[name_col].astype(str)
        return tmp

    cus_opts = build_code_name(sales_df, "cusid", "cusname")
    sp_opts = build_code_name(sales_df, "spid", "spname")
    item_opts = build_code_name(sales_df, "itemcode", "itemname")

    area_opts = sales_df[["area"]].dropna().drop_duplicates()
    area_opts["label"] = area_opts["area"].astype(str)

    return cus_opts, sp_opts, item_opts, area_opts

def display_customer_data_view_page(current_page, zid, data_dict):
    st.header("Customer Data View")

    # -----------------------------
    # Load & prepare data
    # -----------------------------
    sales, returns = common.data_copy_add_columns(
        data_dict["sales"],
        data_dict["return"]
    )

    # -----------------------------
    # Build selector options (cached)
    # -----------------------------
    sales_shape = (sales.shape[0], sales.shape[1])
    sales_cols = tuple(sales.columns)
    cus_opts, sp_opts, item_opts, area_opts = _build_customer_data_view_options(
        zid, sales_shape, sales_cols, sales
    )

    # -----------------------------
    # UI filters
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        customer = st.selectbox(
            "Customer (mandatory)",
            options=[""] + cus_opts["label"].tolist()
        )
        salesman = st.selectbox(
            "Salesman (optional)",
            options=[""] + sp_opts["label"].tolist()
        )

    with col2:
        product = st.selectbox(
            "Product (optional)",
            options=[""] + item_opts["label"].tolist()
        )
        area = st.selectbox(
            "Area (optional)",
            options=[""] + area_opts["label"].tolist()
        )

    # -----------------------------
    # Validation: customer first
    # -----------------------------
    if not customer:
        if salesman or product or area:
            st.warning("Select a customer first to view transactions.")
        return

    cusid = customer.split(" - ")[0]
    spid = salesman.split(" - ")[0] if salesman else None
    itemcode = product.split(" - ")[0] if product else None
    area_val = area if area else None

    # -----------------------------
    # Mask filtering (fast)
    # -----------------------------
    # Only keep columns needed for the final table
    sales_cols_needed = ["date", "voucher", "cusid", "spid", "area", "itemcode", "quantity", "final_sales"]
    ret_cols_needed = ["date", "revoucher", "cusid", "spid", "area", "itemcode", "returnqty", "treturnamt"]

    smask = (sales["cusid"].astype(str) == str(cusid))
    rmask = (returns["cusid"].astype(str) == str(cusid))

    if spid:
        smask &= (sales["spid"].astype(str) == str(spid))
        rmask &= (returns["spid"].astype(str) == str(spid))

    if itemcode:
        smask &= (sales["itemcode"].astype(str) == str(itemcode))
        rmask &= (returns["itemcode"].astype(str) == str(itemcode))

    if area_val:
        smask &= (sales["area"] == area_val)
        rmask &= (returns["area"] == area_val)

    sales_f = sales.loc[smask, sales_cols_needed]
    returns_f = returns.loc[rmask, ret_cols_needed]

    if sales_f.empty and returns_f.empty:
        st.info("No transactions exist for the selected combination.")
        return

    # -----------------------------
    # Normalize to credit/debit rows
    # (table shows codes only, as requested)
    # -----------------------------
    sales_txn = pd.DataFrame({
        "Date": sales_f["date"],
        "Voucher": sales_f["voucher"],
        "Customer Code": sales_f["cusid"],
        "Salesman Code": sales_f["spid"],
        "Area": sales_f["area"],
        "Product Code": sales_f["itemcode"],
        "Qty Sold": sales_f["quantity"],
        "Qty Returned": 0,
        "Sales Value": sales_f["final_sales"],
        "Return Value": 0
    })

    return_txn = pd.DataFrame({
        "Date": returns_f["date"],
        "Voucher": returns_f["revoucher"],
        "Customer Code": returns_f["cusid"],
        "Salesman Code": returns_f["spid"],
        "Area": returns_f["area"],
        "Product Code": returns_f["itemcode"],
        "Qty Sold": 0,
        "Qty Returned": returns_f["returnqty"],
        "Sales Value": 0,
        "Return Value": returns_f["treturnamt"]
    })

    txn = pd.concat([sales_txn, return_txn], ignore_index=True)

    # Replace 0 with '-' for display clarity (credit / debit style)
    display_cols = ["Qty Sold", "Qty Returned", "Sales Value", "Return Value"]

    for col in display_cols:
        txn[col] = txn[col].apply(lambda x: "-" if x == 0 else x)

    # Consistent column order
    txn = txn[
        ["Date", "Voucher", "Customer Code", "Salesman Code", "Area", "Product Code",
         "Qty Sold", "Qty Returned", "Sales Value", "Return Value"]
    ]

    txn = txn.sort_values("Date", ascending=False)

    # -----------------------------
    # Display
    # -----------------------------
    st.dataframe(txn, use_container_width=True)

# -----------------------------
# Basket Analysis v2 (per Bijoy spec)
# -----------------------------

def _table_writeup(title: str, columns: dict[str, str], toggles: list[str] | None = None):
    """
    Render a consistent explanation block under a table.
    columns: {"colname": "meaning", ...}
    toggles: ["Toggle X: meaning", ...]
    """
    st.markdown(f"**{title} — Column Guide**")
    st.markdown("\n".join([f"- `{k}`: {v}" for k, v in columns.items()]))

    if toggles:
        st.markdown("**Controls / Options**")
        st.markdown("\n".join([f"- {t}" for t in toggles]))

def _ensure_datetime(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce")

def _month_name(m: int) -> str:
    try:
        return calendar.month_abbr[int(m)]
    except Exception:
        return str(m)

def _build_order_sets(df: pd.DataFrame, order_id_col: str, key_col: str) -> dict:
    """
    order_id -> set(keys)
    Presence-based (unique keys per order).
    """
    tmp = df[[order_id_col, key_col]].dropna()
    grouped = tmp.groupby(order_id_col)[key_col].agg(lambda x: set(x.astype(str).unique()))
    return grouped.to_dict()

def _anchor_orders(order_sets: dict, anchors: set[str], mode: str) -> set:
    """
    mode: 'ALL' (intersection) or 'ANY' (union)
    """
    if not anchors:
        return set()
    out = set()
    if mode == "ALL":
        for oid, s in order_sets.items():
            if anchors.issubset(s):
                out.add(oid)
    else:
        for oid, s in order_sets.items():
            if s.intersection(anchors):
                out.add(oid)
    return out

def _basket_recommendations_items(df: pd.DataFrame,order_id_col: str,item_col: str,qty_col: str,value_col: str,anchor_items: list[str],anchor_mode: str = "ALL",top_n: int = 200,) -> pd.DataFrame:
    """
    Return ranked recommended items (excluding anchor items) with support/confidence/lift and qty/value extras.
    """
    if df.empty or not anchor_items:
        return pd.DataFrame()

    work = df.copy()
    work[item_col] = work[item_col].astype(str)
    work[order_id_col] = work[order_id_col].astype(str)

    order_sets = _build_order_sets(work, order_id_col, item_col)
    total_orders = len(order_sets)
    if total_orders == 0:
        return pd.DataFrame()

    anchors = set([str(x).split(" - ")[0] for x in anchor_items])
    anchor_orders = _anchor_orders(order_sets, anchors, anchor_mode)
    anchor_orders_n = len(anchor_orders)
    if anchor_orders_n == 0:
        return pd.DataFrame()

    # Item presence counts across all orders
    exploded = []
    for oid, items in order_sets.items():
        for it in items:
            exploded.append((oid, it))
    exp_df = pd.DataFrame(exploded, columns=[order_id_col, item_col])

    item_order_counts = exp_df.groupby(item_col)[order_id_col].nunique()

    # Co-occur counts within anchor orders
    anchor_items_rows = exp_df[exp_df[order_id_col].isin(anchor_orders)]
    co_counts = anchor_items_rows.groupby(item_col)[order_id_col].nunique()

    # Build result (exclude anchors)
    candidates = co_counts.index.difference(pd.Index(list(anchors)))
    if len(candidates) == 0:
        return pd.DataFrame()

    base = pd.DataFrame({
        "itemcode": candidates,
        "orders_with_anchor_and_item": co_counts.loc[candidates].values,
        "orders_with_item": item_order_counts.reindex(candidates).fillna(0).astype(int).values
    })

    base["anchor_orders"] = anchor_orders_n
    base["total_orders"] = total_orders
    base["support"] = base["orders_with_anchor_and_item"] / total_orders
    base["confidence"] = base["orders_with_anchor_and_item"] / anchor_orders_n
    base["item_support"] = base["orders_with_item"] / total_orders
    base["lift"] = np.where(base["item_support"] > 0, base["confidence"] / base["item_support"], np.nan)

    # Qty / value extras inside anchor orders for candidate items
    in_anchor = work[work[order_id_col].isin(anchor_orders)]
    cand_mask = in_anchor[item_col].isin(base["itemcode"])
    in_anchor = in_anchor.loc[cand_mask, [order_id_col, item_col, qty_col, value_col]]

    # Coerce numeric
    in_anchor[qty_col] = pd.to_numeric(in_anchor[qty_col], errors="coerce").fillna(0.0)
    in_anchor[value_col] = pd.to_numeric(in_anchor[value_col], errors="coerce").fillna(0.0)

    agg = in_anchor.groupby(item_col).agg(
        total_qty_when_cooccurs=(qty_col, "sum"),
        avg_qty_when_cooccurs=(qty_col, "mean"),
        total_value_when_cooccurs=(value_col, "sum"),
        avg_value_when_cooccurs=(value_col, "mean"),
    ).reset_index().rename(columns={item_col: "itemcode"})

    base = base.merge(agg, on="itemcode", how="left")

    # Add itemname/itemgroup if available
    meta_cols = [c for c in ["itemname", "itemgroup"] if c in work.columns]
    if meta_cols:
        meta = (
            work[[item_col] + meta_cols]
            .dropna()
            .drop_duplicates(subset=[item_col])
            .rename(columns={item_col: "itemcode"})
        )
        base = base.merge(meta, on="itemcode", how="left")

    # Rank
    base = base.sort_values(["lift", "confidence", "support"], ascending=[False, False, False]).head(top_n)
    return base.reset_index(drop=True)

def _best_months_for_pairs(df: pd.DataFrame,order_id_col: str,month_col: str,item_col: str,anchor_orders: set,itemcodes: list[str],top_k_months: int = 2) -> dict:
    """
    For each itemcode, find the 'best months' based on co-occur frequency within anchor orders.
    Returns dict itemcode -> "Jan, Feb"
    """
    if df.empty or not anchor_orders or not itemcodes:
        return {}

    work = df[[order_id_col, month_col, item_col]].copy()
    work[order_id_col] = work[order_id_col].astype(str)
    work[item_col] = work[item_col].astype(str)
    work = work[work[order_id_col].isin(anchor_orders)]
    work[month_col] = pd.to_numeric(work[month_col], errors="coerce").astype("Int64")

    out = {}
    for code in itemcodes:
        sub = work[work[item_col] == str(code)]
        if sub.empty:
            out[str(code)] = ""
            continue
        cnt = sub.groupby(month_col)[order_id_col].nunique().sort_values(ascending=False)
        months = [m for m in cnt.index.tolist() if pd.notna(m)]
        months = months[:top_k_months]
        out[str(code)] = ", ".join(_month_name(int(m)) for m in months)
    return out

def _basket_recommendations_groups(df: pd.DataFrame,order_id_col: str,group_col: str,qty_col: str,value_col: str,anchor_groups: list[str],anchor_mode: str = "ALL",top_n: int = 200) -> pd.DataFrame:
    """
    Group-to-group basket recommendations.
    """
    if df.empty or not anchor_groups:
        return pd.DataFrame()

    work = df.copy()
    work[group_col] = work[group_col].fillna("").astype(str)
    work[order_id_col] = work[order_id_col].astype(str)

    order_sets = _build_order_sets(work, order_id_col, group_col)
    total_orders = len(order_sets)
    if total_orders == 0:
        return pd.DataFrame()

    anchors = set([str(x) for x in anchor_groups if str(x).strip() != ""])
    anchor_orders = _anchor_orders(order_sets, anchors, anchor_mode)
    anchor_orders_n = len(anchor_orders)
    if anchor_orders_n == 0:
        return pd.DataFrame()

    # Presence counts
    exploded = []
    for oid, groups in order_sets.items():
        for g in groups:
            if g != "":
                exploded.append((oid, g))
    exp_df = pd.DataFrame(exploded, columns=[order_id_col, group_col])
    group_order_counts = exp_df.groupby(group_col)[order_id_col].nunique()

    anchor_rows = exp_df[exp_df[order_id_col].isin(anchor_orders)]
    co_counts = anchor_rows.groupby(group_col)[order_id_col].nunique()

    candidates = co_counts.index.difference(pd.Index(list(anchors)))
    if len(candidates) == 0:
        return pd.DataFrame()

    base = pd.DataFrame({
        "recommended_group": candidates,
        "orders_with_anchor_and_group": co_counts.loc[candidates].values,
        "orders_with_group": group_order_counts.reindex(candidates).fillna(0).astype(int).values
    })
    base["anchor_orders"] = anchor_orders_n
    base["total_orders"] = total_orders
    base["support"] = base["orders_with_anchor_and_group"] / total_orders
    base["confidence"] = base["orders_with_anchor_and_group"] / anchor_orders_n
    base["group_support"] = base["orders_with_group"] / total_orders
    base["lift"] = np.where(base["group_support"] > 0, base["confidence"] / base["group_support"], np.nan)

    # qty/value extras for recommended groups within anchor orders
    in_anchor = work[work[order_id_col].isin(anchor_orders)]
    in_anchor = in_anchor[in_anchor[group_col].isin(base["recommended_group"])]
    in_anchor[qty_col] = pd.to_numeric(in_anchor[qty_col], errors="coerce").fillna(0.0)
    in_anchor[value_col] = pd.to_numeric(in_anchor[value_col], errors="coerce").fillna(0.0)

    agg = in_anchor.groupby(group_col).agg(
        total_qty_when_cooccurs=(qty_col, "sum"),
        total_value_when_cooccurs=(value_col, "sum"),
    ).reset_index().rename(columns={group_col: "recommended_group"})

    base = base.merge(agg, on="recommended_group", how="left")
    base = base.sort_values(["lift", "confidence", "support"], ascending=[False, False, False]).head(top_n)
    return base.reset_index(drop=True)

def display_basket_analysis_page(current_page, zid: str, data_dict: dict, selected_filters: dict):
    st.header("Basket Analysis")

    # Prepare merged datasets (Basket-only merge rule) using navbar filters
    prep = basket_prepare(str(zid), selected_filters or {})
    scope_zids = prep.get("scope_zids", [str(zid)])
    sales_df = prep.get("sales", pd.DataFrame())
    purchase_df = prep.get("purchase", pd.DataFrame())

    # Ensure expected types / columns
    if sales_df is None or sales_df.empty:
        st.warning("No sales data available for Basket Analysis with the selected filters.")
        return

    # Build order_id (zid + DO number)
    sales_df = sales_df.copy()
    sales_df["zid"] = sales_df["zid"].astype(str)
    sales_df["voucher"] = sales_df["voucher"].astype(str)
    sales_df["order_id"] = sales_df["zid"] + "-" + sales_df["voucher"]

    # Parse dates
    if "date" in sales_df.columns:
        sales_df["date"] = _ensure_datetime(sales_df["date"])
    if "month" in sales_df.columns:
        sales_df["month"] = pd.to_numeric(sales_df["month"], errors="coerce").fillna(0).astype(int)

    # -----------------------------
    # Customer Master Audit (top)
    # -----------------------------
    if scope_zids == ["100000", "100001"]:
        with st.expander("Customer Master Audit (100000 vs 100001)", expanded=True):
            c0 = prep.get("cacus_100000", pd.DataFrame()).copy()
            c1 = prep.get("cacus_100001", pd.DataFrame()).copy()

            if c0.empty or c1.empty:
                st.warning("Customer master data not available for audit.")
            else:
                c0["cusid"] = c0["cusid"].astype(str)
                c1["cusid"] = c1["cusid"].astype(str)

                overlap = set(c0["cusid"]).intersection(set(c1["cusid"]))
                only_0 = set(c0["cusid"]) - set(c1["cusid"])
                only_1 = set(c1["cusid"]) - set(c0["cusid"])

                st.write({
                    "overlap_customers": len(overlap),
                    "only_in_100000": len(only_0),
                    "only_in_100001": len(only_1),
                })

                # Compare all shared columns except zid
                cols0 = [c for c in c0.columns if c != "zid"]
                cols1 = [c for c in c1.columns if c != "zid"]
                common_cols = sorted(set(cols0).intersection(set(cols1)))
                if "cusid" not in common_cols:
                    st.error("cacus_simple must include 'cusid' for audit.")
                else:
                    left = c0[common_cols].rename(columns={c: f"{c}_100000" for c in common_cols if c != "cusid"})
                    right = c1[common_cols].rename(columns={c: f"{c}_100001" for c in common_cols if c != "cusid"})

                    merged = left.merge(right, on="cusid", how="inner")

                    mismatch_counts = {}
                    mismatch_mask = pd.Series(False, index=merged.index)
                    for c in common_cols:
                        if c == "cusid":
                            continue
                        a = merged[f"{c}_100000"].astype(str)
                        b = merged[f"{c}_100001"].astype(str)
                        msk = a.ne(b)
                        mismatch_counts[c] = int(msk.sum())
                        mismatch_mask = mismatch_mask | msk

                    st.subheader("Mismatch counts by column")
                    st.dataframe(pd.DataFrame([mismatch_counts]).T.rename(columns={0: "mismatch_count"}))

                    only_mismatches = st.checkbox("Show only mismatches", value=True, key="basket_cacus_only_mismatch")
                    audit_df = merged[mismatch_mask] if only_mismatches else merged
                    st.dataframe(audit_df)

    # -----------------------------
    # Main perspective selector
    # -----------------------------
    perspective = st.radio("Select Perspective", ["Sales Perspective", "Purchase Perspective"], horizontal=True)

    # ==========================================================
    # SALES PERSPECTIVE
    # ==========================================================
    if perspective == "Sales Perspective":
        tabA, tabB, tabC, tabD = st.tabs(["Product(s) Basket", "Customer Pattern", "Area Pattern", "Product Group Basket"])

        # ---- Tab A: Product(s) basket ----
        with tabA:
            st.subheader("Product(s) Basket")

            # Build product options from the (already filtered) sales_df
            prod_meta = (
                sales_df[["itemcode", "itemname"]]
                .dropna()
                .drop_duplicates()
                .sort_values(["itemcode", "itemname"])
            )
            prod_options = (prod_meta["itemcode"].astype(str) + " - " + prod_meta["itemname"].astype(str)).tolist()

            anchor_products = st.multiselect("Select Anchor Products (multi)", prod_options)
            anchor_mode = st.radio("Anchor Rule", ["ALL (intersection)", "ANY (union)"], index=0, horizontal=True)
            anchor_mode_key = "ALL" if anchor_mode.startswith("ALL") else "ANY"

            rank_by = st.selectbox("Rank by", ["lift", "confidence", "support"], index=0)
            show_seasonality = st.toggle("Show seasonality (best months)", value=True)

            if not anchor_products:
                st.info("Select at least one anchor product to see recommendations.")
            else:
                rec = _basket_recommendations_items(
                    df=sales_df,
                    order_id_col="order_id",
                    item_col="itemcode",
                    qty_col="quantity",
                    value_col="totalsales",
                    anchor_items=anchor_products,
                    anchor_mode=anchor_mode_key,
                    top_n=200,
                )

                if rec.empty:
                    st.warning("No anchor orders found (try switching ALL/ANY or widening filters).")
                else:
                    # Add best months (optional)
                    if show_seasonality and "month" in sales_df.columns:
                        # Recompute anchor orders for month calc
                        order_sets = _build_order_sets(sales_df, "order_id", "itemcode")
                        anchors = set([str(x).split(" - ")[0] for x in anchor_products])
                        anchor_orders = _anchor_orders(order_sets, anchors, anchor_mode_key)
                        best = _best_months_for_pairs(
                            df=sales_df,
                            order_id_col="order_id",
                            month_col="month",
                            item_col="itemcode",
                            anchor_orders=anchor_orders,
                            itemcodes=rec["itemcode"].astype(str).head(50).tolist(),
                            top_k_months=2,
                        )
                        rec["best_months"] = rec["itemcode"].astype(str).map(best).fillna("")

                    # Final sort by chosen metric
                    rec = rec.sort_values(rank_by, ascending=False).reset_index(drop=True)

                    # Show KPIs
                    st.write({
                        "anchor_mode": anchor_mode_key,
                        "recommended_items": int(rec.shape[0]),
                    })
                    st.dataframe(rec)
                    _table_writeup(
                        "Product Basket Recommendations",
                        columns={
                            "itemcode": "Product code of the recommended item.",
                            "itemname": "Product name of the recommended item.",
                            "itemgroup": "Product group (itemgroup2) of the recommended item.",
                            "orders_with_anchor": "Number of orders (DO) that contain the selected anchor set.",
                            "orders_with_item": "Number of orders (DO) that contain the recommended item.",
                            "orders_with_anchor_and_item": "Number of orders that contain both the anchor set and the recommended item.",
                            "support": "Share of all orders that contain both anchor and recommended item.",
                            "confidence": "Likelihood the recommended item appears given the anchor order set.",
                            "lift": "Strength above baseline: confidence divided by the overall probability of the item.",
                            "avg_qty_when_cooccurs": "Average quantity of the recommended item within anchor orders where it appears.",
                            "total_qty_when_cooccurs": "Total quantity of the recommended item across anchor orders where it appears.",
                            "avg_value_when_cooccurs": "Average totalsales of the recommended item within anchor orders where it appears.",
                            "total_value_when_cooccurs": "Total totalsales of the recommended item across anchor orders where it appears.",
                            "best_months": "Months where co-occurrence is strongest (if seasonality toggle is ON).",
                        },
                        toggles=[
                            "Anchor Rule: **ALL (intersection)** means orders must contain every selected anchor product. **ANY (union)** means orders may contain at least one anchor product.",
                            "Rank by: Sorts results by lift / confidence / support.",
                            "Show seasonality: Adds best-months signal based on co-occurrence by month."
                        ]
                    )

        # ---- Tab B: Customer pattern ----
        with tabB:
            st.subheader("Customer Pattern")

            if "cusid" not in sales_df.columns or "cusname" not in sales_df.columns:
                st.warning("Customer fields not available in sales data.")
            else:
                cust_meta = (
                    sales_df[["cusid", "cusname"]]
                    .dropna()
                    .drop_duplicates()
                    .sort_values(["cusid", "cusname"])
                )
                cust_options = (cust_meta["cusid"].astype(str) + " - " + cust_meta["cusname"].astype(str)).tolist()
                selected_customer = st.selectbox("Select Customer", [""] + cust_options, index=0)

                if not selected_customer:
                    st.info("Select a customer to see patterns.")
                else:
                    cusid = selected_customer.split(" - ")[0].strip()
                    cdf = sales_df[sales_df["cusid"].astype(str) == cusid].copy()
                    if cdf.empty:
                        st.warning("No sales for this customer in the selected filters.")
                    else:
                        cdf["date"] = _ensure_datetime(cdf["date"])
                        cdf["dow"] = cdf["date"].dt.day_name()
                        cdf["dom"] = cdf["date"].dt.day

                        def dom_bucket(d):
                            if pd.isna(d):
                                return ""
                            d = int(d)
                            if d <= 5:
                                return "01-05"
                            if d <= 10:
                                return "06-10"
                            if d <= 15:
                                return "11-15"
                            if d <= 20:
                                return "16-20"
                            if d <= 25:
                                return "21-25"
                            return "26-31"

                        cdf["dom_bucket"] = cdf["dom"].apply(dom_bucket)

                        # presence by order_id
                        def top_products(group_col):
                            tmp = cdf.copy()
                            tmp["itemcode"] = tmp["itemcode"].astype(str)
                            orders = tmp.groupby([group_col, "itemcode"])["order_id"].nunique().reset_index(name="order_count")
                            qty = tmp.groupby([group_col, "itemcode"])["quantity"].sum().reset_index(name="total_qty")
                            val = tmp.groupby([group_col, "itemcode"])["totalsales"].sum().reset_index(name="total_value")
                            out = orders.merge(qty, on=[group_col, "itemcode"], how="left").merge(val, on=[group_col, "itemcode"], how="left")
                            meta = tmp[["itemcode", "itemname", "itemgroup"]].drop_duplicates(subset=["itemcode"])
                            out = out.merge(meta, on="itemcode", how="left")
                            return out.sort_values([group_col, "order_count"], ascending=[True, False])

                        st.write({
                            "customer_orders": int(cdf["order_id"].nunique()),
                            "customer_total_value": float(pd.to_numeric(cdf["totalsales"], errors="coerce").fillna(0).sum()),
                            "customer_total_qty": float(pd.to_numeric(cdf["quantity"], errors="coerce").fillna(0).sum()),
                        })

                        st.markdown("### By Month")
                        month_tbl = top_products("month")
                        st.dataframe(month_tbl, use_container_width=True)
                        _table_writeup(
                            "Customer Pattern — By Month",
                            columns={
                                "month": "Calendar month (1–12) of the sales date.",
                                "itemcode": "Product code.",
                                "itemname": "Product name.",
                                "itemgroup": "Product group (itemgroup2).",
                                "order_count": "Number of distinct orders (DO) in which the product appears for the selected month.",
                                "total_qty": "Total quantity sold for the product in the selected month.",
                                "total_value": "Total sales value (totalsales) for the product in the selected month."
                            }
                        )

                        st.markdown("### By Day of Week")
                        dow_tbl = top_products("dow")
                        st.dataframe(dow_tbl, use_container_width=True)
                        _table_writeup(
                            "Customer Pattern — By Day of Week",
                            columns={
                                "dow": "Day name (Monday–Sunday) derived from the sales date.",
                                "itemcode": "Product code.",
                                "itemname": "Product name.",
                                "itemgroup": "Product group (itemgroup2).",
                                "order_count": "Number of distinct orders (DO) in which the product appears for the selected day of week.",
                                "total_qty": "Total quantity sold for the product on that day of week.",
                                "total_value": "Total sales value (totalsales) for the product on that day of week."
                            }
                        )

                        st.markdown("### By Day-of-Month Bucket")
                        dom_tbl = top_products("dom_bucket")
                        st.dataframe(dom_tbl, use_container_width=True)
                        _table_writeup(
                            "Customer Pattern — By Day-of-Month Bucket",
                            columns={
                                "dom_bucket": "Day-of-month grouped as 01–05, 06–10, 11–15, 16–20, 21–25, 26–31.",
                                "itemcode": "Product code.",
                                "itemname": "Product name.",
                                "itemgroup": "Product group (itemgroup2).",
                                "order_count": "Number of distinct orders (DO) in which the product appears for the selected day bucket.",
                                "total_qty": "Total quantity sold for the product in that day bucket.",
                                "total_value": "Total sales value (totalsales) for the product in that day bucket."
                            }
                        )


        # ---- Tab C: Area pattern ----
        with tabC:
            st.subheader("Area Pattern (cuscity)")

            area_col = "area" if "area" in sales_df.columns else None
            if not area_col:
                st.warning("Area field not available in sales data.")
            else:
                areas = sorted([a for a in sales_df[area_col].dropna().astype(str).unique().tolist() if a.strip() != ""])
                selected_area = st.selectbox("Select Area", [""] + areas, index=0)
                if not selected_area:
                    st.info("Select an area to see top products.")
                else:
                    adf = sales_df[sales_df[area_col].astype(str) == selected_area].copy()
                    if adf.empty:
                        st.warning("No sales in this area for the selected filters.")
                    else:
                        top = (
                            adf.groupby(["itemcode", "itemname", "itemgroup"])["order_id"]
                            .nunique()
                            .reset_index(name="order_count")
                        )
                        qty = adf.groupby(["itemcode"])["quantity"].sum().reset_index(name="total_qty")
                        val = adf.groupby(["itemcode"])["totalsales"].sum().reset_index(name="total_value")
                        top = top.merge(qty, on="itemcode", how="left").merge(val, on="itemcode", how="left")
                        top = top.sort_values("order_count", ascending=False)
                        st.dataframe(top)
                        _table_writeup(
                        "Area Pattern — Top Products",
                        columns={
                            "area": "Customer city/area (cacus.cuscity) for the selected area filter.",
                            "itemcode": "Product code.",
                            "itemname": "Product name.",
                            "itemgroup": "Product group (itemgroup2).",
                            "order_count": "Number of distinct orders (DO numbers) in which the product appears within the selected area and time range.",
                            "total_qty": "Total quantity sold for the product within the selected area and time range.",
                            "total_value": "Total sales value (totalsales) for the product within the selected area and time range."
                        },
                        toggles=[
                            "Area dropdown: Filters the analysis to customers whose `cuscity` matches the chosen area.",
                            "All results reflect the global navbar filters (date range, salesman, customer, product filters)."
                        ]
                    )


        # ---- Tab D: Product group basket ----
        with tabD:
            st.subheader("Product Group Basket")

            if "itemgroup" not in sales_df.columns:
                st.warning("itemgroup not available in sales data.")
            else:
                groups = sorted([g for g in sales_df["itemgroup"].dropna().astype(str).unique().tolist() if g.strip() != ""])
                anchor_groups = st.multiselect("Select Anchor Product Groups (multi)", groups)

                anchor_mode_g = st.radio("Anchor Rule (Groups)", ["ALL (intersection)", "ANY (union)"], index=0, horizontal=True)
                anchor_mode_g_key = "ALL" if anchor_mode_g.startswith("ALL") else "ANY"
                rank_by_g = st.selectbox("Rank groups by", ["lift", "confidence", "support"], index=0, key="rank_by_group")

                if not anchor_groups:
                    st.info("Select at least one product group.")
                else:
                    grec = _basket_recommendations_groups(
                        df=sales_df,
                        order_id_col="order_id",
                        group_col="itemgroup",
                        qty_col="quantity",
                        value_col="totalsales",
                        anchor_groups=anchor_groups,
                        anchor_mode=anchor_mode_g_key,
                        top_n=200,
                    )
                    if grec.empty:
                        st.warning("No anchor orders found for these groups (try ANY or widen filters).")
                    else:
                        grec = grec.sort_values(rank_by_g, ascending=False).reset_index(drop=True)
                        st.dataframe(grec)
                    _table_writeup(
                        "Product Group Basket — Co-occurring Groups",
                        columns={
                        "recommended_group": "A product group (itemgroup2) that frequently appears in the same order as the selected anchor group(s).",
                        "orders_with_anchor": "Number of distinct orders (DO numbers) that contain the anchor product group set.",
                        "orders_with_group": "Number of distinct orders that contain the recommended group.",
                        "orders_with_anchor_and_group": "Number of distinct orders that contain both the anchor group set and the recommended group.",
                        "support": "Share of all orders that contain both the anchor groups and the recommended group.",
                        "confidence": "Likelihood the recommended group appears given the anchor group set.",
                        "lift": "Strength above baseline: confidence divided by the overall probability of the recommended group appearing in any order.",
                        "total_qty_in_anchor_orders": "Total quantity of items belonging to the recommended group within anchor orders where the group appears.",
                        "total_value_in_anchor_orders": "Total totalsales of items belonging to the recommended group within anchor orders where the group appears.",
                        },
                        toggles=[
                            "Anchor Rule: **ALL (intersection)** means the order must contain at least one item from every selected anchor group. **ANY (union)** means the order must contain at least one item from any selected anchor group.",
                            "Rank by: Sorts group recommendations by lift / confidence / support.",
                            "Presence-based: Group co-occurrence is based on whether a group appears in an order at least once, not the quantity."
                        ]
                    )

    # ==========================================================
    # PURCHASE PERSPECTIVE
    # ==========================================================
    else:
        if purchase_df is None or purchase_df.empty:
            st.warning("No purchase data available for this business scope.")
            return

        # --- Enforce navbar timeline on purchase_df ---
        purchase_df = purchase_df.copy()
        purchase_df["combinedate"] = pd.to_datetime(purchase_df["combinedate"], errors="coerce")

        # Prefer filtering by navbar date range if available
        date_from = selected_filters.get("date_from") or selected_filters.get("start_date")
        date_to   = selected_filters.get("date_to") or selected_filters.get("end_date")

        if date_from and date_to:
            d0 = pd.to_datetime(date_from)
            d1 = pd.to_datetime(date_to)
            purchase_df = purchase_df[(purchase_df["combinedate"] >= d0) & (purchase_df["combinedate"] <= d1)]
        else:
            # If navbar uses years selection instead of explicit dates
            years = selected_filters.get("years") or selected_filters.get("year_list")
            if years:
                years_set = set(int(y) for y in years)
                purchase_df = purchase_df[purchase_df["combinedate"].dt.year.isin(years_set)]

        if str(zid) in ("100000", "100001"):
            purchase_df = purchase_df[
                purchase_df["zid"].astype(str).isin(["100001", "100009"])
            ]
        p = purchase_df.copy()
        p["povoucher"] = p["povoucher"].astype(str)
        p["shipmentname"] = p["shipmentname"].astype(str)
        p["status"] = p["status"].astype(str)
        p["combinedate"] = _ensure_datetime(p["combinedate"])

        # Only Received shipments (phase 1)
        p_received = p[p["status"] == "5-Received"].copy()

        if p_received.empty:
            st.warning("No 'Received' shipments found (status = 5-Received).")
            return

        # --------------------------------------------
        # Build grouped shipment metadata (merge 100001 + 100009 by shipmentname + ship_date)
        # --------------------------------------------
        ship_src = p_received.copy()
        ship_src["zid"] = ship_src["zid"].astype(str)
        ship_src["povoucher"] = ship_src["povoucher"].astype(str)
        ship_src["shipmentname"] = ship_src["shipmentname"].fillna("").astype(str)
        ship_src["combinedate"] = _ensure_datetime(ship_src["combinedate"])

        # Day-level grouping date
        ship_src["ship_date"] = ship_src["combinedate"].dt.date

        # Group key: (shipmentname, ship_date)
        ship_meta = (
            ship_src.groupby(["shipmentname", "ship_date"], as_index=False)
            .agg(
                zids=("zid", lambda s: " + ".join(sorted(set(s.astype(str))))),
                povouchers=("povoucher", lambda s: " + ".join(sorted(set(s.astype(str))))),
                combinedate_min=("combinedate", "min"),
            )
            .sort_values(["combinedate_min", "shipmentname"])
        )

        # Label: "100001 + 100009 | IP--A + IP--B - NAME,YYYY-MM-DD"
        ship_meta["ship_label"] = (
            ship_meta["zids"]
            + " | "
            + ship_meta["povouchers"]
            + " - "
            + ship_meta["shipmentname"]
            + ","
            + ship_meta["ship_date"].astype(str)
        )

        ship_options = ship_meta["ship_label"].tolist()

        selected_ship = st.selectbox(
            "Select Shipment (Merged by shipmentname)",
            [""] + ship_options,
            index=0
        )

        window_days = st.selectbox(
            "Before/After Window (days)",
            [7, 14, 30, 45, 60],
            index=2
        )

        # Resolve selection back to underlying purchase rows
        selected_shipment_df = None
        if selected_ship:
            chosen = ship_meta[ship_meta["ship_label"] == selected_ship]
            if not chosen.empty:
                chosen_name = chosen.iloc[0]["shipmentname"]
                chosen_date = chosen.iloc[0]["ship_date"]

                selected_shipment_df = ship_src[
                    (ship_src["shipmentname"] == chosen_name)
                    & (ship_src["ship_date"] == chosen_date)
                ].copy()


        if not selected_ship:
            st.info("Select a shipment to run the before/after and basket analysis.")
            return

        # Use parsed zid + povoucher selection (prevents collisions across businesses)
        if selected_shipment_df is None or selected_shipment_df.empty:
            st.warning("Shipment not found in Received dataset.")
            return

        # Display label for debug: merged POV list
        pov = " + ".join(sorted(set(selected_shipment_df["povoucher"].astype(str))))
        # Shipment date = earliest combinedate among the merged shipment rows
        combinedate = selected_shipment_df["combinedate"].min()

        # Shipment items = all itemcodes in the merged shipment rows
        ship_items = set(selected_shipment_df["itemcode"].astype(str).unique().tolist())



        # Build before/after windows
        start_before = combinedate - pd.Timedelta(days=int(window_days))
        end_before = combinedate
        start_after = combinedate
        end_after = combinedate + pd.Timedelta(days=int(window_days))

        # Filter sales for before/after
        s = sales_df.copy()
        s["date"] = _ensure_datetime(s["date"])
        before = s[(s["date"] >= start_before) & (s["date"] < end_before)].copy()
        after = s[(s["date"] >= start_after) & (s["date"] <= end_after)].copy()

        st.write({
            "shipment_povoucher": pov,
            "shipment_date_combinedate": str(combinedate.date()) if pd.notna(combinedate) else "",
            "window_days": int(window_days),
            "shipment_items_count": len(ship_items),
        })

        viewA, viewB = st.tabs(["Shipment Impact (Before vs After)", "Shipment Basket + Extra Uplift"])

        # ---- View A ----
        with viewA:
            st.subheader("Shipment Impact: Item-level Before vs After")

            def item_summary(df_in: pd.DataFrame) -> pd.DataFrame:
                if df_in.empty:
                    return pd.DataFrame(columns=["itemcode", "itemname", "qty", "value"])
                tmp = df_in.copy()
                tmp["quantity"] = pd.to_numeric(tmp["quantity"], errors="coerce").fillna(0.0)
                tmp["totalsales"] = pd.to_numeric(tmp["totalsales"], errors="coerce").fillna(0.0)
                out = tmp.groupby(["itemcode", "itemname"], as_index=False).agg(
                    qty=("quantity", "sum"),
                    value=("totalsales", "sum"),
                )
                return out

            bsum = item_summary(before)
            asum = item_summary(after)

            merged = bsum.merge(asum, on=["itemcode", "itemname"], how="outer", suffixes=("_before", "_after")).fillna(0.0)
            merged["delta_qty"] = merged["qty_after"] - merged["qty_before"]
            merged["delta_value"] = merged["value_after"] - merged["value_before"]
            merged["pct_qty_change"] = np.where(merged["qty_before"] > 0, merged["delta_qty"] / merged["qty_before"], np.nan)
            merged["pct_value_change"] = np.where(merged["value_before"] > 0, merged["delta_value"] / merged["value_before"], np.nan)

            # Shipment items table
            # Shipment items table (include ALL shipment items, even if sales are 0)
            ship_base = selected_shipment_df[["itemcode", "itemname"]].copy()
            ship_base["itemcode"] = ship_base["itemcode"].astype(str)
            ship_base["itemname"] = ship_base["itemname"].fillna("").astype(str)
            ship_base = ship_base.drop_duplicates(subset=["itemcode"])

            merged_work = merged.copy()
            merged_work["itemcode"] = merged_work["itemcode"].astype(str)

            ship_tbl = ship_base.merge(
                merged_work,
                on="itemcode",
                how="left",
                suffixes=("_ship", "")
            )

            # Prefer shipment itemname if sales-side itemname is missing
            if "itemname_ship" in ship_tbl.columns and "itemname" in ship_tbl.columns:
                ship_tbl["itemname"] = ship_tbl["itemname"].fillna(ship_tbl["itemname_ship"])
                ship_tbl = ship_tbl.drop(columns=["itemname_ship"], errors="ignore")

            # Fill numeric sales fields with 0
            for c in ["qty_before", "value_before", "qty_after", "value_after", "delta_qty", "delta_value"]:
                if c in ship_tbl.columns:
                    ship_tbl[c] = pd.to_numeric(ship_tbl[c], errors="coerce").fillna(0.0)

            # Recompute pct fields safely
            ship_tbl["pct_qty_change"] = np.where(
                ship_tbl["qty_before"] > 0,
                ship_tbl["delta_qty"] / ship_tbl["qty_before"],
                np.nan
            )
            ship_tbl["pct_value_change"] = np.where(
                ship_tbl["value_before"] > 0,
                ship_tbl["delta_value"] / ship_tbl["value_before"],
                np.nan
            )

            ship_tbl = ship_tbl.sort_values("delta_value", ascending=False)

            st.markdown("**Shipment items (before vs after)**")
            st.dataframe(ship_tbl, use_container_width=True)


            # Non-shipment uplift table
            non_tbl = merged[~merged["itemcode"].astype(str).isin(ship_items)].copy()
            non_tbl = non_tbl.sort_values("delta_value", ascending=False)
            st.markdown("**Non-shipment items uplift (before vs after)**")
            st.dataframe(non_tbl)

            _table_writeup(
                "Shipment Impact — Shipment Items (Before vs After)",
                columns={
                    "itemcode": "Item code for the shipment line (packcode-based mapping already applied where relevant).",
                    "itemname": "Item name.",
                    "qty_before": "Total sales quantity of this item in the **before window** (combinedate − N days to combinedate).",
                    "qty_after": "Total sales quantity of this item in the **after window** (combinedate to combinedate + N days).",
                    "delta_qty": "qty_after − qty_before.",
                    "pct_qty_change": "Percentage change in quantity from before to after. Blank/NA if qty_before is zero.",
                    "value_before": "Total sales value (totalsales) of this item in the before window.",
                    "value_after": "Total sales value (totalsales) of this item in the after window.",
                    "delta_value": "value_after − value_before.",
                    "pct_value_change": "Percentage change in value from before to after. Blank/NA if value_before is zero.",
                },
                toggles=[
                    "Shipment dropdown: Selects a single shipment (povoucher − shipmentname).",
                    "Window (days): Controls the before/after period. Applied symmetrically around `combinedate`.",
                    "Received-only: This analysis is valid only for shipments marked as Received."
                ]
            )


        # ---- View B ----
        with viewB:
            st.subheader("Shipment Basket (after window) + Extra Products Uplift")

            # Find after-window orders that contain any shipment item
            if after.empty:
                st.warning("No sales in the after window.")
            else:
                # Build order sets in after
                after_work = after.copy()
                after_work["itemcode"] = after_work["itemcode"].astype(str)
                order_sets = _build_order_sets(after_work, "order_id", "itemcode")

                shipment_orders = set()
                for oid, items in order_sets.items():
                    if items.intersection(ship_items):
                        shipment_orders.add(oid)

                if not shipment_orders:
                    st.warning("No after-window orders contain shipment items.")
                else:
                    # Build recommendations among non-shipment items, anchored on "shipment item present"
                    # We'll pass anchor_mode ANY with anchor items = shipment items (because condition is any shipment item present)
                    rec = _basket_recommendations_items(
                        df=after_work,
                        order_id_col="order_id",
                        item_col="itemcode",
                        qty_col="quantity",
                        value_col="totalsales",
                        anchor_items=list(ship_items),
                        anchor_mode="ANY",
                        top_n=200,
                    )
                    if rec.empty:
                        st.warning("No extra-product basket results found.")
                    else:
                        # Limit to non-shipment items explicitly
                        rec = rec[~rec["itemcode"].astype(str).isin(ship_items)].copy()
                        st.markdown("**Extra products (not in shipment) most likely to sell with shipment items**")
                        st.dataframe(rec)

                        # Extra products uplift table (top K)
                        top_k = st.slider("Top extra products to compare (before vs after)", min_value=10, max_value=60, value=30, step=5)
                        top_items = rec["itemcode"].astype(str).head(int(top_k)).tolist()

                        def uplift_for(items: list[str]) -> pd.DataFrame:
                            if not items:
                                return pd.DataFrame()
                            b = before[before["itemcode"].astype(str).isin(items)].copy()
                            a = after[after["itemcode"].astype(str).isin(items)].copy()

                            bsum = item_summary(b).rename(columns={"qty": "qty_before", "value": "value_before"})
                            asum = item_summary(a).rename(columns={"qty": "qty_after", "value": "value_after"})
                            u = bsum.merge(asum, on=["itemcode", "itemname"], how="outer").fillna(0.0)
                            u["delta_qty"] = u["qty_after"] - u["qty_before"]
                            u["delta_value"] = u["value_after"] - u["value_before"]
                            u["pct_qty_change"] = np.where(u["qty_before"] > 0, u["delta_qty"] / u["qty_before"], np.nan)
                            u["pct_value_change"] = np.where(u["value_before"] > 0, u["delta_value"] / u["value_before"], np.nan)
                            return u

                        uplift_tbl = uplift_for(top_items)
                        if uplift_tbl.empty:
                            st.warning("No uplift data for the selected extra products in the before/after windows.")
                        else:
                            # Add lift/confidence context from basket rec
                            ctx = rec[["itemcode", "support", "confidence", "lift"]].copy()
                            uplift_tbl = uplift_tbl.merge(ctx, on="itemcode", how="left")
                            uplift_tbl = uplift_tbl.sort_values("delta_value", ascending=False)
                            st.markdown("**Extra products uplift (before vs after) with basket context**")
                            st.dataframe(uplift_tbl)
                            _table_writeup(
                                "Shipment Impact — Non-shipment Items (Before vs After)",
                                columns={
                                    "itemcode": "Item code for products **not** included in the selected shipment.",
                                    "itemname": "Item name.",
                                    "itemgroup": "Product group (itemgroup2).",
                                    "qty_before": "Total quantity sold in the before window.",
                                    "qty_after": "Total quantity sold in the after window.",
                                    "delta_qty": "qty_after − qty_before.",
                                    "pct_qty_change": "Percentage change in quantity from before to after. Blank/NA if qty_before is zero.",
                                    "value_before": "Total sales value (totalsales) in the before window.",
                                    "value_after": "Total sales value (totalsales) in the after window.",
                                    "delta_value": "value_after − value_before.",
                                    "pct_value_change": "Percentage change in value from before to after. Blank/NA if value_before is zero.",
                                },
                                toggles=[
                                    "Ranking (if present): Sorts by delta_value / delta_qty / pct change to highlight the strongest changes.",
                                    "This table is the main test of the hypothesis: whether non-shipment products increase after the shipment arrives."
                                ]
                            )

