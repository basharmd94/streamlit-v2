import streamlit as st
import pandas as pd
import calendar
from datetime import datetime
from processing import common, overall_margin
from utils.utils import timed


@timed
def display_margin_analysis_page(current_page, zid, data_dict):
    st.sidebar.title("Overall Margin Analysis")
    filtered_data,filtered_data_r = common.data_copy_add_columns(data_dict['sales'], data_dict['return'])
    analysis_mode = st.radio("Choose Analysis Mode:",["Overview","Comparison","Distributions","Descriptive Stats","Metric Comparison","📈 Order Analytics"],horizontal=True)

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

    elif analysis_mode == "Comparison":
        all_years = sorted(filtered_data["year"].dropna().unique().astype(int).tolist())
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
            filtered_data["month_label"] = (filtered_data["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + filtered_data["year"].astype(str))
            month_options = sorted(filtered_data["month_label"].dropna().unique().tolist())
            selected_months = st.multiselect("Select Months to Compare", options=month_options, default=month_options[:3])
            if granularity == 'Day of Week':
                average_or_total_MOM_DOW = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
            elif granularity == 'Day of Month':
                average_or_total_MOM_DOM = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
                day_options = list(range(1, 32))
                selected_dom_days = st.multiselect("Select Days (leave empty to include all days)", options=day_options)

        compare_by = st.selectbox("Compare By", ["Salesman", "Customer", "Product", "Product Group", "Area"])
        metric = st.selectbox("Metric", ["Net Sales", "Total Returns", "Total Discounts", "Net Margin", "Net Units Sold"])
        dimension_column_map = {"Salesman": ("spid","spname"), "Customer": ("cusid","cusname"), "Product": ("itemcode","itemname"), "Product Group": ("itemgroup",None), "Area": ("area",None)}
        code_col, name_col = dimension_column_map[compare_by]
        if name_col and name_col in filtered_data.columns and code_col in filtered_data.columns:
            filtered_sub = filtered_data[[code_col, name_col]].dropna().drop_duplicates()
            filtered_sub["combined"] = filtered_sub[code_col].astype(str) + " - " + filtered_sub[name_col].astype(str)
            display_options = sorted(filtered_sub["combined"].tolist())
        elif code_col in filtered_data.columns:
            display_options = sorted(filtered_data[code_col].dropna().unique().astype(str).tolist())
        else:
            display_options = []
        is_mom = compare_type == "Month vs Month"
        if is_mom:
            selected_display = st.multiselect(f"Select {compare_by}(s) to Compare", options=display_options, default=display_options[:3], max_selections=7, key="comp_entity_select")
            selected_codes = [x.split(" - ")[0] for x in selected_display] if name_col else selected_display
        else:
            selected_display = st.selectbox(f"Select {compare_by} to Filter (leave as '(All)' to aggregate)", options=["(All)"] + display_options, key="comp_entity_select")
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []
        if compare_type == "Year-over-Year (YOY)" and granularity == "Monthly":
            overall_margin.plot_yoy_monthly_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, selected_codes=selected_codes, metric=metric, selected_years=selected_years, selected_month_names=selected_months)
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Daily":
            overall_margin.plot_yoy_daily_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, selected_codes=selected_codes, metric=metric, selected_years=selected_years, start_date=start_date, end_date=end_date)
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Day of Week":
            overall_margin.plot_yoy_dow_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, selected_codes=selected_codes, metric=metric, selected_years=selected_years, average_or_total=average_or_total_YOY_DOW)
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Day of Month":
            overall_margin.plot_yoy_dom_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, selected_codes=selected_codes, metric=metric, selected_years=selected_years, selected_month_names=selected_months, average_or_total=average_or_total_YOY_DOM)
        elif compare_type == "Month vs Month" and granularity == "Monthly":
            overall_margin.plot_month_vs_month_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, name_col=name_col, selected_codes=selected_codes, metric=metric, selected_months=selected_months)
        elif compare_type == "Month vs Month" and granularity == "Day of Week":
            overall_margin.plot_month_vs_month_dow_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, name_col=name_col, selected_codes=selected_codes, metric=metric, selected_months=selected_months, aggregation_type=average_or_total_MOM_DOW)
        elif compare_type == "Month vs Month" and granularity == "Day of Month":
            overall_margin.plot_month_vs_month_dom_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, name_col=name_col, selected_codes=selected_codes, metric=metric, selected_months=selected_months, aggregation_type=average_or_total_MOM_DOM, selected_days=selected_dom_days)
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
            selected_group = st.selectbox("Group By", ["Customer", "Product", "Salesman", "Area", "Product Group"])
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

        compare_by = st.selectbox("Compare By", ["Salesman", "Customer", "Product", "Product Group", "Area"])

        metric_choices = [
            "Net Sales", "Total Returns", "Total Discounts", "Net Margin", "Net Units Sold",
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
            "Salesman":      ("spid",      "spname"),
            "Customer":      ("cusid",     "cusname"),
            "Product":       ("itemcode",  "itemname"),
            "Product Group": ("itemgroup", None),
            "Area":          ("area",      None),
        }

        code_col, name_col = dimension_column_map[compare_by]

        if name_col and name_col in filtered_data.columns and code_col in filtered_data.columns:
            filtered_sub = filtered_data[[code_col, name_col]].dropna().drop_duplicates()
            filtered_sub["combined"] = filtered_sub[code_col].astype(str) + " - " + filtered_sub[name_col].astype(str)
            display_options = sorted(filtered_sub["combined"].tolist())
        elif code_col in filtered_data.columns:
            display_options = sorted(filtered_data[code_col].dropna().unique().astype(str).tolist())
        else:
            display_options = []

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

    elif analysis_mode == "📈 Order Analytics":
        st.subheader("📈 Order Analytics")
        st.caption("Filters below apply across all sub-sections. Empty = no filter applied.")

        with st.expander("🔍 Entity Filters", expanded=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                sel_areas = st.multiselect("Area",
                    sorted(filtered_data["area"].dropna().unique().tolist()), key="moa_areas")
            with col2:
                sel_salesmen = st.multiselect("Salesman",
                    sorted(filtered_data["spname"].dropna().unique().tolist()), key="moa_salesmen")
            with col3:
                sel_product_groups = st.multiselect("Product Group",
                    sorted(filtered_data["itemgroup"].dropna().unique().tolist()), key="moa_product_groups")
            with col4:
                sel_customers = st.multiselect("Customer",
                    sorted(filtered_data["cusname"].dropna().unique().tolist()), key="moa_customers")
            with col5:
                sel_products = st.multiselect("Product",
                    sorted(filtered_data["itemname"].dropna().unique().tolist()), key="moa_products")

        sub_mode = st.radio(
            "Sub-section",
            ["Margin vs Order Scatter", "Rolling Average"],
            horizontal=True, key="moa_sub",
        )

        if sub_mode == "Margin vs Order Scatter":
            color_by = st.selectbox(
                "Color By", ["(None)", "Salesman", "Area", "Product Group", "Customer"],
                key="moa_color_by",
            )
            overall_margin.plot_margin_order_scatter(
                filtered_data,
                sel_areas, sel_salesmen, sel_product_groups, sel_customers, sel_products,
                color_by,
            )

        elif sub_mode == "Rolling Average":
            ra_windows = st.multiselect("Rolling Windows (days)", [5, 10, 30, 60],
                                        default=[10, 30], key="moa_ra_windows")
            if ra_windows:
                overall_margin.plot_rolling_margin_average(
                    filtered_data,
                    sel_areas, sel_salesmen, sel_product_groups, sel_customers, sel_products,
                    ra_windows,
                )
            else:
                st.info("Select at least one rolling window.")
