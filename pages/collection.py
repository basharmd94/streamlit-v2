import streamlit as st
import pandas as pd
import calendar
from datetime import datetime
from processing import common, collection
from utils.utils import timed


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
                filtered_data_c=filtered_data_c_d,  # Pass collection data also
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

        # 6. Call the Plot Function
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
