import streamlit as st
import pandas as pd
import calendar
from datetime import datetime
from io import BytesIO
from processing import common, overall_sales
from utils.utils import timed


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
