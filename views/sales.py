import calendar
import streamlit as st
import pandas as pd
from processing import common, overall_sales
from utils.utils import timed


@timed
def display_overall_sales_analysis_page(current_page, zid, data_dict):
    st.title("Overall Sales Analysis")
    filtered_data, filtered_data_r = common.data_copy_add_columns(data_dict['sales'], data_dict['return'])
    analysis_mode = st.radio("Choose Analysis Mode:",["Overview", "Comparison", "Distributions", "Descriptive Stats", "📈 Order Analytics", "👥 Customer Cycles"],horizontal=True)

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

        compare_by = st.selectbox("Compare By", ["Salesman", "Customer", "Product", "Product Group", "Area", "Reason"])
        metric = st.selectbox("Metric", [
            "Net Sales", "Total Returns", "Total Discounts",
            "Number of Orders", "Number of Returns", "Number of Discounts",
            "Number of Customers", "Number of Customer Returns",
            "Number of Products", "Number of Product Returns",
            "Units Sold", "Units Returned", "Net Units Sold",
        ])

        dimension_column_map = {
            "Salesman":      ("spid",      "spname"),
            "Customer":      ("cusid",     "cusname"),
            "Product":       ("itemcode",  "itemname"),
            "Product Group": ("itemgroup", None),
            "Area":          ("area",      None),
            "Reason":        ("reason",    None),
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

        is_mom = compare_type == "Month vs Month"
        if is_mom:
            selected_display = st.multiselect(
                f"Select {compare_by}(s) to Compare",
                options=display_options,
                default=display_options[:3],
                max_selections=7,
                key="comp_entity_select",
            )
            selected_codes = [x.split(" - ")[0] for x in selected_display] if name_col else selected_display
        else:
            selected_display = st.selectbox(
                f"Select {compare_by} to Filter (leave as '(All)' to aggregate)",
                options=["(All)"] + display_options,
                key="comp_entity_select",
            )
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []

        if compare_type == "Year-over-Year (YOY)" and granularity == "Monthly":
            overall_sales.plot_yoy_monthly_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, selected_codes=selected_codes, metric=metric, selected_years=selected_years, selected_month_names=selected_months)
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Daily":
            overall_sales.plot_yoy_daily_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, selected_codes=selected_codes, metric=metric, selected_years=selected_years, start_date=start_date, end_date=end_date)
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Day of Week":
            overall_sales.plot_yoy_dow_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, selected_codes=selected_codes, metric=metric, selected_years=selected_years, average_or_total=average_or_total_YOY_DOW)
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Day of Month":
            overall_sales.plot_yoy_dom_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, selected_codes=selected_codes, metric=metric, selected_years=selected_years, selected_month_names=selected_months, average_or_total=average_or_total_YOY_DOM)
        elif compare_type == "Month vs Month" and granularity == "Monthly":
            overall_sales.plot_month_vs_month_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, name_col=name_col, selected_codes=selected_codes, metric=metric, selected_months=selected_months)
        elif compare_type == "Month vs Month" and granularity == "Day of Week":
            overall_sales.plot_month_vs_month_dow_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, name_col=name_col, selected_codes=selected_codes, metric=metric, selected_months=selected_months, aggregation_type=average_or_total_MOM_DOW)
        elif compare_type == "Month vs Month" and granularity == "Day of Month":
            overall_sales.plot_month_vs_month_dom_comparison(filtered_data=filtered_data, filtered_data_r=filtered_data_r, code_col=code_col, name_col=name_col, selected_codes=selected_codes, metric=metric, selected_months=selected_months, aggregation_type=average_or_total_MOM_DOM, selected_days=selected_dom_days)
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

    elif analysis_mode == "📈 Order Analytics":
        st.subheader("📈 Order Analytics")
        st.caption("Filters below apply across all sub-sections. Empty = no filter applied.")

        with st.expander("🔍 Entity Filters", expanded=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                sel_areas = st.multiselect("Area",
                    sorted(filtered_data["area"].dropna().unique().tolist()), key="oa_areas")
            with col2:
                sel_salesmen = st.multiselect("Salesman",
                    sorted(filtered_data["spname"].dropna().unique().tolist()), key="oa_salesmen")
            with col3:
                sel_product_groups = st.multiselect("Product Group",
                    sorted(filtered_data["itemgroup"].dropna().unique().tolist()), key="oa_product_groups")
            with col4:
                sel_customers = st.multiselect("Customer",
                    sorted(filtered_data["cusname"].dropna().unique().tolist()), key="oa_customers")
            with col5:
                sel_products = st.multiselect("Product",
                    sorted(filtered_data["itemname"].dropna().unique().tolist()), key="oa_products")

        sub_mode = st.radio(
            "Sub-section",
            ["Order Size Distribution", "Return Size Distribution", "Rolling Average"],
            horizontal=True, key="oa_sub",
        )

        if sub_mode in ("Order Size Distribution", "Return Size Distribution"):
            col1, col2, col3 = st.columns(3)
            with col1:
                value_min = st.number_input("Min Value (optional)", value=None, placeholder="e.g. 1000", key="oa_min")
            with col2:
                value_max = st.number_input("Max Value (optional)", value=None, placeholder="e.g. 100000", key="oa_max")
            with col3:
                nbins = st.number_input("Number of Bins", min_value=5, max_value=500, value=50, key="oa_bins")

            overall_sales.plot_order_size_distribution(
                filtered_data, filtered_data_r,
                sel_areas, sel_salesmen, sel_product_groups, sel_customers, sel_products,
                value_min, value_max, nbins,
                "Order Size" if sub_mode == "Order Size Distribution" else "Return Size",
            )

            if sub_mode == "Order Size Distribution":
                with st.expander("📖 How to read this chart"):
                    st.markdown("""
**What you are looking at:** Each bar represents how many sales vouchers (orders) fall within a given value range. The x-axis is the total order value and the y-axis is the count of orders in that bracket.

**What to look for:**
- **Where the bulk sits** — if most orders cluster in the lower brackets (e.g. 1K–10K), your typical customer is placing small orders. A wide spread or a secondary peak at higher values suggests a two-tier customer base.
- **Long right tail** — a few very large orders can skew averages significantly. These are often key accounts and should be tracked separately.
- **Bracket breakdown** (shown in the table below the chart) — use the count and share columns to understand what proportion of your business volume comes from each size tier. A healthy mix typically shows some volume at every level; over-reliance on one bracket is a concentration risk.
- **Filtering by salesman or area** — if one salesman's distribution skews smaller than others, they may need coaching on upselling or are simply serving a different customer segment.
                    """)
            else:
                with st.expander("📖 How to read this chart"):
                    st.markdown("""
**What you are looking at:** Each bar is a return voucher and its total value. High bars on the right indicate large-value returns.

**What to look for:**
- **Frequency vs. value** — many small-value returns typically reflect routine product exchanges or minor quality issues. A few very large returns are more likely to represent a dispute, a bulk rejection, or a customer churning.
- **Comparing to order size** — if your return distribution closely mirrors your order size distribution, returns may be systematic (e.g. a product line issue). If returns skew larger than typical order sizes, investigate whether whole-batch rejections are occurring.
- **Filtering by product group** — a concentration of returns in one group is a strong signal of a quality or expectation mismatch for that category.
- **Trend over time** — use the sidebar date range to compare return patterns across periods. A widening return distribution over time can indicate growing customer dissatisfaction.
                    """)

        elif sub_mode == "Rolling Average":
            col1, col2 = st.columns(2)
            with col1:
                ra_metric = st.selectbox("Metric", ["Sales", "Net Sales", "Returns"], key="oa_ra_metric")
            with col2:
                ra_windows = st.multiselect("Rolling Windows (days)", [5, 10, 30, 60],
                                            default=[10, 30], key="oa_ra_windows")
            if ra_windows:
                overall_sales.plot_rolling_average_sales(
                    filtered_data, filtered_data_r,
                    sel_areas, sel_salesmen, sel_product_groups, sel_customers, sel_products,
                    ra_windows, ra_metric,
                )
                with st.expander("📖 How to read this chart"):
                    st.markdown("""
**What you are looking at:** The grey bars show daily raw sales (or returns, or net sales depending on the metric selected). The coloured lines are rolling averages over the selected day windows (e.g. 10-day, 30-day).

**What to look for:**
- **Direction of the rolling line** — a rising 30-day average over consecutive months is a clear upward trend, even if individual days are noisy. A flattening line after a spike signals momentum stalling.
- **Short vs. long window divergence** — when a short window (5 or 10 days) crosses above a longer one (30 or 60 days), it is an early signal of acceleration. The reverse — short falling below long — is an early warning of a slowdown.
- **Spikes in daily bars** — isolated tall bars are usually one-off large orders. If they happen regularly for a salesman or area, that customer relationship is driving concentrated revenue.
- **Gaps / flat days** — long stretches of zero bars represent days with no transactions. For a field sales team this is normal (Fridays, holidays), but unexpected flat patches during active weeks deserve investigation.
- **Metric choice:** *Sales* shows gross revenue, *Net Sales* subtracts returns, *Returns* isolates return volume — comparing all three reveals how much of apparent growth is being eroded by returns.
                    """)
            else:
                st.info("Select at least one rolling window.")

    elif analysis_mode == "👥 Customer Cycles":
        _render_customer_cycles(filtered_data)


def _render_customer_cycles(df_sales):
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np

    st.subheader("👥 Customer Cycles")
    st.caption("Analysis is based on the date range and ZID selected in the sidebar.")

    if df_sales is None or df_sales.empty:
        st.warning("No sales data available for the selected period.")
        return

    required = {"cusid", "cusname", "area", "year", "month", "date", "voucher", "altsales"}
    missing = required - set(df_sales.columns)
    if missing:
        st.warning(f"Missing columns: {missing}")
        return

    cycle_mode = st.radio(
        "Section",
        ["📊 Monthly Active Customers", "🔄 Customer Flow", "🏷️ Cycle Profiles", "📐 Area Projection"],
        horizontal=True, key="cc_mode",
    )

    # ── 1. Monthly Active Customers ──────────────────────────────────────────
    if cycle_mode == "📊 Monthly Active Customers":
        st.markdown("#### Monthly Active Customers")
        st.caption("Unique customers who placed at least one order in each month.")

        with st.spinner("Computing..."):
            try:
                long_df, pivot_df = overall_sales.compute_monthly_active_customers(df_sales)
            except Exception as e:
                st.error(f"Error computing MAC: {e}")
                return

        if pivot_df.empty:
            st.info("Not enough data.")
            return

        # ── Heatmap ──
        nat_vals = pivot_df.loc["National"].values.astype(float) if "National" in pivot_df.index else None
        area_rows = pivot_df.drop(index="National", errors="ignore")

        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot_df.values.tolist(),
            x=pivot_df.columns.tolist(),
            y=pivot_df.index.tolist(),
            colorscale="Blues",
            text=pivot_df.values.tolist(),
            texttemplate="%{text}",
            showscale=True,
            hovertemplate="Area: %{y}<br>Month: %{x}<br>Active Customers: %{z}<extra></extra>",
        ))
        fig_heat.update_layout(
            title="Active Customers Heatmap (Area × Month)",
            height=max(300, 60 + len(pivot_df) * 35),
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="", yaxis_title="",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Line chart: national + selectable areas ──
        areas_avail = [a for a in pivot_df.index.tolist() if a != "National"]
        sel_areas_cc = st.multiselect(
            "Areas to highlight on line chart (empty = all)",
            areas_avail, key="cc_mac_areas",
        )
        show_areas = sel_areas_cc if sel_areas_cc else areas_avail

        fig_line = go.Figure()
        if nat_vals is not None:
            fig_line.add_trace(go.Scatter(
                x=pivot_df.columns.tolist(), y=nat_vals,
                mode="lines+markers", name="National",
                line=dict(width=3, color="steelblue"), marker=dict(size=7),
            ))
        for area in show_areas:
            if area in pivot_df.index:
                fig_line.add_trace(go.Scatter(
                    x=pivot_df.columns.tolist(),
                    y=pivot_df.loc[area].values.astype(float),
                    mode="lines+markers", name=area,
                    line=dict(width=1.5), marker=dict(size=5),
                ))
        fig_line.update_layout(
            title="Active Customers — Trend",
            xaxis_title="Month", yaxis_title="Unique Customers",
            height=420, legend=dict(orientation="h", y=-0.3),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # ── Table ──
        st.markdown("**Data table**")
        st.dataframe(pivot_df.style.background_gradient(cmap="Blues", axis=1), use_container_width=True)

        with st.expander("📖 How to read this analysis"):
            st.markdown("""
**What you are looking at:** Each cell shows how many unique customers placed at least one order in that area and month. Darker blue = more active customers. The **National** row sums unique customers across all areas (note: a customer in two areas is counted once nationally but once per area).

**What to look for:**
- **Row consistency** — a row that stays uniformly dark has a stable, reliable customer base. Fading towards the right signals declining engagement in that area.
- **Month-on-month change** — a sudden drop in a cell is worth investigating: was there a stock issue, a pricing change, or did a key salesman leave?
- **Seasonality** — look for recurring lighter months (e.g. certain months every year). This is your natural demand trough and should be factored into target-setting.
- **Area vs. National divergence** — if the National row grows but individual area rows stay flat, the growth is coming from new areas, not from deepening existing ones. The reverse (areas growing, National flat) may indicate cannibalisation or data mapping issues.
- **Benchmark** — use the trailing 3-month average of a healthy area as the baseline when setting targets for a recovering area.
            """)

    # ── 2. Customer Flow ──────────────────────────────────────────────────────
    elif cycle_mode == "🔄 Customer Flow":
        st.markdown("#### Customer Flow — Month-on-Month")
        st.caption(
            "**Retained**: ordered both last month and this month. "
            "**Lost**: ordered last month but not this month. "
            "**New**: first order ever. "
            "**Returned**: ordered before, skipped ≥1 month, now back."
        )

        with st.spinner("Computing..."):
            try:
                flow_df = overall_sales.compute_customer_flow(df_sales)
            except Exception as e:
                st.error(f"Error computing flow: {e}")
                return

        if flow_df.empty:
            st.info("Not enough data.")
            return

        # Area selector
        areas_avail = [a for a in flow_df["area"].unique() if a != "National"]
        sel_area_flow = st.selectbox(
            "View area", ["National"] + sorted(areas_avail), key="cc_flow_area"
        )

        sub = flow_df[flow_df["area"] == sel_area_flow].sort_values(["year", "month"]).iloc[1:]

        if sub.empty:
            st.info("Need at least 2 months of data.")
            return

        months = sub["month_label"].tolist()
        colors = {"retained": "#4CAF50", "new_customers": "#2196F3",
                  "returned": "#FF9800", "lost": "#F44336"}

        fig_stack = go.Figure()
        for col, label, color in [
            ("retained",     "Retained",  colors["retained"]),
            ("new_customers","New",        colors["new_customers"]),
            ("returned",     "Returned",  colors["returned"]),
        ]:
            fig_stack.add_trace(go.Bar(
                x=months, y=sub[col].tolist(), name=label,
                marker_color=color,
            ))
        fig_stack.add_trace(go.Bar(
            x=months, y=[-v for v in sub["lost"].tolist()],
            name="Lost", marker_color=colors["lost"],
        ))
        fig_stack.add_trace(go.Scatter(
            x=months, y=sub["total_active"].tolist(),
            mode="lines+markers", name="Total Active",
            yaxis="y2", line=dict(color="black", width=2), marker=dict(size=6),
        ))
        fig_stack.update_layout(
            barmode="relative",
            title=f"Customer Flow — {sel_area_flow}",
            xaxis_title="Month",
            yaxis=dict(title="Customers (Lost shown negative)"),
            yaxis2=dict(title="Total Active", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.3),
            height=480,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        # Summary table
        display_cols = {
            "month_label": "Month", "total_active": "Total Active",
            "retained": "Retained", "new_customers": "New",
            "returned": "Returned", "lost": "Lost", "net_change": "Net Change",
        }
        st.dataframe(
            sub[list(display_cols.keys())].rename(columns=display_cols).reset_index(drop=True),
            use_container_width=True,
        )

        with st.expander("📖 How to read this analysis"):
            st.markdown("""
**What you are looking at:** For each month pair, the chart decomposes the change in your active customer count into four components. Positive bars stack above the axis; Lost customers are shown below (negative) so the net height of the chart approximates your active customer trajectory.

| Bucket | Meaning | What drives it |
|---|---|---|
| **Retained** | Ordered last month AND this month | Core loyalty; your most stable revenue base |
| **New** | First-ever order | Acquisition effectiveness — salesman reach, new areas |
| **Returned** | Previously ordered, skipped ≥1 month, now back | Win-back success; often seasonal or promotion-driven |
| **Lost** | Ordered last month, absent this month | Churn — could be price, service, stock, or competition |

**Key ratios to watch:**
- **Retention rate** = Retained ÷ previous month's Total Active. A healthy figure is typically above 60–70% for a field sales business. Consistently below 50% indicates churn is outpacing acquisition.
- **Net change** = (Retained + New + Returned) − Lost. A positive net change is growth; sustained negative net change means the customer base is shrinking even if sales appear stable (surviving customers are buying more, masking the erosion).
- **Returned vs. Lost balance** — if Returned consistently tracks close to Lost, your dormant pool is being recycled rather than truly lost. That is a win-back opportunity; if Returned is much smaller than Lost, those customers are not coming back.
            """)

    # ── 3. Cycle Profiles ─────────────────────────────────────────────────────
    elif cycle_mode == "🏷️ Cycle Profiles":
        st.markdown("#### Cycle Profiles")
        window = st.slider("Activity window (months)", 6, 24, 12, step=6, key="cc_window")
        st.caption(
            f"Classification over trailing **{window} months**. "
            "Regular ≥10 mo · Active ≥7 · Occasional ≥4 · Rare ≥1 · Lapsed (ordered before, silent 6+ mo) · Inactive."
        )

        with st.spinner("Classifying customers..."):
            try:
                result_df, n_win = overall_sales.classify_customer_cycles(df_sales, window)
            except Exception as e:
                st.error(f"Error classifying: {e}")
                return

        if result_df.empty:
            st.info("No customer data.")
            return

        # Summary cards: national
        class_order = ["Regular", "Active", "Occasional", "Rare", "Lapsed", "Inactive"]
        class_colors = {"Regular": "#4CAF50", "Active": "#8BC34A",
                        "Occasional": "#FF9800", "Rare": "#FF5722",
                        "Lapsed": "#9C27B0", "Inactive": "#9E9E9E"}

        nat_counts = result_df["class"].value_counts()
        cols_cls = st.columns(len(class_order))
        for i, cls in enumerate(class_order):
            with cols_cls[i]:
                st.metric(cls, int(nat_counts.get(cls, 0)))

        # Pie chart national
        pie_data = [int(nat_counts.get(c, 0)) for c in class_order]
        fig_pie = go.Figure(data=go.Pie(
            labels=class_order, values=pie_data,
            marker_colors=[class_colors[c] for c in class_order],
            hole=0.35,
        ))
        fig_pie.update_layout(title="National — Customer Distribution", height=350,
                               margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

        # Area breakdown stacked bar
        st.markdown("**Area-wise class distribution**")
        area_cls = (
            result_df[result_df["area"] != "Unknown"]
            .groupby(["area", "class"]).size().reset_index(name="count")
        )
        if not area_cls.empty:
            fig_ab = go.Figure()
            for cls in class_order:
                sub_cls = area_cls[area_cls["class"] == cls]
                fig_ab.add_trace(go.Bar(
                    x=sub_cls["area"], y=sub_cls["count"],
                    name=cls, marker_color=class_colors[cls],
                ))
            fig_ab.update_layout(
                barmode="stack", height=380,
                xaxis_title="Area", yaxis_title="Customers",
                legend=dict(orientation="h", y=-0.35),
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_ab, use_container_width=True)

        # Filterable detail table
        st.markdown("**Customer detail**")
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_area = st.multiselect(
                "Filter by Area", sorted(result_df["area"].unique()), key="cc_prof_area"
            )
        with col2:
            filter_class = st.multiselect(
                "Filter by Class", class_order, key="cc_prof_class"
            )
        with col3:
            search_name = st.text_input("Search customer name", key="cc_prof_search")

        disp = result_df.copy()
        if filter_area:
            disp = disp[disp["area"].isin(filter_area)]
        if filter_class:
            disp = disp[disp["class"].isin(filter_class)]
        if search_name:
            disp = disp[disp["cusname"].str.contains(search_name, case=False, na=False)]

        show_cols = {
            "cusname": "Customer", "area": "Area", "class": "Class",
            "active_months": "Active Months", "activity_rate": "Activity Rate",
            "avg_gap_months": "Avg Gap (mo)", "avg_order_value": "Avg Order Value",
            "last_order_date": "Last Order",
        }
        st.dataframe(
            disp[list(show_cols.keys())].rename(columns=show_cols).reset_index(drop=True),
            use_container_width=True,
        )
        st.caption(f"{len(disp):,} customers shown · {n_win}-month window")

        with st.expander("📖 How to read this analysis"):
            st.markdown("""
**What you are looking at:** Each customer is assigned a class based on how many months they were active within the selected trailing window.

| Class | Active months (in window) | What it means |
|---|---|---|
| **Regular** | ≥ 10 of the window months | Reliable, recurring customer — protect at all costs |
| **Active** | 7–9 months | Strong relationship but with occasional gaps |
| **Occasional** | 4–6 months | Orders roughly every other month; relationship is real but not cemented |
| **Rare** | 1–3 months | Transactional; easily lost to a competitor or stock issue |
| **Lapsed** | 0 in last 6 months but ordered before | Was a customer, has gone silent — prime win-back target |
| **Inactive** | 0 orders in entire window, no prior history | Either new lead or data artefact; should not appear in a live customer file |

**How to use the detail table:**
- **Activity Rate** — ordered months ÷ window length. Use this for quota-setting: a salesman whose active customers average 0.4 has significant upsell headroom compared to one averaging 0.7.
- **Avg Gap (mo)** — average months between consecutive orders. A gap of 1.0 means the customer skips every other month. Combined with Avg Order Value, this gives you an expected annual revenue per customer.
- **Last Order** — sort by this to identify customers who are approaching the Lapsed boundary. A customer last seen 4–5 months ago needs immediate attention from the salesman.
- **Area-wise stacked bar** — if one area has a disproportionate share of Lapsed/Rare customers relative to its total, that area's salesman is either over-relying on a few accounts or losing the long tail of smaller customers.
            """)

    # ── 4. Area Projection ────────────────────────────────────────────────────
    elif cycle_mode == "📐 Area Projection":
        st.markdown("#### Area Projection")
        trailing = st.slider(
            "Trailing months for rate calculation", 2, 6, 3, key="cc_trail"
        )
        st.caption(
            f"Uses the last **{trailing} months** of retention/return/new-customer rates "
            "to project next month. **Low** = 25th-percentile rates, **Mid** = mean, **High** = 75th-percentile."
        )

        with st.spinner("Computing projections..."):
            try:
                proj_df = overall_sales.compute_area_projection(df_sales, trailing)
            except Exception as e:
                st.error(f"Error computing projection: {e}")
                return

        if proj_df.empty:
            st.info("Not enough monthly data for projection.")
            return

        # ── Summary cards for National ──
        nat = proj_df[proj_df["Area"] == "National"]
        if not nat.empty:
            nat = nat.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Last Month Active", int(nat["Last Month Active"]))
            c2.metric("Proj Active (low–high)",
                      f"{int(nat['Proj Active (lo)'])}–{int(nat['Proj Active (hi)'])}")
            c3.metric("Proj Active (mid)", int(nat["Proj Active (mid)"]))
            c4.metric("Retention Rate", f"{nat['Retention (mean)']}%")

        # ── Bar chart: projected active range ──
        import plotly.express as px

        fig_range = go.Figure()
        areas   = proj_df["Area"].tolist()
        lo_vals = proj_df["Proj Active (lo)"].tolist()
        mid_vals= proj_df["Proj Active (mid)"].tolist()
        hi_vals = proj_df["Proj Active (hi)"].tolist()

        fig_range.add_trace(go.Bar(
            x=areas, y=[h - l for h, l in zip(hi_vals, lo_vals)],
            base=lo_vals, name="Range (lo–hi)",
            marker_color="rgba(33, 150, 243, 0.35)",
        ))
        fig_range.add_trace(go.Scatter(
            x=areas, y=mid_vals, mode="markers+text",
            name="Mid projection", text=[str(v) for v in mid_vals],
            textposition="outside",
            marker=dict(color="steelblue", size=10, symbol="diamond"),
        ))
        fig_range.add_trace(go.Scatter(
            x=areas, y=proj_df["Last Month Active"].tolist(),
            mode="markers", name="Last Month Actual",
            marker=dict(color="orange", size=8, symbol="circle"),
        ))
        fig_range.update_layout(
            title="Projected Active Customers — Next Month",
            xaxis_title="Area", yaxis_title="Customers",
            height=430, barmode="overlay",
            legend=dict(orientation="h", y=-0.3),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_range, use_container_width=True)

        # ── Sales projection chart ──
        fig_sales = go.Figure()
        s_lo  = proj_df["Proj Sales (lo)"].tolist()
        s_mid = proj_df["Proj Sales (mid)"].tolist()
        s_hi  = proj_df["Proj Sales (hi)"].tolist()
        fig_sales.add_trace(go.Bar(
            x=areas, y=[h - l for h, l in zip(s_hi, s_lo)],
            base=s_lo, name="Sales Range",
            marker_color="rgba(76, 175, 80, 0.35)",
        ))
        fig_sales.add_trace(go.Scatter(
            x=areas, y=s_mid, mode="markers+text",
            name="Mid Sales Proj.",
            text=[f"{v/1000:.0f}K" if v >= 1000 else str(int(v)) for v in s_mid],
            textposition="outside",
            marker=dict(color="green", size=10, symbol="diamond"),
        ))
        fig_sales.update_layout(
            title="Projected Sales — Next Month",
            xaxis_title="Area", yaxis_title="Sales (BDT)",
            height=430, barmode="overlay",
            legend=dict(orientation="h", y=-0.3),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_sales, use_container_width=True)

        # ── Detail table ──
        st.markdown("**Projection detail**")
        st.dataframe(proj_df.reset_index(drop=True), use_container_width=True)
        st.caption(
            "Projection = Last Month Active × Retention Rate + Returns + Avg New Customers. "
            "Sales Projection = Projected Active × Avg Order Value. "
            f"Rates derived from trailing {trailing} months."
        )

        with st.expander("📖 How to read this analysis"):
            st.markdown(f"""
**What you are looking at:** A range-based forecast for next month's active customers and sales per area, using the last **{trailing} months** of observed behaviour.

**How the projection is built:**

| Component | Formula | What it captures |
|---|---|---|
| Retained customers | Last month active × Retention rate | Customers expected to re-order |
| Returned customers | Historical return volume (per month) | Dormant customers expected to come back |
| New customers | Historical new-customer rate (per month) | Organic acquisition |
| Projected sales | Projected active × Average order value | Revenue estimate |

**Low / Mid / High range:**
- **Low** uses the 25th-percentile retention and return rates from the trailing window — a pessimistic but realistic floor.
- **Mid** uses the mean — the most likely outcome if recent trends continue.
- **High** uses the 75th-percentile — assumes conditions are favourable (no stock issues, full salesman attendance, no holidays).

**How to use it for target-setting:**
- Set salesman targets between **Mid** and **High** to stretch performance without making targets unachievable.
- If actual results repeatedly come in below **Low**, the issue is structural (customer base erosion, stock, pricing) and targets need revisiting downward, not upward.
- Areas where the High–Low range is very wide have volatile, unpredictable customer behaviour — these need more frequent monitoring, not just a monthly review.
- **Avg Order Value** drives the sales projection. If an area has strong customer counts but a weak projected sales number, the focus should be on order size (product mix, upsell) rather than acquiring more customers.
            """)




@st.cache_data(show_spinner=False, ttl=86400)
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
