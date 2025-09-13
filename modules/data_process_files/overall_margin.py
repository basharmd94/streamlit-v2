import streamlit as st
import pandas as pd
from modules.data_process_files import common
pd.set_option('display.float_format', '{:.2f}'.format)
from utils.utils import timed
import plotly.express as px
import calendar

### for display overall analysis
@timed
def calculate_summary_statistics(filtered_data, filtered_data_r):
    """
    Calculate summary of margin data for the filtered data.

    Args:
    - filtered_data: Filtered sales data.
    - filtered_data_r: Filtered returns data.

    Returns:
    - Dictionary containing the summary statistics.
    """
    return {
        "Net Sales": filtered_data['final_sales'].sum().round(2) - filtered_data_r['treturnamt'].sum().round(2),
        "Total Returns": filtered_data_r['treturnamt'].sum().round(2),
        "Total Discounts": filtered_data['proddiscount'].sum().round(2),
        "Net Margin": filtered_data['gross_margin'].sum().round(2) - filtered_data_r['treturnamt'].sum().round(2),
    }

@timed
def display_summary_statistics(stats):
    """
    Display summary statistics in the Streamlit app in 4 columns.

    Args:
    - stats: Dictionary containing the summary statistics.
    """
    st.sidebar.title("Overall Margin Analysis")

    # Create 4 columns
    col1, col2, col3, col4 = st.columns(4)

    # Split stats into 4 chunks
    stats_items = list(stats.items())
    chunk_size = (len(stats_items) + 3) // 4  # ensure balanced chunks

    chunks = [
        stats_items[0:chunk_size],
        stats_items[chunk_size:2*chunk_size],
        stats_items[2*chunk_size:3*chunk_size],
        stats_items[3*chunk_size:]
    ]

    for col, chunk in zip([col1, col2, col3, col4], chunks):
        with col:
            for stat_name, value in chunk:
                st.markdown(f"**{stat_name}:** {value:,.2f}")

@timed
def display_cross_relation_pivot(filtered_data, filtered_data_r, current_page):
    st.subheader("🔁 Cross Relation Analysis")

    column_options = {
        'Salesman': ['spid', 'spname'],
        'Customer': ['cusid', 'cusname'],
        'Product': ['itemcode', 'itemname'],
        'Product Group': ['itemgroup'],
        'Area': ['area']
    }

    metric_options = [
        "Net Sales", 
        "Total Returns", 
        "Total Discounts",
        "Net Margin"
    ]

    col1, col2, col3 = st.columns(3)
    with col1:
        first_selection = st.selectbox("Select First Column List", list(column_options.keys()), index=0, key="cross_first_col")
    with col2:
        second_selection = st.selectbox("Select Second Column List", list(column_options.keys()), index=1, key="cross_second_col")
    with col3:
        selected_metric = st.selectbox("Select Metric", metric_options, key="cross_metric")

    first_column_list = column_options[first_selection]
    second_column_list = column_options[second_selection]

    try:
        pivot_args = {
            "metric": selected_metric,
            "index": first_column_list,
            "column": second_column_list
        }
        pivot_table_2 = common.net_pivot(filtered_data, filtered_data_r, pivot_args, current_page)
        st.markdown(f"**{selected_metric} by {first_selection} vs {second_selection}**")
        st.write(pivot_table_2)

    except Exception as e:
        st.error(f"Error generating cross relation pivot: {e}")

@timed
def display_entity_metric_pivot(filtered_data, filtered_data_r, current_page):
    st.subheader("📊 Pivot Table Analysis")

    entity_options = {
        "Salesman": ["spid", "spname"],
        "Customer": ["cusid", "cusname"],
        "Product": ["itemcode", "itemname"],
        "Product Group": ["itemgroup"],
        "Area": ["area"]
    }

    metric_options = [
        "Net Sales", 
        "Total Returns", 
        "Total Discounts",
        "Net Margin"
    ]

    col1, col2 = st.columns(2)
    with col1:
        selected_entity = st.selectbox("Select Entity", list(entity_options.keys()))
    with col2:
        selected_metric = st.selectbox("Select Metric", metric_options)

    index_columns = entity_options[selected_entity]

    pivot_args = {
        "metric": selected_metric,
        "index": index_columns,
        "column": ["year", "month"]
    }

    try:
        pivot_table = common.net_pivot(filtered_data, filtered_data_r, pivot_args, current_page)
        st.markdown(f"**{selected_metric} by {selected_entity}**")
        st.write(pivot_table)
    except Exception as e:
        st.error(f"Error generating pivot table: {e}")

@timed
def plot_net(data1,data2,xaxis,yaxis1,yaxis2,bartitle,current_page):

    grouped_data,yaxis = common.net_vertical(data1,data2,xaxis,yaxis1,yaxis2,current_page)

    # Create the bar plot using Plotly
    fig = px.bar(grouped_data, x='xaxis', y=yaxis, title=bartitle, labels={'xaxis': 'Month-Year', yaxis: 'Net Value'})
    fig.update_layout(xaxis_tickangle=-45)  # better readability
    # Display the plot in Streamlit
    st.plotly_chart(fig,use_container_width=True)

@timed
def plot_net_margin(data1,data2,xaxis,yaxis1,yaxis2,bartitle,current_page):

    grouped_data,yaxis = common.net_vertical(data1,data2,xaxis,yaxis1,yaxis2,current_page)

    # Create the bar plot using Plotly
    fig = px.bar(grouped_data, x='xaxis', y=yaxis, title=bartitle, labels={'xaxis': 'Month-Year', yaxis: 'Net Value'})
    fig.update_layout(xaxis_tickangle=-45)  # better readability
    # Display the plot in Streamlit
    st.plotly_chart(fig,use_container_width=True)

@timed
def plot_total_returns(filtered_data_r, current_page):
    df = filtered_data_r.groupby(['year', 'month'])["treturnamt"].sum().reset_index()
    df["month"] = df["month"].astype(int).astype(str).str.zfill(2)
    df["x_label"] = df["year"].astype(str) + "-" + df["month"]

    fig = px.bar(
        df,
        x="x_label",
        y="treturnamt",
        title="Total Returns Over Time",
        labels={"x_label": "Year-Month", "treturnamt": "Total Returns"}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

@timed
def plot_total_discounts(filtered_data, current_page):
    df = filtered_data.groupby(['year', 'month'])["proddiscount"].sum().reset_index()
    df["month"] = df["month"].astype(int).astype(str).str.zfill(2)
    df["x_label"] = df["year"].astype(str) + "-" + df["month"]

    fig = px.bar(
        df,
        x="x_label",
        y="proddiscount",
        title="Total Discounts Over Time",
        labels={"x_label": "Year-Month", "proddiscount": "Total Discounts"}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

@timed
def plot_yoy_monthly_comparison(filtered_data,filtered_data_r,code_col,selected_codes,metric,selected_years,selected_month_names):

    df_sales = filtered_data.copy()
    df_returns = filtered_data_r.copy()

    # Filter entity
    if selected_codes:
        df_sales = df_sales[df_sales[code_col].isin(selected_codes)]
        df_returns = df_returns[df_returns[code_col].isin(selected_codes)]

    # Map months
    month_name_to_num = {calendar.month_abbr[i]: i for i in range(1, 13)}
    selected_months = [month_name_to_num[m] for m in selected_month_names]

    df_sales = df_sales[df_sales["year"].isin(selected_years) & df_sales["month"].isin(selected_months)]
    df_returns = df_returns[df_returns["year"].isin(selected_years) & df_returns["month"].isin(selected_months)]

    # Metric logic
    if metric == "Net Sales":
        sales_grouped = df_sales.groupby(["year", "month"])["final_sales"].sum().reset_index()
        returns_grouped = df_returns.groupby(["year", "month"])["treturnamt"].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=["year", "month"], how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["final_sales"] - df["treturnamt"]
    elif metric == "Total Returns":
        df = df_returns.groupby(["year", "month"])["treturnamt"].sum().reset_index(name="value")
    elif metric == "Net Margin":
        sales_grouped = df_sales.groupby(["year", "month"])["gross_margin"].sum().reset_index()
        returns_grouped = df_returns.groupby(["year", "month"])["treturnamt"].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=["year", "month"], how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["gross_margin"] - df["treturnamt"]
    elif metric == "Total Discounts":
        df = df_sales.groupby(["year", "month"])["proddiscount"].sum().reset_index(name="value")
    else:
        st.error("Unsupported metric.")
        return

    # Add month name
    df["month_name"] = df["month"].apply(lambda x: calendar.month_abbr[int(x)])

    month_abbr_order = [calendar.month_abbr[i] for i in range(1, 13)]
    df["month_name"] = pd.Categorical(df["month_name"], categories=month_abbr_order, ordered=True)
    df = df.sort_values("month_name")

    # Final grouping for bar chart
    pivot = df.pivot(index="month_name", columns="year", values="value").fillna(0)
    df_long = pivot.reset_index().melt(id_vars="month_name", var_name="year", value_name="value")
    # Bar Chart
    fig = px.bar(
            df_long,
            x="month_name",
            y="value",
            color="year",
            barmode="group",
            title=f"{metric} Comparison by Month (YOY)",
            labels={"value": metric, "month_name": "Month"}
        )
    
    st.plotly_chart(fig, use_container_width=True)

    # Show Table
    st.markdown("### 📋 Corresponding Data")
    st.dataframe(pivot.style.format("{:.2f}"), use_container_width=True)

@timed
def plot_yoy_daily_comparison(filtered_data, filtered_data_r, code_col, selected_codes, metric, selected_years, start_date, end_date):
    df_sales = filtered_data.copy()
    df_returns = filtered_data_r.copy()

    # Apply filters for selected entity
    if selected_codes:
        df_sales = df_sales[df_sales[code_col].isin(selected_codes)]
        df_returns = df_returns[df_returns[code_col].isin(selected_codes)]

    # Convert date
    df_sales["date"] = pd.to_datetime(df_sales["date"])
    df_returns["date"] = pd.to_datetime(df_returns["date"])

    # Filter only selected years
    df_sales = df_sales[df_sales["year"].isin(selected_years)]
    df_returns = df_returns[df_returns["year"].isin(selected_years)]

    # Extract day and month to create a day-month key
    df_sales["day_month"] = df_sales["date"].dt.strftime("%m-%d")
    df_returns["day_month"] = df_returns["date"].dt.strftime("%m-%d")

    # Determine selected day-month keys based on user-selected range
    selected_daymonths = pd.date_range(start=start_date, end=end_date).strftime("%m-%d").tolist()
    df_sales = df_sales[df_sales["day_month"].isin(selected_daymonths)]
    df_returns = df_returns[df_returns["day_month"].isin(selected_daymonths)]

    # Metric Logic
    if metric == "Net Sales":
        sales_grouped = df_sales.groupby(["year", "day_month"])["final_sales"].sum().reset_index()
        returns_grouped = df_returns.groupby(["year", "day_month"])["treturnamt"].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=["year", "day_month"], how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["final_sales"] - df["treturnamt"]
    elif metric == "Net Margin":
        sales_grouped = df_sales.groupby(["year", "day_month"])["gross_margin"].sum().reset_index()
        returns_grouped = df_returns.groupby(["year", "day_month"])["treturnamt"].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=["year", "day_month"], how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["gross_margin"] - df["treturnamt"]
    elif metric == "Total Returns":
        df = df_returns.groupby(["year", "day_month"])["treturnamt"].sum().reset_index(name="value")
    elif metric == "Total Discounts":
        df = df_sales.groupby(["year", "day_month"])["proddiscount"].sum().reset_index(name="value")
    else:
        st.error("Unsupported metric.")
        return

    # Format and sort by datetime instead of string
    df["day_month_fmt"] = pd.to_datetime(df["day_month"], format="%m-%d", errors="coerce")
    df = df.dropna(subset=["day_month_fmt"]).sort_values(by=["day_month_fmt", "year"])
    df["day_month_label"] = df["day_month_fmt"].dt.strftime("%b %d")

    # Pivot and melt for bar plot
    pivot = df.pivot(index="day_month_label", columns="year", values="value").fillna(0)
    df_long = pivot.reset_index().melt(id_vars="day_month_label", var_name="year", value_name="value")

    # Bar Chart
    fig = px.bar(
        df_long,
        x="day_month_label",
        y="value",
        color="year",
        barmode="group",
        title=f"{metric} Comparison by Day (YOY)",
        labels={"value": metric, "day_month_label": "Date"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Corresponding Data")
    st.dataframe(pivot.style.format("{:.2f}"), use_container_width=True)

@timed
def plot_yoy_dow_comparison(filtered_data, filtered_data_r, code_col, selected_codes, metric, selected_years, average_or_total):
    df_sales = filtered_data.copy()
    df_returns = filtered_data_r.copy()

    # Apply filters for selected entity
    if selected_codes:
        df_sales = df_sales[df_sales[code_col].isin(selected_codes)]
        df_returns = df_returns[df_returns[code_col].isin(selected_codes)]

    # Convert date
    df_sales["date"] = pd.to_datetime(df_sales["date"])
    df_returns["date"] = pd.to_datetime(df_returns["date"])

    # Filter only selected years
    df_sales = df_sales[df_sales["year"].isin(selected_years)]
    df_returns = df_returns[df_returns["year"].isin(selected_years)]

    # Add weekday column
    df_sales["weekday"] = df_sales["date"].dt.day_name()
    df_returns["weekday"] = df_returns["date"].dt.day_name()

    # Ensure correct weekday order for Bangladesh (Saturday to Friday)
    week_order = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    # Metric Logic
    if metric == "Net Sales":
        sales_grouped = df_sales.groupby(["year", "weekday"])['final_sales'].sum().reset_index()
        returns_grouped = df_returns.groupby(["year", "weekday"])['treturnamt'].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=["year", "weekday"], how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["final_sales"] - df["treturnamt"]
    elif metric == "Net Margin":
        sales_grouped = df_sales.groupby(["year", "weekday"])['gross_margin'].sum().reset_index()
        returns_grouped = df_returns.groupby(["year", "weekday"])['treturnamt'].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=["year", "weekday"], how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["gross_margin"] - df["treturnamt"]
    elif metric == "Total Returns":
        df = df_returns.groupby(["year", "weekday"])["treturnamt"].sum().reset_index(name="value")
    elif metric == "Total Discounts":
        df = df_sales.groupby(["year", "weekday"])["proddiscount"].sum().reset_index(name="value")

    else:
        st.error("Unsupported metric.")
        return

    # Normalize if average selected
    if average_or_total == "Average":
        day_counts = df_sales.groupby(["year", "weekday"]).size().reset_index(name="day_count")
        df = pd.merge(df, day_counts, on=["year", "weekday"], how="left")
        df["value"] = df["value"] / df["day_count"]

    # Sort weekdays by Bangladesh calendar order
    df["weekday"] = pd.Categorical(df["weekday"], categories=week_order, ordered=True)
    df = df.sort_values(["weekday", "year"])

    # Pivot and reshape for plot
    pivot = df.pivot(index="weekday", columns="year", values="value").fillna(0)
    df_long = pivot.reset_index().melt(id_vars="weekday", var_name="year", value_name="value")

    # Bar Chart
    fig = px.bar(
        df_long,
        x="weekday",
        y="value",
        color="year",
        barmode="group",
        title=f"{metric} Comparison by Day of Week (YOY)",
        labels={"value": metric, "weekday": "Day of Week"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Corresponding Data")
    st.dataframe(pivot.style.format("{:.2f}"), use_container_width=True)

@timed
def plot_yoy_dom_comparison(filtered_data, filtered_data_r, code_col, selected_codes, metric, selected_years, selected_month_names, average_or_total):
    # Map month names to numbers
    month_name_to_num = {calendar.month_abbr[i]: i for i in range(1, 13)}
    selected_months = [month_name_to_num[m] for m in selected_month_names]

    df_sales = filtered_data.copy()
    df_returns = filtered_data_r.copy()

    # Filter selected entity, year, month
    if selected_codes:
        df_sales = df_sales[df_sales[code_col].isin(selected_codes)]
        df_returns = df_returns[df_returns[code_col].isin(selected_codes)]

    df_sales = df_sales[df_sales["year"].isin(selected_years) & df_sales["month"].isin(selected_months)]
    df_returns = df_returns[df_returns["year"].isin(selected_years) & df_returns["month"].isin(selected_months)]

    # Convert to datetime
    df_sales["date"] = pd.to_datetime(df_sales["date"])
    df_returns["date"] = pd.to_datetime(df_returns["date"])

    df_sales["day"] = df_sales["date"].dt.day
    df_returns["day"] = df_returns["date"].dt.day

    # Aggregation Logic
    if metric == "Net Sales":
        sales_grouped = df_sales.groupby(["year", "day"])["final_sales"].sum().reset_index()
        returns_grouped = df_returns.groupby(["year", "day"])["treturnamt"].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=["year", "day"], how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["final_sales"] - df["treturnamt"]
    elif metric == "Net Margin":
        sales_grouped = df_sales.groupby(["year", "day"])["gross_margin"].sum().reset_index()
        returns_grouped = df_returns.groupby(["year", "day"])["treturnamt"].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=["year", "day"], how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["gross_margin"] - df["treturnamt"]
    elif metric == "Total Returns":
        df = df_returns.groupby(["year", "day"])["treturnamt"].sum().reset_index(name="value")
    elif metric == "Total Discounts":
        df = df_sales.groupby(["year", "day"])["proddiscount"].sum().reset_index(name="value")
    else:
        st.error("Unsupported metric.")
        return

    # Normalize by number of months if average
    if average_or_total == "Average":
        month_counts = df_sales.groupby("year")["month"].nunique().to_dict()
        df["value"] = df.apply(lambda row: row["value"] / month_counts.get(row["year"], 1), axis=1)

    # Prepare for plotting
    pivot = df.pivot(index="day", columns="year", values="value").fillna(0)
    df_long = pivot.reset_index().melt(id_vars="day", var_name="year", value_name="value")

    # Bar Chart
    fig = px.bar(
        df_long,
        x="day",
        y="value",
        color="year",
        barmode="group",
        title=f"{metric} by Day of Month (YOY)",
        labels={"value": metric, "day": "Day of Month"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Corresponding Data")
    st.dataframe(pivot.style.format("{:.2f}"), use_container_width=True)

@timed
def plot_month_vs_month_comparison(filtered_data,filtered_data_r,code_col,name_col,selected_codes,metric,selected_months):
    df_sales = filtered_data.copy()
    df_returns = filtered_data_r.copy()

    # Create a month_label column
    df_sales["month_label"] = df_sales["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_sales["year"].astype(str)
    df_returns["month_label"] = df_returns["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_returns["year"].astype(str)
    # Filter for selected months
    df_sales = df_sales[df_sales["month_label"].isin(selected_months)]
    df_returns = df_returns[df_returns["month_label"].isin(selected_months)]
    # Filter for selected entities
    if selected_codes:
        df_sales = df_sales[df_sales[code_col].isin(selected_codes)]
        df_returns = df_returns[df_returns[code_col].isin(selected_codes)]
    # Metric logic
    if metric == "Net Sales":
        sales_grouped = df_sales.groupby(["month_label", code_col])["final_sales"].sum().reset_index()
        returns_grouped = df_returns.groupby(["month_label", code_col])["treturnamt"].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=["month_label", code_col], how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["final_sales"] - df["treturnamt"]
    elif metric == "Net Margin":
        sales_grouped = df_sales.groupby(["month_label", code_col])["gross_margin"].sum().reset_index()
        returns_grouped = df_returns.groupby(["month_label", code_col])["treturnamt"].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=["month_label", code_col], how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["gross_margin"] - df["treturnamt"]
    elif metric == "Total Returns":
        df = df_returns.groupby(["month_label", code_col])["treturnamt"].sum().reset_index(name="value")
    elif metric == "Total Discounts":
        df = df_sales.groupby(["month_label", code_col])["proddiscount"].sum().reset_index(name="value")
    else:
        st.error("Unsupported metric.")
        return

    # Join name if applicable
    if name_col:
        lookup = filtered_data[[code_col, name_col]].drop_duplicates()
        df = df.merge(lookup, on=code_col, how="left")
        df["entity_label"] = df[code_col].astype(str) + " - " + df[name_col].astype(str)
    else:
        df["entity_label"] = df[code_col].astype(str)

    if "month_label" not in df.columns:
        st.error("Missing month_label column in processed data.")
        return
    # Sorting for month label (to ensure order like Feb 2023, Jun 2024)
    # Split month_label into month and year integers
    df[["month", "year"]] = df["month_label"].str.split("-", expand=True)
    # Sort by year then month
    df = df.sort_values(["year", "month"])
    # Plot
    fig = px.bar(
        df,
        x="month_label",
        y="value",
        color="entity_label",
        barmode="group",
        title=f"{metric} Comparison across Months",
        labels={"month_label": "Month", "value": metric, "entity_label": "Entity"}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Data Table
    pivot = df.pivot(index="month_label", columns="entity_label", values="value").fillna(0)
    st.markdown("### 📋 Corresponding Data")
    st.dataframe(pivot.style.format("{:.2f}"), use_container_width=True)

@timed
def plot_month_vs_month_dow_comparison(filtered_data,filtered_data_r,code_col,name_col,selected_codes,metric,selected_months,aggregation_type):
    print(filtered_data.columns)
    print(filtered_data_r.columns)
    df_sales = filtered_data.copy()
    df_returns = filtered_data_r.copy()

    # Add day of week
    # df_sales["day_of_week"] = pd.to_datetime(df_sales["date"]).dt.day_name()
    # df_returns["day_of_week"] = pd.to_datetime(df_returns["date"]).dt.day_name()

    # Custom weekday order for Bangladesh
    dow_order = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    df_sales["DOW"] = pd.Categorical(df_sales["DOW"], categories=dow_order, ordered=True)
    df_returns["DOW"] = pd.Categorical(df_returns["DOW"], categories=dow_order, ordered=True)

    # Create a month_label column
    df_sales["month_label"] = df_sales["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_sales["year"].astype(str)
    df_returns["month_label"] = df_returns["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_returns["year"].astype(str)

    # Filter for selected months
    df_sales = df_sales[df_sales["month_label"].isin(selected_months)]
    df_returns = df_returns[df_returns["month_label"].isin(selected_months)]

    # Filter for selected entities
    if selected_codes:
        df_sales = df_sales[df_sales[code_col].isin(selected_codes)]
        df_returns = df_returns[df_returns[code_col].isin(selected_codes)]

    # Metric logic
    if metric == "Net Sales":
        sales_grouped = df_sales.groupby(["month_label", "DOW", code_col])["final_sales"].sum().reset_index()
        returns_grouped = df_returns.groupby(["month_label", "DOW", code_col])["treturnamt"].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=["month_label", "DOW", code_col], how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["final_sales"] - df["treturnamt"]
    elif metric == "Net Margin":
        sales_grouped = df_sales.groupby(["month_label", "DOW", code_col])["gross_margin"].sum().reset_index()
        returns_grouped = df_returns.groupby(["month_label", "DOW", code_col])["treturnamt"].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=["month_label", "DOW", code_col], how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["gross_margin"] - df["treturnamt"]
    elif metric == "Total Returns":
        df = df_returns.groupby(["month_label", "DOW", code_col])["treturnamt"].sum().reset_index(name="value")
    elif metric == "Total Discounts":
        df = df_sales.groupby(["month_label", "DOW", code_col])["proddiscount"].sum().reset_index(name="value")
    else:
        st.error("Unsupported metric.")
        return

    # If aggregation is average, normalize by number of occurrences
    if aggregation_type == "Average":
        count_df = df.groupby(["month_label", "DOW", code_col])["value"].count().reset_index(name="count")
        df = df.merge(count_df, on=["month_label", "DOW", code_col])
        df["value"] = df["value"] / df["count"]

    # Join name if applicable
    if name_col:
        lookup = filtered_data[[code_col, name_col]].drop_duplicates()
        df = df.merge(lookup, on=code_col, how="left")
        df["entity_label"] = df[code_col].astype(str) + " - " + df[name_col].astype(str)
    else:
        df["entity_label"] = df[code_col].astype(str)

    df[["month", "year"]] = df["month_label"].str.split("-", expand=True)
    # Sort by year then month
    df = df.sort_values(["year", "month"])
    # Sort DOW by custom order
    # df["DOW"] = pd.Categorical(df["DOW"], categories=dow_order, ordered=True)

    # Plot
    fig = px.bar(
        df,
        x="DOW",
        y="value",
        color="entity_label",
        barmode="group",
        facet_col="month_label",
        title=f"{metric} by Day of Week (Month vs Month)",
        labels={"value": metric, "DOW": "Day", "entity_label": "Entity"}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Data Table
    pivot = df.pivot_table(index=["DOW"], columns=["entity_label", "month_label"], values="value", aggfunc="sum").fillna(0)
    st.markdown("### 📋 Corresponding Data")
    st.dataframe(pivot.style.format("{:.2f}"), use_container_width=True)

@timed
def plot_month_vs_month_dom_comparison(filtered_data,filtered_data_r,code_col,name_col,selected_codes,metric,selected_months,aggregation_type,selected_days=None):

    df_sales = filtered_data.copy()
    df_returns = filtered_data_r.copy()

    # Create a month_label column
    df_sales["month_label"] = df_sales["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_sales["year"].astype(str)
    df_returns["month_label"] = df_returns["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_returns["year"].astype(str)

    # Extract day
    # df_sales["day"] = pd.to_datetime(df_sales["date"]).dt.day
    # df_returns["day"] = pd.to_datetime(df_returns["date"]).dt.day

    # Filter for selected months
    df_sales = df_sales[df_sales["month_label"].isin(selected_months)]
    df_returns = df_returns[df_returns["month_label"].isin(selected_months)]

    # Filter for selected days if provided
    if selected_days:
        df_sales = df_sales[df_sales["DOM"].isin(selected_days)]
        df_returns = df_returns[df_returns["DOM"].isin(selected_days)]

    # Filter for selected entities
    if selected_codes:
        df_sales = df_sales[df_sales[code_col].isin(selected_codes)]
        df_returns = df_returns[df_returns[code_col].isin(selected_codes)]

    group_cols = ["month_label", "DOM", code_col]

    # Metric logic
    if metric == "Net Sales":
        sales_grouped = df_sales.groupby(group_cols)["final_sales"].sum().reset_index()
        returns_grouped = df_returns.groupby(group_cols)["treturnamt"].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=group_cols, how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["final_sales"] - df["treturnamt"]
    elif metric == "Net Margin":
        sales_grouped = df_sales.groupby(group_cols)["gross_margin"].sum().reset_index()
        returns_grouped = df_returns.groupby(group_cols)["treturnamt"].sum().reset_index()
        df = pd.merge(sales_grouped, returns_grouped, on=group_cols, how="left")
        df["treturnamt"].fillna(0, inplace=True)
        df["value"] = df["gross_margin"] - df["treturnamt"]
    elif metric == "Total Returns":
        df = df_returns.groupby(group_cols)["treturnamt"].sum().reset_index(name="value")
    elif metric == "Total Discounts":
        df = df_sales.groupby(group_cols)["proddiscount"].sum().reset_index(name="value")
    else:
        st.error("Unsupported metric.")
        return

    # Join name if applicable
    if name_col:
        lookup = filtered_data[[code_col, name_col]].drop_duplicates()
        df = df.merge(lookup, on=code_col, how="left")
        df["entity_label"] = df[code_col].astype(str) + " - " + df[name_col].astype(str)
    else:
        df["entity_label"] = df[code_col].astype(str)

    # Aggregation: Total or Average across days
    agg_df = df.groupby(["month_label", "entity_label"])["value"]
    if aggregation_type == "Average":
        agg_df = agg_df.mean().reset_index()
    else:
        agg_df = agg_df.sum().reset_index()

    # Sorting for month_label
    agg_df[["month", "year"]] = agg_df["month_label"].str.split("-", expand=True)
    # Sort by year then month
    agg_df = agg_df.sort_values(["year", "month"])

    # Plot
    fig = px.bar(
        agg_df,
        x="month_label",
        y="value",
        color="entity_label",
        barmode="group",
        title=f"{metric} Comparison across Months (Day of Month Aggregation: {aggregation_type})",
        labels={"month_label": "Month", "value": metric, "entity_label": "Entity"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    pivot = agg_df.pivot(index="month_label", columns="entity_label", values="value").fillna(0)
    st.markdown("### 📋 Corresponding Data")
    st.dataframe(pivot.style.format("{:.2f}"), use_container_width=True)

@timed
def plot_metric_comparison_monthly(filtered_data, filtered_data_r, code_col, selected_codes, metric_x, metric_y, selected_years, selected_month_names):
    df = filtered_data.copy()
    df_r = filtered_data_r.copy()

    if selected_codes:
        df = df[df[code_col].isin(selected_codes)]
        df_r = df_r[df_r[code_col].isin(selected_codes)]

    month_name_to_num = {calendar.month_abbr[i]: i for i in range(1, 13)}
    selected_months = [month_name_to_num[m] for m in selected_month_names]

    df = df[df["year"].isin(selected_years) & df["month"].isin(selected_months)]
    df_r = df_r[df_r["year"].isin(selected_years) & df_r["month"].isin(selected_months)]

    def compute_metric(df, df_r, metric):
        if metric == "Net Sales":
            s = df.groupby(["year", "month"])["final_sales"].sum()
            r = df_r.groupby(["year", "month"])["treturnamt"].sum()
            return s.subtract(r, fill_value=0).reset_index(name="value")
        elif metric == "Total Returns":
            return df_r.groupby(["year", "month"])["treturnamt"].sum().reset_index(name="value")
        elif metric == "Total Discounts":
            return df.groupby(["year", "month"])["proddiscount"].sum().reset_index(name="value")
        elif metric == "Net Margin":
            s = df.groupby(["year", "month"])["gross_margin"].sum()
            r = df_r.groupby(["year", "month"])["treturnamt"].sum()
            return s.subtract(r, fill_value=0).reset_index(name="value")
        else:
            return pd.DataFrame(columns=["year", "month", "value"])

    df_x = compute_metric(df, df_r, metric_x)
    df_x["Metric"] = metric_x
    df_y = compute_metric(df, df_r, metric_y)
    df_y["Metric"] = metric_y

    combined = pd.concat([df_x, df_y], ignore_index=True)
    combined["month"] = combined["month"].astype(int)
    combined["Month-Year"] = combined["year"].astype(str) + "-" + combined["month"].astype(str).str.zfill(2)
    combined = combined.sort_values(["year", "month"])

    # Pivot Table for % of metric_y over metric_x
    pivot_x = df_x.drop(columns="Metric").set_index(["year", "month"]).rename(columns={"value": "Metric_X"})
    pivot_y = df_y.drop(columns="Metric").set_index(["year", "month"]).rename(columns={"value": "Metric_Y"})
    merged = pivot_x.join(pivot_y, how="outer").fillna(0).reset_index()
    merged["Month-Year"] = merged["year"].astype(str) + "-" + merged["month"].astype(int).astype(str).str.zfill(2)
    merged["% B / A"] = (merged["Metric_Y"] / merged["Metric_X"].replace(0, pd.NA)) * 100
    merged = merged[["Month-Year", "Metric_X", "Metric_Y", "% B / A"]]

    fig = px.bar(
        combined,
        x="Month-Year",
        y="value",
        color="Metric",
        barmode="group",
        title=f"{metric_x} vs {metric_y} Over Time",
        labels={"value": "Value"}
    )
    fig.update_layout(xaxis_tickangle=-45)

        # Add % B / A annotations
    for row in merged.itertuples():
        month_year = row._1  # merged["Month-Year"]
        metric_x_val = row.Metric_X
        metric_y_val = row.Metric_Y
        pct = row._4  # merged["% B / A"]

        if pd.notna(pct):
            # Annotate at height of Metric_Y bar
            fig.add_annotation(
                x=month_year,
                y=metric_x_val,
                text=f"{pct:.1f}%",
                showarrow=False,
                font=dict(color="black", size=10),
                yshift=10
            )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Comparison Table")
    st.dataframe(merged.style.format({"Metric_X": "{:,.0f}", "Metric_Y": "{:,.0f}", "% B / A": "{:.1f}%"}), use_container_width=True)

@timed
def plot_distribution_analysis(filtered_data, filtered_data_r, metric, group_by, value_min=None, value_max=None, nbins=100):
    valid_columns = {
        "Customer": "cusname",
        "Product": "itemname",
        "Salesman": "spname",
        "Area": "area",
        "Product Group": "itemgroup",
        "Day of Month": "DOM",
        "Day of Week": "DOW"
    }

    if group_by not in valid_columns:
        st.error(f"Invalid group_by: {group_by}")
        return

    entity_col = valid_columns[group_by]
    df = filtered_data.copy()
    df_r = filtered_data_r.copy()

    try:
        # Step 1: Aggregate per entity (e.g., per customer)
        if metric == "Net Sales":
            sales_sum = df.groupby(entity_col)["final_sales"].sum()
            return_sum = df_r.groupby(entity_col)["treturnamt"].sum()
            agg_series = sales_sum.subtract(return_sum, fill_value=0)
        if metric == "Net Margin":
            sales_sum = df.groupby(entity_col)["gross_margin"].sum()
            return_sum = df_r.groupby(entity_col)["treturnamt"].sum()
            agg_series = sales_sum.subtract(return_sum, fill_value=0)
        elif metric == "Total Returns":
            agg_series = df_r.groupby(entity_col)["treturnamt"].sum()
        elif metric == "Total Discounts":
            agg_series = df.groupby(entity_col)["proddiscount"].sum()
        else:
            st.error("Unsupported metric selected.")
            return

        agg_df = agg_series.reset_index(name="value")

        # Step 2: Apply value filters if needed
        if value_min is not None:
            agg_df = agg_df[agg_df["value"] >= value_min]
        if value_max is not None:
            agg_df = agg_df[agg_df["value"] <= value_max]

        # Step 3: Plot distribution
        fig = px.histogram(
            agg_df,
            x="value",
            nbins=nbins,
            title=f"Distribution of {metric} by {group_by}",
            labels={"value": metric}
        )

        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.markdown("### 📋 Binned Frequency Table")
        bin_table = pd.cut(agg_df["value"], bins=nbins).value_counts().sort_index().reset_index()
        bin_table.columns = ["Range", "Count"]
        st.dataframe(bin_table, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating distribution: {e}")
        st.warning("Check if the selected combination of metric and group by column is valid.")

@timed
def generate_descriptive_statistics(filtered_data, filtered_data_r, group_by):
    valid_columns = {
        "Customer": "cusname",
        "Product": "itemname",
        "Salesman": "spname",
        "Area": "area",
        "Product Group": "itemgroup",
        "Month": "month",
        "Year": "year",
        "Day of Month": "DOM",
        "Day of Week": "DOW"
    }

    if group_by not in valid_columns:
        return pd.DataFrame({"Error": ["Invalid group_by selection."]})

    entity_col = valid_columns[group_by]
    df = filtered_data.copy()
    df_r = filtered_data_r.copy()
    stats = []

    def add_stat_row(name, series):
        desc = series.describe()
        desc["total"] = series.sum()
        desc["coef_var"] = desc["std"] / desc["mean"] if desc["mean"] != 0 else None
        stats.append(desc.rename(name))

    try:
        # Net Sales
        sales_sum = df.groupby(entity_col)["final_sales"].sum()
        return_sum = df_r.groupby(entity_col)["treturnamt"].sum()
        net_sales_series = sales_sum.subtract(return_sum, fill_value=0)
        add_stat_row("Net Sales", net_sales_series)

        # Net Margin
        margin_sum = df.groupby(entity_col)["gross_margin"].sum()
        return_sum = df_r.groupby(entity_col)["treturnamt"].sum()
        net_margin_series = margin_sum.subtract(return_sum, fill_value=0)
        add_stat_row("Net Margin", net_margin_series)

        # Total Returns
        total_returns_series = df_r.groupby(entity_col)["treturnamt"].sum()
        add_stat_row("Total Returns", total_returns_series)

        # TOtal Discounts
        discount_series = df.groupby(entity_col)["proddiscount"].sum()
        add_stat_row("Total Discounts", discount_series)
    except Exception as e:
        return pd.DataFrame({"Error": [f"Failed to compute stats: {str(e)}"]})

    df_stats = pd.DataFrame(stats)
    return df_stats.round(2).reset_index().rename(columns={"index": "Metric"})