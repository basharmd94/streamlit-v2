import streamlit as st
import pandas as pd
from modules.data_process_files import common
pd.set_option('display.float_format', '{:.2f}'.format)
import calendar
import plotly.express as px
from utils.utils import timed


@timed
def calculate_summary_statistics(filtered_data, filtered_data_r):
    """
    Calculate summary of sales data for the filtered data.

    Args:
    - filtered_data: Filtered sales data.
    - filtered_data_r: Filtered returns data.

    Returns:
    - Dictionary containing the summary statistics.
    """
    units_sold     = float(filtered_data["quantity"].sum()) if len(filtered_data) else 0.0
    units_returned = float(filtered_data_r["returnqty"].sum()) if len(filtered_data_r) else 0.0
    net_units      = units_sold - units_returned

    return {
        "Total Sales": filtered_data['final_sales'].sum().round(2),
        "Total Returns": filtered_data_r['treturnamt'].sum().round(2),
        "Net Sales": filtered_data['final_sales'].sum() - filtered_data_r['treturnamt'].sum().round(2),
        "Number of Orders": filtered_data['voucher'].nunique(),
        "Number of Returns": filtered_data_r['revoucher'].nunique(),
        "Number of Customers": filtered_data['cusid'].nunique(),
        "Number of Customer Returned": filtered_data_r['cusid'].nunique(),
        "Number of Products": filtered_data['itemcode'].nunique(),
        "Number of Products Returned": filtered_data_r['itemcode'].nunique(),
        "Units Sold": round(units_sold, 2),
        "Units Returned": round(units_returned, 2),
        "Net Units Sold": round(net_units, 2)
    }

@timed
def display_summary_statistics(stats):
    """
    Display summary statistics in the Streamlit app.

    Args:
    - stats: Dictionary containing the summary statistics.
    """
    st.sidebar.title("Overall Sales Analysis")
    
    # Create a grid-like layout with three columns
    col1, col2, col3 = st.columns(3)
    
    # Split stats into three parts
    stats_items = list(stats.items())
    first_third = stats_items[:len(stats_items)//3]
    second_third = stats_items[len(stats_items)//3:2*len(stats_items)//3]
    third_third = stats_items[2*len(stats_items)//3:]
    
    # Display first third of stats in first column
    with col1:
        for stat_name, value in first_third:
            st.markdown(f"**{stat_name}:** {value:,.2f}")
    
    # Display second third of stats in second column
    with col2:
        for stat_name, value in second_third:
            st.markdown(f"**{stat_name}:** {value:,.2f}")
    
    # Display third third of stats in third column
    with col3:
        for stat_name, value in third_third:
            st.markdown(f"**{stat_name}:** {value:,.2f}")

@timed
def display_cross_relation_pivot(filtered_data, filtered_data_r, current_page):
    st.subheader("🔁 Cross Relation Analysis")

    column_options = {
        'Salesman': ['spid', 'spname'],
        'Customer': ['cusid', 'cusname'],
        'Product': ['itemcode', 'itemname'],
        'Product Group': ['itemgroup'],
        'Area': ['area'],
        'Reason':['reason']
    }

    metric_options = [
        "Net Sales", "Total Returns", "Total Discounts",
        "Number of Orders", "Number of Returns", "Number of Discounts",
        "Number of Customers", "Number of Customer Returns",
        "Number of Products", "Number of Product Returns",
        "Units Sold", "Units Returned", "Net Units Sold"
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
        "Area": ["area"],
        "Reason":["reason"]
    }

    metric_options = [
        "Net Sales", "Total Returns", "Total Discounts",
        "Number of Orders", "Number of Returns", "Number of Discounts",
        "Number of Customers", "Number of Customer Returns",
        "Number of Products", "Number of Product Returns",
        "Units Sold", "Units Returned", "Net Units Sold"
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
def prepare_number_of_orders(filtered_data):
    """
    Prepare data for plotting the number of orders.
    
    Parameters:
    - filtered_data: Filtered sales DataFrame
    
    Returns:
    - DataFrame with the number of orders
    """
    # Calculate number of orders
    orders_data = filtered_data.groupby(['year', 'month']).size().reset_index(name='number_of_orders')
    return orders_data

@timed
def prepare_number_of_returns(filtered_data_r):
    """
    Prepare data for plotting the number of returns.
    
    Parameters:
    - filtered_data_r: Filtered returns DataFrame
    
    Returns:
    - DataFrame with the number of returns
    """
    # Calculate number of returns
    returns_data = filtered_data_r.groupby(['year', 'month']).size().reset_index(name='number_of_returns')
    return returns_data

@timed
def prepare_number_of_customers(filtered_data):
    """
    Prepare data for plotting the number of customers.
    
    Parameters:
    - filtered_data: Filtered sales DataFrame
    
    Returns:
    - DataFrame with the number of customers
    """
    # Calculate number of unique customers
    customers_data = filtered_data.groupby(['year', 'month'])['cusid'].nunique().reset_index(name='number_of_customers')
    return customers_data

@timed
def prepare_number_of_customer_returns(filtered_data_r):
    """
    Prepare data for plotting the number of customer returns.
    
    Parameters:
    - filtered_data_r: Filtered returns DataFrame
    
    Returns:
    - DataFrame with the number of customer returns
    """
    # Calculate number of unique customer returns
    customer_returns_data = filtered_data_r.groupby(['year', 'month'])['cusid'].nunique().reset_index(name='number_of_customer_returns')
    return customer_returns_data

@timed
def prepare_number_of_products(filtered_data):
    """
    Prepare data for plotting the number of products.
    
    Parameters:
    - filtered_data: Filtered sales DataFrame
    
    Returns:
    - DataFrame with the number of products
    """
    # Calculate number of unique products
    products_data = filtered_data.groupby(['year', 'month'])['itemcode'].nunique().reset_index(name='number_of_products')
    return products_data

@timed
def prepare_number_of_product_returns(filtered_data_r):
    """
    Prepare data for plotting the number of product returns.
    
    Parameters:
    - filtered_data_r: Filtered returns DataFrame
    
    Returns:
    - DataFrame with the number of product returns
    """
    # Calculate number of unique product returns
    product_returns_data = filtered_data_r.groupby(['year', 'month'])['itemcode'].nunique().reset_index(name='number_of_product_returns')
    return product_returns_data

@timed
def prepare_net_sales(filtered_data, filtered_data_r):
    """
    Prepare data for plotting the net sales, adjusted for returns.
    
    Parameters:
    - filtered_data: Sales DataFrame
    - filtered_data_r: Returns DataFrame
    
    Returns:
    - DataFrame with the net sales
    """
    # Calculate total sales per month
    sales_data = filtered_data.groupby(['year', 'month'])['final_sales'].sum().reset_index(name='final_sales')
    
    # Calculate total returns per month
    returns_data = filtered_data_r.groupby(['year', 'month'])['treturnamt'].sum().reset_index(name='total_returns')
    
    # Merge sales and returns data
    net_sales_data = sales_data.merge(returns_data, on=['year', 'month'], how='left')
    
    # Fill NaN returns with 0 and calculate net sales
    net_sales_data['total_returns'] = net_sales_data['total_returns'].fillna(0)
    net_sales_data['net_sales'] = net_sales_data['final_sales'] - net_sales_data['total_returns']
    
    # Select and rename columns
    net_sales_data = net_sales_data[['year', 'month', 'net_sales']]
    
    return net_sales_data

@timed
def prepare_sales_performance_ratios(filtered_data, filtered_data_r):
    """
    Calculate various sales performance ratios over time.
    
    Parameters:
    - filtered_data: Sales DataFrame
    - filtered_data_r: Returns DataFrame
    
    Returns:
    - DataFrame with performance ratios pivoted
    """
    # Prepare data for different metrics
    orders_data = prepare_number_of_orders(filtered_data)
    returns_data = prepare_number_of_returns(filtered_data_r)
    customers_data = prepare_number_of_customers(filtered_data)
    customer_returns_data = prepare_number_of_customer_returns(filtered_data_r)
    net_sales_data = prepare_net_sales(filtered_data, filtered_data_r)
    
    # Merge all dataframes on year and month
    merged_data = orders_data.merge(
        returns_data[['year', 'month', 'number_of_returns']], 
        on=['year', 'month'], 
        how='outer'
    ).merge(
        customers_data[['year', 'month', 'number_of_customers']], 
        on=['year', 'month'], 
        how='outer'
    ).merge(
        customer_returns_data[['year', 'month', 'number_of_customer_returns']], 
        on=['year', 'month'], 
        how='outer'
    ).merge(
        net_sales_data[['year', 'month', 'net_sales']], 
        on=['year', 'month'], 
        how='outer'
    )
    
    # Fill NaN values with 0 to prevent division errors
    merged_data = merged_data.fillna(0)
    
    # Calculate ratios
    merged_data['net_sales_per_order'] = merged_data['net_sales'] / merged_data['number_of_orders'].replace(0, 1)
    merged_data['net_sales_per_customer'] = merged_data['net_sales'] / merged_data['number_of_customers'].replace(0, 1)
    merged_data['orders_per_customer'] = merged_data['number_of_orders'] / merged_data['number_of_customers'].replace(0, 1)
    merged_data['returns_per_customer_return'] = merged_data['number_of_customer_returns'] / merged_data['number_of_returns'].replace(0, 1)
    merged_data['orders_to_returns_ratio'] = merged_data['number_of_returns']/merged_data['number_of_orders'].replace(0, 1)
    merged_data['customers_to_customer_returns_ratio'] = merged_data['number_of_customer_returns'] / merged_data['number_of_customers'].replace(0, 1)
    
    # Sort by year and month
    merged_data['month_numeric'] = merged_data['month'].apply(convert_month_to_number)
    merged_data['sort_key'] = merged_data.apply(lambda row: (int(row['year']), row['month_numeric']), axis=1)
    merged_data = merged_data.sort_values('sort_key')
    
    # Format month for display
    merged_data['month_formatted'] = merged_data['month_numeric'].astype(str).str.zfill(2)
    merged_data['period'] = merged_data.apply(lambda row: f"{row['year']}-{row['month_formatted']}", axis=1)
    
    # Select ratio columns for melting
    ratio_columns = [
        'net_sales_per_order', 
        'net_sales_per_customer', 
        'orders_per_customer', 
        'returns_per_customer_return', 
        'orders_to_returns_ratio', 
        'customers_to_customer_returns_ratio'
    ]
    
    # Rename columns for better readability
    ratio_names = {
        'net_sales_per_order': 'Net Sales per Order', 
        'net_sales_per_customer': 'Net Sales per Customer', 
        'orders_per_customer': 'Orders per Customer', 
        'returns_per_customer_return': 'Returns per Customer Return', 
        'orders_to_returns_ratio': 'Orders to Returns Ratio', 
        'customers_to_customer_returns_ratio': 'Customers to Customer Returns Ratio'
    }
    
    # Melt the data to create a long-format DataFrame
    melted_data = merged_data.melt(
        id_vars=['period'], 
        value_vars=ratio_columns, 
        var_name='Ratio', 
        value_name='Value'
    )
    
    # Replace ratio column names with readable names
    melted_data['Ratio'] = melted_data['Ratio'].map(ratio_names)
    
    # Pivot the melted data
    pivoted_data = melted_data.pivot_table(
        index='Ratio', 
        columns='period', 
        values='Value', 
        aggfunc='first'
    )
    
    # Round values
    pivoted_data = pivoted_data.round(2)
    
    return pivoted_data

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
    elif metric == "Total Sales":
        df = df_sales.groupby(["year", "month"])["final_sales"].sum().reset_index(name="value")
    elif metric == "Number of Orders":
        df = df_sales.drop_duplicates("voucher").groupby(["year", "month"])["voucher"].count().reset_index(name="value")
    elif metric == "Total Returns":
        df = df_returns.groupby(["year", "month"])["treturnamt"].sum().reset_index(name="value")
    elif metric == "Number of Returns":
        df = df_returns.drop_duplicates("revoucher").groupby(["year", "month"])["revoucher"].count().reset_index(name="value")
    elif metric == "Number of Products":
        df = df_sales.drop_duplicates(["itemcode", "year", "month"]).groupby(["year", "month"])["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Product Returns":
        df = df_returns.drop_duplicates(["itemcode", "year", "month"]).groupby(["year", "month"])["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Customers":
        df = df_sales.drop_duplicates(["cusid", "year", "month"]).groupby(["year", "month"])["cusid"].count().reset_index(name="value")
    elif metric == "Number of Customer Returns":
        df = df_returns.drop_duplicates(["cusid", "year", "month"]).groupby(["year", "month"])["cusid"].count().reset_index(name="value")
    elif metric == "Total Product Discounts":
        df = df_sales.groupby(["year", "month"])["proddiscount"].sum().reset_index(name="value")
    elif metric == "Number of Product Discounts":
        df = df_sales[df_sales["proddiscount"] > 0].groupby(["year", "month"])["proddiscount"].count().reset_index(name="value")
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
    import pandas as pd
    import plotly.express as px
    import streamlit as st

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
    elif metric == "Total Sales":
        df = df_sales.groupby(["year", "day_month"])["final_sales"].sum().reset_index(name="value")
    elif metric == "Number of Orders":
        df = df_sales.drop_duplicates("voucher").groupby(["year", "day_month"])["voucher"].count().reset_index(name="value")
    elif metric == "Total Returns":
        df = df_returns.groupby(["year", "day_month"])["treturnamt"].sum().reset_index(name="value")
    elif metric == "Number of Returns":
        df = df_returns.drop_duplicates("revoucher").groupby(["year", "day_month"])["revoucher"].count().reset_index(name="value")
    elif metric == "Number of Products":
        df = df_sales.drop_duplicates(["itemcode", "year", "day_month"]).groupby(["year", "day_month"])["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Product Returns":
        df = df_returns.drop_duplicates(["itemcode", "year", "day_month"]).groupby(["year", "day_month"])["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Customers":
        df = df_sales.drop_duplicates(["cusid", "year", "day_month"]).groupby(["year", "day_month"])["cusid"].count().reset_index(name="value")
    elif metric == "Number of Customer Returns":
        df = df_returns.drop_duplicates(["cusid", "year", "day_month"]).groupby(["year", "day_month"])["cusid"].count().reset_index(name="value")
    elif metric == "Total Product Discounts":
        df = df_sales.groupby(["year", "day_month"])["proddiscount"].sum().reset_index(name="value")
    elif metric == "Number of Product Discounts":
        df = df_sales[df_sales["proddiscount"] > 0].groupby(["year", "day_month"])["proddiscount"].count().reset_index(name="value")
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
    elif metric == "Total Sales":
        df = df_sales.groupby(["year", "weekday"])["final_sales"].sum().reset_index(name="value")
    elif metric == "Number of Orders":
        df = df_sales.drop_duplicates("voucher").groupby(["year", "weekday"])["voucher"].count().reset_index(name="value")
    elif metric == "Total Returns":
        df = df_returns.groupby(["year", "weekday"])["treturnamt"].sum().reset_index(name="value")
    elif metric == "Number of Returns":
        df = df_returns.drop_duplicates("revoucher").groupby(["year", "weekday"])["revoucher"].count().reset_index(name="value")
    elif metric == "Number of Products":
        df = df_sales.drop_duplicates(["itemcode", "year", "weekday"]).groupby(["year", "weekday"])["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Product Returns":
        df = df_returns.drop_duplicates(["itemcode", "year", "weekday"]).groupby(["year", "weekday"])["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Customers":
        df = df_sales.drop_duplicates(["cusid", "year", "weekday"]).groupby(["year", "weekday"])["cusid"].count().reset_index(name="value")
    elif metric == "Number of Customer Returns":
        df = df_returns.drop_duplicates(["cusid", "year", "weekday"]).groupby(["year", "weekday"])["cusid"].count().reset_index(name="value")
    elif metric == "Total Product Discounts":
        df = df_sales.groupby(["year", "weekday"])["proddiscount"].sum().reset_index(name="value")
    elif metric == "Number of Product Discounts":
        df = df_sales[df_sales["proddiscount"] > 0].groupby(["year", "weekday"])["proddiscount"].count().reset_index(name="value")
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
    elif metric == "Total Sales":
        df = df_sales.groupby(["year", "day"])["final_sales"].sum().reset_index(name="value")
    elif metric == "Number of Orders":
        df = df_sales.drop_duplicates("voucher").groupby(["year", "day"])["voucher"].count().reset_index(name="value")
    elif metric == "Total Returns":
        df = df_returns.groupby(["year", "day"])["treturnamt"].sum().reset_index(name="value")
    elif metric == "Number of Returns":
        df = df_returns.drop_duplicates("revoucher").groupby(["year", "day"])["revoucher"].count().reset_index(name="value")
    elif metric == "Number of Products":
        df = df_sales.drop_duplicates(["itemcode", "year", "day"]).groupby(["year", "day"])["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Product Returns":
        df = df_returns.drop_duplicates(["itemcode", "year", "day"]).groupby(["year", "day"])["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Customers":
        df = df_sales.drop_duplicates(["cusid", "year", "day"]).groupby(["year", "day"])["cusid"].count().reset_index(name="value")
    elif metric == "Number of Customer Returns":
        df = df_returns.drop_duplicates(["cusid", "year", "day"]).groupby(["year", "day"])["cusid"].count().reset_index(name="value")
    elif metric == "Total Product Discounts":
        df = df_sales.groupby(["year", "day"])["proddiscount"].sum().reset_index(name="value")
    elif metric == "Number of Product Discounts":
        df = df_sales[df_sales["proddiscount"] > 0].groupby(["year", "day"])["proddiscount"].count().reset_index(name="value")
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
    elif metric == "Total Sales":
        df = df_sales.groupby(["month_label", code_col])["final_sales"].sum().reset_index(name="value")
    elif metric == "Number of Orders":
        df = df_sales.drop_duplicates("voucher").groupby(["month_label", code_col])["voucher"].count().reset_index(name="value")
    elif metric == "Total Returns":
        df = df_returns.groupby(["month_label", code_col])["treturnamt"].sum().reset_index(name="value")
    elif metric == "Number of Returns":
        df = df_returns.drop_duplicates("revoucher").groupby(["month_label", code_col])["revoucher"].count().reset_index(name="value")
    elif metric == "Number of Products":
        df = df_sales.drop_duplicates(["itemcode", "month_label"]).groupby(["month_label", code_col])["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Product Returns":
        df = df_returns.drop_duplicates(["itemcode", "month_label"]).groupby(["month_label", code_col])["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Customers":
        df = df_sales.drop_duplicates(["cusid", "month_label"]).groupby(["month_label", code_col])["cusid"].count().reset_index(name="value")
    elif metric == "Number of Customer Returns":
        df = df_returns.drop_duplicates(["cusid", "month_label"]).groupby(["month_label", code_col])["cusid"].count().reset_index(name="value")
    elif metric == "Total Product Discounts":
        df = df_sales.groupby(["month_label", code_col])["proddiscount"].sum().reset_index(name="value")
    elif metric == "Number of Product Discounts":
        df = df_sales[df_sales["proddiscount"] > 0].groupby(["month_label", code_col])["proddiscount"].count().reset_index(name="value")
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
    elif metric == "Total Sales":
        df = df_sales.groupby(["month_label", "DOW", code_col])["final_sales"].sum().reset_index(name="value")
    elif metric == "Number of Orders":
        df = df_sales.drop_duplicates("voucher").groupby(["month_label", "DOW", code_col])["voucher"].count().reset_index(name="value")
    elif metric == "Total Returns":
        df = df_returns.groupby(["month_label", "DOW", code_col])["treturnamt"].sum().reset_index(name="value")
    elif metric == "Number of Returns":
        df = df_returns.drop_duplicates("revoucher").groupby(["month_label", "DOW", code_col])["revoucher"].count().reset_index(name="value")
    elif metric == "Number of Products":
        df = df_sales.drop_duplicates(["itemcode", "month_label", "DOW"]).groupby(["month_label", "DOW", code_col])["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Product Returns":
        df = df_returns.drop_duplicates(["itemcode", "month_label", "DOW"]).groupby(["month_label", "DOW", code_col])["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Customers":
        df = df_sales.drop_duplicates(["cusid", "month_label", "DOW"]).groupby(["month_label", "DOW", code_col])["cusid"].count().reset_index(name="value")
    elif metric == "Number of Customer Returns":
        df = df_returns.drop_duplicates(["cusid", "month_label", "DOW"]).groupby(["month_label", "DOW", code_col])["cusid"].count().reset_index(name="value")
    elif metric == "Total Product Discounts":
        df = df_sales.groupby(["month_label", "DOW", code_col])["proddiscount"].sum().reset_index(name="value")
    elif metric == "Number of Product Discounts":
        df = df_sales[df_sales["proddiscount"] > 0].groupby(["month_label", "DOW", code_col])["proddiscount"].count().reset_index(name="value")
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
    elif metric == "Total Sales":
        df = df_sales.groupby(group_cols)["final_sales"].sum().reset_index(name="value")
    elif metric == "Number of Orders":
        df = df_sales.drop_duplicates("voucher").groupby(group_cols)["voucher"].count().reset_index(name="value")
    elif metric == "Total Returns":
        df = df_returns.groupby(group_cols)["treturnamt"].sum().reset_index(name="value")
    elif metric == "Number of Returns":
        df = df_returns.drop_duplicates("revoucher").groupby(group_cols)["revoucher"].count().reset_index(name="value")
    elif metric == "Number of Products":
        df = df_sales.drop_duplicates(["itemcode", "month_label", "DOM"]).groupby(group_cols)["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Product Returns":
        df = df_returns.drop_duplicates(["itemcode", "month_label", "DOM"]).groupby(group_cols)["itemcode"].count().reset_index(name="value")
    elif metric == "Number of Customers":
        df = df_sales.drop_duplicates(["cusid", "month_label", "DOM"]).groupby(group_cols)["cusid"].count().reset_index(name="value")
    elif metric == "Number of Customer Returns":
        df = df_returns.drop_duplicates(["cusid", "month_label", "DOM"]).groupby(group_cols)["cusid"].count().reset_index(name="value")
    elif metric == "Total Product Discounts":
        df = df_sales.groupby(group_cols)["proddiscount"].sum().reset_index(name="value")
    elif metric == "Number of Product Discounts":
        df = df_sales[df_sales["proddiscount"] > 0].groupby(group_cols)["proddiscount"].count().reset_index(name="value")
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
        elif metric == "Total Sales":
            agg_series = df.groupby(entity_col)["final_sales"].sum()
        elif metric == "Total Returns":
            agg_series = df_r.groupby(entity_col)["treturnamt"].sum()
        elif metric == "Product Discounts":
            agg_series = df.groupby(entity_col)["proddiscount"].sum()
        elif metric == "Number of Orders":
            agg_series = df.drop_duplicates("voucher").groupby(entity_col)["voucher"].count()
        elif metric == "Number of Returns":
            agg_series = df_r.drop_duplicates("revoucher").groupby(entity_col)["revoucher"].count()
        elif metric == "Number of Customers":
            agg_series = df.groupby(entity_col)["cusid"].nunique()
        elif metric == "Number of Products":
            agg_series = df.groupby(entity_col)["itemcode"].nunique()
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

        # Total Sales
        total_sales_series = df.groupby(entity_col)["final_sales"].sum()
        add_stat_row("Total Sales", total_sales_series)

        # Total Returns
        total_returns_series = df_r.groupby(entity_col)["treturnamt"].sum()
        add_stat_row("Total Returns", total_returns_series)

        # Product Discounts
        discount_series = df.groupby(entity_col)["proddiscount"].sum()
        add_stat_row("Product Discounts", discount_series)

        # Number of Orders
        order_series = df.drop_duplicates("voucher").groupby(entity_col)["voucher"].count()
        add_stat_row("Number of Orders", order_series)

        # Number of Returns
        return_series = df_r.drop_duplicates("revoucher").groupby(entity_col)["revoucher"].count()
        add_stat_row("Number of Returns", return_series)

        # Number of Customers
        customer_series = df.groupby(entity_col)["cusid"].nunique()
        add_stat_row("Number of Customers", customer_series)

        # Number of Products
        product_series = df.groupby(entity_col)["itemcode"].nunique()
        add_stat_row("Number of Products", product_series)

    except Exception as e:
        return pd.DataFrame({"Error": [f"Failed to compute stats: {str(e)}"]})

    df_stats = pd.DataFrame(stats)
    return df_stats.round(2).reset_index().rename(columns={"index": "Metric"})

@timed
def convert_month_to_number(month):
    """
    Convert month name or string to its numeric representation.
    
    Parameters:
    - month: Month name or string representation
    
    Returns:
    - Numeric month (1-12)
    """
    month_mapping = {
        'January': 1, 'Jan': 1,
        'February': 2, 'Feb': 2,
        'March': 3, 'Mar': 3,
        'April': 4, 'Apr': 4,
        'May': 5,
        'June': 6, 'Jun': 6,
        'July': 7, 'Jul': 7,
        'August': 8, 'Aug': 8,
        'September': 9, 'Sep': 9,
        'October': 10, 'Oct': 10,
        'November': 11, 'Nov': 11,
        'December': 12, 'Dec': 12
    }
    
    # If it's already a number, convert to int
    if isinstance(month, (int, float)):
        return int(month)
    
    # If it's a string, look up in mapping
    if isinstance(month, str):
        # Try exact match first
        if month in month_mapping:
            return month_mapping[month]
        
        # Try case-insensitive match
        month_lower = month.lower()
        for key, value in month_mapping.items():
            if key.lower() == month_lower:
                return value
    
    # If no match found, raise an error
    raise ValueError(f"Could not convert month: {month}")

@timed
def plot_net(data1,data2,xaxis,yaxis1,yaxis2,bartitle,current_page):

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
def plot_number_of_discounts(filtered_data, current_page):
    df = filtered_data[filtered_data["proddiscount"] > 0].copy()
    df = df.groupby(['year', 'month']).size().reset_index(name="count")
    df["month"] = df["month"].astype(int).astype(str).str.zfill(2)
    df["x_label"] = df["year"].astype(str) + "-" + df["month"]

    fig = px.bar(
        df,
        x="x_label",
        y="count",
        title="Number of Discounts Over Time",
        labels={"x_label": "Year-Month", "count": "Number of Discounts"}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

@timed
def plot_number_of_orders(filtered_data, current_page):
    """
    Plot the number of orders over time as a bar graph.
    
    Parameters:
    - filtered_data: Filtered sales DataFrame
    - current_page: Current page context
    """
    from modules.data_process_files import overall_sales
    
    # Prepare data
    orders_data = overall_sales.prepare_number_of_orders(filtered_data)
    
    # Create a custom sorting key with robust month conversion
    orders_data['month_numeric'] = orders_data['month'].apply(convert_month_to_number)
    orders_data['sort_key'] = orders_data.apply(lambda row: (int(row['year']), row['month_numeric']), axis=1)
    orders_data = orders_data.sort_values('sort_key')
    
    # Ensure month is a string and padded
    orders_data['month_formatted'] = orders_data['month_numeric'].astype(str).str.zfill(2)
    orders_data['x_label'] = orders_data.apply(lambda row: f"{row['year']}-{row['month_formatted']}", axis=1)
    
    # Create Plotly bar graph
    fig = px.bar(
        orders_data, 
        x='x_label', 
        y='number_of_orders',
        title='Number of Orders Over Time',
        labels={'x_label': 'Year-Month', 'number_of_orders': 'Number of Orders'}
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Year-Month',
        yaxis_title='Number of Orders',
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

@timed
def plot_number_of_returns(filtered_data_r, current_page):
    """
    Plot the number of returns over time as a bar graph.
    
    Parameters:
    - filtered_data_r: Filtered returns DataFrame
    - current_page: Current page context
    """
    from modules.data_process_files import overall_sales
    
    # Prepare data
    returns_data = overall_sales.prepare_number_of_returns(filtered_data_r)
    
    # Create a custom sorting key with robust month conversion
    returns_data['month_numeric'] = returns_data['month'].apply(convert_month_to_number)
    returns_data['sort_key'] = returns_data.apply(lambda row: (int(row['year']), row['month_numeric']), axis=1)
    returns_data = returns_data.sort_values('sort_key')
    
    # Ensure month is a string and padded
    returns_data['month_formatted'] = returns_data['month_numeric'].astype(str).str.zfill(2)
    returns_data['x_label'] = returns_data.apply(lambda row: f"{row['year']}-{row['month_formatted']}", axis=1)
    
    # Create Plotly bar graph
    fig = px.bar(
        returns_data, 
        x='x_label', 
        y='number_of_returns',
        title='Number of Returns Over Time',
        labels={'x_label': 'Year-Month', 'number_of_returns': 'Number of Returns'}
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Year-Month',
        yaxis_title='Number of Returns',
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

@timed
def plot_number_of_customers(filtered_data, current_page):
    """
    Plot the number of unique customers over time as a bar graph.
    
    Parameters:
    - filtered_data: Filtered sales DataFrame
    - current_page: Current page context
    """
    from modules.data_process_files import overall_sales
    
    # Prepare data
    customers_data = overall_sales.prepare_number_of_customers(filtered_data)
    
    # Create a custom sorting key with robust month conversion
    customers_data['month_numeric'] = customers_data['month'].apply(convert_month_to_number)
    customers_data['sort_key'] = customers_data.apply(lambda row: (int(row['year']), row['month_numeric']), axis=1)
    customers_data = customers_data.sort_values('sort_key')
    
    # Ensure month is a string and padded
    customers_data['month_formatted'] = customers_data['month_numeric'].astype(str).str.zfill(2)
    customers_data['x_label'] = customers_data.apply(lambda row: f"{row['year']}-{row['month_formatted']}", axis=1)
    
    # Create Plotly bar graph
    fig = px.bar(
        customers_data, 
        x='x_label', 
        y='number_of_customers',
        title='Number of Unique Customers Over Time',
        labels={'x_label': 'Year-Month', 'number_of_customers': 'Number of Customers'}
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Year-Month',
        yaxis_title='Number of Customers',
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

@timed
def plot_number_of_customer_returns(filtered_data_r, current_page):
    """
    Plot the number of unique customer returns over time as a bar graph.
    
    Parameters:
    - filtered_data_r: Filtered returns DataFrame
    - current_page: Current page context
    """
    from modules.data_process_files import overall_sales
    
    # Prepare data
    customer_returns_data = overall_sales.prepare_number_of_customer_returns(filtered_data_r)
    
    # Create a custom sorting key with robust month conversion
    customer_returns_data['month_numeric'] = customer_returns_data['month'].apply(convert_month_to_number)
    customer_returns_data['sort_key'] = customer_returns_data.apply(lambda row: (int(row['year']), row['month_numeric']), axis=1)
    customer_returns_data = customer_returns_data.sort_values('sort_key')
    
    # Ensure month is a string and padded
    customer_returns_data['month_formatted'] = customer_returns_data['month_numeric'].astype(str).str.zfill(2)
    customer_returns_data['x_label'] = customer_returns_data.apply(lambda row: f"{row['year']}-{row['month_formatted']}", axis=1)
    
    # Create Plotly bar graph
    fig = px.bar(
        customer_returns_data, 
        x='x_label', 
        y='number_of_customer_returns',
        title='Number of Unique Customer Returns Over Time',
        labels={'x_label': 'Year-Month', 'number_of_customer_returns': 'Number of Customer Returns'}
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Year-Month',
        yaxis_title='Number of Customer Returns',
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

@timed
def plot_number_of_products(filtered_data, current_page):
    """
    Plot the number of unique products over time as a bar graph.
    
    Parameters:
    - filtered_data: Filtered sales DataFrame
    - current_page: Current page context
    """
    from modules.data_process_files import overall_sales
    
    # Prepare data
    products_data = overall_sales.prepare_number_of_products(filtered_data)
    
    # Create a custom sorting key with robust month conversion
    products_data['month_numeric'] = products_data['month'].apply(convert_month_to_number)
    products_data['sort_key'] = products_data.apply(lambda row: (int(row['year']), row['month_numeric']), axis=1)
    products_data = products_data.sort_values('sort_key')
    
    # Ensure month is a string and padded
    products_data['month_formatted'] = products_data['month_numeric'].astype(str).str.zfill(2)
    products_data['x_label'] = products_data.apply(lambda row: f"{row['year']}-{row['month_formatted']}", axis=1)
    
    # Create Plotly bar graph
    fig = px.bar(
        products_data, 
        x='x_label', 
        y='number_of_products',
        title='Number of Unique Products Over Time',
        labels={'x_label': 'Year-Month', 'number_of_products': 'Number of Products'}
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Year-Month',
        yaxis_title='Number of Products',
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

@timed
def plot_number_of_product_returns(filtered_data_r, current_page):
    """
    Plot the number of unique product returns over time as a bar graph.
    
    Parameters:
    - filtered_data_r: Filtered returns DataFrame
    - current_page: Current page context
    """
    from modules.data_process_files import overall_sales
    
    # Prepare data
    product_returns_data = overall_sales.prepare_number_of_product_returns(filtered_data_r)
    
    # Create a custom sorting key with robust month conversion
    product_returns_data['month_numeric'] = product_returns_data['month'].apply(convert_month_to_number)
    product_returns_data['sort_key'] = product_returns_data.apply(lambda row: (int(row['year']), row['month_numeric']), axis=1)
    product_returns_data = product_returns_data.sort_values('sort_key')
    
    # Ensure month is a string and padded
    product_returns_data['month_formatted'] = product_returns_data['month_numeric'].astype(str).str.zfill(2)
    product_returns_data['x_label'] = product_returns_data.apply(lambda row: f"{row['year']}-{row['month_formatted']}", axis=1)
    
    # Create Plotly bar graph
    fig = px.bar(
        product_returns_data, 
        x='x_label', 
        y='number_of_product_returns',
        title='Number of Unique Product Returns Over Time',
        labels={'x_label': 'Year-Month', 'number_of_product_returns': 'Number of Product Returns'}
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Year-Month',
        yaxis_title='Number of Product Returns',
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)



