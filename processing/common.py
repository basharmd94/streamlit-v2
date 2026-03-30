import streamlit as st
import pandas as pd
import io
import base64
import decimal
import math
import json
from utils.utils import timed

def to_dataframe(data, columns):
    """Convert the fetched data to a pandas dataframe."""
    df = pd.DataFrame(data, columns=columns)
    return df

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def find_unique_overtime(data,group_criteria,count_criteria):
    unique_data = data.groupby(group_criteria)[count_criteria].nunique().reset_index()
    return unique_data

def make_aggregates(data,group_criteria,sum_criteria):
    grouped_data = data.groupby(group_criteria)[sum_criteria].sum().reset_index()
    return grouped_data

def find_mean(data,group_criteria,mean_criteria):
    mean_data = data.groupby(group_criteria)[mean_criteria].mean().reset_index()
    return mean_data

def find_median(data,group_criteria,median_criteria):
    median_data = data.groupby(group_criteria)[median_criteria].median().reset_index()
    return median_data

def numerise_columns(data,non_numeric_list):
    # List of all columns that are not in the non-numeric columns list
    numeric_cols = [col for col in data.columns if col not in non_numeric_list]
    # Convert specific columns to numeric
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data

def handle_infinity_and_round(x):
    if isinstance(x, (int, float)):
        # 1️⃣  Leave NaN alone
        if pd.isna(x):                     # handles both np.nan & None
            return x
        # 2️⃣  Convert ±∞ → NaN
        if math.isinf(x):
            return float("nan")
        # 3️⃣  Round anything finite
        return round(x)
    return x

def filter_data_by_column(df, column, selected_values):
    if selected_values:
        return df[df[column].isin(selected_values)]
    return df

def create_download_link(df, filename="data.xlsx"):
    # ── flatten a MultiIndex header, if present ────────────────
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            f"{c[0]}_{c[1].strip()}" if c[1] else f"{c[0]}"
            for c in df.columns.values
        ]

    # ── write the file to an in-memory buffer ─────────────────
    buffer = io.BytesIO()
    df.to_excel(buffer, index=True)      # ← removed encoding='utf-8'
    buffer.seek(0)

    # ── build the download link ───────────────────────────────
    b64 = base64.b64encode(buffer.read()).decode()
    href = (
        f'<a href="data:application/vnd.openxmlformats-officedocument.'
        f'spreadsheetml.sheet;base64,{b64}" download="{filename}">'
        f'Download Excel File</a>'
    )
    return href

def update_single_options(data, col):
    return data[col].unique().tolist()

def update_pair_options(data, col1, col2):
    unique_pairs = data.drop_duplicates(subset=[col1, col2])[[col1, col2]]
    pairs = (unique_pairs[col1].astype(str) + " - " + unique_pairs[col2]).tolist()
    return pairs

def decimal_to_float(data):
    # Convert all decimal.Decimal columns to float
    """
    Convert all decimal columns in a dataframe to float.

    Args:
    - df: Input dataframe.

    Returns:
    - Dataframe with decimal columns converted to float.
    """
    for col in data.columns:
        if any(isinstance(x, decimal.Decimal) for x in data[col]):
            data[col] = data[col].astype(float)

    return data

def find_stats(grouped_data,metric):
    variance = round(grouped_data[metric].var(),2)
    std_dev = round(grouped_data[metric].std(),2)
    minimum = round(grouped_data[metric].min(),2)
    maximum = round(grouped_data[metric].max(),2)
    # Compute the Interquartile Range (IQR) for the 'totalsales' column again
    Q1 = grouped_data[metric].quantile(0.25)
    Q3 = grouped_data[metric].quantile(0.75)
    IQR = round(Q3 - Q1,2)

    skew = round(grouped_data[metric].skew(),2)
    kurt = round(grouped_data[metric].kurt(),2)

    return variance,std_dev,minimum,maximum,IQR,skew,kurt

def time_filtered_data_purchase(sales_data, purchase_data, selected_time):
    days = selected_time * 365
    sales_data['date'] = pd.to_datetime(sales_data['date'], errors='coerce')
    year_ago = sales_data['date'].max() - pd.Timedelta(days=days)
    
    sales_data = sales_data[sales_data['date'].notna() & (sales_data['date'] > year_ago)]
    
    purchase_data['combinedate'] = pd.to_datetime(purchase_data['combinedate'], errors='coerce')
    purchase_data = purchase_data[purchase_data['grnvoucher'].notna() & purchase_data['combinedate'].notna() & (purchase_data['combinedate'] > year_ago)]
    
    return sales_data, purchase_data, year_ago

@timed
def net_pivot(data1, data2, params, current_page=None, data3=None):
    index_cols = params['index'] if isinstance(params['index'], list) else [params['index']]
    column_cols = params['column']
    metric = params['metric']

    df = data1.copy()
    df_r = data2.copy()
    df_c = data3.copy() if data3 is not None else None  # collections (optional)
    print(index_cols, 'index_cols')
    print(column_cols, 'column_cols')
    # print(metric, 'metric')
    try:
        if metric == "Net Sales":
            grouped_sales = df.groupby(index_cols + column_cols)["final_sales"].sum()
            grouped_returns = df_r.groupby(index_cols + column_cols)["treturnamt"].sum()
            result = grouped_sales.subtract(grouped_returns, fill_value=0)

        elif metric == "Net Margin":
            grouped_sales = df.groupby(index_cols + column_cols)["gross_margin"].sum()
            grouped_returns = df_r.groupby(index_cols + column_cols)["treturnamt"].sum()
            grouped_return_cost = df_r.groupby(index_cols + column_cols)["returncost"].sum()
            result = grouped_sales.subtract(grouped_returns, fill_value=0).add(grouped_return_cost, fill_value=0)

        elif metric == "Total Returns":
            result = df_r.groupby(index_cols + column_cols)["treturnamt"].sum()

        elif metric == "Total Discounts":
            result = df.groupby(index_cols + column_cols)["proddiscount"].sum()

        elif metric == "Number of Orders":
            result = df.drop_duplicates("voucher").groupby(index_cols + column_cols)["voucher"].count()

        elif metric == "Number of Returns":
            result = df_r.drop_duplicates("revoucher").groupby(index_cols + column_cols)["revoucher"].count()

        elif metric == "Number of Discounts":
            result = df[df["proddiscount"] > 0].groupby(index_cols + column_cols).size()

        elif metric == "Number of Customers":
            result = df.groupby(index_cols + column_cols)["cusid"].nunique()

        elif metric == "Number of Customer Returns":
            result = df_r.groupby(index_cols + column_cols)["cusid"].nunique()

        elif metric == "Number of Products":
            result = df.groupby(index_cols + column_cols)["itemcode"].nunique()

        elif metric == "Number of Product Returns":
            result = df_r.groupby(index_cols + column_cols)["itemcode"].nunique()

        elif metric == "Collection":
            if df_c is None:
                raise ValueError("Collection data is required for 'Collection' metric.")
            result = df_c.groupby(index_cols + column_cols)["value"].sum()
        
        elif metric == "Net Units Sold":
            result = df.groupby(index_cols + column_cols)["quantity"].sum() - df_r.groupby(index_cols + column_cols)["returnqty"].sum()

        elif metric == "Units Returned":
            result = df_r.groupby(index_cols + column_cols)["returnqty"].sum()

        elif metric == "Units Sold":
            result = df.groupby(index_cols + column_cols)["quantity"].sum()

        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Pivot formatting
        pivot = result.unstack(column_cols).fillna(0)

        # Sort columns properly
        try:
            if isinstance(pivot.columns, pd.MultiIndex) and len(pivot.columns.levels) == 2:
                def safe_sort_key(col):
                    year = int(float(col[0])) if str(col[0]).replace('.', '', 1).isdigit() else 9999
                    month = int(float(col[1])) if str(col[1]).replace('.', '', 1).isdigit() else 99
                    return (year, month)

                sorted_columns = sorted(pivot.columns, key=safe_sort_key)
                pivot = pivot[sorted_columns]
        except Exception as e:
            print(f"Column sort failed: {e}")

        # Add grand total and format
        pivot["Grand Total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values(by="Grand Total", ascending=False)
        pivot = decimal_to_float(pivot)
        pivot = pivot.applymap(lambda x: round(x) if isinstance(x, (int, float)) else x)

        return pivot

    except Exception as e:
        print(f"Error in net_pivot for metric '{metric}': {e}")
        raise


def net_vertical(data1,data2,xaxis,yaxis1,yaxis2,current_page):
    data1_grouped = data1.groupby(xaxis)[yaxis1].sum().reset_index()
    data2_grouped = data2.groupby(xaxis)[yaxis2].sum().reset_index()

    # Merging the grouped data on 'year' and 'month'
    merged_df = pd.merge(data1_grouped, data2_grouped, on=xaxis, how='left')

    # Filling NaN values with 0 for months without returns
    merged_df[yaxis2].fillna(0, inplace=True)

    # Calculating net sales
    merged_df['net'] = merged_df[yaxis1] - merged_df[yaxis2]

    # Sorting the DataFrame by year and month
    merged_df = merged_df.sort_values(['year', 'month']).reset_index(drop=True)

    # Creating the xaxis column
    merged_df['xaxis'] = merged_df['month'].astype(str).str.zfill(2) + "_" + merged_df['year'].astype(str)

    return merged_df,'net'

def get_pair_columns(column):
    """
    Return the paired ID and Name columns based on the provided column name.

    Args:
    - column: The provided column name.

    Returns:
    - Tuple containing the paired ID and Name columns.
    """
    if column.endswith("name"):
        if "item" in column:  # Special case for 'itemcode' and 'itemname'
            return "itemcode", "itemname"
        return column[:-4] + "id", column
    elif column.endswith("id"):
        return column, column[:-2] + "name"
    elif column.endswith("code"):  # Condition for 'code'
        return column, column[:-4] + "name"
    else:
        return f"{column}id", f"{column}name"

def apply_filter_and_update_options(df, df_r, column, display_name, is_pair=False, default=None):
    """
    Apply selected filter to dataframes and update options.

    Args:
    - df: Sales dataframe.
    - df_r: Returns dataframe.
    - column: Column name to apply the filter on.
    - display_name: Name to display on the Streamlit sidebar.
    - is_pair: If True, expects paired options (e.g., 'id - name').
    - default: Default value for the filter.

    Returns:
    - Filtered dataframes: df, df_r.
    """
    if is_pair:
        col1, col2 = get_pair_columns(column)
        options = update_pair_options(df, col1, col2)
    else:
        options = update_single_options(df, column)

    if column == "year" and default is None:
        current_year = max(set([int(value) for value in options]))
        last_year = current_year - 1
        default = [last_year, current_year]
    
    selected = st.sidebar.multiselect(f"Select {display_name}", options, default=default)
    
    if selected:
        if is_pair:
            selected_values = [x.split(" - ")[1] for x in selected]  # Extract the name part from 'id - name'
            df = df[df[f"{column}"].isin(selected_values)]
            df_r = df_r[df_r[f"{column}"].isin(selected_values)]
        else:
            df = df[df[column].isin(selected)]
            df_r = df_r[df_r[column].isin(selected)]
    
    return df, df_r

def filtered_options(filtered_data, filtered_data_r):
    """
    Filter dataframes based on user selections in Streamlit interface.

    Args:
    - filtered_data: Sales dataframe.
    - filtered_data_r: Returns dataframe.

    Returns:
    - Filtered dataframes: filtered_data, filtered_data_r.
    """
    # Apply filters in sequence: 'year', 'month', 'spname' (salesman), 'cusname' (customer), 'itemname' (product), 'area', 'itemgroup' (product group)
    filters_sequence = [
        ('year', 'Year', False),
        ('month', 'Month', False),
        ('spname', 'Salesman', True),
        ('cusname', 'Customer', True),
        ('itemname', 'Product', True),
        ('area', 'Area', False),
        ('itemgroup', 'Product Group', False)
    ]

    for column, display_name, is_pair in filters_sequence:
        filtered_data, filtered_data_r = apply_filter_and_update_options(filtered_data, filtered_data_r, column, display_name, is_pair)
    
    return filtered_data, filtered_data_r

def data_copy_add_columns(*dfs):
    """
    Clean and transform each dataframe efficiently.
    """
    month_map = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }

    processed_dfs = []

    for df in dfs:
        df = df.copy()

        # Convert date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df[df['date'].notna()]
            df['DOM'] = df['date'].dt.day
            df['DOW'] = df['date'].dt.day_name()

        # Map month numbers to names
        if 'month' in df.columns and pd.api.types.is_integer_dtype(df['month']):
            df['month_name'] = df['month'].map(month_map)

        # Convert numeric columns
        numeric_cols = ['altsales','proddiscount','totalsales', 'treturnamt', 'value', 'quantity', 'cost']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute derived columns
        if 'cost' in df.columns and 'altsales' in df.columns and 'proddiscount' in df.columns:
            df['final_sales'] = df['altsales'] - df['proddiscount']
            df['gross_margin'] = df['final_sales'] - df['cost']
        if 'quantity' in df.columns and 'final_sales' in df.columns:
            df['rate'] = df['final_sales'] / df['quantity']

        # Convert decimals to float
        df = decimal_to_float(df)

        processed_dfs.append(df)

    return tuple(processed_dfs)

def enrich_collection_with_sales_info(filtered_data_c, filtered_data_s):
    # Create mapping from sales data (cusid -> spid, spname)
    print("Before:", filtered_data_c["value"].sum())


    # Step 1: Extract mapping of most recent spid/spname per customer
    latest_sales = (
        filtered_data_s.sort_values("date")  # or use .sort_values("year", "month", "date") if more granular
        .drop_duplicates(subset=["cusid"], keep="last")[["cusid", "spid", "spname", "area"]]
    )

    # Step 2: Merge safely without duplication
    enriched_df = pd.merge(
        filtered_data_c,
        latest_sales,
        on="cusid",
        how="left",
        validate="many_to_one"  # 🚨 ensures each cusid in collection maps to only one row
    )

    print("After:", enriched_df["value"].sum()) 


    return enriched_df

def add_voucher_type_ar(filtered_data_ar):
    filtered_data_ar['voucher_type'] = filtered_data_ar['voucher'].str.extract(r"^([A-Za-z]+)")

    voucher_type_dict = {
    'OB':'Opening', #done
    'INOP':'Sales', #done
    'RCT':'Collection', #done
    'SRJV':'Return', #done
    'SRT':'Return', #done
    'JV':'Adjustment', #done
    'IMSA':'Sales', #done
    'STJV':'Collection', #done
    'CPAY':'Adjustment', #done
    'PAY':'Adjustment', #done
    'BRCT':'Collection', #done
    'CRCT':'Collection', #done
    'CHQ':'Collection', #done
    'ADJV':'Collection', #done
    'TR':'Adjustment', #done
    'BTJV':'Collection' #done
    }

    filtered_data_ar['voucher_type_desc'] = filtered_data_ar['voucher_type'].map(voucher_type_dict)
    filtered_data_ar["sign"] = filtered_data_ar["value"].apply(
        lambda x: 1 if x > 0 else 
                  -1 if x < 0 else 
                  0
    )
    filtered_data_ar = filtered_data_ar.sort_values(['cusid','date'])
    filtered_data_ar['value'] = pd.to_numeric(filtered_data_ar['value'], errors='coerce').fillna(0)

    now = pd.Timestamp.now()
    current_year, current_month = now.year, now.month
    filtered_data_ar = filtered_data_ar[
        (filtered_data_ar['year'] < current_year) |
        ((filtered_data_ar['year'] == current_year) &
        (filtered_data_ar['month'] < current_month))
    ]
    # Add running balance
    filtered_data_ar['running_balance'] = filtered_data_ar.groupby(['cusid','year'])['value'].cumsum()
    return filtered_data_ar