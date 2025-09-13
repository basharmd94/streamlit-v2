import pandas as pd
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from modules.data_process_files import common
from datetime import datetime

def generate_cohort(purchase_data,year_ago,inventory_data,sales_df,cohort_df):
    cohort_df = cohort_latest(cohort_df)
    purchase_order = time_filtered_data_requisition(purchase_data,year_ago)
    
    purchase_order = purchase_order[['itemcode','itemname','shipmentname','quantity','combinedate']]
    purchase_pivot = purchase_order.pivot_table(index=['itemcode'], columns='shipmentname', values='quantity', aggfunc='sum').reset_index()
    

    inventory_df = inventory_data[inventory_data['itemgroup'].isin(['Finished Goods Packaging','RAW Material Packaging','Furniture Fittings','Hardware','Industrial & Household','Sanitary'])]
    inventory_df = inventory_df[['itemcode','itemname','itemgroup','stockqty']]
    caitem = inventory_df[['itemcode','itemname','itemgroup']]
    caitem = caitem.drop_duplicates(subset='itemcode', keep='first')
    
    non_numeric_cols = ['itemcode','itemname','itemgroup']
    inventory_df = common.numerise_columns(inventory_df,non_numeric_cols)
    inventory_df = inventory_df.groupby(['itemcode']).agg({'stockqty': 'sum'}).reset_index()
    inventory_df = pd.merge(inventory_df,caitem,on='itemcode',how='left')
    # inventory_df = inventory_df.groupby(['itemcode','itemname']).agg({'stockqty': 'sum','opstdprice': 'mean'}).reset_index()
    
    # Group by 'itemcode', 'itemname', and month to compute the total sales for each product in each month
    sales_df = sales_df.groupby(['itemcode', 'year', 'month']).agg({'quantity': 'sum'}).reset_index()
    # Compute monthly average for each product
    non_numeric_cols = ['itemcode','year','month']
    sales_df = common.numerise_columns(sales_df,non_numeric_cols)
    monthly_avg = sales_df.groupby(['itemcode'])['quantity'].transform('mean')
    # Filter out months where sales are less than 20% of the monthly average for that product
    sales_df = sales_df[sales_df['quantity'] >= 0.2 * monthly_avg]
    # Compute the average sales per month for each product
    sales_df = sales_df.groupby(['itemcode'])['quantity'].mean().reset_index().sort_values('quantity').rename(columns={'quantity':'n-mean'})

    cohort_df = cohort_df.drop(columns=['itemname'])
    merged_df = pd.merge(inventory_df, cohort_df, on='itemcode', how='left')
    merged_df = pd.merge(merged_df, purchase_pivot, on='itemcode', how='left')
    merged_df = pd.merge(merged_df,sales_df,on='itemcode',how='left').fillna(0).sort_values('n-mean',ascending=False)

    # Dynamically identify all shipment columns
    shipment_columns = [col for col in merged_df.columns if 'MDKF' in col]
    shipment_columns_sorted = sorted(shipment_columns, key=lambda x: x.split(",")[1])

    merged_df = common.decimal_to_float(merged_df)

    # Reinitialize the result DataFrame
    result_df = merged_df.copy()

    # Calculate stock before each shipment
    current_stock = merged_df['stockqty'].copy()

    for i, shipment in enumerate(shipment_columns_sorted):
        # For the first shipment, subtract n-mean from the current date to the shipment date
        if i == 0:
            days_until_shipment = (pd.to_datetime(shipment.split(",")[1]) - datetime.now()).days
            print(days_until_shipment)
        # For subsequent shipments, subtract n-mean for the days between shipments
        else:
            days_until_shipment = (pd.to_datetime(shipment.split(",")[1]) - pd.to_datetime(shipment_columns_sorted[i-1].split(",")[1])).days
            print(days_until_shipment)

        applicable_monthly_sales = days_until_shipment / 30  # Approximate number of months until shipment
        current_stock -= applicable_monthly_sales * merged_df['n-mean']
        current_stock = current_stock.clip(lower=0)  # Ensure stock doesn't go negative

        # Store the predicted stock before the shipment in our result dataframe
        result_df[f'stock_before_{shipment}'] = current_stock
        # Add the shipment quantity to the current stock
        current_stock += merged_df[shipment]

    # Predict stock 1 month after the last shipment
    current_stock -= result_df['n-mean']
    current_stock = current_stock.clip(lower=0)  # Ensure stock doesn't go negative
    result_df['final_stock'] = current_stock
    result_df['Week%'] = result_df['quantity']/result_df['1']

    # Define the desired order of columns, excluding shipment columns for now
    desired_columns = [
        'itemcode','itemname', 'stockqty', 'cost', 'average sales price',
        'combinedate', 'quantity', '1', '2', '3', '4', 'n-mean'
    ]

    # Add 'stock_before_' columns for each shipment to the desired columns list
    for shipment in shipment_columns_sorted:
        desired_columns.append(f'stock_before_{shipment}')
        desired_columns.append(shipment)

    # Add the 'final_stock' column to the desired columns list
    desired_columns.append('final_stock')

    # Filter and reorder the columns of the result dataframe
    result_df = result_df[desired_columns]

    # Create a column to determine if a product has shipments in the last three shipments
    last_three_shipments = shipment_columns_sorted[-3:]
    result_df['has_shipments'] = result_df[last_three_shipments].sum(axis=1) > 0

    # Sort the dataframe first by 'has_shipments' (descending so products with shipments come first)
    # Then by 'final_stock' (ascending) and then by 'n-mean' (descending)
    result_df = result_df.sort_values(by=['has_shipments', 'final_stock', 'n-mean'], ascending=[False, True, False])
    result_df = result_df.drop(columns='has_shipments')
    result_df['combinedate'] = pd.to_datetime(result_df['combinedate'], errors='coerce')
    result_df['combinedate'] = result_df['combinedate'].dt.strftime('%Y-%m-%d')
    result_df['itemname'] = result_df['itemname'].astype(str)

    # products_to_order
    result_df = result_df.applymap(common.handle_infinity_and_round).fillna(0)

    return result_df

def time_filtered_data_requisition(purchase_data,year_ago):
    purchase_data = purchase_data[~purchase_data['grnvoucher'].notna()]
    purchase_data['combinedate'] = pd.to_datetime(purchase_data['combinedate'], errors='coerce')
    purchase_data = purchase_data[purchase_data['combinedate'].notna()]
    purchase_data = purchase_data[purchase_data['combinedate'] > year_ago]
    return purchase_data

def cohort_latest(cohort_df):
    # Initialize an empty list to hold the results
    latest_data = []

    # For each unique itemcode
    for item in cohort_df['itemcode'].unique():
        subset = cohort_df[cohort_df['itemcode'] == item].tail(1)

        # Retrieve the itemcode and itemname
        itemcode = subset['itemcode'].values[0]
        itemname = subset['itemname'].values[0]

        # Retrieve the latest cost and average sales price
        latest_cost = subset['cost'].values[0]
        latest_avg_sales_price = subset['average sales price'].values[0]
        latest_date = subset['combinedate'].values[0]
        latest_quantity = subset['quantity'].values[0]

        # Retrieve the latest values for columns '1','2','3','4'
        col_1 = subset['1'].values[0]
        col_2 = subset['2'].values[0]
        col_3 = subset['3'].values[0]
        col_4 = subset['4'].values[0]

        # Append the results to the list
        latest_data.append([itemcode, itemname, latest_cost, latest_avg_sales_price, latest_date,latest_quantity ,col_1, col_2, col_3, col_4])

    # Convert the latest data to a DataFrame
    latest_df = pd.DataFrame(latest_data, columns=['itemcode', 'itemname', 'cost', 'average sales price','combinedate','quantity', '1', '2', '3', '4'])

    return latest_df

def process_chunk(chunk, sales_df):
    data_rows = []
    
    for _, purchase in chunk.iterrows():
        relevant_sales = sales_df[sales_df['itemcode'] == purchase['itemcode']].copy()
        relevant_sales['relative_week'] = (relevant_sales['date'] - purchase['combinedate']).dt.days // 7 + 1
        
        # Filter relevant sales
        relevant_sales = relevant_sales[(relevant_sales['relative_week'] > 0) & (relevant_sales['relative_week'] <= 12)]
        
        aggregated_sales = relevant_sales.groupby('relative_week').agg({'quantity': 'sum', 'totalsales': 'sum'}).reset_index()
        
        week_sales = {str(week): 0 for week in range(1, 13)}
        for week, qty in zip(aggregated_sales['relative_week'], aggregated_sales['quantity']):
            week_sales[str(week)] = qty
        
        total_qty = aggregated_sales['quantity'].sum()
        total_amount = aggregated_sales['totalsales'].sum()

        avg_price = total_amount / total_qty if total_qty else 0

        data_dict = {
            'itemcode': purchase['itemcode'],
            'itemname': purchase['itemname'],
            'povoucher': purchase['povoucher'],
            'shipmentname': purchase['shipmentname'],
            'combinedate': purchase['combinedate'],
            'quantity': purchase['quantity'],
            'cost': purchase['cost'],
            'average sales price': avg_price,
            **week_sales
        }

        data_rows.append(data_dict)

    return data_rows

def main_purchase_product_cohort_process(sales_df, purchase_df):
    num_processes = 4
    chunks = np.array_split(purchase_df, num_processes)
    partial_process_chunk = partial(process_chunk, sales_df=sales_df)
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(partial_process_chunk, chunks))

    final_df = pd.DataFrame([item for sublist in results for item in sublist])
    final_df = final_df.sort_values(by=['itemcode', 'combinedate'])
    final_df['days_since_last_purchase'] = final_df.groupby('itemcode')['combinedate'].diff().fillna(pd.Timedelta(seconds=0)).dt.days
    final_df['days_since_last_purchase'] = final_df['days_since_last_purchase'].fillna(0).astype(int)
    final_df['combinedate'] = final_df['combinedate'].dt.strftime('%Y-%m-%d')
    final_df = common.decimal_to_float(final_df)

        # Calculate sum and handle division by zero
    sum_values = final_df[[str(i) for i in range(1, 5)]].sum(axis=1)
    sum_values = sum_values.replace(0, np.nan)  # Replace zeros with NaN to avoid division by zero
    final_df['days_of_product_left'] = (final_df['quantity'] * 30.5) / sum_values

    # Handle inf values
    final_df = final_df.replace([np.inf, -np.inf], np.nan)

    final_df = final_df.applymap(lambda x: round(x) if isinstance(x, (int, float)) and not pd.isna(x) else x).fillna(0)

    # final_df['days_of_product_left'] = (final_df['quantity'] * 30.5) / final_df[[str(i) for i in range(1, 5)]].sum(axis=1)
    # final_df = final_df.applymap(lambda x: round(x) if isinstance(x, (int, float)) else x).fillna(0)
    return final_df


