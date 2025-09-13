import pandas as pd
from modules.data_process_files import common

# Function to compute the relative week
def compute_relative_week(start_date, current_date):
    days_difference = (current_date - start_date).days
    relative_week = (days_difference // 7) + 1
    if 1 <= relative_week <= 12:
        return relative_week
    return None

def get_top_associated_items(item, data, num_top=5):
    """
    Get the top associated items purchased with a given item.

    Parameters:
    - item: The item for which we want to find associated items.
    - data: The binary matrix indicating item presence in transactions.
    - num_top: The number of top associated items to return.

    Returns:
    - top_items: A Series of the top associated items with their support values.
    """
    # Filter the data for transactions where the item is purchased
    item_data = data[data[item] == 1]
    
    # Calculate the support for other items in these transactions
    support_data = item_data.mean().drop(item)
    
    # Get the top num_top associated items
    top_items = support_data.nlargest(num_top)
    
    return top_items

def purchase_basket_analysis(purchase_df):
    # Preprocess the data to create a binary matrix for purchase transactions
    purchase_basket = purchase_df.groupby(['grnvoucher', 'itemcode'])['quantity'].sum().unstack().reset_index().fillna(0).set_index('grnvoucher')
    purchase_basket_sets = purchase_basket.applymap(lambda x: 1 if x >= 1 else 0)

    # 1. Calculate the number of times each product was purchased
    purchase_counts = purchase_basket_sets.sum(axis=0)

    # 2. Extract the item names for each product code
    item_names = purchase_df.set_index('itemcode')['itemname'].to_dict()

    # 3. Modify the report structure to include the additional columns
    purchase_report_data = []
    for item in purchase_basket_sets.columns:
        row_data = [item, item_names.get(item, "Unknown")]  # Include the item name
        top_associated = get_top_associated_items(item, purchase_basket_sets)
        for assoc_item, support in top_associated.items():
            row_data.extend([assoc_item, support])
        row_data.append(purchase_counts.get(item, 0))  # Include the number of times the item was purchased
        purchase_report_data.append(row_data)

    # Update the column names to include 'Item Name' and 'Purchase Count'
    extended_purchase_column_names = ['Item of Interest', 'Item Name']
    for i in range(1, 6):
        extended_purchase_column_names.extend([f"Item {i}", f"% Support {i}"])
    extended_purchase_column_names.append('Purchase Count')

    # Create the updated DataFrame for the purchase report
    purchase_report_df = pd.DataFrame(purchase_report_data, columns=extended_purchase_column_names)

    # 4. Sort the report based on the 'Purchase Count' column
    sorted_purchase_report_df = purchase_report_df.sort_values(by='Purchase Count', ascending=False)
    sorted_purchase_report_df = common.decimal_to_float(sorted_purchase_report_df)
    sorted_purchase_report_df = sorted_purchase_report_df.round(4)
    return sorted_purchase_report_df

def sales_basket_analysis(sales_df,inventory_df):
        # Preprocess the data to create a binary matrix for sales transactions
    sales_basket = sales_df.groupby(['voucher', 'itemcode'])['quantity'].sum().unstack().reset_index().fillna(0).set_index('voucher')
    sales_basket_sets = sales_basket.applymap(lambda x: 1 if x >= 1 else 0)

    # Extract the item names for each product code
    item_names_sales = sales_df.drop_duplicates(subset='itemcode').set_index('itemcode')['itemname'].to_dict()

    # Modify the report structure to interleave item names after each item code
    sales_report_data = []
    for item in sales_basket_sets.columns:
        row_data = [item, item_names_sales.get(item, "Unknown")]  # Include the item name for the item of interest
        top_associated = get_top_associated_items(item, sales_basket_sets)
        
        for assoc_item, support in top_associated.items():
            assoc_item_name = item_names_sales.get(assoc_item, "Unknown")
            row_data.extend([assoc_item, assoc_item_name, support])
        
        sales_report_data.append(row_data)

    # Update the column names to include 'Item Name' after each 'Item' column
    extended_sales_column_names = ['Item of Interest', 'Item Name']
    for i in range(1, 6):
        extended_sales_column_names.extend([f"Item {i}", f"Item {i} Name", f"% Support {i}"])

    # Create the DataFrame for the sales report
    sales_report_with_names_df = pd.DataFrame(sales_report_data, columns=extended_sales_column_names)

        # Calculate total sales for each item
    item_totalsales = sales_df.groupby('itemcode')['totalsales'].sum()

    # Create a function to get the total sales for a list of item codes
    def get_total_sales(itemcodes):
        return [item_totalsales.get(code, 0) for code in itemcodes]

    # Initialize lists to store the sales data
    item_of_interest_sales = []
    item1_sales = []
    item2_sales = []
    item3_sales = []
    item4_sales = []
    item5_sales = []
    total_sales = []

    # Loop through the DataFrame to get the total sales for each item code
    for index, row in sales_report_with_names_df.iterrows():
        item_of_interest_code = row['Item of Interest']
        item1_code = row.get('Item 1', None)
        item2_code = row.get('Item 2', None)
        item3_code = row.get('Item 3', None)
        item4_code = row.get('Item 4', None)
        item5_code = row.get('Item 5', None)
        
        item_of_interest_sales.append(get_total_sales([item_of_interest_code])[0])
        item1_sales.append(get_total_sales([item1_code])[0])
        item2_sales.append(get_total_sales([item2_code])[0])
        item3_sales.append(get_total_sales([item3_code])[0])
        item4_sales.append(get_total_sales([item4_code])[0])
        item5_sales.append(get_total_sales([item5_code])[0])
        
        total_sales.append(sum(get_total_sales([item_of_interest_code, item1_code, item2_code, item3_code, item4_code, item5_code])))

    # Add the sales data to the DataFrame
    sales_report_with_names_df['Item of Interest Sales'] = item_of_interest_sales
    sales_report_with_names_df['Item 1 Sales'] = item1_sales
    sales_report_with_names_df['Item 2 Sales'] = item2_sales
    sales_report_with_names_df['Item 3 Sales'] = item3_sales
    sales_report_with_names_df['Item 4 Sales'] = item4_sales
    sales_report_with_names_df['Item 5 Sales'] = item5_sales
    sales_report_with_names_df['Total Sales (All Items)'] = total_sales

    # Sort the DataFrame based on the 'Total Sales (All Items)' column
    sorted_sales_report_with_all_sales = sales_report_with_names_df.sort_values(by='Total Sales (All Items)', ascending=False)
    sorted_sales_report_with_all_sales = sorted_sales_report_with_all_sales.merge(inventory_df,left_on='Item of Interest',right_on='itemcode',how='left').drop('itemcode',axis=1)
    sorted_sales_report_with_all_sales = common.decimal_to_float(sorted_sales_report_with_all_sales)
    sorted_sales_report_with_all_sales = sorted_sales_report_with_all_sales.round(4)
    return sorted_sales_report_with_all_sales

