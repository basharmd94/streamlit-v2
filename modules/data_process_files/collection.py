import streamlit as st
import pandas as pd
import numpy as np
from modules.data_process_files import common
from collections import defaultdict
pd.set_option('display.float_format', '{:.2f}'.format)
from utils.utils import timed
import plotly.express as px
import calendar
from typing import Tuple, Dict

#old

#Collections
def customer_segmentation_by_collection_days(average_collection_df):
    # Define bin edges and labels
    bin_edges = [0, 2, 5, 10, 20, 30, 50, 100]  # Adjust according to your needs
    bin_labels = ['0-2 days', '3-5 days', '6-10 days', '11-20 days', '21-30 days', '31-50 days', '51-100 days']  # Adjust according to your needs

    # Assign each row to a bin
    average_collection_df['range_of_average_days'] = pd.cut(average_collection_df['average_days_to_collection'], bins=bin_edges, labels=bin_labels, right=True)

    # Group by the bins and calculate the count, total sales, total return, and total collection for each bin
    summary_df = average_collection_df.groupby('range_of_average_days').agg(
        count=pd.NamedAgg(column='average_days_to_collection', aggfunc='count'),
        total_sales=pd.NamedAgg(column='total_sale', aggfunc='sum'),  # Replace 'sales' with the actual column name for sales
        total_return=pd.NamedAgg(column='total_return', aggfunc='sum'),  # Replace 'return' with the actual column name for return
        total_collection=pd.NamedAgg(column='total_collection', aggfunc='sum')  # Replace 'collection' with the actual column name for collection
    ).reset_index()

    return summary_df

def filtered_options_for_collection(sales_data, returns_data, collections_data):
    datasets = {
        'sales': sales_data,
        'returns': returns_data,
        'collections': collections_data
    }

    # Year Filter
    year_options = common.update_single_options(sales_data, 'year')
    current_year = max(set([int(value) for value in year_options]))
    selected_year = st.multiselect("Select Year", year_options, default=current_year)
    
    # Month Filter
    month_options = common.update_single_options(sales_data, 'month')
    selected_month = st.multiselect("Select Month", month_options)
    
    # Customer Filter
    customer_options = common.update_pair_options(sales_data, 'cusid', 'cusname')
    selected_customers = st.multiselect("Select Customer", customer_options)
    selected_cusnames = [x.split(" - ")[1] for x in selected_customers]
    selected_cusids = [x.split(" - ")[0] for x in selected_customers]

    for key, data in datasets.items():
        data = common.filter_data_by_column(data, 'year', selected_year)
        data = common.filter_data_by_column(data, 'month', selected_month)
        if key not in ('collections'):
            data = common.filter_data_by_column(data, 'cusname', selected_cusnames)
        elif key == 'collections':
            data = common.filter_data_by_column(data, 'cusid', selected_cusids)
        else:
            data
        datasets[key] = data

    # Salesman Filter 
    salesman_options = common.update_pair_options(sales_data, 'spid', 'spname')
    selected_salesman = st.multiselect("Select Salesman", salesman_options)
    selected_spnames = [x.split(" - ")[1] for x in selected_salesman]
    selected_spids = [x.split(" - ")[0] for x in selected_salesman]

    for key, data in datasets.items():
        data = common.filter_data_by_column(data, 'year', selected_year)
        data = common.filter_data_by_column(data, 'month', selected_month)
        if key not in ('collections'):
            data = common.filter_data_by_column(data, 'spname', selected_spnames)
        elif key == 'collections':
            selected_cusids = sales_data.loc[sales_data['spid'].isin(selected_spids), 'cusid'].tolist()
            data = common.filter_data_by_column(data, 'cusid', selected_cusids)
        else:
            data
        datasets[key] = data

    return datasets['sales'], datasets['returns'], datasets['collections']

def get_grouped_df_collection(sales_df, returns_df, collection_df, timeframe):
    # Step 1: Determine group key
    if timeframe == "Yearly":
        group_key = ["year"]
    elif timeframe == "Monthly":
        group_key = ["year", "month"]
    elif timeframe == "Daily":
        group_key = ["date"]
    else:
        raise ValueError("Invalid timeframe selected. Only 'Daily', 'Monthly', or 'Yearly' are allowed.")

    # Step 2: Aggregations based on the group key
    sales_time = sales_df.groupby(group_key)["final_sales"].sum().reset_index(name='final_sales')
    returns_time = returns_df.groupby(group_key)["treturnamt"].sum().reset_index(name='treturnamt')
    collection_time = collection_df.groupby(group_key)["value"].sum().abs().reset_index(name='collection')
    # Step 3: Merge based on group key
    merged_time = pd.merge(sales_time, returns_time, on=group_key, how='outer')
    merged_time = pd.merge(merged_time, collection_time, on=group_key, how='outer')
    
    # Step 4: DOM and DOW separately
    sales_dom = sales_df.groupby('DOM')["final_sales"].sum().reset_index(name='final_sales')
    returns_dom = returns_df.groupby('DOM')["treturnamt"].sum().reset_index(name='treturnamt')
    collection_dom = collection_df.groupby('DOM')["value"].sum().abs().reset_index(name='collection')

    merged_dom = pd.merge(sales_dom, returns_dom, on="DOM", how="outer")
    merged_dom = pd.merge(merged_dom, collection_dom, on="DOM", how="outer")

    sales_dow = sales_df.groupby('DOW')["final_sales"].sum().reset_index(name='final_sales')
    returns_dow = returns_df.groupby('DOW')["treturnamt"].sum().reset_index(name='treturnamt')
    collection_dow = collection_df.groupby('DOW')["value"].sum().abs().reset_index(name='collection')

    merged_dow = pd.merge(sales_dow, returns_dow, on="DOW", how="outer")
    merged_dow = pd.merge(merged_dow, collection_dow, on="DOW", how="outer")

    return merged_time.fillna(0), merged_dom.fillna(0), merged_dow.fillna(0)

def average_days_to_collection(sales_df, returns_df, collection_df):
    # Combine and sort data
    combined_df = pd.concat([
        sales_df.assign(type='sale'),
        returns_df.assign(type='return'),
        collection_df.assign(type='collection')
    ], ignore_index=True).sort_values(by=['cusid', 'date'])
    
    # Create a mapping for customer names
    cusname_mapping = sales_df.drop_duplicates(subset=['cusid']).set_index('cusid')['cusname'].to_dict()

    # Process combined data
    ongoing_calculations = defaultdict(lambda: {'last_sale_date': None, 'adjusted_sales_amount': 0, 'total_days': 0, 'count': 0})
    collection_days_list = defaultdict(list)

    for _, row in combined_df.iterrows():
        cusid = row['cusid']
        ongoing = ongoing_calculations[cusid]
        ongoing['cusname'] = cusname_mapping.get(cusid, '<Unknown>')
        
        if row['type'] == 'sale':
            ongoing['last_sale_date'] = row['date']
            ongoing['adjusted_sales_amount'] += row['final_sales']
        elif row['type'] == 'return' and ongoing['last_sale_date']:
            ongoing['adjusted_sales_amount'] -= row['treturnamt']
        elif row['type'] == 'collection' and ongoing['last_sale_date'] and ongoing['adjusted_sales_amount'] > 0:
            days = (row['date'] - ongoing['last_sale_date']).days
            ongoing['total_days'] += days
            ongoing['count'] += 1
            collection_days_list[cusid].append({'cusname': ongoing['cusname'], 'year': row['date'].year, 'month_of_collection': row['date'].month, 'collection_days': days})

    # Create average days dataframe
    avg_days_data = {k: {'cusname': v['cusname'], 'average_days_to_collection': v['total_days'] / v['count']} for k, v in ongoing_calculations.items() if v['count'] > 0}
    average_days_df = pd.DataFrame.from_dict(avg_days_data, orient='index').reset_index().rename(columns={'index': 'cusid'})
    
    # Create collection days dataframe
    collection_days_data = [(k, v['cusname'], v['year'], v['month_of_collection'], v['collection_days']) for k, vals in collection_days_list.items() for v in vals]
    collection_days_df = pd.DataFrame(collection_days_data, columns=['cusid', 'cusname', 'year', 'month_of_collection', 'collection_days'])
    
    # Create pivot table
    pivot_df = pd.pivot_table(collection_days_df, values='collection_days', index=['cusid', 'cusname'], columns=['year', 'month_of_collection'], aggfunc='mean').fillna(0)
    pivot_df['Average'] = pivot_df.apply(lambda row: row[row != 0].mean(), axis=1)
    pivot_df = pivot_df.reset_index()
    
    # Merge totals
    totals = {
        'total_sale': sales_df.groupby('cusid')['final_sales'].sum(),
        'total_return': returns_df.groupby('cusid')['treturnamt'].sum(),
        'total_collection': collection_df.groupby('cusid')['value'].sum()
    }
    
    for col, series in totals.items():
        average_days_df = average_days_df.merge(series.rename(col), on='cusid', how='left').fillna(0)
        # pivot_df = pivot_df.merge(series.reset_index(name=col), on=['cusid', 'cusname'], how='left').fillna(0)
    
    # Process collection data for average days between collections
    collection_data = combined_df[combined_df['type'] == 'collection'].copy()
    collection_data['cusname'].fillna(collection_data['cusid'].map(cusname_mapping), inplace=True)
    collection_data.sort_values(by=['cusid', 'cusname', 'date'], inplace=True)
    collection_data['days_between'] = collection_data.groupby(['cusid', 'cusname'])['date'].diff().dt.days
    
    avg_days_between = collection_data.groupby(['cusid', 'cusname']).agg(average_days_between=('days_between', 'mean'), average_collection=('value', 'mean')).reset_index()
    
    return average_days_df, pivot_df, avg_days_between, combined_df

#Accounts Receivables
def compute_order_frequency_metrics(filtered_data_ar: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 1. Copy & parse dates
    df = filtered_data_ar.copy()
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    
    # 2. Keep only Sales
    df_sales = df[df['voucher_type_desc'] == 'Sales'].sort_values(['cusid', 'date'])
    
    # 3. Compute days between orders
    df_sales['Interval'] = df_sales.groupby('cusid')['date'].diff().dt.days

    # 4. Aggregate
    metrics = (
        df_sales
        .groupby(['cusid', 'cusname', 'area', 'year'])
        .agg(
            Order_Count   = ('voucher', 'count'),
            Avg_Interval  = ('Interval', 'mean'),
            Std_Interval  = ('Interval', 'std'),
        )
        .reset_index()
    )

    # 5. Pivot each metric separately

    # --- Order Count pivot ---
    order_count_pivot = metrics.pivot(
        index=['cusid', 'cusname','area'],
        columns='year',
        values='Order_Count'
    )
    order_count_pivot.columns = [f"Order_Count_{yr}" for yr in order_count_pivot.columns]
    order_count_df = order_count_pivot.reset_index()

    # --- Average Interval pivot ---
    avg_interval_pivot = metrics.pivot(
        index=['cusid', 'cusname','area'],
        columns='year',
        values='Avg_Interval'
    )
    avg_interval_pivot.columns = [f"Avg_Interval_{yr}" for yr in avg_interval_pivot.columns]
    avg_interval_df = avg_interval_pivot.reset_index()

    # --- Std Interval pivot ---
    std_interval_pivot = metrics.pivot(
        index=['cusid', 'cusname','area'],
        columns='year',
        values='Std_Interval'
    )
    std_interval_pivot.columns = [f"Std_Interval_{yr}" for yr in std_interval_pivot.columns]
    std_interval_df = std_interval_pivot.reset_index()

    return order_count_df, avg_interval_df, std_interval_df

@st.cache_data
def compute_payment_timeliness_metrics(df: pd.DataFrame,tolerance: float = 10,on_time_limit: int = 30) -> Dict[str, pd.DataFrame]:
    # 1) Parse dates
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    #list of opening balances per customer (opening 2024) 
    ob = (df[df['voucher_type_desc']=='Opening'].groupby(['cusid','cusname','area','year'])['value'].sum().rename('opening_balance').reset_index())
    ob['opening_balance'] = pd.to_numeric(ob['opening_balance'], errors='coerce').fillna(0.0)
    ledger_txns = df[df['voucher_type_desc']!='Opening'].copy()
    ledger_txns['value'] = pd.to_numeric(ledger_txns['value'], errors='coerce').fillna(0.0)

    # 3) Invoices table
    invoices = (
        ledger_txns[ledger_txns['voucher_type_desc']=='Sales']
        .rename(columns={'date':'invoice_date','value':'invoice_amount'})
        [['cusid','cusname','area','invoice_date','invoice_amount','year','month','running_balance']]
    )
    invoices['prior_balance'] = invoices['running_balance'] - invoices['invoice_amount']

    # 4) FIFO allocation
    paid = []
    for cusid, grp in invoices.groupby('cusid'):
        cust_led = ledger_txns.loc[ledger_txns['cusid']==cusid,['date','running_balance']].reset_index(drop=True)
        for _, inv in grp.iterrows():
            thresh = inv['prior_balance'] + tolerance
            sub = cust_led[cust_led['date']>=inv['invoice_date']]
            mask = sub['running_balance'] <= thresh
            paid.append(sub.loc[mask,'date'].iloc[0] if mask.any() else pd.NaT)
    invoices['paid_date'] = pd.Series(paid, index=invoices.index)
    invoices['days_to_pay'] = (invoices['paid_date'] - invoices['invoice_date']).dt.days
    invoices['pay_bucket'] = pd.cut(invoices['days_to_pay'], bins=[0,10,20,30,40,np.inf],labels=['0–10','11–20','21–30','31–40','>45'], right=True)

    # 5) Closing balances per cust-year-month
    opening_per_year = ob.groupby('year')['opening_balance'].sum().reset_index(name='opening_balance')
    tx = (ledger_txns.groupby(['year','month'])['value'].sum().reset_index(name='month_txn_sum'))
    tx = tx.merge(opening_per_year, on=['year'], how='left').fillna({'opening_balance':0})
    tx = tx.sort_values(['year','month'])
    tx['month_txn_sum'] = pd.to_numeric(tx['month_txn_sum'], errors='coerce').fillna(0)
    tx['cum_tx'] = tx.groupby(['year','month'])['month_txn_sum'].cumsum()
    tx['closing_balance'] = tx['opening_balance'] + tx['cum_tx']

    # 6) Year-Month Summary
    ym = tx.groupby(['year','month'])['closing_balance'].sum().rename('Total_Balance')
    sales_ym = invoices.groupby(['year','month'])['invoice_amount'].sum().rename('Total_Sales')
    col_ym = ledger_txns[ledger_txns['value']<0].groupby(['year','month'])['value'].sum().abs().rename('Total_Collection')
    ret_ym = ledger_txns[ledger_txns['voucher_type_desc']=='Return'].groupby(['year','month'])['value'].sum().abs().rename('Total_Returns')
    ym_summary = (pd.concat([ym, sales_ym, col_ym, ret_ym], axis=1).reset_index().fillna(0))

    # 7) Bucket Summary per selected year (placeholder, year param later)
    bucket_info = invoices.copy()  # raw data to produce bucket summary

    # 8) Days-to-Pay Summary per customer-year
    def safe_p90(s): arr = s.dropna().to_numpy(); return np.nan if arr.size==0 else np.percentile(arr,90)
    def pct_on_time(s): return (s <= on_time_limit).mean()
    days = (invoices.groupby(['cusid','cusname','area','year'])
            .agg(Avg_Days_to_Pay=('days_to_pay','mean'),
                 Med_Days_to_Pay=('days_to_pay','median'),
                 P90_Days_to_Pay=('days_to_pay', safe_p90),
                 Pct_On_Time=('days_to_pay', pct_on_time),
                 Total_Sales=('invoice_amount','sum'))
            .reset_index())
    days_summary = days.pivot_table(index=['cusid','cusname','area'], columns='year',
                                    values=['Avg_Days_to_Pay','Med_Days_to_Pay','P90_Days_to_Pay','Pct_On_Time'])
    days_summary.columns = [f"{m}_{y}" for m,y in days_summary.columns]; days_summary=days_summary.reset_index()

    # 9) DSO Summary per customer-month
    last_bal = (ledger_txns.groupby(['cusid','cusname','area','year','month'])['running_balance']
                .last().reset_index(name='AR_Balance'))
    cr_sales = (invoices.groupby(['cusid','cusname','area','year','month'])['invoice_amount']
                .sum().reset_index(name='Credit_Sales'))
    dso = last_bal.merge(cr_sales, on=['cusid','cusname','area','year','month'], how='left').fillna(0)
    dso[['AR_Balance','Credit_Sales']] = dso[['AR_Balance','Credit_Sales']].astype(float)
    dso['days_in_period'] = dso.apply(lambda r: calendar.monthrange(r['year'],r['month'])[1], axis=1)
    dso['DSO'] = np.where(dso['Credit_Sales']>0,
                          (dso['AR_Balance']/dso['Credit_Sales'])*dso['days_in_period'],0)
    dso_summary = dso.pivot(index=['cusid','cusname','area'], columns=['year','month'], values='DSO')
    dso_summary.columns = [f"DSO_{yr}_{mn:02d}" for yr,mn in dso_summary.columns]
    dso_summary = dso_summary.reset_index().fillna(0)

    # We will just take the total and show it below the header for how much non-sales transactions happened within the period
    cus_has_no_sales = (ledger_txns.groupby('cusid')['voucher_type_desc'].apply(lambda vs: ~vs.eq('Sales').any()))
    no_sales_cus = cus_has_no_sales[cus_has_no_sales].index.tolist()
    no_sales_rows = ledger_txns[ledger_txns['cusid'].isin(no_sales_cus)]
    no_non_sales_cust = no_sales_rows['cusid'].nunique()
    total_non_sales_balance = no_sales_rows.sort_values(['cusid', 'date']).groupby('cusid')['running_balance'].last().sum()

    led_pairs = (ledger_txns.loc[:, ['cusid','cusname','area','year']].drop_duplicates())
    check = ob.merge(led_pairs,on=['cusid','cusname','area','year'],how='left',indicator=True)
    no_tx = check[check['_merge']=='left_only'][['cusid','cusname','area','year','opening_balance']]

    return {
        'ym_summary': ym_summary,
        'bucket_info': bucket_info,
        'days_summary': days_summary,
        'dso_summary': dso_summary,
        'invoices': invoices,
        'ledger_txns': ledger_txns,
        'closing_balances': tx,
        'total_non_sales_balance': total_non_sales_balance,
        'no_non_sales_cust': no_non_sales_cust,
        'no_tx': no_tx,
        'df': df
    }

def display_payment_timeliness_page(filtered_data_ar: pd.DataFrame):
    """Display summaries in order: YM, Bucket, Days, DSO."""
    data = compute_payment_timeliness_metrics(filtered_data_ar)

    #year month wise summary
    st.subheader("Year-Month Financial Summary")
    st.dataframe(data['ym_summary'])

    #bucket summary
    st.subheader("Bucket Summary")
    filter_type = st.radio("Filter timeline by:", ["Yearly", "Monthly", "Period"], horizontal=True)

    invoices = data['bucket_info']
    ledger = data['ledger_txns']

    if filter_type == "Yearly":
        years = sorted(invoices['year'].unique())
        year = st.selectbox("Select year", years, index=len(years)-1)
        inv_mask = invoices['year'] == year
        led_mask = ledger['year'] == year

    elif filter_type == "Monthly":
        invoices = invoices.assign(ym=invoices['year'].astype(str) + "-" + invoices['month'].astype(str).str.zfill(2))
        ledger = ledger.assign(ym=ledger['year'].astype(str) + "-" + ledger['month'].astype(str).str.zfill(2))
        options = sorted(invoices['ym'].unique())
        sel = st.selectbox("Select year-month", options, index=len(options)-1)
        inv_mask = invoices['ym'] == sel
        led_mask = ledger['ym'] == sel

    else: 
        min_date = invoices['invoice_date'].min().date()
        max_date = invoices['invoice_date'].max().date()
        start, end = st.date_input("Select period", [min_date, max_date])
        start_ts = pd.to_datetime(start)
        end_ts   = pd.to_datetime(end)
        inv_mask = invoices['invoice_date'].between(start_ts, end_ts)
        led_mask = ledger['date'].between(start_ts, end_ts)

    inv_filt = invoices[inv_mask].copy()
    led_filt = ledger[led_mask].copy()

    cust = (inv_filt.groupby(['cusid','cusname','area']).agg(Avg_Days_to_Pay=('days_to_pay','mean'),Total_Sales=('invoice_amount','sum')).reset_index())
    end_bal = (led_filt.sort_values(['cusid','date']).groupby('cusid')['running_balance'].last().reset_index(name='Ending_Balance'))
    cust = cust.merge(end_bal, on='cusid', how='left')
    cust['pay_bucket'] = (pd.cut(cust['Avg_Days_to_Pay'],bins=[0,10,20,30,40,np.inf],labels=['0–10','11–20','21–30','31–40','>45'],right=True).cat.add_categories(['Unknown']).fillna('Unknown'))
    bucket_summary = (cust.groupby('pay_bucket').agg(Avg_Days_to_Pay=('Avg_Days_to_Pay','mean'),Total_Customers=('cusid','nunique'),Total_Sales=('Total_Sales','sum'),Total_Ending_Balance =('Ending_Balance','sum')).reset_index())
    st.dataframe(bucket_summary)

    #opening balances
    if filter_type == "Yearly": 
        st.subheader(f"Opening Balances Without Transactions in {year}")
        no_tx = data['no_tx'][data['no_tx']['year'] == year]
        st.dataframe(no_tx)

    # --- Days-to-Pay Summary ---
    st.subheader("Days-to-Pay Summary")

    view = st.radio("View by:", ["Yearly", "Monthly"], horizontal=True)

    days_summary = data['days_summary']
    invoices    = data['invoices']
    ledger_txns = data['ledger_txns']

    if view == "Yearly":
        metric = st.selectbox("Select metric", ["Avg_Days_to_Pay", "Med_Days_to_Pay", "P90_Days_to_Pay", "Pct_On_Time"])
        prefix = f"{metric}_"
        year_cols = [c for c in days_summary.columns if c.startswith(prefix)]
        df_year = days_summary[["cusid", "cusname", "area"] + year_cols].copy()

        sales = (invoices.groupby(["cusid","year"])["invoice_amount"].sum().unstack(fill_value=0))
        sales.columns = [f"Sales_{y}" for y in sales.columns]
        sales = sales.reset_index()

        coll = (ledger_txns[ledger_txns["value"] < 0].groupby(["cusid","year"])["value"].sum().abs().unstack(fill_value=0))
        coll.columns = [f"Collection_{y}" for y in coll.columns]
        coll = coll.reset_index()

        df_year = (df_year.merge(sales, on="cusid", how="left").merge(coll, on="cusid", how="left"))
        st.dataframe(df_year)

    else:  # Monthly view
        inv = invoices.copy()
        inv["YM"] = (inv["year"].astype(str) + "-" + inv["month"].astype(str).str.zfill(2))
        pivot = (inv.groupby(["cusid","cusname","area","YM"])["days_to_pay"].mean().unstack(fill_value=np.nan).reset_index())
        total_sales = (invoices.groupby("cusid")["invoice_amount"].sum().rename("Total_Sales_All").reset_index())
        total_coll = (ledger_txns[ledger_txns["value"] < 0].groupby("cusid")["value"].sum().abs().rename("Total_Collection_All").reset_index())
        df_month = (pivot.merge(total_sales, on="cusid", how="left").merge(total_coll, on="cusid", how="left"))
        st.dataframe(df_month)

    # --- DSO Summary ---
    st.subheader("DSO Summary")

    view = st.radio("DSO View", ["Average", "Monthly"], horizontal=True)

    dso = data['dso_summary'].copy()
    invoices = data['invoices']
    ledger_txns = data['ledger_txns']
    total_sales = (invoices.groupby("cusid")["invoice_amount"].sum().rename("Total_Sales_All").reset_index())
    total_coll = (ledger_txns[ledger_txns["value"] < 0].groupby("cusid")["value"].sum().abs().rename("Total_Collection_All").reset_index())

    if view == "Average":
        df_avg = (dso.set_index(['cusid','cusname','area']).mean(axis=1).rename("Avg_DSO").reset_index())
        df_avg = (df_avg.merge(total_sales, on="cusid", how="left").merge(total_coll, on="cusid", how="left"))
        st.dataframe(df_avg)

    else:  # Monthly
        df_monthly = dso.copy()
        df_monthly = (df_monthly.merge(total_sales, on="cusid", how="left").merge(total_coll, on="cusid", how="left"))
        st.dataframe(df_monthly)

def display_composite_scoring_page(filtered_data_ar):
    data = compute_payment_timeliness_metrics(filtered_data_ar)
    
    st.subheader("Composite Scoring")

    # 1) Choose Yearly vs Monthly
    comp_view = st.radio("Score by:", ["All","Yearly", "Monthly"], horizontal=True)

    # grab precomputed tables
    days_summary = data['days_summary']
    dso_summary  = data['dso_summary']
    invoices     = data['invoices']
    ledger_txns  = data['ledger_txns']

    if comp_view == "All":
    # — no time filter, use entire dataset —
        # 1) Avg Days-to-Pay over all time
        df_days = (invoices.groupby(['cusid','cusname','area'])['days_to_pay'].mean().reset_index(name='Days_to_Pay'))

        # 2) Avg DSO over all months
        dso_cols = [c for c in dso_summary.columns if c.startswith("DSO_")]
        df_dso = dso_summary[['cusid'] + dso_cols].copy()
        df_dso['DSO'] = df_dso[dso_cols].mean(axis=1)
        df_dso = df_dso[['cusid','DSO']]

        df_sales = (invoices.groupby(['cusid','cusname','area'])['invoice_amount'].sum().reset_index(name='Total_Sales'))
        df_coll = (ledger_txns[ledger_txns['value'] < 0].groupby('cusid')['value'].sum().abs().reset_index(name='Total_Collection'))
        end_bal = (ledger_txns.sort_values(['cusid','date']).groupby('cusid')['running_balance'].last().reset_index(name='Ending_Balance'))
        df = (df_days.merge(df_dso, on='cusid', how='left').merge(df_sales, on=['cusid','cusname','area'], how='left').merge(df_coll, on='cusid', how='left').merge(end_bal, on='cusid', how='left').fillna(0))

    elif comp_view == "Yearly":
        # — select year and days metric —
        years = sorted(invoices['year'].unique())
        sel_year = st.selectbox("Select Year", years, index=len(years)-1)
        days_metric = st.selectbox("Days metric", ["Avg_Days_to_Pay", "Med_Days_to_Pay", "P90_Days_to_Pay", "Pct_On_Time"])
        df_days = days_summary[['cusid','cusname','area', f"{days_metric}_{sel_year}"]].rename(columns={f"{days_metric}_{sel_year}": "Days_to_Pay"})

        dso_cols = [c for c in dso_summary.columns if c.startswith(f"DSO_{sel_year}_")]
        df_dso = dso_summary[['cusid'] + dso_cols].copy()
        df_dso['DSO'] = df_dso[dso_cols].mean(axis=1)
        df_dso = df_dso[['cusid','DSO']]

        df_sales = (invoices[invoices['year']==sel_year].groupby(['cusid','cusname','area'])['invoice_amount'].sum().reset_index(name='Total_Sales'))
        df_coll = (ledger_txns[(ledger_txns['year']==sel_year)&(ledger_txns['value']<0)].groupby('cusid')['value'].sum().abs().reset_index(name='Total_Collection'))
        end_bal = (ledger_txns[ledger_txns['year']==sel_year].sort_values(['cusid', 'date']).groupby('cusid')['running_balance'].last().reset_index(name='Ending_Balance'))
        df = (df_days.merge(df_dso, on='cusid', how='left').merge(df_sales,on=['cusid','cusname','area'], how='left').merge(df_coll, on='cusid', how='left').merge(end_bal, on='cusid', how='left').fillna(0))

    else:  # Monthly
        # — select year-month —
        inv = invoices.assign(YM=invoices['year'].astype(str) + "-" + invoices['month'].astype(str).str.zfill(2))
        yms = sorted(inv['YM'].unique())
        sel_ym = st.selectbox("Select Year-Month", yms, index=len(yms)-1)
        year, month = map(int, sel_ym.split("-"))

        df_days = (inv[inv['YM']==sel_ym].groupby(['cusid','cusname','area'])['days_to_pay'].mean().reset_index(name='Days_to_Pay'))
        col = f"DSO_{year}_{month:02d}"
        df_dso = (dso_summary[['cusid','cusname','area',col]].rename(columns={col: 'DSO'}))
        df_sales = (inv[inv['YM']==sel_ym].groupby(['cusid','cusname','area'])['invoice_amount'].sum().reset_index(name='Total_Sales'))
        led = ledger_txns.assign(YM=ledger_txns['year'].astype(str) + "-" + ledger_txns['month'].astype(str).str.zfill(2))
        df_coll = (led[(led['YM']==sel_ym)&(led['value']<0)].groupby('cusid')['value'].sum().abs().reset_index(name='Total_Collection'))
        end_bal = (led[led['YM']==sel_ym].sort_values(['cusid', 'date']).groupby('cusid')['running_balance'].last().reset_index(name='Ending_Balance'))
        df = (df_days.merge(df_dso, on=['cusid','cusname','area'], how='outer').merge(df_sales,on=['cusid','cusname','area'], how='left').merge(df_coll, on='cusid', how='left').merge(end_bal, on='cusid', how='left').fillna(0))
    # need default and set entire period as default
    mins = df[['Days_to_Pay','DSO','Total_Sales','Total_Collection']].min()
    maxs = df[['Days_to_Pay','DSO','Total_Sales','Total_Collection']].max()
    denom = maxs - mins
    # avoid zero-division
    denom = denom.replace(0, 1)

    df['norm_days'] = (maxs['Days_to_Pay'] - df['Days_to_Pay']) / denom['Days_to_Pay']
    df['norm_dso']  = (maxs['DSO']        - df['DSO'])        / denom['DSO']
    df['norm_sales']= (df['Total_Sales']  - mins['Total_Sales']) / denom['Total_Sales']
    df['norm_coll'] = (df['Total_Collection'] - mins['Total_Collection']) / denom['Total_Collection']

    # 3) Weight sliders (will auto-normalize)
    w1 = st.slider("Weight: Days-to-Pay",   0.0, 1.0, 1.0, key="w_days")
    w2 = st.slider("Weight: DSO",            0.0, 1.0, 0.1, key="w_dso")
    w3 = st.slider("Weight: Total Sales",    0.0, 1.0, 0.7, key="w_sales")
    w4 = st.slider("Weight: Total Collection",0.0, 1.0, 0.9, key="w_coll")
    total_w = w1 + w2 + w3 + w4
    if total_w == 0:
        st.warning("Please assign at least one non-zero weight.")
    else:
        # normalize weights
        w1, w2, w3, w4 = w1/total_w, w2/total_w, w3/total_w, w4/total_w

        # 4) Composite score
        df['Composite_Score'] = (
            w1*df['norm_days'] +
            w2*df['norm_dso']  +
            w3*df['norm_sales']+
            w4*df['norm_coll']
        )

        # 5) Show results
        st.markdown("### Composite Scores")
        st.dataframe(
            df.sort_values('Composite_Score', ascending=False)
            .loc[:, ['cusid','cusname','area',
                       'Days_to_Pay','DSO','Total_Sales','Total_Collection',
                       'Composite_Score']],
            use_container_width=True
        )
    
    hist_metric = st.selectbox("Histogram Metric", ["Days_to_Pay", "DSO", "Total_Sales", "Total_Collection", "Ending_Balance"])

    fig = px.histogram(df,x=hist_metric,nbins=10,title=f"Distribution of {hist_metric}")
    fig.update_layout(xaxis_title=hist_metric,yaxis_title="Count",bargap=0.1)
    st.plotly_chart(fig)

    segment_map = {
    "0.0-0.1": "Critical Watch",
    "0.1-0.2": "High Risk",
    "0.2-0.3": "Warning Zone",
    "0.3-0.4": "Needs Attention",
    "0.4-0.5": "Developing",
    "0.5-0.6": "Stable",
    "0.6-0.7": "Solid Performer",
    "0.7-0.8": "Valued Partner",
    "0.8-0.9": "Top Tier",
    "0.9-1.0": "Elite Champion"
    }

    # 3) Build dynamic bins
    edges = make_dynamic_bins(df['Composite_Score'])
    labels = [f"{edges[i]:.3f}-{edges[i+1]:.3f}" for i in range(len(edges)-1)]
    df['comp_bin'] = pd.cut(
        df['Composite_Score'],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
        ordered=False
    )

    orig_intervals = [
    (float(k.split('-')[0]), float(k.split('-')[1]), v)
    for k, v in segment_map.items()
    ]

    # map each dynamic label to its base
    bin_to_base = {}
    for lbl in labels:
        low = float(lbl.split('-')[0])
        for o_low, o_high, base in orig_intervals:
            if (o_low <= low < o_high) or (low == 1.0 and o_high == 1.0):
                bin_to_base[lbl] = base
                break

    # group and sort
    grouped = defaultdict(list)
    for lbl, base in bin_to_base.items():
        grouped[base].append(lbl)

    lbl_to_segment = {}
    for base, lbls in grouped.items():
        sorted_lbls = sorted(lbls, key=lambda x: float(x.split('-')[0]))
        for idx, lbl in enumerate(sorted_lbls):
            lbl_to_segment[lbl] = f"{base}-{idx}"

    df['segment_name'] = df['comp_bin'].astype(str).map(lbl_to_segment)
    print(df.columns)
    # 4) Summary table per dynamic bin
    summary_table = (
        df
        .groupby('comp_bin')
        .agg(
            Count=('cusid','nunique'),
            Avg_Days_To_Pay=('Days_to_Pay','mean'),
            Total_Sales=('Total_Sales','sum'),
            Total_Collection=('Total_Collection','sum'),
            Total_Ending_Balance=('Ending_Balance','sum')
        )
        .reset_index()
    )

    st.dataframe(summary_table,use_container_width=True)

    st.subheader("Full composite score list")
    st.dataframe(df,use_container_width=True)


def make_dynamic_bins(scores: pd.Series,seed_width: float = 0.1,top_n: int = 3,split_pcts: list = [0.25, 0.5, 0.75, 0.99]) -> list:
    seed_edges = list(np.arange(0, 1 + seed_width, seed_width))
    seed_bins = pd.cut(
        scores,
        bins=seed_edges,
        right=False,
        include_lowest=True
    )
    top_seeds = seed_bins.value_counts().nlargest(top_n).index
    all_edges = set(seed_edges)
    for interval in top_seeds:
        low, high = interval.left, interval.right
        subset = scores[(scores >= low) & (scores < high)]
        if len(subset) >= len(split_pcts):
            qvals = subset.quantile(split_pcts).tolist()
            for q in qvals:
                if low < q < high:
                    all_edges.add(q)
    return sorted(all_edges)

@timed
def calculate_summary_statistics(filtered_data_c, filtered_data_s, filtered_data_r):
    """
    Calculate summary of margin data for the filtered data.

    Args:
    - filtered_data_c: Filtered collection data.
    - filtered_data_s: Filtered sales data.
    - filtered_data_r: Filtered returns data.

    Returns:
    - Dictionary containing the summary statistics.
    """
    units_sold     = float(filtered_data_s["quantity"].sum()) if len(filtered_data_s) else 0.0
    units_returned = float(filtered_data_r["returnqty"].sum()) if len(filtered_data_r) else 0.0
    net_units      = units_sold - units_returned
    return {
        "Net Sales": filtered_data_s['final_sales'].sum().round(2) - filtered_data_r['treturnamt'].sum().round(2),
        "Total Returns": filtered_data_r['treturnamt'].sum().round(2),
        "Total Discounts": filtered_data_s['proddiscount'].sum().round(2),
        "Net Margin": filtered_data_s['gross_margin'].sum().round(2) - filtered_data_r['treturnamt'].sum().round(2),
        "Net Collection": filtered_data_c['value'].sum().round(2),
        "Net Units Sold": round(net_units, 2)
    }

@timed
def display_summary_statistics(stats):
    """
    Display summary statistics in the Streamlit app across 5 columns.

    Args:
    - stats: Dictionary containing the summary statistics.
    """
    st.sidebar.title("Overall Margin Analysis")

    # Create 5 columns
    col1, col2, col3, col4, col5 = st.columns(5)

    # Split stats into 5 chunks
    stats_items = list(stats.items())
    chunk_size = (len(stats_items) + 4) // 5  # balanced split across 5

    chunks = [
        stats_items[0:chunk_size],
        stats_items[chunk_size:2*chunk_size],
        stats_items[2*chunk_size:3*chunk_size],
        stats_items[3*chunk_size:4*chunk_size],
        stats_items[4*chunk_size:]
    ]

    for col, chunk in zip([col1, col2, col3, col4, col5], chunks):
        with col:
            for stat_name, value in chunk:
                st.markdown(f"**{stat_name}:** {value:,.2f}")

@timed
def display_cross_relation_pivot(filtered_data_c, filtered_data_s, filtered_data_r, current_page):
    st.subheader("🔁 Cross Relation Analysis")

    column_options = {
        'Salesman': ['spid', 'spname'],
        'Customer': ['cusid', 'cusname'],
        'Product': ['itemcode', 'itemname'],
        'Product Group': ['itemgroup'],
        'Area': ['area']
    }

    full_metric_options = [
        "Net Sales", 
        "Total Returns", 
        "Total Discounts",
        "Net Margin",
        "Net Units Sold",
        "Collection"  # Collection only for customer/salesman/area
    ]

    col1, col2 = st.columns(2)
    with col1:
        first_selection = st.selectbox(
            "Select First Column List", 
            list(column_options.keys()), 
            index=0, 
            key="cross_first_col"
        )
    with col2:
        second_selection = st.selectbox(
            "Select Second Column List", 
            list(column_options.keys()), 
            index=1, 
            key="cross_second_col"
        )

    # Only allow Collection if BOTH selected columns are NOT Product or Product Group
    forbidden = ["Product", "Product Group"]
    if first_selection in forbidden or second_selection in forbidden:
        metric_options = [
            "Net Sales", 
            "Total Returns", 
            "Total Discounts",
            "Net Margin",
            "Net Units Sold"
        ]
    else:
        metric_options = full_metric_options

    selected_metric = st.selectbox("Select Metric", metric_options, key="cross_metric")

    first_column_list = column_options[first_selection]
    second_column_list = column_options[second_selection]

    try:
        pivot_args = {
            "metric": selected_metric,
            "index": first_column_list,
            "column": second_column_list
        }

        pivot_table = common.net_pivot(
            filtered_data_s, 
            filtered_data_r, 
            pivot_args, 
            current_page=current_page, 
            data3=filtered_data_c
        )

        st.markdown(f"**{selected_metric} by {first_selection} vs {second_selection}**")
        st.write(pivot_table)

    except Exception as e:
        st.error(f"Error generating cross relation pivot: {e}")

@timed
def display_entity_metric_pivot(filtered_data_c, filtered_data_s, filtered_data_r, current_page):
    st.subheader("📊 Pivot Table Analysis")

    entity_options = {
        "Salesman": ["spid", "spname"],
        "Customer": ["cusid", "cusname"],
        "Product": ["itemcode", "itemname"],
        "Product Group": ["itemgroup"],
        "Area": ["area"]
    }

    full_metric_options = [
        "Net Sales", 
        "Total Returns", 
        "Total Discounts",
        "Net Margin",
        "Net Units Sold",
        "Collection"  # Collection allowed only if entity is customer/salesman/area
    ]

    col1, col2 = st.columns(2)
    with col1:
        selected_entity = st.selectbox(
            "Select Entity", 
            list(entity_options.keys()), 
            key="entity_metric_entity"
        )
    
    # Filter metric options based on entity
    if selected_entity in ["Product", "Product Group"]:
        metric_options = [
            "Net Sales", 
            "Total Returns", 
            "Total Discounts",
            "Net Margin",
            "Net Units Sold"
        ]
    else:
        metric_options = full_metric_options

    with col2:
        selected_metric = st.selectbox(
            "Select Metric", 
            metric_options, 
            key="entity_metric_metric"
        )

    index_columns = entity_options[selected_entity]

    pivot_args = {
        "metric": selected_metric,
        "index": index_columns,
        "column": ["year", "month"]
    }

    try:
        pivot_table = common.net_pivot(
            filtered_data_s, 
            filtered_data_r, 
            pivot_args, 
            current_page=current_page,
            data3=filtered_data_c
        )

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
def plot_collection(filtered_data_c, current_page):
    # Group by year and month
    collection_data = filtered_data_c.groupby(["year", "month"])["value"].sum().reset_index()

    # --- FIX: Convert month names to month numbers safely ---
    if collection_data["month"].dtype == object:
        month_name_to_num = {month: idx for idx, month in enumerate(calendar.month_name) if month}
        collection_data["month_num"] = collection_data["month"].map(month_name_to_num)
    else:
        collection_data["month_num"] = collection_data["month"]

    # Build x_label
    collection_data["month_num_str"] = collection_data["month_num"].astype(int).astype(str).str.zfill(2)
    collection_data["x_label"] = collection_data["year"].astype(str) + "-" + collection_data["month_num_str"]

    # Sort by year, month
    collection_data = collection_data.sort_values(["year", "month_num"])

    # Plot
    fig = px.bar(
        collection_data,
        x="x_label",
        y="value",
        title="Total Collections Over Time",
        labels={"x_label": "Year-Month", "value": "Collection Amount"}
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        template='plotly_white',
        xaxis_title="Year-Month",
        yaxis_title="Collection Amount"
    )

    st.plotly_chart(fig, use_container_width=True)

@timed
def plot_yoy_monthly_comparison(filtered_data_c,filtered_data_s,filtered_data_r,code_col,selected_codes,metric,selected_years,selected_month_names):

    df_collections = filtered_data_c.copy()
    df_sales = filtered_data_s.copy()
    df_returns = filtered_data_r.copy()

    # Filter entity
    if selected_codes:
        df_collections = df_collections[df_collections[code_col].isin(selected_codes)]
        df_sales = df_sales[df_sales[code_col].isin(selected_codes)]
        df_returns = df_returns[df_returns[code_col].isin(selected_codes)]

    # Map months
    month_name_to_num = {calendar.month_abbr[i]: i for i in range(1, 13)}
    selected_months = [month_name_to_num[m] for m in selected_month_names]

    # Filter years and months
    df_collections = df_collections[df_collections["year"].isin(selected_years) & df_collections["month"].isin(selected_months)]
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
    elif metric == "Collection":
        df = df_collections.groupby(["year", "month"])["value"].sum().reset_index(name="value")
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
def plot_yoy_daily_comparison(filtered_data_c,filtered_data_s, filtered_data_r, code_col, selected_codes, metric, selected_years, start_date, end_date):
    df_collections = filtered_data_c.copy()
    df_sales = filtered_data_s.copy()
    df_returns = filtered_data_r.copy()

    # Apply filters for selected entity
    if selected_codes:
        df_collections = df_collections[df_collections[code_col].isin(selected_codes)]
        df_sales = df_sales[df_sales[code_col].isin(selected_codes)]
        df_returns = df_returns[df_returns[code_col].isin(selected_codes)]

    # Convert date
    df_collections["date"] = pd.to_datetime(df_collections["date"])
    df_sales["date"] = pd.to_datetime(df_sales["date"])
    df_returns["date"] = pd.to_datetime(df_returns["date"])

    # Filter only selected years
    df_collections = df_collections[df_collections["year"].isin(selected_years)]
    df_sales = df_sales[df_sales["year"].isin(selected_years)]
    df_returns = df_returns[df_returns["year"].isin(selected_years)]

    # Extract day and month to create a day-month key
    df_collections["day_month"] = df_collections["date"].dt.strftime("%m-%d")
    df_sales["day_month"] = df_sales["date"].dt.strftime("%m-%d")
    df_returns["day_month"] = df_returns["date"].dt.strftime("%m-%d")

    # Determine selected day-month keys based on user-selected range
    selected_daymonths = pd.date_range(start=start_date, end=end_date).strftime("%m-%d").tolist()
    df_collections = df_collections[df_collections["day_month"].isin(selected_daymonths)]
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
    elif metric == "Collection":
        df = df_collections.groupby(["year", "day_month"])["value"].sum().reset_index(name="value")
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
def plot_yoy_dow_comparison(filtered_data_c, filtered_data_s, filtered_data_r, code_col, selected_codes, metric, selected_years, average_or_total):
    df_collections = filtered_data_c.copy()
    df_sales = filtered_data_s.copy()
    df_returns = filtered_data_r.copy()

    # Apply filters for selected entity
    if selected_codes:
        df_collections = df_collections[df_collections[code_col].isin(selected_codes)]
        df_sales = df_sales[df_sales[code_col].isin(selected_codes)]
        df_returns = df_returns[df_returns[code_col].isin(selected_codes)]

    # Convert date
    df_collections["date"] = pd.to_datetime(df_collections["date"])
    df_sales["date"] = pd.to_datetime(df_sales["date"])
    df_returns["date"] = pd.to_datetime(df_returns["date"])

    # Filter only selected years
    df_collections = df_collections[df_collections["year"].isin(selected_years)]
    df_sales = df_sales[df_sales["year"].isin(selected_years)]
    df_returns = df_returns[df_returns["year"].isin(selected_years)]

    # Add weekday column
    df_collections["weekday"] = df_collections["date"].dt.day_name()
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
    elif metric == "Collection":
        df = df_collections.groupby(["year", "weekday"])["value"].sum().reset_index(name="value")
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
def plot_yoy_dom_comparison(filtered_data_c, filtered_data_s, filtered_data_r, code_col, selected_codes, metric, selected_years, selected_month_names, average_or_total):
    # Map month names to numbers
    month_name_to_num = {calendar.month_abbr[i]: i for i in range(1, 13)}
    selected_months = [month_name_to_num[m] for m in selected_month_names]

    df_collections = filtered_data_c.copy()
    df_sales = filtered_data_s.copy()
    df_returns = filtered_data_r.copy()

    # Filter selected entity, year, month
    if selected_codes:
        df_collections = df_collections[df_collections[code_col].isin(selected_codes)]
        df_sales = df_sales[df_sales[code_col].isin(selected_codes)]
        df_returns = df_returns[df_returns[code_col].isin(selected_codes)]

    df_collections = df_collections[df_collections["year"].isin(selected_years) & df_collections["month"].isin(selected_months)]
    df_sales = df_sales[df_sales["year"].isin(selected_years) & df_sales["month"].isin(selected_months)]
    df_returns = df_returns[df_returns["year"].isin(selected_years) & df_returns["month"].isin(selected_months)]

    # Convert to datetime
    df_collections["date"] = pd.to_datetime(df_collections["date"])
    df_sales["date"] = pd.to_datetime(df_sales["date"])
    df_returns["date"] = pd.to_datetime(df_returns["date"])

    df_collections["day"] = df_collections["date"].dt.day
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
    elif metric == "Collection":
        df = df_collections.groupby(["year", "day"])["value"].sum().reset_index(name="value")
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
def plot_month_vs_month_comparison(filtered_data_c,filtered_data_s,filtered_data_r,code_col,name_col,selected_codes,metric,selected_months):
    df_collections = filtered_data_c.copy()
    df_sales = filtered_data_s.copy()
    df_returns = filtered_data_r.copy()

    # Create a month_label column
    df_collections["month_label"] = df_collections["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_collections["year"].astype(str)
    df_sales["month_label"] = df_sales["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_sales["year"].astype(str)
    df_returns["month_label"] = df_returns["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_returns["year"].astype(str)
    # Filter for selected months
    df_collections = df_collections[df_collections["month_label"].isin(selected_months)]
    df_sales = df_sales[df_sales["month_label"].isin(selected_months)]
    df_returns = df_returns[df_returns["month_label"].isin(selected_months)]
    # Filter for selected entities
    if selected_codes:
        df_collections = df_collections[df_collections[code_col].isin(selected_codes)]
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
    elif metric == "Collection":
        df = df_collections.groupby(["month_label", code_col])["value"].sum().reset_index(name="value")
    else:
        st.error("Unsupported metric.")
        return

    # Join name if applicable
    if name_col:
        lookup = filtered_data_s[[code_col, name_col]].drop_duplicates()
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
def plot_month_vs_month_dow_comparison(filtered_data_c,filtered_data_s,filtered_data_r,code_col,name_col,selected_codes,metric,selected_months,aggregation_type):

    df_collections = filtered_data_c.copy()
    df_sales = filtered_data_s.copy()
    df_returns = filtered_data_r.copy()

    # Custom weekday order for Bangladesh
    dow_order = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    df_collections["DOW"] = pd.Categorical(df_collections["DOW"], categories=dow_order, ordered=True)
    df_sales["DOW"] = pd.Categorical(df_sales["DOW"], categories=dow_order, ordered=True)
    df_returns["DOW"] = pd.Categorical(df_returns["DOW"], categories=dow_order, ordered=True)

    # Create a month_label column
    df_collections["month_label"] = df_collections["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_collections["year"].astype(str)
    df_sales["month_label"] = df_sales["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_sales["year"].astype(str)
    df_returns["month_label"] = df_returns["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_returns["year"].astype(str)

    # Filter for selected months
    df_collections = df_collections[df_collections["month_label"].isin(selected_months)]
    df_sales = df_sales[df_sales["month_label"].isin(selected_months)]
    df_returns = df_returns[df_returns["month_label"].isin(selected_months)]

    # Filter for selected entities
    if selected_codes:
        df_collections = df_collections[df_collections[code_col].isin(selected_codes)]
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
    elif metric == "Collection":
        df = df_collections.groupby(["month_label", "DOW", code_col])["value"].sum().reset_index(name="value")
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
        lookup = filtered_data_s[[code_col, name_col]].drop_duplicates()
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
def plot_month_vs_month_dom_comparison(filtered_data_c,filtered_data_s,filtered_data_r,code_col,name_col,selected_codes,metric,selected_months,aggregation_type,selected_days=None):

    df_collections = filtered_data_c.copy()
    df_sales = filtered_data_s.copy()
    df_returns = filtered_data_r.copy()

    # Create a month_label column
    df_collections["month_label"] = df_collections["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_collections["year"].astype(str)
    df_sales["month_label"] = df_sales["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_sales["year"].astype(str)
    df_returns["month_label"] = df_returns["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + df_returns["year"].astype(str)

    # Extract day
    # df_sales["day"] = pd.to_datetime(df_sales["date"]).dt.day
    # df_returns["day"] = pd.to_datetime(df_returns["date"]).dt.day

    # Filter for selected months
    df_collections = df_collections[df_collections["month_label"].isin(selected_months)]
    df_sales = df_sales[df_sales["month_label"].isin(selected_months)]
    df_returns = df_returns[df_returns["month_label"].isin(selected_months)]

    # Filter for selected days if provided
    if selected_days:
        df_collections = df_collections[df_collections["DOM"].isin(selected_days)]
        df_sales = df_sales[df_sales["DOM"].isin(selected_days)]
        df_returns = df_returns[df_returns["DOM"].isin(selected_days)]

    # Filter for selected entities
    if selected_codes:
        df_collections = df_collections[df_collections[code_col].isin(selected_codes)]
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
    elif metric == "Collection":
        df = df_collections.groupby(group_cols)["value"].sum().reset_index(name="value")
    else:
        st.error("Unsupported metric.")
        return

    # Join name if applicable
    if name_col:
        lookup = filtered_data_s[[code_col, name_col]].drop_duplicates()
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
def plot_metric_comparison_monthly(filtered_data_c, filtered_data_s, filtered_data_r, code_col, selected_codes, metric_x, metric_y, selected_years, selected_month_names):
    df_c = filtered_data_c.copy()
    df_s = filtered_data_s.copy()
    df_r = filtered_data_r.copy()

    if selected_codes:
        df_c = df_c[df_c[code_col].isin(selected_codes)]
        df_s = df_s[df_s[code_col].isin(selected_codes)]
        df_r = df_r[df_r[code_col].isin(selected_codes)]

    month_name_to_num = {calendar.month_abbr[i]: i for i in range(1, 13)}
    selected_months = [month_name_to_num[m] for m in selected_month_names]

    df_c = df_c[df_c["year"].isin(selected_years) & df_c["month"].isin(selected_months)]
    df_s = df_s[df_s["year"].isin(selected_years) & df_s["month"].isin(selected_months)]
    df_r = df_r[df_r["year"].isin(selected_years) & df_r["month"].isin(selected_months)]

    def compute_metric(df_c, df_s, df_r, metric):
        if metric == "Net Sales":
            s = df_s.groupby(["year", "month"])["final_sales"].sum()
            r = df_r.groupby(["year", "month"])["treturnamt"].sum()
            return s.subtract(r, fill_value=0).reset_index(name="value")
        elif metric == "Total Returns":
            return df_r.groupby(["year", "month"])["treturnamt"].sum().reset_index(name="value")
        elif metric == "Total Discounts":
            return df_s.groupby(["year", "month"])["proddiscount"].sum().reset_index(name="value")
        elif metric == "Net Margin":
            s = df_s.groupby(["year", "month"])["gross_margin"].sum()
            r = df_r.groupby(["year", "month"])["treturnamt"].sum()
            return s.subtract(r, fill_value=0).reset_index(name="value")
        elif metric == "Collection":
            return df_c.groupby(["year", "month"])["value"].sum().reset_index(name="value")
        else:
            return pd.DataFrame(columns=["year", "month", "value"])

    df_x = compute_metric(df_c, df_s, df_r, metric_x)
    df_x["Metric"] = metric_x
    df_y = compute_metric(df_c, df_s, df_r, metric_y)
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
def plot_distribution_analysis(filtered_data_s, filtered_data_r, filtered_data_c, metric, group_by, value_min=None, value_max=None, nbins=100):
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

    # Always copy safely
    df_s = filtered_data_s.copy()
    df_r = filtered_data_r.copy()
    df_c = filtered_data_c.copy()

    try:
        # Step 1: Aggregate based on metric
        if metric == "Net Sales":
            sales_sum = df_s.groupby(entity_col)["final_sales"].sum()
            return_sum = df_r.groupby(entity_col)["treturnamt"].sum()
            agg_series = sales_sum.subtract(return_sum, fill_value=0)

        elif metric == "Net Margin":
            sales_sum = df_s.groupby(entity_col)["gross_margin"].sum()
            return_sum = df_r.groupby(entity_col)["treturnamt"].sum()
            agg_series = sales_sum.subtract(return_sum, fill_value=0)

        elif metric == "Total Returns":
            agg_series = df_r.groupby(entity_col)["treturnamt"].sum()

        elif metric == "Total Discounts":
            agg_series = df_s.groupby(entity_col)["proddiscount"].sum()

        elif metric == "Collection":
            if entity_col not in df_c.columns:
                st.error(f"Group {group_by} not available in Collection data.")
                return
            agg_series = df_c.groupby(entity_col)["value"].sum()

        else:
            st.error("Unsupported metric selected.")
            return

        # Step 2: Prepare dataframe
        agg_df = agg_series.reset_index(name="value")

        # Step 3: Apply min/max filters
        if value_min is not None:
            agg_df = agg_df[agg_df["value"] >= value_min]
        if value_max is not None:
            agg_df = agg_df[agg_df["value"] <= value_max]

        # Step 4: Plot distribution
        fig = px.histogram(
            agg_df,
            x="value",
            nbins=nbins,
            title=f"Distribution of {metric} by {group_by}",
            labels={"value": metric}
        )

        st.plotly_chart(fig, use_container_width=True)

        # Step 5: Show Binned Frequency Table
        st.markdown("### 📋 Binned Frequency Table")
        bin_table = pd.cut(agg_df["value"], bins=nbins).value_counts().sort_index().reset_index()
        bin_table.columns = ["Range", "Count"]
        st.dataframe(bin_table, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating distribution: {e}")
        st.warning("Check if the selected combination of metric and group by column is valid.")

@timed
def generate_descriptive_statistics(filtered_data, filtered_data_r, filtered_data_c, group_by):
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
    df_c = filtered_data_c.copy()
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

        # Total Discounts
        discount_series = df.groupby(entity_col)["proddiscount"].sum()
        add_stat_row("Total Discounts", discount_series)

        # Collection
        if entity_col in df_c.columns:
            collection_series = df_c.groupby(entity_col)["value"].sum()
            add_stat_row("Collection", collection_series)

    except Exception as e:
        return pd.DataFrame({"Error": [f"Failed to compute stats: {str(e)}"]})

    df_stats = pd.DataFrame(stats)
    return df_stats.round(2).reset_index().rename(columns={"index": "Metric"})


    # The way this works is: first we find the invoices (sales) and then we find the balance before this 
    # invoice hit our ledger, then we check every payments made after to find out when the balance hits this 
    # prior balance mark. This days to payment is calculated for each invoice for each customer.