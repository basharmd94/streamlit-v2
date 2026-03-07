import pandas as pd
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from modules.data_process_files import common
from datetime import datetime
import os
import json
from datetime import date as _date
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st

def generate_cohort(purchase_data,year_ago,inventory_data,sales_df,cohort_df):
    cohort_df = cohort_latest(cohort_df)
    purchase_order = time_filtered_data_requisition(purchase_data,year_ago)
    
    purchase_order = purchase_order[['itemcode','itemname','shipmentname','quantity','combinedate']]
    purchase_pivot = purchase_order.pivot_table(index=['itemcode'], columns='shipmentname', values='quantity', aggfunc='sum').reset_index()
    
    # -------------------------
    # Inventory: compute on-hand from stock_movement ledger
    # -------------------------
    sm = inventory_data.copy()

    # required columns check (fail loud)
    required_cols = {"itemcode", "itemname", "itemgroup", "date", "stockqty"}
    missing = required_cols - set(sm.columns)
    if missing:
        raise KeyError(f"stock_movement is missing columns: {missing}")

    # keep only relevant groups (add/remove as you wish)
    sm = sm[sm["itemgroup"].isin([
        "Finished Goods Packaging",
        "RAW Material Packaging",
        "Import Item",
        "Furniture Fittings",
        "Hardware",
        "Industrial & Household",
        "Sanitary",
    ])].copy()

    # parse dates
    sm["date"] = pd.to_datetime(sm["date"], errors="coerce")
    sm = sm[sm["date"].notna()].copy()

    # IMPORTANT: cutoff date to avoid "future" / messy rows
    # Best cutoff is "today" OR max available date in movement.
    cutoff_date = min(pd.Timestamp.today().normalize(), sm["date"].max().normalize())
    sm = sm[sm["date"] <= cutoff_date].copy()

    # ensure numeric stockqty
    non_numeric_cols = ["itemcode", "itemname", "itemgroup"]
    sm = common.numerise_columns(sm, non_numeric_cols)

    # ON-HAND = net movement sum (if your stockqty already stores movement deltas)
    inventory_df = (
        sm.groupby("itemcode", as_index=False)["stockqty"]
        .sum()
    )

    # keep one name/group per itemcode
    caitem = (
        sm[["itemcode", "itemname", "itemgroup"]]
        .drop_duplicates(subset="itemcode", keep="first")
    )

    inventory_df = pd.merge(inventory_df, caitem, on="itemcode", how="left")
    
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

# ============================================================
# Shipment Profitability Engine (Stock Movement Driven)
# ============================================================

# ------------------------------------------------------------
# Reference: Stock docnum prefix meaning (keep in code)
# ------------------------------------------------------------
# IS-- / ISS- : Special adjustments (stock removed)
# PRE-        : Purchase returns
# DSR-        : Damaged Returns (held as stock)
# IGRN        : Import purchases (shipments)
# RECT        : Returns
# REC-        : Special adjustments (qty received)
# SRE-        : General store / O&A items (usually not in shipments)
# IPTO        : Internal transfers
# MO--        : Manufacturing (RM issue/receive)
# GRN-        : Local purchase
# RECA        : Sales returns (also)
# TO--        : Transfer order
# SR--        : Return
# DO--        : Sales


def _today() -> pd.Timestamp:
    return pd.Timestamp(_date.today()).floor("D")

def _norm_code(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def _prep_purchase_shipment(purchase_df: pd.DataFrame, shipmentname: str) -> pd.DataFrame:
    p = purchase_df.copy()
    p["shipmentname"] = p["shipmentname"].astype(str).str.strip()
    p = p[p["shipmentname"] == str(shipmentname)].copy()
    if p.empty:
        return p

    p["itemcode"] = p["itemcode"].apply(_norm_code)
    p["combinedate"] = pd.to_datetime(p["combinedate"], errors="coerce").dt.floor("D")
    p["quantity"] = pd.to_numeric(p["quantity"], errors="coerce").fillna(0.0)
    p["cost"] = pd.to_numeric(p["cost"], errors="coerce").fillna(0.0)

    # unit_cost is purchase.cost (confirmed)
    grp = (
        p.groupby(["shipmentname", "itemcode", "itemname", "combinedate"], as_index=False)
        .agg(initial_qty=("quantity", "sum"), unit_cost=("cost", "mean"))
    )
    grp["batch_id"] = (
        grp["shipmentname"].astype(str)
        + " | "
        + grp["itemcode"].astype(str)
        + " | "
        + grp["combinedate"].dt.strftime("%Y-%m-%d")
    )
    grp["batch_cost"] = grp["initial_qty"] * grp["unit_cost"]
    return grp.sort_values(["itemcode", "combinedate"]).reset_index(drop=True)

def _prep_stock_movement(stock_mv_df: pd.DataFrame, zids: List[str]) -> pd.DataFrame:
    """
    Stock movement ledger prep.
    We DO NOT filter by project (confirmed).
    We DO allow multiple zids (100001 + 100009).
    """
    if stock_mv_df is None or stock_mv_df.empty:
        return pd.DataFrame()

    s = stock_mv_df.copy()

    s["zid"] = s["zid"].astype(str).str.strip()
    zids_norm = [str(z).strip() for z in (zids or [])]
    if zids_norm:
        s = s[s["zid"].isin(zids_norm)].copy()

    s["itemcode"] = s["itemcode"].apply(_norm_code)
    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.floor("D")
    s = s[s["date"].notna()].copy()

    s["stockqty"] = pd.to_numeric(s["stockqty"], errors="coerce").fillna(0.0)
    s["stockvalue"] = pd.to_numeric(s["stockvalue"], errors="coerce").fillna(0.0)

    s["docnum"] = s.get("docnum", "").astype(str).fillna("").str.strip()
    s["prefix"] = s["docnum"].str.slice(0, 4)

    s["warehouse"] = s.get("warehouse", "").astype(str).fillna("").str.strip()

    return s

def _onhand_series(stock_mv: pd.DataFrame) -> pd.DataFrame:
    """
    Returns cumulative on-hand by date/itemcode for ALL warehouses combined.
    Columns: date, itemcode, onhand_qty, onhand_cost
    """
    if stock_mv.empty:
        return pd.DataFrame(columns=["date", "itemcode", "onhand_qty", "onhand_cost"])

    daily = (
        stock_mv.groupby(["date", "itemcode"], as_index=False)
        .agg(mv_qty=("stockqty", "sum"), mv_cost=("stockvalue", "sum"))
        .sort_values(["itemcode", "date"])
        .reset_index(drop=True)
    )

    daily["onhand_qty"] = daily.groupby("itemcode")["mv_qty"].cumsum()
    daily["onhand_cost"] = daily.groupby("itemcode")["mv_cost"].cumsum()

    return daily[["date", "itemcode", "onhand_qty", "onhand_cost"]]

@st.cache_data
def _prep_stock_timeseries(stock_movement_df: pd.DataFrame, zids: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Shared stock truth engine:
      - cleaned stock movement rows (multi-zid)
      - per-day onhand series per item (qty + cost) using cumsum
      - per-day total inventory cost (sum of onhand_cost across items)
    """
    sm = _prep_stock_movement(stock_movement_df, zids=zids)
    onhand = _onhand_series(sm)
    total_onhand_cost = _total_onhand_cost_series(onhand)

    return {
        "sm": sm,
        "onhand": onhand,
        "total_onhand_cost": total_onhand_cost,
    }

def _total_onhand_cost_series(onhand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Total on-hand cost across all SKUs (all warehouses) per date.
    Columns: date, total_onhand_cost
    """
    if onhand_df.empty:
        return pd.DataFrame(columns=["date", "total_onhand_cost"])

    tot = (
        onhand_df.groupby("date", as_index=False)["onhand_cost"]
        .sum()
        .rename(columns={"onhand_cost": "total_onhand_cost"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    return tot

def _compute_batch_end_and_sold_remaining(
    batch: pd.Series,
    onhand: pd.DataFrame,
    as_of: pd.Timestamp,
    threshold_qty: float = 0.0,) -> Tuple[pd.Timestamp, float, float, float, float]:
    """
    Depletion-based FIFO (virtual bin) using onhand series:

    baseline = onhand_qty(combinedate - 1)
    shipment_qty = initial_qty

    shipment_remaining(date) = min(shipment_qty, max(0, onhand_qty(date) - baseline))

    End date = first date >= combinedate where shipment_remaining(date) <= threshold_qty (with EPS).
    If not found: open batch (end_date = NaT).

    Returns:
      end_date,
      sold_qty,
      remaining_qty,
      baseline_onhand_before (this is your 'threshold_qty' in the report),
      onhand_at_end (onhand qty at end_eff; useful for debugging)
    """
    code = batch["itemcode"]
    start = pd.to_datetime(batch["combinedate"], errors="coerce").floor("D")
    ship_qty = float(batch["initial_qty"])

    if pd.isna(start) or ship_qty <= 0:
        return (pd.NaT, 0.0, ship_qty, 0.0, 0.0)

    sku = onhand[onhand["itemcode"] == code].copy()
    if sku.empty:
        return (pd.NaT, 0.0, ship_qty, 0.0, 0.0)

    # baseline onhand before shipment arrival
    before_date = start - pd.Timedelta(days=1)
    sku_before = sku[sku["date"] <= before_date]
    baseline = float(sku_before["onhand_qty"].iloc[-1]) if not sku_before.empty else 0.0

    # slice after start
    sku_after = sku[(sku["date"] >= start) & (sku["date"] <= as_of)].copy()
    if sku_after.empty:
        # no movements after start, assume still open
        return (pd.NaT, 0.0, ship_qty, baseline, baseline)

    # compute shipment remaining series
    rem = (sku_after["onhand_qty"] - baseline).clip(lower=0.0).clip(upper=ship_qty)

    # float residue tolerance (important for cases like 1.088e-14)
    EPS = 1e-6
    rem = rem.where(rem > EPS, 0.0)

    sku_after["ship_remaining"] = rem

    # find end date
    end_date = pd.NaT
    thr = float(threshold_qty) + EPS
    hit = sku_after[sku_after["ship_remaining"] <= thr]
    if not hit.empty:
        end_date = pd.to_datetime(hit["date"].iloc[0]).floor("D")

    end_eff = end_date if pd.notna(end_date) else as_of

    # remaining at end_eff
    sku_end = sku_after[sku_after["date"] <= end_eff]
    remaining_eff = float(sku_end["ship_remaining"].iloc[-1]) if not sku_end.empty else ship_qty

    sold_eff = ship_qty - remaining_eff
    sold_eff = max(0.0, min(ship_qty, sold_eff))

    # onhand at end_eff (debugging)
    sku_onhand_end = sku_after[sku_after["date"] <= end_eff]
    onhand_at_end = float(sku_onhand_end["onhand_qty"].iloc[-1]) if not sku_onhand_end.empty else baseline

    return end_date, sold_eff, remaining_eff, baseline, onhand_at_end

def _sales_revenue_for_period(sales_df: pd.DataFrame, itemcode: str, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[float, float]:
    """
    Revenue still from sales table. We compute revenue and qty sold in the window.
    """
    if sales_df is None or sales_df.empty:
        return 0.0, 0.0

    s = sales_df.copy()
    s["itemcode"] = s["itemcode"].apply(_norm_code)
    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.floor("D")
    s = s[(s["itemcode"] == itemcode) & (s["date"] >= start) & (s["date"] <= end)].copy()
    if s.empty:
        return 0.0, 0.0

    s["quantity"] = pd.to_numeric(s["quantity"], errors="coerce").fillna(0.0)
    s["totalsales"] = pd.to_numeric(s["totalsales"], errors="coerce").fillna(0.0)

    return float(s["totalsales"].sum()), float(s["quantity"].sum())

def build_shipment_inventory_tables(
    purchase_df: pd.DataFrame,
    stock_movement_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    shipmentname: str,
    project: str = None,
    zid_deplete: str = "100001",) -> Dict[str, pd.DataFrame]:
    """
    Inventory tables bundle for Batch Profitability -> Inventory Check.

    Key rules:
      1) Arrival Check — 100001:
         - Purchases that happened in zid=100001 only
         - Onhand_before / Onhand_end are computed from stock_movement for zid=100001 ONLY

      2) Arrival Check — 100009 Items:
         - Purchases that happened in zid=100009 (itemcode already normalized to the sell code / packcode)
         - Onhand_before / Onhand_end are computed from stock_movement across BOTH zids (100001 + 100009),
           rolled up under the sell code (itemcode)

      3) Sales vs Stock Reconciliation:
         - Expected onhand today (sales model) vs Actual onhand today (stock ledger)
         - Expected = onhand_before_total + shipment_qty_total - sales_qty_window + return_qty_window
         - Actual = stock ledger onhand today across BOTH zids
    """

    # ---- as_of: always a clean day timestamp (ns) ----
    as_of = pd.Timestamp(pd.to_datetime(_today(), errors="coerce")).floor("D")
    if pd.isna(as_of):
        as_of = pd.Timestamp.today().floor("D")

    # ----------------------------
    # 0) Purchase scope: selected shipment only
    # ----------------------------
    p = purchase_df.copy() if isinstance(purchase_df, pd.DataFrame) else pd.DataFrame()
    if p.empty:
        return {
            "arrival_check_100001_only": pd.DataFrame(),
            "arrival_check_100009_items": pd.DataFrame(),
            "reconcile_sales_vs_stock": pd.DataFrame(),
            "warehouse_breakdown": pd.DataFrame(),
        }

    p["shipmentname"] = p["shipmentname"].astype(str).str.strip()
    p = p[p["shipmentname"] == str(shipmentname)].copy()
    if p.empty:
        return {
            "arrival_check_100001_only": pd.DataFrame(),
            "arrival_check_100009_items": pd.DataFrame(),
            "reconcile_sales_vs_stock": pd.DataFrame(),
            "warehouse_breakdown": pd.DataFrame(),
        }

    p["zid"] = p["zid"].astype(str).str.strip()
    p["itemcode"] = p["itemcode"].apply(_norm_code).astype(str).str.strip()
    p["itemname"] = p.get("itemname", "").astype(str)
    p["combinedate"] = pd.to_datetime(p["combinedate"], errors="coerce").dt.floor("D")
    p = p[p["combinedate"].notna()].copy()
    p["quantity"] = pd.to_numeric(p.get("quantity", 0), errors="coerce").fillna(0.0)

    # Purchase qty by zid (vectorized)
    p["qty_100001"] = np.where(p["zid"] == "100001", p["quantity"], 0.0)
    p["qty_100009"] = np.where(p["zid"] == "100009", p["quantity"], 0.0)

    p_sum = (
        p.groupby(["shipmentname", "itemcode", "itemname", "combinedate"], as_index=False)
         .agg(
            purchased_qty_total=("quantity", "sum"),
            purchased_qty_100001=("qty_100001", "sum"),
            purchased_qty_100009=("qty_100009", "sum"),
         )
         .sort_values(["combinedate", "itemcode"])
         .reset_index(drop=True)
    )

    # ----------------------------
    # 1) Stock movement prep (ledger deltas)
    # ----------------------------
    sm = stock_movement_df.copy() if isinstance(stock_movement_df, pd.DataFrame) else pd.DataFrame()
    if sm.empty:
        # Return tables with purchase only (inventory unknown)
        arrival_100001_only = p_sum[p_sum["purchased_qty_100009"] <= 0].copy()
        arrival_100001_only = arrival_100001_only.rename(columns={"purchased_qty_100001": "purchased_qty_total(100001)"})
        arrival_100001_only["onhand_before_total(100001)"] = 0.0
        arrival_100001_only["onhand_end_of_date_total(100001)"] = 0.0
        arrival_100001_only = arrival_100001_only[[
            "shipmentname", "itemcode", "itemname", "combinedate",
            "purchased_qty_total(100001)", "onhand_before_total(100001)", "onhand_end_of_date_total(100001)"
        ]]

        arrival_100009 = p_sum[p_sum["purchased_qty_100009"] > 0].copy()
        arrival_100009 = arrival_100009.rename(columns={"purchased_qty_100009": "purchased_qty_total"})
        arrival_100009["onhand_before_total(100001+100009)"] = 0.0
        arrival_100009["onhand_end_of_date_total(100001+100009)"] = 0.0
        arrival_100009 = arrival_100009[[
            "shipmentname", "itemcode", "itemname", "combinedate",
            "purchased_qty_total", "onhand_before_total(100001+100009)", "onhand_end_of_date_total(100001+100009)"
        ]]

        return {
            "arrival_check_100001_only": arrival_100001_only.reset_index(drop=True),
            "arrival_check_100009_items": arrival_100009.reset_index(drop=True),
            "reconcile_sales_vs_stock": pd.DataFrame(),
            "warehouse_breakdown": pd.DataFrame(),
        }

    sm["zid"] = sm["zid"].astype(str).str.strip()
    sm = sm[sm["zid"].isin(["100001", "100009"])].copy()
    sm["itemcode"] = sm["itemcode"].apply(_norm_code).astype(str).str.strip()
    sm["date"] = pd.to_datetime(sm["date"], errors="coerce").dt.floor("D")
    sm = sm[sm["date"].notna()].copy()
    sm["stockqty"] = pd.to_numeric(sm.get("stockqty", 0), errors="coerce").fillna(0.0)
    sm["warehouse"] = sm.get("warehouse", "").astype(str).fillna("").str.strip()

    def _build_daily_onhand(sm_in: pd.DataFrame) -> pd.DataFrame:
        d = (
            sm_in.groupby(["date", "itemcode"], as_index=False)["stockqty"]
            .sum()
            .copy()
        )
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.floor("D")
        d = d[d["date"].notna()].copy()
        d = d.sort_values(["date", "itemcode"]).reset_index(drop=True)
        d["onhand_qty"] = d.groupby("itemcode")["stockqty"].cumsum()
        return d[["date", "itemcode", "onhand_qty"]]

    daily_total = _build_daily_onhand(sm)                         # 100001+100009
    daily_100001 = _build_daily_onhand(sm[sm["zid"] == "100001"]) # 100001 only

    def _asof_onhand(daily_onhand: pd.DataFrame, q: pd.DataFrame, qdate_col: str, out_col: str) -> pd.DataFrame:
        """
        As-of lookup: for each (itemcode, qdate) find last onhand_qty where date <= qdate.
        """
        if q.empty:
            return pd.DataFrame(columns=["_rowid", out_col])

        qq = q[["itemcode", qdate_col]].copy()
        qq = qq.reset_index(drop=False).rename(columns={"index": "_rowid"})
        qq["itemcode"] = qq["itemcode"].astype(str).str.strip()
        qq[qdate_col] = pd.to_datetime(qq[qdate_col], errors="coerce")
        qq = qq[qq[qdate_col].notna()].copy()

        dd = daily_onhand.copy()
        dd["itemcode"] = dd["itemcode"].astype(str).str.strip()
        dd["date"] = pd.to_datetime(dd["date"], errors="coerce")
        dd = dd[dd["date"].notna()].copy()

        # merge_asof requires sorted by ON key first, then BY key
        qq = qq.sort_values([qdate_col, "itemcode"]).reset_index(drop=True)
        dd = dd.sort_values(["date", "itemcode"]).reset_index(drop=True)

        m = pd.merge_asof(
            qq,
            dd,
            left_on=qdate_col,
            right_on="date",
            by="itemcode",
            direction="backward",
            allow_exact_matches=True,
        )
        m[out_col] = m["onhand_qty"].fillna(0.0).astype(float)
        return m[["_rowid", out_col]]

    # ----------------------------
    # 2) Arrival tables (correct < and <= semantics)
    # ----------------------------
    # IMPORTANT semantics:
    #   before = stock as of strictly before combinedate  => combinedate - 1ns
    #   end    = stock as of end of combinedate           => combinedate + 1day - 1ns

    # ---- Table 1: 100001-only purchases and 100001-only stock ----
    arr1 = p_sum[p_sum["purchased_qty_100009"] <= 0].copy()
    arr1["purchased_qty_total(100001)"] = arr1["purchased_qty_100001"].astype(float)
    arr1["before_ts"] = arr1["combinedate"] - pd.Timedelta(nanoseconds=1)
    arr1["end_ts"] = arr1["combinedate"] + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    b1 = _asof_onhand(daily_100001, arr1, "before_ts", "onhand_before_total(100001)")
    e1 = _asof_onhand(daily_100001, arr1, "end_ts", "onhand_end_of_date_total(100001)")
    arr1 = arr1.join(b1.set_index("_rowid"), how="left").join(e1.set_index("_rowid"), how="left")

    arr1["onhand_before_total(100001)"] = arr1["onhand_before_total(100001)"].fillna(0.0)
    arr1["onhand_end_of_date_total(100001)"] = arr1["onhand_end_of_date_total(100001)"].fillna(0.0)

    arrival_100001_only = arr1[[
        "shipmentname", "itemcode", "itemname", "combinedate",
        "purchased_qty_total(100001)",
        "onhand_before_total(100001)",
        "onhand_end_of_date_total(100001)",
    ]].sort_values(["combinedate", "itemcode"]).reset_index(drop=True)

    # ---- Table 2: 100009 purchases, but stock across BOTH zids under sell code ----
    arr2 = p_sum[p_sum["purchased_qty_100009"] > 0].copy()
    # IMPORTANT: only keep ONE purchased_qty_total column (avoid duplicate columns)
    arr2["purchased_qty_total"] = arr2["purchased_qty_100009"].astype(float)

    arr2["before_ts"] = arr2["combinedate"] - pd.Timedelta(nanoseconds=1)
    arr2["end_ts"] = arr2["combinedate"] + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    b2 = _asof_onhand(daily_total, arr2, "before_ts", "onhand_before_total(100001+100009)")
    e2 = _asof_onhand(daily_total, arr2, "end_ts", "onhand_end_of_date_total(100001+100009)")
    arr2 = arr2.join(b2.set_index("_rowid"), how="left").join(e2.set_index("_rowid"), how="left")

    arr2["onhand_before_total(100001+100009)"] = arr2["onhand_before_total(100001+100009)"].fillna(0.0)
    arr2["onhand_end_of_date_total(100001+100009)"] = arr2["onhand_end_of_date_total(100001+100009)"].fillna(0.0)

    arrival_100009 = arr2[[
        "shipmentname", "itemcode", "itemname", "combinedate",
        "purchased_qty_total",
        "onhand_before_total(100001+100009)",
        "onhand_end_of_date_total(100001+100009)",
    ]].sort_values(["combinedate", "itemcode"]).reset_index(drop=True)

    # ----------------------------
    # 3) Reconcile (expected vs actual)
    # ----------------------------
    base = (
        p_sum.groupby(["itemcode", "itemname", "combinedate"], as_index=False)["purchased_qty_total"]
        .sum()
        .rename(columns={"purchased_qty_total": "shipment_qty_total"})
        .sort_values(["combinedate", "itemcode"])
        .reset_index(drop=True)
    )

    # onhand_before_total across BOTH zids: strictly before combinedate
    base["before_ts"] = base["combinedate"] - pd.Timedelta(nanoseconds=1)
    b_all = _asof_onhand(daily_total, base, "before_ts", "onhand_before_total(100001+100009)")
    base = base.join(b_all.set_index("_rowid"), how="left")
    base["onhand_before_total(100001+100009)"] = base["onhand_before_total(100001+100009)"].fillna(0.0)

    # --- Sales & Returns window: combinedate -> today (end of today) ---
    def _window_qty(df_in: pd.DataFrame, qty_col: str) -> np.ndarray:
        df = df_in.copy() if isinstance(df_in, pd.DataFrame) else pd.DataFrame()
        if df.empty or "itemcode" not in df.columns:
            return np.zeros(len(base), dtype=float)

        df["zid"] = df["zid"].astype(str).str.strip()
        df = df[df["zid"] == str(zid_deplete)].copy()

        dcol = "date" if "date" in df.columns else ("xdate" if "xdate" in df.columns else None)
        if dcol is None:
            return np.zeros(len(base), dtype=float)

        df["itemcode"] = df["itemcode"].apply(_norm_code).astype(str).str.strip()
        df["d"] = pd.to_datetime(df[dcol], errors="coerce").dt.floor("D")
        df = df[df["d"].notna()].copy()
        df[qty_col] = pd.to_numeric(df.get(qty_col, 0), errors="coerce").fillna(0.0)

        daily = (
            df.groupby(["d", "itemcode"], as_index=False)[qty_col]
              .sum()
              .sort_values(["d", "itemcode"])
              .reset_index(drop=True)
        )
        daily["cum"] = daily.groupby("itemcode")[qty_col].cumsum()

        # window start: just before combinedate (strictly < combinedate)
        q0 = base[["itemcode", "combinedate"]].copy()
        q0["qdate"] = q0["combinedate"] - pd.Timedelta(nanoseconds=1)
        q0 = q0[["itemcode", "qdate"]].copy()
        q0 = q0.reset_index(drop=False).rename(columns={"index": "_rowid"})

        # window end: end of today (<= today)
        q1 = base[["itemcode"]].copy()
        q1["qdate"] = pd.Timestamp(as_of).floor("D") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        q1 = q1.reset_index(drop=False).rename(columns={"index": "_rowid"})

        daily_sorted = daily.sort_values(["d", "itemcode"]).reset_index(drop=True)
        q0 = q0.sort_values(["qdate", "itemcode"]).reset_index(drop=True)
        q1 = q1.sort_values(["qdate", "itemcode"]).reset_index(drop=True)

        m0 = pd.merge_asof(q0, daily_sorted, left_on="qdate", right_on="d", by="itemcode",
                           direction="backward", allow_exact_matches=True)
        m1 = pd.merge_asof(q1, daily_sorted, left_on="qdate", right_on="d", by="itemcode",
                           direction="backward", allow_exact_matches=True)

        m0 = m0[["_rowid", "cum"]].rename(columns={"cum": "_c0"})
        m1 = m1[["_rowid", "cum"]].rename(columns={"cum": "_c1"})
        mm = pd.merge(m1, m0, on="_rowid", how="outer").fillna(0.0)

        mm["_rowid"] = pd.to_numeric(mm["_rowid"], errors="coerce")
        mm = mm.set_index("_rowid")

        c1 = mm.reindex(range(len(base)))["_c1"].fillna(0.0).to_numpy()
        c0 = mm.reindex(range(len(base)))["_c0"].fillna(0.0).to_numpy()

        return (c1 - c0).astype(float)

    base["sales_qty_window"] = _window_qty(sales_df, "quantity")
    base["return_qty_window"] = _window_qty(returns_df, "returnqty")

    base["expected_onhand_today_salesmodel"] = (
        base["onhand_before_total(100001+100009)"]
        + base["shipment_qty_total"]
        - base["sales_qty_window"]
        + base["return_qty_window"]
    )

    # Actual onhand today from ledger total (as-of end of today)
    base["today_end_ts"] = pd.Timestamp(as_of).floor("D") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    a_today = _asof_onhand(daily_total, base, "today_end_ts", "actual_onhand_today_stockledger_total")
    base = base.join(a_today.set_index("_rowid"), how="left")

    base["actual_onhand_today_stockledger_total"] = base["actual_onhand_today_stockledger_total"].fillna(0.0)
    base["difference_expected_minus_actual"] = (
        base["expected_onhand_today_salesmodel"] - base["actual_onhand_today_stockledger_total"]
    )

    reconcile = base[[
        "itemcode",
        "itemname",
        "combinedate",
        "onhand_before_total(100001+100009)",
        "shipment_qty_total",
        "sales_qty_window",
        "return_qty_window",
        "expected_onhand_today_salesmodel",
        "actual_onhand_today_stockledger_total",
        "difference_expected_minus_actual",
    ]].sort_values(["combinedate", "itemcode"]).reset_index(drop=True)

    # # ----------------------------
    # # 4) Warehouse breakdown as-of (across BOTH zids)
    # # ----------------------------
    # daily_wh = (
    #     sm.groupby(["warehouse", "date", "itemcode"], as_index=False)["stockqty"]
    #       .sum()
    #       .sort_values(["warehouse", "date", "itemcode"])
    #       .reset_index(drop=True)
    # )
    # daily_wh["onhand_wh"] = daily_wh.groupby(["warehouse", "itemcode"])["stockqty"].cumsum()

    # today_end = pd.Timestamp(as_of).floor("D") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    # wh_asof = daily_wh[daily_wh["date"] <= today_end].copy()
    # wh_asof = (
    #     wh_asof.sort_values(["warehouse", "date", "itemcode"])
    #           .groupby(["warehouse", "itemcode"], as_index=False)
    #           .tail(1)
    # )

    # warehouse_breakdown = wh_asof[["warehouse", "itemcode", "onhand_wh"]].sort_values(
    #     ["warehouse", "itemcode"]
    # ).reset_index(drop=True)

    return {
        "arrival_check_100001_only": arrival_100001_only,
        "arrival_check_100009_items": arrival_100009,
        "reconcile_sales_vs_stock": reconcile,
        # "warehouse_breakdown": warehouse_breakdown,
    }

def run_batch_profitability_engine(
    purchase_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    stock_movement_df: pd.DataFrame,
    glheader_df: pd.DataFrame,
    gldetail_df: pd.DataFrame,
    glmst_df: pd.DataFrame,
    hierarchy_path: str,
    shipmentname: str,
    discount_pct: float = 0.0,
    alloc_base: str = "Cost Share",
    overhead_granularity: str = "Day",
    overhead_mode: str = "Only for this shipment",
    overhead_level: str = "Level 0",
    overhead_node: str = "",
    zid_deplete: str = "100001",
    include_interest_in_overhead: bool = True,
    shipment_overhead_total: float = 0.0,
    vat_pct: float = 0.0,
    manual_overhead_value: float = 0.0,
    inventory_tables: Optional[Dict[str, pd.DataFrame]] = None,) -> pd.DataFrame:
    """
    Final profitability table (shipment -> SKUs), hybrid model:

    - Sold qty & remaining qty come from Inventory Check (reconcile_sales_vs_stock).
    - Closing date (batch_end_date) comes from stock_movement ledger time series,
      detected as first day where ledger onhand reaches <= EPS (fast vectorized).

    Keeps your original columns and overhead logic.
    """
    EPS = 1e-9

    # ----------------------------
    # Dates
    # ----------------------------
    as_of = pd.to_datetime(_today(), errors="coerce")
    if pd.isna(as_of):
        as_of = pd.Timestamp.today()
    as_of = pd.Timestamp(as_of).floor("D")

    # ----------------------------
    # 1) Shipment SKUs
    # ----------------------------
    batches = _prep_purchase_shipment(purchase_df, shipmentname)
    if batches is None or batches.empty:
        return pd.DataFrame()

    df0 = batches.copy()
    df0["itemcode"] = df0["itemcode"].apply(_norm_code).astype(str).str.strip()
    df0["combinedate"] = pd.to_datetime(df0["combinedate"], errors="coerce").dt.floor("D")
    df0 = df0[df0["combinedate"].notna()].copy()

    # Ensure numeric basics exist
    df0["initial_qty"] = pd.to_numeric(df0.get("initial_qty", 0), errors="coerce").fillna(0.0).astype(float)
    df0["unit_cost"] = pd.to_numeric(df0.get("unit_cost", 0), errors="coerce").fillna(0.0).astype(float)

    # ----------------------------
    # 2) Pull Inventory Check reconcile table
    # ----------------------------
    reconcile = None
    if isinstance(inventory_tables, dict):
        reconcile = inventory_tables.get("reconcile_sales_vs_stock")

    if not isinstance(reconcile, pd.DataFrame) or reconcile.empty:
        try:
            tables = build_shipment_inventory_tables(
                purchase_df=purchase_df,
                stock_movement_df=stock_movement_df,
                sales_df=sales_df,
                returns_df=returns_df,
                shipmentname=shipmentname,
                project=None,
                zid_deplete=zid_deplete,
            )
            reconcile = tables.get("reconcile_sales_vs_stock", pd.DataFrame())
        except Exception:
            reconcile = pd.DataFrame()

    if reconcile is None or reconcile.empty:
        return pd.DataFrame()

    rec = reconcile.copy()
    rec["itemcode"] = rec["itemcode"].apply(_norm_code).astype(str).str.strip()
    rec["combinedate"] = pd.to_datetime(rec["combinedate"], errors="coerce").dt.floor("D")
    rec = rec[rec["combinedate"].notna()].copy()

    needed = [
        "itemcode",
        "combinedate",
        "onhand_before_total(100001+100009)",
        "sales_qty_window",
        "return_qty_window",
        "actual_onhand_today_stockledger_total",
    ]
    for c in needed:
        if c not in rec.columns:
            rec[c] = 0.0
    rec = rec[needed].copy()

    df0 = df0.merge(rec, on=["itemcode", "combinedate"], how="left")
    for c in needed[2:]:
        df0[c] = pd.to_numeric(df0[c], errors="coerce").fillna(0.0)

    # ----------------------------
    # 3) Sold qty + remaining qty FROM inventory check
    # ----------------------------
    net_sold_window = (df0["sales_qty_window"] - df0["return_qty_window"]).astype(float)

    sold_qty = np.clip(
        net_sold_window.to_numpy(dtype=float),
        0.0,
        df0["initial_qty"].to_numpy(dtype=float),
    )
    remaining_qty = np.clip(df0["initial_qty"].to_numpy(dtype=float) - sold_qty, 0.0, None)
    remaining_qty = np.where(remaining_qty < EPS, 0.0, remaining_qty)

    df0["sold_qty"] = sold_qty.astype(float)
    df0["remaining_qty"] = remaining_qty.astype(float)

    # baseline / threshold display
    df0["onhand_before"] = df0["onhand_before_total(100001+100009)"].astype(float)
    df0["threshold_qty"] = df0["onhand_before"].astype(float)

    # ----------------------------
    # 4) Closing date FROM stock ledger (FAST, vectorized)
    # ----------------------------
    ts = _prep_stock_timeseries(stock_movement_df, zids=["100001", "100009"])
    onhand = ts.get("onhand")

    df0["batch_end_date"] = pd.NaT  # default

    if onhand is not None and not onhand.empty:
        oh = onhand.copy()

        if "onhand_qty" not in oh.columns and "onhand" in oh.columns:
            oh = oh.rename(columns={"onhand": "onhand_qty"})

        if "itemcode" in oh.columns:
            oh["itemcode"] = oh["itemcode"].apply(_norm_code).astype(str).str.strip()

        oh["date"] = pd.to_datetime(oh["date"], errors="coerce").dt.floor("D")
        oh = oh[oh["date"].notna()].copy()
        oh["onhand_qty"] = pd.to_numeric(oh.get("onhand_qty", 0), errors="coerce").fillna(0.0)

        oh = oh.sort_values(["itemcode", "date"])

        depletion = (
            oh[oh["onhand_qty"] <= EPS]
            .groupby("itemcode", as_index=False)["date"]
            .min()
            .rename(columns={"date": "_ledger_depletion_date"})
        )

        df0 = df0.merge(depletion, on="itemcode", how="left")

        df0["batch_end_date"] = np.where(
            (df0["_ledger_depletion_date"].notna()) &
            (df0["_ledger_depletion_date"] >= df0["combinedate"]),
            df0["_ledger_depletion_date"],
            pd.NaT,
        )

        df0.drop(columns=["_ledger_depletion_date"], inplace=True)

    # MUST close if remaining_qty == 0 (even if ledger didn't hit 0)
    df0["is_closed"] = (df0["remaining_qty"] <= EPS) | (df0["batch_end_date"].notna())

    # ----------------------------
    # 5) days_active / velocity / days_to_clear
    # ----------------------------
    end_eff = df0["batch_end_date"].where(df0["batch_end_date"].notna(), as_of)

    df0["days_active"] = ((end_eff - df0["combinedate"]).dt.days + 1).clip(lower=1).astype(int)

    # raw sku velocity
    df0["velocity"] = np.where(df0["days_active"] > 0, df0["sold_qty"] / df0["days_active"], 0.0)
    df0["velocity"] = pd.to_numeric(df0["velocity"], errors="coerce").fillna(0.0)

    # velocity used:
    # - use sku velocity if sold_qty > 0
    # - fallback floor 0.02 if sold_qty == 0
    df0["velocity_used"] = np.where(df0["sold_qty"] > 0, df0["velocity"], 0.02)
    df0["velocity_used"] = pd.to_numeric(df0["velocity_used"], errors="coerce").fillna(0.02)

    # days_to_clear using velocity_used, capped at 730 days
    df0["days_to_clear"] = np.where(
        df0["velocity_used"] > 0,
        df0["remaining_qty"] / df0["velocity_used"],
        730.0
    )
    df0["days_to_clear"] = pd.to_numeric(df0["days_to_clear"], errors="coerce").fillna(730.0)
    df0["days_to_clear"] = df0["days_to_clear"].clip(lower=0.0, upper=730.0)

    df0["batch_age_days"] = ((as_of - df0["combinedate"]).dt.days).astype(int)

    # ----------------------------
    # 6) Revenue window (vectorized daily cum + merge_asof)
    # ----------------------------
    def _resolve_sales_value_col(sdf: pd.DataFrame) -> str:
        if sdf is None or not isinstance(sdf, pd.DataFrame) or sdf.empty:
            return ""
        if "totalsales" in sdf.columns:
            return "totalsales"
        if "altsales" in sdf.columns:
            return "altsales"
        return ""

    def _window_rev_qty_vectorized(sdf: pd.DataFrame, value_col: str):
        if sdf is None or not isinstance(sdf, pd.DataFrame) or sdf.empty:
            return np.zeros(len(df0), dtype=float), np.zeros(len(df0), dtype=float)

        s = sdf.copy()
        s["zid"] = s["zid"].astype(str).str.strip()
        s = s[s["zid"] == str(zid_deplete)].copy()
        if s.empty:
            return np.zeros(len(df0), dtype=float), np.zeros(len(df0), dtype=float)

        dcol = "date" if "date" in s.columns else ("xdate" if "xdate" in s.columns else None)
        if dcol is None:
            return np.zeros(len(df0), dtype=float), np.zeros(len(df0), dtype=float)

        s["itemcode"] = s["itemcode"].apply(_norm_code).astype(str).str.strip()
        s["d"] = pd.to_datetime(s[dcol], errors="coerce").dt.floor("D")
        s = s[s["d"].notna()].copy()
        if s.empty:
            return np.zeros(len(df0), dtype=float), np.zeros(len(df0), dtype=float)

        s["quantity"] = pd.to_numeric(s.get("quantity", 0), errors="coerce").fillna(0.0)

        if value_col == "totalsales":
            s["totalsales"] = pd.to_numeric(s.get("totalsales", 0), errors="coerce").fillna(0.0)
            s["_rev"] = s["totalsales"]
        elif value_col == "altsales":
            s["altsales"] = pd.to_numeric(s.get("altsales", 0), errors="coerce").fillna(0.0)
            s["_rev"] = s["altsales"]
        else:
            s["_rev"] = 0.0

        daily = (
            s.groupby(["d", "itemcode"], as_index=False)
            .agg(qty=("quantity", "sum"), rev=("_rev", "sum"))
            .sort_values(["d", "itemcode"])
            .reset_index(drop=True)
        )
        daily["cum_qty"] = daily.groupby("itemcode")["qty"].cumsum()
        daily["cum_rev"] = daily.groupby("itemcode")["rev"].cumsum()

        q0 = df0[["itemcode", "combinedate"]].copy()
        q0["qdate"] = q0["combinedate"] - pd.Timedelta(nanoseconds=1)
        q0 = q0.reset_index(drop=False).rename(columns={"index": "_rowid"})
        q0 = q0.sort_values(["qdate", "itemcode"]).reset_index(drop=True)

        q1 = df0[["itemcode"]].copy()
        q1["qdate"] = pd.to_datetime(end_eff, errors="coerce").dt.floor("D") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        q1 = q1.reset_index(drop=False).rename(columns={"index": "_rowid"})
        q1 = q1.sort_values(["qdate", "itemcode"]).reset_index(drop=True)

        daily_sorted = daily.sort_values(["d", "itemcode"]).reset_index(drop=True)

        m0 = pd.merge_asof(q0, daily_sorted, left_on="qdate", right_on="d", by="itemcode",
                           direction="backward", allow_exact_matches=True)
        m1 = pd.merge_asof(q1, daily_sorted, left_on="qdate", right_on="d", by="itemcode",
                           direction="backward", allow_exact_matches=True)

        m0 = m0[["_rowid", "cum_qty", "cum_rev"]].rename(columns={"cum_qty": "_q0", "cum_rev": "_r0"})
        m1 = m1[["_rowid", "cum_qty", "cum_rev"]].rename(columns={"cum_qty": "_q1", "cum_rev": "_r1"})

        mm = pd.merge(m1, m0, on="_rowid", how="outer").fillna(0.0)
        mm["_rowid"] = pd.to_numeric(mm["_rowid"], errors="coerce")
        mm = mm.set_index("_rowid")

        q1v = mm.reindex(range(len(df0)))["_q1"].fillna(0.0).to_numpy()
        q0v = mm.reindex(range(len(df0)))["_q0"].fillna(0.0).to_numpy()
        r1v = mm.reindex(range(len(df0)))["_r1"].fillna(0.0).to_numpy()
        r0v = mm.reindex(range(len(df0)))["_r0"].fillna(0.0).to_numpy()

        qty_w = np.where(q1v - q0v < 0, 0.0, (q1v - q0v)).astype(float)
        rev_w = np.where(r1v - r0v < 0, 0.0, (r1v - r0v)).astype(float)
        return rev_w, qty_w

    df0["sold_revenue"] = 0.0
    df0["avg_price"] = 0.0

    sales_value_col = _resolve_sales_value_col(sales_df)
    if sales_value_col and isinstance(sales_df, pd.DataFrame) and not sales_df.empty:
        rev_w, qty_w = _window_rev_qty_vectorized(sales_df, sales_value_col)
        avg_price = np.where(qty_w > 0, rev_w / qty_w, 0.0)
        sold_revenue = avg_price * df0["sold_qty"].to_numpy(dtype=float)

        df0["avg_price"] = pd.to_numeric(avg_price, errors="coerce")
        df0["sold_revenue"] = pd.to_numeric(sold_revenue, errors="coerce")

    df0["avg_price"] = pd.to_numeric(df0["avg_price"], errors="coerce").fillna(0.0).astype(float)
    df0["sold_revenue"] = pd.to_numeric(df0["sold_revenue"], errors="coerce").fillna(0.0).astype(float)

    df0["scenario_price"] = df0["avg_price"] * (1.0 - float(discount_pct) / 100.0)

    df0["realized_cogs"] = df0["sold_qty"].astype(float) * df0["unit_cost"].astype(float)
    df0["realized_gm"] = df0["sold_revenue"].astype(float) - df0["realized_cogs"].astype(float)

    df0["remaining_cost_value"] = df0["remaining_qty"].astype(float) * df0["unit_cost"].astype(float)
    df0["proj_remaining_revenue"] = df0["remaining_qty"].astype(float) * df0["scenario_price"].astype(float)
    df0["proj_remaining_gm"] = df0["proj_remaining_revenue"].astype(float) - df0["remaining_cost_value"].astype(float)

    # -----------------------------
    # 7) Overhead pool + allocations (UNCHANGED)
    # -----------------------------
    total_sold_revenue = float(df0["sold_revenue"].sum()) if "sold_revenue" in df0 else 0.0

    vat_overhead_value = (float(vat_pct) / 100.0) * max(0.0, total_sold_revenue)
    manual_overhead_value = float(manual_overhead_value or 0.0)

    total_overhead_pool = float(shipment_overhead_total or 0.0) + float(vat_overhead_value) + float(manual_overhead_value)

    if total_sold_revenue > 0:
        share_real = (df0["sold_revenue"] / total_sold_revenue).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        denom = float(df0["realized_cogs"].sum())
        share_real = (df0["realized_cogs"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0) if denom > 0 else 0.0

    df0["overhead_realized"] = total_overhead_pool * share_real
    df0["net_profit_realized"] = df0["realized_gm"] - df0["overhead_realized"]

    days_elapsed = int(df0["days_active"].max()) if "days_active" in df0 else 1
    days_elapsed = max(1, days_elapsed)
    avg_daily_alloc = total_overhead_pool / float(days_elapsed)

    total_proj_remaining_revenue = float(df0["proj_remaining_revenue"].sum())
    if total_proj_remaining_revenue > 0:
        share_rem = (df0["proj_remaining_revenue"] / total_proj_remaining_revenue).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        denom = float(df0["remaining_cost_value"].sum())
        share_rem = (df0["remaining_cost_value"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0) if denom > 0 else 0.0

    dclear = df0["days_to_clear"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dclear = dclear.clip(lower=0.0, upper=730.0)

    # D0 = initial daily overhead allocation
    D0 = avg_daily_alloc

    # Decaying daily overhead:
    # overhead_projected_sku = D0 * (0.97)^(days_to_clear_used / 60) * days_to_clear_used * remaining_share
    decay_factor = np.power(0.97, dclear / 60.0)

    df0["overhead_projected"] = D0 * decay_factor * dclear * share_rem

    df0["Proj_remaining_profit"] = df0["proj_remaining_gm"] - df0["overhead_projected"]
    df0["proj_final_profit"] = df0["net_profit_realized"] + df0["Proj_remaining_profit"]
    df0.drop(columns=["velocity_used"], errors="ignore", inplace=True)
    # -----------------------------
    # 8) Output columns (EXACT headers)
    # -----------------------------
    cols = [
        "shipmentname",
        "batch_id",
        "itemcode",
        "itemname",
        "onhand_before",
        "combinedate",
        "batch_end_date",
        "is_closed",
        "initial_qty",
        "sold_qty",
        "remaining_qty",
        "threshold_qty",
        "unit_cost",
        "sold_revenue",
        "realized_cogs",
        "realized_gm",
        "overhead_realized",
        "net_profit_realized",
        "remaining_cost_value",
        "proj_remaining_revenue",
        "proj_remaining_gm",
        "overhead_projected",
        "Proj_remaining_profit",
        "proj_final_profit",
        "avg_price",
        "scenario_price",
        "days_active",
        "velocity",
        "days_to_clear",
        "batch_age_days",
    ]

    # Guarantee all requested columns exist (prevents any downstream KeyError)
    for c in cols:
        if c not in df0.columns:
            if c in ("combinedate", "batch_end_date"):
                df0[c] = pd.NaT
            elif c in ("shipmentname", "batch_id", "itemcode", "itemname"):
                df0[c] = ""
            elif c == "is_closed":
                df0[c] = False
            else:
                df0[c] = 0.0

    df0 = df0[cols].copy()

    try:
        df0 = common.decimal_to_float(df0)
    except Exception:
        pass

    return df0.sort_values(["is_closed", "proj_final_profit"], ascending=[True, False]).reset_index(drop=True)
# ============================================================
# Accounts Explorer (GL overhead timeline + shipment allocation)
# ============================================================

def _today_d() -> pd.Timestamp:
    return pd.Timestamp(_date.today()).floor("D")

def _norm_code(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def _load_hierarchy_prefix_labels(hierarchy_path: str) -> Dict[str, str]:
    """
    Load prefix labels from hierarchy.json stored at modules/hierarchy.json.

    Expected keys in JSON:
      raw["Income Statement Hierarchy"] is a dict whose keys look like:
        "06-Office & Administrative Expenses"
        "0601-Office Expenses"
    Returns mapping:
      {"06": "06 - Office & Administrative Expenses", "0601": "0601 - Office Expenses", ...}
    """

    if not hierarchy_path:
        return {}

    # Candidate paths (in order)
    here = os.path.dirname(os.path.abspath(__file__))  # .../modules/data_process_files
    modules_dir = os.path.abspath(os.path.join(here, ".."))  # .../modules

    candidates = [
        hierarchy_path,  # absolute or relative from cwd
        os.path.join(modules_dir, os.path.basename(hierarchy_path)),  # .../modules/hierarchy.json
        os.path.join(os.getcwd(), hierarchy_path),  # cwd/hierarchy.json
    ]

    path = None
    for c in candidates:
        if c and os.path.exists(c):
            path = c
            break

    if not path:
        # Debug help: you can temporarily st.write these candidates in views if needed
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    is_tree = raw.get("Income Statement Hierarchy", {})
    if not isinstance(is_tree, dict) or not is_tree:
        return {}

    out: Dict[str, str] = {}

    def split_key(k: str) -> Tuple[str, str]:
        k = (k or "").strip()
        if "-" not in k:
            return "", ""
        left, right = k.split("-", 1)
        return left.strip(), right.strip()

    # Level-2 keys (2 digits) are top-level keys in the tree
    for l2_key, l1_dict in is_tree.items():
        if isinstance(l2_key, str):
            p2, lab2 = split_key(l2_key)
            # Allow 2-digit numeric OR 2-digit + 1 letter (e.g., 06A, 06B)
            if (
                (p2.isdigit() and len(p2) == 2) or
                (len(p2) == 3 and p2[:2].isdigit() and p2[2].isalpha())
            ):
                out[p2] = f"{p2} - {lab2}"

        # Level-1 keys (4 digits) are inside the L2 dict
        if isinstance(l1_dict, dict):
            for l1_key in l1_dict.keys():
                if not isinstance(l1_key, str):
                    continue
                p1, lab1 = split_key(l1_key)
                if p1.isdigit() and len(p1) == 4:
                    out[p1] = f"{p1} - {lab1}"

    return out

def _prep_gl_join(glheader_df: pd.DataFrame, gldetail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join glheader(date) with gldetail(ac_code,value) by voucher.
    Assumes gldetail has already been filtered by project in SQL (your get_gldetail_simple does).
    """
    h = glheader_df.copy()
    d = gldetail_df.copy()

    if h is None or h.empty or d is None or d.empty:
        return pd.DataFrame(columns=["date", "ac_code", "value"])

    h["voucher"] = h["voucher"].astype(str).str.strip()
    d["voucher"] = d["voucher"].astype(str).str.strip()

    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.floor("D")
    h = h[h["date"].notna()].copy()

    d["ac_code"] = d["ac_code"].astype(str).str.strip()
    d["value"] = pd.to_numeric(d["value"], errors="coerce").fillna(0.0)

    m = d.merge(h[["voucher", "date"]], on="voucher", how="left")
    m = m[m["date"].notna()].copy()
    return m[["date", "ac_code", "value"]]

def _extract_prefix(selection: str) -> str:
    """
    Extract numeric prefix from selections like:
      - "05 - Other Expenses"
      - "0501 - Other Expenses"
      - "05010002"  (level0 exact code)
      - "05010002 Packaging & Blister Expense"
    Handles hyphen '-', en-dash '–', em-dash '—'.
    """
    s = "" if selection is None else str(selection).strip()

    # If selection begins with digits, take the leading digit run
    import re
    m = re.match(r"^(\d+)", s)
    if m:
        return m.group(1)

    # Otherwise split on common dash chars
    for dash in ["-", "–", "—"]:
        if dash in s:
            left = s.split(dash, 1)[0].strip()
            m2 = re.match(r"^(\d+)", left)
            if m2:
                return m2.group(1)
            return left
    return s

def _extract_prefix_from_label(selection: str) -> str:
    if not selection:
        return ""
    selection = str(selection).strip()
    # split at first space or dash
    if " -" in selection:
        return selection.split(" -", 1)[0].strip()
    if "-" in selection:
        return selection.split("-", 1)[0].strip()
    return selection

def _level_match_mask(
    gl: pd.DataFrame,
    level: int,
    selection: str,
    hierarchy_path: str,) -> pd.Series:

    if gl is None or gl.empty:
        return pd.Series([False] * len(gl), index=gl.index)

    sel = _extract_prefix_from_label(selection)
    codes = gl["ac_code"].astype(str).str.strip()

    # Level 0 → exact ac_code
    if level == 0:
        return codes == sel

    # Level 1 → 4-digit numeric prefix
    if level == 1:
        return codes.str[:4] == sel

    # Level 2
    if level == 2:
        special_groups = _load_special_level2_groups(hierarchy_path)

        # 06A / 06B special virtual groups
        if sel in special_groups:
            lvl1_prefixes = special_groups.get(sel, [])
            return codes.str[:4].isin(lvl1_prefixes)

        # Normal numeric family (05/06/07)
        return codes.str[:2] == sel

    return pd.Series([False] * len(gl), index=gl.index)

def _prep_stock_for_total_onhand(stock_mv_df: pd.DataFrame, zid: str) -> pd.DataFrame:
    """
    Total on-hand cost base = cumulative stockvalue across ALL SKUs / ALL warehouses
    - Filter to zid (100001)
    - Stock movements already filtered by project in SQL (per your requirement: no blanks)
    """
    s = stock_mv_df.copy()
    if s is None or s.empty:
        return pd.DataFrame(columns=["date", "total_onhand_cost"])

    s["zid"] = s["zid"].astype(str)
    s = s[s["zid"] == str(zid)].copy()

    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.floor("D")
    s = s[s["date"].notna()].copy()

    s["stockvalue"] = pd.to_numeric(s["stockvalue"], errors="coerce").fillna(0.0)

    daily = (
        s.groupby("date", as_index=False)["stockvalue"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["total_onhand_cost"] = daily["stockvalue"].cumsum()
    return daily[["date", "total_onhand_cost"]]

# ============================================================
# Accounts Explorer (NEW): Shipment-level overhead allocation
# ============================================================

def _selection_masks(
    gl: pd.DataFrame,
    level: int,
    selections: List[str],
    hierarchy_path: str,) -> pd.Series:
    """
    Combine masks for multiple selections using OR logic.
    """
    if gl is None or gl.empty or not selections:
        return pd.Series([False] * len(gl), index=gl.index)

    mask = pd.Series([False] * len(gl), index=gl.index)
    for sel in selections:
        mm = _level_match_mask(gl, level=level, selection=sel, hierarchy_path=hierarchy_path)
        mask = mask | mm

    return mask

def _shipment_daily_value_series(
    purchase_df: pd.DataFrame,
    onhand: pd.DataFrame,
    shipmentname: str,
    as_of: pd.Timestamp,
    threshold_qty: float = 0.0,) -> Tuple[pd.DataFrame, pd.Timestamp, bool]:
    """
    Builds daily shipment value series using depletion-based shipment slice:
      baseline = onhand_qty(combinedate - 1)
      ship_remaining(date) = min(ship_qty, max(0, onhand_qty(date) - baseline))
      ship_value_cost(date) = ship_remaining(date) * unit_cost
    Shipment is considered closed only if ALL SKUs reach remaining <= threshold.
    Returns:
      - df(date, shipment_value_cost)
      - end_eff = batch_end_date if closed else as_of
      - is_closed
    """
    p = purchase_df.copy()
    p["shipmentname"] = p["shipmentname"].astype(str).str.strip()
    p = p[p["shipmentname"] == str(shipmentname)].copy()
    if p.empty:
        return pd.DataFrame(columns=["date", "shipment_value_cost"]), pd.NaT, False

    p["combinedate"] = pd.to_datetime(p["combinedate"], errors="coerce").dt.floor("D")
    p = p[p["combinedate"].notna()].copy()
    if p.empty:
        return pd.DataFrame(columns=["date", "shipment_value_cost"]), pd.NaT, False

    start = p["combinedate"].min()

    p["itemcode"] = p["itemcode"].apply(_norm_code)
    p["quantity"] = pd.to_numeric(p["quantity"], errors="coerce").fillna(0.0)
    p["cost"] = pd.to_numeric(p["cost"], errors="coerce").fillna(0.0)

    sku_agg = p.groupby("itemcode", as_index=False).agg(
        initial_qty=("quantity", "sum"),
        unit_cost=("cost", "mean"),
    )

    parts = []
    closed_flags = []

    for _, r in sku_agg.iterrows():
        code = r["itemcode"]
        ship_qty = float(r["initial_qty"])
        unit_cost = float(r["unit_cost"])
        if ship_qty <= 0 or unit_cost <= 0:
            continue

        sku = onhand[onhand["itemcode"] == code].copy()
        if sku.empty:
            idx = pd.date_range(start, as_of, freq="D")
            parts.append(pd.DataFrame({"date": idx, "ship_val": ship_qty * unit_cost}))
            closed_flags.append(False)
            continue

        before_date = start - pd.Timedelta(days=1)
        sku_before = sku[sku["date"] <= before_date]
        baseline = float(sku_before["onhand_qty"].iloc[-1]) if not sku_before.empty else 0.0

        sku_after = sku[(sku["date"] >= start) & (sku["date"] <= as_of)].copy()
        if sku_after.empty:
            idx = pd.date_range(start, as_of, freq="D")
            parts.append(pd.DataFrame({"date": idx, "ship_val": ship_qty * unit_cost}))
            closed_flags.append(False)
            continue

        rem = (sku_after["onhand_qty"] - baseline).clip(lower=0.0).clip(upper=ship_qty)

        EPS = 1e-6
        rem = rem.where(rem > EPS, 0.0)
        sku_after["ship_remaining"] = rem

        closed = bool(len(sku_after) > 0 and float(sku_after["ship_remaining"].iloc[-1]) <= (float(threshold_qty) + EPS))
        closed_flags.append(closed)

        sku_daily = sku_after.set_index("date")[["ship_remaining"]].sort_index()
        idx = pd.date_range(start, as_of, freq="D")
        sku_daily = sku_daily.reindex(idx).ffill().fillna(ship_qty)
        sku_daily = sku_daily.rename_axis("date").reset_index()
        sku_daily["ship_val"] = sku_daily["ship_remaining"] * unit_cost
        parts.append(sku_daily[["date", "ship_val"]])

    if not parts:
        return pd.DataFrame(columns=["date", "shipment_value_cost"]), pd.NaT, False

    v = pd.concat(parts, ignore_index=True)
    v = v.groupby("date", as_index=False)["ship_val"].sum().rename(columns={"ship_val": "shipment_value_cost"})
    v = v.sort_values("date").reset_index(drop=True)

    is_closed = bool(closed_flags) and all(closed_flags)

    if is_closed:
        eps = 1e-9
        hit = v[v["shipment_value_cost"] <= eps]
        end_eff = pd.to_datetime(hit["date"].iloc[0]).floor("D") if not hit.empty else as_of
    else:
        end_eff = as_of

    v = v[v["date"] <= end_eff].copy()
    return v, end_eff, is_closed

def build_accounts_overhead_summary(
    purchase_df: pd.DataFrame,
    stock_movement_df: pd.DataFrame,
    glheader_df: pd.DataFrame,
    gldetail_df: pd.DataFrame,
    glmst_df: pd.DataFrame,
    hierarchy_path: str,
    shipmentname: str,
    level: int,
    selections: List[str],
    include_details: bool = False,
    zids_inventory: Optional[List[str]] = None,
    warehouse_filters: Optional[Dict[str, List[str]]] = None,   # NEW
    warehouse_json_path: str = "warehouse_filters.json",) -> Dict[str, Any]:
    """
    Shipment-level overhead allocation over shipment age:
      date range = combinedate .. (end_of_shipment if closed else today)

    daily allocated = overhead_total_for_day * (shipment_value_cost / total_inventory_value_cost)

    Uses stock movement cumsum value base (your requirement).
    """
    if zids_inventory is None:
        zids_inventory = ["100001", "100009"]

    # ---------------------------------------------------
    # Apply SAME warehouse filters used by warehouse table
    # ---------------------------------------------------
    sm = stock_movement_df.copy()
    sm["zid"] = sm["zid"].astype(str).str.strip()
    sm["warehouse"] = sm["warehouse"].astype(str).fillna("").str.strip()

    zset = set([str(z).strip() for z in zids_inventory])
    sm = sm[sm["zid"].isin(zset)].copy()

    # Determine warehouse allowlist per zid
    if warehouse_filters is None:
        wh_map = load_warehouse_filters(warehouse_json_path)
    else:
        wh_map = {str(k).strip(): [str(x).strip() for x in v] for k, v in warehouse_filters.items()}

    if wh_map:
        keep = pd.Series(False, index=sm.index)
        for zid_ in sm["zid"].unique():
            allowed = set(wh_map.get(str(zid_).strip(), []))
            if allowed:
                keep = keep | ((sm["zid"] == zid_) & (sm["warehouse"].isin(allowed)))
        sm = sm[keep].copy()

    as_of = _today_d()

    # Shared stock truth engine (you already created this earlier)
    ts = _prep_stock_timeseries(stock_movement_df, zids=zids_inventory)
    onhand = ts["onhand"]
    inv_cost_day = ts["total_onhand_cost"]  # columns: date, total_onhand_cost

    if onhand.empty or inv_cost_day.empty:
        return {
            "summary_df": pd.DataFrame(),
            "totals": {"overhead_total_sum": 0.0, "overhead_for_shipment_sum": 0.0, "avg_daily_overhead_for_shipment": 0.0},
            "details_df": pd.DataFrame(),
            "end_eff": pd.NaT,
            "is_closed": False,
        }

    ship_val_day, end_eff, is_closed = _shipment_daily_value_series(
        purchase_df=purchase_df,
        onhand=onhand,
        shipmentname=shipmentname,
        as_of=as_of,
        threshold_qty=0.0,
    )


    if ship_val_day.empty or pd.isna(end_eff):
        return {
            "summary_df": pd.DataFrame(),
            "totals": {"overhead_total_sum": 0.0, "overhead_for_shipment_sum": 0.0, "avg_daily_overhead_for_shipment": 0.0},
            "details_df": pd.DataFrame(),
            "end_eff": end_eff,
            "is_closed": bool(is_closed),
        }

    start_date = pd.to_datetime(ship_val_day["date"].min()).floor("D")

    # ------------------------------
    # Inventory value series (NEW)
    # ------------------------------
    inv_daily = total_inventory_value_timeseries(
        stock_movement_df=sm,  # FILTERED
        start_date=start_date,
        end_date=end_eff,
        zids=zids_inventory,
        warehouse_json_path=warehouse_json_path,
        override_selected_warehouses=warehouse_filters,  # SAME AS WAREHOUSE TABLE
    )

    # rename to match your downstream variable names
    inv = inv_daily.rename(columns={"total_inventory_value": "total_inventory_value_cost"}).copy()

    gl = _prep_gl_join(glheader_df, gldetail_df)
    if gl.empty:
        return {
            "summary_df": pd.DataFrame(),
            "totals": {"overhead_total_sum": 0.0, "overhead_for_shipment_sum": 0.0, "avg_daily_overhead_for_shipment": 0.0},
            "details_df": pd.DataFrame(),
            "end_eff": end_eff,
            "is_closed": bool(is_closed),
        }

    gl = gl[(gl["date"] >= start_date) & (gl["date"] <= end_eff)].copy()

    # multi-select filter
    m = _selection_masks(gl, level=level, selections=selections, hierarchy_path=hierarchy_path)
    gl_sel = gl[m].copy()

    # daily overhead totals
    ov_day = (
        gl_sel.groupby("date", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "overhead_total_for_day"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    ship = ship_val_day.copy()

    # continuous daily base
    idx = pd.date_range(start_date, end_eff, freq="D")
    base = pd.DataFrame({"date": idx})
    base = base.merge(inv, on="date", how="left")
    base = base.merge(ship, on="date", how="left")
    base = base.merge(ov_day, on="date", how="left")

    base["total_inventory_value_cost"] = base["total_inventory_value_cost"].ffill().fillna(0.0)
    base["shipment_value_cost"] = base["shipment_value_cost"].ffill().fillna(0.0)
    base["overhead_total_for_day"] = base["overhead_total_for_day"].fillna(0.0)

    denom = base["total_inventory_value_cost"].replace(0.0, np.nan)
    base["ratio"] = (base["shipment_value_cost"] / denom).fillna(0.0)
    base["overhead_allocated_for_day"] = base["overhead_total_for_day"] * base["ratio"]

    overhead_total_sum = float(base["overhead_total_for_day"].sum())
    overhead_for_ship_sum = float(base["overhead_allocated_for_day"].sum())
    n_days = max(1, int(len(base)))
    avg_daily_alloc = overhead_for_ship_sum / n_days

    # Build "Level in rows" summary table
    prefix_labels = _load_hierarchy_prefix_labels(hierarchy_path)

    def _label_for(sel: str) -> str:
        pfx = _extract_prefix(sel)
        if level == 0:
            code = pfx
            if glmst_df is not None and not glmst_df.empty:
                gm = glmst_df.copy()
                gm["ac_code"] = gm["ac_code"].astype(str).str.strip()
                gm["ac_name"] = gm.get("ac_name", "").astype(str).fillna("").str.strip()
                hit = gm[gm["ac_code"] == code]
                if not hit.empty:
                    return f"{code} - {hit['ac_name'].iloc[0]}"
            return code
        if level == 1:
            return prefix_labels.get(pfx[:4], pfx[:4])
        return prefix_labels.get(pfx[:2], pfx[:2])

    # If selections empty => treat as one aggregated row
    sel_list = selections[:] if selections else ["(ALL SELECTED)"]

    rows = []
    for sel in sel_list:
        if sel == "(ALL SELECTED)":
            gl_part = gl_sel
            lbl = "(All selected accounts)"
            code_out = ""
        else:
            gl_part = gl[_level_match_mask(gl, level=level, selection=sel, hierarchy_path=hierarchy_path)].copy()
            lbl = _label_for(sel)
            code_out = _extract_prefix(sel)

        ovp = (
            gl_part.groupby("date", as_index=False)["value"]
            .sum()
            .rename(columns={"value": "overhead_total_for_day"})
        )
        tmp = base[["date", "total_inventory_value_cost", "shipment_value_cost"]].merge(ovp, on="date", how="left")
        tmp["overhead_total_for_day"] = tmp["overhead_total_for_day"].fillna(0.0)

        denom2 = tmp["total_inventory_value_cost"].replace(0.0, np.nan)
        tmp["ratio"] = (tmp["shipment_value_cost"] / denom2).fillna(0.0)
        tmp["overhead_allocated_for_day"] = tmp["overhead_total_for_day"] * tmp["ratio"]

        rows.append({
            "level": level,
            "selection": code_out,
            "label": lbl,
            "overhead_total": float(tmp["overhead_total_for_day"].sum()),
            "overhead_for_shipment": float(tmp["overhead_allocated_for_day"].sum()),
        })

    summary_df = pd.DataFrame(rows)

    return {
        "summary_df": summary_df,
        "totals": {
            "overhead_total_sum": overhead_total_sum,
            "overhead_for_shipment_sum": overhead_for_ship_sum,
            "avg_daily_overhead_for_shipment": avg_daily_alloc,
        },
        "details_df": base.copy() if include_details else pd.DataFrame(),
        "end_eff": end_eff,
        "is_closed": bool(is_closed),
    }

def build_accounts_overhead_table(purchase_df: pd.DataFrame,stock_movement_df: pd.DataFrame,glheader_df: pd.DataFrame,gldetail_df: pd.DataFrame,
    glmst_df: pd.DataFrame,  # not required for matching, but used for Level 0 list + names
    hierarchy_path: str,
    shipmentname: str,
    level: int,                 # 0/1/2
    selection: str,             # ac_code or label
    granularity: str,           # "Day" or "Month"
    mode: str,                  # "Total" or "Only for this shipment"
    zid_deplete: str = "100001",) -> pd.DataFrame:
    """
    Returns a table:
      Day granularity:
        period, overhead_total, total_onhand_cost, shipment_cost, shipment_share, overhead_for_shipment
      Month granularity:
        period (YYYY-MM), overhead_total, avg_total_onhand_cost, shipment_cost, shipment_share, overhead_for_shipment

    Notes:
      - GL sign: DO NOT abs(). Expenses are positive (your system), revenue negative.
      - We filter GL rows from shipment combinedate onward.
      - Shipment cost uses purchase (both zids in purchase_df) filtered by shipmentname: sum(qty*unit_cost).
      - Total inventory cost base uses cumulative stockvalue (zid=100001, all warehouses).
    """
    as_of = _today_d()

    # Shipment scope for start date and shipment cost
    p = purchase_df.copy()
    p["shipmentname"] = p["shipmentname"].astype(str).str.strip()
    p = p[p["shipmentname"] == str(shipmentname)].copy()
    if p.empty:
        return pd.DataFrame()

    p["combinedate"] = pd.to_datetime(p["combinedate"], errors="coerce").dt.floor("D")
    p = p[p["combinedate"].notna()].copy()

    start_date = p["combinedate"].min()

    p["quantity"] = pd.to_numeric(p["quantity"], errors="coerce").fillna(0.0)
    p["cost"] = pd.to_numeric(p["cost"], errors="coerce").fillna(0.0)
    shipment_cost = float((p["quantity"] * p["cost"]).sum())
    if shipment_cost <= 0:
        shipment_cost = 1.0

    # GL join
    gl = _prep_gl_join(glheader_df, gldetail_df)
 
    if gl.empty:
        return pd.DataFrame()

    # Filter dates from combinedate onward
    gl = gl[(gl["date"] >= start_date) & (gl["date"] <= as_of)].copy()
    # pfx_dbg = _extract_prefix(selection)
    # st.write("DEBUG selection:", selection, " extracted:", pfx_dbg, " level:", level)
    # Filter by level selection
    mask = _level_match_mask(gl, level=level, selection=selection)
    st.write("DEBUG selection prefix:", sel)
    st.write("DEBUG matched rows:", mask.sum())
    gl = gl[mask].copy()

    if gl.empty:
        # return empty but with expected columns
        cols = ["period", "overhead_total"]
        if mode == "Only for this shipment":
            cols += ["total_onhand_cost_base", "shipment_cost", "shipment_share", "overhead_for_shipment"]
        return pd.DataFrame(columns=cols)

    # Build overhead_total per day
    day = gl.groupby("date", as_index=False)["value"].sum().rename(columns={"value": "overhead_total"})
    day = day.sort_values("date").reset_index(drop=True)

    if granularity == "Month":
        day["period"] = day["date"].dt.to_period("M").astype(str)
        overhead = day.groupby("period", as_index=False)["overhead_total"].sum()
    else:
        overhead = day.rename(columns={"date": "period"})[["period", "overhead_total"]]

    if mode == "Total":
        return overhead.reset_index(drop=True)

    # Only for this shipment: need inventory cost base per day/month
    tot_cost_day = _prep_stock_for_total_onhand(stock_movement_df, zid=zid_deplete)
    if tot_cost_day.empty:
        overhead["total_onhand_cost_base"] = np.nan
        overhead["shipment_cost"] = shipment_cost
        overhead["shipment_share"] = np.nan
        overhead["overhead_for_shipment"] = np.nan
        return overhead.reset_index(drop=True)

    if granularity == "Month":
        tot_cost_day["period"] = tot_cost_day["date"].dt.to_period("M").astype(str)
        # You asked month base to be "total stock value" for that month slice.
        # Best interpretation: average total on-hand cost across the days present in that month slice.
        base = tot_cost_day.groupby("period", as_index=False)["total_onhand_cost"].mean().rename(
            columns={"total_onhand_cost": "total_onhand_cost_base"}
        )
    else:
        base = tot_cost_day.rename(columns={"date": "period", "total_onhand_cost": "total_onhand_cost_base"})[
            ["period", "total_onhand_cost_base"]
        ]

    overhead = overhead.merge(base, on="period", how="left")
    overhead["shipment_cost"] = shipment_cost
    overhead["shipment_share"] = overhead["shipment_cost"] / overhead["total_onhand_cost_base"]
    overhead["overhead_for_shipment"] = overhead["overhead_total"] * overhead["shipment_share"]

    return overhead.reset_index(drop=True)

def build_accounts_selector_options(glmst_df: pd.DataFrame, hierarchy_path: str) -> Dict[str, List[str]]:
    """
    Robust option builder for Accounts Explorer.

    Builds options that ACTUALLY exist in glmst_df, while using hierarchy.json labels when available.

    Levels:
      - Level 2:
          * numeric families: 05, 06, 07 (based on ac_code[:2] present in glmst)
          * special families: 06A, 06B (NOT present in ac_code prefixes; pulled from hierarchy.json subtree)
      - Level 1:
          * 4-digit numeric prefixes that exist in glmst (e.g., 0501, 0601, 0629, 0630, 0633, ...)
      - Level 0:
          * exact ac_codes + ac_name (for all 05/06/07 accounts)

    Returns:
      {
        "level2_options": [...],
        "level1_options": [...],
        "level0_options": [...]
      }
    """
    prefix_labels = _load_hierarchy_prefix_labels(hierarchy_path)
    special_groups = _load_special_level2_groups(hierarchy_path)  # {"06A":[...], "06B":[...]}

    if glmst_df is None or glmst_df.empty:
        return {"level2_options": [], "level1_options": [], "level0_options": []}

    gm = glmst_df.copy()
    gm["ac_code"] = gm["ac_code"].astype(str).fillna("").str.strip()
    gm["ac_name"] = gm.get("ac_name", "").astype(str).fillna("").str.strip()

    # Keep only expense families we care about (numeric prefixes only; 06A/06B are hierarchy groupings, not ac_code prefixes)
    gm = gm[gm["ac_code"].str.startswith(("05", "06", "07"))].copy()
    if gm.empty:
        return {"level2_options": [], "level1_options": [], "level0_options": []}

    # -----------------------------
    # Level 2 options
    # -----------------------------
    # Numeric families that exist in data
    p2_numeric = sorted(gm["ac_code"].str[:2].unique().tolist())

    # Add special hierarchy-defined families (06A/06B) if they exist in hierarchy
    p2 = list(p2_numeric)
    for k in ("06A", "06B"):
        if special_groups.get(k):  # non-empty list => exists in hierarchy
            p2.append(k)

    # Stable order
    p2_order = ["05", "06", "06A", "06B", "07"]
    p2 = [x for x in p2_order if x in set(p2)]

    level2_options = [prefix_labels.get(p, p) for p in p2]

    # -----------------------------
    # Level 1 options
    # -----------------------------
    p1 = sorted(gm["ac_code"].str[:4].unique().tolist())
    p1 = [p for p in p1 if p.isdigit() and len(p) == 4]
    level1_options = [prefix_labels.get(p, p) for p in p1]

    # -----------------------------
    # Level 0 options
    # -----------------------------
    gm = gm.sort_values("ac_code")
    level0_options = (gm["ac_code"] + " " + gm["ac_name"]).tolist()

    return {
        "level2_options": level2_options,
        "level1_options": level1_options,
        "level0_options": level0_options,
    }

def _load_special_level2_groups(hierarchy_path: str) -> Dict[str, List[str]]:
    """
    Returns mapping for special Level-2 heads that are not numeric prefixes in ac_code,
    e.g. 06A and 06B, to the Level-1 numeric prefixes under them.

    Example:
      {
        "06A": ["0630", "0633", "0635", ...],
        "06B": ["0629", ...]
      }
    """
    if not hierarchy_path:
        return {}

    # Same candidate path logic you already use
    here = os.path.dirname(os.path.abspath(__file__))  # .../modules/data_process_files
    modules_dir = os.path.abspath(os.path.join(here, ".."))  # .../modules
    candidates = [
        hierarchy_path,
        os.path.join(modules_dir, os.path.basename(hierarchy_path)),
        os.path.join(os.getcwd(), hierarchy_path),
    ]

    path = None
    for c in candidates:
        if c and os.path.exists(c):
            path = c
            break
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tree = raw.get("Income Statement Hierarchy", {})
    if not isinstance(tree, dict):
        return {}

    out: Dict[str, List[str]] = {"06A": [], "06B": []}

    def split_key(k: str) -> Tuple[str, str]:
        k = (k or "").strip()
        if "-" not in k:
            return "", ""
        left, right = k.split("-", 1)
        return left.strip(), right.strip()

    for l2_key, l1_dict in tree.items():
        p2, _lab2 = split_key(l2_key)
        if p2 not in ("06A", "06B"):
            continue
        if isinstance(l1_dict, dict):
            for l1_key in l1_dict.keys():
                p1, _lab1 = split_key(l1_key)
                # Level-1 prefixes are numeric 4 digits (0629/0630/0633/...)
                if p1.isdigit() and len(p1) == 4:
                    out[p2].append(p1)

    # de-dup + sort
    out["06A"] = sorted(list(set(out["06A"])))
    out["06B"] = sorted(list(set(out["06B"])))
    return out

def build_warehouse_total_value_table(
    stock_movement_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    zids: List[str] = None,
    warehouse_filters: Optional[Dict[str, List[str]]] = None,
    warehouse_json_path: str = "warehouse_filters.json",) -> pd.DataFrame:
    """
    Returns zid, warehouse, totalvalue (sum of movement stockvalue) up to as_of_date.

    Optional filtering:
      - warehouse_filters: {"100001": [...], "100009": [...]}
        (explicit selection from UI)
      - otherwise uses modules/warehouse_filters.json via load_warehouse_filters()

    stock_movement_df must have: zid, warehouse, date, stockqty, stockvalue.
    """

    if zids is None:
        zids = ["100001", "100009"]

    if stock_movement_df is None or stock_movement_df.empty:
        return pd.DataFrame(columns=["zid", "warehouse", "totalvalue"])

    df = stock_movement_df.copy()

    # normalize zid
    df["zid"] = df["zid"].astype(str).str.strip()
    zid_set = set(str(z).strip() for z in zids)
    df = df[df["zid"].isin(list(zid_set))].copy()
    if df.empty:
        return pd.DataFrame(columns=["zid", "warehouse", "totalvalue"])

    # required cols
    for col in ["date", "warehouse", "stockqty", "stockvalue"]:
        if col not in df.columns:
            raise KeyError(f"stock_movement_df missing required column: {col}")

    # normalize dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    as_of_date = pd.to_datetime(as_of_date, errors="coerce").normalize()
    df = df[df["date"].notna() & (df["date"] <= as_of_date)].copy()
    if df.empty:
        return pd.DataFrame(columns=["zid", "warehouse", "totalvalue"])

    # normalize warehouse/value
    df["warehouse"] = df["warehouse"].astype(str).fillna("").str.strip()
    df["stockqty"] = pd.to_numeric(df["stockqty"], errors="coerce").fillna(0.0)
    df["stockvalue"] = pd.to_numeric(df["stockvalue"], errors="coerce").fillna(0.0)

    # ---------------------------------------
    # NEW: apply warehouse filters (like overhead summary)
    # ---------------------------------------
    if warehouse_filters is None:
        wh_map = load_warehouse_filters(warehouse_json_path)  # {"100001":[...], "100009":[...]}
    else:
        wh_map = {str(k).strip(): [str(x).strip() for x in v] for k, v in warehouse_filters.items()}

    if wh_map:
        # keep only enabled warehouses for each zid when provided
        keep_mask = pd.Series(False, index=df.index)
        for zid_key, wh_list in wh_map.items():
            wh_set = set(w for w in (wh_list or []) if w)
            if not wh_set:
                continue
            keep_mask = keep_mask | ((df["zid"] == zid_key) & (df["warehouse"].isin(list(wh_set))))
        # If filters exist but nothing matched, return empty
        if keep_mask.any():
            df = df[keep_mask].copy()
        else:
            return pd.DataFrame(columns=["zid", "warehouse", "totalvalue"])

    # drop blank warehouses (after filtering)
    df = df[df["warehouse"].astype(str).str.len() > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=["zid", "warehouse", "totalvalue"])

    # cumulative movement sum per (zid, warehouse)
    out = (
        df.groupby(["zid", "warehouse"], as_index=False)
          .agg(totalvalue=("stockvalue", "sum"))
          .sort_values(["zid", "warehouse"])
          .reset_index(drop=True)
    )
    out["totalvalue"] = out["totalvalue"].astype(float)
    return out

# ============================================================
# Accounts Explorer Details of inventory
# ============================================================

def _resolve_modules_file(path_hint: str) -> Optional[str]:
    """
    Resolve a file path that may live under /modules.
    purchase.py is under modules/data_process_files.
    """
    if not path_hint:
        return None

    here = os.path.dirname(os.path.abspath(__file__))       # .../modules/data_process_files
    modules_dir = os.path.abspath(os.path.join(here, "..")) # .../modules

    candidates = [
        path_hint,  # absolute OR relative from CWD
        os.path.join(modules_dir, path_hint),               # modules/<path_hint>
        os.path.join(modules_dir, os.path.basename(path_hint)),
        os.path.join(os.getcwd(), path_hint),
    ]

    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

def load_warehouse_filters(warehouse_json_path: str = "warehouse_filters.json") -> Dict[str, List[str]]:
    """
    Load enabled warehouses per zid from modules/warehouse_filters.json.

    Returns:
      {"100001": [...], "100009": [...]}
    """
    path = _resolve_modules_file(warehouse_json_path)
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[str, List[str]] = {}
    for zid, cfg in (raw or {}).items():
        zid_s = str(zid).strip()
        enabled = (cfg or {}).get("enabled", [])
        enabled = [str(x).strip() for x in enabled if str(x).strip()]
        out[zid_s] = enabled
    return out

def build_shipment_bridge_table(purchase_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a shipment selector table across BOTH zids (100001 + 100009).
    Output columns:
      shipmentname, combinedate, ip_100001, ip_100009, grn_100001, grn_100009
    """
    if purchase_df is None or purchase_df.empty:
        return pd.DataFrame(columns=[
            "shipmentname","combinedate","ip_100001","ip_100009","grn_100001","grn_100009"
        ])

    df = purchase_df.copy()
    for c in ["zid","shipmentname","povoucher","grnvoucher","combinedate"]:
        if c not in df.columns:
            raise KeyError(f"purchase_df missing required column: {c}")

    df["zid"] = df["zid"].astype(str).str.strip()
    df["shipmentname"] = df["shipmentname"].astype(str).fillna("").str.strip()
    df = df[df["shipmentname"] != ""].copy()

    df["combinedate"] = pd.to_datetime(df["combinedate"], errors="coerce").dt.normalize()
    df["povoucher"] = df["povoucher"].astype(str).fillna("").str.strip()
    df["grnvoucher"] = df["grnvoucher"].astype(str).fillna("").str.strip()

    # combinedate per shipment = MIN combinedate (safe)
    base = (
        df.groupby("shipmentname", as_index=False)
          .agg(combinedate=("combinedate","min"))
          .sort_values("shipmentname")
          .reset_index(drop=True)
    )

    def _pick_one(s: pd.Series) -> str:
        s = s.dropna().astype(str)
        s = [x.strip() for x in s.tolist() if x.strip()]
        return s[0] if s else ""

    piv = (
        df.groupby(["shipmentname","zid"], as_index=False)
          .agg(
              ip=("povoucher", _pick_one),
              grn=("grnvoucher", _pick_one),
          )
    )

    # merge zid-specific ip/grn into columns
    out = base.copy()
    for zid in ["100001","100009"]:
        tmp = piv[piv["zid"] == zid][["shipmentname","ip","grn"]].copy()
        tmp = tmp.rename(columns={"ip": f"ip_{zid}", "grn": f"grn_{zid}"})
        out = out.merge(tmp, on="shipmentname", how="left")

    for c in ["ip_100001","ip_100009","grn_100001","grn_100009"]:
        if c not in out.columns:
            out[c] = ""

    return out

def warehouse_value_snapshot(
    stock_movement_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    zids: List[str],
    warehouse_json_path: str = "warehouse_filters.json",
    override_selected_warehouses: Optional[Dict[str, List[str]]] = None,) -> pd.DataFrame:
    """
    Snapshot: zid, warehouse, totalvalue
    IMPORTANT: stockvalue is treated as NET MOVEMENT (delta), and we SUM deltas up to as_of_date.

    We filter warehouses using either:
      - override_selected_warehouses (from UI multiselect), OR
      - warehouse_filters.json (enabled warehouses)
    """
    if stock_movement_df is None or stock_movement_df.empty:
        return pd.DataFrame(columns=["zid","warehouse","totalvalue"])

    df = stock_movement_df.copy()
    for c in ["zid","date","warehouse","stockvalue"]:
        if c not in df.columns:
            raise KeyError(f"stock_movement_df missing required column: {c}")

    df["zid"] = df["zid"].astype(str).str.strip()
    zset = set([str(z).strip() for z in zids])
    df = df[df["zid"].isin(zset)].copy()
    if df.empty:
        return pd.DataFrame(columns=["zid","warehouse","totalvalue"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    as_of_date = pd.to_datetime(as_of_date, errors="coerce").normalize()
    df = df[df["date"].notna() & (df["date"] <= as_of_date)].copy()

    df["warehouse"] = df["warehouse"].astype(str).fillna("").str.strip()
    df["stockvalue"] = pd.to_numeric(df["stockvalue"], errors="coerce").fillna(0.0)

    # warehouse filters
    if override_selected_warehouses is not None:
        wh_map = {str(k).strip(): [str(x).strip() for x in v] for k, v in override_selected_warehouses.items()}
    else:
        wh_map = load_warehouse_filters(warehouse_json_path)

    if wh_map:
        keep = pd.Series(False, index=df.index)
        for zid in df["zid"].unique():
            allowed = set(wh_map.get(str(zid).strip(), []))
            if allowed:
                keep = keep | ((df["zid"] == zid) & (df["warehouse"].isin(allowed)))
        df = df[keep].copy()

    if df.empty:
        return pd.DataFrame(columns=["zid","warehouse","totalvalue"])

    out = (
        df.groupby(["zid","warehouse"], as_index=False)
          .agg(totalvalue=("stockvalue","sum"))
          .sort_values(["zid","warehouse"])
          .reset_index(drop=True)
    )
    out["totalvalue"] = out["totalvalue"].astype(float)
    return out

@st.cache_data
def total_inventory_value_timeseries(

    stock_movement_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    zids: List[str],
    warehouse_json_path: str = "warehouse_filters.json",
    override_selected_warehouses: Optional[Dict[str, List[str]]] = None,) -> pd.DataFrame:
    """
    Daily inventory value series for ratio logic.
    Since stockvalue is NET MOVEMENT, we compute:
      day_value = SUM(stockvalue) per day
      total_inventory_value = cumulative SUM(day_value)

    Output columns: date, total_inventory_value
    """
    if stock_movement_df is None or stock_movement_df.empty:
        return pd.DataFrame(columns=["date","total_inventory_value"])

    df = stock_movement_df.copy()
    for c in ["zid","date","warehouse","stockvalue"]:
        if c not in df.columns:
            raise KeyError(f"stock_movement_df missing required column: {c}")

    df["zid"] = df["zid"].astype(str).str.strip()
    zset = set([str(z).strip() for z in zids])
    df = df[df["zid"].isin(zset)].copy()
    if df.empty:
        return pd.DataFrame(columns=["date","total_inventory_value"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["warehouse"] = df["warehouse"].astype(str).fillna("").str.strip()
    df["stockvalue"] = pd.to_numeric(df["stockvalue"], errors="coerce").fillna(0.0)

    start_date = pd.to_datetime(start_date, errors="coerce").normalize()
    end_date = pd.to_datetime(end_date, errors="coerce").normalize()

    df = df[df["date"].notna() & (df["date"] <= end_date)].copy()

    # warehouse filters
    if override_selected_warehouses is not None:
        wh_map = {str(k).strip(): [str(x).strip() for x in v] for k, v in override_selected_warehouses.items()}
    else:
        wh_map = load_warehouse_filters(warehouse_json_path)

    if wh_map:
        keep = pd.Series(False, index=df.index)
        for zid in df["zid"].unique():
            allowed = set(wh_map.get(str(zid).strip(), []))
            if allowed:
                keep = keep | ((df["zid"] == zid) & (df["warehouse"].isin(allowed)))
        df = df[keep].copy()

    if df.empty:
        return pd.DataFrame(columns=["date","total_inventory_value"])

    daily = (
        df.groupby("date", as_index=False)
          .agg(day_value=("stockvalue","sum"))
          .sort_values("date")
          .reset_index(drop=True)
    )
    daily["total_inventory_value"] = daily["day_value"].cumsum()

    daily = daily[daily["date"] >= start_date].copy()
    return daily[["date","total_inventory_value"]].reset_index(drop=True)

# ============================================================
# Warehouse Options Helper
# ============================================================

def get_all_warehouse_options(stock_movement_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Returns distinct warehouse names per zid from stock_movement_df.

    Output format:
        {
            "100001": ["HMBR Main Store", ...],
            "100009": ["Finished Goods Store Packaging", ...]
        }
    """

    if stock_movement_df is None or stock_movement_df.empty:
        return {}

    df = stock_movement_df.copy()

    if "zid" not in df.columns or "warehouse" not in df.columns:
        return {}

    df["zid"] = df["zid"].astype(str).str.strip()
    df["warehouse"] = df["warehouse"].astype(str).fillna("").str.strip()

    df = df[df["warehouse"] != ""].copy()

    out: Dict[str, List[str]] = {}

    for zid in sorted(df["zid"].unique()):
        wh_list = (
            df[df["zid"] == zid]["warehouse"]
            .dropna()
            .unique()
            .tolist()
        )
        out[zid] = sorted(wh_list)

    return out