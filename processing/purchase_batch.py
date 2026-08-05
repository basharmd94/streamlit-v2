from __future__ import annotations

import pandas as pd
import numpy as np
from processing import common
from datetime import datetime
import streamlit as st
from datetime import date as _date
from typing import Dict, List, Tuple, Optional, Any


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
    
    # Derive year/month from date if the daily-item MV is being used (no year/month columns)
    if "year" not in sales_df.columns or "month" not in sales_df.columns:
        _d = pd.to_datetime(sales_df["date"], errors="coerce")
        sales_df = sales_df.copy()
        sales_df["year"]  = _d.dt.year
        sales_df["month"] = _d.dt.month

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
    result_df['itemname'] = result_df['itemname'].astype(str)

    # Format combinedate BEFORE numeric cleanup so it survives as a string.
    # Re-parse from whatever type it is (string or NaT) and format cleanly.
    result_df['combinedate'] = (
        pd.to_datetime(result_df['combinedate'], errors='coerce')
        .dt.strftime('%Y-%m-%d')
        .fillna('')   # items with no cohort match get empty string instead of 'NaT'/0
    )

    # Numeric cleanup (leaves string columns like combinedate, itemname untouched)
    numeric_cols = [c for c in result_df.columns if c not in ('itemcode', 'itemname', 'combinedate')]
    result_df[numeric_cols] = result_df[numeric_cols].apply(
        lambda col: col.map(common.handle_infinity_and_round)
    ).fillna(0)

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
    # NOTE: ProcessPoolExecutor is replaced with sequential processing.
    # ProcessPoolExecutor uses 'spawn' on macOS (Python 3.8+), which re-imports all
    # modules in each worker — including 'import streamlit as st' at the top of this
    # file — causing workers to crash silently and return empty results (all 0s).
    data_rows = process_chunk(purchase_df, sales_df)

    if not data_rows:
        return pd.DataFrame(columns=[
            'itemcode', 'itemname', 'povoucher', 'shipmentname',
            'combinedate', 'quantity', 'cost', 'average sales price',
            '1', '2', '3', '4',
        ])

    final_df = pd.DataFrame(data_rows)
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

@st.cache_data(show_spinner=False, ttl=86400)
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
    Batch profitability using FIFO sales allocation across repeated shipments.

    Key changes:
    - sold_qty / remaining_qty / batch_end_date / sold_revenue are driven by FIFO allocation
      across all shipments of the same itemcode.
    - Inventory Check session state is NOT used for profitability math anymore.
      (Leave Inventory Check for audit only.)
    - onhand_before is still computed from stock movement ledger as a display / threshold field.
    """

    EPS = 1e-9

    # ----------------------------
    # Dates
    # ----------------------------
    as_of = pd.to_datetime(_today(), errors="coerce")
    if pd.isna(as_of):
        as_of = pd.Timestamp.today()
    as_of = pd.Timestamp(as_of).floor("D")

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _prep_all_purchase_batches(p_df: pd.DataFrame) -> pd.DataFrame:
        p = p_df.copy()
        if p.empty:
            return pd.DataFrame()

        p["shipmentname"] = p["shipmentname"].astype(str).str.strip()
        p = p[p["shipmentname"] != ""].copy()
        if p.empty:
            return pd.DataFrame()

        p["itemcode"] = p["itemcode"].apply(_norm_code).astype(str).str.strip()
        p["itemname"] = p["itemname"].astype(str)
        p["combinedate"] = pd.to_datetime(p["combinedate"], errors="coerce").dt.floor("D")
        p["quantity"] = pd.to_numeric(p["quantity"], errors="coerce").fillna(0.0)
        p["cost"] = pd.to_numeric(p["cost"], errors="coerce").fillna(0.0)

        p = p[p["combinedate"].notna()].copy()
        if p.empty:
            return pd.DataFrame()

        grp = (
            p.groupby(["shipmentname", "itemcode", "itemname", "combinedate"], as_index=False)
            .agg(
                initial_qty=("quantity", "sum"),
                unit_cost=("cost", "mean"),
            )
        )

        grp["batch_id"] = (
            grp["shipmentname"].astype(str)
            + " | "
            + grp["itemcode"].astype(str)
            + " | "
            + grp["combinedate"].dt.strftime("%Y-%m-%d")
        )

        grp["batch_cost"] = grp["initial_qty"] * grp["unit_cost"]
        grp = grp.sort_values(["itemcode", "combinedate", "shipmentname"]).reset_index(drop=True)
        return grp

    def _build_onhand_before_for_selected(stock_mv_df: pd.DataFrame, selected_batches: pd.DataFrame) -> pd.Series:
        if stock_mv_df is None or stock_mv_df.empty or selected_batches.empty:
            return pd.Series([0.0] * len(selected_batches), index=selected_batches.index)

        sm = stock_mv_df.copy()
        sm["zid"] = sm["zid"].astype(str).str.strip()
        sm = sm[sm["zid"].isin(["100001", "100009"])].copy()
        if sm.empty:
            return pd.Series([0.0] * len(selected_batches), index=selected_batches.index)

        sm["itemcode"] = sm["itemcode"].apply(_norm_code).astype(str).str.strip()
        sm["date"] = pd.to_datetime(sm["date"], errors="coerce").dt.floor("D")
        sm = sm[sm["date"].notna()].copy()
        sm["stockqty"] = pd.to_numeric(sm.get("stockqty", 0), errors="coerce").fillna(0.0)

        daily = (
            sm.groupby(["date", "itemcode"], as_index=False)["stockqty"]
            .sum()
            .sort_values(["date", "itemcode"])
            .reset_index(drop=True)
        )
        daily["onhand_qty"] = daily.groupby("itemcode")["stockqty"].cumsum()

        q = selected_batches[["itemcode", "combinedate"]].copy()
        q["qdate"] = q["combinedate"] - pd.Timedelta(nanoseconds=1)
        q = q.reset_index(drop=False).rename(columns={"index": "_rowid"})
        q["itemcode"] = q["itemcode"].astype(str).str.strip()

        daily["itemcode"] = daily["itemcode"].astype(str).str.strip()
        q = q.sort_values(["qdate", "itemcode"]).reset_index(drop=True)
        daily = daily.sort_values(["date", "itemcode"]).reset_index(drop=True)

        m = pd.merge_asof(
            q,
            daily,
            left_on="qdate",
            right_on="date",
            by="itemcode",
            direction="backward",
            allow_exact_matches=True,
        )

        m = m[["_rowid", "onhand_qty"]].rename(columns={"onhand_qty": "onhand_before"})
        m["onhand_before"] = pd.to_numeric(m["onhand_before"], errors="coerce").fillna(0.0)
        m["_rowid"] = pd.to_numeric(m["_rowid"], errors="coerce")

        out = m.set_index("_rowid").reindex(range(len(selected_batches)))["onhand_before"].fillna(0.0)
        out.index = selected_batches.index
        return out

    def _resolve_sales_value_col(sdf: pd.DataFrame) -> str:
        if sdf is None or not isinstance(sdf, pd.DataFrame) or sdf.empty:
            return ""
        if "totalsales" in sdf.columns:
            return "totalsales"
        if "altsales" in sdf.columns:
            return "altsales"
        return ""

    def _build_daily_events(s_df: pd.DataFrame, r_df: pd.DataFrame, target_zid: str) -> pd.DataFrame:
        # ---- sales daily
        s = s_df.copy() if isinstance(s_df, pd.DataFrame) else pd.DataFrame()
        if not s.empty:
            s["zid"] = s["zid"].astype(str).str.strip()
            s = s[s["zid"] == str(target_zid).strip()].copy()
            s["itemcode"] = s["itemcode"].apply(_norm_code).astype(str).str.strip()
            s["d"] = pd.to_datetime(s["date"], errors="coerce").dt.floor("D")
            s = s[s["d"].notna()].copy()
            s["quantity"] = pd.to_numeric(s.get("quantity", 0), errors="coerce").fillna(0.0)

            val_col = _resolve_sales_value_col(s)
            if val_col == "totalsales":
                s["totalsales"] = pd.to_numeric(s.get("totalsales", 0), errors="coerce").fillna(0.0)
                s["_rev"] = s["totalsales"]
            elif val_col == "altsales":
                s["altsales"] = pd.to_numeric(s.get("altsales", 0), errors="coerce").fillna(0.0)
                s["_rev"] = s["altsales"]
            else:
                s["_rev"] = 0.0

            s_daily = (
                s.groupby(["itemcode", "d"], as_index=False)
                .agg(
                    sales_qty=("quantity", "sum"),
                    sales_rev=("_rev", "sum"),
                )
            )
        else:
            s_daily = pd.DataFrame(columns=["itemcode", "d", "sales_qty", "sales_rev"])

        # ---- returns daily
        r = r_df.copy() if isinstance(r_df, pd.DataFrame) else pd.DataFrame()
        if not r.empty:
            r["zid"] = r["zid"].astype(str).str.strip()
            r = r[r["zid"] == str(target_zid).strip()].copy()
            r["itemcode"] = r["itemcode"].apply(_norm_code).astype(str).str.strip()
            r_dcol = "date" if "date" in r.columns else ("xdate" if "xdate" in r.columns else None)

            if r_dcol is not None:
                r["d"] = pd.to_datetime(r[r_dcol], errors="coerce").dt.floor("D")
                r = r[r["d"].notna()].copy()
                r["returnqty"] = pd.to_numeric(r.get("returnqty", 0), errors="coerce").fillna(0.0)

                r_daily = (
                    r.groupby(["itemcode", "d"], as_index=False)
                    .agg(return_qty=("returnqty", "sum"))
                )
            else:
                r_daily = pd.DataFrame(columns=["itemcode", "d", "return_qty"])
        else:
            r_daily = pd.DataFrame(columns=["itemcode", "d", "return_qty"])

        ev = pd.merge(s_daily, r_daily, on=["itemcode", "d"], how="outer").fillna(0.0)
        if ev.empty:
            return ev

        ev["sales_qty"] = pd.to_numeric(ev["sales_qty"], errors="coerce").fillna(0.0)
        ev["sales_rev"] = pd.to_numeric(ev["sales_rev"], errors="coerce").fillna(0.0)
        ev["return_qty"] = pd.to_numeric(ev["return_qty"], errors="coerce").fillna(0.0)

        ev = ev.sort_values(["itemcode", "d"]).reset_index(drop=True)
        return ev

    # ----------------------------------------------------------
    # 1) Prepare all batches and selected shipment batches
    # ----------------------------------------------------------
    all_batches = _prep_all_purchase_batches(purchase_df)
    if all_batches.empty:
        return pd.DataFrame()

    selected_batches = all_batches[all_batches["shipmentname"].astype(str) == str(shipmentname).strip()].copy()
    if selected_batches.empty:
        return pd.DataFrame()

    selected_batches = selected_batches.sort_values(["itemcode", "combinedate", "shipmentname"]).reset_index(drop=True)

    # display threshold / baseline from stock ledger
    selected_batches["onhand_before"] = _build_onhand_before_for_selected(stock_movement_df, selected_batches).astype(float)
    selected_batches["threshold_qty"] = selected_batches["onhand_before"].astype(float)

    # ----------------------------------------------------------
    # 2) Build daily sales events and FIFO allocate across ALL batches
    #    for the selected shipment itemcodes
    # ----------------------------------------------------------
    target_itemcodes = set(selected_batches["itemcode"].astype(str).tolist())

    alloc_src_batches = all_batches[all_batches["itemcode"].astype(str).isin(target_itemcodes)].copy()
    alloc_src_batches = alloc_src_batches.sort_values(["itemcode", "combinedate", "shipmentname"]).reset_index(drop=True)

    def _fifo_allocate_batches(all_batches: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """
        FIFO allocation across repeated shipments of the same SKU.

        Correct rule:
        - Only sales on/after a batch's combinedate can affect that batch.
        - Earlier batches get consumed first.
        - The quantity available for a batch is based on:
            sales after batch date
            minus older stock still open at batch date
        NOT minus total older purchases.
        """

        EPS = 1e-9

        batches = all_batches.copy()

        batches["sold_qty"] = 0.0
        batches["sold_revenue"] = 0.0
        batches["batch_end_date"] = pd.NaT
        batches["remaining_qty"] = pd.to_numeric(
            batches.get("initial_qty", 0), errors="coerce"
        ).fillna(0.0)
        batches["is_closed"] = False

        if batches.empty or events is None or events.empty:
            return batches

        ev = events.copy()
        ev["itemcode"] = ev["itemcode"].astype(str).str.strip()
        ev["d"] = pd.to_datetime(ev["d"], errors="coerce").dt.floor("D")
        ev = ev[ev["d"].notna()].copy()

        ev["sales_qty"] = pd.to_numeric(ev.get("sales_qty", 0), errors="coerce").fillna(0.0)
        ev["sales_rev"] = pd.to_numeric(ev.get("sales_rev", 0), errors="coerce").fillna(0.0)
        ev["return_qty"] = pd.to_numeric(ev.get("return_qty", 0), errors="coerce").fillna(0.0)
        ev["net_qty"] = ev["sales_qty"] - ev["return_qty"]

        ev = ev.sort_values(["itemcode", "d"]).reset_index(drop=True)

        batches["itemcode"] = batches["itemcode"].astype(str).str.strip()
        batches["combinedate"] = pd.to_datetime(batches["combinedate"], errors="coerce").dt.floor("D")
        batches["initial_qty"] = pd.to_numeric(batches.get("initial_qty", 0), errors="coerce").fillna(0.0)

        for code, bgrp in batches.groupby("itemcode", sort=False):
            sales = ev[ev["itemcode"] == code].copy()
            if sales.empty:
                continue

            sales = sales.sort_values("d").reset_index(drop=True)
            sales["cum_net"] = sales["net_qty"].cumsum()

            batch_positions = bgrp.sort_values(["combinedate", "shipmentname"]).index.tolist()

            for pos in batch_positions:
                batch_qty = float(batches.at[pos, "initial_qty"])
                batch_date = pd.Timestamp(batches.at[pos, "combinedate"]).floor("D")

                # older purchase qty strictly before this batch
                older_batches = batches[
                    (batches["itemcode"] == code) &
                    (batches["combinedate"] < batch_date)
                ].copy()

                older_purchase_qty = float(older_batches["initial_qty"].sum()) if not older_batches.empty else 0.0

                # net sales strictly before this batch date
                sales_before = sales[sales["d"] < batch_date].copy()
                net_sales_before = float(sales_before["net_qty"].sum()) if not sales_before.empty else 0.0

                # older stock still open when this batch arrives
                older_open_at_batch = max(0.0, older_purchase_qty - net_sales_before)

                # sales only after this batch arrived
                sales_after_batch = sales[sales["d"] >= batch_date].copy()
                if sales_after_batch.empty:
                    continue

                total_sales_after_batch = float(sales_after_batch["net_qty"].sum())

                # this batch can only consume sales beyond older stock still open
                available_for_batch = max(0.0, total_sales_after_batch - older_open_at_batch)

                sold = max(0.0, min(batch_qty, available_for_batch))

                if sold <= EPS:
                    continue

                # average realized price using sales after batch date
                total_sales_qty_after_batch = float(sales_after_batch["sales_qty"].sum())
                total_sales_rev_after_batch = float(sales_after_batch["sales_rev"].sum())

                avg_price = (
                    total_sales_rev_after_batch / total_sales_qty_after_batch
                    if total_sales_qty_after_batch > EPS else 0.0
                )
                revenue = sold * avg_price

                batches.at[pos, "sold_qty"] = sold
                batches.at[pos, "sold_revenue"] = revenue

                # batch end date:
                # first day where cumulative net sales since batch date
                # exceeds older_open_at_batch + sold
                # batch_end_date should only exist if the batch is fully depleted
                if sold >= (batch_qty - EPS):
                    sales_after_batch = sales_after_batch.copy()
                    sales_after_batch["cum_qty_from_batch"] = sales_after_batch["net_qty"].cumsum()

                    # To close this batch, sales after batch date must first clear older open stock
                    # and then consume the FULL batch quantity
                    close_target = older_open_at_batch + batch_qty

                    hit = sales_after_batch[
                        sales_after_batch["cum_qty_from_batch"] >= close_target
                    ]

                    if not hit.empty:
                        close_dt = pd.Timestamp(hit.iloc[0]["d"]).floor("D")
                        if close_dt >= batch_date:
                            batches.at[pos, "batch_end_date"] = close_dt
                    else:
                        batches.at[pos, "batch_end_date"] = pd.NaT
                else:
                    batches.at[pos, "batch_end_date"] = pd.NaT
                    
        batches["sold_qty"] = pd.to_numeric(batches["sold_qty"], errors="coerce").fillna(0.0)
        batches["sold_revenue"] = pd.to_numeric(batches["sold_revenue"], errors="coerce").fillna(0.0)

        batches["remaining_qty"] = (
            batches["initial_qty"].astype(float) - batches["sold_qty"].astype(float)
        ).clip(lower=0.0)

        batches["remaining_qty"] = np.where(
            batches["remaining_qty"] < EPS,
            0.0,
            batches["remaining_qty"]
        )

        batches["batch_end_date"] = pd.to_datetime(batches["batch_end_date"], errors="coerce")

        batches.loc[
            batches["batch_end_date"].notna() & (batches["batch_end_date"] < batches["combinedate"]),
            "batch_end_date"
        ] = pd.NaT

        batches["is_closed"] = batches["remaining_qty"] <= EPS

        return batches

    events = _build_daily_events(sales_df, returns_df, zid_deplete)
    if not events.empty:
        events = events[events["itemcode"].astype(str).isin(target_itemcodes)].copy()
        events = events.sort_values(["itemcode", "d"]).reset_index(drop=True)

    print("Events rows:", len(events))
    print("Event SKUs:", events["itemcode"].unique()[:10])
    print("Batch SKUs:", list(target_itemcodes)[:10])
    alloc_src_batches = _fifo_allocate_batches(alloc_src_batches, events)

    alloc_src_batches["batch_end_date"] = pd.to_datetime(alloc_src_batches["batch_end_date"], errors="coerce")
    alloc_src_batches.loc[
        alloc_src_batches["batch_end_date"] < alloc_src_batches["combinedate"],
        "batch_end_date"
    ] = pd.NaT

    # remaining qty on all batches
    alloc_src_batches["sold_qty"] = pd.to_numeric(alloc_src_batches["sold_qty"], errors="coerce").fillna(0.0)
    alloc_src_batches["sold_revenue"] = pd.to_numeric(alloc_src_batches["sold_revenue"], errors="coerce").fillna(0.0)
    alloc_src_batches["remaining_qty"] = np.clip(
        alloc_src_batches["initial_qty"].astype(float) - alloc_src_batches["sold_qty"].astype(float),
        0.0,
        None,
    )
    alloc_src_batches["remaining_qty"] = np.where(alloc_src_batches["remaining_qty"] < EPS, 0.0, alloc_src_batches["remaining_qty"])
    alloc_src_batches["is_closed"] = alloc_src_batches["remaining_qty"] <= EPS

    # pull back only selected shipment rows
    alloc_cols = ["batch_id", "sold_qty", "sold_revenue", "remaining_qty", "batch_end_date", "is_closed"]
    df0 = selected_batches.merge(
        alloc_src_batches[alloc_cols],
        on="batch_id",
        how="left",
    )

    for c in ["sold_qty", "sold_revenue", "remaining_qty"]:
        df0[c] = pd.to_numeric(df0[c], errors="coerce").fillna(0.0)

    df0["is_closed"] = df0["is_closed"].fillna(False)
    df0["batch_end_date"] = pd.to_datetime(df0["batch_end_date"], errors="coerce")

    # ----------------------------------------------------------
    # 3) Derived profitability fields
    # ----------------------------------------------------------
    # -----------------------------------------
    # Price logic:
    # 1) use realized batch avg price if sold_qty > 0
    # 2) else fallback to historical SKU avg price from sales_df
    # 3) else fallback to unit_cost
    # -----------------------------------------

    # batch realized avg price
    df0["avg_price"] = np.where(df0["sold_qty"] > EPS, df0["sold_revenue"] / df0["sold_qty"], np.nan)
    df0["avg_price"] = pd.to_numeric(df0["avg_price"], errors="coerce")

    # build historical SKU avg price fallback from sales_df
    hist_price_map = {}

    if isinstance(sales_df, pd.DataFrame) and not sales_df.empty:
        s_hist = sales_df.copy()
        s_hist["zid"] = s_hist["zid"].astype(str).str.strip()
        s_hist = s_hist[s_hist["zid"] == str(zid_deplete).strip()].copy()

        if not s_hist.empty:
            s_hist["itemcode"] = s_hist["itemcode"].apply(_norm_code).astype(str).str.strip()
            s_hist["quantity"] = pd.to_numeric(s_hist.get("quantity", 0), errors="coerce").fillna(0.0)

            if "totalsales" in s_hist.columns:
                s_hist["sales_value"] = pd.to_numeric(s_hist.get("totalsales", 0), errors="coerce").fillna(0.0)
            elif "altsales" in s_hist.columns:
                s_hist["sales_value"] = pd.to_numeric(s_hist.get("altsales", 0), errors="coerce").fillna(0.0)
            else:
                s_hist["sales_value"] = 0.0

            s_hist_grp = (
                s_hist.groupby("itemcode", as_index=False)
                .agg(total_qty=("quantity", "sum"), total_rev=("sales_value", "sum"))
            )

            s_hist_grp["hist_avg_price"] = np.where(
                s_hist_grp["total_qty"] > EPS,
                s_hist_grp["total_rev"] / s_hist_grp["total_qty"],
                np.nan
            )

            hist_price_map = dict(
                zip(
                    s_hist_grp["itemcode"].astype(str),
                    pd.to_numeric(s_hist_grp["hist_avg_price"], errors="coerce")
                )
            )

    # apply historical fallback
    df0["hist_avg_price"] = df0["itemcode"].astype(str).map(hist_price_map)

    # final avg_price fallback chain:
    # realized batch avg -> historical avg -> unit_cost
    df0["avg_price"] = df0["avg_price"].fillna(df0["hist_avg_price"])
    df0["avg_price"] = df0["avg_price"].fillna(df0["unit_cost"])
    df0["avg_price"] = pd.to_numeric(df0["avg_price"], errors="coerce").fillna(0.0)

    df0["scenario_price"] = df0["avg_price"] * (1.0 - float(discount_pct) / 100.0)

    df0["realized_cogs"] = df0["sold_qty"].astype(float) * df0["unit_cost"].astype(float)
    df0["realized_gm"] = df0["sold_revenue"].astype(float) - df0["realized_cogs"].astype(float)

    df0["remaining_cost_value"] = df0["remaining_qty"].astype(float) * df0["unit_cost"].astype(float)
    df0["proj_remaining_revenue"] = df0["remaining_qty"].astype(float) * df0["scenario_price"].astype(float)
    df0["proj_remaining_gm"] = df0["proj_remaining_revenue"].astype(float) - df0["remaining_cost_value"].astype(float)

    df0.drop(columns=["hist_avg_price"], errors="ignore", inplace=True)

    # ----------------------------------------------------------
    # 4) Activity timing / velocity
    # ----------------------------------------------------------
    end_eff = df0["batch_end_date"].where(df0["batch_end_date"].notna(), as_of)

    df0["days_active"] = ((end_eff - df0["combinedate"]).dt.days + 1).clip(lower=1).astype(int)

    df0["velocity"] = np.where(df0["days_active"] > 0, df0["sold_qty"] / df0["days_active"], 0.0)
    df0["velocity"] = pd.to_numeric(df0["velocity"], errors="coerce").fillna(0.0)

    # user rule:
    # velocity_used = velocity_sku if sales > 0, else fallback floor 0.02
    df0["velocity_used"] = np.where(df0["sold_qty"] > EPS, df0["velocity"], 0.02)
    df0["velocity_used"] = pd.to_numeric(df0["velocity_used"], errors="coerce").fillna(0.02)

    df0["days_to_clear"] = np.where(
        df0["velocity_used"] > EPS,
        df0["remaining_qty"] / df0["velocity_used"],
        730.0,
    )
    df0["days_to_clear"] = pd.to_numeric(df0["days_to_clear"], errors="coerce").fillna(730.0)
    df0["days_to_clear"] = df0["days_to_clear"].clip(lower=0.0, upper=730.0)

    df0["batch_age_days"] = ((as_of - df0["combinedate"]).dt.days).astype(int)

    # ----------------------------------------------------------
    # 5) Overhead logic
    # ----------------------------------------------------------
    total_sold_revenue = float(df0["sold_revenue"].sum()) if "sold_revenue" in df0.columns else 0.0

    vat_overhead_value = (float(vat_pct) / 100.0) * max(0.0, total_sold_revenue)
    manual_overhead_value = float(manual_overhead_value or 0.0)
    total_overhead_pool = float(shipment_overhead_total or 0.0) + float(vat_overhead_value) + float(manual_overhead_value)

    if total_sold_revenue > EPS:
        share_real = (df0["sold_revenue"] / total_sold_revenue).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        denom = float(df0["realized_cogs"].sum())
        share_real = (df0["realized_cogs"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0) if denom > EPS else 0.0

    # D0 = initial daily overhead allocation across the shipment horizon
    days_elapsed = int(df0["days_active"].max()) if "days_active" in df0.columns else 1
    days_elapsed = max(1, days_elapsed)
    D0 = total_overhead_pool / float(days_elapsed)

    # realized overhead: stop at batch_end_date if closed, else run to as_of
    realized_end_eff = df0["batch_end_date"].where(df0["batch_end_date"].notna(), as_of)
    realized_days = ((realized_end_eff - df0["combinedate"]).dt.days + 1).clip(lower=1).astype(float)

    df0["overhead_realized"] = D0 * realized_days * share_real
    df0["net_profit_realized"] = df0["realized_gm"] - df0["overhead_realized"]

    # projected overhead
    total_proj_remaining_revenue = float(df0["proj_remaining_revenue"].sum())
    if total_proj_remaining_revenue > EPS:
        share_rem = (df0["proj_remaining_revenue"] / total_proj_remaining_revenue).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        denom = float(df0["remaining_cost_value"].sum())
        share_rem = (df0["remaining_cost_value"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0) if denom > EPS else 0.0

    dclear = df0["days_to_clear"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dclear = dclear.clip(lower=0.0, upper=730.0)

    # user formula:
    # overhead_projected_sku = D0 * (0.97)^(days_to_clear/60) * days_to_clear * remaining_share
    decay_factor = np.power(0.97, dclear / 60.0)
    df0["overhead_projected"] = D0 * decay_factor * dclear * share_rem

    df0["Proj_remaining_profit"] = df0["proj_remaining_gm"] - df0["overhead_projected"]
    df0["proj_final_profit"] = df0["net_profit_realized"] + df0["Proj_remaining_profit"]

    # ----------------------------------------------------------
    # 6) Final output columns
    # ----------------------------------------------------------
    df0.drop(columns=["velocity_used"], errors="ignore", inplace=True)

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


@st.cache_data(ttl=3600, show_spinner=False)
def build_time_to_sell_percentiles(
    purchase_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    min_qty: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-itemcode P50/P75/P90/P95 days-to-sell from historical closed batches.

    Depletion: for each closed batch (itemcode, combinedate, initial_qty), find the
    first date after combinedate where cumulative 100001 sales >= prior_cumulative +
    initial_qty.  days_to_sell = that_date - combinedate.

    Batches with initial_qty < min_qty are excluded — small test orders skew percentiles.
    Cross-ZID batches (100001 + 100009) for the same item and date are combined into one
    logical arrival since their itemcodes resolve to the same 100001-space code.

    Returns:
        pct_df   — one row per itemcode: P50/P75/P90/P95 (days), n, flag
        detail_df — one row per batch: itemcode, combinedate, initial_qty,
                    days_to_sell (None = not yet depleted), est_end_date
    """
    empty_pct = pd.DataFrame(
        columns=["itemcode", "itemname", "itemgroup", "P50", "P75", "P90", "P95", "n", "flag"]
    )
    empty_detail = pd.DataFrame(
        columns=["itemcode", "itemname", "itemgroup", "combinedate", "initial_qty", "days_to_sell", "est_end_date", "status"]
    )

    if purchase_df is None or purchase_df.empty:
        return empty_pct, empty_detail

    # --- closed batches ---
    closed = purchase_df[purchase_df["status"] != "1-Open"].copy()
    closed["itemcode"] = closed["itemcode"].astype(str).str.strip()
    closed["combinedate"] = pd.to_datetime(closed["combinedate"], errors="coerce").dt.floor("D")
    closed["quantity"] = pd.to_numeric(closed["quantity"], errors="coerce").fillna(0.0)
    closed = closed[closed["combinedate"].notna() & (closed["quantity"] > 0)].copy()

    if closed.empty:
        return empty_pct, empty_detail

    # Aggregate cross-ZID arrivals of the same item on the same date into one batch
    agg_spec: dict = {"initial_qty": ("quantity", "sum"), "itemname": ("itemname", "first")}
    if "itemgroup" in closed.columns:
        agg_spec["itemgroup"] = ("itemgroup", "first")
    closed = closed.groupby(["itemcode", "combinedate"], as_index=False).agg(**agg_spec)
    if "itemgroup" not in closed.columns:
        closed["itemgroup"] = ""

    # Exclude small batches — not representative of typical sell-through behaviour
    closed = closed[closed["initial_qty"] >= min_qty].copy()
    if closed.empty:
        return empty_pct, empty_detail

    # --- cumulative sales series per itemcode (100001, already filtered at query layer) ---
    sales_lookup: dict = {}
    if sales_df is not None and not sales_df.empty:
        s = sales_df[["itemcode", "date", "quantity"]].copy()
        s["itemcode"] = s["itemcode"].astype(str).str.strip()
        s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.floor("D")
        s["quantity"] = pd.to_numeric(s["quantity"], errors="coerce").fillna(0.0).clip(lower=0.0)
        s = s[s["date"].notna()].groupby(["itemcode", "date"], as_index=False)["quantity"].sum()
        s = s.sort_values(["itemcode", "date"]).reset_index(drop=True)
        s["cum_sales"] = s.groupby("itemcode")["quantity"].cumsum()
        for code, grp in s.groupby("itemcode"):
            sales_lookup[code] = (
                grp["date"].values.astype("datetime64[ns]"),
                grp["cum_sales"].values.astype(float),
            )

    # --- per-batch depletion via numpy searchsorted (O(log n) per batch) ---
    records: list = []
    for _, row in closed.iterrows():
        code = str(row["itemcode"])
        combinedate: pd.Timestamp = row["combinedate"]
        initial_qty: float = float(row["initial_qty"])
        meta = {"itemcode": code, "itemname": row["itemname"], "itemgroup": row.get("itemgroup", ""),
                "combinedate": combinedate, "initial_qty": initial_qty}

        if code not in sales_lookup:
            records.append({**meta, "days_to_sell": None, "est_end_date": pd.NaT, "status": "No sales data"})
            continue

        dates, cumvals = sales_lookup[code]
        combine_np = np.datetime64(combinedate, "ns")

        # prior cumulative: last value strictly before combinedate
        idx_before = int(np.searchsorted(dates, combine_np, side="left")) - 1
        prior_cum = float(cumvals[idx_before]) if idx_before >= 0 else 0.0
        target_cum = prior_cum + initial_qty

        # slice from combinedate onwards
        idx_start = int(np.searchsorted(dates, combine_np, side="left"))
        if idx_start >= len(dates):
            records.append({**meta, "days_to_sell": None, "est_end_date": pd.NaT, "status": "Not depleted"})
            continue

        future_cum = cumvals[idx_start:]
        future_dates = dates[idx_start:]
        hit = int(np.searchsorted(future_cum, target_cum - 1e-6, side="left"))

        if hit < len(future_cum) and future_cum[hit] >= target_cum - 1e-6:
            end_date = pd.Timestamp(future_dates[hit])
            days = max(0, (end_date - combinedate).days)
            records.append({**meta, "days_to_sell": days, "est_end_date": end_date, "status": "Depleted"})
        else:
            records.append({**meta, "days_to_sell": None, "est_end_date": pd.NaT, "status": "Not depleted"})

    if not records:
        return empty_pct, empty_detail

    rec_df = pd.DataFrame(records)

    # --- percentiles per itemcode ---
    out_rows: list = []
    for code, grp in rec_df.groupby("itemcode"):
        completed = grp["days_to_sell"].dropna().astype(float).values
        n = int(len(completed))
        p90 = float(np.percentile(completed, 90)) if n > 0 else None

        if n == 0:
            flag = "No data"
        elif n < 5:
            flag = "Low confidence"
        elif p90 is not None and p90 > 120:
            flag = "Dead stock"
        else:
            flag = ""

        out_rows.append({
            "itemcode": code,
            "itemname": grp["itemname"].iloc[0],
            "itemgroup": grp["itemgroup"].iloc[0],
            "P50": int(round(float(np.percentile(completed, 50)))) if n > 0 else None,
            "P75": int(round(float(np.percentile(completed, 75)))) if n > 0 else None,
            "P90": int(round(p90)) if p90 is not None else None,
            "P95": int(round(float(np.percentile(completed, 95)))) if n > 0 else None,
            "n": n,
            "flag": flag,
        })

    pct_df = pd.DataFrame(out_rows).sort_values("itemcode").reset_index(drop=True)
    detail_df = rec_df.sort_values(["itemcode", "combinedate"]).reset_index(drop=True)
    return pct_df, detail_df