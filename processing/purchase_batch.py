from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from processing import common
from datetime import datetime
import streamlit as st
from datetime import date as _date
from typing import Dict, List, Tuple, Optional, Any

_log = logging.getLogger(__name__)


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

    # When baseline < 0, pre-existing backorders were filled the moment this shipment
    # arrived — those units were immediately consumed. Reduce the effective bin size by
    # the backorder depth so remaining_qty can't be inflated by a negative subtrahend.
    # We keep original_ship_qty for sold_eff so that backorder-fill is counted as sold.
    original_ship_qty = ship_qty
    if baseline < 0.0:
        ship_qty = max(0.0, ship_qty + baseline)  # units available for forward tracking
        baseline = 0.0

    # slice after start
    sku_after = sku[(sku["date"] >= start) & (sku["date"] <= as_of)].copy()
    if sku_after.empty:
        # no movements after start, assume still open
        return (pd.NaT, 0.0, original_ship_qty, baseline, baseline)

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

    # sold = original purchased qty minus what's still remaining (includes backorder fill)
    sold_eff = original_ship_qty - remaining_eff
    sold_eff = max(0.0, min(original_ship_qty, sold_eff))

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
    movements_df: pd.DataFrame,
    sales_df: pd.DataFrame,
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
    # 1) Stock movement prep — from mv_imtrn_movements (net daily per item)
    # ----------------------------
    sm = movements_df.copy() if isinstance(movements_df, pd.DataFrame) else pd.DataFrame()
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
    # mv_imtrn_movements uses txn_date; net_qty = xqty * xsign (signed stock delta)
    sm["date"] = pd.to_datetime(sm["txn_date"], errors="coerce").dt.floor("D")
    sm = sm[sm["date"].notna()].copy()
    sm["net_qty"] = pd.to_numeric(sm.get("net_qty", 0), errors="coerce").fillna(0.0)

    def _build_daily_onhand(sm_in: pd.DataFrame) -> pd.DataFrame:
        """Cumulative onhand from mv_imtrn_movements net daily movements."""
        d = (
            sm_in.groupby(["date", "itemcode"], as_index=False)["net_qty"]
            .sum()
            .copy()
        )
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.floor("D")
        d = d[d["date"].notna()].copy()
        d = d.sort_values(["date", "itemcode"]).reset_index(drop=True)
        d["onhand_qty"] = d.groupby("itemcode")["net_qty"].cumsum()
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

    # Derive DO-- sales qty and cust_return qty from mv_imtrn_movements
    # for the reconcile window — captures all physical movements, not just sales orders.
    _do_rows = sm[sm["txn_type"] == "sale"].copy() if "txn_type" in sm.columns else pd.DataFrame()
    if not _do_rows.empty:
        _do_rows = _do_rows.copy()
        _do_rows["quantity"] = pd.to_numeric(_do_rows.get("xqty", 0), errors="coerce").fillna(0.0)
    _ret_rows = sm[sm["txn_type"] == "cust_return"].copy() if "txn_type" in sm.columns else pd.DataFrame()
    if not _ret_rows.empty:
        _ret_rows = _ret_rows.copy()
        _ret_rows["returnqty"] = pd.to_numeric(_ret_rows.get("xqty", 0), errors="coerce").fillna(0.0)

    base["sales_qty_window"] = _window_qty(_do_rows if not _do_rows.empty else sales_df, "quantity")
    base["return_qty_window"] = _window_qty(_ret_rows, "returnqty")

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



# ═══════════════════════════════════════════════════════════════════════════════
# Raw-ledger FIFO helpers
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Raw-ledger FIFO helpers
# Replace the combinedate-grouped approach with individual IGRN receipt lots.
# ═══════════════════════════════════════════════════════════════════════════════

def _build_raw_lots(
    movements_df: pd.DataFrame,
    purchase_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join individual IGRN purchase receipts from mv_imtrn_movements with
    shipmentname and unit_cost from mv_purchase_batches.

    Both 100001 (direct imports) and 100009 (cross-ZID) IGRN rows are included.
    100009 item codes are already mapped to 100001 codes in the MV via
    caitem.xdrawing, so itemcode is consistent across both ZIDs.

    Each row = one FIFO lot: txn_date (actual receipt date), net_qty,
    shipmentname, unit_cost.  No combinedate grouping — no phantom remaining.

    Returns columns: zid, xdocnum, itemcode, itemname, shipmentname,
                     lot_date, lot_qty, unit_cost
    """
    if (movements_df is None or not isinstance(movements_df, pd.DataFrame)
            or movements_df.empty
            or purchase_df is None or not isinstance(purchase_df, pd.DataFrame)
            or purchase_df.empty):
        return pd.DataFrame()

    mv = movements_df.copy()
    mv["zid"]      = mv["zid"].astype(str).str.strip()
    mv["xdocnum"]  = mv["xdocnum"].astype(str).str.strip()
    mv["itemcode"] = mv["itemcode"].apply(_norm_code).astype(str).str.strip()
    mv["lot_date"] = pd.to_datetime(mv["txn_date"], errors="coerce").dt.floor("D")
    mv["lot_qty"]  = pd.to_numeric(mv["net_qty"], errors="coerce").fillna(0.0)

    mv = mv[
        (mv["txn_type"] == "purchase") &
        mv["xdocnum"].str.startswith("IGRN") &
        mv["lot_date"].notna() &
        (mv["lot_qty"] > 0)
    ].copy()

    if mv.empty:
        return pd.DataFrame()

    # Raw purchase_df: may alias unit_cost as 'cost' depending on query
    pb = purchase_df.copy()
    pb["zid"]        = pb["zid"].astype(str).str.strip()
    pb["grnvoucher"] = pb["grnvoucher"].astype(str).str.strip()
    pb["itemcode"]   = pb["itemcode"].apply(_norm_code).astype(str).str.strip()
    _cost_col = "unit_cost" if "unit_cost" in pb.columns else "cost"
    pb = pb.rename(columns={_cost_col: "unit_cost"})
    pb = pb[pb["grnvoucher"].str.startswith("IGRN")].copy()
    pb = pb[["zid", "grnvoucher", "itemcode", "itemname",
             "shipmentname", "unit_cost"]].drop_duplicates()

    # Drop any itemname column from mv before the join so it doesn't collide
    # with pb's itemname (mv_imtrn_movements has no itemname column in production,
    # but be defensive in case a caller passes enriched data).
    mv = mv.drop(columns=[c for c in ["itemname"] if c in mv.columns], errors="ignore")

    # Join: (zid, xdocnum=grnvoucher, itemcode) — all three already xdrawing-resolved in MV
    lots = mv.merge(
        pb,
        left_on=["zid", "xdocnum", "itemcode"],
        right_on=["zid", "grnvoucher", "itemcode"],
        how="inner",
    )

    if lots.empty:
        return pd.DataFrame()

    lots["unit_cost"] = pd.to_numeric(lots["unit_cost"], errors="coerce").fillna(0.0)
    return lots[
        ["zid", "xdocnum", "itemcode", "itemname", "shipmentname",
         "lot_date", "lot_qty", "unit_cost"]
    ].reset_index(drop=True)


def _build_dep_daily(
    movements_df: pd.DataFrame,
    dep_zid: str = "100001",
) -> pd.DataFrame:
    """Daily depletion totals from mv_imtrn_movements, filtered to dep_zid only.

    Depletions = txn_type in ('sale', 'issue') for dep_zid only.

    100009 IS-- (MO-- manufacturing) and 100009 DO-- (inter-company transfers
    to 100001) are excluded by the zid filter.  In the combined entity view
    these cancel with 100001 GRN- receipts and 100009 production outputs
    respectively, so omitting them gives the correct net depletion.

    Returns columns: itemcode, dep_date, dep_qty
    """
    if (movements_df is None or not isinstance(movements_df, pd.DataFrame)
            or movements_df.empty):
        return pd.DataFrame(columns=["itemcode", "dep_date", "dep_qty"])

    mv = movements_df.copy()
    mv["zid"]      = mv["zid"].astype(str).str.strip()
    mv             = mv[mv["zid"] == str(dep_zid).strip()].copy()
    mv["itemcode"] = mv["itemcode"].apply(_norm_code).astype(str).str.strip()
    mv["dep_date"] = pd.to_datetime(mv["txn_date"], errors="coerce").dt.floor("D")
    mv             = mv[mv["txn_type"].isin(["sale", "issue"])
                        & mv["dep_date"].notna()].copy()
    mv["dep_qty"]  = pd.to_numeric(mv["xqty"], errors="coerce").fillna(0.0)

    dep = (
        mv.groupby(["itemcode", "dep_date"], as_index=False)["dep_qty"].sum()
    )
    return dep.sort_values(["itemcode", "dep_date"]).reset_index(drop=True)


def _fifo_lots(lots: pd.DataFrame, deps: pd.DataFrame) -> pd.DataFrame:
    """Greedy oldest-first FIFO depletion engine.

    Parameters
    ----------
    lots : DataFrame — columns [itemcode, lot_date, lot_qty, …]
           All other columns are preserved in the output unchanged.
    deps : DataFrame — columns [itemcode, dep_date, dep_qty]

    Returns lots with added columns:
        sold_qty       — units consumed from this lot by FIFO depletions
        remaining_qty  — lot_qty − sold_qty (≥ 0)
        is_closed      — remaining_qty < EPS
        batch_end_date — first dep_date on which this lot became fully
                         depleted; NaT if still has remaining stock

    Key properties
    --------------
    • Lots with lot_date > dep_date are not yet in the warehouse and are
      skipped (all later lots in sorted order are also skipped, so the
      break is safe).
    • Combined entity: lots from both 100001 + 100009 feed the queue;
      deps must be pre-filtered by the caller to dep_zid='100001'.
    • 100009 internal movements (MO-- IS--, DO-- to 100001) do NOT appear
      in deps because _build_dep_daily applies the zid filter.
    """
    EPS = 1e-9
    result = lots.copy()
    result["sold_qty"]       = 0.0
    result["batch_end_date"] = pd.NaT

    deps = deps.copy()
    deps["dep_date"] = pd.to_datetime(deps["dep_date"], errors="coerce").dt.floor("D")
    deps = deps[deps["dep_date"].notna()].copy()

    for code, code_lots in lots.groupby("itemcode", sort=False):
        code_deps = (
            deps[deps["itemcode"] == str(code)]
            .sort_values("dep_date")
            .reset_index(drop=True)
        )
        if code_deps.empty:
            continue

        sorted_idx = code_lots.sort_values("lot_date").index.tolist()
        remaining  = {i: float(lots.at[i, "lot_qty"]) for i in sorted_idx}
        end_dates  = {i: pd.NaT                       for i in sorted_idx}
        qpos = 0

        for _, dep in code_deps.iterrows():
            to_dep   = float(dep["dep_qty"])
            dep_date = pd.Timestamp(dep["dep_date"]).floor("D")

            while to_dep > EPS and qpos < len(sorted_idx):
                idx      = sorted_idx[qpos]
                lot_date = lots.at[idx, "lot_date"]
                if pd.notna(lot_date) and pd.Timestamp(lot_date).floor("D") > dep_date:
                    break  # lot not yet in warehouse; all later lots are newer too
                absorbed        = min(remaining[idx], to_dep)
                remaining[idx] -= absorbed
                to_dep         -= absorbed
                if remaining[idx] < EPS:
                    remaining[idx] = 0.0
                    if pd.isna(end_dates[idx]):
                        end_dates[idx] = dep_date
                    qpos += 1

        for i in sorted_idx:
            result.at[i, "sold_qty"]       = lots.at[i, "lot_qty"] - remaining[i]
            result.at[i, "batch_end_date"] = end_dates[i]

    result["remaining_qty"] = (result["lot_qty"] - result["sold_qty"]).clip(lower=0.0)
    result["is_closed"]     = result["remaining_qty"] < EPS
    return result



def run_batch_profitability_engine(
    purchase_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    movements_df: pd.DataFrame,
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
    """Batch profitability using raw-ledger FIFO across repeated shipments.

    Each individual IGRN receipt in mv_imtrn_movements becomes one FIFO lot
    (receipt date = txn_date, not a combinedate aggregate).  This eliminates
    the phantom-remaining bug that arose when a single combinedate grouped
    receipts spanning multiple days.

    Combined entity: both 100001 and 100009 IGRN lots are in the FIFO queue.
    Depletions are 100001-only (sale + issue); 100009 IS-- manufacturing issues
    and DO-- inter-company transfers are excluded — they cancel in the combined
    entity view.

    Output columns are identical to the previous implementation so all
    downstream views (SKU Simulator, Inventory Check, etc.) are unaffected.
    """

    EPS   = 1e-9
    as_of = pd.Timestamp(_today()).floor("D")

    # ── inner helper: onhand just before each batch arrived ───────────────────
    def _build_onhand_before(mv_df: pd.DataFrame, batches: pd.DataFrame) -> pd.Series:
        if mv_df is None or (isinstance(mv_df, pd.DataFrame) and mv_df.empty) or batches.empty:
            return pd.Series([0.0] * len(batches), index=batches.index)
        mv = mv_df.copy()
        mv["zid"]      = mv["zid"].astype(str).str.strip()
        mv             = mv[mv["zid"].isin(["100001", "100009"])].copy()
        if mv.empty:
            return pd.Series([0.0] * len(batches), index=batches.index)
        mv["itemcode"] = mv["itemcode"].apply(_norm_code).astype(str).str.strip()
        mv["date"]     = pd.to_datetime(mv["txn_date"], errors="coerce").dt.floor("D")
        mv             = mv[mv["date"].notna()].copy()
        mv["net_qty"]  = pd.to_numeric(mv.get("net_qty", 0), errors="coerce").fillna(0.0)
        daily = (
            mv.groupby(["date", "itemcode"], as_index=False)["net_qty"].sum()
              .sort_values(["date", "itemcode"]).reset_index(drop=True)
        )
        daily["onhand_qty"] = daily.groupby("itemcode")["net_qty"].cumsum()
        q = batches[["itemcode", "combinedate"]].copy()
        q["qdate"]    = q["combinedate"] - pd.Timedelta(nanoseconds=1)
        q             = q.reset_index(drop=False).rename(columns={"index": "_rowid"})
        q["itemcode"] = q["itemcode"].astype(str).str.strip()
        daily["itemcode"] = daily["itemcode"].astype(str).str.strip()
        q     = q.sort_values(["qdate", "itemcode"]).reset_index(drop=True)
        daily = daily.sort_values(["date", "itemcode"]).reset_index(drop=True)
        m = pd.merge_asof(
            q, daily,
            left_on="qdate", right_on="date",
            by="itemcode", direction="backward", allow_exact_matches=True,
        )
        m = m[["_rowid", "onhand_qty"]].rename(columns={"onhand_qty": "onhand_before"})
        m["onhand_before"] = pd.to_numeric(m["onhand_before"], errors="coerce").fillna(0.0)
        m["_rowid"]        = pd.to_numeric(m["_rowid"],        errors="coerce")
        out = m.set_index("_rowid").reindex(range(len(batches)))["onhand_before"].fillna(0.0)
        out.index = batches.index
        return out

    # ── 1. Build raw lots (individual IGRN receipts, both ZIDs) ──────────────
    all_lots = _build_raw_lots(movements_df, purchase_df)
    if all_lots.empty:
        return pd.DataFrame()

    target_shipment  = str(shipmentname).strip()
    sel_mask         = all_lots["shipmentname"] == target_shipment
    if not sel_mask.any():
        return pd.DataFrame()

    target_itemcodes = set(all_lots.loc[sel_mask, "itemcode"].tolist())

    # FIFO queue: all lots for these items (100001 + 100009)
    fifo_input = all_lots[all_lots["itemcode"].isin(target_itemcodes)].copy()

    # ── 2. Build depletions (100001 only: sale + issue) ───────────────────────
    dep_daily = _build_dep_daily(movements_df, dep_zid=str(zid_deplete))
    dep_daily = dep_daily[dep_daily["itemcode"].isin(target_itemcodes)].copy()

    # ── 3. Run FIFO ───────────────────────────────────────────────────────────
    fifo_result = _fifo_lots(fifo_input, dep_daily)

    # ── 4. Aggregate lots → (shipmentname, itemcode) batch rows ──────────────
    # combinedate = earliest receipt date for the shipment×item pair
    # unit_cost   = quantity-weighted average across lots
    agg_basic = (
        fifo_result
        .groupby(["shipmentname", "itemcode"], as_index=False)
        .agg(
            itemname     =("itemname",      "first"),
            combinedate  =("lot_date",      "min"),
            initial_qty  =("lot_qty",       "sum"),
            sold_qty     =("sold_qty",      "sum"),
            remaining_qty=("remaining_qty", "sum"),
        )
    )

    # weighted unit_cost
    _wc = fifo_result[["shipmentname", "itemcode", "lot_qty", "unit_cost"]].copy()
    _wc["_wc"] = _wc["lot_qty"] * _wc["unit_cost"]
    _wc_agg = (
        _wc.groupby(["shipmentname", "itemcode"], as_index=False)
           .agg(_sq=("lot_qty", "sum"), _sw=("_wc", "sum"))
    )
    _wc_agg["unit_cost"] = np.where(
        _wc_agg["_sq"] > EPS, _wc_agg["_sw"] / _wc_agg["_sq"], 0.0
    )

    # batch_end_date: max close-date across closed lots; NaT if any lot is open
    _closed  = fifo_result[fifo_result["is_closed"]].copy()
    _bed_agg = (
        _closed.groupby(["shipmentname", "itemcode"], as_index=False)["batch_end_date"].max()
    )

    batch_grp = (
        agg_basic
        .merge(_wc_agg[["shipmentname", "itemcode", "unit_cost"]],
               on=["shipmentname", "itemcode"], how="left")
        .merge(_bed_agg, on=["shipmentname", "itemcode"], how="left")
    )
    batch_grp["batch_end_date"] = pd.to_datetime(batch_grp["batch_end_date"], errors="coerce")
    batch_grp.loc[batch_grp["remaining_qty"] >= EPS, "batch_end_date"] = pd.NaT
    batch_grp["is_closed"]  = batch_grp["remaining_qty"] < EPS
    batch_grp["batch_cost"] = batch_grp["initial_qty"] * batch_grp["unit_cost"]
    batch_grp["batch_id"]   = (
        batch_grp["shipmentname"].astype(str)
        + " | " + batch_grp["itemcode"].astype(str)
        + " | " + pd.to_datetime(batch_grp["combinedate"]).dt.strftime("%Y-%m-%d")
    )
    batch_grp["combinedate"] = pd.to_datetime(batch_grp["combinedate"], errors="coerce").dt.floor("D")

    # ── 5. Pull selected shipment rows ────────────────────────────────────────
    df0 = batch_grp[batch_grp["shipmentname"] == target_shipment].copy().reset_index(drop=True)
    if df0.empty:
        return pd.DataFrame()

    df0 = df0.sort_values(["itemcode", "combinedate"]).reset_index(drop=True)

    df0["onhand_before"] = _build_onhand_before(movements_df, df0).astype(float)
    df0["threshold_qty"] = df0["onhand_before"].astype(float)

    # ── 6. Revenue / avg price ────────────────────────────────────────────────
    # Build per-item cumulative (qty, rev) arrays from sales_df for price lookup
    rev_lkp: dict = {}
    if isinstance(sales_df, pd.DataFrame) and not sales_df.empty:
        s = sales_df.copy()
        s["zid"]      = s["zid"].astype(str).str.strip()
        s             = s[s["zid"] == str(zid_deplete).strip()].copy()
        s["itemcode"] = s["itemcode"].apply(_norm_code).astype(str).str.strip()
        s["d"]        = pd.to_datetime(s.get("date"), errors="coerce").dt.floor("D")
        s             = s[s["d"].notna()].copy()
        _val_col = next((c for c in ["totalsales", "altsales", "sales_rev"] if c in s.columns), None)
        _qty_col2 = "quantity" if "quantity" in s.columns else "sales_qty"
        s["_rev"] = pd.to_numeric(s[_val_col],       errors="coerce").fillna(0.0) if _val_col else 0.0
        s["_qty"] = pd.to_numeric(s.get(_qty_col2, 0), errors="coerce").fillna(0.0).clip(lower=0.0)
        sr = (
            s.groupby(["itemcode", "d"], as_index=False)
             .agg(_qty=("_qty", "sum"), _rev=("_rev", "sum"))
             .sort_values(["itemcode", "d"]).reset_index(drop=True)
        )
        sr["cum_qty"] = sr.groupby("itemcode")["_qty"].cumsum()
        sr["cum_rev"] = sr.groupby("itemcode")["_rev"].cumsum()
        for code_r, grp_r in sr.groupby("itemcode"):
            rev_lkp[str(code_r)] = (
                grp_r["d"].values.astype("datetime64[ns]"),
                grp_r["cum_qty"].values.astype(float),
                grp_r["cum_rev"].values.astype(float),
            )

    def _cum_at(dn, cn, T):
        if pd.isna(T):
            return 0.0
        idx = int(np.searchsorted(dn, np.datetime64(T, "ns"), side="right")) - 1
        return float(cn[idx]) if idx >= 0 else 0.0

    # avg realized price per (itemcode, combinedate) window
    # window: combinedate → day before next batch of same item (or today)
    price_cache: dict = {}
    for code_p, item_df in df0.groupby("itemcode"):
        sorted_dates = sorted(item_df["combinedate"].dropna().unique())
        for i_p, Di_p in enumerate(sorted_dates):
            price_to = (sorted_dates[i_p + 1] - pd.Timedelta(days=1)
                        if i_p + 1 < len(sorted_dates) else as_of)
            key = (str(code_p), Di_p)
            if str(code_p) in rev_lkp:
                dn, cq, cr = rev_lkp[str(code_p)]
                Di_excl = Di_p - pd.Timedelta(days=1)
                qty_w   = max(0.0, _cum_at(dn, cq, price_to) - _cum_at(dn, cq, Di_excl))
                rev_w   = max(0.0, _cum_at(dn, cr, price_to) - _cum_at(dn, cr, Di_excl))
                price_cache[key] = (rev_w / qty_w) if qty_w > EPS else np.nan
            else:
                price_cache[key] = np.nan

    # global historical fallback per item
    hist_price_map: dict = {}
    if isinstance(sales_df, pd.DataFrame) and not sales_df.empty:
        s_h = sales_df.copy()
        s_h["zid"]      = s_h["zid"].astype(str).str.strip()
        s_h             = s_h[s_h["zid"] == str(zid_deplete).strip()].copy()
        s_h["itemcode"] = s_h["itemcode"].apply(_norm_code).astype(str).str.strip()
        _vq = "quantity" if "quantity" in s_h.columns else "sales_qty"
        _vr = next((c for c in ["totalsales", "altsales", "sales_rev"] if c in s_h.columns), None)
        s_h["_qty"] = pd.to_numeric(s_h.get(_vq, 0), errors="coerce").fillna(0.0)
        s_h["_rev"] = pd.to_numeric(s_h[_vr],        errors="coerce").fillna(0.0) if _vr else 0.0
        hg = s_h.groupby("itemcode", as_index=False).agg(tq=("_qty", "sum"), tr=("_rev", "sum"))
        hg["hp"] = np.where(hg["tq"] > EPS, hg["tr"] / hg["tq"], np.nan)
        hist_price_map = dict(
            zip(hg["itemcode"].astype(str), pd.to_numeric(hg["hp"], errors="coerce"))
        )

    # fallback chain: window realized → historical global → unit_cost
    df0["avg_price"] = df0.apply(
        lambda r: price_cache.get((str(r["itemcode"]), r["combinedate"]), np.nan), axis=1
    )
    df0["avg_price"] = df0["avg_price"].where(
        df0["avg_price"].notna(),
        df0["itemcode"].astype(str).map(hist_price_map),
    )
    df0["avg_price"] = df0["avg_price"].fillna(df0["unit_cost"])
    df0["avg_price"] = pd.to_numeric(df0["avg_price"], errors="coerce").fillna(0.0)

    df0["scenario_price"] = df0["avg_price"] * (1.0 - float(discount_pct) / 100.0)
    df0["sold_revenue"]   = df0["sold_qty"].astype(float) * df0["avg_price"].astype(float)

    # ── 7. COGS / gross margin ────────────────────────────────────────────────
    df0["realized_cogs"]         = df0["sold_qty"].astype(float) * df0["unit_cost"].astype(float)
    df0["realized_gm"]           = df0["sold_revenue"].astype(float) - df0["realized_cogs"].astype(float)
    df0["remaining_cost_value"]  = df0["remaining_qty"].astype(float) * df0["unit_cost"].astype(float)
    df0["proj_remaining_revenue"] = df0["remaining_qty"].astype(float) * df0["scenario_price"].astype(float)
    df0["proj_remaining_gm"]     = df0["proj_remaining_revenue"].astype(float) - df0["remaining_cost_value"].astype(float)

    # ── 8. Activity timing / velocity ─────────────────────────────────────────
    end_eff              = df0["batch_end_date"].where(df0["batch_end_date"].notna(), as_of)
    df0["days_active"]   = ((end_eff - df0["combinedate"]).dt.days + 1).clip(lower=1).astype(int)
    df0["velocity"]      = np.where(df0["days_active"] > 0, df0["sold_qty"] / df0["days_active"], 0.0)
    df0["velocity"]      = pd.to_numeric(df0["velocity"], errors="coerce").fillna(0.0)
    _vel_used            = np.where(df0["sold_qty"] > EPS, df0["velocity"], 0.02)
    df0["days_to_clear"] = np.where(
        _vel_used > EPS, df0["remaining_qty"] / _vel_used, 730.0
    )
    df0["days_to_clear"] = (
        pd.to_numeric(df0["days_to_clear"], errors="coerce").fillna(730.0).clip(0.0, 730.0)
    )
    df0["batch_age_days"] = ((as_of - df0["combinedate"]).dt.days).astype(int)

    # ── 9. Overhead ────────────────────────────────────────────────────────────
    total_sold_revenue    = float(df0["sold_revenue"].sum()) if "sold_revenue" in df0.columns else 0.0
    vat_overhead_value    = (float(vat_pct) / 100.0) * max(0.0, total_sold_revenue)
    manual_overhead_value = float(manual_overhead_value or 0.0)
    total_overhead_pool   = float(shipment_overhead_total or 0.0) + vat_overhead_value + manual_overhead_value

    if total_sold_revenue > EPS:
        share_real = (df0["sold_revenue"] / total_sold_revenue).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        _denom = float(df0["realized_cogs"].sum())
        share_real = (df0["realized_cogs"] / _denom).replace([np.inf, -np.inf], np.nan).fillna(0.0) \
            if _denom > EPS else 0.0

    days_elapsed          = max(1, int(df0["days_active"].max()) if "days_active" in df0.columns else 1)
    D0                    = total_overhead_pool / float(days_elapsed)
    realized_end_eff      = df0["batch_end_date"].where(df0["batch_end_date"].notna(), as_of)
    realized_days         = ((realized_end_eff - df0["combinedate"]).dt.days + 1).clip(lower=1).astype(float)
    df0["overhead_realized"]   = D0 * realized_days * share_real
    df0["net_profit_realized"] = df0["realized_gm"] - df0["overhead_realized"]

    _tprr = float(df0["proj_remaining_revenue"].sum())
    if _tprr > EPS:
        share_rem = (df0["proj_remaining_revenue"] / _tprr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        _denom2 = float(df0["remaining_cost_value"].sum())
        share_rem = (df0["remaining_cost_value"] / _denom2).replace([np.inf, -np.inf], np.nan).fillna(0.0) \
            if _denom2 > EPS else 0.0

    dclear = df0["days_to_clear"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 730.0)
    df0["overhead_projected"]    = D0 * np.power(0.97, dclear / 60.0) * dclear * share_rem
    df0["Proj_remaining_profit"] = df0["proj_remaining_gm"] - df0["overhead_projected"]
    df0["proj_final_profit"]     = df0["net_profit_realized"] + df0["Proj_remaining_profit"]

    # ── 10. Final output ───────────────────────────────────────────────────────
    cols = [
        "shipmentname", "batch_id", "itemcode", "itemname",
        "onhand_before", "combinedate", "batch_end_date", "is_closed",
        "initial_qty", "sold_qty", "remaining_qty", "threshold_qty",
        "unit_cost", "sold_revenue", "realized_cogs", "realized_gm",
        "overhead_realized", "net_profit_realized",
        "remaining_cost_value", "proj_remaining_revenue", "proj_remaining_gm",
        "overhead_projected", "Proj_remaining_profit", "proj_final_profit",
        "avg_price", "scenario_price",
        "days_active", "velocity", "days_to_clear", "batch_age_days",
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

    _numeric_round_cols = [
        "onhand_before", "initial_qty", "sold_qty", "remaining_qty", "threshold_qty",
        "unit_cost", "sold_revenue", "realized_cogs", "realized_gm",
        "overhead_realized", "net_profit_realized",
        "remaining_cost_value", "proj_remaining_revenue", "proj_remaining_gm",
        "overhead_projected", "Proj_remaining_profit", "proj_final_profit",
        "avg_price", "scenario_price", "velocity", "days_to_clear",
    ]
    for _c in _numeric_round_cols:
        if _c in df0.columns:
            df0[_c] = pd.to_numeric(df0[_c], errors="coerce").round(0).fillna(0.0)

    return df0


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
        target_cum = prior_cum + initial_qty * 0.99  # 99% sell-through = depleted

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
        elif n < 3:
            flag = "Low confidence"
        elif p90 is not None and p90 > 120:
            flag = "Dead stock"
        else:
            flag = "Good"

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


@st.cache_data(ttl=3600, show_spinner=False)
def build_abc_xyz(sales_df: pd.DataFrame) -> pd.DataFrame:
    """ABC-XYZ inventory classification.

    ABC: cumulative revenue contribution across full sales history.
         A = top 80%, B = next 15%, C = bottom 5%.
    XYZ: coefficient of variation (CV) of monthly sales quantity.
         X = CV < 0.5 (stable), Y = 0.5–1.0 (variable), Z ≥ 1.0 (erratic).
         Products with < 6 calendar months of data → 'Insufficient'.

    Returns one row per itemcode with columns:
    itemcode, total_revenue, abc, n_months, cv, xyz, class_combined.
    """
    empty = pd.DataFrame(
        columns=["itemcode", "total_revenue", "abc", "n_months", "cv", "xyz", "class_combined"]
    )
    if sales_df is None or sales_df.empty:
        return empty

    s = sales_df.copy()
    s["itemcode"] = s["itemcode"].astype(str).str.strip()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s["quantity"] = pd.to_numeric(s["quantity"], errors="coerce").fillna(0.0).clip(lower=0.0)
    rev_col = next((c for c in ["totalsales", "sales_rev", "altsales"] if c in s.columns), None)
    s["revenue"] = pd.to_numeric(s[rev_col], errors="coerce").fillna(0.0) if rev_col else 0.0
    s = s[s["date"].notna()].copy()
    if s.empty:
        return empty

    # ---- ABC: cumulative revenue rank ----
    abc_df = s.groupby("itemcode", as_index=False)["revenue"].sum()
    abc_df.columns = ["itemcode", "total_revenue"]
    abc_df = abc_df.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    total_rev = abc_df["total_revenue"].sum()
    abc_df["cum_pct"] = (abc_df["total_revenue"].cumsum() / total_rev * 100) if total_rev > 0 else 100.0

    def _abc(pct: float) -> str:
        if pct <= 80: return "A"
        if pct <= 95: return "B"
        return "C"

    abc_df["abc"] = abc_df["cum_pct"].apply(_abc)

    # ---- XYZ: monthly quantity CV ----
    s["ym"] = s["date"].dt.to_period("M")
    monthly = s.groupby(["itemcode", "ym"], as_index=False)["quantity"].sum()

    xyz_rows: list = []
    for code, grp in monthly.groupby("itemcode"):
        n_months = int(grp["ym"].nunique())
        vals = grp["quantity"].values.astype(float)
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        if n_months < 6:
            xyz, cv = "Insufficient", None
        elif mean_v == 0:
            xyz, cv = "Z", None
        else:
            cv = std_v / mean_v
            if cv < 0.5:   xyz = "X"
            elif cv < 1.0: xyz = "Y"
            else:           xyz = "Z"
        xyz_rows.append({"itemcode": code, "n_months": n_months,
                         "cv": round(cv, 3) if cv is not None else None, "xyz": xyz})

    xyz_df = pd.DataFrame(xyz_rows)
    result = abc_df[["itemcode", "total_revenue", "abc"]].merge(xyz_df, on="itemcode", how="outer")
    result["total_revenue"] = pd.to_numeric(result["total_revenue"], errors="coerce").fillna(0.0)
    result["abc"] = result["abc"].fillna("C")
    result["xyz"] = result["xyz"].fillna("Insufficient")
    result["class_combined"] = result["abc"] + result["xyz"].replace("Insufficient", "?")
    return result.sort_values(["abc", "xyz", "total_revenue"], ascending=[True, True, False]).reset_index(drop=True)



@st.cache_data(ttl=3600, show_spinner=False)
def build_batch_consolidation(
    purchase_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    movements_df: pd.DataFrame | None = None,
) -> tuple:
    """Shipment-level consolidation using raw-ledger FIFO stock snapshots.

    Replaces the combinedate-aggregated approach with individual IGRN receipt
    lots from mv_imtrn_movements (via _build_raw_lots / _build_dep_daily /
    _fifo_lots), eliminating phantom remaining quantities.

    Combined entity: lots from both 100001 + 100009 IGRN receipts;
    depletions from 100001 only (sale + issue), so 100009 MO-- manufacturing
    issues and DO-- inter-company transfers are excluded.

    Returns (out[_COLS], pd.DataFrame(debug_rows)) — same tuple as before.
    """
    EPS = 1e-9

    today           = pd.Timestamp.today().normalize()
    yesterday       = today - pd.Timedelta(days=1)
    day_before_yest = yesterday - pd.Timedelta(days=1)
    month_start     = today.replace(day=1)
    snap_ms         = month_start - pd.Timedelta(days=1)   # end of prev month
    window_start    = pd.Timestamp(year=today.year - 3, month=1, day=1)

    _COLS = [
        "Shipment Name", "Shipment Date", "Arrival Value (BDT)",
        "Month Start Stock (BDT)", "Current Stock Yesterday (BDT)",
        "MTD Sales (BDT)", "Yesterday Sales (BDT)",
    ]
    EMPTY = pd.DataFrame(columns=_COLS)

    if purchase_df is None or purchase_df.empty:
        return EMPTY, pd.DataFrame()
    if sales_df is None or sales_df.empty:
        return EMPTY, pd.DataFrame()

    # ── 1. Identify non-open shipments within the analysis window ─────────────
    p = purchase_df.copy()
    p["shipmentname"] = p["shipmentname"].astype(str).str.strip()
    p["combinedate"]  = pd.to_datetime(p["combinedate"], errors="coerce").dt.floor("D")
    _status_col = "status" if "status" in p.columns else None
    valid_mask = (
        p["combinedate"].notna() &
        (p["combinedate"] >= window_start)
    )
    if _status_col:
        valid_mask &= (p[_status_col] != "1-Open")
    valid_shipments = set(p.loc[valid_mask, "shipmentname"].unique())
    if not valid_shipments:
        return EMPTY, pd.DataFrame()

    # ── 2. Build raw lots (both ZIDs, IGRN only) ──────────────────────────────
    all_lots = _build_raw_lots(movements_df, purchase_df)
    if all_lots.empty:
        return EMPTY, pd.DataFrame()
    all_lots = all_lots[all_lots["shipmentname"].isin(valid_shipments)].copy()
    if all_lots.empty:
        return EMPTY, pd.DataFrame()

    # ── 3. Depletion table (100001 only) ─────────────────────────────────────
    dep_daily = _build_dep_daily(movements_df, dep_zid="100001")
    target_items = set(all_lots["itemcode"].tolist())
    dep_daily = dep_daily[dep_daily["itemcode"].isin(target_items)].copy()

    # ── 4. Price cache per lot (per-item cumulative sales arrays) ─────────────
    s = sales_df.copy()
    if "zid" in s.columns:
        s["zid"] = s["zid"].astype(str).str.strip()
        s = s[s["zid"] == "100001"].copy()
    s["itemcode"] = s["itemcode"].astype(str).str.strip()
    s["date"]     = pd.to_datetime(s.get("date"), errors="coerce").dt.floor("D")
    _qty_s = "quantity" if "quantity" in s.columns else "sales_qty"
    _rev_s = next((c for c in ["totalsales", "altsales", "sales_rev"] if c in s.columns), None)
    s["_qty"] = pd.to_numeric(s.get(_qty_s, 0), errors="coerce").fillna(0.0).clip(lower=0.0)
    s["_rev"] = pd.to_numeric(s[_rev_s], errors="coerce").fillna(0.0) if _rev_s else 0.0
    s = (
        s[s["date"].notna()]
        .groupby(["itemcode", "date"], as_index=False)
        .agg(qty=("_qty", "sum"), rev=("_rev", "sum"))
        .sort_values(["itemcode", "date"]).reset_index(drop=True)
    )
    s["cum_qty"] = s.groupby("itemcode")["qty"].cumsum()
    s["cum_rev"] = s.groupby("itemcode")["rev"].cumsum()

    sales_lkp: dict = {}
    for code_s, grp_s in s.groupby("itemcode"):
        sales_lkp[str(code_s)] = (
            grp_s["date"].values.astype("datetime64[ns]"),
            grp_s["cum_qty"].values.astype(float),
            grp_s["cum_rev"].values.astype(float),
        )

    def _cum_at(dates_np, cum_np, T: pd.Timestamp) -> float:
        if pd.isna(T):
            return 0.0
        idx = int(np.searchsorted(dates_np, np.datetime64(T, "ns"), side="right")) - 1
        return float(cum_np[idx]) if idx >= 0 else 0.0

    # Price window: lot_date → day before next lot of same item (or yesterday)
    price_cache: dict = {}
    for code_p, item_lots in all_lots.groupby("itemcode"):
        sorted_dates = sorted(item_lots["lot_date"].dropna().unique())
        for i_p, Di_p in enumerate(sorted_dates):
            price_to = (
                sorted_dates[i_p + 1] - pd.Timedelta(days=1)
                if i_p + 1 < len(sorted_dates) else yesterday
            )
            key_p = (str(code_p), Di_p)
            if str(code_p) in sales_lkp:
                dn, cq, cr = sales_lkp[str(code_p)]
                Di_excl = Di_p - pd.Timedelta(days=1)
                qty_w   = max(0.0, _cum_at(dn, cq, price_to) - _cum_at(dn, cq, Di_excl))
                rev_w   = max(0.0, _cum_at(dn, cr, price_to) - _cum_at(dn, cr, Di_excl))
                price_cache[key_p] = (rev_w / qty_w) if qty_w > EPS else np.nan
            else:
                price_cache[key_p] = np.nan

    # Fallback: global historical avg price per item
    hist_price_map: dict = {}
    for code_h, (dn_h, cq_h, cr_h) in sales_lkp.items():
        tq_h = float(cq_h[-1]) if len(cq_h) > 0 else 0.0
        tr_h = float(cr_h[-1]) if len(cr_h) > 0 else 0.0
        hist_price_map[code_h] = (tr_h / tq_h) if tq_h > EPS else np.nan

    # Assign price per lot (price_cache → hist_price_map → unit_cost)
    def _lot_price(row) -> float:
        code = str(row["itemcode"])
        ld   = pd.Timestamp(row["lot_date"]).floor("D")
        pv   = price_cache.get((code, ld), np.nan)
        if pd.isna(pv) or pv <= 0:
            pv = hist_price_map.get(code, np.nan)
        if pd.isna(pv) or pv <= 0:
            pv = float(row["unit_cost"])
        return pv

    all_lots["price"] = all_lots.apply(_lot_price, axis=1).fillna(0.0)

    # ── 5. FIFO snapshots (run 3 times, filtered by date) ────────────────────
    def _snap_remaining(T: pd.Timestamp) -> pd.DataFrame:
        """FIFO remaining_qty per lot at date T."""
        l_t = all_lots[
            pd.to_datetime(all_lots["lot_date"], errors="coerce").dt.floor("D") <= T
        ].copy()
        d_t = dep_daily[
            pd.to_datetime(dep_daily["dep_date"], errors="coerce").dt.floor("D") <= T
        ].copy()
        if l_t.empty:
            return pd.DataFrame(columns=["zid", "xdocnum", "itemcode", "remaining_qty"])
        return _fifo_lots(l_t, d_t)[["zid", "xdocnum", "itemcode", "remaining_qty"]]

    snap_ms_r  = _snap_remaining(snap_ms)
    snap_y_r   = _snap_remaining(yesterday)
    snap_dby_r = _snap_remaining(day_before_yest)

    # ── 6. Merge snapshots onto lot master ────────────────────────────────────
    base = all_lots[
        ["zid", "xdocnum", "itemcode", "itemname",
         "shipmentname", "lot_date", "lot_qty", "unit_cost", "price"]
    ].copy()

    def _merge_rem(base_df, snap_df, col_name):
        if snap_df.empty:
            base_df[col_name] = 0.0
            return base_df
        snp = snap_df.rename(columns={"remaining_qty": col_name})
        merged = base_df.merge(snp, on=["zid", "xdocnum", "itemcode"], how="left")
        merged[col_name] = pd.to_numeric(merged[col_name], errors="coerce").fillna(0.0).clip(lower=0.0)
        return merged

    base = _merge_rem(base, snap_ms_r,  "rem_ms")
    base = _merge_rem(base, snap_y_r,   "rem_y")
    base = _merge_rem(base, snap_dby_r, "rem_dby")

    base["arrival_val"]   = base["lot_qty"]  * base["price"]
    base["ms_stock_val"]  = base["rem_ms"]   * base["price"]
    base["cur_stock_val"] = base["rem_y"]    * base["price"]
    base["mtd_val"]       = (base["rem_ms"]  - base["rem_y"] ).clip(lower=0.0) * base["price"]
    base["yest_val"]      = (base["rem_dby"] - base["rem_y"] ).clip(lower=0.0) * base["price"]

    # ── 7. Debug rows (item-level per shipment) ───────────────────────────────
    debug_grp = (
        base
        .groupby(["shipmentname", "itemcode", "itemname", "lot_date"], as_index=False)
        .agg(
            initial_qty =("lot_qty",   "sum"),
            rem_y       =("rem_y",     "sum"),
            price       =("price",     "first"),
            unit_cost   =("unit_cost", "first"),
        )
    )
    debug_rows = []
    for _, r in debug_grp.iterrows():
        debug_rows.append({
            "Shipment":      r["shipmentname"],
            "Item Code":     r["itemcode"],
            "Item Name":     r["itemname"],
            "Batch Date":    r["lot_date"],
            "Initial Qty":   round(r["initial_qty"]),
            "Sold (FIFO)":   round(max(0.0, r["initial_qty"] - r["rem_y"])),
            "Remaining Qty": round(r["rem_y"]),
            "Price":         round(float(r["price"]), 2),
            "Stock Val":     round(r["rem_y"] * float(r["price"])),
        })

    # ── 8. Aggregate to shipment level ────────────────────────────────────────
    ship_dates = (
        base.groupby("shipmentname")["lot_date"].min()
        .rename("Shipment Date").reset_index()
    )
    out = (
        base
        .groupby("shipmentname", as_index=False)
        .agg(
            arrival_val  =("arrival_val",   "sum"),
            ms_stock_val =("ms_stock_val",  "sum"),
            cur_stock_val=("cur_stock_val", "sum"),
            mtd_val      =("mtd_val",       "sum"),
            yest_val     =("yest_val",      "sum"),
        )
        .merge(ship_dates, on="shipmentname", how="left")
        .sort_values("Shipment Date", ascending=False)
        .reset_index(drop=True)
        .rename(columns={
            "shipmentname":  "Shipment Name",
            "arrival_val":   "Arrival Value (BDT)",
            "ms_stock_val":  "Month Start Stock (BDT)",
            "cur_stock_val": "Current Stock Yesterday (BDT)",
            "mtd_val":       "MTD Sales (BDT)",
            "yest_val":      "Yesterday Sales (BDT)",
        })
    )

    for col in [
        "Arrival Value (BDT)", "Month Start Stock (BDT)",
        "Current Stock Yesterday (BDT)", "MTD Sales (BDT)", "Yesterday Sales (BDT)",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(0).fillna(0).astype("Int64")

    # Drop shipments with zero current stock (fully depleted)
    out = out[out["Current Stock Yesterday (BDT)"] > 0].reset_index(drop=True)

    # Ensure all required columns exist
    for c in _COLS:
        if c not in out.columns:
            out[c] = pd.NA

    return out[_COLS], pd.DataFrame(debug_rows)
