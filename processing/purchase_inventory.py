from __future__ import annotations

import pandas as pd
import numpy as np
import os
import json
import re
from datetime import date as _date
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st


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

    s["docnum"] = (s["docnum"].astype(str).fillna("").str.strip()
                   if "docnum" in s.columns else "")
    s["prefix"] = s["docnum"].str.slice(0, 4)

    s["warehouse"] = (s["warehouse"].astype(str).fillna("").str.strip()
                      if "warehouse" in s.columns else "")

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

@st.cache_data(ttl=86400)
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

@st.cache_data(show_spinner=False, ttl=86400)
def build_accounts_overhead_summary(
    purchase_df: pd.DataFrame,
    stock_movement_df: pd.DataFrame,
    gl_overhead_df: pd.DataFrame,
    glmst_df: pd.DataFrame,
    hierarchy_path: str,
    shipmentname: str,
    level: int,
    selections: tuple,
    include_details: bool = False,
    zids_inventory: Optional[List[str]] = None,
    warehouse_filters: Optional[Dict[str, List[str]]] = None,
    warehouse_json_path: str = "data/warehouse_filters.json",
    revenue_selections: tuple = (),) -> Dict[str, Any]:
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

    if gl_overhead_df is None or (isinstance(gl_overhead_df, pd.DataFrame) and gl_overhead_df.empty):
        return {
            "summary_df": pd.DataFrame(),
            "totals": {"overhead_total_sum": 0.0, "overhead_for_shipment_sum": 0.0, "avg_daily_overhead_for_shipment": 0.0},
            "details_df": pd.DataFrame(),
            "end_eff": end_eff,
            "is_closed": bool(is_closed),
        }

    _gl_raw = gl_overhead_df.copy()
    _gl_raw["date"]    = pd.to_datetime(_gl_raw["date"], errors="coerce").dt.floor("D")
    _gl_raw["ac_code"] = _gl_raw["ac_code"].astype(str).str.strip()
    _gl_raw["value"]   = pd.to_numeric(_gl_raw["value"], errors="coerce").fillna(0.0)
    gl = _gl_raw[_gl_raw["date"].notna()][["date", "ac_code", "value"]].copy()

    if gl.empty:
        return {
            "summary_df": pd.DataFrame(),
            "totals": {"overhead_total_sum": 0.0, "overhead_for_shipment_sum": 0.0, "avg_daily_overhead_for_shipment": 0.0},
            "details_df": pd.DataFrame(),
            "end_eff": end_eff,
            "is_closed": bool(is_closed),
        }

    gl = gl[(gl["date"] >= start_date) & (gl["date"] <= end_eff)].copy()

    # multi-select filter (selections may arrive as tuple for cache-key stability)
    selections_list = list(selections) if selections else []
    m = _selection_masks(gl, level=level, selections=selections_list, hierarchy_path=hierarchy_path)
    gl_sel = gl[m].copy()

    # Revenue adjustments: income accounts whose negative GL values reduce the overhead pool
    if revenue_selections:
        rev_codes = [str(s).strip() for s in revenue_selections]
        rev_mask = gl["ac_code"].isin(rev_codes)
        gl_rev = gl[rev_mask].copy()
        if not gl_rev.empty:
            gl_sel = pd.concat([gl_sel, gl_rev], ignore_index=True)

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
    sel_list = list(selections) if selections else ["(ALL SELECTED)"]

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

    # Revenue adjustments summary row — show each selected revenue account separately
    if revenue_selections:
        rev_codes = [str(s).strip() for s in revenue_selections]
        for rev_code in rev_codes:
            gl_rev_part = gl[gl["ac_code"] == rev_code].copy()
            if gl_rev_part.empty:
                continue
            rev_lbl = rev_code
            if glmst_df is not None and not glmst_df.empty:
                gm_tmp = glmst_df.copy()
                gm_tmp["ac_code"] = gm_tmp["ac_code"].astype(str).str.strip()
                hit = gm_tmp[gm_tmp["ac_code"] == rev_code]
                if not hit.empty:
                    rev_lbl = f"{rev_code} - {hit['ac_name'].iloc[0]}"
            ovp_r = (
                gl_rev_part.groupby("date", as_index=False)["value"]
                .sum()
                .rename(columns={"value": "overhead_total_for_day"})
            )
            tmp_r = base[["date", "total_inventory_value_cost", "shipment_value_cost"]].merge(ovp_r, on="date", how="left")
            tmp_r["overhead_total_for_day"] = tmp_r["overhead_total_for_day"].fillna(0.0)
            denom_r = tmp_r["total_inventory_value_cost"].replace(0.0, np.nan)
            tmp_r["ratio"] = (tmp_r["shipment_value_cost"] / denom_r).fillna(0.0)
            tmp_r["overhead_allocated_for_day"] = tmp_r["overhead_total_for_day"] * tmp_r["ratio"]
            rows.append({
                "level": 0,
                "selection": rev_code,
                "label": f"[Revenue Adj] {rev_lbl}",
                "overhead_total": float(tmp_r["overhead_total_for_day"].sum()),
                "overhead_for_shipment": float(tmp_r["overhead_allocated_for_day"].sum()),
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

def build_accounts_overhead_table(purchase_df: pd.DataFrame,stock_movement_df: pd.DataFrame,gl_overhead_df: pd.DataFrame,
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

    # GL join (use pre-joined MV data directly)
    if gl_overhead_df is None or (isinstance(gl_overhead_df, pd.DataFrame) and gl_overhead_df.empty):
        return pd.DataFrame()
    _gl_r = gl_overhead_df.copy()
    _gl_r["date"] = pd.to_datetime(_gl_r["date"], errors="coerce").dt.floor("D")
    _gl_r["ac_code"] = _gl_r["ac_code"].astype(str).str.strip()
    _gl_r["value"] = pd.to_numeric(_gl_r["value"], errors="coerce").fillna(0.0)
    gl = _gl_r[_gl_r["date"].notna()][["date", "ac_code", "value"]].copy()

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
        return {"level2_options": [], "level1_options": [], "level0_options": [], "revenue_options": []}

    gm = glmst_df.copy()
    gm["ac_code"] = gm["ac_code"].astype(str).fillna("").str.strip()
    gm["ac_name"] = gm.get("ac_name", "").astype(str).fillna("").str.strip()

    # Revenue accounts (Income type) — negative GL values reduce overhead when selected
    if "ac_type" in gm.columns:
        gm_rev = gm[gm["ac_type"].astype(str).str.lower() == "income"].copy()
    else:
        gm_rev = gm[gm["ac_code"].str.startswith("08")].copy()
    gm_rev = gm_rev.sort_values("ac_code")
    revenue_options = (gm_rev["ac_code"] + " " + gm_rev["ac_name"]).tolist()

    # Keep only expense families we care about (numeric prefixes only; 06A/06B are hierarchy groupings, not ac_code prefixes)
    gm = gm[gm["ac_code"].str.startswith(("05", "06", "07"))].copy()
    if gm.empty:
        return {"level2_options": [], "level1_options": [], "level0_options": [], "revenue_options": revenue_options}

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
        "revenue_options": revenue_options,
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

@st.cache_data(show_spinner=False, ttl=86400)
def build_warehouse_total_value_table(
    stock_movement_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    zids: List[str] = None,
    warehouse_filters: Optional[Dict[str, List[str]]] = None,
    warehouse_json_path: str = "data/warehouse_filters.json",) -> pd.DataFrame:
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

def load_warehouse_filters(warehouse_json_path: str = "data/warehouse_filters.json") -> Dict[str, List[str]]:
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
    warehouse_json_path: str = "data/warehouse_filters.json",
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

@st.cache_data(ttl=86400)
def total_inventory_value_timeseries(

    stock_movement_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    zids: List[str],
    warehouse_json_path: str = "data/warehouse_filters.json",
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

@st.cache_data(show_spinner=False, ttl=86400)
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


# ============================================================
# Cross-ZID Item Mapping (100009 FG/RM -> 100001 catalog)
# ============================================================

def build_crosszid_item_mapping(gulshan_df: pd.DataFrame, hmbr_df: pd.DataFrame) -> pd.DataFrame:
    """Relate every Gulshan Packaging (100009) FG/RM item to its claimed HMBR
    (100001) counterpart via caitem.xdrawing -- resolved here in Python
    against a full 100001 catalog lookup, not via a SQL JOIN. This matters:
    a JOIN with the usual "valid xdrawing" WHERE filter silently drops any
    item whose xdrawing is blank/'NO'/'KH*' from the result set entirely,
    so it never even gets a chance to be flagged "no duplicate" -- it just
    vanishes. Resolving in Python means every FG/RM item is guaranteed to
    produce exactly one output row, whatever its xdrawing looks like.

    gulshan_df: get_gulshan_fg_rm_items() output -- one row per 100009 item
        (itemcode prefixed 'FH' or 'HPI') with item_100009/name_100009/
        group_100009/xdrawing.
    hmbr_df: get_hmbr_catalog_lookup() output -- full 100001 catalog
        (itemcode/name_100001/xabc_100001), used purely as a lookup table.

    An xdrawing of NULL/''/'NO'/or starting with 'KH' is treated as "no real
    link" -- the same CASE used everywhere else this column is read in this
    codebase (see caitem's packcode CASE in CLAUDE.md) -- so those rows are
    flagged has_duplicate=False without attempting a lookup at all.

    Output columns match the original single-query version exactly (so the
    view layer needs no changes to its Match/Mismatch/No-Duplicate logic),
    plus the raw xdrawing value so a "No Duplicate" row's cause is visible
    at a glance (blank vs 'NO' vs 'KH...' vs a mistyped/broken code):
    itemcode, name_100001, name_100009, item_100009, group_100009,
    xabc_100001, xdrawing.
    """
    if gulshan_df is None or gulshan_df.empty:
        return pd.DataFrame(columns=[
            "itemcode", "name_100001", "name_100009", "item_100009",
            "group_100009", "xabc_100001", "xdrawing",
        ])

    g = gulshan_df.copy()
    g["xdrawing"] = g["xdrawing"].fillna("").astype(str).str.strip()

    valid_drawing = (
        (g["xdrawing"] != "")
        & (g["xdrawing"].str.upper() != "NO")
        & (~g["xdrawing"].str.upper().str.startswith("KH"))
    )
    g["_drawing_key"] = g["xdrawing"].where(valid_drawing, other=pd.NA)

    if hmbr_df is None or hmbr_df.empty:
        hmbr = pd.DataFrame(columns=["itemcode", "name_100001", "xabc_100001"])
    else:
        hmbr = hmbr_df.drop_duplicates("itemcode").copy()
        hmbr["itemcode"] = hmbr["itemcode"].astype(str)

    merged = g.merge(hmbr, left_on="_drawing_key", right_on="itemcode", how="left")
    merged = merged.drop(columns=["_drawing_key"])

    keep = [
        "itemcode", "name_100001", "name_100009", "item_100009",
        "group_100009", "xabc_100001", "xdrawing",
    ]
    return (
        merged[[c for c in keep if c in merged.columns]]
        .sort_values("item_100009")
        .reset_index(drop=True)
    )

    return out