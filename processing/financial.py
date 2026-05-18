import pandas as pd
from datetime import datetime
from core import queries
from core.db import get_dataframe
import streamlit as st
import numpy as np
from pathlib import Path
import json
from collections import OrderedDict
from typing import Dict, Tuple, List, Set
import re, ast

HERE = Path(__file__).resolve().parent.parent
JSON_PATH = HERE / "data" / "hierarchy.json"

@st.cache_data(ttl=86400)
def _load_raw() -> dict:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(ttl=86400)
def get_filtered_master(zid, excluded_acctypes):
    _sql, _params = queries.get_gl_master(zid)
    df_master = get_dataframe(_sql, _params)

    # Some businesses (e.g. 100007) have NULL ac_type in glmst.
    # Infer the correct type from the ac_code prefix so that IS/BS
    # filtering works identically to businesses with fully-populated glmst.
    _null_mask = df_master['ac_type'].isna() | (df_master['ac_type'].astype(str).str.strip() == '')
    if _null_mask.any():
        _codes = df_master.loc[_null_mask, 'ac_code'].astype(str)
        df_master.loc[_null_mask & _codes.str.startswith('08'),                          'ac_type'] = 'Income'
        df_master.loc[_null_mask & _codes.str.startswith(('04','05','06','07','14','15')), 'ac_type'] = 'Expenditure'
        df_master.loc[_null_mask & _codes.str.startswith(('01','02','03')),               'ac_type'] = 'Asset'
        df_master.loc[_null_mask & _codes.str.startswith(('09','10','11','12','13')),     'ac_type'] = 'Liability'

    return df_master[~df_master['ac_type'].isin(excluded_acctypes)]

@st.cache_data(ttl=86400)
def process_data_month(zid, year, start_month, end_month,label_col, label_df, project=None, account_types=None):
    df_master = get_filtered_master(zid, account_types)
    df_new    = df_master.copy()

    cs_month = start_month          # keep caller’s value
    ce_month = end_month            # ...likewise

    now = datetime.now()

    # 1️⃣  basic bounds
    if not (1 <= cs_month <= 12 and 1 <= ce_month <= 12):
        raise ValueError("start_month and end_month must be between 1 and 12")

    # 2️⃣  caller’s rule: start must be ≤ end
    if cs_month > ce_month:
        raise ValueError("start_month must be less than or equal to end_month")

    # 3️⃣  don’t request the current (incomplete) month
    #     → shift the window back by one if needed
    if ce_month == now.month:
        ce_month -= 1

    # 4️⃣  after adjustment, ensure window is still valid
    if ce_month < cs_month:
        raise ValueError(
            "After adjusting for the current month, end_month is "
            "earlier than start_month; nothing to pull."
        )

    # =================================================================
    # BALANCE-SHEET  (cumulative YTD each month)
    # =================================================================
    if 'Balance Sheet' in label_col:
        # ---- current fiscal year ----
        _sql, _params = queries.get_gl_details(
            zid=zid, project=project, year=year,
            smonth=cs_month, emonth=ce_month,
            is_bs=True, is_project=bool(project))
        df = get_dataframe(_sql, _params)

        max_m = df['month'].max()
        if not df.empty and pd.notna(max_m):
            for i, m in enumerate(range(int(max_m) + 1)):
                part = (df[df['month'] <= m]
                        .groupby('ac_code')['sum'].sum()
                        .reset_index().round(1))
                if i == 0:
                    df_new_c = (df_new
                                .merge(part, on='ac_code', how='left')
                                .fillna(0)
                                .rename(columns={'sum': (year, m)}))
                else:
                    df_new_c = (df_new_c
                                .merge(part, on='ac_code', how='left')
                                .fillna(0)
                                .rename(columns={'sum': (year, m)}))
        else:
            df_new_c = df_new.copy()

        # ---- previous fiscal year ----
        _sql, _params = queries.get_gl_details(
            zid=zid, project=project, year=year - 1,
            smonth=1, emonth=12,
            is_bs=True, is_project=bool(project))
        df = get_dataframe(_sql, _params)

        max_m = df['month'].max()
        if not df.empty and pd.notna(max_m):
            for i, m in enumerate(range(int(max_m) + 1)):
                part = (df[df['month'] <= m]
                        .groupby('ac_code')['sum'].sum()
                        .reset_index().round(1))
                if i == 0:
                    df_new = (df_new
                              .merge(part, on='ac_code', how='left')
                              .fillna(0)
                              .rename(columns={'sum': (year - 1, m)}))
                else:
                    df_new = (df_new
                              .merge(part, on='ac_code', how='left')
                              .fillna(0)
                              .rename(columns={'sum': (year - 1, m)}))
        # ---- combine the 2 FYs and tidy ----
        df_l = (df_new_c
                .merge(df_new, on='ac_code', how='left')
                .drop(columns=[
                    'ac_name_y', 'ac_type_y',
                    'ac_lv1_y', 'ac_lv2_y', 'ac_lv3_y', 'ac_lv4_y',
                    (year, 0), (year - 1, 0)
                ], errors='ignore')
                .rename(columns={
                    'ac_name_x': 'ac_name',
                    'ac_type_x': 'ac_type',
                    'ac_lv1_x' : 'ac_lv1',
                    'ac_lv2_x' : 'ac_lv2',
                    'ac_lv3_x' : 'ac_lv3',
                    'ac_lv4_x' : 'ac_lv4',
                }))
        
        month_cols = sorted(c for c in df_l.columns if isinstance(c, tuple))
        ordered = (
            ["ac_code", "ac_name", "ac_type", "ac_lv1", "ac_lv2", "ac_lv3", "ac_lv4"]
            + month_cols
        )

        df_l = df_l[ordered]

        return (df_l
                .merge(label_df[['ac_lv4', label_col]],
                       on='ac_lv4', how='left')
                .sort_values('ac_type', ascending=True)
                .fillna(0)
                .rename(columns={'Balance Sheet': 'ac_lv5'}))

    # =================================================================
    # INCOME-STATEMENT  (flat pivot, two fiscal years)
    # =================================================================
    _sql, _params = queries.get_gl_details(
        zid=zid, project=project, year=year,
        smonth=cs_month, emonth=ce_month,
        is_bs=False, is_project=bool(project))
    df_curr = get_dataframe(_sql, _params)
    _sql, _params = queries.get_gl_details(
        zid=zid, project=project, year=year - 1,
        smonth=1, emonth=12,
        is_bs=False, is_project=bool(project))
    df_prev = get_dataframe(_sql, _params)

    # ---- concat replaces deprecated append ----
    df = pd.concat([df_curr, df_prev], ignore_index=True)

    # ---- pivot with flat header ----
    df = (df.pivot_table(values='sum',
                         index='ac_code',
                         columns=['year', 'month'],
                         aggfunc='sum')
            .reset_index())

    # ---- ac_code sometimes becomes a 2-level col; flatten it ----
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[0] == 'ac_code' else c for c in df.columns]

    # ---- join to master & labels ----
    df_new = (df
              .merge(df_new[['ac_code', 'ac_name', 'ac_type',
                             'ac_lv1', 'ac_lv2', 'ac_lv3', 'ac_lv4']],
                     on='ac_code', how='right')
              .merge(label_df[['ac_lv4', 'Income Statement']],
                     on='ac_lv4', how='left')
              .fillna(0))

    # ---- drop stray column created by pivot ----
    df_new = df_new.drop(columns=[('ac_code', '', '')], errors='ignore')

    # ---- ('sum', yr, mo) → (yr, mo) ----
    df_new.columns = [col[1:] if isinstance(col, tuple) and col[0] == 'sum' else col
                      for col in df_new.columns]

    # ---- reorder columns exactly as before ----
    month_cols = sorted(c for c in df_new.columns if isinstance(c, tuple))
    ordered = (['ac_code', 'ac_name', 'ac_type',
                'ac_lv1', 'ac_lv2', 'ac_lv3', 'ac_lv4']
               + month_cols + ['Income Statement'])
    df_new = df_new[ordered]

    return df_new.rename(columns={'Income Statement': 'ac_lv5'})

@st.cache_data(ttl=86400)
def process_data(
    zid,
    year_list,                 # list[int] – e.g. [2024, 2023]
    start_month: int,
    end_month: int,
    label_col: str,            # "Balance Sheet" or "Income Statement"
    label_df: pd.DataFrame,
    project=None,
    account_types=None):
    """
    Build a wide table of account balances (one column per fiscal year)
    and attach the chosen label column.

    Returns
    -------
    DataFrame
        master-chart columns + {year₁, year₂, …} + ac_lv5
    """
    # ------------------------------------------------------------------
    # 0 . Reference data: filtered chart of accounts
    # ------------------------------------------------------------------
    df_master = get_filtered_master(zid, account_types)

    # ------------------------------------------------------------------
    # 1 . Pull GL details for every requested fiscal year
    #     and stack them vertically.
    # ------------------------------------------------------------------
    is_bs = 'Balance Sheet' in label_col
    frames = []
    for yr in year_list:
        _sql, _params = queries.get_gl_details(
            zid=zid,
            project=project,
            year=yr,
            smonth=start_month,
            emonth=end_month,
            is_bs=is_bs,
            is_project=bool(project))
        raw = get_dataframe(_sql, _params)
        if raw.empty:
            continue

        yearly = (raw
                  .groupby('ac_code', as_index=False)['sum']
                  .sum()
                  .round(1))
        yearly['year'] = yr                # keep track for pivot
        frames.append(yearly)

    # If we received nothing, create an empty shell so downstream
    # logic still works.
    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        df_wide = (df_all
                   .pivot_table(values='sum',
                                index='ac_code',
                                columns='year',
                                aggfunc='sum')
                   .reset_index())
    else:
        df_wide = pd.DataFrame(columns=['ac_code'] + year_list)

    # ------------------------------------------------------------------
    # 2 . Merge with the master COA and ensure every year column exists
    # ------------------------------------------------------------------
    df_new = (df_master.merge(df_wide, on='ac_code', how='left').fillna(0))
    # 2b. Make sure every year column exists
    for yr in year_list:
        if yr not in df_new.columns:
            df_new[yr] = 0.0
            df_new[yr] = df_new[yr].astype(int)

    year_cols   = [yr for yr in year_list if yr in df_new.columns]
    other_cols  = [c for c in df_new.columns if c not in year_cols]
    df_new      = df_new[other_cols[:7] + year_cols + other_cols[7:]]

    # ------------------------------------------------------------------
    # 3 . Attach label (Income- or Balance-sheet mapping)
    # ------------------------------------------------------------------
    df_new = (df_new
              .merge(label_df[['ac_lv4', label_col]],
                     on='ac_lv4', how='left')
              .sort_values('ac_type', ascending=True))

    rename_map = {'Balance Sheet': 'ac_lv5',
                  'Income Statement': 'ac_lv5'}
    return df_new.rename(columns=rename_map)

def _period_key(col):
    if isinstance(col, tuple) and len(col) == 2:
        return int(col[0]), int(col[1])
    try:
        if isinstance(col, str) and col.startswith("("):
            y, m = ast.literal_eval(col)
            return int(y), int(m)
        nums = re.findall(r"\d+", str(col))
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        if len(nums) == 1:
            return int(nums[0]), 0
    except Exception:
        pass
    return (9999, 99) 

@st.cache_data(ttl=86400)
def _cash_codes_from_json() -> List[str]:
    raw = _load_raw()                            # ← uses your cached loader
    bs_tree = raw.get("Balance Sheet Hierarchy", {})

    want = {
        "0101-CASH & CASH EQUIVALENT",
        "0102-BANK BALANCE",
    }

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in want:
                    for code, _ in v:            # v = [ [code, name], … ]
                        yield code
                else:
                    yield from walk(v)

    return list(walk(bs_tree))

@st.cache_data(ttl=86400)
def _codes_to_exclude() -> Set[str]:
    raw = _load_raw()                                  # already cached JSON
    _EXCLUDE_BUCKETS = {
        "0101-CASH & CASH EQUIVALENT",
        "0102-BANK BALANCE",
        # "1301A-Non-Cash Capital",
        # "1302-Error Adjustment For Retained Earning",
    }
    bs_tree = raw.get("Balance Sheet Hierarchy", {})

    codes: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if key in _EXCLUDE_BUCKETS:
                    # val is a list[ [code, name], … ]
                    for code, _ in val:
                        codes.add(code)
                else:
                    walk(val)                          # recurse
        # nothing to do if node is a list at this level

    walk(bs_tree)
    return codes

LV0_PREFIX_ORDER = [
    "08",  # Revenue
    "04",  # Cost of Goods Sold
    "05",  # Other Direct Expenses
    "06",  # Office & Admin
    "07",  # Sales & Distribution
    "14",  # Purchase Return
    "15",  # Sales Return
]

PREFIX_TO_SECTION_SIGN = {
    "01": ("operating",  1),   # Current Asset
    "02": ("investing",  1),   # Non Current Asset
    "03": ("investing",  1),   # Fixed / Other Asset
    "09": ("operating",  1),   # Current Liability
    "10": ("financing",  1),   # LT / ST Liability
    "11": ("financing",  1),   # Reserve & Fund
    "12": ("financing",  1),   # Long term Liability
    "13": ("financing",  1),   # Owners Equity
}

SPECIAL_FINANCING_CODES = {"09040007", "09040001", "09040003"}
SPECIAL_FIN_BUCKET_L1 = "10-Special Financing (CL Reclass)"
SPECIAL_FIN_BUCKET_L2 = "10-Special Financing (CL Reclass)"

def _prefix_lookup(code):
    code = "" if code is None else str(code)

    if code in {"09040007", "09040001", "09040003"}:
        return "financing", 1

    for pfx, (sect, sgn) in PREFIX_TO_SECTION_SIGN.items():
        if code.startswith(pfx):
            return sect, sgn
    return "operating", 0 

def _prefix_rank(code: str) -> int:
    """Return 0–5 for known IFRS bucket, 99 otherwise (for sorting)."""
    for i, pfx in enumerate(LV0_PREFIX_ORDER):
        if code.startswith(pfx):
            return i
    return 99

def sort_pl_level0(pl_df: pd.DataFrame,
                   code_col: str = "ac_code",
                   name_col: str = "ac_name", selected_perspective: str = "Yearly"):
    """
    Returns
    -------
    pl_sorted : DataFrame  – full detail, IFRS order + Net Profit row
    net_profit: pd.Series  – totals per year
    """
    df = pl_df.copy()
    col_headers = df.select_dtypes(include=[np.number]).columns

    df["_rank"] = df[code_col].apply(_prefix_rank)
    df = (
        df.sort_values(["_rank", code_col], ignore_index=True)
          .drop(columns="_rank")
    )

    if selected_perspective.lower() == "monthly":
        # 1) column order → chronological
        if all(isinstance(c, tuple) and len(c) == 2 for c in col_headers):
            ordered = sorted(col_headers, key=lambda t: (int(t[0]), int(t[1])))
        else:
            def _key(col):
                try:
                    y, m = ast.literal_eval(col)  # parses "('2023','1')" etc.
                    return int(y), int(m)
                except Exception:
                    return (9999, 99)
            ordered = sorted(col_headers, key=_key)

        # 2) totals per month
        month_tot = df[ordered].sum()

        # 3) cumulative within each fiscal year
        years_lbl = [str(c[0]) if isinstance(c, tuple) else str(ast.literal_eval(c)[0])
                     for c in ordered]
        net_profit_m = month_tot.groupby(years_lbl).cumsum()
        net_profit_m.index = ordered     # restore original column labels
    else:
        net_profit_m = None
    net_profit = df[col_headers].sum()

    np_row = pd.DataFrame(
        {code_col: [""], name_col: ["Net Profit/Loss"], **net_profit.to_dict()}
    )

     # ⇓⇓  NEW  extract depreciation row (ac_code == "06360001")  ⇓⇓
    dep_series = (
        df.loc[df[code_col] == "06360001", col_headers]
          .sum()                                   # safe even if code missing
    )
    dep_row = pd.DataFrame(
        {code_col: [""], name_col: ["Depreciation"], **dep_series.to_dict()}
    )

    pl_sorted = pd.concat([df, np_row], ignore_index=True)
    return pl_sorted, net_profit, net_profit_m, dep_row

def append_net_profit_to_bs_level0(bs_df: pd.DataFrame,net_profit: pd.Series,code_col: str = "ac_code",name_col: str = "ac_name") -> pd.DataFrame:
    """
    Return a new B/S dataframe that has two extra rows appended:

        • Net Profit/Loss  – copies the yearly NP series
        • Balance Check    – simple sum of the entire column
                             (now includes the NP row)

    The two new rows have a blank `ac_code`.
    """
    col_headers = net_profit.index.tolist()        # e.g. ["2023", "2024"]

    # --- 1.  build Net-Profit row ---------------------------------------
    np_row = {code_col: "", name_col: "Net Profit/Loss"}
    np_row.update(net_profit.to_dict())
    bs_plus_np = pd.concat(
        [bs_df, pd.DataFrame([np_row])], ignore_index=True
    )

    # --- 2.  build Balance-Check row ------------------------------------
    bal_row = {code_col: "", name_col: "Balance Check"}
    bal_row.update(bs_plus_np[col_headers].sum().to_dict())
    bs_final = pd.concat(
        [bs_plus_np, pd.DataFrame([bal_row])], ignore_index=True
    )

    return bs_final

def cash_open_close(bs_df: pd.DataFrame,code_col: str = "ac_code",name_col: str = "ac_name") -> pd.DataFrame:
    """
    Returns a 2-row DataFrame:
        • Opening Cash & CE
        • Closing Cash & CE
    Columns match the B/S period headers, sorted chronologically.
    Opening for the first period is 0.
    """
    cash_codes = _cash_codes_from_json()

    # slice & sum
    cash_df  = bs_df[bs_df[code_col].isin(cash_codes)]
    num_cols = cash_df.select_dtypes("number").columns
    closing  = cash_df[num_cols].sum(axis=0)

    opening  = closing.shift(1).fillna(0)        # first period → 0

    # assemble output
    out = pd.DataFrame({
        name_col: ["Opening Cash & CE", "Closing Cash & CE"],
        **{col: [opening[col], closing[col]] for col in num_cols}
    })

    # reorder numeric columns chronologically
    sorted_cols = sorted(num_cols, key=_period_key)
    return out[[name_col] + sorted_cols].replace({np.nan: 0})

def _normalize_monthly_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Convert (year, month) tuple columns to a consistent string form: "(YYYY, M)"
    conv = {}
    for c in df.columns:
        if isinstance(c, tuple) and len(c) == 2:
            y, m = c
            conv[c] = f"({int(y)}, {int(m)})"
    return df.rename(columns=conv)

def make_cashflow_statement_level0(pl_df: pd.DataFrame,bs_df: pd.DataFrame, coc_df: pd.DataFrame,code_col: str = "ac_code",name_col: str = "ac_name",tol: float = 4_000.0,selected_perspective: str = "Yearly",):
    """
    Build a multi-year Cash-Flow Statement with
        • Net Profit/Loss
        • detailed Working-Capital rows  →  WC subtotal
        • Cash from Operations
        • detailed CapEx rows           →  CapEx subtotal
        • detailed Financing rows       →  Financing subtotal
        • Net ΔCash
    Returns
    -------
    cfs_df   : DataFrame  (rows in IFRS hierarchy, cols = all years after base)
    balanced : dict[int,bool]  (Assets ≈ Liab+Equity+NP test per year)
    """

    # ---- A  normalise year labels to str & intersect years --------------
    pl_df = pl_df.rename(columns={c: str(c) for c in pl_df.select_dtypes(include=[np.number]).columns})
    bs_df = bs_df.rename(columns={c: str(c) for c in bs_df.select_dtypes(include=[np.number]).columns})

    excl_codes = _codes_to_exclude()
    bs_df = bs_df[~bs_df[code_col].isin(excl_codes)]

    if selected_perspective.lower() == "monthly":

        pl_df = _normalize_monthly_labels(pl_df)
        bs_df = _normalize_monthly_labels(bs_df)
        bs_periods = [c for c in bs_df.columns if isinstance(c, str) and c.startswith("(")]
        pl_periods = [c for c in pl_df.columns if isinstance(c, str) and c.startswith("(")]

        common = [c for c in bs_periods if c in pl_periods]
        col_head = sorted(common, key=_period_key)
    else:
        num_bs = [c for c in bs_df.columns if str(c).isdigit()]   # year-like cols as strings
        common = [c for c in num_bs if c in pl_df.columns]        # overlap with P/L
        col_head = common

    if len(col_head) < 2:
        raise ValueError("No data available for the period selected. Please choose a range that spans at least two years.")

    # ---- B  Net-Profit series (strip any pre-existing NP row) -----------
    pl_no_np  = pl_df[pl_df[name_col] != "Net Profit/Loss"]
    net_profit = (pl_no_np[col_head].sum()) #* -1

    # ---- C  Year-on-year deltas for every ac_code -----------------------
    bs_work   = bs_df.set_index([code_col, name_col])[col_head]

    # --- Special handling: BS Net Profit/Loss should become "Prior Period Net Profit/Loss" and be shifted ---
    np_key = ("", "Net Profit/Loss")  # BS NP row: blank code, name "Net Profit/Loss"

    prior_np_key = ("", "Prior Period Net Profit/Loss")
    prior_np_series = None

    if np_key in bs_work.index:
        # Shift RIGHT: current period shows previous period's NP
        prior_np_series = bs_work.loc[np_key].shift(1).loc[col_head[1:]]  # aligns with delta columns

        # Monthly mode: the BS "Net Profit/Loss" is a YTD cumulative figure.
        # Shifting it right gives the prior month's cumulative NP — which is only
        # a legitimate financing event in January (the year-closing transfer of the
        # full prior year's P/L to retained earnings). For Feb–Dec the cumulative
        # NP is already captured in the top-line "Net Profit/Loss" row from the
        # P/L statement, so including it in financing double-counts it. Zero those
        # months out so only January carries the prior-period value.
        if selected_perspective.lower() == "monthly":
            for col in prior_np_series.index:
                key = _period_key(col)
                month = key[1] if isinstance(key, tuple) and len(key) == 2 else None
                if month != 1:
                    prior_np_series[col] = 0.0

        # Remove the original BS NP row from delta processing
        bs_work = bs_work.drop(index=np_key)

    bs_delta  = bs_work.diff(axis=1).iloc[:, 1:]          # target − base
    bs_delta.columns = col_head[1:]

    # Inject the shifted prior NP row into bs_delta (no diff)
    if prior_np_series is not None:
        bs_delta.loc[prior_np_key, :] = prior_np_series.values

    # ---- D  Allocate deltas & build detailed frames ---------------------
    section_frames = {"operating": [], "investing": [], "financing": []}

    for (code, name), delta_row in bs_delta.iterrows():
        if (code == "" and str(name).strip() == "Prior Period Net Profit/Loss"):
            section, sign = "financing", -1
        else:
            section, sign = _prefix_lookup(code)
        if sign == 0:
            continue
        signed = sign * delta_row  # flip sign for cash logic
        detail = pd.DataFrame(
            {code_col: [code], name_col: [name], **signed.to_dict()}
        )
        section_frames[section].append(detail)

    def _concat_or_empty(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    op_details  = _concat_or_empty(section_frames["operating"])
    inv_details = _concat_or_empty(section_frames["investing"])
    fin_details = _concat_or_empty(section_frames["financing"])

    op_total  = op_details[col_head[1:]].sum() if not op_details.empty else pd.Series(0, index=col_head[1:])
    inv_total = inv_details[col_head[1:]].sum() if not inv_details.empty else pd.Series(0, index=col_head[1:])
    fin_total = fin_details[col_head[1:]].sum() if not fin_details.empty else pd.Series(0, index=col_head[1:])

    def _total_row(label, total_series):
        return pd.DataFrame({code_col: [""], name_col: [label], **total_series.to_dict()})

    # ---- E  Assemble rows in the required order -------------------------
    np_row       = _total_row("Net Profit/Loss",    net_profit.loc[col_head[1:]])
    # Depreciation is NOT added back here: the accumulated-depreciation BS account
    # (03xx codes) is already captured in inv_total via the BS delta, so adding it
    # back in CFO would double-count it and break the cash-flow check.
    wc_total_row = _total_row("Δ Working Capital",  op_total)
    cfo_row      = _total_row("Cash from Operations", op_total + net_profit.loc[col_head[1:]])
    capex_row    = _total_row("CapEx / Investments", inv_total)
    fin_row      = _total_row("Cash from Financing", fin_total)
    ndc_row      = _total_row("Net ΔCash",
                              cfo_row.iloc[0, 2:] + inv_total + fin_total)

    if coc_df is not None and not coc_df.empty:
    # a) normalise coc_df column labels to match cfs_df
        num_cols_coc = coc_df.select_dtypes("number").columns
        num_cols_cfs = col_head[1:]                    # 2023 / (2023,6) …

        # map every period column in cfs → a key (year, month)
        key2cfs = { _period_key(c): c for c in num_cols_cfs }

        # build aligned Series
        close_raw  = coc_df.loc[coc_df[name_col] == "Closing Cash & CE",
                                num_cols_coc].iloc[0]
        open_raw   = coc_df.loc[coc_df[name_col] == "Opening Cash & CE",
                                num_cols_coc].iloc[0]

        opening = pd.Series(0, index=num_cols_cfs, dtype=float)
        closing = pd.Series(0, index=num_cols_cfs, dtype=float)

        for col in num_cols_coc:
            k = _period_key(col)
            if k in key2cfs:
                opening[key2cfs[k]] = open_raw[col]
                closing[key2cfs[k]] = close_raw[col]

        opening_row  = _total_row("Opening Cash & CE", opening)
    else:
        # fallback if coc_df not supplied
        opening      = pd.Series(0, index=col_head[1:], dtype=float)
        closing      = pd.Series(0, index=col_head[1:], dtype=float)
        opening_row  = _total_row("Opening Cash & CE", opening)

    ndc_row      = _total_row("Net ΔCash",
                            cfo_row.iloc[0, 2:] + inv_total + fin_total)

    calc_close   = opening - ndc_row.iloc[0, 2:]
    calc_row     = _total_row("Calculated Closing Cash & CE", calc_close)
    close_row    = _total_row("Closing Cash & CE", closing)
    check_row    = _total_row("Cash-flow Check", calc_close - closing)
    # ───────────────────────────────────────────────────────────────────────

    # Helper: blank spacer line
    def _spacer():
        return pd.DataFrame({code_col: [""], name_col: [""],
                            **{yr: [np.nan] for yr in col_head[1:]}})

    cfs_df = pd.concat(
        [
            np_row,
            op_details, wc_total_row, cfo_row,
            inv_details, capex_row,
            fin_details, fin_row,
            opening_row,
            ndc_row,
            calc_row,
            close_row,
            check_row,
        ],
        ignore_index=True
    )

    if selected_perspective.lower() == "monthly":
        label_cols = [code_col, name_col]

        # --- 1. detect genuine tuples --------------------------------------
        tuple_cols = [c for c in cfs_df.columns if isinstance(c, tuple)]

        if tuple_cols:
            # normal case: columns kept their (year, month) tuple form
            tuple_cols = sorted(tuple_cols, key=lambda t: (int(t[0]), int(t[1])))
            cfs_df = cfs_df[label_cols + tuple_cols]

        else:
            # columns were strings ("('2023','1')" or "2023_1" etc.)
            num_cols = [c for c in cfs_df.columns if c not in label_cols]
            num_cols = sorted(num_cols, key=_period_key)   # ← use shared helper
            cfs_df   = cfs_df[label_cols + num_cols]

    summary_df = (cfs_df.loc[cfs_df[code_col] == ""]
                          .drop(columns=code_col)
                          .reset_index(drop=True))

    return cfs_df, summary_df

@st.cache_data(ttl=86400)
def _build_lookup() -> Tuple[Dict[str, Tuple[str, str]],Dict[str, Tuple[str, str]],Dict[str, list[str]],Dict[str, list[str]]]:
    raw = _load_raw()

    def walk(branch):
        code2, o1, o2 = {}, [], []
        for l2, sub in branch.items():
            o2.append(l2)
            for l1, rows in sub.items():
                o1.append(l1)
                for code, _ in rows:           # rows = [code, name]
                    code2[code] = (l1, l2)
        # de-dup while keeping order
        return (code2,
                list(OrderedDict.fromkeys(o1)),
                list(OrderedDict.fromkeys(o2)))

    is_map, o1_is, o2_is = walk(raw["Income Statement Hierarchy"])
    bs_map, o1_bs, o2_bs = walk(raw["Balance Sheet Hierarchy"])

    return ({"IS": is_map, "BS": bs_map},   
            {"IS": o1_is,  "BS": o1_bs},    
            {"IS": o2_is,  "BS": o2_bs})    

def level_builder(df_l0: pd.DataFrame,
                  section: str,
                  code_col="ac_code",
                  name_col="ac_name") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_L1, df_L2) already summed *and* JSON-ordered."""
    section = section.upper()
    maps, order1_tbl, order2_tbl = _build_lookup()
    lookup = maps[section]     # maps[0]=IS, [1]=BS
    order1 = order1_tbl[section]
    order2 = order2_tbl[section]

    # map each code
    lv1, lv2 = zip(*(lookup.get(c, (np.nan, np.nan)) for c in df_l0[code_col]))
    work = df_l0.copy()
    work["L1"] = lv1
    work["L2"] = lv2
    num = work.select_dtypes("number").columns

    # Level-1
    l1 = (work.groupby("L1", as_index=False)[num].sum()
                .rename(columns={"L1": name_col}))
    l1[name_col] = pd.Categorical(l1[name_col], order1, ordered=True)
    l1 = l1.sort_values(name_col).reset_index(drop=True)

    # Level-2
    l2 = (work.groupby("L2", as_index=False)[num].sum()
                .rename(columns={"L2": name_col}))
    l2[name_col] = pd.Categorical(l2[name_col], order2, ordered=True)
    l2 = l2.sort_values(name_col).reset_index(drop=True)

    return l1, l2

def consolidate_cfs(cfs_l0: pd.DataFrame,
                    level: int,
                    section: str = "BS",
                    debug: bool = False) -> pd.DataFrame:
    """
    Collapse arrow-detail rows in a Level-0 CFS to Level-1 or Level-2.

    Parameters
    ----------
    cfs_l0 : DataFrame from make_cashflow_statement_level0
    level  : 1 or 2
    section: "BS" (arrow rows are B/S codes) or "IS"
    debug  : True → write diagnostic tables via Streamlit

    Returns
    -------
    Consolidated DataFrame
    """
    assert level in (1, 2)
    section = section.upper()

    maps, order1_tbl, order2_tbl = _build_lookup()
    lookup = maps[section]
    order  = order1_tbl[section] if level == 1 else order2_tbl[section]

    # --- ensure our special financing bucket exists in the category order ---
    special_bucket = SPECIAL_FIN_BUCKET_L1 if level == 1 else SPECIAL_FIN_BUCKET_L2
    if special_bucket not in order:
        order = list(order) + [special_bucket]

    # Guess columns
    code_col = cfs_l0.columns[0]     # arrow lives here
    name_col = cfs_l0.columns[1]
    num_cols = cfs_l0.select_dtypes("number").columns

    # ---------------------------------------------------------
    def bucket(code: str):
        """
        Return the level-1 / level-2 bucket for a detail row.
        A detail row is any row whose code_col is a non-empty string.
        Header / subtotal rows have code_col == "".
        """
        if not isinstance(code, str) or code.strip() == "":
            return None                           # header / subtotal
        acct = code.strip()

        # --- override: reclass these current-liability codes into financing bucket ---
        if acct in SPECIAL_FINANCING_CODES:
            return special_bucket

        return lookup.get(acct, (None, None))[level - 1]

    work = cfs_l0.copy()
    work["bucket"] = work[code_col].apply(bucket)

    arrow_mask = work["bucket"].notna()
    # Aggregate numeric columns
    agg = (work.loc[arrow_mask]
                .groupby("bucket", as_index=False)[num_cols].sum())

    agg[code_col] = ""                           # no arrow in output
    agg[name_col] = pd.Categorical(
        agg["bucket"], categories=order, ordered=True
    )
    agg = agg.sort_values(name_col).drop(columns="bucket")
    headers = work.loc[~arrow_mask, [code_col, name_col] + list(num_cols)]

    # ---- reorder & clean up columns ------------------------------------
    agg = agg.drop(columns=code_col)                     # ① drop empty code col
    agg = agg[[name_col] + list(num_cols)]               # ② label first, then years
    agg = agg.reset_index(drop=True)                     # ③ tidy index
    
    # 🆕  keep header / subtotal rows such as “Net Profit/Loss”
    headers = headers.drop(columns=code_col)            # drop empty code col
    headers = headers[[name_col] + list(num_cols)]      # same column order
    headers = headers.reset_index(drop=True)
    headers = headers[headers[name_col].isin([
        "Change in Net Profit/Loss",
        "Prior Period Net Profit/Loss"
    ])]

    # 🆕  combine and return
    final = pd.concat([agg, headers], ignore_index=True)
    return final

def add_np_and_balance_lv1(pl_lv1: pd.DataFrame,bs_lv1: pd.DataFrame,name_col: str = "ac_name",selected_perspective: str = "Yearly") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    • Adds Net Profit/Loss (and Depreciation) to P/L
    • Appends NP to B/S + Balance-Check row
    • Returns updated P/L, updated B/S, net-profit Series, dep_row DF
    """

    # ---------------------------------------------------------------
    # 1️⃣  compute Net Profit/Loss
    # ---------------------------------------------------------------
    col_headers = pl_lv1.select_dtypes("number").columns

    if selected_perspective.lower() == "monthly":
        # ---- chronological order ----------------------------------
        if all(isinstance(c, tuple) and len(c) == 2 for c in col_headers):
            ordered = sorted(col_headers, key=lambda t: (int(t[0]), int(t[1])))
        else:
            def _key(col):
                try:
                    y, m = ast.literal_eval(col)   # parses "('2023','1')" etc.
                    return int(y), int(m)
                except Exception:
                    return (9999, 99)
            ordered = sorted(col_headers, key=_key)

        # ---- monthly totals --------------------------------------
        month_tot = pl_lv1[ordered].sum()

        # ---- cumulative within each FY ---------------------------
        years_lbl = [
            str(c[0]) if isinstance(c, tuple) else str(ast.literal_eval(c)[0])
            for c in ordered
        ]
        net_profit_m = month_tot.groupby(years_lbl).cumsum()
        net_profit_m.index = ordered          # restore original labels
    else:
        net_profit_m = []
        # Yearly (or default) perspective
    net_profit = pl_lv1[col_headers].sum()

    # Net-profit & depreciation rows
    np_row = pd.DataFrame({name_col: ["Net Profit/Loss"], **net_profit.to_dict()})
    dep_series = pl_lv1.loc[
        pl_lv1[name_col].str.contains("0636-Depreciation", na=False),
        col_headers
    ].sum()
    dep_row = pd.DataFrame({name_col: ["Depreciation"], **dep_series.to_dict()})

    pl_out = pd.concat([pl_lv1, np_row], ignore_index=True)

    # ---------------------------------------------------------------
    # 3️⃣  B/S with NP + balance-check
    # ---------------------------------------------------------------
    if selected_perspective.lower() == "monthly":
        np_row_m = pd.DataFrame({name_col: ["Net Profit/Loss"], **net_profit_m.to_dict()})
        bs_with_np = pd.concat([bs_lv1, np_row_m], ignore_index=True)
    else:
        bs_with_np = pd.concat([bs_lv1, np_row], ignore_index=True)
    bal_row = pd.DataFrame(
        {name_col: ["Balance Check"], **bs_with_np[col_headers].sum().to_dict()}
    )
    bs_out = pd.concat([bs_with_np, bal_row], ignore_index=True)

    return pl_out, bs_out, net_profit, dep_row

def _as_year_series(obj, years):
    """Return a 1-D Series indexed by years (strings). Accepts Series/DataFrame/scalar."""
    s = getattr(obj, "squeeze", lambda: obj)()

    if isinstance(s, pd.Series):
        s.index.name = None
        s.index = s.index.map(str)
        return s.reindex(years, fill_value=0)

    if isinstance(s, pd.DataFrame):
        # collapse to 1-D
        if s.shape[0] == 1:
            s = s.iloc[0]
        elif s.shape[1] == 1:
            s = s.iloc[:, 0]
        else:
            # fallback: collapse across rows to get per-year totals
            s = s.sum(axis=0)
        s.index = s.index.map(str)
        return s.reindex(years, fill_value=0)

    # scalar fallback (numpy.float64, int, etc.)
    label = years[0] if years else "Total"
    return pd.Series({label: float(s)}).reindex(years, fill_value=0)

def build_cfs_level1_summary_df(cfs_lv1: pd.DataFrame,net_profit: pd.DataFrame,depreciation_df: pd.DataFrame,coc_df: pd.DataFrame | None = None,name_col: str = "ac_name") -> Tuple[pd.DataFrame, pd.DataFrame]:

    # ------------------------------------------------------------------
    # 1️⃣  normalise NP to Series aligned to cfs columns
    # ------------------------------------------------------------------
    years_raw = cfs_lv1.select_dtypes("number").columns.tolist()  # e.g., Int64Index([2022, 2023,...])
    years_str = [str(y) for y in years_raw]

    np_series  = _as_year_series(net_profit, years_str)

    def _row(label: str, series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({name_col: [label], **series.to_dict()})

    np_row  = _row("Net Profit/Loss", np_series)

    # 3) Tag detail rows
    def bucket_section(label: str) -> str | None:
        if label in ("Change in Net Profit/Loss", "Prior Period Net Profit/Loss"):
            return "financing"
        if not isinstance(label, str) or "-" not in label:
            return None
        return _prefix_lookup(label.split("-")[0])[0]

    work = cfs_lv1.copy()
    work["section"] = work[name_col].apply(bucket_section)

    op_mask  = work["section"] == "operating"
    inv_mask = work["section"] == "investing"
    fin_mask = work["section"] == "financing"

    # 4) Sum with RAW numeric columns, then convert to string-indexed series
    wc_total  = _as_year_series(work.loc[op_mask,  years_raw].sum(), years_str)
    inv_total = _as_year_series(work.loc[inv_mask, years_raw].sum(), years_str)
    fin_total = _as_year_series(work.loc[fin_mask, years_raw].sum(), years_str)

    wc_row  = _row("Δ Working Capital", wc_total)
    cfo_row = _row("Cash from Operations", wc_total + np_series)

    inv_tot  = _row("Cash from Investing", inv_total)
    fin_tot  = _row("Cash from Financing", fin_total)

    # ------------------------------------------------------------------
    # 3️⃣  opening / closing cash reconciliation (uses coc_df)
    # ------------------------------------------------------------------
    if coc_df is not None and not coc_df.empty:
        num_cols_coc = coc_df.select_dtypes("number").columns
        num_cols_cfs = years_raw

        # map period-key → canonical col label in cfs
        key2cfs = { _period_key(c): c for c in num_cols_cfs }

        close_raw = coc_df.loc[coc_df[name_col] == "Closing Cash & CE",
                               num_cols_coc].iloc[0]
        open_raw  = coc_df.loc[coc_df[name_col] == "Opening Cash & CE",
                               num_cols_coc].iloc[0]

        opening = pd.Series(0, index=num_cols_cfs, dtype=float)
        closing = pd.Series(0, index=num_cols_cfs, dtype=float)

        for col in num_cols_coc:
            k = _period_key(col)
            if k in key2cfs:
                opening[key2cfs[k]] = open_raw[col]
                closing[key2cfs[k]] = close_raw[col]
    else:
        opening = closing = pd.Series(0, index=years_in_cfs, dtype=float)

    opening_row = _row("Opening Cash & CE", opening)

    ndc_row = _row("Net ΔCash",
                   cfo_row.iloc[0, 1:] + inv_total + fin_total)

    calc_close = opening - ndc_row.iloc[0, 1:]
    calc_row   = _row("Calculated Closing Cash & CE", calc_close)
    close_row  = _row("Closing Cash & CE", closing)
    check_row  = _row("Cash-flow Check", calc_close - closing)

    # ------------------------------------------------------------------
    # 4️⃣  assemble full Level-1 CFS in correct order
    # ------------------------------------------------------------------
    full_df = pd.concat(
        [
            np_row,
            work.loc[op_mask],
            wc_row,
            cfo_row,
            work.loc[inv_mask],
            inv_tot,
            work.loc[fin_mask],
            fin_tot,
            opening_row,
            ndc_row,
            calc_row,
            close_row,
            check_row,
        ],
        ignore_index=True
    ).drop(columns="section")

    summary_labels = [
        "Net Profit/Loss", "Δ Working Capital",
        "Cash from Operations", "Cash from Investing",
        "Cash from Financing","Opening Cash & CE",
        "Net ΔCash","Calculated Closing Cash & CE",
        "Closing Cash & CE","Cash-flow Check"
    ]
    sum_df = (full_df[full_df[name_col].isin(summary_labels)]
                     .reset_index(drop=True))

    return full_df, sum_df

def _get_series(df: pd.DataFrame, bucket: str) -> pd.Series:
    num_cols = df.select_dtypes("number").columns
    row = df.loc[df["ac_name"] == bucket, num_cols]
    if row.empty:
        return pd.Series(0, index=num_cols)
    return row.iloc[0]

def add_np_and_balance_lv2(pl_lv2: pd.DataFrame,bs_lv2: pd.DataFrame,name_col: str = "ac_name",selected_perspective: str = "Yearly") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Build Gross Profit, EBIT, Net Profit for Level-2 P/L and sync B/S.

    Returns
    -------
    pl_out : P/L with Gross Profit, EBIT, Net Profit rows appended
    bs_out : B/S with Net Profit + Balance Check appended
    net_profit_series : pd.Series (year columns)
    """

    # numeric year columns (assumed identical in P/L & B/S)
    col_headers = pl_lv2.select_dtypes("number").columns

    # ---- 1️⃣  fetch needed bucket series -------------------------------
    rev  = _get_series(pl_lv2, "08-Revenue")
    cogs = _get_series(pl_lv2, "04-Cost of Goods Sold")
    oth_dir  = _get_series(pl_lv2, "05-OTHERS DIRECT EXPENSES")
    oth_exp  = _get_series(pl_lv2, "05-Other Expenses")
    office   = _get_series(pl_lv2, "06-Office & Administrative Expenses")
    salesd   = _get_series(pl_lv2, "07-Sales & Distribution Expenses")
    int_exp  = _get_series(pl_lv2, "06A-Interest Expenses")
    tax_exp  = _get_series(pl_lv2, "06B-Taxes & Duties Expenses")

    # ---- 2️⃣  build derived lines --------------------------------------
    gross_profit = rev + cogs              # COGS already negative
    ebit = (gross_profit + oth_dir + oth_exp +
            office + salesd)
    net_profit = ebit + int_exp + tax_exp

    if selected_perspective.lower() == "monthly":
        ordered = sorted(col_headers, key=_period_key)      # shared helper
        net_profit_m = net_profit[ordered]

        # group columns by fiscal year and cum-sum within each year
        years_lbl = []
        for col in ordered:
            if isinstance(col, tuple) and len(col) == 2:        # (2024, 2)
                years_lbl.append(str(col[0]))
            elif isinstance(col, str) and col.startswith("("):  # "('2024','2')"
                years_lbl.append(str(ast.literal_eval(col)[0]))
            else:                                               # "2024_02" etc.
                import re
                nums = re.findall(r"\d+", str(col))
                years_lbl.append(nums[0] if nums else "")
        net_profit_m = net_profit_m.groupby(years_lbl).cumsum()
    # ---- 3️⃣  helper to create a single-row DF -------------------------
    def _row(label: str, series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {name_col: [label], **series.to_dict()}
        )

    rows_to_add = [
        _row("Gross Profit",gross_profit),
        _row("EBIT",ebit),
        _row("Net Profit/Loss",net_profit),
    ]

    # ---- 4️⃣  append & reorder P/L -------------------------------------
    pl_out = pd.concat([pl_lv2] + rows_to_add, ignore_index=True)

    desired_order = [
        "08-Revenue",
        "04-Cost of Goods Sold",
        "Gross Profit",
        "05-OTHERS DIRECT EXPENSES",
        "05-Other Expenses",
        "06-Office & Administrative Expenses",
        "07-Sales & Distribution Expenses",
        "EBIT",
        "06A-Interest Expenses",
        "06B-Taxes & Duties Expenses",
        "Net Profit/Loss",
    ]
    cat = pd.Categorical(pl_out[name_col], categories=desired_order, ordered=True)
    pl_out = pl_out.sort_values(name_col, key=lambda s: cat).reset_index(drop=True)

    # ---- 5️⃣  augment B/S ----------------------------------------------
    if selected_perspective.lower() == "monthly":
        np_row_bs = _row("Net Profit/Loss", net_profit_m)
    else:
        np_row_bs = _row("Net Profit/Loss", net_profit)
    bs_with_np = pd.concat([bs_lv2, np_row_bs], ignore_index=True)

    bal_row = _row("Balance Check", bs_with_np[col_headers].sum())
    bs_out = pd.concat([bs_with_np, bal_row], ignore_index=True)

    return pl_out, bs_out, net_profit

def build_cfs_level2_summary(cfs_lv2: pd.DataFrame,net_profit_lv2: pd.DataFrame,depreciation_lv1_row: pd.DataFrame,coc_df: pd.DataFrame | None = None,name_col: str = "ac_name") -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    full_df : Level-2 cash-flow with reconciliation lines
    sum_df  : 6-row summary
    """
        # Use name-based year detection and ensure strings
    # Derive raw & string years
    years_raw = cfs_lv2.select_dtypes("number").columns.tolist()
    years_str = [str(c) for c in years_raw]

    # 1) Net-profit as aligned Series (string-indexed)
    np_series  = _as_year_series(net_profit_lv2, years_str)

    def _row(label: str, series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({name_col: [label], **series.to_dict()})

    np_row  = _row("Net Profit/Loss", np_series)

    def section_of(label: str) -> str | None:
        if label in ("Change in Net Profit/Loss", "Prior Period Net Profit/Loss"):
            return "financing"
        if not isinstance(label, str) or "-" not in label:
            return None
        return _prefix_lookup(label.split("-")[0])[0]

    work = cfs_lv2.copy()
    work["section"] = work[name_col].apply(section_of)

    op_mask = work["section"] == "operating"
    inv_mask = work["section"] == "investing"
    fin_mask = work["section"] == "financing"

    # Sum with RAW numeric columns → then normalize to string-indexed series
    wc_tot  = _as_year_series(work.loc[op_mask,  years_raw].sum(), years_str)
    inv_tot = _as_year_series(work.loc[inv_mask, years_raw].sum(), years_str)
    fin_tot = _as_year_series(work.loc[fin_mask, years_raw].sum(), years_str)

    wc_row  = _row("Δ Working Capital", wc_tot)
    cfo_row = _row("Cash from Operations", wc_tot + np_series)

    inv_row = _row("Cash from Investing",  inv_tot)
    fin_row = _row("Cash from Financing",  fin_tot)

    # ── 3️⃣  Opening / Closing Cash reconciliation (uses coc_df) ───────
    if coc_df is not None and not coc_df.empty:
        num_cols_coc = coc_df.select_dtypes("number").columns
        key2cfs = { _period_key(c): c for c in years_raw }

        close_raw = coc_df.loc[coc_df[name_col] == "Closing Cash & CE",
                               num_cols_coc].iloc[0]
        open_raw  = coc_df.loc[coc_df[name_col] == "Opening Cash & CE",
                               num_cols_coc].iloc[0]

        opening = pd.Series(0, index=years_raw, dtype=float)
        closing = pd.Series(0, index=years_raw, dtype=float)

        for col in num_cols_coc:
            k = _period_key(col)
            if k in key2cfs:
                opening[key2cfs[k]] = open_raw[col]
                closing[key2cfs[k]] = close_raw[col]
    else:
        opening = closing = pd.Series(0, index=col_headers, dtype=float)

    opening_row = _row("Opening Cash & CE", opening)

    ndc_row   = _row("Net ΔCash",
                     cfo_row.iloc[0, 1:] + inv_tot + fin_tot)

    calc_close = opening - ndc_row.iloc[0, 1:]
    calc_row   = _row("Calculated Closing Cash & CE", calc_close)
    close_row  = _row("Closing Cash & CE", closing)
    check_row  = _row("Cash-flow Check", calc_close - closing)

    # ── 4️⃣  assemble full statement in IFRS order ─────────────────────
    full_df = pd.concat(
        [
            np_row,
            work.loc[op_mask],  wc_row,  cfo_row,
            work.loc[inv_mask], inv_row,
            work.loc[fin_mask], fin_row,
            opening_row,
            ndc_row,
            calc_row, close_row, check_row
        ],
        ignore_index=True
    ).drop(columns="section")

    summary_labels = [
        "Net Profit/Loss", "Δ Working Capital",
        "Cash from Operations", "Cash from Investing",
        "Cash from Financing","Opening Cash & CE",
        "Net ΔCash","Calculated Closing Cash & CE",
        "Closing Cash & CE","Cash-flow Check"
    ]
    sum_df = (full_df[full_df[name_col].isin(summary_labels)]
                     .reset_index(drop=True))

    return full_df, sum_df


# =============================================================================
# LEVEL S — Management View
# =============================================================================

def _ls_num_cols(df: pd.DataFrame) -> pd.Index:
    """Return the numeric (period) columns of a DataFrame."""
    return df.select_dtypes("number").columns


def _ls_sum(df: pd.DataFrame, codes: list, code_col: str = "ac_code") -> pd.Series:
    """
    Sum Level 0 numeric values for the given ac_codes and NEGATE.
    Flips from Level 0 convention (revenue=negative, expense=positive)
    to Level S convention (revenue=positive, expense=negative).
    """
    num = _ls_num_cols(df)
    mask = df[code_col].isin(codes)
    return -(df.loc[mask, num].sum())


def _ls_sum_raw(df: pd.DataFrame, codes: list, code_col: str = "ac_code") -> pd.Series:
    """
    Sum Level 0 numeric values for given ac_codes WITHOUT sign flip.
    Used for rows with mixed signs (e.g. 0501-Others Direct Expenses).
    """
    num = _ls_num_cols(df)
    mask = df[code_col].isin(codes)
    return df.loc[mask, num].sum()


def _ls_sum_raw_by_name(
    df: pd.DataFrame,
    codes: list,
    names: list,
    code_col: str = "ac_code",
    name_col: str = "ac_name",
) -> pd.Series:
    """
    Sum Level 0 numeric values filtered by BOTH ac_code AND ac_name WITHOUT sign flip.
    Used for the consolidated BS/CFS where the same ac_code (e.g. 01030001) appears
    with different ac_names to distinguish external from internal AR groups.
    """
    num = _ls_num_cols(df)
    mask = df[code_col].isin(codes) & df[name_col].isin(names)
    return df.loc[mask, num].sum()


def _ls_prefix_sum(df: pd.DataFrame, prefix: str, code_col: str = "ac_code") -> pd.Series:
    """Sum all codes that start with the given prefix (raw, no sign flip — used for BS)."""
    num = _ls_num_cols(df)
    mask = df[code_col].str.startswith(prefix, na=False)
    return df.loc[mask, num].sum()


def _ls_row(name: str, series: pd.Series, name_col: str = "ac_name") -> pd.DataFrame:
    return pd.DataFrame({name_col: [name], **series.to_dict()})


def _ls_zero(num_cols: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=num_cols)


def compute_vat_is_rows(
    gl_vat: pd.DataFrame,
    period_cols,
    selected_perspective: str = "Yearly",
) -> dict:
    """
    Derive the four VAT/Tax informational rows for the Level S IS from a
    raw gldetail slice containing ac_code IN (1050007, 6290001).

    Rules
    -----
    From 01050007 (AT(+)VAT Rebate Claim — BS account):
      vat_rebate  = CPAY totals + JV totals where value < 0   → shown negative (expense)
      vat_cash    = CPAY totals only                          → shown negative (expense)

    From 06290001 (Net VAT Expenses Cash — P&L account):
      vat_office  = CPAY totals                               → shown negative (expense)
      others_tax  = ADJV totals, sign flipped                 → shown positive (tax recovered/adjustment)

    Parameters
    ----------
    gl_vat : DataFrame
        Columns: ac_code, year, month, voucher, value
    period_cols : Index or list
        The numeric period columns from pl_raw (years for Yearly, (yr,mo) for Monthly).
    selected_perspective : str

    Returns
    -------
    dict with keys: 'vat_rebate', 'vat_cash', 'vat_office', 'others_tax'
        Each value is a pd.Series indexed by period_cols.
    """
    zero = pd.Series(0.0, index=period_cols)

    if gl_vat is None or gl_vat.empty:
        return {k: zero.copy() for k in ("vat_rebate", "vat_cash", "vat_office", "others_tax")}

    df = gl_vat.copy()
    df["ac_code"] = df["ac_code"].astype(str).str.strip()
    df["pfx"] = df["voucher"].astype(str).str[:4].str.upper()

    is_monthly = selected_perspective.strip().lower() == "monthly"

    def _period(row):
        yr, mo = int(row["year"]), int(row["month"])
        return (yr, mo) if is_monthly else yr

    df["period"] = df.apply(_period, axis=1)

    def _agg(mask) -> pd.Series:
        sub = df[mask].groupby("period")["value"].sum()
        return sub.reindex(period_cols, fill_value=0.0)

    # ── 01050007 ─────────────────────────────────────────────────────────────
    m07 = df["ac_code"] == "01050007"
    cpay_07  = _agg(m07 & (df["pfx"] == "CPAY"))
    jv_neg07 = _agg(m07 & (df["pfx"] == "JV--") & (df["value"] < 0))

    # Both components are positive amounts in L0 (asset increases / outflows)
    # → negate to show as expense (negative) in Level S
    vat_rebate = -(cpay_07 + jv_neg07)
    vat_cash   = -cpay_07

    # ── 06290001 ─────────────────────────────────────────────────────────────
    m01 = df["ac_code"] == "06290001"
    cpay_01 = _agg(m01 & (df["pfx"] == "CPAY"))
    adjv_01 = _agg(m01 & (df["pfx"] == "ADJV"))

    # CPAY on 06290001: positive in L0 = cash paid for VAT → negative in IS
    vat_office  = -cpay_01
    # ADJV on 06290001: flip sign → positive = tax recovered / adjustment credit
    others_tax  = -adjv_01

    return {
        "vat_rebate": vat_rebate,
        "vat_cash":   vat_cash,
        "vat_office": vat_office,
        "others_tax": others_tax,
    }


def build_pl_level_s(
    pl_raw: pd.DataFrame,
    selected_perspective: str = "Yearly",
    code_col: str = "ac_code",
    name_col: str = "ac_name",
    vat_rows: dict = None,
) -> pd.DataFrame:
    """
    Build the Level S (Management View) Income Statement.

    Parameters
    ----------
    pl_raw : DataFrame
        Raw Level 0 P&L data — must still contain ac_code (before drop_cols strip).
    selected_perspective : str
        'Yearly' or 'Monthly'.

    Returns
    -------
    DataFrame with columns [ac_name] + period columns.
    Sign convention: positive = revenue/profit, negative = cost/loss.
    """
    num = _ls_num_cols(pl_raw)
    z   = _ls_zero(num)

    def s(codes):  return _ls_sum(pl_raw, codes, code_col)      # negate
    def r(codes):  return _ls_sum_raw(pl_raw, codes, code_col)  # raw

    # ── Direct rows ──────────────────────────────────────────────────────────
    revenue       = s(["08010001"])
    others_rev    = s(["08010002","08020001","08020002","08020003","08030001","08030002",
                        "08040001","08050001","08050002","08050003","08050004",
                        "08050005","08050006","08050007","08050008","08050009",
                        "08050010","08050011",
                        "15010001"])   # Sales Return — revenue deduction (pre-2016 only)
    mrp_discount  = s(["07080002"])

    # Adjusted Revenue = Revenue + MRP Discount (display only, no effect on other rows)
    adj_revenue   = revenue + mrp_discount

    # COGS: purchase-related codes only. 05010001/03/04 (0501-Others Direct
    # Expenses) are reported on their own dedicated IS line below GP.
    # 04010008 (Customs Tax & VAT on imports) and 04010011 included as
    # purchase-related costs.
    cogs          = s(["04010020","04010002","04010004","04010008","04010011"])

    # SGA includes Director Remuneration (06150001–3) for full reconciliation.
    sga           = s(["06010001","06010002","06010003","06010004","06010005",
                        "06010006","06010007","06010008","06010009","06020001","06030001","06030002",
                        "06030003","06030004","06030005","06030006","06030007",
                        "06030008","06030009","06030010","06030011","06030012",
                        "06030013","06030014","06040001","06040002","06040003",
                        "06040004","06040005","06040006","06040007","06040008",
                        "06060001","06060002","06070001","06070002","06070003",
                        "06100001","06160001","06160002","06160003","06160004",
                        "06160005","06160007","06160008","06160009","06160010",
                        "06170001","06180001","06180002","06180003","06180004",
                        "06190001","06190002","06190003","06190004","06190005",
                        "06200001","06210001","06210002","06210003","06220001",
                        "06220002","06220003","06220004","06220005","06220006",
                        "06220007","06220008","06220009","06220010","06220011",
                        "06220012","06220013","06220014","06230001","06230002","06230003","06230004",
                        "06240001","06240002","06250001","06250002","06250003",
                        "06260001","06260002","06260003","06270001","06280001",
                        "06310001","06310002","06310003","06310004","06310005",
                        "06310006","06310007","06320001","06320002","06340001",
                        "06360001","06370001","06380001","06390001","06290003",
                        "06290004","05010002",
                        "06160006"])   # Paper bill — previously missing from SGA
    salary        = s(["06120001","06120002"])
    bonus         = s(["06130001","06130002"])
    overtime      = s(["06140001","06140002"])
    director_rem  = s(["06150001","06150002","06150003"])

    discount_paid = s(["07080001"])
    sd_expenses   = s(["07010001","07010002","07010003","07010004","07010005",
                        "07010006","07010007","07010008","07010009","07020001","07020002","07020003","07020004",
                        "07030001","07030002","07030003","07040001","07040002",
                        "07040003","07040005","07050001","07050002","07050003","07060001","07060002",
                        "07060003","07060004","07060005","07090001","07100001",
                        "07100002","07110001","07110002","07110003","07110004",
                        "07120001","07120002","07120003","07120004","07120005",
                        "07130001","07130002","07130003","07140001"])

    # 0501-Others Direct Expenses: 05010002 stays in sga (sign-corrected there).
    # 05010001/03/04 reported here — no longer in COGS.
    others_direct = s(["05010001","05010003","05010004"])

    bank_interest = s(["06300001","06300002","06300003"])
    loan_interest = s(["06330001","06350001","06350002"])

    net_vat_cash  = s(["06290001"])
    income_tax    = s(["06290002"])

    # ── Calculated rows ──────────────────────────────────────────────────────
    # GP = Revenue + Others Revenue + MRP Discount + COGS (all revenue lines above the line)
    gross_profit   = revenue + others_rev + mrp_discount + cogs
    total_sga      = sga + salary + bonus + overtime + director_rem
    total_sd       = discount_paid + sd_expenses
    # others_rev/mrp_discount now in GP; EBITDA = GP + opex lines below
    ebitda         = gross_profit + total_sga + total_sd + others_direct
    total_interest = bank_interest + loan_interest
    vat_tax_total  = net_vat_cash + income_tax
    net_income     = ebitda + total_interest + vat_tax_total

    # ── VAT informational rows (display-only, from gl_vat direct query) ─────
    _vr = vat_rows or {}
    _vat_rebate = _vr.get("vat_rebate", z.copy())
    _vat_cash   = _vr.get("vat_cash",   z.copy())
    _vat_office = _vr.get("vat_office", z.copy())
    _others_tax = _vr.get("others_tax", z.copy())

    rows = [
        _ls_row("Revenue",                           revenue,        name_col),
        _ls_row("Others Revenue",                    others_rev,     name_col),
        _ls_row("MRP Discount",                      mrp_discount,   name_col),
        _ls_row("Adjusted Revenue (Pending)",         adj_revenue,    name_col),
        _ls_row("COGS",                              cogs,           name_col),
        _ls_row("Gross Profit",                      gross_profit,   name_col),
        _ls_row("SG&A",                              sga,            name_col),
        _ls_row("0612-Salary Expenses",              salary,         name_col),
        _ls_row("0613-Employee Bonus",               bonus,          name_col),
        _ls_row("0614-Overtime",                     overtime,       name_col),
        _ls_row("0615-Director Remuneration",        director_rem,   name_col),
        _ls_row("Total SG&A",                        total_sga,      name_col),
        _ls_row("0708-Discount Paid",                discount_paid,  name_col),
        _ls_row("Sales & Distribution Expenses",     sd_expenses,    name_col),
        _ls_row("Total Sales & Distribution",        total_sd,       name_col),
        _ls_row("0501-Others Direct Expenses",       others_direct,  name_col),
        _ls_row("EBITDA",                            ebitda,         name_col),
        _ls_row("0630-Bank Interest & Charges",      bank_interest,  name_col),
        _ls_row("0633-Interest-Loan",                loan_interest,  name_col),
        _ls_row("Total Interest & Charges",          total_interest, name_col),
        _ls_row("VAT Expenses from Rebate (A)",      _vat_rebate,    name_col),
        _ls_row("VAT through Cash (i)",              _vat_cash,      name_col),
        _ls_row("Others Company Tax (ii)",           _others_tax,    name_col),
        _ls_row("VAT Expenses Office (iii)",         _vat_office,    name_col),
        _ls_row("Net VAT Expenses Cash (B)",         net_vat_cash,   name_col),
        _ls_row("0629-Income Tax Expenses (C)",      income_tax,     name_col),
        _ls_row("0629-VAT & Tax Total (A+B+C)",      vat_tax_total,  name_col),
        _ls_row("Net Income",                        net_income,     name_col),
    ]

    return pd.concat(rows, ignore_index=True)


# ── Per-business Level S BS code routing ─────────────────────────────────────
# Each entry overrides which ac_codes map to which AR/AP buckets.
# Businesses not listed use the default routing below.
_BS_LEVEL_S_CODE_ROUTES: dict = {
    "100000": {
        # 01030002 = Internal Receivable; 01030003 = Recognized Agent (Dealers)
        "ar_main":      ["01030001"],
        "ar_internal":  ["01030002"],
        "ar_agent":     ["01030003"],
        "ar_local":     [],
        "ar_defaulted": [],
        # 09030002 = Internal AP; 09030003 = International AP
        "ap_local":     ["09030001"],
        "ap_internal":  ["09030002"],
        "ap_intl":      ["09030003"],
    },
    "100001": {
        # 01030003 = Defaulted Receivables (not local AR)
        "ar_main":      ["01030001"],
        "ar_internal":  [],
        "ar_agent":     ["01030002"],
        "ar_local":     [],
        "ar_defaulted": ["01030003"],
        "ap_local":     ["09030001", "09030002"],
        "ap_internal":  ["09030003"],
        "ap_intl":      ["09030004"],
    },
    "100005": {
        # 09030002 = Internal AP; 09030003 = International AP
        "ar_main":      ["01030001"],
        "ar_internal":  [],
        "ar_agent":     ["01030002"],
        "ar_local":     ["01030003"],
        "ar_defaulted": [],
        "ap_local":     ["09030001"],
        "ap_internal":  ["09030002"],
        "ap_intl":      ["09030003"],
    },
}

_BS_LEVEL_S_CODE_ROUTES_DEFAULT: dict = {
    "ar_main":      ["01030001"],
    "ar_internal":  [],
    "ar_agent":     ["01030002"],
    "ar_local":     ["01030003"],
    "ar_defaulted": [],
    "ap_local":     ["09030001", "09030002"],
    "ap_internal":  ["09030003"],
    "ap_intl":      ["09030004"],
}

# Consolidated routing: merges code variants from different businesses into
# unified group-level buckets.  No internal AR/AP bucket — those intercompany
# amounts net at the total assets / liabilities level via natural sign netting.
_BS_LEVEL_S_CODE_ROUTES["consolidated"] = {
    "ar_main":      ["01030001"],
    "ar_internal":  [],
    "ar_agent":     ["01030002", "01030003"],   # both Recognised Agent variants
    "ar_local":     [],
    "ar_defaulted": [],
    "ap_local":     ["09030001", "09030002"],   # incl. 100001 Construction Materials AP
    "ap_internal":  [],
    "ap_intl":      ["09030003", "09030004"],   # both AP International variants
}


def build_bs_level_s(
    bs_raw: pd.DataFrame,
    net_profit: pd.Series,
    zid=None,
    code_col: str = "ac_code",
    name_col: str = "ac_name",
) -> pd.DataFrame:
    """
    Build the Level S (Management View) Balance Sheet.

    Parameters
    ----------
    bs_raw : DataFrame
        Raw Level 0 BS with ac_code still present.
    net_profit : Series
        Net Income per period from build_pl_level_s() (Level S sign: positive=profit).
    zid : str or int, optional
        Business ZID used to look up per-business AR/AP code routing.
        If None or not in the routing table, the default routing is used.

    Returns
    -------
    DataFrame with [ac_name] + period columns.
    Level 0 accounting sign convention: assets positive, liabilities negative.
    """
    num = _ls_num_cols(bs_raw)
    z   = _ls_zero(num)

    def d(codes):  return _ls_sum_raw(bs_raw, codes, code_col)
    def pfx(p):    return _ls_prefix_sum(bs_raw, p, code_col)

    # ── Per-business AR/AP code routing ──────────────────────────────────────
    _routes = _BS_LEVEL_S_CODE_ROUTES.get(
        str(zid) if zid is not None else "",
        _BS_LEVEL_S_CODE_ROUTES_DEFAULT,
    )

    # ── Current Assets ───────────────────────────────────────────────────────
    cash_ce       = d(["01010001","01010003","01010004","01010006","01010008"])
    bank_bal      = d(["01020001","01020002","01020003"])
    if str(zid) == "consolidated":
        # Consolidated C2 BS AR mapping:
        #   ar_main      → (01030001, "Accounts Receivable")
        #                  external ZIDs: 100001 / 100005 / 100000
        #   ar_internal  → (01030001, "Accounts Receivable (Internal)") [ARAP ZIDs]
        #                  + (01030002, "Accounts Receivable (Previous) GTC & Zepto") [100000]
        #                  both are internal/legacy receivables grouped together
        #   ar_agent     → (01030002, "Recognized Agent")
        #                  external trade agent AR
        #   ar_local     → zero (no local AR bucket at consolidated level)
        #   ar_defaulted → (01030003, "Accounts Receivable for future Collections")
        #                  100001's long-term / defaulted receivable bucket
        def dn(codes, names):
            return _ls_sum_raw_by_name(bs_raw, codes, names, code_col, "ac_name")
        ar_main      = dn(["01030001"], ["Accounts Receivable"])
        ar_internal  = (dn(["01030001"], ["Accounts Receivable (Internal)"])
                        + dn(["01030002"], ["Accounts Receivable (Previous) GTC & Zepto"]))
        # "Recognized Agent" rows appear under 01030002 (100001/100005) and also
        # under 01030003 for ZID 100000 — both must be captured in ar_agent.
        ar_agent     = dn(["01030002", "01030003"], ["Recognized Agent"])
        ar_local     = z
        ar_defaulted = dn(["01030003"], ["Accounts Receivable for future Collections"])
    else:
        ar_main      = d(_routes["ar_main"])
        ar_internal  = d(_routes["ar_internal"])
        ar_agent     = d(_routes["ar_agent"])
        ar_local     = d(_routes["ar_local"])
        ar_defaulted = d(_routes["ar_defaulted"])
    ar_total      = ar_main + ar_internal + ar_agent + ar_local + ar_defaulted
    prepaid       = d(["01040005"])
    advance       = d(["01050001","01050002","01050003","01050004",
                        "01050005","01050006","01050007","01050008"])
    stock         = d(["01060003","01060001","01060002"])
    total_ca      = cash_ce + bank_bal + ar_total + prepaid + advance + stock

    # ── Other Assets ─────────────────────────────────────────────────────────
    def_capex     = d(["02010001"])
    gift_items    = d(["02020001"])
    loan_hosp     = d(["02030001"])
    loan_surma    = d(["02030002"])
    security_dep  = d(["02040001"])
    loan_others   = d(["02050001","02050002","02050003","02050004","02050005",
                        "02050006","02050007","02050008","02050009","02050010",
                        "02050011","02050012","02050013","02050014","02050015",
                        "02050016","02050017"])
    other_invest  = d(["02060001"])
    total_oa      = (def_capex + gift_items + loan_hosp + loan_surma +
                     security_dep + loan_others + other_invest)

    # ── Fixed Assets — aggregate by prefix ───────────────────────────────────
    fa_office_eq  = pfx("03010")
    fa_corp_eq    = pfx("03020")
    fa_furn       = pfx("03030")
    fa_trade_veh  = pfx("03040")
    fa_priv_veh   = pfx("03050")
    fa_plant      = pfx("03060")
    fa_intang     = pfx("03070")
    fa_land       = pfx("03080")
    total_fa      = (fa_office_eq + fa_corp_eq + fa_furn + fa_trade_veh +
                     fa_priv_veh + fa_plant + fa_intang + fa_land)

    total_assets  = total_ca + total_oa + total_fa

    # ── Current Liabilities ───────────────────────────────────────────────────
    accrued       = d(["09010001","09010002","09010003","09010004",
                        "09010005","09010006"])
    if str(zid) == "consolidated":
        # AP code mapping varies by ZID — 100001 uses 09030003 for Internal and
        # 09030004 for International, while all other ZIDs use 09030002/09030003.
        # Filter by ac_name so each row lands in the correct bucket regardless of code.
        #   ap_local:    all 09030001 (handles "Accounts Payable(Local)" and
        #                "Accounts Payable" name variants for ZIDs 100004/100006)
        #                + 100001's 09030002 "Constraction Materials Suppliers(M.B)"
        #                which goes to _allother_bs_rows (not in arap_bs) and should
        #                be classified as local AP per 100001's per-ZID route.
        #   ap_internal: 09030002 (most ZIDs) + 09030003 (100001) named "Accounts Payable(Internal)"
        #   ap_intl:     09030003 (most ZIDs) + 09030004 (100001) named "Accounts Payable(International)"
        ap_local    = (d(["09030001"])   # all 09030001 regardless of name variant
                       + dn(["09030002"], ["Constraction Materials Suppliers(M.B)"]))
        ap_internal = dn(["09030002", "09030003"],
                         ["Accounts Payable(Internal)"])
        ap_intl     = dn(["09030003", "09030004"],
                         ["Accounts Payable(International)"])
    else:
        ap_local    = d(_routes["ap_local"])
        ap_internal = d(_routes["ap_internal"])
        ap_intl     = d(_routes["ap_intl"])
    ap_total      = ap_local + ap_internal + ap_intl
    money_agent   = d(["09040001","09040002","09040003","09040004",
                        "09040005","09040007"])
    recon_liab    = d(["09040006"])
    cf_liab       = pfx("09050")   # all C&F / shipping / customs liability codes
    others_liab   = d(["09060001"])
    total_cl      = accrued + ap_total + money_agent + recon_liab + cf_liab + others_liab

    # ── Short-Term Bank Loans (B) ─────────────────────────────────────────────
    stb_loan      = d(["10010001","10010002","10010003","10010004",
                        "10010005","10010006","10010007","10010008"])

    # ── Short-Term Loans (other) ──────────────────────────────────────────────
    st_loan       = d(["10020001","10020002","10020003","10020004","10020005",
                        "10020006","10020007","10020008","10020009","10020010",
                        "10020011","10020012","10020013","10020014","10020015",
                        "10020016","10020017"])
    total_stl     = stb_loan + st_loan

    # ── Long-Term Bank Loans (E) ──────────────────────────────────────────────
    ltl_loan      = d(["12010001","12010002"])

    # ── Reserve & Funds ───────────────────────────────────────────────────────
    emp_fund      = d(["11010001","11010002","11010003"])
    dir_award     = d(["11020002"])
    rent_tax_fund = d(["11020001","11020003"])   # 11020003 Office Rent Tax Fund
    edu_fund      = d(["11030001"])
    sec_fund      = d(["11040001","11040002"])
    total_rf      = emp_fund + dir_award + rent_tax_fund + edu_fund + sec_fund

    # total_stl already includes stb_loan; do not add stb_loan again
    total_liab    = total_cl + total_stl + ltl_loan + total_rf

    # ── Equity ────────────────────────────────────────────────────────────────
    share_cap     = d(["13010001"])
    retained      = d(["13010003","13010004","13010005"])
    error_adj     = d(["13020001","13020002","13020003","13020004",
                        "13020005","13020006","13020007","13021001"])
    # net_profit from Level S IS is positive=profit (Level S sign).
    # BS equity convention (Level 0): profit increases equity as negative (credit).
    np_for_bs     = -net_profit.reindex(num, fill_value=0.0)
    total_equity  = share_cap + retained + error_adj + np_for_bs
    total_le      = total_liab + total_equity
    balance_check = total_assets + total_le   # should be near-zero

    # ── Consolidated-only netting adjustments ────────────────────────────────
    # Applied after all raw variables are set so sub-totals stay consistent.
    # The affected totals are recomputed immediately after.
    if str(zid) == "consolidated":
        # Work on mutable copies so the original Series are not aliased elsewhere.
        ar_internal = ar_internal.copy()
        ap_local    = ap_local.copy()
        ap_internal = ap_internal.copy()
        retained    = retained.copy()
        error_adj   = error_adj.copy()

        for _col in num:
            _yr = _period_key(_col)[0]   # calendar-year component of period key

            # ── ARAP netting ──────────────────────────────────────────────────
            # <= 2022: AR(Internal) + AP(Local) — pre-split era, both sides
            #          accumulated intercompany balances in these two buckets.
            # >= 2023: AR(Internal) + AP(Internal) — post-split era, correct
            #          intercompany buckets on each side.
            # Sign convention: AR is positive (asset), AP is negative (liability).
            # Net value → AP bucket if negative, AR bucket if positive; other = 0.
            _ar_val = float(ar_internal.iat[ar_internal.index.get_loc(_col)])
            if _yr <= 2022:
                _ap_val = float(ap_local.iat[ap_local.index.get_loc(_col)])
                _net    = _ar_val + _ap_val
                if _net <= 0:
                    ap_local.at[_col]    = _net
                    ar_internal.at[_col] = 0.0
                else:
                    ar_internal.at[_col] = _net
                    ap_local.at[_col]    = 0.0
            else:                                  # >= 2023
                _ap_val = float(ap_internal.iat[ap_internal.index.get_loc(_col)])
                _net    = _ar_val + _ap_val
                if _net <= 0:
                    ap_internal.at[_col] = _net
                    ar_internal.at[_col] = 0.0
                else:
                    ar_internal.at[_col] = _net
                    ap_internal.at[_col] = 0.0

        # ── Equity netting ────────────────────────────────────────────────────
        # Combine 1301A-Non-Cash Capital and 1302-Error Adjustment into one row.
        # The sum always lands in retained; error_adj is zeroed out.
        retained  = retained + error_adj
        error_adj = pd.Series(0.0, index=num)

        # ── Recompute all totals that depend on the netted variables ──────────
        ar_total      = ar_main + ar_internal + ar_agent + ar_local + ar_defaulted
        ap_total      = ap_local + ap_internal + ap_intl
        total_ca      = cash_ce + bank_bal + ar_total + prepaid + advance + stock
        total_cl      = accrued + ap_total + money_agent + recon_liab + cf_liab + others_liab
        total_equity  = share_cap + retained + error_adj + np_for_bs
        total_liab    = total_cl + total_stl + ltl_loan + total_rf
        total_le      = total_liab + total_equity
        total_assets  = total_ca + total_oa + total_fa
        balance_check = total_assets + total_le

    rows = [
        _ls_row("0101-Cash & Cash Equivalent",            cash_ce,       name_col),
        _ls_row("0102-Bank Balance",                      bank_bal,      name_col),
        _ls_row("Accounts Receivable",                    ar_main,       name_col),
        _ls_row("Accounts Receivable (Internal)",         ar_internal,   name_col),
        _ls_row("Recognized Agent (Dealers)",             ar_agent,      name_col),
        _ls_row("Accounts Receivable (Local)",            ar_local,      name_col),
        _ls_row("Defaulted Receivables",                  ar_defaulted,  name_col),
        _ls_row("0103-Accounts Receivable (Total)",       ar_total,      name_col),
        _ls_row("0104-Prepaid Expenses",                  prepaid,       name_col),
        _ls_row("0105-Advance Accounts",                  advance,       name_col),
        _ls_row("0106-Stock in Hand",                     stock,         name_col),
        _ls_row("01-Total Current Assets (A)",            total_ca,      name_col),
        _ls_row("0201-Deferred Capital Expenditure",      def_capex,     name_col),
        _ls_row("0202-Gift Items",                        gift_items,    name_col),
        _ls_row("0203-Loan to Hospital (Staff Salary)",   loan_hosp,     name_col),
        _ls_row("0203-Loan to Others Concern (Surma)",    loan_surma,    name_col),
        _ls_row("0204-Security Deposit",                  security_dep,  name_col),
        _ls_row("0205-Loan to Others Concern",            loan_others,   name_col),
        _ls_row("0206-Other Investment",                  other_invest,  name_col),
        _ls_row("Total Other Assets (B)",                 total_oa,      name_col),
        _ls_row("0301-Office Equipment",                  fa_office_eq,  name_col),
        _ls_row("0302-Corporate Office Equipments",       fa_corp_eq,    name_col),
        _ls_row("0303-Furniture & Fixture",               fa_furn,       name_col),
        _ls_row("0304-Trading Vehicles",                  fa_trade_veh,  name_col),
        _ls_row("0305-Private Vehicles",                  fa_priv_veh,   name_col),
        _ls_row("0306-Plants & Machinery",                fa_plant,      name_col),
        _ls_row("0307-Intangible Asset",                  fa_intang,     name_col),
        _ls_row("0308-Land & Building",                   fa_land,       name_col),
        _ls_row("Total Fixed Assets (C)",                 total_fa,      name_col),
        _ls_row("Total Assets (A+B+C)",                   total_assets,  name_col),
        _ls_row("0901-Accrued Expenses",                  accrued,       name_col),
        _ls_row("Accounts Payable (Local)",               ap_local,      name_col),
        _ls_row("Accounts Payable (Internal)",            ap_internal,   name_col),
        _ls_row("Accounts Payable (International)",       ap_intl,       name_col),
        _ls_row("0903-Accounts Payable (Total)",          ap_total,      name_col),
        _ls_row("0904-Money Agent Liability",             money_agent,   name_col),
        _ls_row("0904-Reconciliation Liability",          recon_liab,    name_col),
        _ls_row("0905-C & F Liability",                   cf_liab,       name_col),
        _ls_row("0906-Others Liability",                  others_liab,   name_col),
        _ls_row("Current Liability (A)",                  total_cl,      name_col),
        _ls_row("1001-Short Term Bank Loan (B)",          stb_loan,      name_col),
        _ls_row("1001-Short Term Loan (Related Parties)", st_loan,       name_col),
        _ls_row("Total Short Term Liability (C)",         total_stl,     name_col),
        _ls_row("1201-Long Term Bank Loan (E)",           ltl_loan,      name_col),
        _ls_row("1101-Employee Fund",                     emp_fund,      name_col),
        _ls_row("1102-Directors Award Fund",              dir_award,     name_col),
        _ls_row("1102-Office Rent Tax Fund",              rent_tax_fund, name_col),
        _ls_row("1103-Employee Educational Fund",         edu_fund,      name_col),
        _ls_row("1104-Security Deposit Fund",             sec_fund,      name_col),
        _ls_row("Total Reserve & Funds (D)",              total_rf,      name_col),
        _ls_row("Total Liabilities (A+B+C+D+E)",           total_liab,    name_col),
        _ls_row("1301-Share Capital",                     share_cap,     name_col),
        _ls_row("1301A-Non-Cash Capital (Retained Earning)", retained,   name_col),
        _ls_row("1302-Error Adjustment For Retained Earning", error_adj, name_col),
        _ls_row("Net Profit/Loss",                        np_for_bs,     name_col),
        _ls_row("Total Equity",                           total_equity,  name_col),
        _ls_row("Total Liabilities & Equity",             total_le,      name_col),
        _ls_row("Balance Check",                          balance_check, name_col),
    ]

    return pd.concat(rows, ignore_index=True)


def build_cfs_level_s(
    pl_raw: pd.DataFrame,
    bs_raw: pd.DataFrame,
    coc_lv0: pd.DataFrame,
    net_income_series: pd.Series,
    zid=None,
    code_col: str = "ac_code",
    name_col: str = "ac_name",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the Level S (Management View) Cash Flow Statement.

    Uses the same BS account groupings as build_bs_level_s() to produce
    grouped CFS rows (not individual accounts).  Sections:
      Operating WC  — 0103 AR, 0104 Prepaid, 0105 Advance, 0106 Stock,
                       0901 Accrued, 0903 AP, 0904 Recon, 0905 C&F, 0906 Others
      Investing     — 0201–0206 Other Assets, 0301–0308 Fixed Assets
      Financing     — 0904 Money Agent, 1001 STB Loan, 1001 ST Loan,
                       1201 LT Loan, 1101–1104 Funds, 1301 Share Cap,
                       1301A Non-Cash Cap, 1302 Error Adj,
                       Prior Period Net Profit/Loss

    Parameters
    ----------
    pl_raw : DataFrame
        Raw Level 0 P&L with ac_code (5 years).
    bs_raw : DataFrame
        Raw Level 0 BS with ac_code (5 years).
    coc_lv0 : DataFrame
        Opening/closing cash from cash_open_close() (same object as Level 0 CFS).
    net_income_series : Series
        Net Income per period from build_pl_level_s() (positive=profit).
        Must be indexed by the same period labels as bs_raw numeric columns.

    Returns
    -------
    (full_df, summary_df) — same structure as Level 0/1/2 CFS functions.
    """
    num_all     = _ls_num_cols(bs_raw)
    sorted_cols = sorted(num_all, key=_period_key)
    col_head    = sorted_cols               # e.g. [2021, 2022, 2023, 2024, 2025]
    delta_cols  = col_head[1:]              # 4 delta columns (CFS window)

    # ── Helpers ────────────────────────────────────────────────────────────
    def d(codes) -> pd.Series:
        """Raw BS group sum across all periods, indexed by col_head."""
        return _ls_sum_raw(bs_raw, codes, code_col).reindex(col_head, fill_value=0.0)

    def pfx(prefix) -> pd.Series:
        """Raw BS prefix sum across all periods, indexed by col_head."""
        return _ls_prefix_sum(bs_raw, prefix, code_col).reindex(col_head, fill_value=0.0)

    def delta(series: pd.Series) -> pd.Series:
        """Year-on-year change of a period Series; result indexed by delta_cols."""
        s = pd.Series(series.values, index=col_head, dtype=float)
        d_s = s.diff().iloc[1:]
        d_s.index = delta_cols
        return d_s

    def _row(label: str, series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({name_col: [label],
                             **dict(zip(delta_cols, series.values))})

    # ── BS group series (same account codes as build_bs_level_s) ──────────

    # — Current Assets (WC items only; cash/bank excluded) —
    _z_col = pd.Series(0.0, index=col_head)   # zero series for unused AR buckets
    if str(zid) == "consolidated":
        # Consolidated C2 BS has name-differentiated rows for the same ac_code.
        # Use name-based filtering to mirror the same AR split as build_bs_level_s.
        def dn_cfs(codes, names):
            return _ls_sum_raw_by_name(bs_raw, codes, names, code_col, "ac_name").reindex(col_head, fill_value=0.0)
        ar_main      = dn_cfs(["01030001"], ["Accounts Receivable"])
        ar_internal  = (dn_cfs(["01030001"], ["Accounts Receivable (Internal)"])
                        + dn_cfs(["01030002"], ["Accounts Receivable (Previous) GTC & Zepto"]))
        # "Recognized Agent" appears under 01030002 (100001/100005) and also
        # under 01030003 for ZID 100000 — both must be captured in ar_agent.
        ar_agent     = dn_cfs(["01030002", "01030003"], ["Recognized Agent"])
        ar_local     = _z_col
        ar_defaulted = dn_cfs(["01030003"], ["Accounts Receivable for future Collections"])
    else:
        ar_main      = d(["01030001"])
        ar_internal  = _z_col
        ar_agent     = d(["01030002"])
        ar_local     = d(["01030003"])
        ar_defaulted = _z_col
    ar_total      = ar_main + ar_internal + ar_agent + ar_local + ar_defaulted
    prepaid       = d(["01040005"])
    advance       = d(["01050001","01050002","01050003","01050004",
                        "01050005","01050006","01050007","01050008"])
    stock         = d(["01060003","01060001","01060002"])

    # — Current Liabilities (WC items; money_agent goes to Financing) —
    accrued       = d(["09010001","09010002","09010003","09010004",
                        "09010005","09010006"])
    if str(zid) == "consolidated":
        # Mirror the exact same AP split as build_bs_level_s consolidated branch:
        #   ap_local:    all 09030001 (any name variant) +
        #                100001's 09030002 "Constraction Materials Suppliers(M.B)"
        #   ap_internal: 09030002/"09030003" named "Accounts Payable(Internal)"
        #   ap_intl:     09030003/09030004 named "Accounts Payable(International)"
        ap_local    = (d(["09030001"])
                       + dn_cfs(["09030002"], ["Constraction Materials Suppliers(M.B)"]))
        ap_internal = dn_cfs(["09030002", "09030003"],
                             ["Accounts Payable(Internal)"])
        ap_intl     = dn_cfs(["09030003", "09030004"],
                             ["Accounts Payable(International)"])
    else:
        ap_local    = d(["09030001","09030002"])
        ap_internal = d(["09030003"])
        ap_intl     = d(["09030004"])
    ap_total      = ap_local + ap_internal + ap_intl
    recon_liab    = d(["09040006"])
    cf_liab       = pfx("09050")   # all C&F / shipping / customs liability codes
    others_liab   = d(["09060001"])

    # — Investing: Other Assets —
    def_capex     = d(["02010001"])
    gift_items    = d(["02020001"])
    loan_hosp     = d(["02030001"])
    loan_surma    = d(["02030002"])
    security_dep  = d(["02040001"])
    loan_others   = d(["02050001","02050002","02050003","02050004","02050005",
                        "02050006","02050007","02050008","02050009","02050010",
                        "02050011","02050012","02050013","02050014","02050015",
                        "02050016","02050017"])
    other_invest  = d(["02060001"])

    # — Investing: Fixed Assets —
    fa_office_eq  = pfx("03010")
    fa_corp_eq    = pfx("03020")
    fa_furn       = pfx("03030")
    fa_trade_veh  = pfx("03040")
    fa_priv_veh   = pfx("03050")
    fa_plant      = pfx("03060")
    fa_intang     = pfx("03070")
    fa_land       = pfx("03080")

    # — Financing: Liabilities & Equity —
    money_agent   = d(["09040001","09040002","09040003","09040004",
                        "09040005","09040007"])
    stb_loan      = d(["10010001","10010002","10010003","10010004",
                        "10010005","10010006","10010007","10010008"])
    st_loan       = d(["10020001","10020002","10020003","10020004","10020005",
                        "10020006","10020007","10020008","10020009","10020010",
                        "10020011","10020012","10020013","10020014","10020015",
                        "10020016","10020017"])
    ltl_loan      = d(["12010001","12010002"])
    emp_fund      = d(["11010001","11010002","11010003"])
    dir_award     = d(["11020002"])
    rent_tax_fund = d(["11020001","11020003"])   # 11020003 Office Rent Tax Fund
    edu_fund      = d(["11030001"])
    sec_fund      = d(["11040001","11040002"])
    share_cap     = d(["13010001"])
    retained      = d(["13010003","13010004","13010005"])
    error_adj     = d(["13020001","13020002","13020003","13020004",
                        "13020005","13020006","13020007","13021001"])

    # ── Consolidated-only netting (mirrors build_bs_level_s logic) ───────────
    # Applied to the balance-level Series before any deltas are taken, so the
    # CFS movements reflect the netted BS positions at each period boundary.
    if str(zid) == "consolidated":
        ar_internal = ar_internal.copy()
        ap_local    = ap_local.copy()
        ap_internal = ap_internal.copy()
        retained    = retained.copy()
        error_adj   = error_adj.copy()

        for _col in col_head:
            _yr = _period_key(_col)[0]

            # ARAP: net AR(Internal) against AP(Local) for <=2022,
            #        net AR(Internal) against AP(Internal) for >=2023.
            _ar_val = float(ar_internal.iat[ar_internal.index.get_loc(_col)])
            if _yr <= 2022:
                _ap_val = float(ap_local.iat[ap_local.index.get_loc(_col)])
                _net    = _ar_val + _ap_val
                if _net <= 0:
                    ap_local.at[_col]    = _net
                    ar_internal.at[_col] = 0.0
                else:
                    ar_internal.at[_col] = _net
                    ap_local.at[_col]    = 0.0
            else:
                _ap_val = float(ap_internal.iat[ap_internal.index.get_loc(_col)])
                _net    = _ar_val + _ap_val
                if _net <= 0:
                    ap_internal.at[_col] = _net
                    ar_internal.at[_col] = 0.0
                else:
                    ar_internal.at[_col] = _net
                    ap_internal.at[_col] = 0.0

        # Equity: combine retained + error_adj into retained; zero out error_adj.
        retained  = retained + error_adj
        error_adj = pd.Series(0.0, index=col_head)

        # Recompute totals that feed into op_total / delta calculations.
        ar_total = ar_main + ar_internal + ar_agent + ar_local + ar_defaulted
        ap_total = ap_local + ap_internal + ap_intl

    # ── Net Profit for CFS ────────────────────────────────────────────────
    # net_income_series is Level S (positive=profit).  Negate for CFS convention.
    np_cfs = (-net_income_series).reindex(delta_cols, fill_value=0.0)
    np_cfs = pd.Series(np_cfs.values, index=delta_cols, dtype=float)

    # ── Prior Period Net Profit/Loss ──────────────────────────────────────
    # Yearly mode: each delta year's Prior Period NP = the PRIOR year's full NI.
    # Monthly mode: Prior Period NP appears ONLY in January (month=1), where it
    # equals the FULL prior year's NI (sum of all monthly NI for that year).
    # For Feb–Dec it is zero — the incremental monthly NI is already captured in
    # the top-line "Net Profit/Loss" row and must not be double-counted.
    ni_all = net_income_series.reindex(sorted_cols, fill_value=0.0)

    first_pk   = _period_key(sorted_cols[0])
    is_monthly = isinstance(first_pk, tuple) and first_pk[1] != 0

    if is_monthly:
        # Aggregate full-year NI per calendar year across ALL sorted periods.
        year_totals: Dict[int, float] = {}
        for col in sorted_cols:
            yr = _period_key(col)[0]
            year_totals[yr] = year_totals.get(yr, 0.0) + float(ni_all[col])

        prior_np_fin = pd.Series(0.0, index=delta_cols)
        for col in delta_cols:
            pk = _period_key(col)
            yr, mo = pk[0], pk[1]
            if mo == 1:                               # January only
                prior_np_fin[col] = year_totals.get(yr - 1, 0.0)
    else:
        # Yearly: prior period NI shifts one slot right into each delta year.
        prior_np_fin = pd.Series(ni_all.values[:-1], index=delta_cols, dtype=float)

    # ── Operating WC detail rows ──────────────────────────────────────────
    op_details = pd.concat([
        _row("Accounts Receivable",                   delta(ar_main)),
        _row("Accounts Receivable (Internal)",        delta(ar_internal)),
        _row("Recognized Agent (Dealers)",            delta(ar_agent)),
        _row("Accounts Receivable (Local)",           delta(ar_local)),
        _row("Defaulted Receivables",                 delta(ar_defaulted)),
        _row("0103-Accounts Receivable (Total)",      delta(ar_total)),
        _row("0104-Prepaid Expenses",                 delta(prepaid)),
        _row("0105-Advance Accounts",                 delta(advance)),
        _row("0106-Stock in Hand",                    delta(stock)),
        _row("0901-Accrued Expenses",                 delta(accrued)),
        _row("Accounts Payable (Local)",              delta(ap_local)),
        _row("Accounts Payable (Internal)",           delta(ap_internal)),
        _row("Accounts Payable (International)",      delta(ap_intl)),
        _row("0903-Accounts Payable (Total)",         delta(ap_total)),
        _row("0904-Reconciliation Liability",         delta(recon_liab)),
        _row("0905-C & F Liability",                  delta(cf_liab)),
        _row("0906-Others Liability",                 delta(others_liab)),
    ], ignore_index=True)

    # op_total sums the nine WC group totals only (sub-rows excluded to avoid double-count)
    op_total = (
        delta(ar_total) + delta(prepaid) + delta(advance) + delta(stock)
        + delta(accrued) + delta(ap_total)
        + delta(recon_liab) + delta(cf_liab) + delta(others_liab)
    )

    # ── Investing detail rows ─────────────────────────────────────────────
    inv_details = pd.concat([
        _row("0201-Deferred Capital Expenditure",      delta(def_capex)),
        _row("0202-Gift Items",                        delta(gift_items)),
        _row("0203-Loan to Hospital (Staff Salary)",   delta(loan_hosp)),
        _row("0203-Loan to Others Concern (Surma)",    delta(loan_surma)),
        _row("0204-Security Deposit",                  delta(security_dep)),
        _row("0205-Loan to Others Concern",            delta(loan_others)),
        _row("0206-Other Investment",                  delta(other_invest)),
        _row("0301-Office Equipment",                  delta(fa_office_eq)),
        _row("0302-Corporate Office Equipments",       delta(fa_corp_eq)),
        _row("0303-Furniture & Fixture",               delta(fa_furn)),
        _row("0304-Trading Vehicles",                  delta(fa_trade_veh)),
        _row("0305-Private Vehicles",                  delta(fa_priv_veh)),
        _row("0306-Plants & Machinery",                delta(fa_plant)),
        _row("0307-Intangible Asset",                  delta(fa_intang)),
        _row("0308-Land & Building",                   delta(fa_land)),
    ], ignore_index=True)

    inv_total = pd.Series(
        inv_details[[c for c in delta_cols]].sum().values,
        index=delta_cols, dtype=float,
    )

    # ── Financing detail rows ─────────────────────────────────────────────
    fin_details = pd.concat([
        _row("0904-Money Agent Liability",                 delta(money_agent)),
        _row("1001-Short Term Bank Loan (B)",              delta(stb_loan)),
        _row("1001-Short Term Loan (Related Parties)",     delta(st_loan)),
        _row("1201-Long Term Bank Loan (E)",               delta(ltl_loan)),
        _row("1101-Employee Fund",                         delta(emp_fund)),
        _row("1102-Directors Award Fund",                  delta(dir_award)),
        _row("1102-Office Rent Tax Fund",                  delta(rent_tax_fund)),
        _row("1103-Employee Educational Fund",             delta(edu_fund)),
        _row("1104-Security Deposit Fund",                 delta(sec_fund)),
        _row("1301-Share Capital",                         delta(share_cap)),
        _row("1301A-Non-Cash Capital (Retained Earning)",  delta(retained)),
        _row("1302-Error Adjustment For Retained Earning", delta(error_adj)),
        _row("Prior Period Net Profit/Loss",               prior_np_fin),
    ], ignore_index=True)

    fin_total = pd.Series(
        fin_details[[c for c in delta_cols]].sum().values,
        index=delta_cols, dtype=float,
    )

    # Depreciation is NOT added back: the accumulated-depreciation BS account
    # (03xx codes) is already captured in inv_total via the BS delta, so adding
    # dep back in CFO would double-count it and break the cash-flow check.
    cfo_series = np_cfs + op_total

    # ── Opening / closing cash (reuse Level 0 coc_lv0) ───────────────────
    num_coc   = coc_lv0.select_dtypes("number").columns
    key2delta = {_period_key(c): c for c in delta_cols}

    opening = pd.Series(0.0, index=delta_cols)
    closing = pd.Series(0.0, index=delta_cols)

    if not coc_lv0.empty:
        close_raw = coc_lv0.loc[coc_lv0[name_col] == "Closing Cash & CE", num_coc]
        open_raw  = coc_lv0.loc[coc_lv0[name_col] == "Opening Cash & CE", num_coc]
        if not close_raw.empty and not open_raw.empty:
            for col in num_coc:
                k = _period_key(col)
                if k in key2delta:
                    opening[key2delta[k]] = open_raw.iloc[0][col]
                    closing[key2delta[k]] = close_raw.iloc[0][col]

    np_row      = _row("Net Profit/Loss",               np_cfs)
    wc_row      = _row("Δ Working Capital",             op_total)
    cfo_row     = _row("Cash from Operations",          cfo_series)
    inv_row     = _row("Cash from Investing",           inv_total)
    fin_row     = _row("Cash from Financing",           fin_total)
    opening_row = _row("Opening Cash & CE",             opening)
    ndc_series  = cfo_series + inv_total + fin_total
    ndc_row     = _row("Net ΔCash",                     ndc_series)
    calc_close  = opening - ndc_series
    calc_row    = _row("Calculated Closing Cash & CE",  calc_close)
    close_row   = _row("Closing Cash & CE",             closing)
    check_row   = _row("Cash-flow Check",               calc_close - closing)

    full_df = pd.concat(
        [
            np_row,
            op_details, wc_row, cfo_row,
            inv_details, inv_row,
            fin_details, fin_row,
            opening_row, ndc_row,
            calc_row, close_row, check_row,
        ],
        ignore_index=True,
    )

    summary_labels = [
        "Net Profit/Loss", "Δ Working Capital",
        "Cash from Operations", "Cash from Investing",
        "Cash from Financing", "Opening Cash & CE",
        "Net ΔCash", "Calculated Closing Cash & CE",
        "Closing Cash & CE", "Cash-flow Check",
    ]
    sum_df = full_df[full_df[name_col].isin(summary_labels)].reset_index(drop=True)

    return full_df, sum_df


# ══════════════════════════════════════════════════════════════════════════════
# Level T — Trading Adjustments (ZID 100001, Full Year vs YTD / Lifetime only)
#
# These functions take already-generated Level S DataFrames as input and apply
# row-level overrides.  Level S generation is completely untouched.
#
# adj_data : dict of named pd.Series, each indexed to the numeric period
#            columns of the Level S DataFrames.  Add future adjustments here:
#              "ind_hh_net_sales" — I&H net sales subtracted from Revenue (years <= 2024)
#              future: "cogs_adj", "sga_adj", etc.
# ══════════════════════════════════════════════════════════════════════════════

def _t_get(df: pd.DataFrame, name: str, name_col: str = "ac_name") -> pd.Series:
    """Return the numeric row values for a named row in a Level S DataFrame."""
    num = df.select_dtypes("number").columns
    mask = df[name_col] == name
    if not mask.any():
        return pd.Series(0.0, index=num)
    return df.loc[mask, num].iloc[0].copy()


def _t_set(df: pd.DataFrame, name: str, values: pd.Series,
           name_col: str = "ac_name") -> None:
    """Overwrite the numeric values for a named row in a Level S DataFrame (in-place)."""
    num = df.select_dtypes("number").columns
    mask = df[name_col] == name
    if mask.any():
        df.loc[mask, num] = values.reindex(num).values


def get_level_t_vat_prior(num_cols) -> pd.Series:
    """
    Load the pre-2022 VAT Rebate reclassification amounts from
    data/level_t_vat_prior.json and return a Series aligned to *num_cols*.

    Values are non-zero only for year columns <= 2021.  Update the JSON file
    at data/level_t_vat_prior.json when exact figures are available from
    physical records.
    """
    _path = HERE / "data" / "level_t_vat_prior.json"
    _raw: dict = {}
    if _path.exists():
        with open(_path, "r", encoding="utf-8") as _f:
            _raw = json.load(_f).get("vat_rebate_prior", {})

    series = pd.Series(0.0, index=num_cols)
    for col in num_cols:
        yr = _period_key(col)[0]
        if yr <= 2021:
            key = str(yr)
            if key in _raw:
                series[col] = float(_raw[key])
    return series


def adjust_pl_level_t(pl_s: pd.DataFrame, adj_data: dict) -> pd.DataFrame:
    """
    Apply Level T adjustments to the Level S Income Statement.

    Current adjustments
    -------------------
    - Revenue reduced by Industrial & Household net sales (years <= 2024 only;
      the caller zeroes out later years in adj_data["ind_hh_net_sales"]).
    - Adjusted Revenue, Gross Profit, EBITDA and Net Income are recalculated
      to cascade the revenue change through the IS.

    Future adjustments
    ------------------
    Add entries to adj_data and corresponding _t_set calls in the
    ADJUSTMENTS BLOCK below.  All other rows stay identical to Level S.
    """
    df = pl_s.copy()
    num = df.select_dtypes("number").columns
    z = pd.Series(0.0, index=num)

    # ── ADJUSTMENTS BLOCK ─────────────────────────────────────────────────────
    ind_hh        = adj_data.get("ind_hh_net_sales", z.copy())
    cogs_adj      = adj_data.get("cogs_adj",          z.copy())
    # 8 % of I&H revenue is the expense reduction applied proportionally across
    # all SG&A and S&D sub-rows (bank interest and tax rows are untouched).
    exp_reduction = 0.08 * ind_hh   # positive value per year
    # Pre-2022 VAT rebate amounts that were embedded inside COGS; loaded from JSON.
    # Non-zero only for years <= 2021 (placeholder 5 000 000 BDT — update from records).
    vat_rebate_prior = get_level_t_vat_prior(num)
    # ──────────────────────────────────────────────────────────────────────────

    # ── Read Level S rows ─────────────────────────────────────────────────────
    revenue        = _t_get(df, "Revenue")
    others_rev     = _t_get(df, "Others Revenue")
    mrp_discount   = _t_get(df, "MRP Discount")
    cogs           = _t_get(df, "COGS")
    vat_rebate_row = _t_get(df, "VAT Expenses from Rebate (A)")

    # SG&A individual rows
    sga            = _t_get(df, "SG&A")
    salary         = _t_get(df, "0612-Salary Expenses")
    bonus          = _t_get(df, "0613-Employee Bonus")
    overtime       = _t_get(df, "0614-Overtime")
    director_rem   = _t_get(df, "0615-Director Remuneration")

    # S&D individual rows
    discount_paid  = _t_get(df, "0708-Discount Paid")
    sd_expenses    = _t_get(df, "Sales & Distribution Expenses")

    others_direct  = _t_get(df, "0501-Others Direct Expenses")
    total_interest = _t_get(df, "Total Interest & Charges")
    vat_tax_total  = _t_get(df, "0629-VAT & Tax Total (A+B+C)")

    # ── Revenue & COGS adjustments ─────────────────────────────────────────────
    new_revenue      = revenue - ind_hh
    new_adj_revenue  = new_revenue + mrp_discount

    # Remove I&H net cost AND reclassify pre-2022 embedded VAT out of COGS.
    # Both adjustments are positive, making COGS less negative (lower cost).
    new_cogs         = cogs + cogs_adj + vat_rebate_prior
    new_gross_profit = new_revenue + others_rev + mrp_discount + new_cogs

    # ── Pre-2022 VAT reclassification — move out of COGS into VAT line ────────
    # The same amount is added to VAT Expenses from Rebate (making it more
    # negative = larger expense) so that Net Income remains unchanged.
    new_vat_rebate    = vat_rebate_row - vat_rebate_prior
    new_vat_tax_total = vat_tax_total  - vat_rebate_prior

    # ── Proportional expense reduction across SG&A + S&D ──────────────────────
    # total_base is the sum of all sub-rows — negative (expenses).
    # scale = exp_reduction / total_base  →  negative value.
    # new_row = row * (1 + scale)  →  row magnitude shrinks proportionally.
    total_base  = (sga + salary + bonus + overtime + director_rem
                   + discount_paid + sd_expenses)
    safe_base   = total_base.where(total_base != 0, other=np.nan)
    scale       = (exp_reduction / safe_base).fillna(0)

    new_sga          = sga          * (1 + scale)
    new_salary       = salary       * (1 + scale)
    new_bonus        = bonus        * (1 + scale)
    new_overtime     = overtime     * (1 + scale)
    new_director_rem = director_rem * (1 + scale)
    new_disc_paid    = discount_paid * (1 + scale)
    new_sd_expenses  = sd_expenses  * (1 + scale)

    new_total_sga    = new_sga + new_salary + new_bonus + new_overtime + new_director_rem
    new_total_sd     = new_disc_paid + new_sd_expenses

    new_ebitda       = new_gross_profit + new_total_sga + new_total_sd + others_direct
    # VAT tax total already incorporates the rebate reclassification above.
    new_net_income   = new_ebitda + total_interest + new_vat_tax_total

    # ── Write back ────────────────────────────────────────────────────────────
    _t_set(df, "Revenue",                        new_revenue)
    _t_set(df, "Adjusted Revenue (Pending)",     new_adj_revenue)
    _t_set(df, "COGS",                           new_cogs)
    _t_set(df, "Gross Profit",                   new_gross_profit)
    _t_set(df, "VAT Expenses from Rebate (A)",   new_vat_rebate)
    _t_set(df, "0629-VAT & Tax Total (A+B+C)",   new_vat_tax_total)
    _t_set(df, "SG&A",                           new_sga)
    _t_set(df, "0612-Salary Expenses",           new_salary)
    _t_set(df, "0613-Employee Bonus",            new_bonus)
    _t_set(df, "0614-Overtime",                  new_overtime)
    _t_set(df, "0615-Director Remuneration",     new_director_rem)
    _t_set(df, "Total SG&A",                     new_total_sga)
    _t_set(df, "0708-Discount Paid",             new_disc_paid)
    _t_set(df, "Sales & Distribution Expenses",  new_sd_expenses)
    _t_set(df, "Total Sales & Distribution",     new_total_sd)
    _t_set(df, "EBITDA",                         new_ebitda)
    _t_set(df, "Net Income",                     new_net_income)

    return df


def get_level_t_stock_inh(num_cols) -> pd.Series:
    """
    Load the I&H ending stock balance per year from
    data/level_t_stock_inh.json and return a Series aligned to *num_cols*.

    Values cover 2016–2023 (manually calculated). Years outside that range
    return 0 (no adjustment). Update the JSON when revised figures are available.
    """
    _path = HERE / "data" / "level_t_stock_inh.json"
    _raw: dict = {}
    if _path.exists():
        with open(_path, "r", encoding="utf-8") as _f:
            _raw = json.load(_f).get("inh_stock_balance", {})

    series = pd.Series(0.0, index=num_cols)
    for col in num_cols:
        key = str(_period_key(col)[0])
        if key in _raw:
            series[col] = float(_raw[key])
    return series


def adjust_bs_level_t(bs_s: pd.DataFrame, adj_data: dict) -> pd.DataFrame:
    """
    Apply Level T adjustments to the Level S Balance Sheet.

    Adjustments applied
    -------------------
    1. Stock in Hand     — subtract the I&H ending inventory balance (from JSON)
                           from '0106-Stock in Hand'.
    2. Accounts Receivable — scale all AR rows by the non-I&H revenue share:
                           keep_ratio = 1 − (ind_hh_net_sales / level_s_revenue)
                           Sub-rows (AR Main, Agent, Local) and AR Total are all
                           scaled by the same ratio so they remain consistent.
    3. Cash & Bank       — scale '0101-Cash & Cash Equivalent' and
                           '0102-Bank Balance' by the same keep_ratio, removing
                           the estimated cash float held for I&H operations.
    4. Equity balancing  — after all asset reductions, the '1302-Error Adjustment
                           For Retained Earning' row is offset by the exact gap
                           so that Balance Check returns to zero.  Total Equity
                           and Total Liabilities & Equity are recalculated
                           accordingly.

    adj_data keys consumed
    ----------------------
    ind_hh_net_sales : pd.Series  — I&H net sales (from PL revenue adjustment)
    level_s_revenue  : pd.Series  — Level S total revenue (pre-T adjustments)
    """
    df = bs_s.copy()
    num = df.select_dtypes("number").columns
    z   = pd.Series(0.0, index=num)

    # ── Load inputs ───────────────────────────────────────────────────────────
    inh_stock      = get_level_t_stock_inh(num)
    ind_hh_rev     = adj_data.get("ind_hh_net_sales", z.copy()).reindex(num, fill_value=0.0)
    level_s_rev    = adj_data.get("level_s_revenue",  z.copy()).reindex(num, fill_value=0.0)

    # I&H share of total revenue (clamped to [0, 1] for safety)
    safe_rev   = level_s_rev.where(level_s_rev != 0, other=np.nan)
    ih_ratio   = (ind_hh_rev / safe_rev).fillna(0.0).clip(0.0, 1.0)
    keep_ratio = 1.0 - ih_ratio   # fraction to retain (non-I&H share)

    # ── Read current BS rows ──────────────────────────────────────────────────
    cash_ce    = _t_get(df, "0101-Cash & Cash Equivalent")
    bank_bal   = _t_get(df, "0102-Bank Balance")
    ar_main    = _t_get(df, "Accounts Receivable")
    ar_agent   = _t_get(df, "Recognized Agent (Dealers)")
    ar_local   = _t_get(df, "Accounts Receivable (Local)")
    ar_total   = _t_get(df, "0103-Accounts Receivable (Total)")
    prepaid    = _t_get(df, "0104-Prepaid Expenses")
    advance    = _t_get(df, "0105-Advance Accounts")
    stock      = _t_get(df, "0106-Stock in Hand")
    total_oa   = _t_get(df, "Total Other Assets (B)")
    total_fa   = _t_get(df, "Total Fixed Assets (C)")
    share_cap  = _t_get(df, "1301-Share Capital")
    retained   = _t_get(df, "1301A-Non-Cash Capital (Retained Earning)")
    error_adj  = _t_get(df, "1302-Error Adjustment For Retained Earning")
    np_for_bs  = _t_get(df, "Net Profit/Loss")
    total_liab = _t_get(df, "Total Liabilities (A+B+C+D+E)")

    # ── Adjustment 1: Stock in Hand ───────────────────────────────────────────
    new_stock = stock - inh_stock

    # ── Adjustment 2: Accounts Receivable (non-I&H share only) ───────────────
    new_ar_main  = ar_main  * keep_ratio
    new_ar_agent = ar_agent * keep_ratio
    new_ar_local = ar_local * keep_ratio
    new_ar_total = ar_total * keep_ratio

    # ── Adjustment 3: Cash & Bank (non-I&H share only) ───────────────────────
    new_cash_ce  = cash_ce  * keep_ratio
    new_bank_bal = bank_bal * keep_ratio

    # ── Recalculate asset subtotals ───────────────────────────────────────────
    new_total_ca     = new_cash_ce + new_bank_bal + new_ar_total + prepaid + advance + new_stock
    new_total_assets = new_total_ca + total_oa + total_fa

    # ── Adjustment 4: Equity balancing ───────────────────────────────────────
    # Compute the imbalance that exists before correction.  All removed assets
    # reduce total_assets; without a matching liability/equity offset the
    # Balance Check deviates from zero.  We absorb the gap into error_adj so
    # that Balance Check = new_total_assets + new_total_le = 0 exactly.
    intermediate_check = new_total_assets + total_liab + share_cap + retained + error_adj + np_for_bs
    new_error_adj      = error_adj - intermediate_check
    new_total_equity   = share_cap + retained + new_error_adj + np_for_bs
    new_total_le       = total_liab + new_total_equity

    # ── Write back ────────────────────────────────────────────────────────────
    _t_set(df, "0101-Cash & Cash Equivalent",                new_cash_ce)
    _t_set(df, "0102-Bank Balance",                          new_bank_bal)
    _t_set(df, "0106-Stock in Hand",                         new_stock)
    _t_set(df, "Accounts Receivable",                        new_ar_main)
    _t_set(df, "Recognized Agent (Dealers)",                 new_ar_agent)
    _t_set(df, "Accounts Receivable (Local)",                new_ar_local)
    _t_set(df, "0103-Accounts Receivable (Total)",           new_ar_total)
    _t_set(df, "01-Total Current Assets (A)",                new_total_ca)
    _t_set(df, "Total Assets (A+B+C)",                       new_total_assets)
    _t_set(df, "1302-Error Adjustment For Retained Earning", new_error_adj)
    _t_set(df, "Total Equity",                               new_total_equity)
    _t_set(df, "Total Liabilities & Equity",                 new_total_le)
    _t_set(df, "Balance Check",                              new_total_assets + new_total_le)

    return df


def adjust_cfs_level_t(
    cfs_s: pd.DataFrame,
    summary_s: pd.DataFrame,
    adj_data: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Level T adjustments to the Level S Cash Flow Statement.
    Currently a pass-through — placeholder for future row-level CFS overrides.
    CFS is rebuilt from scratch with the updated net income before this is called,
    so the net income change propagates automatically.
    """
    # future: add _t_set calls here as new adj_data keys are introduced
    return cfs_s.copy(), summary_s.copy()


def build_cfs_from_bs_t(
    bs_t: pd.DataFrame,
    net_income_series: pd.Series,
    name_col: str = "ac_name",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the Level T Cash Flow Statement directly from the adjusted Level T
    Balance Sheet.

    Unlike build_cfs_level_s(), which reads raw GL account codes, this function
    reads the named rows that already exist in bs_t (the output of
    adjust_bs_level_t).  Because bs_t is fully balanced (Balance Check = 0),
    the resulting CFS check is guaranteed to be 0 as well.

    Opening and closing cash are derived from the adjusted '0101-Cash & Cash
    Equivalent' + '0102-Bank Balance' rows, so they reflect the I&H-removed
    cash balance consistently.

    This function is designed for yearly mode only (matching Level T).

    Parameters
    ----------
    bs_t             : adjusted Level T Balance Sheet from adjust_bs_level_t()
    net_income_series: Net Income per period from adjust_pl_level_t() (positive=profit)

    Returns
    -------
    (full_df, summary_df)  — same structure as build_cfs_level_s()
    """
    num_all     = bs_t.select_dtypes("number").columns
    sorted_cols = sorted(num_all, key=_period_key)
    col_head    = sorted_cols          # all period columns (e.g. 5 years)
    delta_cols  = col_head[1:]         # 4 delta columns (CFS window)

    # ── Helpers ────────────────────────────────────────────────────────────
    def get(row_name: str) -> pd.Series:
        """Read a named BS row and align to col_head."""
        return _t_get(bs_t, row_name).reindex(col_head, fill_value=0.0)

    def delta(series: pd.Series) -> pd.Series:
        """Year-on-year change; result indexed by delta_cols."""
        s = pd.Series(series.values, index=col_head, dtype=float)
        d_s = s.diff().iloc[1:]
        d_s.index = delta_cols
        return d_s

    def _row(label: str, series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({name_col: [label],
                             **dict(zip(delta_cols, series.values))})

    # ── Read all BS rows (mirrors the named rows in build_bs_level_s) ──────

    # Current Assets — WC items (cash excluded; tracked separately as open/close)
    ar_main      = get("Accounts Receivable")
    ar_agent     = get("Recognized Agent (Dealers)")
    ar_local     = get("Accounts Receivable (Local)")
    ar_total     = get("0103-Accounts Receivable (Total)")
    prepaid      = get("0104-Prepaid Expenses")
    advance      = get("0105-Advance Accounts")
    stock        = get("0106-Stock in Hand")

    # Current Liabilities — WC items (money_agent goes to Financing)
    accrued      = get("0901-Accrued Expenses")
    ap_local     = get("Accounts Payable (Local)")
    ap_internal  = get("Accounts Payable (Internal)")
    ap_intl      = get("Accounts Payable (International)")
    ap_total     = get("0903-Accounts Payable (Total)")
    recon_liab   = get("0904-Reconciliation Liability")
    cf_liab      = get("0905-C & F Liability")
    others_liab  = get("0906-Others Liability")

    # Investing: Other Assets
    def_capex    = get("0201-Deferred Capital Expenditure")
    gift_items   = get("0202-Gift Items")
    loan_hosp    = get("0203-Loan to Hospital (Staff Salary)")
    loan_surma   = get("0203-Loan to Others Concern (Surma)")
    security_dep = get("0204-Security Deposit")
    loan_others  = get("0205-Loan to Others Concern")
    other_invest = get("0206-Other Investment")

    # Investing: Fixed Assets
    fa_office_eq  = get("0301-Office Equipment")
    fa_corp_eq    = get("0302-Corporate Office Equipments")
    fa_furn       = get("0303-Furniture & Fixture")
    fa_trade_veh  = get("0304-Trading Vehicles")
    fa_priv_veh   = get("0305-Private Vehicles")
    fa_plant      = get("0306-Plants & Machinery")
    fa_intang     = get("0307-Intangible Asset")
    fa_land       = get("0308-Land & Building")

    # Financing: Liabilities & Equity
    money_agent   = get("0904-Money Agent Liability")
    stb_loan      = get("1001-Short Term Bank Loan (B)")
    st_loan       = get("1001-Short Term Loan (Related Parties)")
    ltl_loan      = get("1201-Long Term Bank Loan (E)")
    emp_fund      = get("1101-Employee Fund")
    dir_award     = get("1102-Directors Award Fund")
    rent_tax_fund = get("1102-Office Rent Tax Fund")
    edu_fund      = get("1103-Employee Educational Fund")
    sec_fund      = get("1104-Security Deposit Fund")
    share_cap     = get("1301-Share Capital")
    retained      = get("1301A-Non-Cash Capital (Retained Earning)")
    error_adj     = get("1302-Error Adjustment For Retained Earning")

    # Adjusted cash total (cash_ce + bank_bal, both already scaled by keep_ratio)
    cash_ce      = get("0101-Cash & Cash Equivalent")
    bank_bal     = get("0102-Bank Balance")
    cash_total   = cash_ce + bank_bal

    # ── Net Profit for CFS ────────────────────────────────────────────────
    np_cfs = (-net_income_series).reindex(delta_cols, fill_value=0.0)
    np_cfs = pd.Series(np_cfs.values, index=delta_cols, dtype=float)

    # ── Prior Period Net Profit/Loss (yearly mode only for Level T) ───────
    ni_all       = net_income_series.reindex(sorted_cols, fill_value=0.0)
    prior_np_fin = pd.Series(ni_all.values[:-1], index=delta_cols, dtype=float)

    # ── Operating WC detail rows ──────────────────────────────────────────
    op_details = pd.concat([
        _row("Accounts Receivable",                   delta(ar_main)),
        _row("Recognized Agent (Dealers)",            delta(ar_agent)),
        _row("Accounts Receivable (Local)",           delta(ar_local)),
        _row("0103-Accounts Receivable (Total)",      delta(ar_total)),
        _row("0104-Prepaid Expenses",                 delta(prepaid)),
        _row("0105-Advance Accounts",                 delta(advance)),
        _row("0106-Stock in Hand",                    delta(stock)),
        _row("0901-Accrued Expenses",                 delta(accrued)),
        _row("Accounts Payable (Local)",              delta(ap_local)),
        _row("Accounts Payable (Internal)",           delta(ap_internal)),
        _row("Accounts Payable (International)",      delta(ap_intl)),
        _row("0903-Accounts Payable (Total)",         delta(ap_total)),
        _row("0904-Reconciliation Liability",         delta(recon_liab)),
        _row("0905-C & F Liability",                  delta(cf_liab)),
        _row("0906-Others Liability",                 delta(others_liab)),
    ], ignore_index=True)

    op_total = (
        delta(ar_total) + delta(prepaid) + delta(advance) + delta(stock)
        + delta(accrued) + delta(ap_total)
        + delta(recon_liab) + delta(cf_liab) + delta(others_liab)
    )

    # ── Investing detail rows ─────────────────────────────────────────────
    inv_details = pd.concat([
        _row("0201-Deferred Capital Expenditure",      delta(def_capex)),
        _row("0202-Gift Items",                        delta(gift_items)),
        _row("0203-Loan to Hospital (Staff Salary)",   delta(loan_hosp)),
        _row("0203-Loan to Others Concern (Surma)",    delta(loan_surma)),
        _row("0204-Security Deposit",                  delta(security_dep)),
        _row("0205-Loan to Others Concern",            delta(loan_others)),
        _row("0206-Other Investment",                  delta(other_invest)),
        _row("0301-Office Equipment",                  delta(fa_office_eq)),
        _row("0302-Corporate Office Equipments",       delta(fa_corp_eq)),
        _row("0303-Furniture & Fixture",               delta(fa_furn)),
        _row("0304-Trading Vehicles",                  delta(fa_trade_veh)),
        _row("0305-Private Vehicles",                  delta(fa_priv_veh)),
        _row("0306-Plants & Machinery",                delta(fa_plant)),
        _row("0307-Intangible Asset",                  delta(fa_intang)),
        _row("0308-Land & Building",                   delta(fa_land)),
    ], ignore_index=True)

    inv_total = pd.Series(
        inv_details[[c for c in delta_cols]].sum().values,
        index=delta_cols, dtype=float,
    )

    # ── Financing detail rows ─────────────────────────────────────────────
    fin_details = pd.concat([
        _row("0904-Money Agent Liability",                 delta(money_agent)),
        _row("1001-Short Term Bank Loan (B)",              delta(stb_loan)),
        _row("1001-Short Term Loan (Related Parties)",     delta(st_loan)),
        _row("1201-Long Term Bank Loan (E)",               delta(ltl_loan)),
        _row("1101-Employee Fund",                         delta(emp_fund)),
        _row("1102-Directors Award Fund",                  delta(dir_award)),
        _row("1102-Office Rent Tax Fund",                  delta(rent_tax_fund)),
        _row("1103-Employee Educational Fund",             delta(edu_fund)),
        _row("1104-Security Deposit Fund",                 delta(sec_fund)),
        _row("1301-Share Capital",                         delta(share_cap)),
        _row("1301A-Non-Cash Capital (Retained Earning)",  delta(retained)),
        _row("1302-Error Adjustment For Retained Earning", delta(error_adj)),
        _row("Prior Period Net Profit/Loss",               prior_np_fin),
    ], ignore_index=True)

    fin_total = pd.Series(
        fin_details[[c for c in delta_cols]].sum().values,
        index=delta_cols, dtype=float,
    )

    # Depreciation is NOT added back (same rationale as build_cfs_level_s).
    cfo_series = np_cfs + op_total

    # ── Opening / closing cash from adjusted BS ───────────────────────────
    # cash_total is indexed by col_head (all years including the base year).
    # For each delta column i (year[i] vs year[i-1]):
    #   opening = cash_total at year[i-1]
    #   closing = cash_total at year[i]
    opening = pd.Series(
        cash_total.values[:-1], index=delta_cols, dtype=float
    )
    closing = pd.Series(
        cash_total.values[1:], index=delta_cols, dtype=float
    )

    np_row      = _row("Net Profit/Loss",               np_cfs)
    wc_row      = _row("Δ Working Capital",             op_total)
    cfo_row     = _row("Cash from Operations",          cfo_series)
    inv_row     = _row("Cash from Investing",           inv_total)
    fin_row     = _row("Cash from Financing",           fin_total)
    opening_row = _row("Opening Cash & CE",             opening)
    ndc_series  = cfo_series + inv_total + fin_total
    ndc_row     = _row("Net ΔCash",                     ndc_series)
    calc_close  = opening - ndc_series
    calc_row    = _row("Calculated Closing Cash & CE",  calc_close)
    close_row   = _row("Closing Cash & CE",             closing)
    check_row   = _row("Cash-flow Check",               calc_close - closing)

    full_df = pd.concat(
        [
            np_row,
            op_details, wc_row, cfo_row,
            inv_details, inv_row,
            fin_details, fin_row,
            opening_row, ndc_row,
            calc_row, close_row, check_row,
        ],
        ignore_index=True,
    )

    summary_labels = [
        "Net Profit/Loss", "Δ Working Capital",
        "Cash from Operations", "Cash from Investing",
        "Cash from Financing", "Opening Cash & CE",
        "Net ΔCash", "Calculated Closing Cash & CE",
        "Closing Cash & CE", "Cash-flow Check",
    ]
    sum_df = full_df[full_df[name_col].isin(summary_labels)].reset_index(drop=True)

    return full_df, sum_df


# ══════════════════════════════════════════════════════════════════════════════
# Level T — GI Adjustments  (ZID 100000, Full Year vs YTD / Lifetime only)
#
# adj_data keys consumed / expected:
#   'ind_hh_net_sales'  — I&H net sales Series from 100001 (aligned to GI cols)
#   'cogs_adj'          — I&H net cost Series from 100001  (future use)
#   'level_s_revenue'   — 100001 Level S Revenue Series    (future use)
#   ... additional keys added as each instruction is implemented
# ══════════════════════════════════════════════════════════════════════════════

def adjust_pl_level_t_gi(
    pl_gi_s: pd.DataFrame,
    adj_data: dict,
) -> pd.DataFrame:
    """
    Apply GI (100000) Level T adjustments to the GI Level S Income Statement.

    Adjustments applied  (years < 2024 only — 2024+ equals Level S)
    -------------------
    1. Revenue    — replaced entirely with I&H Net Sales from ZID 100001.
    2. COGS       — replaced entirely with sum of COGS from 100002/3/4/7/8.
    3. Total SG&A — replaced with sum of Total SG&A from 100002/3/4/7/8.
    4. Total S&D  — sum of Total S&D from 100002/3/4/7/8, plus a 5% of
                    adjusted revenue distribution charge.

    All adjustments cascade through: Adjusted Revenue, Gross Profit,
    EBITDA, and Net Income.  For 2024 and beyond, Level T = Level S
    (no Level T interaction).

    adj_data keys consumed
    ----------------------
    ind_hh_net_sales       : pd.Series  — I&H net sales from 100001
    cogs_from_subsidiaries : pd.Series  — sum of COGS from 100002/3/4/7/8
    sga_from_subsidiaries  : pd.Series  — sum of Total SG&A from 100002/3/4/7/8
    sd_from_subsidiaries   : pd.Series  — sum of Total S&D from 100002/3/4/7/8
    """
    df  = pl_gi_s.copy()
    num = df.select_dtypes("number").columns
    z   = pd.Series(0.0, index=num)

    ind_hh_rev  = adj_data.get("ind_hh_net_sales",        z.copy()).reindex(num, fill_value=0.0)
    subs_cogs   = adj_data.get("cogs_from_subsidiaries",  z.copy()).reindex(num, fill_value=0.0)
    subs_sga    = adj_data.get("sga_from_subsidiaries",   z.copy()).reindex(num, fill_value=0.0)
    subs_sd     = adj_data.get("sd_from_subsidiaries",    z.copy()).reindex(num, fill_value=0.0)

    # ── Read rows needed for cascade ─────────────────────────────────────────
    revenue    = _t_get(df, "Revenue")
    cogs_ls    = _t_get(df, "COGS")
    others_rev = _t_get(df, "Others Revenue")
    mrp_disc   = _t_get(df, "MRP Discount")
    total_sga  = _t_get(df, "Total SG&A")
    total_sd   = _t_get(df, "Total Sales & Distribution")
    others_dir = _t_get(df, "0501-Others Direct Expenses")
    tot_int    = _t_get(df, "Total Interest & Charges")
    vat_tax    = _t_get(df, "0629-VAT & Tax Total (A+B+C)")

    # ── Adjustment 1: Revenue ────────────────────────────────────────────────
    #   < 2024 : replace entirely with I&H Net Sales from 100001
    #   ≥ 2024 : keep Level S as-is (no Level T interaction)
    new_revenue = revenue.copy()
    for col in num:
        if _period_key(col)[0] < 2024:
            new_revenue[col] = ind_hh_rev.get(col, 0.0)

    # ── Adjustment 2: COGS ───────────────────────────────────────────────────
    #   < 2024 : replace entirely with sum of COGS from subsidiaries 100002/3/4/7/8
    #   ≥ 2024 : keep Level S as-is (no Level T interaction)
    new_cogs = cogs_ls.copy()
    for col in num:
        if _period_key(col)[0] < 2024:
            new_cogs[col] = subs_cogs.get(col, 0.0)

    # ── Adjustment 3: Total SG&A ─────────────────────────────────────────────
    #   < 2024 : replace with sum of Total SG&A from subsidiaries 100002/3/4/7/8
    #   ≥ 2024 : keep Level S as-is (no Level T interaction)
    new_total_sga = total_sga.copy()
    for col in num:
        if _period_key(col)[0] < 2024:
            new_total_sga[col] = subs_sga.get(col, 0.0)

    # ── Adjustment 4: Total Sales & Distribution ─────────────────────────────
    #   < 2024 : sum of Total S&D from subsidiaries 100002/3/4/7/8
    #            + 5% of adjusted revenue as a distribution charge (negative = expense)
    #   ≥ 2024 : keep Level S as-is (no Level T interaction)
    new_total_sd = total_sd.copy()
    for col in num:
        if _period_key(col)[0] < 2024:
            new_total_sd[col] = subs_sd.get(col, 0.0) + (-0.05 * new_revenue[col])

    # ── Cascade through derived rows ──────────────────────────────────────────
    new_adj_rev    = new_revenue + mrp_disc
    new_gp         = new_revenue + others_rev + mrp_disc + new_cogs
    new_ebitda     = new_gp + new_total_sga + new_total_sd + others_dir
    new_net_income = new_ebitda + tot_int + vat_tax

    # ── Write back ────────────────────────────────────────────────────────────
    _t_set(df, "Revenue",                    new_revenue)
    _t_set(df, "Adjusted Revenue (Pending)", new_adj_rev)
    _t_set(df, "COGS",                       new_cogs)
    _t_set(df, "Gross Profit",               new_gp)
    _t_set(df, "Total SG&A",                 new_total_sga)
    _t_set(df, "Total Sales & Distribution", new_total_sd)
    _t_set(df, "EBITDA",                     new_ebitda)
    _t_set(df, "Net Income",                 new_net_income)

    return df


def adjust_bs_level_t_gi(
    bs_gi_s: pd.DataFrame,
    adj_data: dict,
) -> pd.DataFrame:
    """
    Apply GI (100000) Level T adjustments to the GI Level S Balance Sheet.

    Sign convention (inherited from build_bs_level_s / Level 0):
        Assets  → POSITIVE   (debit balances, _ls_sum_raw no flip)
        Liabilities / Equity → NEGATIVE  (credit balances)

    Adjustments by section
    ──────────────────────
    CURRENT ASSETS  (all adjustments apply for years < 2024 only;
                     2024+ = Level S, no Level T interaction)
      0101 Cash CE  } — ADD the I&H portions from 100001 (positive) for years < 2024
      0102 Bank     }
      AR sub-rows   } — same < 2024 addition; total and CA totals recalculated
      0106 Stock    — ADD inh_stock (JSON, positive; covers 2016–2023) + ADD Level S
                       stock from subs 100002/3/4/7/8 for years < 2024

    OTHER ASSETS (B)
      0205 Loan to Others Concern
      0206 Other Investment
        — ADD from subs 100002/3/4/7/8, years < 2024 only; Total Other Assets recalc.

    FIXED ASSETS (C)
      0301–0308 — ADD from subs 100002/3/4/7/8, years < 2024 only;
                  Total Fixed Assets and Total Assets recalculated.

    CURRENT LIABILITIES (A)
      0901 Accrued, AP sub-rows + total, 0904 Money Agent, 0904 Recon,
      0905 C&F, 0906 Others
        — ADD from subs, years < 2024 only (values stay negative = correct).

    SHORT / LONG-TERM LOANS (B, C, E)
      1001 STB, 1001 ST, Total STL, 1201 LTL
        — ADD from subs, < 2024.

    RESERVE & FUNDS (D)
      1101 Emp Fund, 1102 Dir Award, 1102 Rent Tax, 1103 Edu, 1104 Sec
        — ADD from subs, < 2024; Total RF and Total Liabilities recalculated.

    EQUITY
      1301 Share Capital, 1301A Retained Earning
        — ADD from subs, < 2024 only.

    BALANCING (step 12)
      After all rows are set, error_adj absorbs the gap so Balance Check = 0.
      Total Equity, Total LE, Balance Check recalculated last.

    adj_data keys consumed
    ──────────────────────
    stock_inh     : pd.Series — I&H stock from JSON (positive, 2016–2023)
    cash_add      : pd.Series — cash CE to add (100001 cash × ih_ratio, positive)
    bank_add      : pd.Series — bank to add (100001 bank × ih_ratio, positive)
    ar_main_add   : pd.Series — AR main to add (positive)
    ar_agent_add  : pd.Series — AR agent to add (positive)
    ar_local_add  : pd.Series — AR local to add (positive)
    sub_bs_raws   : list[DataFrame] — raw Level 0 BS DataFrames for subs 100002/3/4/7/8
    """
    df  = bs_gi_s.copy()
    num = df.select_dtypes("number").columns
    z   = pd.Series(0.0, index=num)

    # ── Year masks ────────────────────────────────────────────────────────────
    pre2024 = np.array([_period_key(c)[0] <  2024 for c in num])
    le2024  = np.array([_period_key(c)[0] <= 2024 for c in num])

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _madd(base: pd.Series, add: pd.Series, mask: np.ndarray) -> pd.Series:
        """Return base + add where mask is True, else base unchanged."""
        return base + pd.Series(np.where(mask, add.values, 0.0), index=num)

    sub_bs_raws: list = adj_data.get("sub_bs_raws", [])

    def _sub_codes(codes: list) -> pd.Series:
        """Sum account codes across subsidiary raw BS DataFrames (no sign flip)."""
        tot = z.copy()
        for raw in sub_bs_raws:
            if raw is None or raw.empty:
                continue
            tot = tot + _ls_sum_raw(raw, codes, "ac_code").reindex(num, fill_value=0.0)
        return tot

    def _sub_pfx(prefix: str) -> pd.Series:
        """Sum prefix-matched codes across subsidiary raw BS DataFrames."""
        tot = z.copy()
        for raw in sub_bs_raws:
            if raw is None or raw.empty:
                continue
            tot = tot + _ls_prefix_sum(raw, prefix, "ac_code").reindex(num, fill_value=0.0)
        return tot

    # ── Unpack adj_data ───────────────────────────────────────────────────────
    inh_stock    = adj_data.get("stock_inh",    z.copy()).reindex(num, fill_value=0.0)
    cash_add     = adj_data.get("cash_add",     z.copy()).reindex(num, fill_value=0.0)
    bank_add     = adj_data.get("bank_add",     z.copy()).reindex(num, fill_value=0.0)
    # ar_local_add carries the FULL 100001 AR total (all sub-rows combined × ih_ratio)
    # — placed entirely into "Accounts Receivable (Local)" for years ≤ 2024
    ar_local_add = adj_data.get("ar_local_add", z.copy()).reindex(num, fill_value=0.0)

    # ── Read all Level S rows from GI's BS ───────────────────────────────────
    cash_ce      = _t_get(df, "0101-Cash & Cash Equivalent")
    bank_bal     = _t_get(df, "0102-Bank Balance")
    ar_main      = _t_get(df, "Accounts Receivable")
    ar_agent     = _t_get(df, "Recognized Agent (Dealers)")
    ar_local     = _t_get(df, "Accounts Receivable (Local)")
    prepaid      = _t_get(df, "0104-Prepaid Expenses")
    advance      = _t_get(df, "0105-Advance Accounts")
    stock        = _t_get(df, "0106-Stock in Hand")
    def_capex    = _t_get(df, "0201-Deferred Capital Expenditure")
    gift_items   = _t_get(df, "0202-Gift Items")
    loan_hosp    = _t_get(df, "0203-Loan to Hospital (Staff Salary)")
    loan_surma   = _t_get(df, "0203-Loan to Others Concern (Surma)")
    security_dep = _t_get(df, "0204-Security Deposit")
    loan_others  = _t_get(df, "0205-Loan to Others Concern")
    other_invest = _t_get(df, "0206-Other Investment")
    fa_office_eq = _t_get(df, "0301-Office Equipment")
    fa_corp_eq   = _t_get(df, "0302-Corporate Office Equipments")
    fa_furn      = _t_get(df, "0303-Furniture & Fixture")
    fa_trade_veh = _t_get(df, "0304-Trading Vehicles")
    fa_priv_veh  = _t_get(df, "0305-Private Vehicles")
    fa_plant     = _t_get(df, "0306-Plants & Machinery")
    fa_intang    = _t_get(df, "0307-Intangible Asset")
    fa_land      = _t_get(df, "0308-Land & Building")
    accrued      = _t_get(df, "0901-Accrued Expenses")
    ap_local     = _t_get(df, "Accounts Payable (Local)")
    ap_internal  = _t_get(df, "Accounts Payable (Internal)")
    ap_intl      = _t_get(df, "Accounts Payable (International)")
    money_agent  = _t_get(df, "0904-Money Agent Liability")
    recon_liab   = _t_get(df, "0904-Reconciliation Liability")
    cf_liab      = _t_get(df, "0905-C & F Liability")
    others_liab  = _t_get(df, "0906-Others Liability")
    stb_loan     = _t_get(df, "1001-Short Term Bank Loan (B)")
    st_loan      = _t_get(df, "1001-Short Term Loan (Related Parties)")
    ltl_loan     = _t_get(df, "1201-Long Term Bank Loan (E)")
    emp_fund     = _t_get(df, "1101-Employee Fund")
    dir_award    = _t_get(df, "1102-Directors Award Fund")
    rent_tax     = _t_get(df, "1102-Office Rent Tax Fund")
    edu_fund     = _t_get(df, "1103-Employee Educational Fund")
    sec_fund     = _t_get(df, "1104-Security Deposit Fund")
    share_cap    = _t_get(df, "1301-Share Capital")
    retained     = _t_get(df, "1301A-Non-Cash Capital (Retained Earning)")
    error_adj    = _t_get(df, "1302-Error Adjustment For Retained Earning")
    np_for_bs    = _t_get(df, "Net Profit/Loss")

    # ════════════════════════════════════════════════════════════════════════
    # CURRENT ASSETS
    # ════════════════════════════════════════════════════════════════════════
    # Cash & Bank: ADD trading-adj amounts (positive) for years < 2024
    # (2024+ = Level S; no Level T interaction)
    new_cash_ce  = _madd(cash_ce,  cash_add,     pre2024)
    new_bank_bal = _madd(bank_bal, bank_add,     pre2024)

    # AR adjustments:
    #   "Accounts Receivable"        ← ADD sum of all AR from subs 100002/3/4/7/8
    #                                   (codes 01030001+01030002+01030003), years < 2024
    #   "Recognized Agent (Dealers)" ← unchanged
    #   "Accounts Receivable (Local)"← ADD full 100001 AR total × ih_ratio, years < 2024
    # (2024+ = Level S; no Level T interaction for any AR row)
    sub_ar_all   = _sub_codes(["01030001","01030002","01030003"])
    new_ar_main  = _madd(ar_main,  sub_ar_all,   pre2024)   # subs → ar_main,  < 2024
    new_ar_agent = ar_agent                                  # unchanged
    new_ar_local = _madd(ar_local, ar_local_add, pre2024)   # 100001 total → ar_local, < 2024
    new_ar_total = new_ar_main + new_ar_agent + new_ar_local

    # Stock: ADD inh_stock (positive from JSON; zero outside 2016–2023), years < 2024
    #        ADD subsidiary stock for years < 2024
    # (2024+ = Level S; no Level T interaction)
    sub_stock    = _sub_codes(["01060003","01060001","01060002"])
    new_stock    = stock + _madd(z, inh_stock, pre2024) + _madd(z, sub_stock, pre2024)

    new_total_ca = (new_cash_ce + new_bank_bal + new_ar_total +
                    prepaid + advance + new_stock)

    # ════════════════════════════════════════════════════════════════════════
    # OTHER ASSETS (B)  — add 0205 + 0206 from subs, < 2024 only
    # ════════════════════════════════════════════════════════════════════════
    sub_loan_oth  = _sub_codes(["02050001","02050002","02050003","02050004","02050005",
                                 "02050006","02050007","02050008","02050009","02050010",
                                 "02050011","02050012","02050013","02050014","02050015",
                                 "02050016","02050017"])
    sub_oth_inv   = _sub_codes(["02060001"])
    new_loan_oth  = _madd(loan_others,  sub_loan_oth, pre2024)
    new_oth_inv   = _madd(other_invest, sub_oth_inv,  pre2024)
    new_total_oa  = (def_capex + gift_items + loan_hosp + loan_surma +
                     security_dep + new_loan_oth + new_oth_inv)

    # ════════════════════════════════════════════════════════════════════════
    # FIXED ASSETS (C)  — add 0301–0308 from subs, < 2024 only
    # ════════════════════════════════════════════════════════════════════════
    new_fa_oe    = _madd(fa_office_eq, _sub_pfx("03010"), pre2024)
    new_fa_ce    = _madd(fa_corp_eq,   _sub_pfx("03020"), pre2024)
    new_fa_furn  = _madd(fa_furn,      _sub_pfx("03030"), pre2024)
    new_fa_tveh  = _madd(fa_trade_veh, _sub_pfx("03040"), pre2024)
    new_fa_pveh  = _madd(fa_priv_veh,  _sub_pfx("03050"), pre2024)
    new_fa_plant = _madd(fa_plant,     _sub_pfx("03060"), pre2024)
    new_fa_int   = _madd(fa_intang,    _sub_pfx("03070"), pre2024)
    new_fa_land  = _madd(fa_land,      _sub_pfx("03080"), pre2024)
    new_total_fa = (new_fa_oe + new_fa_ce + new_fa_furn + new_fa_tveh +
                    new_fa_pveh + new_fa_plant + new_fa_int + new_fa_land)

    new_total_assets = new_total_ca + new_total_oa + new_total_fa

    # ════════════════════════════════════════════════════════════════════════
    # CURRENT LIABILITIES (A)  — add from subs, < 2024 only
    # (liabilities are negative in Level 0 / Level S → stays negative)
    # ════════════════════════════════════════════════════════════════════════
    new_accrued     = _madd(accrued,     _sub_codes(["09010001","09010002","09010003",
                                                      "09010004","09010005","09010006"]), pre2024)
    new_ap_local    = _madd(ap_local,    _sub_codes(["09030001","09030002"]),             pre2024)
    new_ap_internal = _madd(ap_internal, _sub_codes(["09030003"]),                        pre2024)
    new_ap_intl     = _madd(ap_intl,     _sub_codes(["09030004"]),                        pre2024)
    new_ap_total    = new_ap_local + new_ap_internal + new_ap_intl
    new_money_agent = _madd(money_agent, _sub_codes(["09040001","09040002","09040003",
                                                      "09040004","09040005","09040007"]), pre2024)
    new_recon_liab  = _madd(recon_liab,  _sub_codes(["09040006"]),                        pre2024)
    new_cf_liab     = _madd(cf_liab,     _sub_pfx("09050"),                               pre2024)
    new_others_liab = _madd(others_liab, _sub_codes(["09060001"]),                        pre2024)
    new_total_cl    = (new_accrued + new_ap_total + new_money_agent +
                       new_recon_liab + new_cf_liab + new_others_liab)

    # ════════════════════════════════════════════════════════════════════════
    # SHORT / LONG-TERM LOANS  (B, C, E)  — add from subs, < 2024 only
    # ════════════════════════════════════════════════════════════════════════
    new_stb_loan  = _madd(stb_loan, _sub_codes(["10010001","10010002","10010003",
                                                 "10010004","10010005","10010006",
                                                 "10010007","10010008"]),           pre2024)
    new_st_loan   = _madd(st_loan,  _sub_codes(["10020001","10020002","10020003",
                                                 "10020004","10020005","10020006",
                                                 "10020007","10020008","10020009",
                                                 "10020010","10020011","10020012",
                                                 "10020013","10020014","10020015",
                                                 "10020016","10020017"]),           pre2024)
    new_total_stl = new_stb_loan + new_st_loan
    new_ltl_loan  = _madd(ltl_loan, _sub_codes(["12010001","12010002"]),            pre2024)

    # ════════════════════════════════════════════════════════════════════════
    # RESERVE & FUNDS (D)  — add from subs, < 2024 only
    # ════════════════════════════════════════════════════════════════════════
    new_emp_fund  = _madd(emp_fund,  _sub_codes(["11010001","11010002","11010003"]), pre2024)
    new_dir_award = _madd(dir_award, _sub_codes(["11020002"]),                       pre2024)
    new_rent_tax  = _madd(rent_tax,  _sub_codes(["11020001","11020003"]),             pre2024)
    new_edu_fund  = _madd(edu_fund,  _sub_codes(["11030001"]),                        pre2024)
    new_sec_fund  = _madd(sec_fund,  _sub_codes(["11040001","11040002"]),             pre2024)
    new_total_rf  = (new_emp_fund + new_dir_award + new_rent_tax +
                     new_edu_fund + new_sec_fund)

    new_total_liab = new_total_cl + new_total_stl + new_ltl_loan + new_total_rf

    # ════════════════════════════════════════════════════════════════════════
    # EQUITY  — add share capital + retained from subs, < 2024 only
    # (equity is negative in Level 0 / Level S → stays negative)
    # ════════════════════════════════════════════════════════════════════════
    new_share_cap = _madd(share_cap, _sub_codes(["13010001"]),                        pre2024)
    new_retained  = _madd(retained,  _sub_codes(["13010003","13010004","13010005"]),  pre2024)

    # ════════════════════════════════════════════════════════════════════════
    # BALANCING — error_adj absorbs the gap; Balance Check → 0  (step 12)
    # All asset / liability / equity rows must be finalised BEFORE this block.
    # ════════════════════════════════════════════════════════════════════════
    intermediate_check = (new_total_assets + new_total_liab +
                          new_share_cap + new_retained + error_adj + np_for_bs)
    new_error_adj  = error_adj - intermediate_check
    new_total_equity = new_share_cap + new_retained + new_error_adj + np_for_bs
    new_total_le     = new_total_liab + new_total_equity

    # ── Write back every adjusted row ────────────────────────────────────────
    # Current Assets
    _t_set(df, "0101-Cash & Cash Equivalent",            new_cash_ce)
    _t_set(df, "0102-Bank Balance",                      new_bank_bal)
    _t_set(df, "Accounts Receivable",                    new_ar_main)
    _t_set(df, "Recognized Agent (Dealers)",             new_ar_agent)
    _t_set(df, "Accounts Receivable (Local)",            new_ar_local)
    _t_set(df, "0103-Accounts Receivable (Total)",       new_ar_total)
    _t_set(df, "0106-Stock in Hand",                     new_stock)
    _t_set(df, "01-Total Current Assets (A)",            new_total_ca)
    # Other Assets
    _t_set(df, "0205-Loan to Others Concern",            new_loan_oth)
    _t_set(df, "0206-Other Investment",                  new_oth_inv)
    _t_set(df, "Total Other Assets (B)",                 new_total_oa)
    # Fixed Assets
    _t_set(df, "0301-Office Equipment",                  new_fa_oe)
    _t_set(df, "0302-Corporate Office Equipments",       new_fa_ce)
    _t_set(df, "0303-Furniture & Fixture",               new_fa_furn)
    _t_set(df, "0304-Trading Vehicles",                  new_fa_tveh)
    _t_set(df, "0305-Private Vehicles",                  new_fa_pveh)
    _t_set(df, "0306-Plants & Machinery",                new_fa_plant)
    _t_set(df, "0307-Intangible Asset",                  new_fa_int)
    _t_set(df, "0308-Land & Building",                   new_fa_land)
    _t_set(df, "Total Fixed Assets (C)",                 new_total_fa)
    _t_set(df, "Total Assets (A+B+C)",                   new_total_assets)
    # Current Liabilities
    _t_set(df, "0901-Accrued Expenses",                  new_accrued)
    _t_set(df, "Accounts Payable (Local)",               new_ap_local)
    _t_set(df, "Accounts Payable (Internal)",            new_ap_internal)
    _t_set(df, "Accounts Payable (International)",       new_ap_intl)
    _t_set(df, "0903-Accounts Payable (Total)",          new_ap_total)
    _t_set(df, "0904-Money Agent Liability",             new_money_agent)
    _t_set(df, "0904-Reconciliation Liability",          new_recon_liab)
    _t_set(df, "0905-C & F Liability",                   new_cf_liab)
    _t_set(df, "0906-Others Liability",                  new_others_liab)
    _t_set(df, "Current Liability (A)",                  new_total_cl)
    # Short / Long-term Loans
    _t_set(df, "1001-Short Term Bank Loan (B)",          new_stb_loan)
    _t_set(df, "1001-Short Term Loan (Related Parties)", new_st_loan)
    _t_set(df, "Total Short Term Liability (C)",         new_total_stl)
    _t_set(df, "1201-Long Term Bank Loan (E)",           new_ltl_loan)
    # Reserve & Funds
    _t_set(df, "1101-Employee Fund",                     new_emp_fund)
    _t_set(df, "1102-Directors Award Fund",              new_dir_award)
    _t_set(df, "1102-Office Rent Tax Fund",              new_rent_tax)
    _t_set(df, "1103-Employee Educational Fund",         new_edu_fund)
    _t_set(df, "1104-Security Deposit Fund",             new_sec_fund)
    _t_set(df, "Total Reserve & Funds (D)",              new_total_rf)
    _t_set(df, "Total Liabilities (A+B+C+D+E)",          new_total_liab)
    # Equity (balancing last)
    _t_set(df, "1301-Share Capital",                     new_share_cap)
    _t_set(df, "1301A-Non-Cash Capital (Retained Earning)", new_retained)
    _t_set(df, "1302-Error Adjustment For Retained Earning", new_error_adj)
    _t_set(df, "Total Equity",                           new_total_equity)
    _t_set(df, "Total Liabilities & Equity",             new_total_le)
    _t_set(df, "Balance Check",                          new_total_assets + new_total_le)
    return df


def build_cfs_from_bs_t_gi(
    bs_gi_t: pd.DataFrame,
    net_income_gi: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the GI (100000) Level T Cash Flow Statement from the adjusted BS.

    Delegates to build_cfs_from_bs_t().  Once adjust_bs_level_t_gi() produces
    a fully balanced BS (Balance Check = 0), the CFS check will also be zero
    automatically because build_cfs_from_bs_t reads named rows from the BS.
    """
    return build_cfs_from_bs_t(bs_gi_t, net_income_gi)
