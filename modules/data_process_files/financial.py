import pandas as pd
from datetime import datetime
from db import db_utils
import streamlit as st
import numpy as np
from pathlib import Path
import json
from collections import OrderedDict
from typing import Dict, Tuple, List, Set
import re, ast

HERE = Path(__file__).resolve().parent.parent
JSON_PATH = HERE / "hierarchy.json"

@st.cache_data
def _load_raw() -> dict:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def get_filtered_master(zid, excluded_acctypes):
    df_master = db_utils.get_gl_master(zid)
    return df_master[~df_master['ac_type'].isin(excluded_acctypes)]

@st.cache_data
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
        df = db_utils.get_gl_details(
            zid=zid, project=project, year=year,
            smonth=cs_month, emonth=ce_month,
            is_bs=True, is_project=bool(project))

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
        df = db_utils.get_gl_details(
            zid=zid, project=project, year=year - 1,
            smonth=1, emonth=12,
            is_bs=True, is_project=bool(project))

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
    df_curr = db_utils.get_gl_details(
        zid=zid, project=project, year=year,
        smonth=cs_month, emonth=ce_month,
        is_bs=False, is_project=bool(project))
    df_prev = db_utils.get_gl_details(
        zid=zid, project=project, year=year - 1,
        smonth=1, emonth=12,
        is_bs=False, is_project=bool(project))

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

@st.cache_data
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
        raw = db_utils.get_gl_details(
            zid=zid,
            project=project,
            year=yr,
            smonth=start_month,
            emonth=end_month,
            is_bs=is_bs,
            is_project=bool(project),
        )
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

@st.cache_data
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

@st.cache_data
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

def _prefix_lookup(code):
    for pfx, (sect, sgn) in PREFIX_TO_SECTION_SIGN.items():
        if code.startswith(pfx):
            return sect, sgn
    return "operating", 0      # ignore unmapped

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
    bs_df.loc[bs_df[name_col].eq("Net Profit/Loss") & bs_df[code_col].eq(""),name_col] = "Change in Net Profit/Loss"

    print(bs_df.columns,"bs_df.columns")
    print(pl_df.columns,"pl_df.columns")

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

    # print("Years (B/S):", num_bs, "Common:", common)  # optional: remove after testing
    print(col_head,"col_head")

    if len(col_head) < 2:
        raise ValueError("Need at least two overlapping years between P/L and B/S")

    # ---- B  Net-Profit series (strip any pre-existing NP row) -----------
    pl_no_np  = pl_df[pl_df[name_col] != "Net Profit/Loss"]
    net_profit = (pl_no_np[col_head].sum()) #* -1

    # ---- C  Year-on-year deltas for every ac_code -----------------------
    bs_work   = bs_df.set_index([code_col, name_col])[col_head]
    bs_delta  = bs_work.diff(axis=1).iloc[:, 1:]          # target − base
    bs_delta.columns = col_head[1:]

    # ---- D  Allocate deltas & build detailed frames ---------------------
    section_frames = {"operating": [], "investing": [], "financing": []}

    for (code, name), delta_row in bs_delta.iterrows():
        if (not code) and str(name).strip() == "Change in Net Profit/Loss":
            section, sign = "financing", 1      # treat as financing item
        else:
            section, sign = _prefix_lookup(code)
        # section, sign = _prefix_lookup(code)
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
    dep_series   = pl_df.loc[pl_df[code_col].str.contains("06360001", case=False, na=False),col_head].sum()
    dep_row      = _total_row("Depreciation", dep_series.loc[col_head[1:]])
    wc_total_row = _total_row("Δ Working Capital",  op_total)
    cfo_row      = _total_row("Cash from Operations",
                                 op_total) #+ net_profit.loc[col_head[1:]]) dep_series.loc[col_head[1:]]
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
            dep_row,
            op_details, wc_total_row, cfo_row,
            inv_details, capex_row,
            fin_details, fin_row,
            opening_row,          # ← INSERTED between financing & ΔCash
            ndc_row,
            calc_row,             # ← reconciliation lines
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

@st.cache_data
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
        acct = code.strip()                       # e.g. "01010101"
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
    headers = headers[headers[name_col] == "Change in Net Profit/Loss"]

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
    # 1️⃣  normalise NP & Depreciation to Series aligned to cfs columns
    # ------------------------------------------------------------------
    years_raw = cfs_lv1.select_dtypes("number").columns.tolist()  # e.g., Int64Index([2022, 2023,...])
    years_str = [str(y) for y in years_raw]

    # 2) Normalise NP & Depreciation → Series indexed by string years
    np_series  = _as_year_series(net_profit, years_str)
    dep_series = _as_year_series(depreciation_df.select_dtypes("number").iloc[0], years_str)

    def _row(label: str, series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({name_col: [label], **series.to_dict()})

    np_row  = _row("Net Profit/Loss", np_series)
    dep_row = _row("Depreciation",    dep_series)

    # 3) Tag detail rows
    def bucket_section(label: str) -> str | None:
        if label == "Change in Net Profit/Loss":
            return "financing"
        if not isinstance(label, str) or "-" not in label:
            return None
        return _prefix_lookup(label.split("-")[0][:2])[0]

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
    cfo_row = _row("Cash from Operations", wc_total)  # (+ dep_series + np_series) if you later enable them

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
            dep_row,
            work.loc[op_mask],
            wc_row,
            cfo_row,
            work.loc[inv_mask],
            inv_tot,
            work.loc[fin_mask],
            fin_tot,
            opening_row,          # ← inserted before Net ΔCash
            ndc_row,
            calc_row,
            close_row,
            check_row,
        ],
        ignore_index=True
    ).drop(columns="section")

    summary_labels = [
        "Net Profit/Loss", "Depreciation","Δ Working Capital",
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

    # 1) Net-profit & Depreciation as aligned Series (string-indexed)
    np_series  = _as_year_series(net_profit_lv2,       years_str) * -1
    dep_series = _as_year_series(depreciation_lv1_row, years_str)

    def _row(label: str, series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({name_col: [label], **series.to_dict()})

    np_row  = _row("Net Profit/Loss", np_series)
    dep_row = _row("Depreciation",    dep_series)

    # classify
    def section_of(label: str) -> str | None:
        if label == "Change in Net Profit/Loss":
            return "financing"
        if not isinstance(label, str) or "-" not in label:
            return None
        return _prefix_lookup(label.split("-")[0][:2])[0]

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
    cfo_row = _row("Cash from Operations", wc_tot)  # (+ dep_series + np_series) if needed

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
            dep_row,
            work.loc[op_mask],  wc_row,  cfo_row,
            work.loc[inv_mask], inv_row,
            work.loc[fin_mask], fin_row,
            opening_row,                   # inserted before ΔCash
            ndc_row,
            calc_row, close_row, check_row
        ],
        ignore_index=True
    ).drop(columns="section")

    summary_labels = [
        "Net Profit/Loss", "Depreciation","Δ Working Capital",
        "Cash from Operations", "Cash from Investing",
        "Cash from Financing","Opening Cash & CE", 
        "Net ΔCash","Calculated Closing Cash & CE",
        "Closing Cash & CE","Cash-flow Check"
    ]
    sum_df = (full_df[full_df[name_col].isin(summary_labels)]
                     .reset_index(drop=True))

    return full_df, sum_df

