"""
processing/consolidation.py
============================
Consolidated financial statements for the group.

Levels produced
---------------
  Level C   — Simple concatenation of all ZID Level-0 statements.
               Retains the `zid` column.  No eliminations.
               Columns: zid | ac_code | ac_name | <period …>

  Level C2  — Proper consolidation applying the rules in
               data/consolidation_rules.json:
                 • InternalLoans  – paired interco loans; eliminate if
                                    every year nets to zero, else keep both rows
                 • External BS    – real external AR; sum by code+name
                 • ARAP2 BS       – external AR/AP; sum by code+name
                 • Allother2 BS   – all remaining BS items; sum by code+name
                 • SalesCOGS IS   – keep external-ZID sales; net internal
                                    sales out of COGS
                 • Expenses IS    – all other IS lines; sum by code+name
               No `zid` column in the output.

  Level 1   — Aggregated from Level C2 via the shared hierarchy / level_builder
  Level 2   — Same
  Level S   — Management view from Level C2 via build_cfs_level_s

Cashflow
--------
  Level C   — separate CFS builder (retains zid grouping)
  Level C2+ — standard make_cashflow_statement_level0 / level builders

Public API
----------
  load_consolidation_rules()                  → dict
  build_level_c(zid_statements, kind)         → DataFrame
  build_level_c2_bs(zid_bs_frames)            → DataFrame   (BS)
  build_level_c2_is(zid_is_frames)            → DataFrame   (IS)
  build_level_c_cfs(c_bs, c_is, perspective)  → (full_df, summary_df)
"""

from __future__ import annotations

import json
import functools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ── path helpers ──────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_RULES_PATH = _DATA_DIR / "consolidation_rules.json"


# ══════════════════════════════════════════════════════════════════════════════
# Rules loader (cached)
# ══════════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=1)
def load_consolidation_rules() -> dict:
    """Load and cache consolidation_rules.json."""
    with open(_RULES_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

_META_COLS = {"zid", "ac_code", "ac_name"}


def _num_cols(df: pd.DataFrame, extra_exclude: set | None = None) -> list:
    """Return numeric (period) column names, excluding meta columns."""
    excl = _META_COLS | (extra_exclude or set())
    return [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]


def _sum_by_code_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a Level-C or Level-C2 DataFrame by (ac_code, ac_name).
    Numeric columns are summed; rows where ALL numeric values are zero are dropped.
    """
    if df.empty:
        return df
    num = _num_cols(df)
    agg = (
        df.groupby(["ac_code", "ac_name"], as_index=False, sort=False)[num]
        .sum()
    )
    # drop all-zero rows
    agg = agg[agg[num].abs().sum(axis=1) > 0].reset_index(drop=True)
    return agg


def _reorder_periods(df: pd.DataFrame, num: list) -> pd.DataFrame:
    """Ensure period columns appear in chronological order."""
    from processing.financial import _period_key  # shared helper
    try:
        ordered = sorted(num, key=_period_key)
    except Exception:
        ordered = num
    meta = [c for c in df.columns if c not in num]
    return df[meta + ordered]


# ══════════════════════════════════════════════════════════════════════════════
# Level C  —  simple concatenation, retains ZID
# ══════════════════════════════════════════════════════════════════════════════

def build_level_c(
    zid_frames: Dict[int, pd.DataFrame],
    kind: str = "bs",
) -> pd.DataFrame:
    """
    Concatenate per-ZID Level-0 statements into a single Level-C frame.

    Parameters
    ----------
    zid_frames : dict  {zid: DataFrame}
        Each DataFrame has at minimum ac_code, ac_name, and period columns.
        The `kind` hint ('bs' or 'is') is for documentation; no logic differs.
    kind : str
        'bs' or 'is' — used only as documentation in callers.

    Returns
    -------
    DataFrame with columns: zid | ac_code | ac_name | <period …>
    """
    if not zid_frames:
        return pd.DataFrame(columns=["zid", "ac_code", "ac_name"])

    parts: list[pd.DataFrame] = []
    for zid, df in zid_frames.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        # drop any pre-existing zid column before inserting the authoritative one
        if "zid" in tmp.columns:
            tmp = tmp.drop(columns=["zid"])
        tmp.insert(0, "zid", int(zid))
        parts.append(tmp)

    if not parts:
        return pd.DataFrame(columns=["zid", "ac_code", "ac_name"])

    combined = pd.concat(parts, ignore_index=True)
    num = _num_cols(combined)
    combined[num] = combined[num].fillna(0)

    # Deduplicate: if the same (zid, ac_code) appears more than once (e.g. from
    # multiple data sources), sum the numeric columns and keep one row per key.
    if combined.duplicated(subset=["zid", "ac_code"]).any():
        agg_num = combined.groupby(["zid", "ac_code", "ac_name"], as_index=False)[num].sum()
        combined = agg_num

    return _reorder_periods(combined, num)


# ══════════════════════════════════════════════════════════════════════════════
# Level C2  —  Balance Sheet
# ══════════════════════════════════════════════════════════════════════════════

def _internal_loans_rows(
    level_c_bs: pd.DataFrame,
    rules: dict,
) -> pd.DataFrame:
    """
    Process InternalLoans sheet rules.

    For each Remarks group:
      • Sum all entries' values year-by-year.
      • If every year nets to zero → eliminate (omit both rows).
      • Otherwise → keep both rows as-is (stripped of ZID).

    Returns a DataFrame of rows to include in Level C2 BS.
    """
    num = _num_cols(level_c_bs, extra_exclude={"zid"})

    # Build a quick lookup: (zid, ac_code) → row of numeric values
    level_c_bs = level_c_bs.copy()
    level_c_bs["zid"] = level_c_bs["zid"].astype(int)

    keep_rows: list[pd.DataFrame] = []

    for group in rules["internal_loans"]:
        entries = group["entries"]

        # Slice the actual current-period values from Level C
        group_rows: list[pd.DataFrame] = []
        for entry in entries:
            zid   = int(entry["zid"])
            code  = str(entry["ac_code"])
            mask  = (level_c_bs["zid"] == zid) & (level_c_bs["ac_code"] == code)
            match = level_c_bs.loc[mask]
            if match.empty:
                # Row not present for this period (zero value) — create a
                # placeholder so the net check still accounts for it correctly.
                row_dict = {"ac_code": code, "ac_name": entry["ac_name"]}
                row_dict.update({c: 0.0 for c in num})
                group_rows.append(pd.DataFrame([row_dict]))
            else:
                row = match.drop(columns=["zid"]).head(1).copy()
                group_rows.append(row)

        if not group_rows:
            continue

        group_df = pd.concat(group_rows, ignore_index=True)

        # Net per year
        net = group_df[num].sum()

        # Eliminate only if ALL years net to zero
        if (net.abs() <= 0.005).all():
            continue  # perfectly matched — drop both sides

        # Not fully matched — keep all entries
        for row in group_rows:
            keep_rows.append(row)

    if not keep_rows:
        return pd.DataFrame(columns=["ac_code", "ac_name"] + num)

    result = pd.concat(keep_rows, ignore_index=True)
    result = result[result[num].abs().sum(axis=1) > 0]
    return result.reset_index(drop=True)


def _external_bs_rows(
    level_c_bs: pd.DataFrame,
    rules: dict,
) -> pd.DataFrame:
    """
    Extract External BS entries (real external AR) and sum by code+name.
    """
    num = _num_cols(level_c_bs, extra_exclude={"zid"})
    level_c_bs = level_c_bs.copy()
    level_c_bs["zid"] = level_c_bs["zid"].astype(int)

    zid_code_pairs = {
        (int(e["zid"]), str(e["ac_code"]))
        for e in rules["external_bs"]
    }

    mask = level_c_bs.apply(
        lambda r: (int(r["zid"]), str(r["ac_code"])) in zid_code_pairs, axis=1
    )
    subset = level_c_bs.loc[mask].drop(columns=["zid"])
    return _sum_by_code_name(subset)


def _arap_bs_rows(
    level_c_bs: pd.DataFrame,
    rules: dict,
) -> pd.DataFrame:
    """
    Extract ARAP2 BS entries and sum by code+name.
    """
    num = _num_cols(level_c_bs, extra_exclude={"zid"})
    level_c_bs = level_c_bs.copy()
    level_c_bs["zid"] = level_c_bs["zid"].astype(int)

    zid_code_pairs = {
        (int(e["zid"]), str(e["ac_code"]))
        for e in rules["arap_bs"]
    }

    mask = level_c_bs.apply(
        lambda r: (int(r["zid"]), str(r["ac_code"])) in zid_code_pairs, axis=1
    )
    subset = level_c_bs.loc[mask].drop(columns=["zid"])
    return _sum_by_code_name(subset)


def _allother_bs_rows(
    level_c_bs: pd.DataFrame,
    rules: dict,
) -> pd.DataFrame:
    """
    All remaining BS rows not claimed by InternalLoans / External / ARAP2.
    Summed by code+name.
    """
    level_c_bs = level_c_bs.copy()
    level_c_bs["zid"] = level_c_bs["zid"].astype(int)

    # Build exclusion sets
    il_pairs: set = set()
    for group in rules["internal_loans"]:
        for e in group["entries"]:
            il_pairs.add((int(e["zid"]), str(e["ac_code"])))

    ext_pairs = {(int(e["zid"]), str(e["ac_code"])) for e in rules["external_bs"]}
    arap_pairs = {(int(e["zid"]), str(e["ac_code"])) for e in rules["arap_bs"]}
    excluded = il_pairs | ext_pairs | arap_pairs

    mask = level_c_bs.apply(
        lambda r: (int(r["zid"]), str(r["ac_code"])) not in excluded, axis=1
    )
    subset = level_c_bs.loc[mask].drop(columns=["zid"])
    return _sum_by_code_name(subset)


def build_level_c2_bs(level_c_bs: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Level C2 consolidated Balance Sheet from the Level C BS.

    Processing order:
      1. InternalLoans  – interco loan pairs (eliminate if fully matched)
      2. External BS    – real external AR, summed by code+name
      3. ARAP2 BS       – external AR/AP, summed by code+name
      4. Allother2      – all remaining BS rows, summed by code+name

    Returns
    -------
    DataFrame with columns: ac_code | ac_name | <period …>
    No ZID column.
    """
    if level_c_bs.empty:
        return level_c_bs

    rules = load_consolidation_rules()
    num   = _num_cols(level_c_bs, extra_exclude={"zid"})

    parts = [
        _internal_loans_rows(level_c_bs, rules),
        _external_bs_rows(level_c_bs, rules),
        _arap_bs_rows(level_c_bs, rules),
        _allother_bs_rows(level_c_bs, rules),
    ]

    result = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    result[num] = result[num].fillna(0)
    # Final dedup: the same ac_code can appear in multiple sections when different
    # ZIDs route it to different buckets (e.g., ARAP for one ZID, allother for
    # another). Sum into one row so make_cashflow_statement_level0 gets a unique
    # (ac_code, ac_name) MultiIndex and .loc[key] returns a Series, not a DataFrame.
    result = _sum_by_code_name(result)
    return _reorder_periods(result, _num_cols(result))


# ══════════════════════════════════════════════════════════════════════════════
# Level C2  —  Income Statement
# ══════════════════════════════════════════════════════════════════════════════

def _sales_cogs_rows(
    level_c_is: pd.DataFrame,
    rules: dict,
) -> pd.DataFrame:
    """
    Consolidated Sales and COGS rows.

    Sales:
      Keep only external ZIDs (100000, 100001, 100005, 100006).
      Sum into a single Sales row.

    COGS:
      Consolidated COGS = sum(ALL ZIDs' COGS) − sum(internal ZIDs' Sales)
      Output as a single COGS row.
    """
    sc  = rules["sales_cogs"]
    ext_zids  = [int(z) for z in sc["external_sales_zids"]]
    int_zids  = [int(z) for z in sc["internal_sales_zids"]]
    sales_code = str(sc["sales_ac_code"])
    cogs_code  = str(sc["cogs_ac_code"])

    level_c_is = level_c_is.copy()
    level_c_is["zid"] = level_c_is["zid"].astype(int)
    num = _num_cols(level_c_is, extra_exclude={"zid"})

    def _slice(zids: list, code: str) -> pd.DataFrame:
        mask = level_c_is["zid"].isin(zids) & (level_c_is["ac_code"] == code)
        return level_c_is.loc[mask, num]

    # Consolidated Sales = sum of external ZIDs' sales
    ext_sales = _slice(ext_zids, sales_code).sum()

    # Consolidated COGS = all COGS - internal Sales
    all_cogs  = _slice(ext_zids + int_zids, cogs_code).sum()
    int_sales = _slice(int_zids, sales_code).sum()
    cons_cogs = all_cogs - int_sales

    rows = []
    if (ext_sales.abs() > 0).any():
        rows.append({"ac_code": sales_code, "ac_name": "Sales", **ext_sales.to_dict()})
    if (cons_cogs.abs() > 0).any():
        rows.append({"ac_code": cogs_code, "ac_name": "Cost Of Goods Sold", **cons_cogs.to_dict()})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ac_code", "ac_name"] + num)


def _expenses_rows(
    level_c_is: pd.DataFrame,
    rules: dict,
) -> pd.DataFrame:
    """
    All IS rows except the main Sales (08010001) and COGS (04010020) lines.
    Summed by code+name.
    """
    sc = rules["sales_cogs"]
    excl_codes = {str(sc["sales_ac_code"]), str(sc["cogs_ac_code"])}

    subset = level_c_is[~level_c_is["ac_code"].isin(excl_codes)].drop(
        columns=["zid"], errors="ignore"
    )
    return _sum_by_code_name(subset)


def build_level_c2_is(level_c_is: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Level C2 consolidated Income Statement from the Level C IS.

    Processing:
      1. SalesCOGS  – consolidated Sales and COGS
      2. Expenses   – all other IS lines, summed by code+name

    Returns
    -------
    DataFrame with columns: ac_code | ac_name | <period …>
    No ZID column.
    """
    if level_c_is.empty:
        return level_c_is

    rules = load_consolidation_rules()
    num   = _num_cols(level_c_is, extra_exclude={"zid"})

    parts = [
        _sales_cogs_rows(level_c_is, rules),
        _expenses_rows(level_c_is, rules),
    ]

    result = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    result[num] = result[num].fillna(0)
    # Defensive dedup: same ac_code can appear in multiple IS sections when
    # different ZIDs route it differently. Sum into one row per (ac_code, ac_name).
    result = _sum_by_code_name(result)
    return _reorder_periods(result, _num_cols(result))


# ══════════════════════════════════════════════════════════════════════════════
# Level C  —  Cashflow Statement
# ══════════════════════════════════════════════════════════════════════════════

def build_level_c_cfs(
    level_c_bs: pd.DataFrame,
    level_c_is: pd.DataFrame,
    selected_perspective: str = "Yearly",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the Level C Cash Flow Statement.

    Level C retains a `zid` column, so we cannot use the standard
    make_cashflow_statement_level0 directly (which expects ac_code as the
    first key column).  This function strips the ZID, sums by code+name to
    produce a deduplicated combined BS/IS, then delegates to the standard CFS
    builder.

    Parameters
    ----------
    level_c_bs : DataFrame  — Level C balance sheet (has `zid` column)
    level_c_is : DataFrame  — Level C income statement (has `zid` column)
    selected_perspective : str — 'Yearly' | 'Monthly'

    Returns
    -------
    (cfs_full_df, cfs_summary_df)  — same contract as make_cashflow_statement_level0
    """
    from processing.financial import make_cashflow_statement_level0, cash_open_close

    # Sum by code+name (collapse ZIDs) to get a clean aggregate BS/IS
    bs_agg = _sum_by_code_name(
        level_c_bs.drop(columns=["zid"], errors="ignore")
    )
    is_agg = _sum_by_code_name(
        level_c_is.drop(columns=["zid"], errors="ignore")
    )

    coc = cash_open_close(bs_agg)
    return make_cashflow_statement_level0(
        pl_df=is_agg,
        bs_df=bs_agg,
        coc_df=coc,
        selected_perspective=selected_perspective,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Level 1 / 2 / S  from Level C2
# ══════════════════════════════════════════════════════════════════════════════

def build_cons_level1(
    c2_bs: pd.DataFrame,
    c2_is: pd.DataFrame,
    selected_perspective: str = "Yearly",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build consolidated Level 1 BS and IS from Level C2.

    Delegates to the standard level_builder + add_np_and_balance_lv1.

    Returns
    -------
    (pl_lv1, bs_lv1, net_profit_series, dep_row)
    """
    from processing.financial import level_builder, add_np_and_balance_lv1

    pl_lv1, _ = level_builder(c2_is, section="IS")
    bs_lv1, _ = level_builder(c2_bs, section="BS")
    return add_np_and_balance_lv1(pl_lv1, bs_lv1, selected_perspective=selected_perspective)


def build_cons_level2(
    c2_bs: pd.DataFrame,
    c2_is: pd.DataFrame,
    selected_perspective: str = "Yearly",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Build consolidated Level 2 BS and IS from Level C2.

    Returns
    -------
    (pl_lv2, bs_lv2, net_profit_series)
    """
    from processing.financial import level_builder, add_np_and_balance_lv2

    _, pl_lv2 = level_builder(c2_is, section="IS")
    _, bs_lv2 = level_builder(c2_bs, section="BS")
    return add_np_and_balance_lv2(pl_lv2, bs_lv2, selected_perspective=selected_perspective)


def build_cons_level_s(
    c2_bs: pd.DataFrame,
    c2_is: pd.DataFrame,
    coc_lv0: pd.DataFrame,
    net_income_series: "pd.Series",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build consolidated Level S (management view) CFS from Level C2.

    Parameters
    ----------
    c2_bs            : Level C2 consolidated BS (no ZID)
    c2_is            : Level C2 consolidated IS (no ZID)
    coc_lv0          : Opening/closing cash from cash_open_close(c2_bs)
    net_income_series: Net income series (from Level C2 IS net profit calc)

    Returns
    -------
    (full_df, summary_df)  — same contract as build_cfs_level_s
    """
    from processing.financial import build_cfs_level_s

    return build_cfs_level_s(
        pl_raw=c2_is,
        bs_raw=c2_bs,
        coc_lv0=coc_lv0,
        net_income_series=net_income_series,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: build all CFS levels from C2 in one call
# ══════════════════════════════════════════════════════════════════════════════

def build_cons_cfs_all(
    c2_bs: pd.DataFrame,
    c2_is: pd.DataFrame,
    selected_perspective: str = "Yearly",
) -> dict:
    """
    Build all consolidated CFS levels (C2 base, L1, L2) in one call.

    Returns
    -------
    dict with keys:
        'cfs_c2'        : (full_df, summary_df)   — Level C2 cashflow
        'cfs_lv1'       : (full_df, summary_df)   — Level 1 cashflow
        'cfs_lv2'       : (full_df, summary_df)   — Level 2 cashflow
        'coc'           : opening/closing cash DataFrame
    """
    from processing.financial import (
        make_cashflow_statement_level0,
        cash_open_close,
        consolidate_cfs,
        build_cfs_level1_summary_df,
        build_cfs_level2_summary,
        add_np_and_balance_lv1,
        add_np_and_balance_lv2,
        level_builder,
        sort_pl_level0,
    )

    coc = cash_open_close(c2_bs)

    # C2 cashflow (Level 0 equivalent)
    cfs_c2, _ = make_cashflow_statement_level0(
        pl_df=c2_is, bs_df=c2_bs, coc_df=coc,
        selected_perspective=selected_perspective,
    )

    # Level 1
    cfs_lv1_raw = consolidate_cfs(cfs_c2, level=1)
    pl_lv1, bs_lv1, np_series, dep_row = add_np_and_balance_lv1(
        *level_builder(c2_is, "IS")[:1],
        *level_builder(c2_bs, "BS")[:1],
        selected_perspective=selected_perspective,
    )
    cfs_lv1, _ = build_cfs_level1_summary_df(
        cfs_lv1_raw, net_profit=np_series.to_frame().T,
        depreciation_df=dep_row, coc_df=coc,
    )

    # Level 2
    cfs_lv2_raw = consolidate_cfs(cfs_c2, level=2)
    _, pl_lv2 = level_builder(c2_is, "IS")
    _, bs_lv2 = level_builder(c2_bs, "BS")
    _, _, np_series2 = add_np_and_balance_lv2(
        pl_lv2, bs_lv2, selected_perspective=selected_perspective
    )
    cfs_lv2, _ = build_cfs_level2_summary(
        cfs_lv2_raw, net_profit_lv2=np_series2.to_frame().T,
        depreciation_lv1_row=dep_row, coc_df=coc,
    )

    return {
        "cfs_c2":  (cfs_c2, _),
        "cfs_lv1": (cfs_lv1, _),
        "cfs_lv2": (cfs_lv2, _),
        "coc":     coc,
    }
