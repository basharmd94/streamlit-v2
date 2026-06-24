# processing/salesman_score.py
# Composite performance score per salesman for Target Management's
# "Salesman Score" tab. Balances reuse salesman_due.build_trickledown_balances
# directly (same FIFO due-by-origin-month methodology as Collection Analysis
# -> Salesman Due -> main due report) so the numbers match exactly.

import numpy as np
import pandas as pd

from processing import salesman_due as sd


def month_choices(today: pd.Timestamp) -> list:
    """[(label, year, month), ...] for the real current calendar month and
    the 2 months immediately before it, most recent first."""
    cur = pd.Timestamp(today.year, today.month, 1)
    return [
        ((cur - pd.DateOffset(months=i)).strftime("%b %Y"),
         int((cur - pd.DateOffset(months=i)).year),
         int((cur - pd.DateOffset(months=i)).month))
        for i in range(3)
    ]


def is_real_current_month(year: int, month: int, today: pd.Timestamp) -> bool:
    return int(year) == today.year and int(month) == today.month


def compute_salesman_balances_trickledown(ar_clean: pd.DataFrame, months_back: int = 5) -> pd.DataFrame:
    """Per-salesman AR balance broken down by origin month, reusing
    salesman_due.build_trickledown_balances directly — the same FIFO
    customer-credit allocation Collection Analysis -> Salesman Due ->
    main due report uses — so the numbers match exactly.

    months_back only controls which months get their own named "YYYY_MM"
    column vs. fold into the opening-balance bucket; the per-month value
    itself is unaffected. Default of 5 covers every (year, month) the
    Salesman Score month dropdown can ever need: the dropdown offers the
    current month and the 2 before it, and for each selection we look back
    2 more months, so the oldest possible origin month is (today - 4
    months) — exactly the start of a 5-month window anchored on today.

    Returns a DataFrame indexed by xsp (salesman id, str), one column per
    "YYYY_MM" window month plus salesman_due._OPENING_LABEL for anything
    older (not expected to be looked up given the months_back=5 guarantee
    above, but present for completeness).
    """
    if ar_clean is None or ar_clean.empty:
        return pd.DataFrame()

    report = sd.build_trickledown_balances(ar_clean, months_back=months_back)
    if report.empty:
        return pd.DataFrame()

    value_cols = [c for c in report.columns if c not in ("xsp", "xsub")]
    return report.groupby("xsp")[value_cols].sum()


def _peer_relative(series: pd.Series) -> pd.Series:
    """Scales to [0, 1] by dividing by the series' own max (peer-relative —
    the top value in this table gets 1.0, others scaled against it). Negative
    inputs are clipped to 0 first (e.g. a customer credit balance shouldn't
    count as a negative deduction). Returns all 0s if every value is <= 0."""
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    peak = s.max()
    if not peak or peak <= 0:
        return pd.Series(0.0, index=s.index)
    return (s / peak).clip(0.0, 1.0)


def compute_salesman_scores(rows_df: pd.DataFrame) -> pd.DataFrame:
    """Adds score component + final score columns to a per-salesman rows_df.

    Required input columns:
      spid, target, sales (gross), net_sales, coll (collection amount),
      uniq_prods, uniq_cust, balance_recent2 (sum of the 2 months before the
      selected one), balance_this_month (the selected month's own balance).

    Score (0-100, higher is better):
      +45 x min(net_sales/target, 100%)        — 0 if target is unset (<=0)
      +45 x min(coll/(1.02*sales), 100%)        — 0 if sales is 0
      +5  x peer-relative(uniq_prods)
      +5  x peer-relative(uniq_cust)
      -6  x peer-relative(return% = (sales-net_sales)/sales)   [30% of the 20% negative bucket]
      -12 x peer-relative(balance_recent2)                     [60% of the 20% negative bucket]
      -2  x peer-relative(balance_this_month)                  [10% of the 20% negative bucket]
    Clipped to [0, 100]. has_target flags target<=0 (scored 0 on that component).
    """
    d = rows_df.copy()
    for c in ["target", "sales", "net_sales", "coll", "uniq_prods", "uniq_cust",
              "balance_recent2", "balance_this_month"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    d["has_target"] = d["target"] > 0
    pct_target = np.where(d["has_target"], d["net_sales"] / d["target"].replace(0, np.nan), 0.0)
    d["score_target"] = np.clip(pct_target, 0.0, 1.0) * 45.0

    pct_coll = np.where(d["sales"] > 0, d["coll"] / (1.02 * d["sales"].replace(0, np.nan)), 0.0)
    d["score_collection"] = np.clip(pct_coll, 0.0, 1.0) * 45.0

    d["score_products"] = _peer_relative(d["uniq_prods"]) * 5.0
    d["score_customers"] = _peer_relative(d["uniq_cust"]) * 5.0

    return_pct = np.where(d["sales"] > 0, (d["sales"] - d["net_sales"]) / d["sales"] * 100.0, 0.0)
    d["return_pct"] = return_pct
    d["neg_returns"] = _peer_relative(pd.Series(return_pct, index=d.index)) * 6.0
    d["neg_balance_recent2"] = _peer_relative(d["balance_recent2"]) * 12.0
    d["neg_balance_this_month"] = _peer_relative(d["balance_this_month"]) * 2.0

    d["negative_total"] = d["neg_returns"] + d["neg_balance_recent2"] + d["neg_balance_this_month"]
    d["score"] = (
        d["score_target"] + d["score_collection"] + d["score_products"] + d["score_customers"]
        - d["negative_total"]
    ).clip(lower=0.0, upper=100.0).round(1)

    return d.sort_values("score", ascending=True).reset_index(drop=True)
