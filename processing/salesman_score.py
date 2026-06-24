# processing/salesman_score.py
# Composite performance score per salesman for Target Management's
# "Salesman Score" tab. Builds on the same AR ledger cleanup as
# processing/salesman_due.py (prep_ar_ledger) for point-in-time balance
# snapshots, which is a different number from that module's trickledown
# due-by-origin-month columns — see compute_salesman_balance_asof.

import numpy as np
import pandas as pd


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


def balance_cutoff_date(year: int, month: int, today: pd.Timestamp) -> pd.Timestamp:
    """As-of date for a balance snapshot: today if (year, month) is the real
    current calendar month (running, incomplete); otherwise that month's
    last day (a completed past month)."""
    if is_real_current_month(year, month, today):
        return today
    return pd.Timestamp(year, month, 1) + pd.DateOffset(months=1) - pd.Timedelta(days=1)


def compute_salesman_balance_asof(ar_clean: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Per-salesman AR balance as of cutoff (inclusive) — a true historical
    snapshot of the running balance, NOT the trickledown due-by-origin-month
    breakdown in salesman_due.py (that shows how much of *today's* balance
    originated in month X; this shows what the balance actually was on a
    specific past date).

    ar_clean: output of salesman_due.prep_ar_ledger (per-row running_balance
    per customer, xsp already bfilled/ffilled).

    For each customer, takes their last transaction at or before cutoff and
    reads its running_balance, then sums per salesman. Customers with no
    transactions yet at or before cutoff are absent (no balance existed for
    them at that point).
    """
    if ar_clean is None or ar_clean.empty:
        return pd.Series(dtype=float)

    d = ar_clean[ar_clean["xdate"] <= cutoff]
    if d.empty:
        return pd.Series(dtype=float)

    last = (
        d.sort_values(["xsub", "xdate", "xrow"], kind="mergesort")
         .groupby("xsub", as_index=False)
         .last()[["xsub", "xsp", "running_balance"]]
    )
    last["xsp"] = last["xsp"].astype(str)
    return last.groupby("xsp")["running_balance"].sum()


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
