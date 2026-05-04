"""
processing/buying_pattern.py

Computes six buying-pattern dimensions per customer from an already-built
customer-wise monthly net-sales pivot.  Pure pandas — no st.* calls.

Dimensions
----------
1. Recency          – months since last non-zero purchase
2. Frequency        – active months / total months in loaded range
3. Spend Tier       – Platinum / Gold / Silver / Bronze bands
4. Trend            – recent-3 vs prior-3 non-zero month averages
5. Seasonal Pattern – calendar months bought in ≥50% of loaded years
6. Product Breadth  – distinct item codes from item-level sales_df

Priority Score = recency(0.30) + spend_tier(0.25) + trend(0.25) + frequency(0.20)
"""

import calendar
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label_to_mi(label: str) -> int:
    """'Jan-25' → monotonic month index (year*12 + month)."""
    abbr = {v: k for k, v in enumerate(calendar.month_abbr) if k}
    mon, yr = label.split("-", 1)
    return (2000 + int(yr)) * 12 + abbr.get(mon, 0)


def _label_to_cal_month(label: str) -> int:
    abbr = {v: k for k, v in enumerate(calendar.month_abbr) if k}
    return abbr.get(label.split("-", 1)[0], 0)


def _label_to_year(label: str) -> int:
    return 2000 + int(label.split("-", 1)[1])


# ── Scoring functions ─────────────────────────────────────────────────────────

def _recency_score(months: int) -> int:
    if months <= 1: return 1
    if months == 2: return 2
    if months == 3: return 3
    if months <= 5: return 4
    return 5


def _frequency_score(ratio: float) -> int:
    if ratio >= 0.70: return 5
    if ratio >= 0.50: return 4
    if ratio >= 0.30: return 3
    if ratio >= 0.10: return 2
    return 1


def _tier_label(total: float) -> str:
    if total >= 150_000: return "🥇 Platinum"
    if total >= 50_000:  return "🥈 Gold"
    if total >= 10_000:  return "🥉 Silver"
    return "Bronze"


def _tier_score(tier: str) -> int:
    return {"🥇 Platinum": 5, "🥈 Gold": 4, "🥉 Silver": 3, "Bronze": 1}.get(tier, 1)


def _compute_trend(nonzero_vals: list) -> str:
    n = len(nonzero_vals)
    if n < 3:
        return "—"
    if n < 6:
        return "Insufficient data"
    recent_avg = sum(nonzero_vals[-3:]) / 3
    prior_avg  = sum(nonzero_vals[-6:-3]) / 3
    if prior_avg == 0:
        return "—"
    ratio = recent_avg / prior_avg
    if ratio > 1.15:  return "📈 Growing"
    if ratio < 0.85:  return "📉 Declining"
    return "➡ Flat"


def _trend_score(trend: str) -> int:
    return {"📈 Growing": 5, "➡ Flat": 3, "📉 Declining": 1,
            "Insufficient data": 2, "—": 2}.get(trend, 2)


# ── Main function ─────────────────────────────────────────────────────────────

def compute_buying_pattern(
    pivot_df:   pd.DataFrame,
    sales_df:   pd.DataFrame,
    id_cols:    list,
    month_cols: list,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    pivot_df   : customer-wise net-sales pivot (original column names, not renamed).
    sales_df   : item-level sales DataFrame — used only for product breadth.
    id_cols    : identity columns present in pivot_df
                 e.g. ['spid','spname','cusid','cusname','cusmobile','area']
    month_cols : ordered list of month-label columns in pivot_df
                 e.g. ['Jan-25','Feb-25', …]

    Returns
    -------
    DataFrame with id_cols + dimension columns + priority_score.
    Customers with zero total net sales are excluded.
    """
    if pivot_df.empty or not month_cols:
        return pd.DataFrame()

    now        = pd.Timestamp.today()
    current_mi = now.year * 12 + now.month
    total_months = len(month_cols)

    mi_map   = {c: _label_to_mi(c)       for c in month_cols}
    cal_map  = {c: _label_to_cal_month(c) for c in month_cols}
    yr_map   = {c: _label_to_year(c)      for c in month_cols}
    years_in_data = sorted({yr_map[c] for c in month_cols})
    n_years  = max(len(years_in_data), 1)

    records = []
    for _, row in pivot_df.iterrows():
        vals         = [float(row.get(c, 0) or 0) for c in month_cols]
        nonzero_vals = [v for v in vals if v > 0]
        nonzero_cols = [c for c, v in zip(month_cols, vals) if v > 0]
        total_sales  = sum(vals)

        if total_sales <= 0:
            continue  # excluded — covered by Report 4

        # 1. Recency
        if nonzero_cols:
            months_since = current_mi - mi_map[nonzero_cols[-1]]
        else:
            months_since = total_months
        rec_score = _recency_score(months_since)

        # 2. Frequency
        active_months = len(nonzero_vals)
        freq_ratio    = active_months / total_months if total_months > 0 else 0
        freq_score    = _frequency_score(freq_ratio)
        freq_display  = f"{active_months} / {total_months}"

        # 3. Spend Tier
        tier      = _tier_label(total_sales)
        t_score   = _tier_score(tier)

        # 4. Trend
        trend     = _compute_trend(nonzero_vals)
        tr_score  = _trend_score(trend)

        # 5. Seasonal Pattern
        cal_hits: dict = {}
        for col, v in zip(month_cols, vals):
            if v > 0:
                cal_hits.setdefault(cal_map[col], set()).add(yr_map[col])
        peak = [
            calendar.month_abbr[m]
            for m in range(1, 13)
            if len(cal_hits.get(m, set())) / n_years >= 0.5
        ]
        seasonal = ", ".join(peak) if peak else "No pattern"

        # 6. Priority Score (product breadth added via merge below)
        priority = round(
            rec_score * 0.30 +
            t_score   * 0.25 +
            tr_score  * 0.25 +
            freq_score * 0.20,
            1,
        )

        rec = {c: row[c] for c in id_cols if c in row.index}
        rec.update({
            "total_sales":       total_sales,
            "spend_tier":        tier,
            "_tier_score":       t_score,
            "months_since_last": months_since,
            "_rec_score":        rec_score,
            "active_months":     active_months,
            "_total_months":     total_months,
            "freq_display":      freq_display,
            "_freq_score":       freq_score,
            "trend":             trend,
            "_trend_score":      tr_score,
            "peak_months":       seasonal,
            "priority_score":    priority,
        })
        records.append(rec)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    # 6. Product Breadth — item-level join
    if (not sales_df.empty
            and "cusid" in sales_df.columns
            and "itemcode" in sales_df.columns):
        pb = (
            sales_df
            .groupby("cusid")["itemcode"]
            .nunique()
            .reset_index()
            .rename(columns={"itemcode": "product_count"})
        )
        result = result.merge(pb, on="cusid", how="left")
        result["product_count"] = result["product_count"].fillna(0).astype(int)
    else:
        result["product_count"] = 0

    return result


def priority_band(score: float) -> str:
    if score >= 4.0: return "🔴 This Week"
    if score >= 2.8: return "🟡 This Month"
    return "🟢 Monitor"
