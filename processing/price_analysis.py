# processing/price_analysis.py
# Pure pandas for Accounting Analysis -> "Price Analysis". No st.* calls.
#
# Two independent pieces, both rendered by views/price_analysis_view.py:
#   1. Total operating-expense change across a user-chosen timeline, built
#      from the same Level 0 income-statement frame Financial Statements
#      uses (so the numbers tie to its "Expense Changes" table).
#   2. A per-product price table (stock, wholesale tier, realized price).
#      Built so extra analysis columns can be appended after the base set.

from __future__ import annotations

import calendar
import math

import numpy as np
import pandas as pd

# 06xx = SG&A, 07xx = S&D. Mirrors views/financial.py::_render_expense_changes'
# own row filter -- keep the two in sync.
EXPENSE_CODE_PATTERN = r"^0[67]"

PRODUCT_TABLE_COLUMNS = [
    "Product Code", "Name", "Avg Monthly Sale", "Current Stock",
    "Std Price", "WH Qty", "WH Price", "Avg Price",
]


def period_columns(df: pd.DataFrame) -> list:
    """Period columns of a Level 0 frame in chronological order: (year, month)
    or (year, quarter) tuples, or plain int years. Meta columns (ac_code,
    ac_name, ac_lv*, ...) are skipped."""
    cols = []
    for c in df.columns:
        if isinstance(c, tuple) and len(c) == 2:
            try:
                if int(c[1]) >= 1:
                    int(c[0])
                    cols.append(c)
            except (TypeError, ValueError):
                continue
        elif isinstance(c, (int, np.integer)) and not isinstance(c, bool):
            cols.append(c)

    def _key(c):
        return (int(c[0]), int(c[1])) if isinstance(c, tuple) else (int(c), 0)

    return sorted(cols, key=_key)


def period_label(col, timeline: str, suffix: str = "") -> str:
    """`suffix` is appended to plain-year labels only (e.g. " (Jan–Sep)" for
    the Yearly - Custom Range timeline, where a year means a month window)."""
    if isinstance(col, tuple):
        year, part = int(col[0]), int(col[1])
        if timeline == "Quarterly":
            return f"Q{part} {year}"
        return f"{calendar.month_abbr[part]} {year}"
    return f"{int(col)}{suffix}"


def build_expense_totals(
    pl_df: pd.DataFrame, timeline: str, n_periods: int, label_suffix: str = ""
) -> pd.DataFrame:
    """Total 06xx+07xx expense for the last `n_periods` periods in `pl_df`,
    with the change vs the previous row. Level 0 sign: expense is positive,
    so a positive Change means costs went up. Empty frame if fewer than two
    periods are available."""
    out_cols = ["Period", "Total Expenses", "Change", "% Change"]
    cols = period_columns(pl_df)
    if len(cols) < 2 or "ac_code" not in pl_df.columns:
        return pd.DataFrame(columns=out_cols)

    cols = cols[-max(int(n_periods), 2):]
    mask = pl_df["ac_code"].astype(str).str.match(EXPENSE_CODE_PATTERN)
    exp = pl_df.loc[mask, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if exp.empty:
        return pd.DataFrame(columns=out_cols)

    out = pd.DataFrame({
        "Period": [period_label(c, timeline, label_suffix) for c in cols],
        "Total Expenses": exp.sum(axis=0).to_numpy(dtype=float),
    })
    out["Change"] = out["Total Expenses"].diff()
    prev = out["Total Expenses"].shift().abs().replace(0, np.nan)
    out["% Change"] = out["Change"] / prev * 100
    return out


def summarize_expense_change(totals: pd.DataFrame) -> dict | None:
    """Net change from the first to the last period of `totals` (the sum of
    every period-over-period change in between)."""
    if totals is None or len(totals) < 2:
        return None
    first, last = totals.iloc[0], totals.iloc[-1]
    change = float(last["Total Expenses"] - first["Total Expenses"])
    base = abs(float(first["Total Expenses"]))
    return {
        "first_label": first["Period"],
        "last_label": last["Period"],
        "first_total": float(first["Total Expenses"]),
        "last_total": float(last["Total Expenses"]),
        "change": change,
        "pct": (change / base * 100) if base else None,
    }


def build_product_price_table(
    items_df: pd.DataFrame,
    wh_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per catalog item.

    items_df : final_items_view (item_id, item_name, stock, avg_monthly_sales,
               std_price) -- the stock view's own 3-month trailing average.
    wh_df    : opspprc (item_id, wh_qty, wh_price) -- lowest wholesale tier;
               items with no tier get blanks, not zeros.
    price_df : item_price_window (itemcode, units, amount) -- Avg Price is
               amount / units over that window; blank when nothing sold.
    """
    items = items_df[["item_id", "item_name", "avg_monthly_sales", "stock", "std_price"]].copy()
    for c in ("avg_monthly_sales", "stock", "std_price"):
        items[c] = pd.to_numeric(items[c], errors="coerce")
    items["item_id"] = items["item_id"].astype(str)

    if wh_df is not None and not wh_df.empty:
        wh = wh_df[["item_id", "wh_qty", "wh_price"]].copy()
        wh["item_id"] = wh["item_id"].astype(str)
        for c in ("wh_qty", "wh_price"):
            wh[c] = pd.to_numeric(wh[c], errors="coerce")
        wh = wh.drop_duplicates("item_id")
    else:
        wh = pd.DataFrame(columns=["item_id", "wh_qty", "wh_price"])

    if price_df is not None and not price_df.empty:
        price = price_df[["itemcode", "units", "amount"]].copy()
        price["item_id"] = price["itemcode"].astype(str)
        units = pd.to_numeric(price["units"], errors="coerce")
        amount = pd.to_numeric(price["amount"], errors="coerce")
        price["avg_price"] = (amount / units.where(units > 0)).astype(float)
        price = price[["item_id", "avg_price"]].drop_duplicates("item_id")
    else:
        price = pd.DataFrame(columns=["item_id", "avg_price"])

    merged = items.merge(wh, on="item_id", how="left").merge(price, on="item_id", how="left")
    merged = merged.rename(columns={
        "item_id": "Product Code",
        "item_name": "Name",
        "avg_monthly_sales": "Avg Monthly Sale",
        "stock": "Current Stock",
        "std_price": "Std Price",
        "wh_qty": "WH Qty",
        "wh_price": "WH Price",
        "avg_price": "Avg Price",
    })
    return (
        merged[PRODUCT_TABLE_COLUMNS]
        .sort_values("Avg Monthly Sale", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


# ── Price-increase sizing ────────────────────────────────────────────────────
# Model: raise every selected item's price by the same fraction x. A fraction
# h = elasticity * x of units is lost. Lost units cost margin, not revenue
# (their COGS is avoided), so with monthly revenue R and gross margin M:
#     new margin = (1 - h) * (M + R*x)
#     monthly gain G(x) = new margin - M = x*(R - e*M) - e*R*x^2
# G rises, peaks, then falls (volume loss eventually outruns the price gain),
# so a target can be unreachable -- required_price_rise reports that.

def months_in_period(timeline: str, start_month: int = 1, end_month: int = 12) -> int:
    """Months one period of the chosen timeline spans (the default recovery
    horizon): 1 Monthly, 3 Quarterly, 12 Full-Year-vs-YTD, start..end for
    Custom Range."""
    if timeline == "Monthly":
        return 1
    if timeline == "Quarterly":
        return 3
    if timeline == "Yearly - Custom Range":
        return max(int(end_month) - int(start_month) + 1, 1)
    return 12


def build_item_base(price_df: pd.DataFrame, window_months: int) -> pd.DataFrame:
    """Per-item monthly revenue / gross margin for items that sold in the
    window (item_price_window rows), highest revenue first, with the running
    share of total revenue. Columns: itemcode, rev_m, mar_m, cum_share."""
    cols = ["itemcode", "rev_m", "mar_m", "cum_share"]
    if price_df is None or price_df.empty:
        return pd.DataFrame(columns=cols)
    d = price_df[["itemcode", "units", "amount", "cost"]].copy()
    for c in ("units", "amount", "cost"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    d = d[(d["units"] > 0) & (d["amount"] > 0)]
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["rev_m"] = d["amount"] / window_months
    d["mar_m"] = (d["amount"] - d["cost"]) / window_months
    d = d.sort_values("rev_m", ascending=False).reset_index(drop=True)
    d["cum_share"] = d["rev_m"].cumsum() / d["rev_m"].sum()
    return d[cols]


def select_top_coverage(base: pd.DataFrame, share: float) -> pd.DataFrame:
    """The fewest top-revenue items whose revenue reaches `share` (0-1) of the total."""
    if base.empty:
        return base
    n = min(int((base["cum_share"] < share - 1e-12).sum()) + 1, len(base))
    return base.head(n)


def required_price_rise(gap: float, revenue: float, margin: float, elasticity: float) -> dict:
    """Smallest uniform price rise x (fraction) whose monthly gain G(x)
    reaches `gap`. Returns {reachable, x, volume_loss, max_gain}; when the
    gap is out of reach, max_gain is the most any uniform rise can recover."""
    if gap <= 0:
        return {"reachable": True, "x": 0.0, "volume_loss": 0.0, "max_gain": None}
    if revenue <= 0:
        return {"reachable": False, "x": None, "volume_loss": None, "max_gain": 0.0}
    if elasticity <= 0:
        x = gap / revenue
        return {"reachable": True, "x": x, "volume_loss": 0.0, "max_gain": None}
    a = revenue - elasticity * margin
    if a <= 0:
        return {"reachable": False, "x": None, "volume_loss": None, "max_gain": 0.0}
    disc = a * a - 4.0 * elasticity * revenue * gap
    if disc < 0:
        return {"reachable": False, "x": None, "volume_loss": None,
                "max_gain": a * a / (4.0 * elasticity * revenue)}
    x = (a - math.sqrt(disc)) / (2.0 * elasticity * revenue)
    return {"reachable": True, "x": x, "volume_loss": elasticity * x, "max_gain": None}


def coverage_scenarios(
    base: pd.DataFrame, gap: float, elasticity: float,
    shares: tuple = (1.0, 0.8, 0.6, 0.4, 0.2),
) -> pd.DataFrame:
    """Required uniform rise if only the top-revenue items covering each
    share of revenue are raised -- fewer items need a bigger % but lose less
    margin to volume."""
    rows = []
    for sh in shares:
        sub = select_top_coverage(base, sh)
        R, M = float(sub["rev_m"].sum()), float(sub["mar_m"].sum())
        res = required_price_rise(gap, R, M, elasticity)
        rows.append({
            "Revenue Covered": f"{sh:.0%}",
            "Items": len(sub),
            "Monthly Revenue": R,
            "Monthly Margin": M,
            "Price Rise Needed": res["x"] * 100 if res["reachable"] else np.nan,
            "Volume Loss": res["volume_loss"] * 100 if res["reachable"] else np.nan,
            "Note": "" if res["reachable"] else f"Not reachable (max {res['max_gain']:,.0f}/mo)",
        })
    return pd.DataFrame(rows)
