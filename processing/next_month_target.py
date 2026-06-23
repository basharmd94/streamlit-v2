# processing/next_month_target.py
# Pure-pandas helpers for the "Next Month Target" estimate in Target Management.

import numpy as np
import pandas as pd


def trailing_12mo_window(today: pd.Timestamp) -> tuple:
    """Last 12 *completed* calendar months, excluding the current (partial) month."""
    cur_month_start = pd.Timestamp(today.year, today.month, 1)
    end = cur_month_start - pd.Timedelta(days=1)
    start = cur_month_start - pd.DateOffset(months=12)
    return start, end


def month_grid(start: pd.Timestamp, end: pd.Timestamp) -> list:
    months = []
    cur = pd.Timestamp(start.year, start.month, 1)
    while cur <= end:
        months.append((cur.year, cur.month))
        cur = cur + pd.DateOffset(months=1)
    return months


def incoming_proration(
    entry_date: pd.Timestamp, month_start: pd.Timestamp, month_end: pd.Timestamp
) -> tuple:
    """Fraction of the target month an incoming shipment is actually sellable for.

    Arrives at/before the month starts -> 1.0 (full month). Arrives after the
    month ends -> 0.0 (should be excluded by the caller before this matters).
    Arrives mid-month -> the inclusive day-count from entry_date to month_end,
    divided by the month's total day count.

    Returns (fraction, days_remaining, days_in_month).
    """
    days_in_month = month_end.day
    if entry_date <= month_start:
        return 1.0, days_in_month, days_in_month
    if entry_date > month_end:
        return 0.0, 0, days_in_month
    days_remaining = (month_end - entry_date).days + 1
    fraction = max(0.0, min(1.0, days_remaining / days_in_month))
    return fraction, days_remaining, days_in_month


def _prep_sales(sales_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if sales_df is None or sales_df.empty or "itemcode" not in sales_df.columns:
        return pd.DataFrame(columns=["itemcode", "year", "month", "qty", "amt"])
    d = sales_df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[(d["date"] >= start) & (d["date"] <= end)]
    if d.empty:
        return pd.DataFrame(columns=["itemcode", "year", "month", "qty", "amt"])
    d["itemcode"] = d["itemcode"].astype(str)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["qty"] = pd.to_numeric(d["quantity"], errors="coerce").fillna(0.0)
    altsales = pd.to_numeric(d["altsales"], errors="coerce").fillna(0.0)
    proddiscount = pd.to_numeric(d.get("proddiscount", 0.0), errors="coerce").fillna(0.0)
    d["amt"] = altsales - proddiscount
    return d.groupby(["itemcode", "year", "month"], as_index=False)[["qty", "amt"]].sum()


def _prep_returns(returns_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if returns_df is None or returns_df.empty or "itemcode" not in returns_df.columns:
        return pd.DataFrame(columns=["itemcode", "year", "month", "qty", "amt"])
    d = returns_df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[(d["date"] >= start) & (d["date"] <= end)]
    if d.empty:
        return pd.DataFrame(columns=["itemcode", "year", "month", "qty", "amt"])
    d["itemcode"] = d["itemcode"].astype(str)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["qty"] = pd.to_numeric(d["returnqty"], errors="coerce").fillna(0.0)
    d["amt"] = pd.to_numeric(d["treturnamt"], errors="coerce").fillna(0.0)
    return d.groupby(["itemcode", "year", "month"], as_index=False)[["qty", "amt"]].sum()


def compute_item_monthly_net(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """One row per itemcode x (year, month) across the trailing window, net of returns.
    Months with no sales for an item are zero-filled here; compute_item_low_high()
    ignores those zero months when picking the item's 'low' figure.
    """
    months = month_grid(start, end)
    s = _prep_sales(sales_df, start, end)
    r = _prep_returns(returns_df, start, end)

    items = sorted(set(s["itemcode"]) | set(r["itemcode"]))
    if not items:
        return pd.DataFrame(columns=["itemcode", "year", "month", "net_qty", "net_amt"])

    grid = pd.DataFrame(
        [(it, y, m) for it in items for (y, m) in months],
        columns=["itemcode", "year", "month"],
    )
    grid = grid.merge(s, on=["itemcode", "year", "month"], how="left")
    grid = grid.rename(columns={"qty": "sales_qty", "amt": "sales_amt"})
    grid = grid.merge(r, on=["itemcode", "year", "month"], how="left")
    grid = grid.rename(columns={"qty": "return_qty", "amt": "return_amt"})

    for c in ["sales_qty", "sales_amt", "return_qty", "return_amt"]:
        grid[c] = grid[c].fillna(0.0)

    grid["net_qty"] = grid["sales_qty"] - grid["return_qty"]
    grid["net_amt"] = grid["sales_amt"] - grid["return_amt"]
    return grid[["itemcode", "year", "month", "net_qty", "net_amt"]]


def _item_stats(g: pd.DataFrame) -> pd.Series:
    """low/median/avg come from the item's active (non-zero net_qty) months
    only, so quiet months don't drag the floor, midpoint, or average down
    toward zero. The matching net_amt for those same active months gives the
    median amt — paired to the same months as median_qty, not independently
    filtered.
    """
    active = g[g["net_qty"] > 0]
    if active.empty:
        low_qty = median_qty = avg_qty = median_amt = 0.0
    else:
        low_qty = float(active["net_qty"].min())
        median_qty = float(active["net_qty"].median())
        avg_qty = float(active["net_qty"].mean())
        median_amt = float(active["net_amt"].median())
    high_qty = float(max(g["net_qty"].max(), 0.0))
    return pd.Series({
        "low_qty": low_qty,
        "median_qty": median_qty,
        "high_qty": high_qty,
        "avg_qty": avg_qty,
        "median_amt": median_amt,
        "total_net_qty": float(g["net_qty"].sum()),
        "total_net_amt": float(g["net_amt"].sum()),
        "months_with_sales": int((g["net_qty"] > 0).sum()),
    })


def compute_item_low_high(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Per item: lowest non-zero / median (non-zero) / highest net qty month in
    the window, the average *active*-month net qty (used to project current
    stock forward to next month — see project_current_stock), the matching
    median net amt, and an overall average selling price (total net amt /
    total net qty, used to turn a capped quantity estimate into a BDT amount
    estimate).

    Months with zero net sales are ignored for the "low", "median", and
    "avg_qty" figures — a quiet month doesn't count as the item's floor,
    typical month, or average, only its active months do. avg_price is a
    ratio of 12-month totals, so zero-sale months don't skew it either way
    (they add 0 to both the numerator and denominator).
    """
    cols = ["itemcode", "low_qty", "median_qty", "high_qty", "avg_qty", "avg_price",
            "median_amt", "total_net_qty", "total_net_amt", "months_with_sales"]
    if monthly_df is None or monthly_df.empty:
        return pd.DataFrame(columns=cols)

    out = monthly_df.groupby("itemcode")[["net_qty", "net_amt"]].apply(_item_stats).reset_index()
    out["avg_price"] = np.where(
        out["total_net_qty"] > 0, out["total_net_amt"] / out["total_net_qty"], 0.0
    )
    for c in ["low_qty", "median_qty", "high_qty", "median_amt"]:
        out[c] = out[c].round(0)
    out["avg_qty"] = out["avg_qty"].round(2)  # keep precision — feeds the daily-rate projection
    return out[cols]


def get_open_shipments(purchase_df: pd.DataFrame) -> pd.DataFrame:
    """Distinct open (not-yet-received) shipments with item/qty counts, for the
    picker. Grouped by (zid, shipmentname) — 100001 (HMBR import) and 100009
    (Gulshan Packaging) keep separate PO books, so what's physically one
    incoming shipment can show up as two distinct (zid, shipmentname) rows
    here; the caller lets the user select both.
    """
    cols = ["zid", "shipmentname", "n_items", "total_qty"]
    if purchase_df is None or purchase_df.empty or "status" not in purchase_df.columns:
        return pd.DataFrame(columns=cols)

    d = purchase_df[purchase_df["status"] == "1-Open"].copy()
    d = d[d["shipmentname"].notna() & (d["shipmentname"].astype(str).str.strip() != "")]
    if d.empty:
        return pd.DataFrame(columns=cols)

    d["zid"] = d["zid"].astype(str)
    d["quantity"] = pd.to_numeric(d["quantity"], errors="coerce").fillna(0.0)
    out = d.groupby(["zid", "shipmentname"], as_index=False).agg(
        n_items=("itemcode", "nunique"),
        total_qty=("quantity", "sum"),
    )
    return out.sort_values(["shipmentname", "zid"]).reset_index(drop=True)


def get_shipment_items(purchase_df: pd.DataFrame, selections: list) -> pd.DataFrame:
    """Item-level incoming quantity for one or more selected (zid, shipmentname)
    shipments. Itemcodes coming from a 100009 selection are already resolved to
    the matching 100001 packaged-item code at the query layer (caitem.xdrawing,
    same mechanism the Current Stock tab uses) — so a 100001-side and a
    100009-side selection for the same physical shipment naturally sum onto
    the same finished-item rows.

    selections: list of (zid, shipmentname) tuples.
    """
    cols = ["itemcode", "itemname", "incoming_qty"]
    if purchase_df is None or purchase_df.empty or not selections:
        return pd.DataFrame(columns=cols)

    d = purchase_df.copy()
    d["zid"] = d["zid"].astype(str)
    mask = pd.Series(False, index=d.index)
    for sel_zid, sel_name in selections:
        mask |= (d["zid"] == str(sel_zid)) & (d["shipmentname"] == sel_name)
    d = d[mask & (d["status"] == "1-Open")]
    if d.empty:
        return pd.DataFrame(columns=cols)

    d["itemcode"] = d["itemcode"].astype(str)
    d["quantity"] = pd.to_numeric(d["quantity"], errors="coerce").fillna(0.0)
    out = d.groupby("itemcode", as_index=False).agg(
        incoming_qty=("quantity", "sum"),
        itemname=("itemname", "first"),
    )
    return out[cols]


def days_remaining_in_month(today: pd.Timestamp) -> tuple:
    """(days_remaining, days_in_month) for today's calendar month, inclusive of today."""
    days_in_month = (
        pd.Timestamp(today.year, today.month, 1) + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    ).day
    days_remaining = days_in_month - today.day + 1
    return days_remaining, days_in_month


def build_next_month_estimate(
    stock_df: pd.DataFrame,
    low_high_df: pd.DataFrame,
    shipment_items_df: pd.DataFrame = None,
    incoming_fraction: float = 1.0,
    today: pd.Timestamp = None,
) -> pd.DataFrame:
    """Combine current stock (+ optional incoming shipment) with each item's
    12-month low/high net-qty band, capped at what's actually available to
    sell. Items with zero available stock (no current stock and nothing
    incoming) are dropped — they can't contribute to next month's target.

    incoming_fraction scales the shipment's quantity down to the portion of
    the target month it's actually available for — a shipment landing mid-month
    can only be sold for the days remaining, not the full month. 1.0 = arrives
    at/before the start of the target month (fully available all month).

    today, if given, projects current_stock forward to the start of next month:
    "current stock" is a snapshot as of today, but some of it will sell between
    today and month-end before next month even begins. That depletion is
    estimated as (avg_qty / days_in_this_month) * days_remaining_in_this_month,
    using the trailing-12-month average *active*-month net qty (avg_qty already
    excludes zero-sale months) as the daily rate. Without today, current_stock
    is used as-is.
    """
    if stock_df is None or stock_df.empty:
        stock = pd.DataFrame(columns=["itemcode", "itemname", "itemgroup", "current_stock"])
    else:
        stock = stock_df.rename(columns={
            "item_id": "itemcode", "item_name": "itemname",
            "item_group": "itemgroup", "stock": "current_stock",
        })[["itemcode", "itemname", "itemgroup", "current_stock"]].copy()
        stock["itemcode"] = stock["itemcode"].astype(str)
        stock["current_stock"] = pd.to_numeric(stock["current_stock"], errors="coerce").fillna(0.0)

    if low_high_df is None or low_high_df.empty:
        lh = pd.DataFrame(columns=["itemcode", "low_qty", "median_qty", "high_qty", "avg_qty", "avg_price"])
    else:
        lh = low_high_df[["itemcode", "low_qty", "median_qty", "high_qty", "avg_qty", "avg_price"]].copy()
        lh["itemcode"] = lh["itemcode"].astype(str)

    df = stock.merge(lh, on="itemcode", how="outer")

    if shipment_items_df is not None and not shipment_items_df.empty:
        sd = shipment_items_df.copy()
        sd["itemcode"] = sd["itemcode"].astype(str)
        sd["incoming_qty"] = pd.to_numeric(sd["incoming_qty"], errors="coerce").fillna(0.0) * incoming_fraction
        df = df.merge(sd, on="itemcode", how="outer", suffixes=("", "_ship"))
        if "itemname_ship" in df.columns:
            df["itemname"] = df["itemname"].fillna(df.pop("itemname_ship"))
    else:
        df["incoming_qty"] = 0.0

    for c in ["current_stock", "low_qty", "median_qty", "high_qty", "avg_qty", "avg_price", "incoming_qty"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0.0)

    if today is not None:
        days_remaining, days_in_month = days_remaining_in_month(today)
        daily_rate = df["avg_qty"] / days_in_month if days_in_month > 0 else 0.0
        projected_sold = daily_rate * days_remaining
        df["projected_stock"] = (df["current_stock"] - projected_sold).clip(lower=0.0)
    else:
        df["projected_stock"] = df["current_stock"]
    df["projected_stock"] = df["projected_stock"].round(0)

    df["available_qty"] = df["projected_stock"] + df["incoming_qty"]
    df = df[df["available_qty"] > 0].copy()
    if df.empty:
        return df

    df["est_low_qty"] = df[["low_qty", "available_qty"]].min(axis=1)
    df["est_median_qty"] = df[["median_qty", "available_qty"]].min(axis=1)
    df["est_high_qty"] = df[["high_qty", "available_qty"]].min(axis=1)
    # low <= median <= high is already guaranteed pre-cap; re-clamp defensively
    # since each was capped independently against the same available_qty.
    df["est_low_qty"] = df[["est_low_qty", "est_high_qty"]].min(axis=1)
    df["est_median_qty"] = df[["est_median_qty", "est_high_qty"]].min(axis=1)
    df["est_median_qty"] = df[["est_median_qty", "est_low_qty"]].max(axis=1)

    df["est_low_amt"] = (df["est_low_qty"] * df["avg_price"]).round(0)
    df["est_median_amt"] = (df["est_median_qty"] * df["avg_price"]).round(0)
    df["est_high_amt"] = (df["est_high_qty"] * df["avg_price"]).round(0)
    df["est_low_qty"] = df["est_low_qty"].round(0)
    df["est_median_qty"] = df["est_median_qty"].round(0)
    df["est_high_qty"] = df["est_high_qty"].round(0)

    return df.sort_values("est_high_amt", ascending=False).reset_index(drop=True)


def extract_item_names(sales_df: pd.DataFrame) -> pd.DataFrame:
    """itemcode -> itemname/itemgroup lookup, pulled straight from sales history."""
    cols = ["itemcode", "itemname", "itemgroup"]
    if sales_df is None or sales_df.empty or "itemcode" not in sales_df.columns:
        return pd.DataFrame(columns=cols)
    d = sales_df.copy()
    d["itemcode"] = d["itemcode"].astype(str)
    for c in ["itemname", "itemgroup"]:
        if c not in d.columns:
            d[c] = ""
    return d[cols].drop_duplicates(subset="itemcode", keep="first").reset_index(drop=True)


def build_sales_only_estimate(
    low_high_df: pd.DataFrame, item_names_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Pure sales-history estimate for businesses with no stock/shipment
    tracking in this tool (e.g. GI, Zepto): each item's low/median/high net
    qty & amt, not capped by stock on hand. Items with no sales history at
    all in the trailing window are dropped.
    """
    cols = ["itemcode", "itemname", "itemgroup", "low_qty", "median_qty", "high_qty",
            "avg_price", "est_low_qty", "est_median_qty", "est_high_qty",
            "est_low_amt", "est_median_amt", "est_high_amt"]
    if low_high_df is None or low_high_df.empty:
        return pd.DataFrame(columns=cols)

    df = low_high_df.copy()
    df["itemcode"] = df["itemcode"].astype(str)
    df = df[df["high_qty"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    if item_names_df is not None and not item_names_df.empty:
        df = df.merge(item_names_df, on="itemcode", how="left")
    if "itemname" not in df.columns:
        df["itemname"] = ""
    if "itemgroup" not in df.columns:
        df["itemgroup"] = ""
    df["itemname"] = df["itemname"].fillna("")
    df["itemgroup"] = df["itemgroup"].fillna("")

    df["est_low_qty"] = df["low_qty"]
    df["est_median_qty"] = df["median_qty"]
    df["est_high_qty"] = df["high_qty"]
    df["est_low_amt"] = (df["low_qty"] * df["avg_price"]).round(0)
    df["est_median_amt"] = (df["median_qty"] * df["avg_price"]).round(0)
    df["est_high_amt"] = (df["high_qty"] * df["avg_price"]).round(0)

    return df.sort_values("est_high_amt", ascending=False).reset_index(drop=True)[cols]


def compute_salesman_area_monthly_net(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """One row per (spid, area) x (year, month) across the trailing window —
    net sales amount (after returns), summed across every product that
    salesman sold in that area. Missing months are zero-filled, mirroring
    compute_item_monthly_net.
    """
    months = month_grid(start, end)
    base_cols = ["spid", "area", "year", "month", "amt"]

    def _prep(df: pd.DataFrame, amt_fn) -> pd.DataFrame:
        if df is None or df.empty or "spid" not in df.columns:
            return pd.DataFrame(columns=base_cols)
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d[(d["date"] >= start) & (d["date"] <= end)]
        if d.empty:
            return pd.DataFrame(columns=base_cols)
        d["spid"] = d["spid"].astype(str)
        d["area"] = d["area"].fillna("Unknown") if "area" in d.columns else "Unknown"
        d["year"] = d["date"].dt.year
        d["month"] = d["date"].dt.month
        d["amt"] = amt_fn(d)
        return d.groupby(["spid", "area", "year", "month"], as_index=False)["amt"].sum()

    def _sales_amt(d: pd.DataFrame) -> pd.Series:
        altsales = pd.to_numeric(d["altsales"], errors="coerce").fillna(0.0)
        proddiscount = pd.to_numeric(d.get("proddiscount", 0.0), errors="coerce").fillna(0.0)
        return altsales - proddiscount

    def _return_amt(d: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(d["treturnamt"], errors="coerce").fillna(0.0)

    s = _prep(sales_df, _sales_amt)
    r = _prep(returns_df, _return_amt)

    keys = sorted(set(zip(s["spid"], s["area"])) | set(zip(r["spid"], r["area"])))
    if not keys:
        return pd.DataFrame(columns=["spid", "area", "year", "month", "net_amt"])

    grid = pd.DataFrame(
        [(sp, ar, y, m) for (sp, ar) in keys for (y, m) in months],
        columns=["spid", "area", "year", "month"],
    )
    grid = grid.merge(s, on=["spid", "area", "year", "month"], how="left").rename(columns={"amt": "sales_amt"})
    grid = grid.merge(r, on=["spid", "area", "year", "month"], how="left").rename(columns={"amt": "return_amt"})
    for c in ["sales_amt", "return_amt"]:
        grid[c] = grid[c].fillna(0.0)
    grid["net_amt"] = grid["sales_amt"] - grid["return_amt"]
    return grid[["spid", "area", "year", "month", "net_amt"]]


def _amt_stats(g: pd.DataFrame) -> pd.Series:
    """low/median from active (non-zero net_amt) months only, same rule as items."""
    active = g[g["net_amt"] > 0]
    if active.empty:
        low_amt = median_amt = 0.0
    else:
        low_amt = float(active["net_amt"].min())
        median_amt = float(active["net_amt"].median())
    high_amt = float(max(g["net_amt"].max(), 0.0))
    return pd.Series({"low_amt": low_amt, "median_amt": median_amt, "high_amt": high_amt})


def compute_salesman_area_low_high(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Per (spid, area): lowest non-zero / median (non-zero) / highest monthly
    net sales amount, accumulated across every product that salesman sold in
    that area over the trailing window.
    """
    cols = ["spid", "area", "low_amt", "median_amt", "high_amt"]
    if monthly_df is None or monthly_df.empty:
        return pd.DataFrame(columns=cols)
    out = monthly_df.groupby(["spid", "area"])[["net_amt"]].apply(_amt_stats).reset_index()
    for c in ["low_amt", "median_amt", "high_amt"]:
        out[c] = out[c].round(0)
    return out[cols]


def extract_salesman_names(sales_df: pd.DataFrame) -> pd.DataFrame:
    """spid -> spname lookup, pulled straight from sales history."""
    cols = ["spid", "spname"]
    if sales_df is None or sales_df.empty or "spid" not in sales_df.columns:
        return pd.DataFrame(columns=cols)
    d = sales_df.copy()
    d["spid"] = d["spid"].astype(str)
    if "spname" not in d.columns:
        d["spname"] = ""
    return d[cols].drop_duplicates(subset="spid", keep="first").reset_index(drop=True)


# ── Stock-aware salesman allocation (HMBR/100009 only — businesses where a ──
# ── shared, limited stock pool means salesmen can't all be credited their ──
# ── own uncapped historical numbers at the same time) ───────────────────────

def compute_item_spid_monthly_net(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """One row per (itemcode, spid) x (year, month) across the trailing window,
    net of returns. Same shape/zero-fill rule as compute_item_monthly_net, but
    split by salesman too — the basis for sharing one item's available stock
    fairly across every salesman who sells it.
    """
    months = month_grid(start, end)
    base_cols = ["itemcode", "spid", "year", "month", "qty", "amt"]

    def _prep(df: pd.DataFrame, qty_col: str, amt_fn) -> pd.DataFrame:
        if df is None or df.empty or "itemcode" not in df.columns or "spid" not in df.columns:
            return pd.DataFrame(columns=base_cols)
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d[(d["date"] >= start) & (d["date"] <= end)]
        if d.empty:
            return pd.DataFrame(columns=base_cols)
        d["itemcode"] = d["itemcode"].astype(str)
        d["spid"] = d["spid"].astype(str)
        d["year"] = d["date"].dt.year
        d["month"] = d["date"].dt.month
        d["qty"] = pd.to_numeric(d[qty_col], errors="coerce").fillna(0.0)
        d["amt"] = amt_fn(d)
        return d.groupby(["itemcode", "spid", "year", "month"], as_index=False)[["qty", "amt"]].sum()

    def _sales_amt(d: pd.DataFrame) -> pd.Series:
        altsales = pd.to_numeric(d["altsales"], errors="coerce").fillna(0.0)
        proddiscount = pd.to_numeric(d.get("proddiscount", 0.0), errors="coerce").fillna(0.0)
        return altsales - proddiscount

    def _return_amt(d: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(d["treturnamt"], errors="coerce").fillna(0.0)

    s = _prep(sales_df, "quantity", _sales_amt)
    r = _prep(returns_df, "returnqty", _return_amt)

    keys = sorted(set(zip(s["itemcode"], s["spid"])) | set(zip(r["itemcode"], r["spid"])))
    if not keys:
        return pd.DataFrame(columns=["itemcode", "spid", "year", "month", "net_qty", "net_amt"])

    grid = pd.DataFrame(
        [(it, sp, y, m) for (it, sp) in keys for (y, m) in months],
        columns=["itemcode", "spid", "year", "month"],
    )
    grid = grid.merge(s, on=["itemcode", "spid", "year", "month"], how="left").rename(
        columns={"qty": "sales_qty", "amt": "sales_amt"}
    )
    grid = grid.merge(r, on=["itemcode", "spid", "year", "month"], how="left").rename(
        columns={"qty": "return_qty", "amt": "return_amt"}
    )
    for c in ["sales_qty", "sales_amt", "return_qty", "return_amt"]:
        grid[c] = grid[c].fillna(0.0)
    grid["net_qty"] = grid["sales_qty"] - grid["return_qty"]
    grid["net_amt"] = grid["sales_amt"] - grid["return_amt"]
    return grid[["itemcode", "spid", "year", "month", "net_qty", "net_amt"]]


def compute_item_spid_low_high(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Per (itemcode, spid): this salesman's own lowest non-zero / median
    (non-zero) / highest net qty month for that item, and their own average
    selling price for it. Same active-month rule as compute_item_low_high —
    this is each salesman's individual demand band, before any stock cap.
    """
    cols = ["itemcode", "spid", "low_qty", "median_qty", "high_qty", "avg_price", "total_net_qty"]
    if monthly_df is None or monthly_df.empty:
        return pd.DataFrame(columns=cols)

    out = monthly_df.groupby(["itemcode", "spid"])[["net_qty", "net_amt"]].apply(_item_stats).reset_index()
    out["avg_price"] = np.where(
        out["total_net_qty"] > 0, out["total_net_amt"] / out["total_net_qty"], 0.0
    )
    for c in ["low_qty", "median_qty", "high_qty"]:
        out[c] = out[c].round(0)
    return out[cols]


def allocate_item_estimate_by_salesman(
    item_spid_low_high: pd.DataFrame, item_estimate_df: pd.DataFrame
) -> pd.DataFrame:
    """Splits each item's already stock-capped low/median/high estimate (the
    same Est. Low/Median/High figures shown in the main product table — built
    from build_next_month_estimate, already bounded by available stock) across
    the salesmen who sell it, by their historical share of that item's net qty.

    Deliberately a share-of-the-final-number split, not each salesman's own
    independently computed low/median/high: independent per-salesman stats
    (each one's own best/worst/typical month) don't add back up to the item's
    combined-series low/median/high, because different salesmen's best (or
    worst) months rarely land on the same calendar month — summing them can
    massively overshoot the item's actual stock-capped total. Splitting the
    item's final number by share instead guarantees the allocated figures,
    summed across every salesman, reconstruct that item's capped estimate
    exactly — never more, so salesmen can't collectively target stock that
    isn't there.
    """
    cols = ["itemcode", "spid", "low_qty", "median_qty", "high_qty", "avg_price",
            "est_low_qty", "est_median_qty", "est_high_qty",
            "est_low_amt", "est_median_amt", "est_high_amt"]
    est_cols = ["est_low_qty", "est_median_qty", "est_high_qty",
                "est_low_amt", "est_median_amt", "est_high_amt"]
    if item_spid_low_high is None or item_spid_low_high.empty:
        return pd.DataFrame(columns=cols)

    d = item_spid_low_high.copy()
    d["itemcode"] = d["itemcode"].astype(str)
    d["total_net_qty"] = d["total_net_qty"].clip(lower=0.0)
    item_totals = d.groupby("itemcode")["total_net_qty"].transform("sum")
    d["share"] = np.where(item_totals > 0, d["total_net_qty"] / item_totals, 0.0)

    if item_estimate_df is None or item_estimate_df.empty:
        for c in est_cols:
            d[c] = 0.0
        return d[cols]

    est = item_estimate_df[["itemcode"] + est_cols].copy()
    est["itemcode"] = est["itemcode"].astype(str)
    d = d.merge(est, on="itemcode", how="left")
    for c in est_cols:
        d[c] = d[c].fillna(0.0) * d["share"]

    return d[cols]


def compute_item_spid_area_totals(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Per (itemcode, spid, area): total net qty over the trailing window
    (summed across months, not month-by-month) — used to split a salesman's
    already stock-capped allocation for an item across the areas they sell it
    in, for the Salesman x Area table.
    """
    cols = ["itemcode", "spid", "area", "total_net_qty"]

    def _prep(df: pd.DataFrame, qty_col: str, sign: float) -> pd.DataFrame:
        if df is None or df.empty or "itemcode" not in df.columns or "spid" not in df.columns:
            return pd.DataFrame(columns=["itemcode", "spid", "area", "net_qty"])
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d[(d["date"] >= start) & (d["date"] <= end)]
        if d.empty:
            return pd.DataFrame(columns=["itemcode", "spid", "area", "net_qty"])
        d["itemcode"] = d["itemcode"].astype(str)
        d["spid"] = d["spid"].astype(str)
        d["area"] = d["area"].fillna("Unknown") if "area" in d.columns else "Unknown"
        d["net_qty"] = sign * pd.to_numeric(d[qty_col], errors="coerce").fillna(0.0)
        return d.groupby(["itemcode", "spid", "area"], as_index=False)["net_qty"].sum()

    s = _prep(sales_df, "quantity", 1.0)
    r = _prep(returns_df, "returnqty", -1.0)
    combined = pd.concat([s, r], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=cols)
    out = combined.groupby(["itemcode", "spid", "area"], as_index=False)["net_qty"].sum()
    return out.rename(columns={"net_qty": "total_net_qty"})[cols]


def allocate_salesman_amt_by_area(
    salesman_capped: pd.DataFrame, item_spid_area_totals: pd.DataFrame
) -> pd.DataFrame:
    """Splits each salesman's already stock-capped per-item low/median/high
    amount (from allocate_item_stock_by_salesman) across the areas they sell
    that item in — by their historical qty share within that item, per area —
    then sums across every item to a per (spid, area) low/median/high amount.
    """
    cols = ["spid", "area", "low_amt", "median_amt", "high_amt"]
    if (
        salesman_capped is None or salesman_capped.empty
        or item_spid_area_totals is None or item_spid_area_totals.empty
    ):
        return pd.DataFrame(columns=cols)

    areas = item_spid_area_totals.copy()
    areas["itemcode"] = areas["itemcode"].astype(str)
    areas["spid"] = areas["spid"].astype(str)
    areas["total_net_qty"] = areas["total_net_qty"].clip(lower=0.0)
    sp_item_totals = areas.groupby(["itemcode", "spid"])["total_net_qty"].transform("sum")
    areas["area_share"] = np.where(sp_item_totals > 0, areas["total_net_qty"] / sp_item_totals, 0.0)

    d = salesman_capped.copy()
    d["itemcode"] = d["itemcode"].astype(str)
    d["spid"] = d["spid"].astype(str)

    merged = areas.merge(d, on=["itemcode", "spid"], how="inner")
    merged["low_amt"] = merged["est_low_amt"] * merged["area_share"]
    merged["median_amt"] = merged["est_median_amt"] * merged["area_share"]
    merged["high_amt"] = merged["est_high_amt"] * merged["area_share"]

    out = merged.groupby(["spid", "area"], as_index=False)[["low_amt", "median_amt", "high_amt"]].sum()
    for c in ["low_amt", "median_amt", "high_amt"]:
        out[c] = out[c].round(0)
    return out[cols]
