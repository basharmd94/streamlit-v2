# processing/manufacturing.py
# Pure-pandas helpers for Manufacturing Analysis (moord/moodt) — 100000/100005/100009.
# Costing here is material cost only (BOM consumption qty x rate) — there is no
# labor/overhead in moord/moodt; overhead is layered in separately via the
# admin-allocation functions below, sourced from GL "06" (Office & Admin) accounts.

import numpy as np
import pandas as pd


def trailing_n_months_window(today: pd.Timestamp, n_months: int) -> tuple:
    """Last n_months *completed* calendar months, excluding the current (partial) month."""
    cur_month_start = pd.Timestamp(today.year, today.month, 1)
    end = cur_month_start - pd.Timedelta(days=1)
    start = cur_month_start - pd.DateOffset(months=n_months)
    return start, end


def _n_months_in_window(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return max(1, (end.year - start.year) * 12 + (end.month - start.month) + 1)


def merge_mo_lines(mo_header: pd.DataFrame, mo_detail: pd.DataFrame) -> pd.DataFrame:
    """One row per BOM line consumed against an MO, with the finished good's
    own context (itemcode_fg/itemname_fg/itemgroup_fg/qtyprd/date/year/month)
    attached from mo_header. itemcode_rm/itemname_rm/itemgroup_rm are the raw
    material consumed on that line. line_cost = qty x rate (material cost).
    """
    cols = [
        "zid", "monumber", "lineno", "itemcode_rm", "itemname_rm", "itemgroup_rm",
        "warehouse", "qty", "qtyord", "unit_rm", "rate", "line_cost",
        "itemcode_fg", "itemname_fg", "itemgroup_fg", "qtyprd", "unit_fg",
        "date", "year", "month",
    ]
    if mo_header is None or mo_header.empty or mo_detail is None or mo_detail.empty:
        return pd.DataFrame(columns=cols)

    h = mo_header.copy()
    h["monumber"] = h["monumber"].astype(str)
    h["itemcode"] = h["itemcode"].astype(str)
    h["qtyprd"] = pd.to_numeric(h["qtyprd"], errors="coerce").fillna(0.0)
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h["year"] = pd.to_numeric(h["year"], errors="coerce")
    h["month"] = pd.to_numeric(h["month"], errors="coerce")

    d = mo_detail.copy()
    d["monumber"] = d["monumber"].astype(str)
    d["itemcode"] = d["itemcode"].astype(str)
    d["qty"] = pd.to_numeric(d["qty"], errors="coerce").fillna(0.0)
    d["qtyord"] = pd.to_numeric(d.get("qtyord"), errors="coerce").fillna(0.0)
    d["rate"] = pd.to_numeric(d["rate"], errors="coerce").fillna(0.0)

    merged = d.merge(
        h[["monumber", "itemcode", "itemname", "itemgroup", "qtyprd", "unit", "date", "year", "month"]],
        on="monumber", how="inner", suffixes=("_rm", "_fg"),
    )
    merged["line_cost"] = merged["qty"] * merged["rate"]
    return merged[cols]


def compute_mo_cost(mo_lines: pd.DataFrame) -> pd.DataFrame:
    """Collapses BOM lines back to one row per MO: total material cost and
    cost per unit produced (material_cost / qtyprd).
    """
    cols = ["monumber", "itemcode", "itemname", "itemgroup", "date", "year", "month",
            "qtyprd", "material_cost", "cost_per_unit"]
    if mo_lines is None or mo_lines.empty:
        return pd.DataFrame(columns=cols)

    g = mo_lines.groupby("monumber").agg(
        itemcode=("itemcode_fg", "first"),
        itemname=("itemname_fg", "first"),
        itemgroup=("itemgroup_fg", "first"),
        date=("date", "first"),
        year=("year", "first"),
        month=("month", "first"),
        qtyprd=("qtyprd", "first"),
        material_cost=("line_cost", "sum"),
    ).reset_index()
    g["cost_per_unit"] = np.where(g["qtyprd"] > 0, g["material_cost"] / g["qtyprd"], 0.0)
    return g[cols]


def compute_fg_cost_summary(mo_cost: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Per finished good, within the window: qty-weighted average cost/unit,
    the most recent single batch's cost/unit, batch count, and total material
    cost — the all-FG overview for the FG Costing tab.
    """
    cols = ["itemcode", "itemname", "itemgroup", "total_qtyprd", "total_material_cost",
            "n_batches", "latest_date", "avg_cost_per_unit", "latest_cost_per_unit"]
    if mo_cost is None or mo_cost.empty:
        return pd.DataFrame(columns=cols)

    d = mo_cost[(mo_cost["date"] >= start) & (mo_cost["date"] <= end)]
    if d.empty:
        return pd.DataFrame(columns=cols)

    g = d.groupby(["itemcode", "itemname", "itemgroup"], as_index=False).agg(
        total_qtyprd=("qtyprd", "sum"),
        total_material_cost=("material_cost", "sum"),
        n_batches=("monumber", "nunique"),
        latest_date=("date", "max"),
    )
    g["avg_cost_per_unit"] = np.where(g["total_qtyprd"] > 0, g["total_material_cost"] / g["total_qtyprd"], 0.0)

    latest = (
        d.sort_values("date").groupby("itemcode").tail(1)[["itemcode", "cost_per_unit"]]
        .rename(columns={"cost_per_unit": "latest_cost_per_unit"})
    )
    g = g.merge(latest, on="itemcode", how="left")
    return g.sort_values("total_material_cost", ascending=False).reset_index(drop=True)[cols]


def compute_fg_cost_history(mo_cost: pd.DataFrame, itemcode: str) -> pd.DataFrame:
    """Monthly qty-weighted average cost/unit trend for one finished good,
    across all history available in mo_cost (caller slices the window)."""
    cols = ["year", "month", "total_qtyprd", "total_material_cost", "n_batches", "avg_cost_per_unit"]
    if mo_cost is None or mo_cost.empty:
        return pd.DataFrame(columns=cols)

    d = mo_cost[mo_cost["itemcode"] == str(itemcode)]
    if d.empty:
        return pd.DataFrame(columns=cols)

    g = d.groupby(["year", "month"], as_index=False).agg(
        total_qtyprd=("qtyprd", "sum"),
        total_material_cost=("material_cost", "sum"),
        n_batches=("monumber", "nunique"),
    )
    g["avg_cost_per_unit"] = np.where(g["total_qtyprd"] > 0, g["total_material_cost"] / g["total_qtyprd"], 0.0)
    return g.sort_values(["year", "month"]).reset_index(drop=True)[cols]


def compute_cost_driver_breakdown(
    mo_lines: pd.DataFrame, itemcode: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """For one finished good within the window: each raw-material line's share
    of that FG's total material cost — which ingredient/component is the
    biggest cost driver."""
    cols = ["itemcode", "itemname", "total_qty", "total_cost", "avg_rate", "pct_of_total"]
    if mo_lines is None or mo_lines.empty:
        return pd.DataFrame(columns=cols)

    d = mo_lines[
        (mo_lines["itemcode_fg"] == str(itemcode))
        & (mo_lines["date"] >= start) & (mo_lines["date"] <= end)
    ]
    if d.empty:
        return pd.DataFrame(columns=cols)

    g = d.groupby(["itemcode_rm", "itemname_rm"], as_index=False).agg(
        total_qty=("qty", "sum"),
        total_cost=("line_cost", "sum"),
        avg_rate=("rate", "mean"),
    ).rename(columns={"itemcode_rm": "itemcode", "itemname_rm": "itemname"})

    total = g["total_cost"].sum()
    g["pct_of_total"] = np.where(total > 0, g["total_cost"] / total * 100, 0.0)
    return g.sort_values("total_cost", ascending=False).reset_index(drop=True)[cols]


def compute_admin_allocation_for_fg(
    mo_cost: pd.DataFrame, admin_expense_df: pd.DataFrame, itemcode: str,
    start: pd.Timestamp, end: pd.Timestamp,
) -> tuple:
    """Allocates each month's GL '06' (Office & Admin) expense across finished
    goods by their share of that month's total production material cost, then
    spreads the selected FG's allocated share across its units produced that
    month. Returns (per-month breakdown, window-weighted-avg admin cost/unit).

    allocated_admin[month] = total_06[month] x (fg_material_cost[month] / total_fg_material_cost[month])
    admin_cost_per_unit[month] = allocated_admin[month] / fg_qtyprd[month]
    """
    cols = ["year", "month", "fg_material_cost", "fg_qtyprd", "total_fg_cost",
            "admin_expense", "fg_share", "allocated_admin", "admin_cost_per_unit"]
    empty = pd.DataFrame(columns=cols)
    if mo_cost is None or mo_cost.empty:
        return empty, 0.0

    d = mo_cost[(mo_cost["date"] >= start) & (mo_cost["date"] <= end)]
    if d.empty:
        return empty, 0.0

    fg = d[d["itemcode"] == str(itemcode)].groupby(["year", "month"], as_index=False).agg(
        fg_material_cost=("material_cost", "sum"),
        fg_qtyprd=("qtyprd", "sum"),
    )
    if fg.empty:
        return empty, 0.0

    monthly_totals = d.groupby(["year", "month"], as_index=False)["material_cost"].sum().rename(
        columns={"material_cost": "total_fg_cost"}
    )

    if admin_expense_df is not None and not admin_expense_df.empty:
        admin = admin_expense_df[["year", "month", "value"]].copy()
        admin["year"] = pd.to_numeric(admin["year"], errors="coerce")
        admin["month"] = pd.to_numeric(admin["month"], errors="coerce")
        admin["value"] = pd.to_numeric(admin["value"], errors="coerce").fillna(0.0)
    else:
        admin = pd.DataFrame(columns=["year", "month", "value"])

    m = fg.merge(monthly_totals, on=["year", "month"], how="left")
    m = m.merge(admin, on=["year", "month"], how="left").rename(columns={"value": "admin_expense"})
    m["total_fg_cost"] = m["total_fg_cost"].fillna(0.0)
    m["admin_expense"] = m["admin_expense"].fillna(0.0)
    m["fg_share"] = np.where(m["total_fg_cost"] > 0, m["fg_material_cost"] / m["total_fg_cost"], 0.0)
    m["allocated_admin"] = m["admin_expense"] * m["fg_share"]
    m["admin_cost_per_unit"] = np.where(m["fg_qtyprd"] > 0, m["allocated_admin"] / m["fg_qtyprd"], 0.0)

    total_qtyprd = m["fg_qtyprd"].sum()
    weighted_avg = float(m["allocated_admin"].sum() / total_qtyprd) if total_qtyprd > 0 else 0.0
    return m.sort_values(["year", "month"]).reset_index(drop=True)[cols], weighted_avg


def compute_rm_rate_trend(mo_lines: pd.DataFrame, itemcode: str) -> tuple:
    """For one raw material: every BOM line it appeared on (chronological) and
    a monthly qty-weighted average rate trend."""
    detail_cols = ["date", "monumber", "itemcode_fg", "itemname_fg", "qty", "rate"]
    monthly_cols = ["year", "month", "avg_rate", "total_qty"]
    if mo_lines is None or mo_lines.empty:
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=monthly_cols)

    d = mo_lines[mo_lines["itemcode_rm"] == str(itemcode)].copy()
    if d.empty:
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=monthly_cols)

    detail = d.sort_values("date")[detail_cols].reset_index(drop=True)

    def _monthly_stats(g: pd.DataFrame) -> pd.Series:
        total_qty = float(g["qty"].sum())
        avg_rate = float(np.average(g["rate"], weights=g["qty"])) if total_qty > 0 else float(g["rate"].mean())
        return pd.Series({"avg_rate": avg_rate, "total_qty": total_qty})

    monthly = d.groupby(["year", "month"])[["rate", "qty"]].apply(_monthly_stats).reset_index()
    return detail, monthly.sort_values(["year", "month"]).reset_index(drop=True)[monthly_cols]


def compute_rm_price_movers(mo_lines: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Ranks raw materials by the % change in rate from their first to last
    BOM line within the window — the biggest movers (up or down) first.
    Raw materials with only a single observation in the window are excluded
    (no real "change" to report)."""
    cols = ["itemcode", "itemname", "first_date", "first_rate", "last_date", "last_rate", "pct_change", "n_lines"]
    if mo_lines is None or mo_lines.empty:
        return pd.DataFrame(columns=cols)

    d = mo_lines[(mo_lines["date"] >= start) & (mo_lines["date"] <= end)].sort_values("date")
    if d.empty:
        return pd.DataFrame(columns=cols)

    g = d.groupby(["itemcode_rm", "itemname_rm"], as_index=False).agg(
        first_date=("date", "first"), first_rate=("rate", "first"),
        last_date=("date", "last"), last_rate=("rate", "last"),
        n_lines=("rate", "size"),
    ).rename(columns={"itemcode_rm": "itemcode", "itemname_rm": "itemname"})

    g = g[g["n_lines"] > 1].copy()
    if g.empty:
        return pd.DataFrame(columns=cols)
    g["pct_change"] = np.where(g["first_rate"] != 0, (g["last_rate"] - g["first_rate"]) / g["first_rate"] * 100, 0.0)
    return g.sort_values("pct_change", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)[cols]


def compute_rm_requirement(mo_lines: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Total raw-material qty and value (qty x rate) actually consumed across
    every MO in the window, for the whole ZID — doubles as both 'RM needed for
    the last N months' and 'Top RM by spend' (just sort by either column)."""
    cols = ["itemcode", "itemname", "itemgroup", "unit", "total_qty", "total_value", "n_lines"]
    if mo_lines is None or mo_lines.empty:
        return pd.DataFrame(columns=cols)

    d = mo_lines[(mo_lines["date"] >= start) & (mo_lines["date"] <= end)]
    if d.empty:
        return pd.DataFrame(columns=cols)

    g = d.groupby(["itemcode_rm", "itemname_rm", "itemgroup_rm", "unit_rm"], as_index=False).agg(
        total_qty=("qty", "sum"),
        total_value=("line_cost", "sum"),
        n_lines=("qty", "size"),
    ).rename(columns={"itemcode_rm": "itemcode", "itemname_rm": "itemname",
                       "itemgroup_rm": "itemgroup", "unit_rm": "unit"})
    return g.sort_values("total_value", ascending=False).reset_index(drop=True)[cols]


def compute_bom_ratio(mo_header: pd.DataFrame, mo_lines: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Per (finished good, raw material): how much RM qty it actually took to
    produce one unit of that FG, within the window — total RM qty consumed
    across all that FG's MOs divided by the FG's total qty produced. Feeds
    the RM Stock Coverage tab (FG sales demand -> RM demand)."""
    cols = ["fg_itemcode", "rm_itemcode", "rm_itemname", "bom_ratio"]
    if mo_header is None or mo_header.empty or mo_lines is None or mo_lines.empty:
        return pd.DataFrame(columns=cols)

    h = mo_header.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h["itemcode"] = h["itemcode"].astype(str)
    h["qtyprd"] = pd.to_numeric(h["qtyprd"], errors="coerce").fillna(0.0)
    h = h[(h["date"] >= start) & (h["date"] <= end)]
    if h.empty:
        return pd.DataFrame(columns=cols)
    fg_qtyprd = h.groupby("itemcode", as_index=False)["qtyprd"].sum().rename(
        columns={"itemcode": "fg_itemcode", "qtyprd": "fg_total_qtyprd"}
    )

    d = mo_lines[(mo_lines["date"] >= start) & (mo_lines["date"] <= end)]
    if d.empty:
        return pd.DataFrame(columns=cols)
    rm_qty = d.groupby(["itemcode_fg", "itemcode_rm", "itemname_rm"], as_index=False)["qty"].sum().rename(
        columns={"itemcode_fg": "fg_itemcode", "itemcode_rm": "rm_itemcode",
                 "itemname_rm": "rm_itemname", "qty": "rm_total_qty"}
    )

    m = rm_qty.merge(fg_qtyprd, on="fg_itemcode", how="left")
    m["fg_total_qtyprd"] = m["fg_total_qtyprd"].fillna(0.0)
    m["bom_ratio"] = np.where(m["fg_total_qtyprd"] > 0, m["rm_total_qty"] / m["fg_total_qtyprd"], 0.0)
    return m[cols]


def compute_avg_monthly_fg_sales(
    sales_df: pd.DataFrame, returns_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Average monthly net (sales - returns) qty per finished good over the
    window — plain total/n_months, matching the Daily/Monthly Avg (3M)
    convention used elsewhere in the app (not a non-zero-month filter)."""
    cols = ["itemcode", "avg_monthly_qty"]
    n_months = _n_months_in_window(start, end)

    def _prep(df: pd.DataFrame, qty_col: str, sign: float) -> pd.DataFrame:
        if df is None or df.empty or "itemcode" not in df.columns:
            return pd.DataFrame(columns=["itemcode", "qty"])
        x = df.copy()
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        x = x[(x["date"] >= start) & (x["date"] <= end)]
        if x.empty:
            return pd.DataFrame(columns=["itemcode", "qty"])
        x["itemcode"] = x["itemcode"].astype(str)
        x["qty"] = sign * pd.to_numeric(x[qty_col], errors="coerce").fillna(0.0)
        return x.groupby("itemcode", as_index=False)["qty"].sum()

    s = _prep(sales_df, "quantity", 1.0)
    r = _prep(returns_df, "returnqty", -1.0)
    combined = pd.concat([s, r], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=cols)

    net = combined.groupby("itemcode", as_index=False)["qty"].sum().rename(columns={"qty": "total_net_qty"})
    net["avg_monthly_qty"] = (net["total_net_qty"] / n_months).clip(lower=0.0)
    return net[cols]


def compute_current_stock_from_imtrn(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Current stock per item = cumulative balance across all of imtrn's
    history (sum of every xqty x xsign movement ever recorded), i.e. the
    'stock' Analytics table summed with no cutoff. final_items_view is NOT
    used for this — it returned 0 for several RM/FG items in 100005 that
    imtrn shows do have real stock (e.g. a view warehouse-scope gap), so raw
    imtrn is the reliable source here, per how this feature was specified.
    """
    cols = ["itemcode", "current_stock"]
    if stock_df is None or stock_df.empty:
        return pd.DataFrame(columns=cols)
    d = stock_df.copy()
    d["itemcode"] = d["itemcode"].astype(str)
    d["stockqty"] = pd.to_numeric(d["stockqty"], errors="coerce").fillna(0.0)
    g = d.groupby("itemcode", as_index=False)["stockqty"].sum().rename(columns={"stockqty": "current_stock"})
    return g[cols]


def compute_rm_stock_coverage(
    bom_ratio_df: pd.DataFrame, avg_fg_sales_df: pd.DataFrame, current_stock_df: pd.DataFrame,
    threshold_months: float = 1.0,
) -> pd.DataFrame:
    """Projected monthly RM need = sum over every FG that uses this RM of
    (that FG's avg monthly sales qty x its BOM ratio for this RM), compared
    against current RM stock (current_stock_df: itemcode, current_stock —
    see compute_current_stock_from_imtrn). coverage_months = current_stock /
    projected need; is_short flags RMs below threshold_months of coverage.
    RMs with zero projected need (no FG currently selling) get NaN coverage —
    nothing to compare against, not a shortage.
    """
    cols = ["itemcode", "itemname", "projected_monthly_need", "current_stock", "coverage_months", "is_short"]
    if bom_ratio_df is None or bom_ratio_df.empty:
        return pd.DataFrame(columns=cols)

    m = bom_ratio_df.merge(
        avg_fg_sales_df if avg_fg_sales_df is not None else pd.DataFrame(columns=["itemcode", "avg_monthly_qty"]),
        left_on="fg_itemcode", right_on="itemcode", how="left",
    )
    m["avg_monthly_qty"] = m["avg_monthly_qty"].fillna(0.0)
    m["projected_need"] = m["bom_ratio"] * m["avg_monthly_qty"]

    g = m.groupby(["rm_itemcode", "rm_itemname"], as_index=False)["projected_need"].sum().rename(
        columns={"rm_itemcode": "itemcode", "rm_itemname": "itemname", "projected_need": "projected_monthly_need"}
    )

    if current_stock_df is not None and not current_stock_df.empty:
        st = current_stock_df[["itemcode", "current_stock"]].copy()
        st["itemcode"] = st["itemcode"].astype(str)
        g = g.merge(st, on="itemcode", how="left")
    else:
        g["current_stock"] = 0.0
    g["current_stock"] = g["current_stock"].fillna(0.0)

    g["coverage_months"] = np.where(
        g["projected_monthly_need"] > 0, g["current_stock"] / g["projected_monthly_need"], np.nan
    )
    g["is_short"] = (g["projected_monthly_need"] > 0) & (g["coverage_months"] < threshold_months)
    return g.sort_values("coverage_months", na_position="last").reset_index(drop=True)[cols]


def compute_bom_variance(mo_lines: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Per raw material, within the window: actual qty issued (xqty) vs the
    standard/planned BOM qty (xqtyord) — a wastage/efficiency signal where
    actual consumption runs persistently above the standard recipe."""
    cols = ["itemcode", "itemname", "total_qty", "total_qtyord", "variance_qty", "variance_pct", "over_consumption"]
    if mo_lines is None or mo_lines.empty:
        return pd.DataFrame(columns=cols)

    d = mo_lines[(mo_lines["date"] >= start) & (mo_lines["date"] <= end)]
    if d.empty:
        return pd.DataFrame(columns=cols)

    g = d.groupby(["itemcode_rm", "itemname_rm"], as_index=False).agg(
        total_qty=("qty", "sum"),
        total_qtyord=("qtyord", "sum"),
    ).rename(columns={"itemcode_rm": "itemcode", "itemname_rm": "itemname"})

    g["variance_qty"] = g["total_qty"] - g["total_qtyord"]
    g["variance_pct"] = np.where(g["total_qtyord"] > 0, g["variance_qty"] / g["total_qtyord"] * 100, 0.0)
    g["over_consumption"] = g["variance_qty"] > 0
    return g.sort_values("variance_qty", ascending=False).reset_index(drop=True)[cols]
