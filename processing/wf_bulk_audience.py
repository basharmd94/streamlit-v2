# processing/wf_bulk_audience.py
"""
Audience-filter builder for Marketing -> WhatsFly -> Bulk Messaging
(views/marketing.py's Bulk Messaging mode). Pure pandas/query logic; the
view layer owns all st.* widgets and session-state bookkeeping.

Design, per the discussion that led here:
  - A free-form "add a filter" builder, not a fixed pipeline -- any filter
    type can be added in any order (Area -> Salesman is the one real
    dependency: Salesman's own options only make sense once an Area is
    picked, since both come off the same opdor rows).
  - Every filter narrows whatever candidate cusid set already exists from
    the ones added before it (AND across filter types, by explicit rule).
  - One shared rolling window (start_date/end_date, from a "months back"
    slider in the view) governs: Area/Salesman, unique-products-bought
    band, Net Sales, Total Returns, and the Product filter. It does NOT
    apply to Days Since Last Sale, Avg Collection Days, Customer Score,
    Current Balance, or Inactive (all five keep their own established,
    all-time/independent-window definitions), or the two exact-date
    filters (Ordered On / Collection Received On).
  - Avg Collection Days is the one expensive computation (a per-customer
    running-balance walk across sales/returns/collections) and is
    therefore always added LAST -- once it's active, no more filters can
    be added on top of it (enforced in the view layer).
  - Net Sales means sales MINUS returns (same "Net Sales" meaning used
    elsewhere in this app, e.g. Target Management's %Collection formula)
    -- not gross sales. Total Returns is its own separate filter.
  - Candidates are recomputed from scratch on every rerun by replaying
    the whole filter list against current widget state (no incremental
    caching) -- so changing the shared window slider after filters are
    already set DOES re-evaluate every window-dependent filter already
    in the list, it is not frozen at add-time.
"""

from datetime import date, timedelta

import pandas as pd

from core.analytics import Analytics

DEFAULT_WINDOW_MONTHS = 6


def window_dates(months: int, today: date = None) -> tuple:
    """(start_date, end_date) for a rolling N-month window ending today."""
    today = today or date.today()
    return today - timedelta(days=int(months) * 30), today


# ---------------------------------------------------------------------------
# Area / Salesman -- opdor.xdiv / opdor.xsp, within the shared window.
# ---------------------------------------------------------------------------

def load_area_salesman_pool(zid: str, start_date: date, end_date: date) -> pd.DataFrame:
    df = Analytics(
        "bulk_area_salesman_orders", zid=zid,
        filters={"start_date": start_date, "end_date": end_date},
    ).data
    if df is None or df.empty:
        return pd.DataFrame(columns=["cusid", "area", "spid", "spname"])
    return df


def area_options(pool_df: pd.DataFrame) -> list:
    """Distinct xdiv values, real ones only -- a blank/NULL xdiv means "no
    territory assigned on this order", not a meaningful area to filter to
    (same blank-bucket handling as elsewhere in this app, e.g. item group)."""
    areas = pool_df["area"].dropna().astype(str).str.strip()
    return sorted(a for a in areas.unique() if a)


def _as_set(values) -> set:
    """Normalize a single value or an iterable of values into a stripped
    string set, for isin-style matching -- Area/Salesman used to be
    single-value (== comparison); both are multi-select now."""
    if isinstance(values, str):
        values = [values]
    return {str(v).strip() for v in values}


def apply_area(pool_df: pd.DataFrame, areas) -> set:
    """OR across selected areas -- a customer qualifies if sold to in ANY
    one of them (a territory filter, not an every-item-required one like
    Product's own multiselect)."""
    wanted = _as_set(areas)
    mask = pool_df["area"].astype(str).str.strip().isin(wanted)
    return set(pool_df.loc[mask, "cusid"].dropna().astype(str))


def salesman_options(pool_df: pd.DataFrame, areas) -> list:
    """[(spid, spname), ...] sorted by name. A salesman appears here if
    they sold in ANY of `areas` (union/OR across areas) -- real-world
    territories are mostly exclusive per salesman, not shared, so
    requiring a salesman to cover EVERY selected area (an earlier version
    of this) surfaced empty almost every time in practice and was mistaken
    for a bug -- see salesman_area_coverage below for showing which of the
    selected areas each option actually covers. Blank/NULL spid (no
    salesman on that order) is excluded, same reasoning as area_options.
    Picking one of these names still only ever surfaces THEIR customers
    within the selected areas (apply_salesman's row-level AND) -- never an
    area they sell in but that wasn't selected here."""
    wanted = _as_set(areas)
    sub = pool_df.copy()
    sub["area"] = sub["area"].astype(str).str.strip()
    sub["spid"] = sub["spid"].astype(str).str.strip()
    sub = sub[(sub["spid"] != "") & (sub["area"].isin(wanted))]
    opts = (
        sub[["spid", "spname"]]
        .drop_duplicates(subset=["spid"])
        .sort_values("spname", na_position="last")
    )
    return list(opts.itertuples(index=False, name=None))


def salesman_area_coverage(pool_df: pd.DataFrame, areas) -> dict:
    """spid -> sorted list of areas (from the selected `areas`, and ONLY
    those) that salesman actually sold in -- lets the Salesman dropdown
    show e.g. "Ziaur Rahman (Asulia)" so it's clear at a glance which of
    the selected areas picking them will actually pull customers from,
    without having to select them first to find out."""
    wanted = _as_set(areas)
    sub = pool_df.copy()
    sub["area"] = sub["area"].astype(str).str.strip()
    sub["spid"] = sub["spid"].astype(str).str.strip()
    sub = sub[(sub["spid"] != "") & (sub["area"].isin(wanted))]
    return sub.groupby("spid")["area"].apply(lambda s: sorted(set(s))).to_dict()


def apply_salesman(pool_df: pd.DataFrame, areas, spids) -> set:
    """Row-level AND between area-membership and salesman-membership (the
    same order must satisfy both), OR-within each of the two sets -- a
    customer qualifies if ANY selected salesman sold to them in ANY
    selected area on the same order, not "any selected area at all" OR'd
    independently with "any selected salesman at all"."""
    wanted_areas = _as_set(areas)
    wanted_spids = _as_set(spids)
    mask = (
        pool_df["area"].astype(str).str.strip().isin(wanted_areas)
        & pool_df["spid"].astype(str).str.strip().isin(wanted_spids)
    )
    return set(pool_df.loc[mask, "cusid"].dropna().astype(str))


# ---------------------------------------------------------------------------
# Shared sales-lines loader -- backs the unique-products band, the Product
# filter, and Inactive. Restricting to `cusids` (already-narrowed
# candidates) when available keeps every downstream pull small.
# ---------------------------------------------------------------------------

def load_sales_lines(zid: str, start_date: date, end_date: date, cusids=None) -> pd.DataFrame:
    filters = {"start_date": start_date, "end_date": end_date}
    if cusids is not None:
        filters["cusids"] = list(cusids)
    df = Analytics("bulk_sales_lines", zid=zid, filters=filters).data
    if df is None or df.empty:
        return pd.DataFrame(columns=["cusid", "cusname", "itemcode", "itemname", "date", "final_sales"])
    # `final_sales` is computed in SQL as a NUMERIC subtraction (altsales -
    # proddiscount), so psycopg2 hands it back as python Decimal objects,
    # not float -- an object-dtype column of Decimals sums fine in pandas
    # but crashes Streamlit's Arrow serialization the moment it's rendered
    # or downloaded (real bug hit live: "Could not convert Decimal(...) to
    # double"). Cast once here so every consumer (Total Sales filter, the
    # always-shown "Total Sales (window)" column, etc.) gets a clean float.
    df["final_sales"] = df["final_sales"].astype(float)
    return df


def unique_product_counts(sales_lines: pd.DataFrame) -> pd.Series:
    """Per-customer count of DISTINCT items bought in the window."""
    if sales_lines.empty:
        return pd.Series(dtype=int)
    return sales_lines.groupby("cusid")["itemcode"].nunique()


def apply_product_count_band(sales_lines: pd.DataFrame, min_n: int, max_n: int) -> set:
    counts = unique_product_counts(sales_lines)
    return set(counts[(counts >= min_n) & (counts <= max_n)].index.astype(str))


def product_options(sales_lines: pd.DataFrame) -> list:
    """[(itemcode, itemname), ...] sorted by name, for the multiselect."""
    if sales_lines.empty:
        return []
    opts = sales_lines[["itemcode", "itemname"]].dropna(subset=["itemcode"]).drop_duplicates()
    opts = opts.sort_values("itemname", na_position="last")
    return list(opts.itertuples(index=False, name=None))


def apply_product_filter(sales_lines: pd.DataFrame, selected_itemcodes: list) -> set:
    """AND semantics -- a customer qualifies only if they bought EVERY
    selected product at least once in the window (confirmed: multiple
    values within a filter combine the same AND way as across filters)."""
    if sales_lines.empty or not selected_itemcodes:
        return set()
    wanted = set(selected_itemcodes)
    bought_sets = sales_lines.groupby("cusid")["itemcode"].apply(set)
    qualifying = bought_sets[bought_sets.apply(lambda s: wanted.issubset(s))].index
    return set(qualifying.astype(str))


def apply_inactive(candidate_cusids: set, sales_lines: pd.DataFrame) -> set:
    """Customers among the current candidates with ZERO purchases in the
    window passed in. NOTE: as of the follow-up that added its own
    "Time window" slider to this filter, `sales_lines` is loaded against
    THIS filter's own independent window, not the shared one -- inactivity
    is a genuinely different question ("how long since anything happened")
    from what the shared slider otherwise scopes."""
    active = set(sales_lines["cusid"].dropna().astype(str).unique())
    return set(candidate_cusids) - active


def load_returns_lines(zid: str, start_date: date, end_date: date, cusids=None) -> pd.DataFrame:
    """Same return population as get_return_data (opcdt/opcrn UNION
    imtemptdt/imtemptrn) -- Avg Collection Days already pulled this
    directly via Analytics; this wrapper gives Net Sales/Total Returns
    the same float-safe loading path as load_sales_lines."""
    filters = {"start_date": start_date, "end_date": end_date}
    if cusids is not None:
        filters["cusids"] = list(cusids)
    df = Analytics("bulk_returns_lines", zid=zid, filters=filters).data
    if df is None or df.empty:
        return pd.DataFrame(columns=["cusid", "date", "treturnamt"])
    # Same Decimal-from-NUMERIC-SQL issue as final_sales in load_sales_lines
    # above -- cast once here so nothing downstream has to guard against it.
    df["treturnamt"] = df["treturnamt"].astype(float)
    return df


def total_sales_by_customer(sales_lines: pd.DataFrame) -> pd.Series:
    """Per-customer sum of final_sales (GROSS -- before netting out
    returns) within whatever window `sales_lines` was loaded for. Kept as
    a building block for net_sales_by_customer below; nothing user-facing
    reads this directly anymore -- see Common Pitfall #1 in CLAUDE.md on
    why a "sales" figure shown on its own should be net of returns."""
    if sales_lines.empty:
        return pd.Series(dtype=float)
    return sales_lines.groupby("cusid")["final_sales"].sum()


def total_returns_by_customer(returns_lines: pd.DataFrame) -> pd.Series:
    """Per-customer sum of treturnamt within whatever window
    `returns_lines` was loaded for -- backs the Total Returns filter."""
    if returns_lines.empty:
        return pd.Series(dtype=float)
    return returns_lines.groupby("cusid")["treturnamt"].sum()


def net_sales_by_customer(sales_lines: pd.DataFrame, returns_lines: pd.DataFrame) -> pd.Series:
    """Per-customer NET sales (sales - returns) within the window -- what
    "Net Sales" means everywhere else in this app (e.g. the %Collection
    formula in Target Management), and confirmed as the intended meaning
    for this filter/column too, not gross sales. fill_value=0 on the
    subtraction means a customer who returned something but has zero
    matching sales in this exact window still nets out to a (negative)
    real value instead of being silently dropped."""
    sales_total = total_sales_by_customer(sales_lines)
    returns_total = total_returns_by_customer(returns_lines)
    return sales_total.sub(returns_total, fill_value=0.0)


def apply_net_sales_band(sales_lines: pd.DataFrame, returns_lines: pd.DataFrame, min_amt: float, max_amt: float) -> set:
    net = net_sales_by_customer(sales_lines, returns_lines)
    return set(net[(net >= min_amt) & (net <= max_amt)].index.astype(str))


def apply_total_returns_band(returns_lines: pd.DataFrame, min_amt: float, max_amt: float) -> set:
    totals = total_returns_by_customer(returns_lines)
    return set(totals[(totals >= min_amt) & (totals <= max_amt)].index.astype(str))


# ---------------------------------------------------------------------------
# Contactable-only gate -- final audience-table requirement, not a filter
# in the free-form builder above (it always applies, per explicit ask).
# ---------------------------------------------------------------------------

def apply_contactable_only(audience_df: pd.DataFrame) -> tuple:
    """Keep only rows with BOTH `cusmobile` and `whatsapp` populated
    (blank/whitespace-only in either one disqualifies the row) -- this is
    deliberately stricter than processing/common.py::customer_whatsapp_numbers'
    own "prefer whatsapp, fall back to cusmobile" logic used for the
    single-message WhatsFly panel elsewhere in this app; for a bulk
    campaign audience the explicit ask was to require both on file, not
    just one. cacus_directory's own query already COALESCEs both columns
    to '' (never NaN), so a plain stripped-empty-string check is enough.
    Returns (kept_df, dropped_count) so the drop is surfaced, never silent."""
    if audience_df.empty:
        return audience_df, 0
    has_mobile = audience_df["cusmobile"].astype(str).str.strip() != ""
    has_whatsapp = audience_df["whatsapp"].astype(str).str.strip() != ""
    keep_mask = has_mobile & has_whatsapp
    return audience_df[keep_mask].copy(), int((~keep_mask).sum())


# ---------------------------------------------------------------------------
# Days Since Last Sale -- all-time last sale date, NOT window-bound (same
# definition already established elsewhere in this app).
# ---------------------------------------------------------------------------

def load_last_sale_dates(zid: str, cusids=None) -> pd.DataFrame:
    filters = {}
    if cusids is not None:
        filters["cusids"] = list(cusids)
    df = Analytics("bulk_last_sale_dates", zid=zid, filters=filters).data
    if df is None or df.empty:
        return pd.DataFrame(columns=["cusid", "last_sale_date", "days_since"])
    df["last_sale_date"] = pd.to_datetime(df["last_sale_date"])
    df["days_since"] = (pd.Timestamp.today().normalize() - df["last_sale_date"]).dt.days
    return df


def apply_days_since_last_sale(last_sale_df: pd.DataFrame, min_days: int, max_days: int) -> set:
    sub = last_sale_df[(last_sale_df["days_since"] >= min_days) & (last_sale_df["days_since"] <= max_days)]
    return set(sub["cusid"].dropna().astype(str))


# ---------------------------------------------------------------------------
# Avg Collection Days -- reuses processing/collection.py's own
# average_days_to_collection exactly, scoped to the shared window AND the
# already-narrowed candidate set (this is the expensive one -- a per-
# customer running-balance walk -- which is why it's always added last).
# ---------------------------------------------------------------------------

def compute_avg_collection_days(zid: str, start_date: date, end_date: date, candidate_cusids: set) -> pd.DataFrame:
    from processing import collection as _collection  # local import: avoids a hard, always-paid dependency for every other filter

    cusids = list(candidate_cusids)
    sales = Analytics("bulk_sales_lines", zid=zid, filters={"start_date": start_date, "end_date": end_date, "cusids": cusids}).data
    returns = Analytics("bulk_returns_lines", zid=zid, filters={"start_date": start_date, "end_date": end_date, "cusids": cusids}).data
    coll = Analytics("bulk_collection_lines", zid=zid, filters={"start_date": start_date, "end_date": end_date, "cusids": cusids}).data

    sales = sales if sales is not None else pd.DataFrame(columns=["cusid", "cusname", "date", "final_sales"])
    returns = returns if returns is not None else pd.DataFrame(columns=["cusid", "date", "treturnamt"])
    coll = coll if coll is not None else pd.DataFrame(columns=["cusid", "cusname", "date", "value"])

    if sales.empty or coll.empty:
        return pd.DataFrame(columns=["cusid", "cusname", "average_days_to_collection"])

    for df in (sales, returns, coll):
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])

    avg_days, _pivot_df, _avg_days_between, _combined_df = _collection.average_days_to_collection(sales, returns, coll)
    return avg_days


def apply_avg_collection_days(avg_days_df: pd.DataFrame, min_days: float, max_days: float) -> set:
    if avg_days_df is None or avg_days_df.empty:
        return set()
    sub = avg_days_df[
        (avg_days_df["average_days_to_collection"] >= min_days)
        & (avg_days_df["average_days_to_collection"] <= max_days)
    ]
    return set(sub["cusid"].dropna().astype(str))


# ---------------------------------------------------------------------------
# Exact-date filters -- independent of the window slider, per explicit ask.
# ---------------------------------------------------------------------------

def apply_order_date(zid: str, order_date: date) -> set:
    df = Analytics("bulk_order_date", zid=zid, filters={"order_date": order_date}).data
    if df is None or df.empty:
        return set()
    return set(df["cusid"].dropna().astype(str))


def apply_collection_date(zid: str, collection_date: date) -> set:
    df = Analytics(
        "bulk_collection_lines", zid=zid,
        filters={"start_date": collection_date, "end_date": collection_date},
    ).data
    if df is None or df.empty:
        return set()
    return set(df["cusid"].dropna().astype(str))
