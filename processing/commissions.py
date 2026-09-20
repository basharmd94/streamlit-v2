# processing/commissions.py
# Pure pandas logic for the Sales Commissions & Incentive Tracking feature.
# See commission_tracking_design.md for the full agreed design — this
# module currently only implements Product Tracking (the one fully-scoped,
# no-open-questions piece); the ranking/campaign pieces get their own
# functions here as each is picked up.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent.parent / "data"
_WATCHLIST_FILE = _DATA_DIR / "commission_product_watchlist.json"
_DATE_FMT = "%Y-%m-%d"

# "more wouldn't make sense" — user's own cap, per commission_tracking_design.md
MAX_WATCHLIST_ITEMS = 20
# "the choice can only go back 6 months and not more" — user's own cap on the
# Before window's start date, relative to the cutoff.
MAX_BACK_MONTHS = 6


def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _save_json(path: Path, data: dict) -> None:
    _DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def min_back_to(cutoff: pd.Timestamp) -> pd.Timestamp:
    """Earliest allowed 'back to' date for a given cutoff — at most 6 months back."""
    return cutoff - pd.DateOffset(months=MAX_BACK_MONTHS)


# ── Watchlist persistence ───────────────────────────────────────────────────
# Per product: only the item code + its own cutoff/back-to dates are saved —
# nothing computed. {zid: {itemcode: {"cutoff": "YYYY-MM-DD", "back_to": "YYYY-MM-DD"}}}

def load_watchlist(zid: str) -> dict:
    """Watched items + their own saved dates, for one ZID — products have
    different catalogs per business, so scoped per ZID like everything else
    in this app."""
    data = _load_json(_WATCHLIST_FILE)
    return dict(data.get(str(zid), {}))


def save_watchlist(zid: str, watchlist: dict) -> None:
    data = _load_json(_WATCHLIST_FILE)
    # cap enforced here too (belt-and-suspenders alongside add_product's own check)
    trimmed = dict(list(watchlist.items())[:MAX_WATCHLIST_ITEMS])
    data[str(zid)] = trimmed
    _save_json(_WATCHLIST_FILE, data)


def valid_dates(cutoff: pd.Timestamp, back_to: pd.Timestamp) -> bool:
    today = pd.Timestamp.today().normalize()
    if pd.isna(cutoff) or pd.isna(back_to):
        return False
    if cutoff > today:
        return False
    if back_to >= cutoff:
        return False
    if back_to < min_back_to(cutoff):
        return False
    return True


def add_product(zid: str, itemcode: str, cutoff: pd.Timestamp, back_to: pd.Timestamp) -> bool:
    """Adds a product with its own cutoff/back-to dates. False (no-op) if
    already tracked, at the cap, or the dates fail validation."""
    itemcode = str(itemcode).strip()
    if not itemcode or not valid_dates(cutoff, back_to):
        return False
    watchlist = load_watchlist(zid)
    if itemcode in watchlist or len(watchlist) >= MAX_WATCHLIST_ITEMS:
        return False
    watchlist[itemcode] = {"cutoff": cutoff.strftime(_DATE_FMT), "back_to": back_to.strftime(_DATE_FMT)}
    save_watchlist(zid, watchlist)
    return True


def remove_product(zid: str, itemcode: str) -> None:
    watchlist = load_watchlist(zid)
    watchlist.pop(str(itemcode), None)
    save_watchlist(zid, watchlist)


def update_product_dates(zid: str, itemcode: str, cutoff: pd.Timestamp, back_to: pd.Timestamp) -> bool:
    """Re-saves an already-tracked product's cutoff/back-to dates. False (no-op)
    if the item isn't tracked or the new dates fail validation — the old dates
    are left untouched in that case."""
    itemcode = str(itemcode).strip()
    if not valid_dates(cutoff, back_to):
        return False
    watchlist = load_watchlist(zid)
    if itemcode not in watchlist:
        return False
    watchlist[itemcode] = {"cutoff": cutoff.strftime(_DATE_FMT), "back_to": back_to.strftime(_DATE_FMT)}
    save_watchlist(zid, watchlist)
    return True


# ── Before/After computation (live, nothing computed here is persisted) ────

def _window_sum(df: pd.DataFrame, itemcode: str, start: pd.Timestamp, end: pd.Timestamp, col: str) -> float:
    """Sum of `col` for one item within [start, end) — end EXCLUSIVE, so a
    Before window and the After window starting at the same cutoff date never
    both count that date."""
    if df.empty:
        return 0.0
    sub = df[(df["itemcode"] == itemcode) & (df["_dt"] >= start) & (df["_dt"] < end)]
    if sub.empty:
        return 0.0
    return float(sub[col].sum())


def build_product_comparisons(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    items_df: pd.DataFrame,
    watchlist: dict,
    as_of: pd.Timestamp,
) -> list:
    """One entry per watched product — its own Before/After metrics, computed
    from its own saved cutoff/back_to dates (not one shared window for the
    whole list).

      Before = [back_to, cutoff)   After = [cutoff, as_of]

    The cutoff date itself belongs to After only, so the two windows never
    overlap. A malformed/unparseable saved entry is skipped rather than
    crashing the whole page.
    """
    if not watchlist:
        return []

    s = pd.DataFrame()
    if sales_df is not None and not sales_df.empty and {"itemcode", "date"}.issubset(sales_df.columns):
        s = sales_df.copy()
        s["itemcode"] = s["itemcode"].astype(str)
        s["_dt"] = pd.to_datetime(s["date"], errors="coerce")
        s["quantity"] = pd.to_numeric(s["quantity"], errors="coerce")
        s["totalsales"] = pd.to_numeric(s["totalsales"], errors="coerce")

    r = pd.DataFrame()
    if returns_df is not None and not returns_df.empty and {"itemcode", "date"}.issubset(returns_df.columns):
        r = returns_df.copy()
        r["itemcode"] = r["itemcode"].astype(str)
        r["_dt"] = pd.to_datetime(r["date"], errors="coerce")
        r["returnqty"] = pd.to_numeric(r["returnqty"], errors="coerce")

    meta_lookup = {}
    if items_df is not None and not items_df.empty and "item_id" in items_df.columns:
        m = items_df[["item_id", "item_name", "item_group", "stock"]].copy()
        m["item_id"] = m["item_id"].astype(str)
        # DB NUMERIC -> Decimal/object dtype -- crashes Arrow serialization
        # downstream if left as-is (same class of bug as WhatsFly Bulk
        # Messaging's Decimal columns, see CLAUDE.md).
        m["stock"] = pd.to_numeric(m["stock"], errors="coerce")
        meta_lookup = m.drop_duplicates("item_id").set_index("item_id").to_dict("index")

    # end-exclusive window, so make the After window include `as_of` itself
    after_end = as_of + pd.Timedelta(days=1)

    rows = []
    for itemcode, dates in watchlist.items():
        try:
            cutoff = pd.Timestamp(dates["cutoff"])
            back_to = pd.Timestamp(dates["back_to"])
        except Exception:
            continue
        if pd.isna(cutoff) or pd.isna(back_to):
            continue

        before_sold = _window_sum(s, itemcode, back_to, cutoff, "quantity")
        before_rev = _window_sum(s, itemcode, back_to, cutoff, "totalsales")
        before_ret = _window_sum(r, itemcode, back_to, cutoff, "returnqty")

        after_sold = _window_sum(s, itemcode, cutoff, after_end, "quantity")
        after_rev = _window_sum(s, itemcode, cutoff, after_end, "totalsales")
        after_ret = _window_sum(r, itemcode, cutoff, after_end, "returnqty")

        meta = meta_lookup.get(itemcode, {})
        rows.append({
            "itemcode": itemcode,
            "itemname": meta.get("item_name"),
            "itemgroup": meta.get("item_group"),
            "current_stock": meta.get("stock"),
            "cutoff": cutoff,
            "back_to": back_to,
            "as_of": as_of,
            "before": {
                "qty_sold": before_sold, "sales_revenue": before_rev,
                "qty_returned": before_ret, "net_qty": before_sold - before_ret,
            },
            "after": {
                "qty_sold": after_sold, "sales_revenue": after_rev,
                "qty_returned": after_ret, "net_qty": after_sold - after_ret,
            },
        })

    rows.sort(key=lambda d: (d["itemname"] or "").lower())
    return rows
