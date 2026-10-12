# processing/customer_tracking.py
# Pure data logic for Target Management -> Customer Tracking — two
# independent, manager-curated customer watchlists (Collection / Returns),
# plus a standalone Promised Payment list. Originally built 2026-10-10 as a
# Commission Results section; moved to its own Target Management radio and
# redesigned 2026-10-12 per explicit spec (see CLAUDE.md's "Customer
# Tracking" section for the full writeup) — color-coded tables instead of
# plain Yes/No summaries, and always pooled across 100001+100000 (the two
# ZIDs sharing one field sales team).
#
# Setup (the two comma-list watchlist inputs) still lives on the admin
# Commissions page, per-ZID as originally built — see load_watchlist_pooled
# below for how the results side unions both ZIDs' own saved lists.
# JSON-backed, same convention as commission_product_watchlist.json — this
# isn't a payout mechanism, so it doesn't belong in the commission_campaigns
# table at all.

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from processing.holidays import next_working_day

_DATA_DIR = Path(__file__).parent.parent / "data"
_WATCHLIST_FILE = _DATA_DIR / "customer_tracking_watchlist.json"

GROUPS = ("collection", "returns")

# "light red with visible writing" — explicit ask, deliberately lighter than
# processing/common.py::highlight_overdue_date's high-contrast #ff4b4b/white
# (that one flags a single date cell; these mark a whole wide row).
_FLAG_STYLE = "background-color: #ffd6d6; color: #7a0000; font-weight: 600;"


def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _save_json(path: Path, data: dict) -> None:
    _DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def parse_codes(text: str) -> list:
    """Comma-separated free text -> cleaned, de-duped customer code list.
    Uppercased since every real cacus.xcus code in this ERP is uppercase
    (e.g. "CUS-005596") — guards against a manual paste with stray case."""
    if not text:
        return []
    seen = set()
    out = []
    for raw in text.split(","):
        code = raw.strip().upper()
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def load_watchlist(zid: str) -> dict:
    """{"collection": [...], "returns": [...]} for one ZID — customer codes
    are only unique within one ZID, so scoped per ZID like every other
    watchlist in this app."""
    data = _load_json(_WATCHLIST_FILE)
    zid_data = data.get(str(zid), {})
    return {g: list(zid_data.get(g, [])) for g in GROUPS}


def save_group(zid: str, group: str, codes: list) -> None:
    if group not in GROUPS:
        raise ValueError(f"Unknown watchlist group: {group!r}")
    data = _load_json(_WATCHLIST_FILE)
    zid_key = str(zid)
    zid_data = dict(data.get(zid_key, {}))
    zid_data[group] = list(codes)
    data[zid_key] = zid_data
    _save_json(_WATCHLIST_FILE, data)


def load_watchlist_pooled(zids: list) -> dict:
    """Union of each given ZID's own saved watchlist — results are always
    pooled 100001+100000 (per explicit ask), but setup stays per-ZID
    (whichever business happens to be active on the Commissions page when
    an admin adds a code) so an entry made under either ZID's sidebar
    selection counts. De-duped, order preserved (first-seen zid wins)."""
    out = {g: [] for g in GROUPS}
    for zid in zids:
        wl = load_watchlist(zid)
        for g in GROUPS:
            for code in wl.get(g, []):
                if code not in out[g]:
                    out[g].append(code)
    return out


# ── Collection Watchlist: overdue-balance row highlight ─────────────────────

def style_overdue_balance(report_df: pd.DataFrame, days_threshold: int = 3, today=None):
    """Light-red row highlight where Current Balance is non-zero AND more
    than `days_threshold` days have passed since the customer's own Sales
    Date — "pending balance greater than 3 days [since the] sale," per
    explicit ask. Expects report_df shaped like
    salesman_due.build_latest_sale_collection_report's own output
    ("Sales Date", "Current Balance" columns)."""
    if report_df is None or report_df.empty:
        return report_df
    if "Sales Date" not in report_df.columns or "Current Balance" not in report_df.columns:
        return report_df
    ref = pd.Timestamp.today().normalize() if today is None else today
    sales_date = pd.to_datetime(report_df["Sales Date"], errors="coerce")
    balance = pd.to_numeric(report_df["Current Balance"], errors="coerce").fillna(0.0)
    days_since = (ref - sales_date).dt.days
    mask = (balance.abs() > 0.01) & (days_since > days_threshold)
    return _style_rows(report_df, mask)


# ── Returns Watchlist: next-working-day-return candidate detection ──────────

def compute_next_day_return_matches(
    raw_df: pd.DataFrame, holidays: set, window_start: pd.Timestamp, window_end: pd.Timestamp
) -> pd.DataFrame:
    """Filters core/queries.py::get_do_next_day_returns' raw (DO, SR) pairs
    down to genuine "returned on the very next working day" candidates —
    customers who refuse delivery on the spot, posted into the ERP a day
    late. DOs are scoped to [window_start, window_end]; a pair matches when
    its SR date equals processing.holidays.next_working_day(do_date) —
    Thursday DOs correctly expect a Saturday return (skipping Friday), every
    other weekday expects the very next calendar day. Confirmed against real
    data (2026-10-12): 422 same-trip-looking pairs at a 1-day gap spanning
    every weekday except Thursday, and 85 of 87 2-day-gap pairs are
    Thursday-dated DOs — exactly this pattern, not noise.
    """
    cols = ["zid", "xdornum", "do_date", "xcrnnum", "sr_date", "cusid", "cusname",
            "return_amount", "day_gap"]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=cols)

    d = raw_df.copy()
    d["do_date"] = pd.to_datetime(d["do_date"], errors="coerce")
    d["sr_date"] = pd.to_datetime(d["sr_date"], errors="coerce")
    d = d[d["do_date"].notna() & d["sr_date"].notna()]
    d = d[(d["do_date"] >= window_start) & (d["do_date"] <= window_end)]
    if d.empty:
        return pd.DataFrame(columns=cols)

    d["expected_next"] = pd.to_datetime(
        d["do_date"].dt.date.map(lambda x: next_working_day(x, holidays))
    )
    d = d[d["sr_date"].dt.normalize() == d["expected_next"]].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)

    d["day_gap"] = (d["sr_date"] - d["do_date"]).dt.days
    d["cusid"] = d["cusid"].astype(str)
    d["return_amount"] = pd.to_numeric(d["return_amount"], errors="coerce").fillna(0.0)
    return d.sort_values("do_date", ascending=False).reset_index(drop=True)[cols]


def style_watchlist_rows(df: pd.DataFrame, watchlist_codes: list, customer_col: str = "cusid"):
    """Light-red row highlight for whichever rows' customer code is on the
    given watchlist — used so the Returns Watchlist's full candidate list
    (every customer matching the next-working-day-return pattern) visibly
    marks which ones the manager already flagged, without hiding the rest."""
    if df is None or df.empty or not watchlist_codes or customer_col not in df.columns:
        return df
    codes = {str(c) for c in watchlist_codes}
    mask = df[customer_col].astype(str).isin(codes)
    return _style_rows(df, mask)


# ── Promised Payment list ────────────────────────────────────────────────────

def filter_promised_payment_window(report_df: pd.DataFrame, months_back: int = 3, today=None) -> pd.DataFrame:
    """Rows from the pooled Latest Sale & Collection report that have a
    "Promised Payment" date set within the trailing `months_back` months.
    That column already comes pre-gated by salesman_due.
    merge_delivery_payment_promise (NaT whenever a newer sale has happened
    since the promise), so "no sales after the [promise's underlying] sales
    date" is already built in — nothing extra to check here."""
    if report_df is None or report_df.empty or "Promised Payment" not in report_df.columns:
        return pd.DataFrame()
    ref = pd.Timestamp.today().normalize() if today is None else today
    cutoff = ref - pd.DateOffset(months=months_back)
    d = report_df.copy()
    d["Promised Payment"] = pd.to_datetime(d["Promised Payment"], errors="coerce")
    d = d[d["Promised Payment"].notna() & (d["Promised Payment"] >= cutoff)]
    return d.sort_values("Promised Payment", ascending=False).reset_index(drop=True)


def style_promised_payment_overdue(df: pd.DataFrame, today=None):
    """Light-red row highlight where the Promised Payment date has already
    passed AND a balance is still outstanding — "red colored if there is a
    balance and date has passed," per explicit ask."""
    if df is None or df.empty:
        return df
    if "Promised Payment" not in df.columns or "Current Balance" not in df.columns:
        return df
    ref = pd.Timestamp.today().normalize() if today is None else today
    promised = pd.to_datetime(df["Promised Payment"], errors="coerce")
    balance = pd.to_numeric(df["Current Balance"], errors="coerce").fillna(0.0)
    mask = promised.notna() & (promised < ref) & (balance.abs() > 0.01)
    return _style_rows(df, mask)


def _style_rows(df: pd.DataFrame, mask: pd.Series):
    def _row_style(row):
        return [_FLAG_STYLE] * len(row) if bool(mask.loc[row.name]) else [""] * len(row)
    return df.style.apply(_row_style, axis=1)
