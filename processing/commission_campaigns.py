# processing/commission_campaigns.py
# B.1/3/4 — Product-Specific / Stock Clearance / Slow-Moving commission
# campaigns. See commission_tracking_design.md §B.1/3/4 for the full spec.
#
# Architecture (confirmed 2026-09-19): `commission_campaigns` (Postgres) holds
# campaign DEFINITIONS only. Payout is computed LIVE every time a campaign is
# queried, from that row + existing sales/collection data — nothing computed
# is persisted. Salesman -> payout-group membership is likewise derived LIVE
# (confirmed 2026-09-20) from prmst.xdisease (a salesman's own current
# working area) + cacus.xstate (that area's zone/channel classification) —
# see build_area_group_map/derive_spid_group_map below. This replaced an
# earlier hand-maintained data/commission_payout_groups.json roster: group
# membership changes whenever a salesman's area changes on the ERP side
# (prmst), so deriving it live means there's nothing to keep in sync by hand.

from __future__ import annotations

import json
from collections import deque

import numpy as np
import pandas as pd

from core.db import execute_write, execute_write_returning, get_data
from processing.commissions import _prorate_before, _window_metrics


# ── Payout-group derivation (Cacus xstate + PRMST xdisease) ─────────────────
# Confirmed 2026-09-20. cacus.xstate (despite the "state" name) already
# encodes a sales-zone/channel classification per customer area (xcity) —
# used elsewhere in the app as a Retail/District split
# (processing/ar_analysis.py's "Market" column). prmst.xdisease (despite the
# name) holds a salesman's own current working area(s), comma-separated.
# Matching the two derives each salesman's payout group live, at query time.

_SYLHET_RETAIL = "sylhet retail"


def build_area_group_map() -> dict:
    """{xcity (lowercased) -> payout group}, derived from cacus.xstate. Group
    names are the real xstate values themselves — "Dhaka retail", "District",
    "Dhaka General", "Nawab pur", "Kawranbazar", "Alubazar", "Imamgonj",
    "Rahima enterprise" as of 2026-09-20 — NOT collapsed to a plain
    Dhaka/District binary, confirmed by the user. The one fold: "Sylhet
    Retail" counts as "District" (it says "retail" but Sylhet isn't Dhaka,
    and a collection deadline is about physical distance, not sales
    channel). Per xcity, takes the majority xstate value across pooled
    100001+100000 (same field sales team/territories) — a handful of stray
    mistagged rows per area (confirmed on real data, e.g. Chittagong is
    "District" in 371/373 rows) don't change the winner."""
    from core.analytics import Analytics

    frames = []
    for zid in ("100001", "100000"):
        df = Analytics("cacus_master", zid=zid, filters={}).data
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return {}

    d = pd.concat(frames, ignore_index=True)
    d["xcity"] = d["xcity"].astype(str).str.strip()
    d["xstate"] = d["xstate"].astype(str).str.strip()
    d = d[(d["xcity"] != "") & (d["xstate"] != "") & (d["xcity"].str.lower() != "nan") & (d["xstate"].str.lower() != "nan")]
    if d.empty:
        return {}

    winners = (
        d.groupby(["xcity", "xstate"]).size().rename("n").reset_index()
         .sort_values("n", ascending=False)
         .drop_duplicates("xcity")
    )
    return {
        row.xcity.lower(): ("District" if row.xstate.lower() == _SYLHET_RETAIL else row.xstate)
        for row in winners.itertuples()
    }


def derive_spid_group_map(area_group_map: dict | None = None, valid_spids: set | None = None) -> dict:
    """{spid: group}, derived live from prmst.xdisease (the salesman's own
    comma-separated area list) matched against build_area_group_map(). A
    salesman whose listed areas span more than one group takes the majority
    group; a tie is broken by whichever tied group's area is listed first.
    A salesman with a blank xdisease, or none of whose listed areas resolve
    to a known group, is simply left out of the returned dict — downstream
    this is handled exactly like an unassigned salesman always was
    (excluded with reason "salesman not in any payout group").

    `valid_spids`, if given, restricts the result to real salesmen only —
    per CLAUDE.md's "Key column mappings" rule, a person is a real,
    currently-active salesman if they appear in the op* tables (real sales
    activity), not merely by having a prmst row with an xemp code. Without
    this, prmst also has non-salesman staff (accounts, admin, etc.) who
    happen to have an area on file, and they'd otherwise show up in the
    derived roster despite never actually appearing on a DO.

    **Active-employment filter — confirmed 2026-09-20: only currently-active
    employees count.** `prmst.xstatusemp` is checked against `'A-Active'`
    (case-insensitive) — an employee who has resigned/been terminated/is on
    hold (other real values: `R-Resigned`, `T-Terminated`, `H-Hold`,
    `D-Dismissed`) is excluded here even if `valid_spids` would otherwise
    have let them through (e.g. they made real sales while still employed,
    long before leaving). This is deliberately checked in ADDITION to
    `valid_spids`, not instead of it — `valid_spids` answers "is this really
    a salesman," this answers "are they still one today." A DO whose
    salesman drops out for this reason is excluded downstream the same way
    any unassigned salesman already was ("salesman not in any payout
    group").

    CAVEAT (2026-09-20, see CLAUDE.md "Key column mappings"): both
    prmst.xdisease AND prmst.xstatusemp are mid-rollout on the live server
    — the local Postgres mirror doesn't reflect real data for either yet.
    This function is correct against real data but won't resolve much
    locally until that rollout finishes.
    """
    from core.analytics import Analytics

    if area_group_map is None:
        area_group_map = build_area_group_map()
    if not area_group_map:
        return {}

    frames = []
    for zid in ("100001", "100000"):
        df = Analytics("prmst_area", zid=zid, filters={}).data
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return {}

    d = pd.concat(frames, ignore_index=True).dropna(subset=["spid"]).drop_duplicates("spid")
    if valid_spids is not None:
        d = d[d["spid"].astype(str).isin({str(s) for s in valid_spids})]
    d = d[d["status"].astype(str).str.strip().str.upper() == "A-ACTIVE"]

    out = {}
    for r in d.itertuples():
        raw = str(r.area or "").strip()
        if not raw or raw.lower() == "nan":
            continue
        areas = [a.strip() for a in raw.split(",") if a.strip()]
        groups = [area_group_map[a.lower()] for a in areas if a.lower() in area_group_map]
        if not groups:
            continue
        counts = pd.Series(groups).value_counts()
        top = counts.max()
        out[str(r.spid)] = next(g for g in groups if counts[g] == top)
    return out


# ── commission_campaigns (Postgres) — CRUD ──────────────────────────────────

def create_campaign(
    campaign_name: str, campaign_type: str, recipient_type: str, product_rates: dict,
    cap: float | None, window_start, window_end, payout_groups: dict,
    uptick_baseline_months: int, created_by: str, notes: str = "",
) -> int | None:
    """Inserts a new campaign row, returns its id (None on failure).

    `recipient_type`: "salesman" or "customer" — who the commission is paid
    to. Confirmed 2026-09-19: the underlying payment/date/collection-deadline
    structure is identical either way; only who receives the payout differs.
    `product_rates`: {itemcode: float} — rate is the per-unit
    INCENTIVE/discount amount (e.g. 5 or 3 BDT), not the product's sales
    price, and varies per product.
    `cap`: ONE ceiling for the whole campaign (or None for no cap) — caps a
    single recipient's TOTAL payout, summed across every product in this
    campaign. Confirmed 2026-09-20 (same day as the recipient_type/
    per-product-rate correction above): the cap is campaign-wide, not
    per-product — an earlier version of this function had it per product,
    which was a misunderstanding, corrected same-day.
    """
    sql = """
        INSERT INTO commission_campaigns
            (campaign_name, campaign_type, recipient_type, product_rates, cap,
             window_start, window_end, payout_groups, uptick_baseline_months,
             created_by, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """
    params = (
        campaign_name, campaign_type, recipient_type, json.dumps(product_rates), cap,
        window_start, window_end, json.dumps(payout_groups),
        uptick_baseline_months, created_by, notes,
    )
    row = execute_write_returning(sql, params)
    return row[0] if row else None


def list_campaigns() -> pd.DataFrame:
    sql = """
        SELECT id, campaign_name, campaign_type, recipient_type, product_rates, cap,
               window_start, window_end, payout_groups, uptick_baseline_months,
               created_by, created_at, notes
        FROM commission_campaigns
        ORDER BY created_at DESC
    """
    records, cols = get_data(sql)
    if not records:
        return pd.DataFrame(columns=cols or [])
    df = pd.DataFrame(records, columns=cols)
    df["product_rates"] = df["product_rates"].apply(lambda v: v if isinstance(v, dict) else json.loads(v))
    df["payout_groups"] = df["payout_groups"].apply(lambda v: v if isinstance(v, dict) else json.loads(v))
    df["cap"] = pd.to_numeric(df["cap"], errors="coerce")
    return df


def get_campaign(campaign_id: int) -> dict | None:
    df = list_campaigns()
    if df.empty:
        return None
    row = df[df["id"] == campaign_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def update_campaign(
    campaign_id: int, campaign_name: str, campaign_type: str, recipient_type: str,
    product_rates: dict, cap: float | None, window_start, window_end, payout_groups: dict,
    uptick_baseline_months: int, notes: str = "",
) -> bool:
    """Updates an existing campaign's definition in place (id/created_by/
    created_at untouched) — added 2026-09-20 so a setup mistake (e.g. wrong
    window dates) can be fixed directly instead of delete+recreate. Safe to
    edit freely: nothing about a campaign is ever pre-computed (payout is
    always live off the row + current sales/collection data), so there's no
    stale derived data anywhere to invalidate."""
    sql = """
        UPDATE commission_campaigns
        SET campaign_name = %s, campaign_type = %s, recipient_type = %s,
            product_rates = %s, cap = %s, window_start = %s, window_end = %s,
            payout_groups = %s, uptick_baseline_months = %s, notes = %s
        WHERE id = %s
    """
    params = (
        campaign_name, campaign_type, recipient_type, json.dumps(product_rates), cap,
        window_start, window_end, json.dumps(payout_groups), uptick_baseline_months,
        notes, campaign_id,
    )
    return execute_write(sql, params)


def delete_campaign(campaign_id: int) -> bool:
    return execute_write("DELETE FROM commission_campaigns WHERE id = %s", (campaign_id,))


# ── A.2 — Highest Product Sales (ranked) ────────────────────────────────────
# Confirmed 2026-09-20: reuses the SAME `commission_campaigns` table as
# B.1/3/4 — explicit ask, "I don't want to create a different table for
# every campaign... designate data to the right columns if possible, if not
# just use the JSONB column that's there." `campaign_name`/`recipient_type`/
# `window_start`/`window_end` are reused as-is (same meaning as for B.1/3/4);
# `product_rates` (JSONB) is repurposed to hold this ranking's own config
# instead of per-product rates: {"num_winners": N, "payouts_by_rank":
# [amt_rank1, amt_rank2, ...]}. `cap`/`payout_groups`/`uptick_baseline_months`
# are unused for these rows (no cap concept, no collection-deadline groups).
# `campaign_type` (a free-text label, "just a UI category" for B.1/3/4's own
# 3 values) now also doubles as the discriminator between mechanisms —
# views/commission_rankings_view.py filters commission_campaigns rows by
# `campaign_type == "Highest Product Sales"` so B.1/3/4's own campaign
# picker never sees these rows and vice versa.

def compute_highest_product_sales_ranking(campaign: dict, sales_df: pd.DataFrame) -> dict:
    """A.2: ranks recipients (salesman or customer, per `recipient_type`) by
    DISTINCT product count sold within the campaign's window — breadth, not
    volume, confirmed 2026-09-20 — and pays a fixed BDT amount per rank
    position to the top `num_winners` (2-5, admin-set). Both the winner
    count and each rank's own payout amount are read from `product_rates`
    (see module note above).

    Ties (identical distinct-product-count) are broken by total quantity
    sold, then by recipient code for a fully deterministic order — this
    tie-break isn't an explicit rule from the user, it's an assumption
    flagged here for visibility, not confirmed.

    Returns `{"recipient_type", "ranking" (DataFrame: rank, recipient,
    recipient_name, distinct_products, total_qty, payout — payout is 0 for
    every non-winning row, never blank/NaN), "num_winners", "total_payout"}`.
    """
    recipient_type = campaign.get("recipient_type") or "salesman"
    window_start, window_end = campaign["window_start"], campaign["window_end"]
    config = campaign.get("product_rates") or {}
    payouts_by_rank = [float(v) for v in config.get("payouts_by_rank", [])]
    num_winners = int(config.get("num_winners") or len(payouts_by_rank))

    recip_key = "spid" if recipient_type == "salesman" else "cusid"
    recip_name_key = "spname" if recipient_type == "salesman" else "cusname"

    empty = pd.DataFrame(columns=["rank", "recipient", "recipient_name", "distinct_products", "total_qty", "payout"])
    if sales_df is None or sales_df.empty or recip_key not in sales_df.columns:
        return {"recipient_type": recipient_type, "ranking": empty, "num_winners": num_winners, "total_payout": 0.0}

    d = sales_df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    start, end = pd.Timestamp(window_start), pd.Timestamp(window_end)
    d = d[(d["date"] >= start) & (d["date"] <= end)]
    if d.empty:
        return {"recipient_type": recipient_type, "ranking": empty, "num_winners": num_winners, "total_payout": 0.0}

    d["itemcode"] = d["itemcode"].astype(str)
    d["quantity"] = pd.to_numeric(d["quantity"], errors="coerce")

    grouped = (
        d.groupby(recip_key)
         .agg(
             recipient_name=(recip_name_key, "first"),
             distinct_products=("itemcode", "nunique"),
             total_qty=("quantity", "sum"),
         )
         .reset_index()
         .rename(columns={recip_key: "recipient"})
    )
    grouped = grouped.sort_values(
        ["distinct_products", "total_qty", "recipient"], ascending=[False, False, True]
    ).reset_index(drop=True)
    grouped["rank"] = grouped.index + 1
    grouped["payout"] = 0.0
    for i in range(min(num_winners, len(payouts_by_rank), len(grouped))):
        grouped.loc[i, "payout"] = payouts_by_rank[i]

    total_payout = float(grouped["payout"].sum())
    return {
        "recipient_type": recipient_type,
        "ranking": grouped[["rank", "recipient", "recipient_name", "distinct_products", "total_qty", "payout"]],
        "num_winners": num_winners,
        "total_payout": total_payout,
    }


# ── Pooled FIFO DO/collection resolver ──────────────────────────────────────

def build_do_totals(sales_df: pd.DataFrame) -> pd.DataFrame:
    """One row per DO voucher: cusid, cusname, spid, spname, date, amount
    (= sum of final_sales across the DO's line items). Names are carried
    along (not just codes) so the payout UI can show a readable recipient
    either way (recipient_type can be salesman OR customer — confirmed
    2026-09-19). `sales_df` must already have `final_sales` (via
    processing.common.data_copy_add_columns)."""
    cols = ["voucher", "cusid", "cusname", "spid", "spname", "date", "amount"]
    if sales_df is None or sales_df.empty:
        return pd.DataFrame(columns=cols)
    d = sales_df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    grouped = (
        d.groupby("voucher", as_index=False)
         .agg(
             cusid=("cusid", "first"),
             cusname=("cusname", "first"),
             spid=("spid", "first"),
             spname=("spname", "first"),
             date=("date", "first"),
             amount=("final_sales", "sum"),
         )
    )
    return grouped[grouped["amount"] > 0].reset_index(drop=True)


def build_collection_totals(collection_df: pd.DataFrame) -> pd.DataFrame:
    """One row per collection voucher: cusid, date, amount."""
    if collection_df is None or collection_df.empty:
        return pd.DataFrame(columns=["glvoucher", "cusid", "date", "amount"])
    d = collection_df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["amount"] = pd.to_numeric(d["value"], errors="coerce").fillna(0.0)
    return d[["glvoucher", "cusid", "date", "amount"]][d["amount"] > 0].reset_index(drop=True)


def resolve_do_paid_dates(do_totals: pd.DataFrame, collection_totals: pd.DataFrame) -> pd.DataFrame:
    """Pooled (100001+100000), per-customer, full-history, chronological FIFO
    walk — all-or-nothing per DO. For each customer: DOs are a FIFO queue of
    "amount owed"; each collection pays off the OLDEST open DO(s) first, in
    full, before moving to the next. A DO's `paid_date` is the date its
    remaining balance first reaches zero. Any collection amount left over
    after every open DO is paid becomes a prepaid credit pool, applied to
    that customer's next DO(s) as they appear in the chronological walk
    (handles an advance payment that precedes the DO it's really for).

    Returns one row per DO voucher: voucher, cusid, spid, date, amount, paid_date
    (NaT if never fully paid as of the data's own history).
    """
    if do_totals.empty:
        return do_totals.assign(paid_date=pd.NaT)

    do = do_totals.copy()
    coll = collection_totals.copy()

    paid_date = {v: pd.NaT for v in do["voucher"]}

    # Pre-group ONCE (a per-customer re-filter of the full collection frame
    # inside the loop below is O(customers x rows) and was measured to never
    # finish on real data — 5,000+ customers x tens of thousands of
    # collection rows). A single groupby pass + dict lookup is O(rows) total.
    do_groups = {cusid: g for cusid, g in do.groupby("cusid", sort=False)}
    coll_groups = {cusid: g for cusid, g in coll.groupby("cusid", sort=False)} if not coll.empty else {}

    for cusid, do_grp in do_groups.items():
        coll_grp = coll_groups.get(cusid)
        events = [
            (r.date, 0, "do", r.voucher, r.spid, float(r.amount))
            for r in do_grp.itertuples()
        ]
        if coll_grp is not None and not coll_grp.empty:
            events.extend(
                (r.date, 1, "coll", None, None, float(r.amount))
                for r in coll_grp.itertuples()
            )
        # DOs before collections on the same date (sort key 0 < 1) — a same-day
        # collection can pay off a same-day DO, not just earlier ones.
        events.sort(key=lambda e: (e[0], e[1]))

        queue = deque()  # [{"voucher":, "remaining":}]
        prepaid_pool = 0.0

        for date, _, etype, voucher, spid, amount in events:
            if etype == "do":
                remaining = amount
                if prepaid_pool > 0:
                    take = min(prepaid_pool, remaining)
                    prepaid_pool -= take
                    remaining -= take
                if remaining <= 1e-6:
                    paid_date[voucher] = date
                else:
                    queue.append({"voucher": voucher, "remaining": remaining})
            else:
                c = amount
                while c > 1e-6 and queue:
                    head = queue[0]
                    take = min(c, head["remaining"])
                    head["remaining"] -= take
                    c -= take
                    if head["remaining"] <= 1e-6:
                        paid_date[head["voucher"]] = date
                        queue.popleft()
                if c > 1e-6:
                    prepaid_pool += c

    out = do.copy()
    out["paid_date"] = out["voucher"].map(paid_date)
    return out


# ── 100001/100000 target gate ───────────────────────────────────────────────

def _months_spanned(window_start, window_end) -> list:
    start = pd.Timestamp(window_start).replace(day=1)
    end = pd.Timestamp(window_end).replace(day=1)
    months = []
    cur = start
    while cur <= end:
        months.append((cur.year, cur.month))
        cur = cur + pd.DateOffset(months=1)
    return months


def zid_target_sum(zid: str, window_start, window_end) -> float:
    """Sum of every individual salesman's target for this ZID across the
    calendar months the window touches — the confirmed source of "did this
    ZID hit its target" (commission_tracking_design.md §The 100001/100000
    gate) — explicitly NOT a new standalone company-level target value."""
    from views._tm_shared import _load_json as _load_tm_json, _TARGETS_FILE
    data = _load_tm_json(_TARGETS_FILE)
    months = {f"{y}-{m:02d}" for y, m in _months_spanned(window_start, window_end)}
    total = 0.0
    prefix = f"{zid}_"
    for key, value in data.items():
        if not key.startswith(prefix):
            continue
        month_part = key.rsplit("_", 1)[-1]
        if month_part in months:
            total += float(value or 0)
    return total


def zid_actual_sales(zid: str, sales_df: pd.DataFrame, window_start, window_end) -> float:
    """final_sales summed for one ZID within the window — compared against
    zid_target_sum for the gate. `sales_df` must have `zid` and `final_sales`
    (via processing.common.data_copy_add_columns) and already be pooled/whole
    (this filters to the one zid itself, not scoped ahead of time)."""
    if sales_df is None or sales_df.empty:
        return 0.0
    d = sales_df[sales_df["zid"].astype(str) == str(zid)].copy()
    if d.empty:
        return 0.0
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    start, end = pd.Timestamp(window_start), pd.Timestamp(window_end)
    d = d[(d["date"] >= start) & (d["date"] <= end)]
    return float(pd.to_numeric(d["final_sales"], errors="coerce").sum())


def check_gate(sales_df: pd.DataFrame, window_start, window_end) -> dict:
    """{"passed": bool, "100001": {"target":, "actual":, "hit": bool}, "100000": {...}}
    — confirmed: if EITHER zid individually misses its own target, the whole
    campaign's commission isn't paid out at all."""
    result = {}
    for zid in ("100001", "100000"):
        target = zid_target_sum(zid, window_start, window_end)
        actual = zid_actual_sales(zid, sales_df, window_start, window_end)
        result[zid] = {"target": target, "actual": actual, "hit": actual >= target}
    result["passed"] = all(result[z]["hit"] for z in ("100001", "100000"))
    return result


# ── Sales-uptick companion metric ───────────────────────────────────────────

def compute_uptick(sales_df: pd.DataFrame, product_codes: list, window_start, window_end, baseline_months: int) -> dict:
    """Campaign-window sales of the picked product(s) vs. their previous
    average (baseline_months trailing the window's own start) — an
    analytical companion answering "did the campaign lift sales," separate
    from the payout math. Not a gate on payout."""
    if sales_df is None or sales_df.empty or not product_codes:
        return {"campaign_qty": 0.0, "campaign_revenue": 0.0, "baseline_avg_qty": 0.0,
                "baseline_avg_revenue": 0.0, "qty_uptick_pct": None, "revenue_uptick_pct": None}

    d = sales_df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["itemcode"] = d["itemcode"].astype(str)
    codes = {str(c) for c in product_codes}
    d = d[d["itemcode"].isin(codes)]

    start, end = pd.Timestamp(window_start), pd.Timestamp(window_end)
    campaign = d[(d["date"] >= start) & (d["date"] <= end)]
    campaign_qty = float(pd.to_numeric(campaign["quantity"], errors="coerce").sum())
    campaign_revenue = float(pd.to_numeric(campaign["final_sales"], errors="coerce").sum())

    baseline_start = start - pd.DateOffset(months=baseline_months)
    baseline_end = start - pd.Timedelta(days=1)
    baseline = d[(d["date"] >= baseline_start) & (d["date"] <= baseline_end)]
    n_months = max(baseline_months, 1)
    baseline_avg_qty = float(pd.to_numeric(baseline["quantity"], errors="coerce").sum()) / n_months
    baseline_avg_revenue = float(pd.to_numeric(baseline["final_sales"], errors="coerce").sum()) / n_months

    def _pct(cur, base):
        if base <= 0:
            return None
        return (cur - base) / base * 100.0

    return {
        "campaign_qty": campaign_qty, "campaign_revenue": campaign_revenue,
        "baseline_avg_qty": baseline_avg_qty, "baseline_avg_revenue": baseline_avg_revenue,
        "qty_uptick_pct": _pct(campaign_qty, baseline_avg_qty),
        "revenue_uptick_pct": _pct(campaign_revenue, baseline_avg_revenue),
    }


def compute_campaign_product_before_after(
    sales_df: pd.DataFrame, returns_df: pd.DataFrame, product_codes: list,
    window_start, window_end, baseline_months: int, items_meta: dict,
) -> list:
    """Per-product Before/After for a campaign's own picked product(s) —
    added 2026-09-20, explicit ask: "in a secondary table, how about we see
    the before after, like you do in product tracking." Same shared engine
    as processing/commissions.py::build_product_comparisons
    (_window_metrics/_prorate_before), different window source:

      Before = the baseline window compute_uptick already uses (baseline_months
               trailing window_start)
      After  = the campaign's own sales window, capped to today if the
               campaign is still ongoing (window_end >= today) so an
               in-progress campaign's After reflects what's actually
               elapsed so far, not padded with future zeros — mirrors
               Product Tracking's own "After runs to today" framing.

    Returns the same per-product dict shape build_product_comparisons does,
    so both feed views/commission_shared.py::render_before_after_table.
    `items_meta` = {itemcode: {"item_name":, "item_group":, "stock":}} (a
    combined-catalog lookup dict, built by the caller from
    views/commission_campaigns_view.py::_load_combined_item_catalog()).
    """
    if not product_codes:
        return []

    today = pd.Timestamp.today().normalize()
    window_start_ts = pd.Timestamp(window_start)
    window_end_ts = pd.Timestamp(window_end)
    after_end = min(window_end_ts, today)
    baseline_start = window_start_ts - pd.DateOffset(months=baseline_months)

    s = pd.DataFrame()
    if sales_df is not None and not sales_df.empty and {"itemcode", "date"}.issubset(sales_df.columns):
        s = sales_df.copy()
        s["itemcode"] = s["itemcode"].astype(str)
        s["_dt"] = pd.to_datetime(s["date"], errors="coerce")
        s["quantity"] = pd.to_numeric(s["quantity"], errors="coerce")
        s["final_sales"] = pd.to_numeric(s["final_sales"], errors="coerce")

    r = pd.DataFrame()
    if returns_df is not None and not returns_df.empty and {"itemcode", "date"}.issubset(returns_df.columns):
        r = returns_df.copy()
        r["itemcode"] = r["itemcode"].astype(str)
        r["_dt"] = pd.to_datetime(r["date"], errors="coerce")
        r["returnqty"] = pd.to_numeric(r["returnqty"], errors="coerce")
        if "treturnamt" in r.columns:
            r["treturnamt"] = pd.to_numeric(r["treturnamt"], errors="coerce")

    before_days = (window_start_ts - baseline_start).days
    after_days = (after_end - window_start_ts).days + 1
    after_end_excl = after_end + pd.Timedelta(days=1)

    rows = []
    for itemcode in product_codes:
        before = _window_metrics(s, r, itemcode, baseline_start, window_start_ts, revenue_col="final_sales")
        after = _window_metrics(s, r, itemcode, window_start_ts, after_end_excl, revenue_col="final_sales")
        before_avg_prorated = _prorate_before(before, before_days, after_days)

        meta = items_meta.get(itemcode, {})
        rows.append({
            "itemcode": itemcode,
            "itemname": meta.get("item_name"),
            "itemgroup": meta.get("item_group"),
            "current_stock": meta.get("stock"),
            "before_days": before_days,
            "after_days": after_days,
            "before": before,
            "before_avg_prorated": before_avg_prorated,
            "after": after,
        })
    return rows


# ── Payout computation ──────────────────────────────────────────────────────

def compute_campaign_payout(
    campaign: dict, sales_line_items: pd.DataFrame, resolved_do: pd.DataFrame,
    spid_group_map: dict,
) -> dict:
    """The full B.1/3/4 payout for one campaign, computed live — nothing here
    is persisted. Returns a dict with a single unified line-level table
    (`line_items`, see below), the per-recipient totals, the gate result,
    and the uptick metric.

    `recipient_type` ("salesman" or "customer", confirmed 2026-09-19) only
    changes WHO the payout is grouped/attributed to — DO eligibility (fully
    collected, by which deadline) is unchanged either way and always keyed
    off the DO's own salesman's payout group, since the deadline is a
    logistics/territory concept, not a recipient concept.

    `product_rates` = {itemcode: float} — rate is the per-unit
    incentive/discount amount (confirmed 2026-09-19, e.g. 5 or 3 BDT — NOT
    the product's sales price), and varies per product.

    `cap` (on the campaign dict, may be None) — ONE ceiling for the whole
    campaign, confirmed 2026-09-20: caps a single recipient's TOTAL payout,
    summed across every product in the campaign — NOT a per-product cap
    (an earlier version had it per-product; corrected same day). Applied at
    the recipient_total level below, after summing every product's payout
    for that recipient.

    `spid_group_map` = {spid: group}, from derive_spid_group_map() — derived
    live from prmst.xdisease + cacus.xstate (confirmed 2026-09-20), not a
    hand-maintained roster.

    `line_items` — added 2026-09-20, replacing the earlier separate
    `recipient_product`/`excluded` pieces, per explicit ask: "I want to see
    what salesmen were removed and what not," in ONE table, not a separate
    excluded-DOs table. One row per (voucher, recipient, itemcode) for EVERY
    in-window DO that carries at least one picked product — eligible AND
    excluded alike — with a `status` column ("Eligible" or the specific
    exclusion reason) and `payout` (quantity × rate for Eligible rows, 0 for
    excluded ones — never a phantom uncapped number for a DO that isn't
    actually being paid).
    """
    product_rates = campaign["product_rates"]
    product_codes = list(product_rates.keys())
    cap = campaign.get("cap")
    cap = float(cap) if cap not in (None, "") and not pd.isna(cap) else None
    recipient_type = campaign.get("recipient_type") or "salesman"
    window_start, window_end = campaign["window_start"], campaign["window_end"]
    groups_deadlines = campaign["payout_groups"]  # {"Dhaka retail": "2026-10-15", ...}
    spid_group = spid_group_map

    start, end = pd.Timestamp(window_start), pd.Timestamp(window_end)
    do = resolved_do.copy()
    do["date"] = pd.to_datetime(do["date"], errors="coerce")
    in_window = do[(do["date"] >= start) & (do["date"] <= end)].copy()

    # Status for EVERY in-window DO, not just excluded ones — so the unified
    # line table below can show every DO's outcome in one place.
    status_map = {}
    for r in in_window.itertuples():
        group = spid_group.get(r.spid)
        if group is None:
            status_map[r.voucher] = "Excluded: salesman not in any payout group"
        elif not groups_deadlines.get(group):
            status_map[r.voucher] = f"Excluded: group '{group}' has no deadline set on this campaign"
        elif pd.isna(r.paid_date):
            status_map[r.voucher] = "Excluded: not yet fully collected"
        elif pd.Timestamp(r.paid_date) > pd.Timestamp(groups_deadlines[group]):
            status_map[r.voucher] = f"Excluded: collected after the {group} deadline ({groups_deadlines[group]})"
        else:
            status_map[r.voucher] = "Eligible"

    # Recipient identity (both salesman and customer id/name) for EVERY
    # in-window DO — carried along regardless of recipient_type, and
    # regardless of eligibility, so an excluded DO's recipient is still
    # visible in the unified table.
    recip_key = "spid" if recipient_type == "salesman" else "cusid"
    recip_name_key = "spname" if recipient_type == "salesman" else "cusname"
    voucher_meta = (
        in_window[["voucher", recip_key, recip_name_key]]
        .rename(columns={recip_key: "recipient", recip_name_key: "recipient_name"})
    )
    voucher_meta["status"] = voucher_meta["voucher"].map(status_map)

    # Every in-window DO's picked-product lines, eligible or not — a DO with
    # none of the picked products never appears here at all, matching "the
    # DOs of that product" scope.
    li = sales_line_items.copy()
    li["itemcode"] = li["itemcode"].astype(str)
    li = li[li["voucher"].isin(set(in_window["voucher"])) & li["itemcode"].isin(set(product_codes))]
    li = li.merge(voucher_meta, on="voucher", how="left")

    line_detail = pd.DataFrame(columns=["voucher", "recipient", "recipient_name", "status", "itemcode", "quantity"])
    if not li.empty:
        line_detail = (
            li.groupby(["voucher", "recipient", "recipient_name", "status", "itemcode"], as_index=False)
              .agg(quantity=("quantity", "sum"))
        )

    rates_df = pd.DataFrame(
        [{"itemcode": code, "rate": float(rate)} for code, rate in product_rates.items()]
    )

    line_items = pd.DataFrame(columns=["voucher", "recipient", "recipient_name", "status", "itemcode", "quantity", "rate", "payout"])
    if not line_detail.empty:
        line_items = line_detail.merge(rates_df, on="itemcode", how="left")
        line_items["quantity"] = pd.to_numeric(line_items["quantity"], errors="coerce")
        line_items["payout"] = line_items["quantity"] * line_items["rate"]
        # Payout is only real for Eligible lines — an excluded DO shows 0,
        # never an uncapped number for a DO that isn't actually being paid.
        line_items.loc[line_items["status"] != "Eligible", "payout"] = 0.0

    # recipient_total / total_payout: eligible lines only. The single
    # campaign-wide cap (if set) applies once here, to a recipient's TOTAL
    # across every product — never per line/per product.
    eligible_lines = line_items[line_items["status"] == "Eligible"] if not line_items.empty else line_items
    recipient_total = pd.DataFrame(columns=["recipient", "recipient_name", "raw_total_payout", "total_payout"])
    if not eligible_lines.empty:
        recipient_total = (
            eligible_lines.groupby(["recipient", "recipient_name"], as_index=False)
                          .agg(raw_total_payout=("payout", "sum"))
        )
        cap_upper = cap if cap is not None else float("inf")
        recipient_total["total_payout"] = recipient_total["raw_total_payout"].clip(upper=cap_upper)

    gate = check_gate(sales_line_items, window_start, window_end)
    uptick = compute_uptick(sales_line_items, product_codes, window_start, window_end, campaign["uptick_baseline_months"])

    total_payout = float(recipient_total["total_payout"].sum()) if not recipient_total.empty else 0.0

    # DO counts scoped to DOs that actually carry a picked product (matching
    # what line_items shows), not every DO in the window regardless of
    # product — the original scoping (any DO in window) is still what drives
    # the gate above, unrelated to this count.
    relevant_vouchers = set(line_items["voucher"].unique()) if not line_items.empty else set()
    eligible_do_count = sum(1 for v in relevant_vouchers if status_map.get(v) == "Eligible")
    excluded_do_count = len(relevant_vouchers) - eligible_do_count

    return {
        "recipient_type": recipient_type,
        "cap": cap,
        "gate": gate,
        "gate_passed": gate["passed"],
        "total_payout_if_gate_passed": total_payout,
        "total_payout": total_payout if gate["passed"] else 0.0,
        "recipient_total": recipient_total,
        "line_items": line_items,
        "eligible_do_count": eligible_do_count,
        "excluded_do_count": excluded_do_count,
        "uptick": uptick,
    }


# ── A.1 — Best Performer (ranked) ───────────────────────────────────────────
# Confirmed 2026-09-20: a 3rd campaign_type sharing the same commission_campaigns
# table (same reuse-the-columns/JSONB pattern as A.2 above) — `product_rates`
# holds {"num_months": M (1-3), "num_winners": N (2-10), "payouts_by_rank":
# [amt_rank1, ...]}. Salesman-only (no recipient_type toggle) — a deliberate
# deviation from the design doc's "for future note" table, since the scoring
# formula's own inputs (target achievement, collection %, AR balance) don't
# have a customer-equivalent meaning. Explicitly required to reuse
# processing.salesman_score.compute_salesman_scores' exact existing weights
# (not the design doc's own "proposed" recalibrated ones) and the same
# 100001/100000 pooling + target gate as B.1/3/4/A.2.

def compute_best_performer_ranking(
    campaign: dict, monthly_scored_tables: list, pooled_sales_df: pd.DataFrame,
) -> dict:
    """Averages each salesman's `compute_salesman_scores` score across the
    campaign's configured number of months, ranks descending, and pays a
    fixed BDT amount per rank position to the top `num_winners` — same
    per-rank-payout mechanism as compute_highest_product_sales_ranking
    above, different scoring source (the existing Salesman Score engine,
    called via processing.salesman_score.build_pooled_monthly_scores, not
    reimplemented here).

    `monthly_scored_tables` — one build_pooled_monthly_scores() output
    DataFrame per evaluation month (1-3 of them, per `num_months` in
    product_rates), built by the CALLER — the view owns loading/pooling
    sales/returns/collection/AR data per calendar year, since a 3-month
    window can cross a year boundary and each month's own scored table
    needs its own year's data. A salesman missing from a given month's
    table (no sales activity that whole calendar year) is averaged over
    only the months they DO appear in — not penalized with a phantom 0 for
    a year they had no presence in at all.

    `pooled_sales_df` — pooled (100001+100000) sales for the SAME
    100001/100000 target gate B.1/3/4 and now A.1 both use (`check_gate`,
    confirmed 2026-09-20 the gate applies here too), evaluated over the
    campaign's own window_start/window_end (the full N-month evaluation
    span, set live by the view from `today` + `num_months` — window_start/
    window_end on the stored row are informational only, same as A.2).

    Returns `{"ranking": DataFrame[rank, recipient, recipient_name,
    avg_score, months_scored, payout], "num_winners", "num_months", "gate",
    "gate_passed", "total_payout_if_gate_passed", "total_payout"}` — same
    gate-zeroes-total-but-keeps-the-computed-figure convention as
    compute_campaign_payout.
    """
    config = campaign.get("product_rates") or {}
    num_months = int(config.get("num_months") or len(monthly_scored_tables) or 1)
    payouts_by_rank = [float(v) for v in config.get("payouts_by_rank", [])]
    num_winners = int(config.get("num_winners") or len(payouts_by_rank))
    window_start, window_end = campaign["window_start"], campaign["window_end"]

    empty = pd.DataFrame(columns=["rank", "recipient", "recipient_name", "avg_score", "months_scored", "payout"])
    gate = check_gate(pooled_sales_df, window_start, window_end)
    tables = [t for t in (monthly_scored_tables or []) if t is not None and not t.empty]
    if not tables:
        return {
            "ranking": empty, "num_winners": num_winners, "num_months": num_months,
            "gate": gate, "gate_passed": gate["passed"],
            "total_payout_if_gate_passed": 0.0, "total_payout": 0.0,
        }

    combined = pd.concat(
        [t[["spid", "spname", "score"]].assign(spid=t["spid"].astype(str)) for t in tables],
        ignore_index=True,
    )
    agg = (
        combined.groupby("spid")
                .agg(spname=("spname", "first"), avg_score=("score", "mean"), months_scored=("score", "count"))
                .reset_index()
    )
    agg["avg_score"] = agg["avg_score"].round(1)
    agg = agg.sort_values(["avg_score", "spid"], ascending=[False, True]).reset_index(drop=True)
    agg["rank"] = agg.index + 1
    agg["payout"] = 0.0
    for i in range(min(num_winners, len(payouts_by_rank), len(agg))):
        agg.loc[i, "payout"] = payouts_by_rank[i]

    raw_total = float(agg["payout"].sum())
    ranking = agg.rename(columns={"spid": "recipient", "spname": "recipient_name"})[
        ["rank", "recipient", "recipient_name", "avg_score", "months_scored", "payout"]
    ]
    return {
        "ranking": ranking, "num_winners": num_winners, "num_months": num_months,
        "gate": gate, "gate_passed": gate["passed"],
        "total_payout_if_gate_passed": raw_total,
        "total_payout": raw_total if gate["passed"] else 0.0,
    }


# ── Customer Best Performer (ranked) ────────────────────────────────────────
# New 2026-09-20 — not in the original commission_tracking_design.md scope, a
# customer-side sibling to A.1 requested directly: "same logic, everything...
# but this would follow the customer scoring that is already in marketing
# analysis." A 4th campaign_type sharing the same commission_campaigns table
# (same JSONB-reuse pattern as A.1/A.2), but with two deliberate departures
# from A.1, both confirmed with the user before building:
#
#   1. Pooled by ZID GROUP, not always all-3-separate and not always
#      pooled-like-A.1 either — corrected 2026-09-20, same day, right after
#      first shipping this as single-ZID-only. The user ran a real audit:
#      100001 (HMBR) and 100000 (GI Corporation) share the SAME cacus
#      customer code 99.9% of the time (~10 mismatches out of ~10,000, none
#      of them current customers) — "when a salesman goes to a customer,
#      the customer doesn't understand the difference between 100,000 and
#      100,001." So those two ARE pooled together for scoring, same as A.1
#      (salesmen) and B.1/3/4 already pool them, for the same underlying
#      reason (one shared field sales team/customer relationship). **100005
#      (Zepto) stays separate** — a genuinely independent consumer brand,
#      no shared codes, explicitly confirmed to stay its own group. See
#      `views/commission_customer_best_performer_view.py::_ZID_GROUPS` —
#      the single source of truth for this grouping. The campaign's own
#      zid GROUP (a list, not one zid) is chosen at setup (whichever
#      business is active then) and PINNED in product_rates — same "pin at
#      setup, don't drift" discipline as A.1's reporting month.
#   2. A reporting YEAR, not a reporting month, no month-averaging —
#      processing.marketing.build_customer_marketing_table (the existing
#      Customer Score engine, reused unmodified per the explicit ask) only
#      filters/aggregates by calendar YEAR (YoY growth, monthly activity
#      rate assuming a full 12-month year) — there's no clean way to cap it
#      to a partial month without changing its own internal logic, which
#      "follow the customer scoring that is already in marketing analysis"
#      explicitly meant NOT to do. Current year or later only, same
#      no-retroactive-setup rule as A.1's reporting month, applied at the
#      year level (see views/commission_customer_best_performer_view.py).
#
# No 100001/100000 target gate — that check is specific to the shared sales
# team between those two ZIDs; it doesn't generalize to a single-ZID
# mechanism that can also be used for 100005 (Zepto, no shared team at all).

def compute_customer_best_performer_ranking(campaign: dict, customer_score_df: pd.DataFrame) -> dict:
    """Ranks customers by their EXISTING Customer Score
    (processing.marketing.build_customer_marketing_table's own
    composite_score, unmodified) and pays out by RANK-BAND TIER, not a
    distinct amount per individual rank — confirmed 2026-09-20, same day,
    right after first shipping this with per-rank amounts: with up to 100
    winners, entering a separate BDT figure for every single rank was
    "hectic" (explicit ask). `product_rates["tiers"]` is a list of
    `{"start_rank", "end_rank", "amount", "gift"}` — a tier can be a single
    rank (start == end, for an individually-differentiated top prize) or a
    wide band (e.g. 11-50) sharing one reward. Each tier can carry a BDT
    `amount`, a physical `gift` (free text — "it doesn't necessarily have
    to be a cash prize... a mug or a pen or a cap"), or both.

    `customer_score_df` — the caller's own already-scoped, already-scored
    output of build_customer_marketing_table (one zid GROUP — 100001+100000
    pooled, or 100005 alone — one reporting year) — this function only
    ranks/pays off an existing `composite_score` column, it doesn't compute
    the score itself.

    Returns `{"ranking": DataFrame[rank, recipient, recipient_name,
    composite_score, payout, gift], "num_winners", "total_payout"}` — no
    gate, so total_payout (the sum of BDT `payout` only, gifts aren't
    priced) is always the raw computed figure, never zeroed.
    """
    config = campaign.get("product_rates") or {}
    tiers = config.get("tiers") or []
    num_winners = int(config.get("num_winners") or 0)

    empty = pd.DataFrame(columns=["rank", "recipient", "recipient_name", "composite_score", "payout", "gift"])
    if (
        customer_score_df is None or customer_score_df.empty
        or "composite_score" not in customer_score_df.columns
    ):
        return {"ranking": empty, "num_winners": num_winners, "total_payout": 0.0}

    d = customer_score_df.dropna(subset=["composite_score"]).copy()
    d["cusid"] = d["cusid"].astype(str)
    d = d.sort_values(["composite_score", "cusid"], ascending=[False, True]).reset_index(drop=True)
    d["rank"] = d.index + 1

    def _tier_for_rank(rank: int) -> tuple:
        if rank > num_winners:
            return 0.0, ""
        for t in tiers:
            if int(t.get("start_rank", 0)) <= rank <= int(t.get("end_rank", 0)):
                return float(t.get("amount", 0) or 0), str(t.get("gift", "") or "")
        return 0.0, ""

    payout_gift = d["rank"].apply(_tier_for_rank)
    d["payout"] = payout_gift.apply(lambda pg: pg[0])
    d["gift"] = payout_gift.apply(lambda pg: pg[1])

    total_payout = float(d["payout"].sum())
    ranking = d.rename(columns={"cusid": "recipient", "cusname": "recipient_name"})[
        ["rank", "recipient", "recipient_name", "composite_score", "payout", "gift"]
    ]
    return {"ranking": ranking, "num_winners": num_winners, "total_payout": total_payout}


# ── A.4 — App Usage Commission ──────────────────────────────────────────────
# New 2026-09-21 — resolves A.4's original "deferred by design" status
# (see commission_tracking_design.md §A.4): the user supplied the real
# mobile ERP API data map (mobile_order_api_data_map.md/.json) and, after
# discussion, specified 5 components with confirmed weights (4% + 24%×4 =
# 100%) and a THRESHOLD payout (not ranked, not tiered) — a genuinely
# different mechanism from A.1/A.1b/A.2's rank-based payouts, closer to
# B.2's "clear your own bar, get a fixed amount" shape:
#
#   4%  Orders           — order count this month, peer-relative (higher
#                           better). "Actively used the app" at its most
#                           basic — placed real orders via the mobile API.
#   24% Location          — 50% GPS fill rate (opmob.xlat/xlong present)
#                            + 50% GPS distinctness rate (distinct
#                            coordinates ÷ GPS-tagged orders) — the second
#                            half is what actually catches "same fake spot
#                            every time," confirmed as a real, detectable
#                            pattern against live data before building
#                            (one real salesman: 133 distinct coordinates
#                            out of 3,632 GPS-tagged orders, ~3.7%, vs a
#                            healthy peer at ~31%).
#   24% Return hygiene    — % of the salesman's OWN returns (opcrn.xemp)
#                            NOT still stuck "1-Open" more than a 14-day
#                            grace period (measured against real TODAY, not
#                            the reporting month's own end — so a return
#                            opened near month-end still gets a fair grace
#                            period even after the month has technically
#                            closed). Confirmed against live data this is a
#                            weak/near-universal signal on its own (~99.9%
#                            of ALL returns eventually reach "3-Issued"
#                            regardless of salesman) — kept anyway per the
#                            user's own weighting, since it still penalizes
#                            genuinely-stuck outliers even if most score
#                            near the top.
#   24% Promised payment   — % of the salesman's delivery orders
#                            (opdor.xsp) this month with xdatepay filled
#                            in. Confirmed near-zero adoption in real data
#                            (334 of 594,723 opdor rows ever, ~0% for top
#                            2026 salesmen) — the user confirmed this
#                            reflects real (not a local-mirror gap) low
#                            usage, so this component is explicitly meant
#                            to reward genuine early adopters from a near-0
#                            baseline, not to penalize everyone equally.
#   24% Collections        — count of glpmt entries (xemp) this month,
#                            peer-relative. Same near-zero-adoption
#                            confirmation as promised payment (9 total
#                            glpmt rows locally, ever).
#
# Salesman population = every spid appearing in `opmob` (i.e. genuinely
# placed at least one order via the app) within the reporting month — a
# salesman with zero orders that month isn't scored at all, matching the
# feature's own premise ("an incentive to all who actively used the app").
# A salesman with zero ELIGIBLE returns/delivery-orders that specific
# month (nothing to evaluate for just that one component) gets a NEUTRAL
# 50 on that one component rather than being punished with a 0 or
# rewarded with a 100 — same "no signal -> neutral" principle
# _peer_relative-style helpers elsewhere in this app already use for a
# zero-variance population, just applied per-salesman instead of
# per-population here.

def _peer_scale_app_usage(series: pd.Series, lower_is_better: bool = False) -> pd.Series:
    """Peer-relative min-max scaling to [0, 100] — same pattern
    processing/marketing.py::_compute_composite_score already uses for
    Customer Score, applied here too (uniformly, even for inputs that are
    already 0-100 percentages) so a component still differentiates
    performers clustered in a narrow high band, not just compress them.
    NaN entries (nothing to evaluate for that salesman) pass through as
    NaN — the caller fills those with a neutral 50 afterward, not 0."""
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index)
    mn, mx = valid.min(), valid.max()
    if mx <= mn:
        scaled = pd.Series(50.0, index=s.index)
        scaled[s.isna()] = np.nan
        return scaled
    scaled = (s - mn) / (mx - mn) * 100.0
    return (100.0 - scaled) if lower_is_better else scaled


def compute_app_usage_scores(
    orders_df: pd.DataFrame, returns_df: pd.DataFrame, delivery_df: pd.DataFrame,
    glpmt_df: pd.DataFrame, month_start, month_end, today,
) -> pd.DataFrame:
    """Per-salesman App Usage composite score (0-100) for one reporting
    month. Pure function — all four input DataFrames are the caller's own
    already-pooled (100001+100000), already-month-scoped raw pulls (see
    module note above for exact table/column sourcing); this function only
    aggregates and scores.

    `month_start`/`month_end`/`today` — `today` caps an ongoing reporting
    month's evaluation window (same "live while ongoing" principle as
    A.1), and separately drives the return-hygiene 14-day grace period
    regardless of the reporting month's own boundary.

    Returns one row per real app-using salesman: `spid`, the 5 raw metrics
    (`orders_count`, `gps_fill_rate`, `gps_distinct_rate`,
    `return_stuck_rate`, `promised_pay_rate`, `collection_count`), the 5
    weighted component scores, and the final `score` (0-100).
    """
    eval_end = min(pd.Timestamp(month_end), pd.Timestamp(today))
    eval_start = pd.Timestamp(month_start)

    if orders_df is None or orders_df.empty:
        return pd.DataFrame()

    o = orders_df.copy()
    o["date"] = pd.to_datetime(o["date"], errors="coerce")
    o = o[(o["date"] >= eval_start) & (o["date"] <= eval_end)]
    if o.empty:
        return pd.DataFrame()
    o["spid"] = o["spid"].astype(str)

    # One row per ORDER (not per line item) — first line item by xroword,
    # matching the API's own "first valid coordinate" convention for
    # location_records (see module note in the data-map doc).
    order_first = (
        o.sort_values(["spid", "invoiceno", "invoicesl", "xroword"])
         .drop_duplicates(subset=["spid", "invoiceno", "invoicesl"])
         .copy()
    )

    sp_list = order_first[["spid"]].drop_duplicates().reset_index(drop=True)
    if sp_list.empty:
        return pd.DataFrame()

    orders_count = order_first.groupby("spid").size()

    order_first["has_gps"] = order_first["xlat"].notna() & (order_first["xlat"] != 0)
    gps_fill_rate = order_first.groupby("spid")["has_gps"].mean() * 100.0

    gps_rows = order_first[order_first["has_gps"]].copy()
    gps_distinct_rate = pd.Series(dtype=float)
    if not gps_rows.empty:
        gps_rows["latlong"] = list(zip(gps_rows["xlat"], gps_rows["xlong"]))
        distinct_ct = gps_rows.groupby("spid")["latlong"].nunique()
        gps_orders_ct = gps_rows.groupby("spid").size()
        gps_distinct_rate = (distinct_ct / gps_orders_ct * 100.0)

    # Returns: stuck-rate among returns old enough (>= 14 days as of real
    # today) to have had a fair chance to close, scoped to this salesman's
    # own returns dated within the reporting month.
    return_stuck_rate = pd.Series(dtype=float)
    if returns_df is not None and not returns_df.empty:
        r = returns_df.copy()
        r["spid"] = r["spid"].astype(str)
        r["date"] = pd.to_datetime(r["date"], errors="coerce")
        r = r[(r["date"] >= eval_start) & (r["date"] <= eval_end) & (r["spid"].isin(sp_list["spid"]))]
        cutoff = pd.Timestamp(today) - pd.Timedelta(days=14)
        eligible = r[r["date"] <= cutoff]
        if not eligible.empty:
            is_stuck = (eligible["status"].astype(str) == "1-Open")
            return_stuck_rate = is_stuck.groupby(eligible["spid"]).mean() * 100.0

    # Promised payment: % of this month's delivery orders with xdatepay set
    # (excluding the 2999-12-31 "unset" sentinel — see CLAUDE.md Common
    # Pitfall #11).
    promised_pay_rate = pd.Series(dtype=float)
    if delivery_df is not None and not delivery_df.empty:
        d = delivery_df.copy()
        d["spid"] = d["spid"].astype(str)
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d[(d["date"] >= eval_start) & (d["date"] <= eval_end) & (d["spid"].isin(sp_list["spid"]))]
        if not d.empty:
            d["paydate"] = pd.to_datetime(d["paydate"], errors="coerce")
            has_pay = d["paydate"].notna() & (d["paydate"] != pd.Timestamp("2999-12-31"))
            promised_pay_rate = has_pay.groupby(d["spid"]).mean() * 100.0

    # Collections: raw glpmt entry count this month, keyed off the entry's
    # own promised-payment date (paydate) — matches how every other
    # commission section date-scopes glpmt.
    collection_count = pd.Series(dtype=float)
    if glpmt_df is not None and not glpmt_df.empty:
        g = glpmt_df.copy()
        g["spid"] = g["spid"].astype(str)
        g["paydate"] = pd.to_datetime(g["paydate"], errors="coerce")
        g = g[(g["paydate"] >= eval_start) & (g["paydate"] <= eval_end) & (g["spid"].isin(sp_list["spid"]))]
        if not g.empty:
            collection_count = g.groupby("spid").size()

    rows = sp_list.copy()
    rows["orders_count"] = rows["spid"].map(orders_count).fillna(0.0)
    rows["gps_fill_rate"] = rows["spid"].map(gps_fill_rate).fillna(0.0)
    rows["gps_distinct_rate"] = rows["spid"].map(gps_distinct_rate)  # NaN if 0 GPS orders -> neutral, not 0
    rows["return_stuck_rate"] = rows["spid"].map(return_stuck_rate)  # NaN if 0 eligible returns -> neutral
    rows["promised_pay_rate"] = rows["spid"].map(promised_pay_rate)  # NaN if 0 delivery orders -> neutral
    rows["collection_count"] = rows["spid"].map(collection_count).fillna(0.0)

    rows["score_orders"] = _peer_scale_app_usage(rows["orders_count"]).fillna(50.0) * 0.04
    loc = (
        _peer_scale_app_usage(rows["gps_fill_rate"]).fillna(50.0) * 0.5
        + _peer_scale_app_usage(rows["gps_distinct_rate"]).fillna(50.0) * 0.5
    )
    rows["score_location"] = loc * 0.24
    rows["score_returns"] = _peer_scale_app_usage(rows["return_stuck_rate"], lower_is_better=True).fillna(50.0) * 0.24
    rows["score_promised_pay"] = _peer_scale_app_usage(rows["promised_pay_rate"]).fillna(50.0) * 0.24
    rows["score_collections"] = _peer_scale_app_usage(rows["collection_count"]).fillna(50.0) * 0.24

    rows["score"] = (
        rows["score_orders"] + rows["score_location"] + rows["score_returns"]
        + rows["score_promised_pay"] + rows["score_collections"]
    ).clip(lower=0.0, upper=100.0).round(1)

    return rows.sort_values("score", ascending=False).reset_index(drop=True)


def compute_app_usage_bonus(campaign: dict, usage_scores_df: pd.DataFrame) -> dict:
    """Threshold payout, NOT a ranking — confirmed 2026-09-21, explicit
    ask: "create a score from 1 to 100, whoever scores more than 90 gets a
    fixed commission." Every salesman whose `score` clears
    `product_rates["threshold"]` gets the SAME flat
    `product_rates["bonus_amount"]` (BDT) — same "clear your own bar" shape
    as B.2's own (not-yet-built) individual-target mechanism, distinct
    from A.1/A.1b/A.2's rank-based per-position payouts.

    Returns `{"scores": DataFrame[spid, spname, score, ...raw metrics...,
    qualified, payout], "threshold", "bonus_amount", "qualified_count",
    "total_payout"}` — `total_payout` = `qualified_count × bonus_amount`,
    always the raw figure (no gate on this mechanism)."""
    config = campaign.get("product_rates") or {}
    threshold = float(config.get("threshold", 90))
    bonus_amount = float(config.get("bonus_amount", 0) or 0)

    empty = pd.DataFrame(columns=["spid", "score", "qualified", "payout"])
    if usage_scores_df is None or usage_scores_df.empty:
        return {
            "scores": empty, "threshold": threshold, "bonus_amount": bonus_amount,
            "qualified_count": 0, "total_payout": 0.0,
        }

    d = usage_scores_df.copy()
    d["qualified"] = d["score"] > threshold
    d["payout"] = np.where(d["qualified"], bonus_amount, 0.0)
    d = d.sort_values("score", ascending=False).reset_index(drop=True)

    qualified_count = int(d["qualified"].sum())
    total_payout = float(d["payout"].sum())
    return {
        "scores": d, "threshold": threshold, "bonus_amount": bonus_amount,
        "qualified_count": qualified_count, "total_payout": total_payout,
    }
