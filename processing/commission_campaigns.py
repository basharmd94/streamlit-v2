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
#   1. SINGLE ZID, never pooled 100001+100000 — a customer code is only
#      unique WITHIN one ZID (unlike a salesman, who genuinely appears in
#      both), so pooling would risk merging two unrelated customers who
#      happen to share a code under one ranking row. The campaign's own ZID
#      is chosen at setup (whichever business is active then) and PINNED in
#      product_rates — same "pin at setup, don't drift" discipline as A.1's
#      reporting month.
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
    composite_score, unmodified) and pays a fixed BDT amount per rank
    position to the top `num_winners` (2-100, admin-set) — same per-rank
    mechanism as A.1/A.2, just a wider winner range (up to 100, explicit
    ask) since a customer campaign can reasonably span far more recipients
    than a salesman one.

    `customer_score_df` — the caller's own already-scoped, already-scored
    output of build_customer_marketing_table (one ZID, one reporting year)
    — this function only ranks/pays off an existing `composite_score`
    column, it doesn't compute the score itself.

    Returns `{"ranking": DataFrame[rank, recipient, recipient_name,
    composite_score, payout], "num_winners", "total_payout"}` — no gate,
    so total_payout is always the raw computed figure (never zeroed).
    """
    config = campaign.get("product_rates") or {}
    payouts_by_rank = [float(v) for v in config.get("payouts_by_rank", [])]
    num_winners = int(config.get("num_winners") or len(payouts_by_rank))

    empty = pd.DataFrame(columns=["rank", "recipient", "recipient_name", "composite_score", "payout"])
    if (
        customer_score_df is None or customer_score_df.empty
        or "composite_score" not in customer_score_df.columns
    ):
        return {"ranking": empty, "num_winners": num_winners, "total_payout": 0.0}

    d = customer_score_df.dropna(subset=["composite_score"]).copy()
    d["cusid"] = d["cusid"].astype(str)
    d = d.sort_values(["composite_score", "cusid"], ascending=[False, True]).reset_index(drop=True)
    d["rank"] = d.index + 1
    d["payout"] = 0.0
    for i in range(min(num_winners, len(payouts_by_rank), len(d))):
        d.loc[i, "payout"] = payouts_by_rank[i]

    total_payout = float(d["payout"].sum())
    ranking = d.rename(columns={"cusid": "recipient", "cusname": "recipient_name"})[
        ["rank", "recipient", "recipient_name", "composite_score", "payout"]
    ]
    return {"ranking": ranking, "num_winners": num_winners, "total_payout": total_payout}
