# views/commission_best_performer_view.py
# UI for A.1 — Best Performer (ranked by average Salesman Score, not a
# rate/threshold or product-breadth mechanism like B.1/3/4 or A.2). See
# commission_tracking_design.md §A.1 and
# processing/commission_campaigns.py::compute_best_performer_ranking for the
# engine — this file is UI only.
#
# Reuses the SAME commission_campaigns table as B.1/3/4 and A.2 (confirmed
# 2026-09-20, same "no separate table per campaign type" ask) — CAMPAIGN_TYPE
# below is the free-text value that distinguishes these rows from the other
# two, so each section's own campaign picker only ever lists its own rows.
#
# Salesman-only (no recipient_type toggle) — the scoring engine's own inputs
# (target achievement, collection %, AR balance) don't have a
# customer-equivalent meaning, a deliberate deviation from the design doc's
# "for future note" table.

from __future__ import annotations

import calendar

import pandas as pd
import streamlit as st

from processing import commission_campaigns as cc
from processing import common, salesman_score as ssc
from views._tm_shared import _get_target, _load_ar_ledger_clean

CAMPAIGN_TYPE = "Best Performer"
MIN_MONTHS = 1
MAX_MONTHS = 3
MIN_WINNERS = 2
MAX_WINNERS = 10

# 100001 + 100000 share one field sales team — same pooling every other
# commission section (B.1/3/4, A.2) already uses, confirmed 2026-09-20 this
# applies to A.1 too (same "one hundred thousand one and one hundred
# thousand" pooling + target gate).
_ZID_PROJ = {"100001": "GULSHAN TRADING", "100000": "GI Corporation"}


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _fmt_bdt(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"৳{v:,.0f}"


# ── Pooled data loaders (own cache, per year — mirrors views/salesman_score.py's
# per-(zid, year) loaders, pooled across both ZIDs like commission_rankings_view's
# own _load_pooled_sales) ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_pooled_sales_year(year: int) -> pd.DataFrame:
    from core.analytics import Analytics
    frames = []
    for zid in _ZID_PROJ:
        df = Analytics("sales", zid=zid, filters={"year": [year]}).data
        if df is not None and not df.empty:
            frames.append(df)
    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if pooled.empty:
        return pooled
    (pooled,) = common.data_copy_add_columns(pooled)
    return pooled


@st.cache_data(show_spinner=False, ttl=3600)
def _load_pooled_returns_year(year: int) -> pd.DataFrame:
    from core.analytics import Analytics
    frames = []
    for zid in _ZID_PROJ:
        df = Analytics("return", zid=zid, filters={"year": [year]}).data
        if df is not None and not df.empty:
            frames.append(df)
    pooled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if pooled.empty:
        return pooled
    (pooled,) = common.data_copy_add_columns(pooled)
    return pooled


@st.cache_data(show_spinner=False, ttl=3600)
def _load_pooled_collection_year(year: int) -> pd.DataFrame:
    from core.analytics import Analytics
    frames = []
    for zid in _ZID_PROJ:
        df = Analytics("collection", zid=zid, filters={"year": [year]}).data
        if df is not None and not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pooled_ar_ledger() -> pd.DataFrame:
    """_load_ar_ledger_clean is already @st.cache_data-wrapped per (zid, proj)
    in views/_tm_shared.py — no extra caching needed here, just the pool."""
    frames = []
    for zid, proj in _ZID_PROJ.items():
        df = _load_ar_ledger_clean(zid, proj)
        if df is not None and not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _target_by_sp(sales_df: pd.DataFrame, year: int, month: int) -> dict:
    """{spid: target}, summed across BOTH ZIDs for that spid regardless of
    which ZID's sales rows it came from — same consolidated-target logic
    views/salesman_score.py's own consolidation toggle already uses (100001
    + 100000 share one field sales team/spid codes)."""
    if sales_df is None or sales_df.empty or "spid" not in sales_df.columns:
        return {}
    spids = sales_df["spid"].dropna().astype(str).unique()
    return {
        spid: float(_get_target("100001", spid, year, month) or 0.0)
        + float(_get_target("100000", spid, year, month) or 0.0)
        for spid in spids
    }


def _anchor_month_choices(today: pd.Timestamp, months_forward: int = 12) -> list:
    """[(label, year, month), ...] earliest first: the current month plus up
    to `months_forward` FUTURE ones — options for the "reporting month"
    picker (see module note above `_render_form`). **Current month or later
    only — confirmed 2026-09-20, explicit correction from the user: "I can
    only set this current month and future months... I can't set it for a
    month that already passed."** A campaign can be set up mid-month for
    the rest of the current month, or in advance for a future one; never
    retroactively for one that's already closed (see `_render_form` for the
    one narrow exception: an already-existing campaign keeps showing its
    own past reporting month so editing it can't silently move it)."""
    cur = pd.Timestamp(today.year, today.month, 1)
    return [
        ((cur + pd.DateOffset(months=i)).strftime("%b %Y"),
         int((cur + pd.DateOffset(months=i)).year),
         int((cur + pd.DateOffset(months=i)).month))
        for i in range(months_forward + 1)
    ]


def _months_for_window(anchor_year: int, anchor_month: int, num_months: int) -> list:
    """[(year, month), ...] oldest first, num_months entries ending at the
    campaign's own reporting month (anchor_year, anchor_month) — fixed at
    setup time. Confirmed 2026-09-20: a campaign must keep showing its own
    reporting month's results after that month closes, not silently roll
    forward to whatever month happens to be current when it's re-opened —
    so this is anchored to the STORED reporting month, never to wall-clock
    "today" at view time (that's a separate, real `today` still passed into
    `build_pooled_monthly_scores` below, only to decide whether the
    reporting month itself is still live-in-progress)."""
    choices = ssc.month_choices(pd.Timestamp(anchor_year, anchor_month, 1))[:num_months]
    return list(reversed([(y, m) for (_lbl, y, m) in choices]))


def _window_for_anchor(anchor_year: int, anchor_month: int, num_months: int) -> tuple:
    """(window_start, window_end) dates for the campaign row — real,
    load-bearing values now (previously informational-only, corrected
    2026-09-20): window_end is the LAST day of the reporting month
    (anchor_year, anchor_month), window_start is the first day of the
    oldest of the num_months months ending there. `window_end`'s own
    (year, month) is what `_compute_ranking` reads back out as the anchor
    on every later view — the campaign's reporting month never moves once
    saved."""
    months = _months_for_window(anchor_year, anchor_month, num_months)
    oldest_year, oldest_month = months[0]
    last_day = calendar.monthrange(anchor_year, anchor_month)[1]
    window_start = pd.Timestamp(oldest_year, oldest_month, 1).date()
    window_end = pd.Timestamp(anchor_year, anchor_month, last_day).date()
    return window_start, window_end


def _compute_ranking(campaign: dict, num_months: int) -> dict:
    today = pd.Timestamp.today().normalize()
    anchor_end = pd.Timestamp(campaign["window_end"])
    anchor_year, anchor_month = int(anchor_end.year), int(anchor_end.month)
    months = _months_for_window(anchor_year, anchor_month, num_months)
    years_needed = sorted({y for y, _m in months})

    sales_by_year = {y: _load_pooled_sales_year(y) for y in years_needed}
    returns_by_year = {y: _load_pooled_returns_year(y) for y in years_needed}
    collection_by_year = {y: _load_pooled_collection_year(y) for y in years_needed}
    ar_clean = _pooled_ar_ledger()

    monthly_tables = []
    for (y, m) in months:
        sales_df = sales_by_year.get(y, pd.DataFrame())
        target_by_sp = _target_by_sp(sales_df, y, m)
        # `today` here is real wall-clock today, not the campaign's anchor —
        # build_pooled_monthly_scores uses it only to decide whether THIS
        # specific (y, m) is still the genuinely current month (live,
        # capped to today) or already closed (full completed month).
        table = ssc.build_pooled_monthly_scores(
            y, m, today, sales_df, returns_by_year.get(y, pd.DataFrame()),
            collection_by_year.get(y, pd.DataFrame()), ar_clean, target_by_sp,
        )
        monthly_tables.append(table)

    pooled_sales_full = (
        pd.concat(list(sales_by_year.values()), ignore_index=True) if sales_by_year else pd.DataFrame()
    )
    oldest_year, oldest_month = months[0]
    last_day = calendar.monthrange(anchor_year, anchor_month)[1]
    campaign_for_gate = dict(campaign)
    campaign_for_gate["window_start"] = pd.Timestamp(oldest_year, oldest_month, 1)
    campaign_for_gate["window_end"] = pd.Timestamp(anchor_year, anchor_month, last_day)

    return cc.compute_best_performer_ranking(campaign_for_gate, monthly_tables, pooled_sales_full)


# ── Create / Edit Best Performer campaign (admin-only) — shared body ───────
#
# Reporting month, confirmed 2026-09-20 after the user flagged a real bug:
# the window must be pinned to whichever month the campaign is SET UP for
# (chosen here, stored as the real window_end), not recomputed from
# wall-clock "today" on every later view. Concretely: while the reporting
# month is still the real, ongoing month, results track live (capped to
# today) — matching the original "current month" behavior. Once that month
# closes, results stay frozen at its final numbers forever after, even if
# the campaign is reopened months later — they never roll forward to
# whatever month happens to be current at view time.
#
# **Current month or later ONLY — corrected same day, second explicit
# correction from the user**: "I can only set this current month and
# future months... I can't set it for a month that already passed." A
# campaign can be created mid-month for the rest of the current month (a
# partial month is fine), or in advance for any future month; never
# retroactively for one that's already closed. `_anchor_month_choices`
# offers current + up to 12 future months, never past ones — the one
# exception is editing an already-existing campaign whose own reporting
# month has since closed, which keeps showing/selecting that one real
# month (not offered as a fresh choice) so Save can't silently move it.

def _render_form(existing: dict | None = None) -> None:
    is_edit = existing is not None
    key_prefix = f"a1_edit_{existing['id']}" if is_edit else "a1_new"

    name = st.text_input(
        "Campaign name", value=(existing["campaign_name"] if is_edit else ""), key=f"{key_prefix}_name",
    )

    today = pd.Timestamp.today().normalize()
    month_opts = _anchor_month_choices(today)
    if is_edit:
        existing_end = pd.Timestamp(existing["window_end"])
        already_offered = any(y == existing_end.year and m == existing_end.month for _l, y, m in month_opts)
        if not already_offered:
            # The campaign's own reporting month has already closed (this is
            # editing an old campaign after the fact, e.g. just to fix the
            # name/payouts) — keep showing/selecting its real month so Save
            # can't silently move it forward to the current month, without
            # offering any OTHER past month as a fresh choice.
            month_opts = [(existing_end.strftime("%b %Y"), int(existing_end.year), int(existing_end.month))] + month_opts
        default_label = existing_end.strftime("%b %Y")
    else:
        default_label = month_opts[0][0]
    month_labels = [lbl for lbl, _y, _m in month_opts]
    sel_label = st.selectbox(
        "Reporting month — the month this campaign is for", month_labels,
        index=month_labels.index(default_label), key=f"{key_prefix}_month",
    )
    anchor_year, anchor_month = next((y, m) for lbl, y, m in month_opts if lbl == sel_label)
    st.caption(
        "Current month or later only — you can set this up for the rest of this month, or in "
        "advance for a future one, but not for a month that's already passed. Tracks live while "
        "the reporting month is the current, ongoing one; once it ends, results stay fixed at "
        "that month's final numbers — reopening this campaign later won't roll the window "
        "forward to whatever month is current by then."
    )

    existing_config = (existing.get("product_rates") or {}) if is_edit else {}
    months_default = int(existing_config.get("num_months", 1)) if is_edit else 1
    months_default = min(max(months_default, MIN_MONTHS), MAX_MONTHS)
    num_months = st.number_input(
        f"Months to average ({MIN_MONTHS}-{MAX_MONTHS}), trailing back from the reporting month",
        min_value=MIN_MONTHS, max_value=MAX_MONTHS, value=months_default, step=1, key=f"{key_prefix}_nummonths",
    )

    winners_default = int(existing_config.get("num_winners", 3)) if is_edit else 3
    winners_default = min(max(winners_default, MIN_WINNERS), MAX_WINNERS)
    num_winners = st.number_input(
        f"Number of winners ({MIN_WINNERS}-{MAX_WINNERS})", min_value=MIN_WINNERS, max_value=MAX_WINNERS,
        value=winners_default, step=1, key=f"{key_prefix}_numwinners",
    )

    existing_payouts = existing_config.get("payouts_by_rank", []) if is_edit else []
    st.caption("Payout amount per rank position (ranked by average Salesman Score, highest first):")
    payouts_by_rank = []
    n_cols = min(int(num_winners), 5)
    for row_start in range(0, int(num_winners), n_cols):
        payout_cols = st.columns(min(n_cols, int(num_winners) - row_start))
        for j, col in enumerate(payout_cols):
            i = row_start + j
            default_amt = float(existing_payouts[i]) if i < len(existing_payouts) else 0.0
            with col:
                amt = st.number_input(
                    f"{_ordinal(i + 1)} (BDT)", min_value=0.0, step=100.0, value=default_amt,
                    key=f"{key_prefix}_payout_{i}",
                )
            payouts_by_rank.append(amt)

    notes = st.text_area(
        "Notes (optional)", value=(existing.get("notes") or "" if is_edit else ""), key=f"{key_prefix}_notes",
    )

    all_payouts_set = all(a > 0 for a in payouts_by_rank)
    can_submit = bool(name.strip()) and all_payouts_set
    if not all_payouts_set:
        st.caption("Every rank position needs a payout amount > 0.")

    btn_label = "💾 Save Changes" if is_edit else "✅ Create Best Performer Campaign"
    if st.button(btn_label, key=f"{key_prefix}_submit_btn", disabled=not can_submit):
        product_rates_payload = {
            "num_months": int(num_months), "num_winners": int(num_winners), "payouts_by_rank": payouts_by_rank,
        }
        window_start, window_end = _window_for_anchor(anchor_year, anchor_month, int(num_months))
        if is_edit:
            ok = cc.update_campaign(
                campaign_id=int(existing["id"]), campaign_name=name.strip(), campaign_type=CAMPAIGN_TYPE,
                recipient_type="salesman", product_rates=product_rates_payload, cap=None,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups={}, uptick_baseline_months=3, notes=notes,
            )
            if ok:
                st.success("Best Performer campaign updated.", icon="✅")
                st.rerun()
            else:
                st.error("Could not update campaign.")
        else:
            cid = cc.create_campaign(
                campaign_name=name.strip(), campaign_type=CAMPAIGN_TYPE, recipient_type="salesman",
                product_rates=product_rates_payload, cap=None,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups={}, uptick_baseline_months=3,
                created_by=st.session_state.get("username", ""), notes=notes,
            )
            if cid:
                st.success(f"Best Performer campaign #{cid} created.", icon="✅")
                st.rerun()
            else:
                st.error("Could not create campaign.")


def _render_create() -> None:
    with st.expander("➕ Create New Best Performer Campaign", expanded=False):
        _render_form()


def _render_edit(campaign: dict) -> None:
    with st.expander("✏️ Edit Best Performer Campaign", expanded=False):
        _render_form(existing=campaign)


def _list_best_performer_campaigns() -> pd.DataFrame:
    df = cc.list_campaigns()
    if df.empty:
        return df
    return df[df["campaign_type"] == CAMPAIGN_TYPE].reset_index(drop=True)


def _render_gate_banner(gate: dict) -> None:
    if gate["passed"]:
        st.success("✅ Gate passed — both 100001 and 100000 hit their sales target for this window.", icon="✅")
    else:
        st.warning(
            "⚠️ Gate NOT passed — payout is ৳0 until both ZIDs hit their sales target "
            "(the computed figures below still show for visibility).", icon="⚠️",
        )
    c1, c2 = st.columns(2)
    for col, zid in zip((c1, c2), ("100001", "100000")):
        g = gate[zid]
        status = "✅" if g["hit"] else "❌"
        col.caption(f"{status} ZID {zid}: {g['actual']:,.0f} / {g['target']:,.0f} target")


# ── Campaign detail / results view ──────────────────────────────────────────

def _render_detail(campaign: dict, read_only: bool, key_suffix: str) -> None:
    config = campaign.get("product_rates") or {}
    num_months = int(config.get("num_months") or 1)
    num_winners = config.get("num_winners")
    payouts_by_rank = config.get("payouts_by_rank", [])

    st.markdown(f"### {campaign['campaign_name']}")
    month_label = "month" if num_months == 1 else "months"
    reporting_month = pd.Timestamp(campaign["window_end"]).strftime("%B %Y")
    today = pd.Timestamp.today().normalize()
    is_ongoing = (
        int(pd.Timestamp(campaign["window_end"]).year) == today.year
        and int(pd.Timestamp(campaign["window_end"]).month) == today.month
    )
    status_note = "still tracking live" if is_ongoing else "closed — final numbers"
    st.caption(
        f"Best Performer (ranked by average Salesman Score) · Averaged over "
        f"{num_months} {month_label} ending {reporting_month} ({status_note})"
    )
    payouts_line = " · ".join(
        f"{_ordinal(i + 1)}: {_fmt_bdt(a)}" for i, a in enumerate(payouts_by_rank)
    )
    st.caption(f"{num_winners} winner(s) — {payouts_line}")

    if not read_only:
        if st.session_state.get("user_role") == "admin":
            st.divider()
            _render_edit(campaign)
            if st.button("🗑 Delete This Campaign", key=f"a1_delete_{campaign['id']}{key_suffix}"):
                cc.delete_campaign(int(campaign["id"]))
                st.rerun()
        return

    # ── Results (Target Management "💰 Commission Results") ────────────────
    with st.spinner(f"Scoring salesmen across {num_months} month(s) (pooled 100001+100000)…"):
        result = _compute_ranking(campaign, num_months)

    _render_gate_banner(result["gate"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Payout", _fmt_bdt(result["total_payout"]))
    m2.metric("Winners", f"{result['num_winners']}")
    m3.metric("Months Averaged", f"{result['num_months']}")

    ranking_df = result["ranking"]
    if ranking_df.empty:
        st.info("No scored salesmen for this window yet.")
        return

    disp = ranking_df.rename(columns={
        "rank": "Rank", "recipient": "Salesman Code", "recipient_name": "Salesman",
        "avg_score": "Avg Score", "months_scored": "Months Scored", "payout": "Payout (BDT)",
    })
    disp["Avg Score"] = disp["Avg Score"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
    disp["Payout (BDT)"] = disp["Payout (BDT)"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) and v > 0 else "—")
    st.dataframe(disp, width="stretch", hide_index=True)


# ── Top-level section entry point ───────────────────────────────────────────

def render(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    """`read_only=True` (Target Management's "💰 Commission Results") shows
    only the ranking result — no create/edit/delete controls, regardless of
    the viewer's own role. `read_only=False` (admin Commissions page) shows
    only setup, no ranking computation at all."""
    st.subheader("🏆 A.1 — Best Performer")
    st.caption(
        "Ranks salesmen by their average Salesman Score — the EXACT same engine as "
        "Target Management's own 🎯 Salesman Score tab (same weights: +45 target, "
        "+45 collection, +5 products, +5 customers, -6 returns, -12/-2 AR balance) — "
        "averaged over the admin-chosen number of months, ending at a reporting month "
        "picked at setup. Tracks live while that month is still ongoing, then stays "
        "fixed at its final numbers once the month closes — reopening a campaign later "
        "never rolls the window forward. Pooled across 100001+100000 and gated on both "
        "ZIDs hitting their own sales target, same as the other commission sections."
    )

    is_admin = (not read_only) and st.session_state.get("user_role") == "admin"
    if is_admin:
        _render_create()

    campaigns = _list_best_performer_campaigns()
    if campaigns.empty:
        st.info("No Best Performer campaigns yet." + (" Create one above." if is_admin else " Ask an admin to create one."))
        return

    labels = {
        f"#{r.id} — {r.campaign_name} ({pd.Timestamp(r.window_end).strftime('%b %Y')})": r.id
        for r in campaigns.itertuples()
    }
    pick = st.selectbox("Select a campaign", list(labels.keys()), key=f"a1_select_campaign{key_suffix}")
    campaign = cc.get_campaign(labels[pick])
    if campaign:
        _render_detail(campaign, read_only=read_only, key_suffix=key_suffix)
