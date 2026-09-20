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


def _eval_months(num_months: int, today: pd.Timestamp) -> list:
    """[(year, month), ...] oldest first — reuses ssc.month_choices (already
    correct on year-rollover), sliced to num_months and reversed so the
    evaluation window reads oldest-to-newest, ending at the CURRENT, ongoing
    month (never "last month" — confirmed 2026-09-20, commission standards
    need to be set ahead of time)."""
    choices = ssc.month_choices(today)[:num_months]
    return list(reversed([(y, m) for (_lbl, y, m) in choices]))


def _current_window(num_months: int, today: pd.Timestamp) -> tuple:
    """Informational window_start/window_end for the stored campaign row
    only — NOT used to gate computation (see _compute_ranking, which always
    recomputes the live window from num_months + today at view time)."""
    months = _eval_months(num_months, today)
    oldest_year, oldest_month = months[0]
    return pd.Timestamp(oldest_year, oldest_month, 1).date(), today.date()


def _compute_ranking(campaign: dict, num_months: int) -> dict:
    today = pd.Timestamp.today().normalize()
    months = _eval_months(num_months, today)
    years_needed = sorted({y for y, _m in months})

    sales_by_year = {y: _load_pooled_sales_year(y) for y in years_needed}
    returns_by_year = {y: _load_pooled_returns_year(y) for y in years_needed}
    collection_by_year = {y: _load_pooled_collection_year(y) for y in years_needed}
    ar_clean = _pooled_ar_ledger()

    monthly_tables = []
    for (y, m) in months:
        sales_df = sales_by_year.get(y, pd.DataFrame())
        target_by_sp = _target_by_sp(sales_df, y, m)
        table = ssc.build_pooled_monthly_scores(
            y, m, today, sales_df, returns_by_year.get(y, pd.DataFrame()),
            collection_by_year.get(y, pd.DataFrame()), ar_clean, target_by_sp,
        )
        monthly_tables.append(table)

    pooled_sales_full = (
        pd.concat(list(sales_by_year.values()), ignore_index=True) if sales_by_year else pd.DataFrame()
    )
    oldest_year, oldest_month = months[0]
    campaign_for_gate = dict(campaign)
    campaign_for_gate["window_start"] = pd.Timestamp(oldest_year, oldest_month, 1)
    campaign_for_gate["window_end"] = today

    return cc.compute_best_performer_ranking(campaign_for_gate, monthly_tables, pooled_sales_full)


# ── Create / Edit Best Performer campaign (admin-only) — shared body ───────

def _render_form(existing: dict | None = None) -> None:
    is_edit = existing is not None
    key_prefix = f"a1_edit_{existing['id']}" if is_edit else "a1_new"

    name = st.text_input(
        "Campaign name", value=(existing["campaign_name"] if is_edit else ""), key=f"{key_prefix}_name",
    )

    existing_config = (existing.get("product_rates") or {}) if is_edit else {}
    months_default = int(existing_config.get("num_months", 1)) if is_edit else 1
    months_default = min(max(months_default, MIN_MONTHS), MAX_MONTHS)
    num_months = st.number_input(
        f"Months to average ({MIN_MONTHS}-{MAX_MONTHS}) — always ending at the current, ongoing month",
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
        today = pd.Timestamp.today().normalize()
        window_start, window_end = _current_window(int(num_months), today)
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
    st.caption(
        f"Best Performer (ranked by average Salesman Score) · Averaged over "
        f"{num_months} {month_label} ending this month"
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
        "averaged over the admin-chosen number of months, always ending at the "
        "CURRENT, ongoing month (never a lagging 'last month' figure, since "
        "commission standards need to be set ahead of time). Pooled across "
        "100001+100000 and gated on both ZIDs hitting their own sales target, same "
        "as the other commission sections."
    )

    is_admin = (not read_only) and st.session_state.get("user_role") == "admin"
    if is_admin:
        _render_create()

    campaigns = _list_best_performer_campaigns()
    if campaigns.empty:
        st.info("No Best Performer campaigns yet." + (" Create one above." if is_admin else " Ask an admin to create one."))
        return

    labels = {
        f"#{r.id} — {r.campaign_name}": r.id
        for r in campaigns.itertuples()
    }
    pick = st.selectbox("Select a campaign", list(labels.keys()), key=f"a1_select_campaign{key_suffix}")
    campaign = cc.get_campaign(labels[pick])
    if campaign:
        _render_detail(campaign, read_only=read_only, key_suffix=key_suffix)
