# views/commission_rankings_view.py
# UI for A.2 — Highest Product Sales (ranked by distinct product count, not
# a rate/threshold mechanism like B.1/3/4). See commission_tracking_design.md
# §A.2 and processing/commission_campaigns.py::compute_highest_product_sales_ranking
# for the engine — this file is UI only.
#
# Reuses the SAME commission_campaigns table as B.1/3/4 (confirmed
# 2026-09-20, explicit ask: no separate table per campaign type) —
# CAMPAIGN_TYPE below is the free-text value that distinguishes these rows
# from B.1/3/4's, so each section's own campaign picker only ever lists its
# own rows.

from __future__ import annotations

import pandas as pd
import streamlit as st

from processing import commission_campaigns as cc

CAMPAIGN_TYPE = "Highest Product Sales"
MIN_WINNERS = 2
MAX_WINNERS = 5
_ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}


@st.cache_data(show_spinner=False, ttl=3600)
def _load_pooled_sales() -> pd.DataFrame:
    """Same pooled (100001+100000) sales pull B.1/3/4 uses — the two ZIDs
    share one field sales team, so a salesman's product breadth spans both."""
    from core.analytics import Analytics
    from processing import common
    s1 = Analytics("sales", zid="100001", filters={}).data
    s0 = Analytics("sales", zid="100000", filters={}).data
    pooled = pd.concat([s1, s0], ignore_index=True) if s1 is not None and s0 is not None else pd.DataFrame()
    if pooled.empty:
        return pooled
    (pooled,) = common.data_copy_add_columns(pooled)
    return pooled


def _list_ranking_campaigns() -> pd.DataFrame:
    df = cc.list_campaigns()
    if df.empty:
        return df
    return df[df["campaign_type"] == CAMPAIGN_TYPE].reset_index(drop=True)


def _clamp_min_session_date(key: str, min_v) -> bool:
    """Same pattern as commission_campaigns_view.py's own version — a
    widget's persisted value (window end) can fall below a newly-raised
    min_value (window start) on a rerun; pre-clamp before instantiating."""
    if key in st.session_state:
        cur = st.session_state[key]
        if cur < min_v:
            st.session_state[key] = min_v
            return True
    return False


def _fmt_bdt(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"৳{v:,.0f}"


# ── Create / Edit ranking campaign (admin-only) — shared body ──────────────

def _render_ranking_form(existing: dict | None = None) -> None:
    is_edit = existing is not None
    key_prefix = f"a2_edit_{existing['id']}" if is_edit else "a2_new"

    name = st.text_input(
        "Campaign name", value=(existing["campaign_name"] if is_edit else ""), key=f"{key_prefix}_name",
    )
    recip_options = ["salesman", "customer"]
    recip_default = existing.get("recipient_type", "salesman") if is_edit else "salesman"
    recip_idx = recip_options.index(recip_default) if recip_default in recip_options else 0
    recipient_type = st.radio(
        "Winners are", recip_options, index=recip_idx, key=f"{key_prefix}_recipient",
        horizontal=True, format_func=lambda v: v.capitalize(),
    )

    today = pd.Timestamp.today().normalize().date()
    wstart_default = pd.Timestamp(existing["window_start"]).date() if is_edit else today.replace(day=1)
    wend_default = pd.Timestamp(existing["window_end"]).date() if is_edit else today
    c1, c2 = st.columns(2)
    with c1:
        window_start = st.date_input("Window start", value=wstart_default, key=f"{key_prefix}_wstart")
    just_clamped = _clamp_min_session_date(f"{key_prefix}_wend", window_start)
    with c2:
        wend_kwargs = dict(min_value=window_start, key=f"{key_prefix}_wend")
        if not just_clamped:
            wend_kwargs["value"] = max(wend_default, window_start)
        window_end = st.date_input("Window end", **wend_kwargs)

    existing_config = (existing.get("product_rates") or {}) if is_edit else {}
    num_default = int(existing_config.get("num_winners", 3)) if is_edit else 3
    num_default = min(max(num_default, MIN_WINNERS), MAX_WINNERS)
    num_winners = st.number_input(
        f"Number of winners ({MIN_WINNERS}-{MAX_WINNERS})", min_value=MIN_WINNERS, max_value=MAX_WINNERS,
        value=num_default, step=1, key=f"{key_prefix}_numwinners",
    )

    existing_payouts = existing_config.get("payouts_by_rank", []) if is_edit else []
    st.caption("Payout amount per rank position (ranked by distinct product count, highest first):")
    payouts_by_rank = []
    payout_cols = st.columns(int(num_winners))
    for i in range(int(num_winners)):
        default_amt = float(existing_payouts[i]) if i < len(existing_payouts) else 0.0
        with payout_cols[i]:
            amt = st.number_input(
                f"{_ORDINAL[i + 1]} (BDT)", min_value=0.0, step=100.0, value=default_amt,
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

    btn_label = "💾 Save Changes" if is_edit else "✅ Create Ranking Campaign"
    if st.button(btn_label, key=f"{key_prefix}_submit_btn", disabled=not can_submit):
        product_rates_payload = {"num_winners": int(num_winners), "payouts_by_rank": payouts_by_rank}
        if is_edit:
            ok = cc.update_campaign(
                campaign_id=int(existing["id"]), campaign_name=name.strip(), campaign_type=CAMPAIGN_TYPE,
                recipient_type=recipient_type, product_rates=product_rates_payload, cap=None,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups={}, uptick_baseline_months=3, notes=notes,
            )
            if ok:
                st.success("Ranking campaign updated.", icon="✅")
                st.rerun()
            else:
                st.error("Could not update campaign.")
        else:
            cid = cc.create_campaign(
                campaign_name=name.strip(), campaign_type=CAMPAIGN_TYPE, recipient_type=recipient_type,
                product_rates=product_rates_payload, cap=None,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups={}, uptick_baseline_months=3,
                created_by=st.session_state.get("username", ""), notes=notes,
            )
            if cid:
                st.success(f"Ranking campaign #{cid} created.", icon="✅")
                st.rerun()
            else:
                st.error("Could not create campaign.")


def _render_create() -> None:
    with st.expander("➕ Create New Ranking Campaign", expanded=False):
        _render_ranking_form()


def _render_edit(campaign: dict) -> None:
    with st.expander("✏️ Edit Ranking Campaign", expanded=False):
        _render_ranking_form(existing=campaign)


# ── Campaign detail / results view ──────────────────────────────────────────

def _render_detail(campaign: dict, read_only: bool, key_suffix: str) -> None:
    recipient_type = campaign.get("recipient_type") or "salesman"
    config = campaign.get("product_rates") or {}
    num_winners = config.get("num_winners")
    payouts_by_rank = config.get("payouts_by_rank", [])

    today = pd.Timestamp.today().normalize().date()
    window_end = pd.Timestamp(campaign["window_end"]).date()
    status_badge = "🟢 Ongoing" if window_end >= today else "🔴 Closed"

    st.markdown(f"### {campaign['campaign_name']} — {status_badge}")
    st.caption(
        f"Highest Product Sales (ranked) · Winners: **{recipient_type.capitalize()}** · "
        f"Window: {campaign['window_start']} → {campaign['window_end']}"
    )
    payouts_line = " · ".join(
        f"{_ORDINAL.get(i + 1, f'{i + 1}th')}: {_fmt_bdt(a)}" for i, a in enumerate(payouts_by_rank)
    )
    st.caption(f"{num_winners} winner(s) — {payouts_line}")

    if not read_only:
        if st.session_state.get("user_role") == "admin":
            st.divider()
            _render_edit(campaign)
            if st.button("🗑 Delete This Campaign", key=f"a2_delete_{campaign['id']}{key_suffix}"):
                cc.delete_campaign(int(campaign["id"]))
                st.rerun()
        return

    # ── Results (Target Management "💰 Commission Results") ────────────────
    with st.spinner("Computing ranking (pooled 100001+100000)…"):
        sales = _load_pooled_sales()
        result = cc.compute_highest_product_sales_ranking(campaign, sales)

    m1, m2 = st.columns(2)
    m1.metric("Total Payout", _fmt_bdt(result["total_payout"]))
    m2.metric("Winners", f"{result['num_winners']}")

    ranking_df = result["ranking"]
    if ranking_df.empty:
        st.info("No sales activity in this window yet.")
        return

    recipient_label = result["recipient_type"].capitalize()
    disp = ranking_df.rename(columns={
        "rank": "Rank", "recipient": f"{recipient_label} Code", "recipient_name": recipient_label,
        "distinct_products": "Distinct Products", "total_qty": "Total Qty", "payout": "Payout (BDT)",
    })
    disp["Total Qty"] = disp["Total Qty"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
    disp["Payout (BDT)"] = disp["Payout (BDT)"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) and v > 0 else "—")
    st.dataframe(disp, width="stretch", hide_index=True)


# ── Top-level section entry point ───────────────────────────────────────────

def render(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    """`read_only=True` (Target Management's "💰 Commission Results") shows
    only the ranking result — no create/edit/delete controls, regardless of
    the viewer's own role. `read_only=False` (admin Commissions page) shows
    only setup, no ranking computation at all."""
    st.subheader("📦 A.2 — Highest Product Sales")
    st.caption(
        "Ranked by DISTINCT product count sold in the window (breadth, not volume) — "
        "the top-ranked recipients (salesman or customer) each get a fixed BDT payout "
        "for their rank position, admin-configured per campaign."
    )

    is_admin = (not read_only) and st.session_state.get("user_role") == "admin"
    if is_admin:
        _render_create()

    campaigns = _list_ranking_campaigns()
    if campaigns.empty:
        st.info("No ranking campaigns yet." + (" Create one above." if is_admin else " Ask an admin to create one."))
        return

    labels = {
        f"#{r.id} — {r.campaign_name} ({r.window_start} → {r.window_end})": r.id
        for r in campaigns.itertuples()
    }
    pick = st.selectbox("Select a campaign", list(labels.keys()), key=f"a2_select_campaign{key_suffix}")
    campaign = cc.get_campaign(labels[pick])
    if campaign:
        _render_detail(campaign, read_only=read_only, key_suffix=key_suffix)
