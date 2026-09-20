# views/commission_customer_best_performer_view.py
# UI for "Best Performer — Customers" — a customer-ranking sibling to A.1
# (Best Performer, salesman-only). Not in the original commission_tracking_
# design.md scope — designed 2026-09-20 per explicit ask: "same logic,
# everything [as the salesman version]... but this would follow the
# customer scoring that is already in marketing analysis."
#
# See processing/commission_campaigns.py::compute_customer_best_performer_ranking
# for the reporting-YEAR (not month) departure from A.1 and why. This file
# is UI only.
#
# ── ZID pooling — corrected 2026-09-20, same day, right after first
# shipping this with single-ZID scoping. Original assumption: customer
# codes aren't unique across ZIDs, so never pool. **Wrong for 100001/100000
# specifically** — the user ran a real audit: out of ~10,000 customers,
# 100001 (HMBR/Gulshan Trading) and 100000 (GI Corporation) share the SAME
# cacus customer code 99.9% of the time (~10 mismatches, none of them
# current/active customers). Business reason: "when a salesman goes to a
# customer, the customer doesn't understand the difference between 100,000
# and 100,001 — they just buy products from us directly." So scoring must
# pool 100001+100000 together for this to be fair — same pooling A.1
# (salesmen) and B.1/3/4 already use, for the same underlying reason (one
# shared field sales team/customer relationship). **100005 (Zepto) stays
# separate** — a genuinely independent consumer brand, no shared codes,
# explicitly confirmed to stay its own group.
#
# _ZID_GROUPS below is the single source of truth for this grouping —
# whichever ZID is active when a campaign is created resolves to its
# group, and the group's full zid LIST (not a single zid) is what gets
# pinned into product_rates and used for scoring from then on.
#
# Reuses the SAME commission_campaigns table as every other commission
# section — CAMPAIGN_TYPE below is the free-text value that distinguishes
# these rows from A.1/A.2/B.1/3/4's own, so each section's own campaign
# picker only ever lists its own rows.

from __future__ import annotations

import pandas as pd
import streamlit as st

from processing import commission_campaigns as cc
from processing import common, marketing as mkt
from views.marketing import _load_ar_balance, _load_cacus

CAMPAIGN_TYPE = "Customer Best Performer"
MIN_WINNERS = 2
MAX_WINNERS = 100

_ZID_GROUPS = {
    "100001": {"label": "HMBR + GI Corporation (pooled)", "zids": ["100001", "100000"]},
    "100000": {"label": "HMBR + GI Corporation (pooled)", "zids": ["100001", "100000"]},
    "100005": {"label": "Zepto Chemicals", "zids": ["100005"]},
}
_PROJ_BY_ZID = {"100001": "GULSHAN TRADING", "100000": "GI Corporation", "100005": "Zepto Chemicals"}


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


def _group_for_zid(zid: str) -> dict:
    return _ZID_GROUPS.get(str(zid), {"label": str(zid), "zids": [str(zid)]})


def _label_for_zids(zids: list) -> str:
    """Derives the display label live from a zids list, rather than
    storing it redundantly — a campaign's own group can only ever be one
    of the two real _ZID_GROUPS combinations, so this can never drift out
    of sync with _ZID_GROUPS the way a separately-stored label could."""
    key = sorted(str(z) for z in zids)
    for group in _ZID_GROUPS.values():
        if sorted(group["zids"]) == key:
            return group["label"]
    return ", ".join(key) if key else "?"


def _reporting_year_choices(today: pd.Timestamp, years_forward: int = 5) -> list:
    """[year, ...] current year through `years_forward` FUTURE ones —
    current year or later only, same no-retroactive-setup rule as A.1's
    reporting month, applied at the year level (confirmed 2026-09-20: "I
    can't set it for a [period] that already passed")."""
    return [today.year + i for i in range(years_forward + 1)]


# ── Pooled data loaders (own cache, per (zid, year) — pooled across a
# group's zids at call time, same pattern A.1's own pooled loaders use) ──

@st.cache_data(show_spinner=False, ttl=3600)
def _load_sales_year(zid: str, year: int) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("sales", zid=zid, filters={"year": [year]}).data
    if df is None or df.empty:
        return pd.DataFrame()
    (df,) = common.data_copy_add_columns(df.copy())
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def _load_collection_year(zid: str, year: int) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("collection", zid=zid, filters={"year": [year]}).data
    return df if df is not None else pd.DataFrame()


def _pooled_sales_year(zids: list, year: int) -> pd.DataFrame:
    frames = [df for z in zids if not (df := _load_sales_year(z, year)).empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pooled_collection_year(zids: list, year: int) -> pd.DataFrame:
    frames = [df for z in zids if not (df := _load_collection_year(z, year)).empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pooled_ar_balance(zids: list) -> pd.DataFrame:
    frames = []
    for z in zids:
        df = _load_ar_balance(z, _PROJ_BY_ZID.get(z, ""))
        if df is not None and not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pooled_cacus(zids: list) -> pd.DataFrame:
    frames = [df for z in zids if not (df := _load_cacus(z)).empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["cusid"], keep="first")


def _score_customers(zids: list, year: int) -> pd.DataFrame:
    """Calls processing.marketing.build_customer_marketing_table exactly as
    Marketing Analysis's own Customer Scoring does — same engine, same
    weights, unmodified. Only the data-loading (this group's zid(s), one
    reporting year, pooled when the group has more than one zid) is
    specific to this campaign type."""
    sales_df = _pooled_sales_year(zids, year)
    collection_df = _pooled_collection_year(zids, year)
    ar_df = _pooled_ar_balance(zids)
    cacus_df = _pooled_cacus(zids)
    return mkt.build_customer_marketing_table(
        sales_df=sales_df, collection_df=collection_df, ar_df=ar_df,
        selected_years=(year,), cacus_df=cacus_df if not cacus_df.empty else None,
    )


# ── Create / Edit (admin-only) — shared body ────────────────────────────────

def _render_form(zid: str = "", existing: dict | None = None) -> None:
    is_edit = existing is not None
    key_prefix = f"a1c_edit_{existing['id']}" if is_edit else "a1c_new"

    existing_config = (existing.get("product_rates") or {}) if is_edit else {}
    # The campaign's own ZID GROUP is pinned at creation and NEVER changes
    # on edit, even if the admin has since switched businesses in the
    # sidebar — same "pin at setup, don't let a later view silently move
    # it" discipline as A.1's reporting month.
    campaign_zids = (
        [str(z) for z in existing_config.get("zids", [])] if is_edit else _group_for_zid(zid)["zids"]
    )
    group_label = _label_for_zids(campaign_zids)

    name = st.text_input(
        "Campaign name", value=(existing["campaign_name"] if is_edit else ""), key=f"{key_prefix}_name",
    )

    today = pd.Timestamp.today().normalize()
    year_opts = _reporting_year_choices(today)
    if is_edit:
        existing_year = int(existing_config.get("reporting_year", today.year))
        if existing_year not in year_opts:
            # The campaign's own reporting year has already closed — keep
            # it selected/available (not offered as a fresh choice, just
            # preserved) so Save can't silently reassign it.
            year_opts = [existing_year] + year_opts
        default_year = existing_year
    else:
        default_year = year_opts[0]
    reporting_year = st.selectbox(
        "Reporting year — the year this campaign's customer ranking is for",
        year_opts, index=year_opts.index(default_year), key=f"{key_prefix}_year",
    )
    st.caption(
        f"Current year or later only — never a year that's already passed. Business group: "
        f"**{group_label}** (fixed to whichever business was active when this campaign was created "
        f"— not editable afterward). Uses the SAME Customer Score already shown in Marketing "
        f"Analysis → 📊 Customer Scoring, unmodified."
    )

    winners_default = int(existing_config.get("num_winners", 10)) if is_edit else 10
    winners_default = min(max(winners_default, MIN_WINNERS), MAX_WINNERS)
    num_winners = st.number_input(
        f"Number of winners ({MIN_WINNERS}-{MAX_WINNERS})", min_value=MIN_WINNERS, max_value=MAX_WINNERS,
        value=winners_default, step=1, key=f"{key_prefix}_numwinners",
    )

    existing_payouts = existing_config.get("payouts_by_rank", []) if is_edit else []
    st.caption("Payout amount per rank position (ranked by Customer Score, highest first):")
    payouts_by_rank = []
    n_cols = min(int(num_winners), 10)
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

    btn_label = "💾 Save Changes" if is_edit else "✅ Create Customer Best Performer Campaign"
    if st.button(btn_label, key=f"{key_prefix}_submit_btn", disabled=not can_submit):
        product_rates_payload = {
            "zids": campaign_zids, "reporting_year": int(reporting_year),
            "num_winners": int(num_winners), "payouts_by_rank": payouts_by_rank,
        }
        # window_start/window_end are informational display bounds only
        # here (the whole reporting year) — computation always reads
        # reporting_year straight out of product_rates, not these dates.
        window_start = pd.Timestamp(int(reporting_year), 1, 1).date()
        window_end = pd.Timestamp(int(reporting_year), 12, 31).date()
        if is_edit:
            ok = cc.update_campaign(
                campaign_id=int(existing["id"]), campaign_name=name.strip(), campaign_type=CAMPAIGN_TYPE,
                recipient_type="customer", product_rates=product_rates_payload, cap=None,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups={}, uptick_baseline_months=3, notes=notes,
            )
            if ok:
                st.success("Customer Best Performer campaign updated.", icon="✅")
                st.rerun()
            else:
                st.error("Could not update campaign.")
        else:
            cid = cc.create_campaign(
                campaign_name=name.strip(), campaign_type=CAMPAIGN_TYPE, recipient_type="customer",
                product_rates=product_rates_payload, cap=None,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups={}, uptick_baseline_months=3,
                created_by=st.session_state.get("username", ""), notes=notes,
            )
            if cid:
                st.success(f"Customer Best Performer campaign #{cid} created.", icon="✅")
                st.rerun()
            else:
                st.error("Could not create campaign.")


def _render_create(zid: str) -> None:
    with st.expander("➕ Create New Customer Best Performer Campaign", expanded=False):
        _render_form(zid=zid)


def _render_edit(campaign: dict) -> None:
    with st.expander("✏️ Edit Customer Best Performer Campaign", expanded=False):
        _render_form(existing=campaign)


def _list_campaigns() -> pd.DataFrame:
    df = cc.list_campaigns()
    if df.empty:
        return df
    return df[df["campaign_type"] == CAMPAIGN_TYPE].reset_index(drop=True)


# ── Campaign detail / results view ──────────────────────────────────────────

def _render_detail(campaign: dict, read_only: bool, key_suffix: str) -> None:
    config = campaign.get("product_rates") or {}
    campaign_zids = [str(z) for z in config.get("zids", [])]
    group_label = _label_for_zids(campaign_zids)
    reporting_year = int(config.get("reporting_year") or pd.Timestamp.today().year)
    num_winners = config.get("num_winners")
    payouts_by_rank = config.get("payouts_by_rank", [])

    today = pd.Timestamp.today().normalize()
    if reporting_year == today.year:
        status_note = "still tracking live"
    elif reporting_year < today.year:
        status_note = "closed — final numbers"
    else:
        status_note = "not started yet"

    st.markdown(f"### {campaign['campaign_name']}")
    st.caption(
        f"Customer Best Performer (ranked by Customer Score) · Business group: {group_label} "
        f"· Reporting year {reporting_year} ({status_note})"
    )
    payouts_line = " · ".join(
        f"{_ordinal(i + 1)}: {_fmt_bdt(a)}" for i, a in enumerate(payouts_by_rank)
    )
    st.caption(f"{num_winners} winner(s) — {payouts_line}")

    if not read_only:
        if st.session_state.get("user_role") == "admin":
            st.divider()
            _render_edit(campaign)
            if st.button("🗑 Delete This Campaign", key=f"a1c_delete_{campaign['id']}{key_suffix}"):
                cc.delete_campaign(int(campaign["id"]))
                st.rerun()
        return

    # ── Results (Target Management "💰 Commission Results") ────────────────
    with st.spinner(f"Scoring {group_label} customers for {reporting_year}…"):
        score_df = _score_customers(campaign_zids, reporting_year)
        result = cc.compute_customer_best_performer_ranking(campaign, score_df)

    m1, m2 = st.columns(2)
    m1.metric("Total Payout", _fmt_bdt(result["total_payout"]))
    m2.metric("Winners", f"{result['num_winners']}")

    ranking_df = result["ranking"]
    if ranking_df.empty:
        st.info("No scored customers for this reporting year yet.")
        return

    disp = ranking_df.rename(columns={
        "rank": "Rank", "recipient": "Customer Code", "recipient_name": "Customer",
        "composite_score": "Score", "payout": "Payout (BDT)",
    })
    disp["Score"] = disp["Score"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
    disp["Payout (BDT)"] = disp["Payout (BDT)"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) and v > 0 else "—")
    st.dataframe(disp, width="stretch", hide_index=True)


# ── Top-level section entry point ───────────────────────────────────────────

def render(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    """`read_only=True` (Target Management's "💰 Commission Results") shows
    only the ranking result — no create/edit/delete controls, regardless of
    the viewer's own role. `read_only=False` (admin Commissions page) shows
    only setup, no ranking computation at all.

    `zid` here only matters for CREATING a new campaign (which business
    GROUP it defaults to, via _group_for_zid) — the campaign picker itself
    lists every Customer Best Performer campaign regardless of which
    business is currently active in the sidebar, since each campaign
    carries its own pinned zid group."""
    st.subheader("🏆 Best Performer — Customers")
    st.caption(
        "Ranks customers by their existing Customer Score — the EXACT same engine as Marketing "
        "Analysis's own 📊 Customer Scoring (unmodified) — for an admin-chosen reporting year, and "
        "pays a fixed BDT amount per rank position to the top 2-100 customers. HMBR (100001) and GI "
        "Corporation (100000) are pooled together (their customers share the same code — confirmed "
        "by a real audit, and a customer doesn't distinguish between the two businesses); Zepto "
        "(100005) is scored separately. Pinned to whichever business group was active when the "
        "campaign was created."
    )

    is_admin = (not read_only) and st.session_state.get("user_role") == "admin"
    if is_admin:
        _render_create(zid)

    campaigns = _list_campaigns()
    if campaigns.empty:
        st.info(
            "No Customer Best Performer campaigns yet."
            + (" Create one above." if is_admin else " Ask an admin to create one.")
        )
        return

    labels = {}
    for r in campaigns.itertuples():
        cfg = r.product_rates or {}
        c_zids = [str(z) for z in cfg.get("zids", [])]
        c_year = cfg.get("reporting_year", "?")
        labels[f"#{r.id} — {r.campaign_name} ({_label_for_zids(c_zids)}, {c_year})"] = r.id
    pick = st.selectbox("Select a campaign", list(labels.keys()), key=f"a1c_select_campaign{key_suffix}")
    campaign = cc.get_campaign(labels[pick])
    if campaign:
        _render_detail(campaign, read_only=read_only, key_suffix=key_suffix)
