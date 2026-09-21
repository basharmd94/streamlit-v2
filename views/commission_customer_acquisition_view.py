# views/commission_customer_acquisition_view.py
# UI for the merged "Customer Acquisition" section -- two independent
# mechanisms under one admin picker, per explicit ask 2026-09-21: "this one
# merge with the New customer creation option which was planned before...
# 2 options similar to A.2 for unique customers and another option will be
# for new customer creation... on a per customer basis."
#
# Mode 1 -- Unique Customers (ranked): same shape as A.2 (Highest Product
# Sales) but ranks salesmen by DISTINCT CUSTOMER count instead of distinct
# product count -- the "Number of Unique Customers" idea noted for later in
# A.4's own closing bullet, picked up here. Salesman-only (see
# processing/commission_campaigns.py::compute_unique_customers_ranking).
#
# Mode 2 -- New Customer Creation: resolves B.5 from commission_tracking_
# design.md §B.5 (previously "needs design work") -- per-customer flat
# bonus, salesman-only (design doc's own "For future note" table already
# confirmed this). A customer counts as genuinely NEW only if their cacus
# row's own creation timestamp (ztime -- see core/queries.py::
# get_cacus_creation_detail for why NOT xdatecre/xdatefst) falls in the
# campaign window AND no OTHER customer code (any ZID/any date in the
# pooled population) shares one of its phone numbers -- explicit
# anti-gaming ask: "If phone number matches this will not be counted as a
# new customer... some customers have multiple phone numbers with a comma
# ... need to check all the phone numbers."
#
# Both modes pool 100001+100000 (same field sales team/shared customer
# codes as A.1/A.4/A.1b) and reuse the SAME commission_campaigns table
# (campaign_type "Unique Customers" / "New Customer Creation" respectively)
# -- no separate table, same convention as every other section. Plain
# admin-picked window_start/window_end (like A.2/B.1/3/4), not a pinned
# "reporting month" like A.1/A.4 -- neither mode re-derives its window from
# wall-clock today at view time, so there's no live-recompute risk to pin
# against.

from __future__ import annotations

import pandas as pd
import streamlit as st

from processing import commission_campaigns as cc

CAMPAIGN_TYPE_UNIQUE = "Unique Customers"
CAMPAIGN_TYPE_NEW = "New Customer Creation"
_ZIDS = ["100001", "100000"]
UC_MIN_WINNERS = 2
UC_MAX_WINNERS = 5
_ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}


def _fmt_bdt(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"৳{v:,.0f}"


def _clamp_min_session_date(key: str, min_v) -> bool:
    """Same pattern as every other section's window-end field -- a widget's
    persisted value can fall below a newly-raised min_value on a rerun;
    pre-clamp before instantiating the dependent date_input."""
    if key in st.session_state:
        cur = st.session_state[key]
        if cur < min_v:
            st.session_state[key] = min_v
            return True
    return False


# ── Shared pooled loaders ────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_sales(zid: str) -> pd.DataFrame:
    from core.analytics import Analytics
    from processing import common
    df = Analytics("sales", zid=zid, filters={}).data
    if df is None or df.empty:
        return pd.DataFrame()
    (df,) = common.data_copy_add_columns(df)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def _load_cacus_creation(zid: str) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("cacus_creation_detail", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_spnames(zid: str) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("prmst_area", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


def _pooled(loader) -> pd.DataFrame:
    frames = []
    for z in _ZIDS:
        df = loader(z)
        if df is not None and not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _spname_map() -> dict:
    df = _pooled(_load_spnames)
    if df.empty or "spid" not in df.columns or "spname" not in df.columns:
        return {}
    d = df.dropna(subset=["spid"]).drop_duplicates("spid")
    return dict(zip(d["spid"].astype(str), d["spname"]))


def _list_campaigns(campaign_type: str) -> pd.DataFrame:
    df = cc.list_campaigns()
    if df.empty:
        return df
    return df[df["campaign_type"] == campaign_type].reset_index(drop=True)


# ── Mode 1: Unique Customers (ranked) ───────────────────────────────────────

def _render_uc_form(existing: dict | None = None) -> None:
    is_edit = existing is not None
    key_prefix = f"uc_edit_{existing['id']}" if is_edit else "uc_new"

    name = st.text_input(
        "Campaign name", value=(existing["campaign_name"] if is_edit else ""), key=f"{key_prefix}_name",
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
    num_default = min(max(num_default, UC_MIN_WINNERS), UC_MAX_WINNERS)
    num_winners = st.number_input(
        f"Number of winners ({UC_MIN_WINNERS}-{UC_MAX_WINNERS})",
        min_value=UC_MIN_WINNERS, max_value=UC_MAX_WINNERS,
        value=num_default, step=1, key=f"{key_prefix}_numwinners",
    )

    existing_payouts = existing_config.get("payouts_by_rank", []) if is_edit else []
    st.caption("Payout amount per rank position (ranked by distinct CUSTOMER count sold to, highest first):")
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

    btn_label = "💾 Save Changes" if is_edit else "✅ Create Unique Customers Campaign"
    if st.button(btn_label, key=f"{key_prefix}_submit_btn", disabled=not can_submit):
        product_rates_payload = {"num_winners": int(num_winners), "payouts_by_rank": payouts_by_rank}
        if is_edit:
            ok = cc.update_campaign(
                campaign_id=int(existing["id"]), campaign_name=name.strip(), campaign_type=CAMPAIGN_TYPE_UNIQUE,
                recipient_type="salesman", product_rates=product_rates_payload, cap=None,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups={}, uptick_baseline_months=3, notes=notes,
            )
            if ok:
                st.success("Unique Customers campaign updated.", icon="✅")
                st.rerun()
            else:
                st.error("Could not update campaign.")
        else:
            cid = cc.create_campaign(
                campaign_name=name.strip(), campaign_type=CAMPAIGN_TYPE_UNIQUE, recipient_type="salesman",
                product_rates=product_rates_payload, cap=None,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups={}, uptick_baseline_months=3,
                created_by=st.session_state.get("username", ""), notes=notes,
            )
            if cid:
                st.success(f"Unique Customers campaign #{cid} created.", icon="✅")
                st.rerun()
            else:
                st.error("Could not create campaign.")


def _render_uc_create() -> None:
    with st.expander("➕ Create New Unique Customers Campaign", expanded=False):
        _render_uc_form()


def _render_uc_edit(campaign: dict) -> None:
    with st.expander("✏️ Edit Unique Customers Campaign", expanded=False):
        _render_uc_form(existing=campaign)


def _render_uc_detail(campaign: dict, read_only: bool, key_suffix: str) -> None:
    config = campaign.get("product_rates") or {}
    num_winners = config.get("num_winners")
    payouts_by_rank = config.get("payouts_by_rank", [])

    today = pd.Timestamp.today().normalize().date()
    window_end = pd.Timestamp(campaign["window_end"]).date()
    status_badge = "🟢 Ongoing" if window_end >= today else "🔴 Closed"

    st.markdown(f"#### {campaign['campaign_name']} — {status_badge}")
    st.caption(
        f"Unique Customers (ranked) · Window: {campaign['window_start']} → {campaign['window_end']} · "
        f"Pooled 100001+100000"
    )
    payouts_line = " · ".join(
        f"{_ORDINAL.get(i + 1, f'{i + 1}th')}: {_fmt_bdt(a)}" for i, a in enumerate(payouts_by_rank)
    )
    st.caption(f"{num_winners} winner(s) — {payouts_line}")

    if not read_only:
        if st.session_state.get("user_role") == "admin":
            st.divider()
            _render_uc_edit(campaign)
            if st.button("🗑 Delete This Campaign", key=f"uc_delete_{campaign['id']}{key_suffix}"):
                cc.delete_campaign(int(campaign["id"]))
                st.rerun()
        return

    # ── Results (Target Management "💰 Commission Results") ────────────────
    with st.spinner("Computing ranking (pooled 100001+100000)…"):
        sales = _pooled(_load_sales)
        result = cc.compute_unique_customers_ranking(campaign, sales)

    m1, m2 = st.columns(2)
    m1.metric("Total Payout", _fmt_bdt(result["total_payout"]))
    m2.metric("Winners", f"{result['num_winners']}")

    ranking_df = result["ranking"]
    if ranking_df.empty:
        st.info("No sales activity in this window yet.")
        return

    disp = ranking_df.rename(columns={
        "rank": "Rank", "recipient": "Salesman Code", "recipient_name": "Salesman",
        "distinct_customers": "Distinct Customers", "total_qty": "Total Qty", "payout": "Payout (BDT)",
    })
    disp["Total Qty"] = disp["Total Qty"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
    disp["Payout (BDT)"] = disp["Payout (BDT)"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) and v > 0 else "—")
    st.dataframe(disp, width="stretch", hide_index=True)


def _render_unique_customers(read_only: bool, key_suffix: str) -> None:
    st.caption(
        "Ranked by DISTINCT CUSTOMER count sold to in the window (breadth, not volume) — the "
        "top-ranked salesmen each get a fixed BDT payout for their rank position, admin-configured "
        "per campaign. Same shape as A.2 (Highest Product Sales), salesman-only."
    )
    is_admin = (not read_only) and st.session_state.get("user_role") == "admin"
    if is_admin:
        _render_uc_create()

    campaigns = _list_campaigns(CAMPAIGN_TYPE_UNIQUE)
    if campaigns.empty:
        st.info(
            "No Unique Customers campaigns yet."
            + (" Create one above." if is_admin else " Ask an admin to create one.")
        )
        return

    labels = {
        f"#{r.id} — {r.campaign_name} ({r.window_start} → {r.window_end})": r.id
        for r in campaigns.itertuples()
    }
    pick = st.selectbox("Select a campaign", list(labels.keys()), key=f"uc_select_campaign{key_suffix}")
    campaign = cc.get_campaign(labels[pick])
    if campaign:
        _render_uc_detail(campaign, read_only=read_only, key_suffix=key_suffix)


# ── Mode 2: New Customer Creation (per-customer flat bonus, B.5) ───────────

def _render_nc_form(existing: dict | None = None) -> None:
    is_edit = existing is not None
    key_prefix = f"nc_edit_{existing['id']}" if is_edit else "nc_new"

    name = st.text_input(
        "Campaign name", value=(existing["campaign_name"] if is_edit else ""), key=f"{key_prefix}_name",
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
    bonus_default = float(existing_config.get("bonus_amount", 0.0)) if is_edit else 0.0
    bonus_amount = st.number_input(
        "Bonus amount (BDT) per genuinely new customer created",
        min_value=0.0, step=100.0, value=bonus_default, key=f"{key_prefix}_bonus",
    )
    st.caption(
        "A customer counts only if their record was created in this window AND no phone number on "
        "file matches any other existing customer (checked across every number in both phone "
        "fields, comma-separated numbers included) — otherwise it's flagged as a likely re-entry of "
        "an existing customer and isn't paid."
    )

    notes = st.text_area(
        "Notes (optional)", value=(existing.get("notes") or "" if is_edit else ""), key=f"{key_prefix}_notes",
    )

    can_submit = bool(name.strip()) and bonus_amount > 0
    if bonus_amount <= 0:
        st.caption("Bonus amount must be > 0.")

    btn_label = "💾 Save Changes" if is_edit else "✅ Create New Customer Campaign"
    if st.button(btn_label, key=f"{key_prefix}_submit_btn", disabled=not can_submit):
        product_rates_payload = {"bonus_amount": float(bonus_amount)}
        if is_edit:
            ok = cc.update_campaign(
                campaign_id=int(existing["id"]), campaign_name=name.strip(), campaign_type=CAMPAIGN_TYPE_NEW,
                recipient_type="salesman", product_rates=product_rates_payload, cap=None,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups={}, uptick_baseline_months=3, notes=notes,
            )
            if ok:
                st.success("New Customer Creation campaign updated.", icon="✅")
                st.rerun()
            else:
                st.error("Could not update campaign.")
        else:
            cid = cc.create_campaign(
                campaign_name=name.strip(), campaign_type=CAMPAIGN_TYPE_NEW, recipient_type="salesman",
                product_rates=product_rates_payload, cap=None,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups={}, uptick_baseline_months=3,
                created_by=st.session_state.get("username", ""), notes=notes,
            )
            if cid:
                st.success(f"New Customer Creation campaign #{cid} created.", icon="✅")
                st.rerun()
            else:
                st.error("Could not create campaign.")


def _render_nc_create() -> None:
    with st.expander("➕ Create New Customer Creation Campaign", expanded=False):
        _render_nc_form()


def _render_nc_edit(campaign: dict) -> None:
    with st.expander("✏️ Edit New Customer Creation Campaign", expanded=False):
        _render_nc_form(existing=campaign)


def _render_nc_detail(campaign: dict, read_only: bool, key_suffix: str) -> None:
    config = campaign.get("product_rates") or {}
    bonus_amount = float(config.get("bonus_amount", 0))

    today = pd.Timestamp.today().normalize().date()
    window_end = pd.Timestamp(campaign["window_end"]).date()
    status_badge = "🟢 Ongoing" if window_end >= today else "🔴 Closed"

    st.markdown(f"#### {campaign['campaign_name']} — {status_badge}")
    st.caption(
        f"New Customer Creation · Window: {campaign['window_start']} → {campaign['window_end']} · "
        f"Pooled 100001+100000 · Bonus: {_fmt_bdt(bonus_amount)} per genuinely new customer"
    )

    if not read_only:
        if st.session_state.get("user_role") == "admin":
            st.divider()
            _render_nc_edit(campaign)
            if st.button("🗑 Delete This Campaign", key=f"nc_delete_{campaign['id']}{key_suffix}"):
                cc.delete_campaign(int(campaign["id"]))
                st.rerun()
        return

    # ── Results (Target Management "💰 Commission Results") ────────────────
    with st.spinner("Checking new customer creation (pooled 100001+100000)…"):
        cacus_100001 = _load_cacus_creation("100001")
        cacus_100000 = _load_cacus_creation("100000")
        names = _spname_map()
        result = cc.compute_new_customer_creation(campaign, cacus_100001, cacus_100000, names)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Payout", _fmt_bdt(result["total_payout"]))
    m2.metric("New (Paid)", f"{result['new_count']}")
    m3.metric("Excluded", f"{result['excluded_count']}")

    rows = result["rows"]
    if rows.empty:
        st.info("No customer records created in this window yet.")
        return

    disp = rows.rename(columns={
        "cusid": "Cust Code", "cusname": "Customer", "spid": "Salesman Code", "spname": "Salesman",
        "created_at": "Created", "phone_numbers_display": "Phone Number(s)", "status": "Status",
        "payout": "Payout (BDT)",
    })
    disp["Created"] = pd.to_datetime(disp["Created"]).dt.strftime("%Y-%m-%d")
    disp["Payout (BDT)"] = disp["Payout (BDT)"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) and v > 0 else "—")
    st.dataframe(disp, width="stretch", hide_index=True)

    csv = rows.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download full list (CSV)", csv, file_name=f"new_customer_creation_{campaign['id']}.csv",
        mime="text/csv", key=f"nc_csv_{campaign['id']}{key_suffix}",
    )


def _render_new_customer_creation(read_only: bool, key_suffix: str) -> None:
    st.caption(
        "Pays a flat BDT bonus per genuinely NEW customer a salesman creates in the window — on a "
        "per-customer basis, not a ranking. A candidate is excluded if any of its phone numbers "
        "already belongs to an existing customer (likely a re-entry, not a real new customer) or if "
        "it has no salesman on file to pay."
    )
    is_admin = (not read_only) and st.session_state.get("user_role") == "admin"
    if is_admin:
        _render_nc_create()

    campaigns = _list_campaigns(CAMPAIGN_TYPE_NEW)
    if campaigns.empty:
        st.info(
            "No New Customer Creation campaigns yet."
            + (" Create one above." if is_admin else " Ask an admin to create one.")
        )
        return

    labels = {
        f"#{r.id} — {r.campaign_name} ({r.window_start} → {r.window_end})": r.id
        for r in campaigns.itertuples()
    }
    pick = st.selectbox("Select a campaign", list(labels.keys()), key=f"nc_select_campaign{key_suffix}")
    campaign = cc.get_campaign(labels[pick])
    if campaign:
        _render_nc_detail(campaign, read_only=read_only, key_suffix=key_suffix)


# ── Top-level section entry point ───────────────────────────────────────────

def render(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    """`read_only=True` (Target Management's "💰 Commission Results") shows
    only results — no create/edit/delete controls. `read_only=False` (admin
    Commissions page) shows only setup, no computation ever runs there."""
    st.subheader("🆕 Customer Acquisition")
    mode = st.radio(
        "Mode", ["🎯 Unique Customers", "🆕 New Customer Creation"],
        horizontal=True, key=f"ca_mode{key_suffix}",
    )
    st.divider()
    if mode == "🎯 Unique Customers":
        _render_unique_customers(read_only, key_suffix)
    else:
        _render_new_customer_creation(read_only, key_suffix)
