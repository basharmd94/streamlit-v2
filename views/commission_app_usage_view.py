# views/commission_app_usage_view.py
# UI for A.4 — App Usage Commission. Resolves A.4's original "deferred by
# design" status in commission_tracking_design.md: the user supplied the
# real mobile ERP API data map (mobile_order_api_data_map.md/.json,
# provided 2026-09-21) and, after discussion, specified the exact 5
# components + weights + a threshold payout mechanism. See
# processing/commission_campaigns.py::compute_app_usage_scores /
# compute_app_usage_bonus for the full engine and the real-data findings
# that shaped it — this file is UI only.
#
# Reporting month, current-or-future only, pinned at setup — same rule and
# same reasoning as A.1 (confirmed 2026-09-20 there, reused here rather
# than re-litigated): "I can only set this current month and future
# months... I can't set it for a month that already passed." Single month
# only (no "months to average") — this is a monthly compliance bonus, not
# a multi-month-averaged ranking.
#
# Pooled 100001+100000 only, same scope as A.1 (not A.1b's per-ZID-group
# design) — confirmed against real opmob data: 346,606 orders on 100001,
# 31,994 on 100000, vs only 63 on 100005 (Zepto). App usage is fundamentally
# a salesman behavior metric like A.1, not a per-business customer metric
# like A.1b, so it follows A.1's precedent.

from __future__ import annotations

import pandas as pd
import streamlit as st

from processing import commission_campaigns as cc

CAMPAIGN_TYPE = "App Usage"
_ZIDS = ["100001", "100000"]


def _fmt_bdt(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"৳{v:,.0f}"


def _anchor_month_choices(today: pd.Timestamp, months_forward: int = 12) -> list:
    """[(label, year, month), ...] earliest first: the current month plus
    up to `months_forward` FUTURE ones — current month or later only, same
    rule as A.1's reporting month (confirmed 2026-09-20: no retroactive
    setup)."""
    cur = pd.Timestamp(today.year, today.month, 1)
    return [
        ((cur + pd.DateOffset(months=i)).strftime("%b %Y"),
         int((cur + pd.DateOffset(months=i)).year),
         int((cur + pd.DateOffset(months=i)).month))
        for i in range(months_forward + 1)
    ]


# ── Pooled data loaders (own cache, per (zid, year, month)) ─────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_orders(zid: str, year: int, month: int) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("app_usage_orders", zid=zid, filters={"year": [year], "month": [month]}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_returns(zid: str, year: int, month: int) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("app_usage_returns", zid=zid, filters={"year": [year], "month": [month]}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_delivery_orders(zid: str, year: int, month: int) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("app_usage_delivery_orders", zid=zid, filters={"year": [year], "month": [month]}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_glpmt_all(zid: str) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("glpmt", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_spnames(zid: str) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("prmst_area", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


def _pooled(loader, *args) -> pd.DataFrame:
    frames = []
    for z in _ZIDS:
        df = loader(z, *args)
        if df is not None and not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _spname_map() -> dict:
    df = _pooled(lambda z: _load_spnames(z))
    if df.empty or "spid" not in df.columns or "spname" not in df.columns:
        return {}
    d = df.dropna(subset=["spid"]).drop_duplicates("spid")
    return dict(zip(d["spid"].astype(str), d["spname"]))


def _compute_results(anchor_year: int, anchor_month: int) -> dict:
    today = pd.Timestamp.today().normalize()
    month_start = pd.Timestamp(anchor_year, anchor_month, 1)
    month_end = month_start + pd.offsets.MonthEnd(0)

    orders_df = _pooled(_load_orders, anchor_year, anchor_month)
    returns_df = _pooled(_load_returns, anchor_year, anchor_month)
    delivery_df = _pooled(_load_delivery_orders, anchor_year, anchor_month)
    glpmt_df = _pooled(lambda z: _load_glpmt_all(z))

    return orders_df, returns_df, delivery_df, glpmt_df, month_start, month_end, today


# ── Create / Edit (admin-only) — shared body ────────────────────────────────

def _render_form(existing: dict | None = None) -> None:
    is_edit = existing is not None
    key_prefix = f"a4_edit_{existing['id']}" if is_edit else "a4_new"

    existing_config = (existing.get("product_rates") or {}) if is_edit else {}

    name = st.text_input(
        "Campaign name", value=(existing["campaign_name"] if is_edit else ""), key=f"{key_prefix}_name",
    )

    today = pd.Timestamp.today().normalize()
    month_opts = _anchor_month_choices(today)
    if is_edit:
        existing_end = pd.Timestamp(existing["window_end"])
        already_offered = any(y == existing_end.year and m == existing_end.month for _l, y, m in month_opts)
        if not already_offered:
            month_opts = [(existing_end.strftime("%b %Y"), int(existing_end.year), int(existing_end.month))] + month_opts
        default_label = existing_end.strftime("%b %Y")
    else:
        default_label = month_opts[0][0]
    month_labels = [lbl for lbl, _y, _m in month_opts]
    sel_label = st.selectbox(
        "Reporting month — the month this bonus is for",
        month_labels, index=month_labels.index(default_label), key=f"{key_prefix}_month",
    )
    anchor_year, anchor_month = next((y, m) for lbl, y, m in month_opts if lbl == sel_label)
    st.caption(
        "Current month or later only — never a month that's already passed. Tracks live while "
        "this is the current, ongoing month; once it closes, results stay fixed at that month's "
        "final numbers. Pooled across 100001+100000 (same scope as A.1 — Zepto's mobile-app usage "
        "is negligible)."
    )

    threshold_default = float(existing_config.get("threshold", 90.0)) if is_edit else 90.0
    threshold = st.number_input(
        "Score threshold (0-100) — a salesman must score ABOVE this to qualify",
        min_value=0.0, max_value=100.0, step=1.0, value=threshold_default, key=f"{key_prefix}_threshold",
    )
    bonus_default = float(existing_config.get("bonus_amount", 0.0)) if is_edit else 0.0
    bonus_amount = st.number_input(
        "Bonus amount (BDT) — same flat amount for every salesman who qualifies",
        min_value=0.0, step=100.0, value=bonus_default, key=f"{key_prefix}_bonus",
    )

    notes = st.text_area(
        "Notes (optional)", value=(existing.get("notes") or "" if is_edit else ""), key=f"{key_prefix}_notes",
    )

    can_submit = bool(name.strip()) and bonus_amount > 0
    if bonus_amount <= 0:
        st.caption("Bonus amount must be > 0.")

    btn_label = "💾 Save Changes" if is_edit else "✅ Create App Usage Campaign"
    if st.button(btn_label, key=f"{key_prefix}_submit_btn", disabled=not can_submit):
        product_rates_payload = {"threshold": float(threshold), "bonus_amount": float(bonus_amount)}
        window_start = pd.Timestamp(anchor_year, anchor_month, 1).date()
        window_end = (pd.Timestamp(anchor_year, anchor_month, 1) + pd.offsets.MonthEnd(0)).date()
        if is_edit:
            ok = cc.update_campaign(
                campaign_id=int(existing["id"]), campaign_name=name.strip(), campaign_type=CAMPAIGN_TYPE,
                recipient_type="salesman", product_rates=product_rates_payload, cap=None,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups={}, uptick_baseline_months=3, notes=notes,
            )
            if ok:
                st.success("App Usage campaign updated.", icon="✅")
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
                st.success(f"App Usage campaign #{cid} created.", icon="✅")
                st.rerun()
            else:
                st.error("Could not create campaign.")


def _render_create() -> None:
    with st.expander("➕ Create New App Usage Campaign", expanded=False):
        _render_form()


def _render_edit(campaign: dict) -> None:
    with st.expander("✏️ Edit App Usage Campaign", expanded=False):
        _render_form(existing=campaign)


def _list_campaigns() -> pd.DataFrame:
    df = cc.list_campaigns()
    if df.empty:
        return df
    return df[df["campaign_type"] == CAMPAIGN_TYPE].reset_index(drop=True)


# ── Campaign detail / results view ──────────────────────────────────────────

def _render_detail(campaign: dict, read_only: bool, key_suffix: str) -> None:
    config = campaign.get("product_rates") or {}
    threshold = float(config.get("threshold", 90))
    bonus_amount = float(config.get("bonus_amount", 0))
    window_end = pd.Timestamp(campaign["window_end"])
    anchor_year, anchor_month = int(window_end.year), int(window_end.month)

    today = pd.Timestamp.today().normalize()
    if anchor_year == today.year and anchor_month == today.month:
        status_note = "still tracking live"
    elif pd.Timestamp(anchor_year, anchor_month, 1) < pd.Timestamp(today.year, today.month, 1):
        status_note = "closed — final numbers"
    else:
        status_note = "not started yet"

    st.markdown(f"### {campaign['campaign_name']}")
    st.caption(
        f"App Usage Commission · Reporting month {window_end.strftime('%B %Y')} ({status_note})"
    )
    st.caption(f"Score threshold: > {threshold:.0f} · Bonus: {_fmt_bdt(bonus_amount)} per qualifying salesman")

    if not read_only:
        if st.session_state.get("user_role") == "admin":
            st.divider()
            _render_edit(campaign)
            if st.button("🗑 Delete This Campaign", key=f"a4_delete_{campaign['id']}{key_suffix}"):
                cc.delete_campaign(int(campaign["id"]))
                st.rerun()
        return

    # ── Results (Target Management "💰 Commission Results") ────────────────
    with st.spinner(f"Scoring app usage for {window_end.strftime('%B %Y')} (pooled 100001+100000)…"):
        orders_df, returns_df, delivery_df, glpmt_df, month_start, month_end, today_ts = _compute_results(
            anchor_year, anchor_month
        )
        scores_df = cc.compute_app_usage_scores(
            orders_df, returns_df, delivery_df, glpmt_df, month_start, month_end, today_ts,
        )
        result = cc.compute_app_usage_bonus(campaign, scores_df)
        names = _spname_map()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Payout", _fmt_bdt(result["total_payout"]))
    m2.metric("Qualified", f"{result['qualified_count']}")
    m3.metric("Scored Salesmen", f"{len(result['scores'])}")

    scores = result["scores"]
    if scores.empty:
        st.info("No salesmen placed any orders via the app for this reporting month yet.")
        return

    disp = scores.copy()
    disp["Salesman"] = disp["spid"].map(names).fillna("")
    disp = disp.rename(columns={
        "spid": "Salesman Code", "score": "Score", "orders_count": "Orders",
        "gps_fill_rate": "GPS Fill %", "gps_distinct_rate": "GPS Distinct %",
        "return_stuck_rate": "Return Stuck %", "promised_pay_rate": "Promised Pay %",
        "collection_count": "Collections Logged", "qualified": "Qualified", "payout": "Payout (BDT)",
    })
    cols = [
        "Salesman Code", "Salesman", "Score", "Orders", "GPS Fill %", "GPS Distinct %",
        "Return Stuck %", "Promised Pay %", "Collections Logged", "Qualified", "Payout (BDT)",
    ]
    disp = disp[[c for c in cols if c in disp.columns]]
    for c in ["GPS Fill %", "GPS Distinct %", "Return Stuck %", "Promised Pay %"]:
        disp[c] = disp[c].apply(lambda v: f"{v:,.0f}%" if pd.notna(v) else "—")
    disp["Score"] = disp["Score"].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
    disp["Qualified"] = disp["Qualified"].apply(lambda v: "✅" if v else "")
    disp["Payout (BDT)"] = disp["Payout (BDT)"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) and v > 0 else "—")
    st.dataframe(disp, width="stretch", hide_index=True)


# ── Top-level section entry point ───────────────────────────────────────────

def render(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    """`read_only=True` (Target Management's "💰 Commission Results") shows
    only the score table — no create/edit/delete controls. `read_only=False`
    (admin Commissions page) shows only setup, no scoring computation."""
    st.subheader("📱 A.4 — App Usage Commission")
    st.caption(
        "Scores every salesman who placed at least one mobile-app order this reporting month on a "
        "0-100 composite (4% order volume, 24% honest GPS location, 24% return hygiene, 24% "
        "promised-payment-date entry, 24% collection logging) — every salesman who scores ABOVE the "
        "admin-set threshold gets the same flat BDT bonus. Not a ranking — clear your own bar, get "
        "paid, same as everyone else who does."
    )

    is_admin = (not read_only) and st.session_state.get("user_role") == "admin"
    if is_admin:
        _render_create()

    campaigns = _list_campaigns()
    if campaigns.empty:
        st.info("No App Usage campaigns yet." + (" Create one above." if is_admin else " Ask an admin to create one."))
        return

    labels = {
        f"#{r.id} — {r.campaign_name} ({pd.Timestamp(r.window_end).strftime('%b %Y')})": r.id
        for r in campaigns.itertuples()
    }
    pick = st.selectbox("Select a campaign", list(labels.keys()), key=f"a4_select_campaign{key_suffix}")
    campaign = cc.get_campaign(labels[pick])
    if campaign:
        _render_detail(campaign, read_only=read_only, key_suffix=key_suffix)
