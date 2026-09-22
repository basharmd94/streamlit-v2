# views/commission_individual_target_view.py
# UI for B.2 — Individual Target Achievement. Built 2026-09-21 from a
# simplified spec the user gave directly, replacing the design doc's
# original fixed-named-list guess: "This will be set in advance and will
# follow the target set within target management for that month. If they
# achieve 100% of the target they will be rewarded one fixed amount for
# all." See processing/commission_campaigns.py::compute_individual_target_bonus
# for the engine — this file is UI only.
#
# Reporting month, current-or-future only, pinned at setup — same rule and
# reasoning as A.1/A.4 (confirmed there, reused here rather than
# re-litigated): a campaign is set up in advance for the current or a
# future month, never backdated.
#
# Pooled 100001+100000 — same scope as A.1/A.4, and the only scope that
# makes sense here: data/targets.json's own per-salesman targets are what
# this section evaluates, and A.1 already established the "consolidated
# target" convention of summing a spid's target across both ZIDs.
#
# Gated on the 100001/100000 company target, confirmed 2026-09-22 — same
# check_gate A.1/B.1/3/4 already use: if either ZID misses its own sales
# target for the reporting month, total_payout zeroes out (the computed
# per-salesman figures still show, for visibility).
#
# Population = every salesman who HAS a target entry for the reporting
# month (views/_tm_shared.py::_targets_for_month), not every salesman with
# sales activity — a salesman with no target assigned can't be evaluated
# for target achievement at all, so they simply don't appear.

from __future__ import annotations

import pandas as pd
import streamlit as st

from processing import commission_campaigns as cc
from views._tm_shared import _targets_for_month

CAMPAIGN_TYPE = "Individual Target Achievement"
_ZIDS = ["100001", "100000"]


def _fmt_bdt(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"৳{v:,.0f}"


def _anchor_month_choices(today: pd.Timestamp, months_forward: int = 12) -> list:
    """[(label, year, month), ...] earliest first: the current month plus
    up to `months_forward` FUTURE ones — current month or later only, same
    rule as A.1/A.4's reporting month (no retroactive setup)."""
    cur = pd.Timestamp(today.year, today.month, 1)
    return [
        ((cur + pd.DateOffset(months=i)).strftime("%b %Y"),
         int((cur + pd.DateOffset(months=i)).year),
         int((cur + pd.DateOffset(months=i)).month))
        for i in range(months_forward + 1)
    ]


# ── Pooled data loaders ─────────────────────────────────────────────────────

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
def _load_returns(zid: str) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("return", zid=zid, filters={}).data
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


def _target_by_sp_pooled(year: int, month: int) -> dict:
    """{spid: target}, summed across 100001+100000 for this (year, month) —
    same "consolidated target" convention A.1's own pooled scoring uses.
    A spid with a target on only one ZID still gets that one value."""
    result: dict = {}
    for z in _ZIDS:
        for spid, target in _targets_for_month(z, year, month).items():
            result[spid] = result.get(spid, 0.0) + float(target or 0)
    return result


def _compute_results(anchor_year: int, anchor_month: int) -> dict:
    today = pd.Timestamp.today().normalize()
    month_start = pd.Timestamp(anchor_year, anchor_month, 1)
    month_end = month_start + pd.offsets.MonthEnd(0)

    sales_df = _pooled(_load_sales)
    returns_df = _pooled(_load_returns)
    target_by_sp = _target_by_sp_pooled(anchor_year, anchor_month)

    return sales_df, returns_df, target_by_sp, month_start, month_end, today


# ── Create / Edit (admin-only) — shared body ────────────────────────────────

def _render_form(existing: dict | None = None) -> None:
    is_edit = existing is not None
    key_prefix = f"b2_edit_{existing['id']}" if is_edit else "b2_new"

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
        "final numbers. Pooled across 100001+100000 (targets summed per salesman across both)."
    )

    bonus_default = float(existing_config.get("bonus_amount", 0.0)) if is_edit else 0.0
    bonus_amount = st.number_input(
        "Bonus amount (BDT) — same flat amount for every salesman who reaches 100% of their own target",
        min_value=0.0, step=100.0, value=bonus_default, key=f"{key_prefix}_bonus",
    )

    notes = st.text_area(
        "Notes (optional)", value=(existing.get("notes") or "" if is_edit else ""), key=f"{key_prefix}_notes",
    )

    can_submit = bool(name.strip()) and bonus_amount > 0
    if bonus_amount <= 0:
        st.caption("Bonus amount must be > 0.")

    btn_label = "💾 Save Changes" if is_edit else "✅ Create Individual Target Campaign"
    if st.button(btn_label, key=f"{key_prefix}_submit_btn", disabled=not can_submit):
        product_rates_payload = {"bonus_amount": float(bonus_amount)}
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
                st.success("Individual Target campaign updated.", icon="✅")
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
                st.success(f"Individual Target campaign #{cid} created.", icon="✅")
                st.rerun()
            else:
                st.error("Could not create campaign.")


def _render_create() -> None:
    with st.expander("➕ Create New Individual Target Campaign", expanded=False):
        _render_form()


def _render_edit(campaign: dict) -> None:
    with st.expander("✏️ Edit Individual Target Campaign", expanded=False):
        _render_form(existing=campaign)


def _list_campaigns() -> pd.DataFrame:
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
        f"Individual Target Achievement · Reporting month {window_end.strftime('%B %Y')} ({status_note})"
    )
    st.caption(
        f"Bonus: {_fmt_bdt(bonus_amount)} per salesman who reaches 100% of their own target "
        f"— gated on the 100001/100000 company target, same as B.1/3/4."
    )

    if not read_only:
        if st.session_state.get("user_role") == "admin":
            st.divider()
            _render_edit(campaign)
            if st.button("🗑 Delete This Campaign", key=f"b2_delete_{campaign['id']}{key_suffix}"):
                cc.delete_campaign(int(campaign["id"]))
                st.rerun()
        return

    # ── Results (Target Management "💰 Commission Results") ────────────────
    with st.spinner(f"Checking target achievement for {window_end.strftime('%B %Y')} (pooled 100001+100000)…"):
        sales_df, returns_df, target_by_sp, month_start, month_end, today_ts = _compute_results(
            anchor_year, anchor_month
        )
        names = _spname_map()
        result = cc.compute_individual_target_bonus(
            campaign, sales_df, returns_df, target_by_sp, names, month_start, month_end, today_ts,
        )

    _render_gate_banner(result["gate"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Payout", _fmt_bdt(result["total_payout"]))
    m2.metric("Qualified", f"{result['qualified_count']}")
    m3.metric("Salesmen With a Target", f"{len(result['scores'])}")

    scores = result["scores"]
    if scores.empty:
        st.info("No salesman has a target set for this reporting month yet.")
        return

    disp = scores.copy()
    disp = disp.rename(columns={
        "spid": "Salesman Code", "spname": "Salesman", "target": "Target",
        "net_sales": "Net Sales", "achievement_pct": "Achievement %",
        "qualified": "Qualified", "payout": "Payout (BDT)",
    })
    for c in ["Target", "Net Sales"]:
        disp[c] = disp[c].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
    disp["Achievement %"] = disp["Achievement %"].apply(lambda v: f"{v:,.1f}%" if pd.notna(v) else "—")
    disp["Qualified"] = disp["Qualified"].apply(lambda v: "✅" if v else "")
    disp["Payout (BDT)"] = disp["Payout (BDT)"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) and v > 0 else "—")
    st.dataframe(disp, width="stretch", hide_index=True)


# ── Top-level section entry point ───────────────────────────────────────────

def render(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    """`read_only=True` (Target Management's "💰 Commission Results") shows
    only the results table — no create/edit/delete controls. `read_only=False`
    (admin Commissions page) shows only setup, no computation ever runs there."""
    st.subheader("🎯 B.2 — Individual Target Achievement")
    st.caption(
        "Every salesman with a target set in Target Management for the reporting month gets the "
        "same flat BDT bonus if their own net sales reach 100% of that target — a threshold check "
        "per person, not a ranking, not proportional."
    )

    is_admin = (not read_only) and st.session_state.get("user_role") == "admin"
    if is_admin:
        _render_create()

    campaigns = _list_campaigns()
    if campaigns.empty:
        st.info(
            "No Individual Target campaigns yet."
            + (" Create one above." if is_admin else " Ask an admin to create one.")
        )
        return

    labels = {
        f"#{r.id} — {r.campaign_name} ({pd.Timestamp(r.window_end).strftime('%b %Y')})": r.id
        for r in campaigns.itertuples()
    }
    pick = st.selectbox("Select a campaign", list(labels.keys()), key=f"b2_select_campaign{key_suffix}")
    campaign = cc.get_campaign(labels[pick])
    if campaign:
        _render_detail(campaign, read_only=read_only, key_suffix=key_suffix)
