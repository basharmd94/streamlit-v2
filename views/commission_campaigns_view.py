# views/commission_campaigns_view.py
# UI for B.1/3/4 — Product-Specific / Stock Clearance / Slow-Moving campaigns.
# See commission_tracking_design.md §B.1/3/4 and processing/commission_campaigns.py
# for the full spec/engine — this file is UI only, no business logic.

from __future__ import annotations

import pandas as pd
import streamlit as st

from processing import commission_campaigns as cc
from views.marketing import _load_final_items

_CAMPAIGN_TYPES = ["Product-Specific", "Stock Clearance", "Slow-Moving"]


# ── Cached loaders — the FIFO resolution over full history is the expensive
# part (~15-20s), so it's cached hourly and shared across every campaign
# rather than recomputed per-campaign-view. ─────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_pooled_sales() -> pd.DataFrame:
    from core.analytics import Analytics
    from processing import common
    s1 = Analytics("sales", zid="100001", filters={}).data
    s0 = Analytics("sales", zid="100000", filters={}).data
    pooled = pd.concat([s1, s0], ignore_index=True) if s1 is not None and s0 is not None else pd.DataFrame()
    if pooled.empty:
        return pooled
    (pooled,) = common.data_copy_add_columns(pooled)
    return pooled


@st.cache_data(show_spinner=False, ttl=3600)
def _load_pooled_collections() -> pd.DataFrame:
    from core.analytics import Analytics
    c1 = Analytics("collection", zid="100001", filters={}).data
    c0 = Analytics("collection", zid="100000", filters={}).data
    return pd.concat([c1, c0], ignore_index=True) if c1 is not None and c0 is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_resolved_do_ledger() -> pd.DataFrame:
    sales = _load_pooled_sales()
    coll = _load_pooled_collections()
    do_totals = cc.build_do_totals(sales)
    coll_totals = cc.build_collection_totals(coll)
    return cc.resolve_do_paid_dates(do_totals, coll_totals)


@st.cache_data(show_spinner=False, ttl=3600)
def _load_combined_item_catalog() -> pd.DataFrame:
    """100001 + 100000 catalogs combined — a campaign's products can come
    from either business (scope is always both ZIDs together)."""
    a = _load_final_items("100001")
    b = _load_final_items("100000")
    frames = [d for d in (a, b) if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("item_id")


@st.cache_data(show_spinner=False, ttl=3600)
def _load_area_group_map() -> dict:
    return cc.build_area_group_map()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_spid_group_map() -> dict:
    sales = _load_pooled_sales()
    valid_spids = set(sales["spid"].dropna().unique()) if not sales.empty and "spid" in sales.columns else None
    return cc.derive_spid_group_map(_load_area_group_map(), valid_spids=valid_spids)


# ── Payout groups — derived live, read-only (admin-only view) ──────────────

def _render_derived_groups() -> None:
    """Payout-group membership is derived live from each salesman's own
    current area on PRMST (xdisease) matched against Cacus's area/zone
    classification (xstate) — confirmed 2026-09-20, replacing an earlier
    hand-maintained JSON roster. Nothing to edit here: to fix a salesman's
    group, correct their area on the ERP side (prmst), not in this app."""
    area_group_map = _load_area_group_map()
    spid_group_map = _load_spid_group_map()
    sales = _load_pooled_sales()
    name_lookup = {}
    if not sales.empty and {"spid", "spname"}.issubset(sales.columns):
        name_lookup = dict(
            sales[["spid", "spname"]].dropna().drop_duplicates("spid").values.tolist()
        )

    with st.expander(f"🗺️ Payout Groups — derived from PRMST area ({len(spid_group_map)} salesman(s) resolved)"):
        st.caption(
            "No roster to maintain — group membership comes live from each salesman's "
            "current working area on PRMST (`xdisease`), matched against Cacus's own "
            "area classification (`xstate`). To change a salesman's group, correct "
            "their area on the ERP side, not here."
        )
        if not area_group_map:
            st.warning("No Cacus customer/area data available.")
            return
        st.caption(
            f"{len(set(area_group_map.values()))} group(s) found in Cacus: "
            + ", ".join(sorted(set(area_group_map.values())))
        )
        if not spid_group_map:
            st.info(
                "No salesman currently resolves to a group — PRMST's `xdisease` "
                "(working area) field is still being populated on the live server "
                "as of 2026-09-20 (see CLAUDE.md). This will populate automatically "
                "once that rollout finishes — nothing to do here."
            )
            return
        rows = [
            {"Salesman Code": spid, "Salesman": name_lookup.get(spid, "(name unavailable)"), "Group": group}
            for spid, group in sorted(spid_group_map.items(), key=lambda kv: (kv[1], kv[0]))
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ── Create / Edit campaign form (admin-only) — shared body ─────────────────
# `existing=None` -> Create New Campaign (blank, calls create_campaign).
# `existing=<campaign dict>` -> Edit Campaign (pre-filled, calls
# update_campaign) — added 2026-09-20 so a setup mistake (e.g. wrong window
# dates) can be fixed directly rather than delete+recreate; campaigns have
# no pre-computed data to invalidate, so editing is always safe.

def _clamp_min_session_date(key: str, min_v) -> bool:
    """Same pattern as views/commissions.py::_clamp_session_date (min-only
    variant) — a widget's persisted value (window end) can fall below a
    newly-raised min_value (window start, if the user pushes it forward on
    a rerun) and Streamlit raises if that's not pre-clamped first. Returns
    True if a clamp happened, so the caller skips passing `value=` this run."""
    if key in st.session_state:
        cur = st.session_state[key]
        if cur < min_v:
            st.session_state[key] = min_v
            return True
    return False


def _render_campaign_form(available_groups: list, items_df: pd.DataFrame, existing: dict | None = None) -> None:
    is_edit = existing is not None
    key_prefix = f"cc_edit_{existing['id']}" if is_edit else "cc_new"

    name = st.text_input(
        "Campaign name", value=(existing["campaign_name"] if is_edit else ""), key=f"{key_prefix}_name",
    )
    c0a, c0b = st.columns(2)
    with c0a:
        type_default = existing.get("campaign_type") if is_edit else None
        type_idx = _CAMPAIGN_TYPES.index(type_default) if type_default in _CAMPAIGN_TYPES else 0
        ctype = st.selectbox("Campaign type", _CAMPAIGN_TYPES, index=type_idx, key=f"{key_prefix}_type")
    with c0b:
        recip_options = ["salesman", "customer"]
        recip_default = existing.get("recipient_type", "salesman") if is_edit else "salesman"
        recip_idx = recip_options.index(recip_default) if recip_default in recip_options else 0
        recipient_type = st.radio(
            "Commission paid to", recip_options, index=recip_idx, key=f"{key_prefix}_recipient",
            horizontal=True, format_func=lambda v: v.capitalize(),
        )

    opts_df = (
        items_df[["item_id", "item_name"]].dropna().drop_duplicates()
        .assign(item_id=lambda d: d["item_id"].astype(str)).sort_values("item_name")
    )
    label_map = {f"{r.item_id} - {r.item_name}": r.item_id for r in opts_df.itertuples()}
    code_to_label = {v: k for k, v in label_map.items()}

    default_labels = (
        [code_to_label[c] for c in existing["product_rates"].keys() if c in code_to_label] if is_edit else []
    )
    picked_labels = st.multiselect(
        "Product(s)", list(label_map.keys()), default=default_labels, key=f"{key_prefix}_products",
    )
    product_codes = [label_map[l] for l in picked_labels]

    product_rates = {}
    if product_codes:
        st.caption(
            "Rate = the per-unit incentive/discount BDT amount for that product "
            "(e.g. 5 or 3) — not the product's sales price. Varies per product; "
            "set the Cap for the whole campaign below."
        )
        for code in product_codes:
            label = next((l for l, c in label_map.items() if c == code), code)
            rc1, rc2 = st.columns([3, 2])
            with rc1:
                st.markdown(f"`{code}`")
                st.caption(label.split(" - ", 1)[-1])
            with rc2:
                rate_default = float(existing["product_rates"].get(code, 0.0)) if is_edit else 0.0
                rate = st.number_input(
                    "Rate (BDT/unit)", min_value=0.0, step=0.5, value=rate_default,
                    key=f"{key_prefix}_rate_{code}",
                )
            product_rates[code] = rate

    existing_cap = existing.get("cap") if is_edit else None
    cap_default = float(existing_cap) if is_edit and existing_cap not in (None, "") and not pd.isna(existing_cap) else 0.0
    cap_val = st.number_input(
        "Cap (BDT, 0 = no cap)", min_value=0.0, step=100.0, value=cap_default, key=f"{key_prefix}_cap",
        help=f"ONE ceiling for the whole campaign — caps a single {recipient_type}'s TOTAL "
             "payout, summed across every product picked above (not a separate cap per product).",
    )
    campaign_cap = cap_val if cap_val > 0 else None

    today = pd.Timestamp.today().normalize().date()
    wstart_default = pd.Timestamp(existing["window_start"]).date() if is_edit else today
    wend_default = pd.Timestamp(existing["window_end"]).date() if is_edit else today
    c3, c4 = st.columns(2)
    with c3:
        window_start = st.date_input("Sales window start", value=wstart_default, key=f"{key_prefix}_wstart")
    just_clamped = _clamp_min_session_date(f"{key_prefix}_wend", window_start)
    with c4:
        wend_kwargs = dict(min_value=window_start, key=f"{key_prefix}_wend")
        if not just_clamped:
            wend_kwargs["value"] = max(wend_default, window_start)
        window_end = st.date_input("Sales window end", **wend_kwargs)

    st.caption("Collection deadline per payout group, for this campaign:")
    payout_groups = {}
    existing_deadlines = existing.get("payout_groups", {}) if is_edit else {}
    # Union with the campaign's own already-set groups — a group can stop
    # showing up here if Cacus's live data shifts; don't silently drop a
    # deadline the campaign already has just because of that.
    all_groups = list(dict.fromkeys(list(available_groups) + list(existing_deadlines.keys())))
    for group in all_groups:
        try:
            deadline_default = pd.Timestamp(existing_deadlines[group]).date() if group in existing_deadlines else today
        except Exception:
            deadline_default = today
        deadline = st.date_input(
            f"{group} deadline", value=deadline_default, key=f"{key_prefix}_deadline_{group}",
        )
        payout_groups[group] = str(deadline)

    baseline_options = [3, 6]
    baseline_default = existing.get("uptick_baseline_months", 3) if is_edit else 3
    baseline_idx = baseline_options.index(baseline_default) if baseline_default in baseline_options else 0
    baseline_months = st.selectbox(
        "Uptick baseline window (months)", baseline_options, index=baseline_idx, key=f"{key_prefix}_baseline",
    )
    notes = st.text_area(
        "Notes (optional)", value=(existing.get("notes") or "" if is_edit else ""), key=f"{key_prefix}_notes",
    )

    all_rates_set = bool(product_rates) and all(v > 0 for v in product_rates.values())
    can_submit = bool(name.strip()) and all_rates_set
    if not product_codes:
        st.caption("Pick at least one product above.")
    elif not all_rates_set:
        st.caption("Every picked product needs a rate > 0.")

    btn_label = "💾 Save Changes" if is_edit else "✅ Create Campaign"
    if st.button(btn_label, key=f"{key_prefix}_submit_btn", disabled=not can_submit):
        if is_edit:
            ok = cc.update_campaign(
                campaign_id=int(existing["id"]), campaign_name=name.strip(), campaign_type=ctype,
                recipient_type=recipient_type, product_rates=product_rates, cap=campaign_cap,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups=payout_groups, uptick_baseline_months=baseline_months, notes=notes,
            )
            if ok:
                st.success("Campaign updated.", icon="✅")
                st.rerun()
            else:
                st.error("Could not update campaign.")
        else:
            cid = cc.create_campaign(
                campaign_name=name.strip(), campaign_type=ctype, recipient_type=recipient_type,
                product_rates=product_rates, cap=campaign_cap,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups=payout_groups, uptick_baseline_months=baseline_months,
                created_by=st.session_state.get("username", ""), notes=notes,
            )
            if cid:
                st.success(f"Campaign #{cid} created.", icon="✅")
                st.rerun()
            else:
                st.error("Could not create campaign.")


def _render_create_campaign() -> None:
    available_groups = sorted(set(_load_area_group_map().values()))
    items_df = _load_combined_item_catalog()

    with st.expander("➕ Create New Campaign", expanded=False):
        if not available_groups:
            st.warning("No payout groups available — Cacus customer/area data couldn't be loaded.")
            return
        if items_df.empty or "item_id" not in items_df.columns:
            st.warning("No product catalog available.")
            return
        _render_campaign_form(available_groups, items_df)


def _render_edit_campaign(campaign: dict) -> None:
    available_groups = sorted(set(_load_area_group_map().values()))
    items_df = _load_combined_item_catalog()

    with st.expander("✏️ Edit Campaign", expanded=False):
        if items_df.empty or "item_id" not in items_df.columns:
            st.warning("No product catalog available.")
            return
        _render_campaign_form(available_groups, items_df, existing=campaign)


# ── Campaign detail / payout view ───────────────────────────────────────────

def _fmt_bdt(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"৳{v:,.0f}"


def _render_campaign_detail(campaign: dict, read_only: bool = False, key_suffix: str = "") -> None:
    recipient_type = campaign.get("recipient_type") or "salesman"
    product_rates = campaign["product_rates"]
    cap = campaign.get("cap")
    cap = float(cap) if cap not in (None, "") and not pd.isna(cap) else None

    today = pd.Timestamp.today().normalize().date()
    window_end = pd.Timestamp(campaign["window_end"]).date()
    is_ongoing = window_end >= today
    status_badge = "🟢 Ongoing" if is_ongoing else "🔴 Closed"

    st.markdown(f"### {campaign['campaign_name']} — {status_badge}")
    st.caption(
        f"{campaign.get('campaign_type') or '—'} · Paid to: **{recipient_type.capitalize()}** · "
        f"Window: {campaign['window_start']} → {campaign['window_end']}"
    )
    rates_line = " · ".join(f"`{code}`: {_fmt_bdt(rate)}/unit" for code, rate in product_rates.items())
    st.caption(f"Rates: {rates_line}")
    st.caption(
        f"Cap: **{_fmt_bdt(cap)}** per {recipient_type}, across every product in this campaign combined"
        if cap is not None else
        f"Cap: **none** — no ceiling on a {recipient_type}'s total payout"
    )

    with st.spinner("Computing payout (pooled 100001+100000, full-history FIFO)…"):
        sales = _load_pooled_sales()
        resolved = _load_resolved_do_ledger()
        spid_group_map = _load_spid_group_map()
        result = cc.compute_campaign_payout(campaign, sales, resolved, spid_group_map)

    gate = result["gate"]
    if result["gate_passed"]:
        st.success(
            f"✅ Gate passed — 100001 ({_fmt_bdt(gate['100001']['actual'])} vs target "
            f"{_fmt_bdt(gate['100001']['target'])}) and 100000 ({_fmt_bdt(gate['100000']['actual'])} vs "
            f"target {_fmt_bdt(gate['100000']['target'])}) both hit their target.",
            icon="✅",
        )
    else:
        failed = [z for z in ("100001", "100000") if not gate[z]["hit"]]
        st.error(
            f"🚫 Gate FAILED — {', '.join(failed)} missed its target. No commission payable, even though "
            f"{_fmt_bdt(result['total_payout_if_gate_passed'])} was computed below (shown for reference).",
            icon="🚫",
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Payout", _fmt_bdt(result["total_payout"]))
    m2.metric("Eligible DOs", f"{result['eligible_do_count']:,}")
    m3.metric("Excluded DOs", f"{result['excluded_do_count']:,}")

    recipient_label = result["recipient_type"].capitalize()
    st.markdown(f"**Per-{recipient_label} Payout**")
    rt_df = result["recipient_total"]
    if rt_df.empty:
        st.info("No eligible payout yet.")
    else:
        rename_cols = {
            "recipient": f"{recipient_label} Code", "recipient_name": recipient_label,
            "raw_total_payout": "Raw Total (BDT)", "total_payout": "Payout (BDT)",
        }
        # Only show the "Raw Total" column (pre-cap) when a cap is actually
        # set — with no cap the two columns would always be identical, which
        # is just noise.
        cols = ["recipient", "recipient_name"] + (["raw_total_payout"] if cap is not None else []) + ["total_payout"]
        disp = rt_df[cols].rename(columns=rename_cols)
        fmt = {"Payout (BDT)": "{:,.0f}"}
        if cap is not None:
            fmt["Raw Total (BDT)"] = "{:,.0f}"
        st.dataframe(disp.style.format(fmt), width="stretch", hide_index=True)

    with st.expander(f"Per-{recipient_label}-Per-Product breakdown"):
        rp_df = result["recipient_product"]
        if rp_df.empty:
            st.info("No eligible lines yet.")
        else:
            st.caption(
                "Payout per line is uncapped (quantity × rate) — the campaign-wide cap "
                f"shown above applies once to each {recipient_type}'s TOTAL across every "
                "product, not to any one line here."
            )
            disp = rp_df.rename(columns={
                "recipient": f"{recipient_label} Code", "recipient_name": recipient_label,
                "itemcode": "Item Code", "quantity": "Qty", "rate": "Rate", "payout": "Payout",
            })
            st.dataframe(
                disp.style.format({
                    "Qty": "{:,.0f}", "Rate": "{:,.2f}", "Payout": "{:,.0f}",
                }),
                width="stretch", hide_index=True,
            )

    with st.expander(f"Excluded DOs ({result['excluded_do_count']:,}) — why they didn't qualify"):
        exc = result["excluded"]
        if exc.empty:
            st.info("None excluded.")
        else:
            st.dataframe(exc["reason"].value_counts().rename("Count"), width="stretch")
            st.dataframe(exc, width="stretch", hide_index=True, height=300)

    uptick = result["uptick"]
    st.markdown("**Sales Uptick (companion metric, not a payout gate)**")
    u1, u2 = st.columns(2)
    with u1:
        pct = uptick["qty_uptick_pct"]
        u1.metric(
            "Qty vs. baseline avg", f"{uptick['campaign_qty']:,.0f}",
            f"{pct:+.1f}% vs {uptick['baseline_avg_qty']:,.0f}" if pct is not None else "no baseline data",
        )
    with u2:
        pct = uptick["revenue_uptick_pct"]
        u2.metric(
            "Revenue vs. baseline avg", _fmt_bdt(uptick["campaign_revenue"]),
            f"{pct:+.1f}% vs {_fmt_bdt(uptick['baseline_avg_revenue'])}" if pct is not None else "no baseline data",
        )

    if not read_only and st.session_state.get("user_role") == "admin":
        st.divider()
        _render_edit_campaign(campaign)
        if st.button("🗑 Delete This Campaign", key=f"cc_delete_{campaign['id']}{key_suffix}"):
            cc.delete_campaign(int(campaign["id"]))
            st.rerun()


# ── Top-level section entry point ───────────────────────────────────────────

def render(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    """`read_only=True` (used by Target Management's manager-facing Commission
    Results view, see views/commissions.py::render_section_picker) hides
    every setup control (roster, create, delete) regardless of the viewer's
    own role — that view is results-only by design, never a place to set up
    a campaign, even for an admin who happens to open it from there.
    `key_suffix` keeps widget keys unique when this is mounted at more than
    one page (the admin Commissions page and Target Management both call
    this, same pattern as e.g. views/glpmt_shared.py's key_suffix)."""
    st.subheader("🎯 Product / Stock Clearance / Slow-Moving Campaign")
    st.caption(
        "Rate per unit sold, paid only if the customer's DO is fully collected by a "
        "deadline (all-or-nothing, FIFO-matched, pooled across 100001+100000). "
        "Gated on both ZIDs hitting their own sales target. Nothing is pre-computed — "
        "every number below is calculated live from current sales/collection data."
    )

    is_admin = (not read_only) and st.session_state.get("user_role") == "admin"
    if is_admin:
        _render_derived_groups()
        _render_create_campaign()

    campaigns = cc.list_campaigns()
    if campaigns.empty:
        st.info("No campaigns yet." + (" Create one above." if is_admin else " Ask an admin to create one."))
        return

    labels = {
        f"#{r.id} — {r.campaign_name} ({r.window_start} → {r.window_end})": r.id
        for r in campaigns.itertuples()
    }
    pick = st.selectbox("Select a campaign", list(labels.keys()), key=f"cc_select_campaign{key_suffix}")
    campaign = cc.get_campaign(labels[pick])
    if campaign:
        _render_campaign_detail(campaign, read_only=read_only, key_suffix=key_suffix)
