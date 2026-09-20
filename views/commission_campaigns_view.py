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


# ── Create campaign form (admin-only) ───────────────────────────────────────

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

        name = st.text_input("Campaign name", key="cc_new_name")
        c0a, c0b = st.columns(2)
        with c0a:
            ctype = st.selectbox("Campaign type", _CAMPAIGN_TYPES, key="cc_new_type")
        with c0b:
            recipient_type = st.radio(
                "Commission paid to", ["salesman", "customer"], key="cc_new_recipient",
                horizontal=True, format_func=lambda v: v.capitalize(),
            )

        opts_df = (
            items_df[["item_id", "item_name"]].dropna().drop_duplicates()
            .assign(item_id=lambda d: d["item_id"].astype(str)).sort_values("item_name")
        )
        label_map = {f"{r.item_id} - {r.item_name}": r.item_id for r in opts_df.itertuples()}
        picked_labels = st.multiselect("Product(s)", list(label_map.keys()), key="cc_new_products")
        product_codes = [label_map[l] for l in picked_labels]

        product_rates = {}
        if product_codes:
            st.caption(
                "Rate = the per-unit incentive/discount BDT amount for that product "
                "(e.g. 5 or 3) — not the product's sales price. Cap is optional, per "
                f"product per {recipient_type}."
            )
            for code in product_codes:
                label = next((l for l, c in label_map.items() if c == code), code)
                rc1, rc2, rc3 = st.columns([3, 2, 2])
                with rc1:
                    st.markdown(f"`{code}`")
                    st.caption(label.split(" - ", 1)[-1])
                with rc2:
                    rate = st.number_input(
                        "Rate (BDT/unit)", min_value=0.0, step=0.5, key=f"cc_new_rate_{code}",
                    )
                with rc3:
                    cap_val = st.number_input(
                        "Cap (BDT, 0 = no cap)", min_value=0.0, step=100.0, key=f"cc_new_cap_{code}",
                    )
                product_rates[code] = {"rate": rate, "cap": (cap_val if cap_val > 0 else None)}

        today = pd.Timestamp.today().normalize().date()
        c3, c4 = st.columns(2)
        with c3:
            window_start = st.date_input("Sales window start", value=today, key="cc_new_wstart")
        with c4:
            window_end = st.date_input("Sales window end", value=today, min_value=window_start, key="cc_new_wend")

        st.caption("Collection deadline per payout group, for this campaign:")
        payout_groups = {}
        for group in available_groups:
            deadline = st.date_input(
                f"{group} deadline", value=today, key=f"cc_new_deadline_{group}",
            )
            payout_groups[group] = str(deadline)

        baseline_months = st.selectbox("Uptick baseline window (months)", [3, 6], key="cc_new_baseline")
        notes = st.text_area("Notes (optional)", key="cc_new_notes")

        all_rates_set = bool(product_rates) and all(v["rate"] > 0 for v in product_rates.values())
        can_create = bool(name.strip()) and all_rates_set
        if not product_codes:
            st.caption("Pick at least one product above.")
        elif not all_rates_set:
            st.caption("Every picked product needs a rate > 0.")
        if st.button("✅ Create Campaign", key="cc_new_create_btn", disabled=not can_create):
            cid = cc.create_campaign(
                campaign_name=name.strip(), campaign_type=ctype, recipient_type=recipient_type,
                product_rates=product_rates,
                window_start=str(window_start), window_end=str(window_end),
                payout_groups=payout_groups, uptick_baseline_months=baseline_months,
                created_by=st.session_state.get("username", ""), notes=notes,
            )
            if cid:
                st.success(f"Campaign #{cid} created.", icon="✅")
                st.rerun()
            else:
                st.error("Could not create campaign.")


# ── Campaign detail / payout view ───────────────────────────────────────────

def _fmt_bdt(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"৳{v:,.0f}"


def _render_campaign_detail(campaign: dict) -> None:
    recipient_type = campaign.get("recipient_type") or "salesman"
    product_rates = campaign["product_rates"]
    st.markdown(f"### {campaign['campaign_name']}")
    st.caption(
        f"{campaign.get('campaign_type') or '—'} · Paid to: **{recipient_type.capitalize()}** · "
        f"Window: {campaign['window_start']} → {campaign['window_end']}"
    )
    rates_line = " · ".join(
        f"`{code}`: {_fmt_bdt(v['rate'])}/unit" + (f" (cap {_fmt_bdt(v['cap'])})" if v.get("cap") else "")
        for code, v in product_rates.items()
    )
    st.caption(f"Rates: {rates_line}")

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
        disp = rt_df.rename(columns={
            "recipient": f"{recipient_label} Code", "recipient_name": recipient_label,
            "total_payout": "Payout (BDT)",
        })
        st.dataframe(
            disp.style.format({"Payout (BDT)": "{:,.0f}"}), width="stretch", hide_index=True,
        )

    with st.expander(f"Per-{recipient_label}-Per-Product breakdown"):
        rp_df = result["recipient_product"]
        if rp_df.empty:
            st.info("No eligible lines yet.")
        else:
            disp = rp_df.rename(columns={
                "recipient": f"{recipient_label} Code", "recipient_name": recipient_label,
                "itemcode": "Item Code", "quantity": "Qty", "rate": "Rate", "cap": "Cap",
                "raw_payout": "Raw Payout", "capped_payout": "Capped Payout",
            })
            # Pre-format Cap as a string ("—" for no-cap) rather than relying on
            # Styler na_rep — st.dataframe doesn't reliably respect it for a NaN
            # in an otherwise-numeric column (same class of issue as CLAUDE.md's
            # documented TOTAL-row/Styler pitfall).
            disp["Cap"] = disp["Cap"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
            st.dataframe(
                disp.style.format({
                    "Qty": "{:,.0f}", "Rate": "{:,.2f}",
                    "Raw Payout": "{:,.0f}", "Capped Payout": "{:,.0f}",
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

    if st.session_state.get("user_role") == "admin":
        st.divider()
        if st.button("🗑 Delete This Campaign", key=f"cc_delete_{campaign['id']}"):
            cc.delete_campaign(int(campaign["id"]))
            st.rerun()


# ── Top-level section entry point ───────────────────────────────────────────

def render(zid: str) -> None:
    st.subheader("🎯 Product / Stock Clearance / Slow-Moving Campaign")
    st.caption(
        "Rate per unit sold, paid only if the customer's DO is fully collected by a "
        "deadline (all-or-nothing, FIFO-matched, pooled across 100001+100000). "
        "Gated on both ZIDs hitting their own sales target. Nothing is pre-computed — "
        "every number below is calculated live from current sales/collection data."
    )

    is_admin = st.session_state.get("user_role") == "admin"
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
    pick = st.selectbox("Select a campaign", list(labels.keys()), key="cc_select_campaign")
    campaign = cc.get_campaign(labels[pick])
    if campaign:
        _render_campaign_detail(campaign)
