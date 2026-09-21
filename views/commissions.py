# views/commissions.py
# "Commissions" top-level page — Sales Reward, Commission & Incentive
# Tracking. Design: commission_tracking_design.md (repo root), status
# "agreed design." Product Tracking and the B.1/3/4 campaign mechanism are
# built so far; every other entry in the dropdown below is a placeholder
# until it's picked up and scoped with the user, per the doc's own
# "discuss before building" rule.

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.analytics import Analytics
from processing import commissions as comm, usage_log
from views import (
    commission_app_usage_view,
    commission_best_performer_view,
    commission_campaigns_view,
    commission_customer_acquisition_view,
    commission_customer_best_performer_view,
    commission_rankings_view,
    commission_shared,
)
from views.marketing import _load_final_items  # noqa: F401 — re-exported; avoids duplicate cache


# ── Cached loaders ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_sales_daily(zid: str) -> pd.DataFrame:
    df = Analytics("sales_daily_item", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_returns_full(zid: str) -> pd.DataFrame:
    """Full return line-item history (itemcode/date/returnqty/treturnamt) —
    switched from the lightweight returns_daily_item MV (qty only) since Net
    Revenue (added 2026-09-20) needs the return's own BDT value (treturnamt),
    which that MV doesn't carry."""
    df = Analytics("return", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


# ── Product Tracking ────────────────────────────────────────────────────────
# Each watched product carries its OWN cutoff + "back to" date (not one
# shared window for the whole list) — see commission_tracking_design.md
# §Product Tracking, "redesign requested 2026-09-19."

def _clamp_session_date(key: str, min_v, max_v) -> bool:
    """A widget's persisted session_state value can fall outside new
    min/max bounds when another widget it depends on (the cutoff date)
    changes — Streamlit raises if that happens, so pre-clamp before the
    date_input call that would otherwise blow up. Returns True if a clamp
    happened, so the caller can skip passing `value=` this run — passing
    both a `value` and a just-written session_state entry in the same run
    triggers a (harmless but noisy) Streamlit warning."""
    if key in st.session_state:
        cur = st.session_state[key]
        if cur < min_v or cur > max_v:
            st.session_state[key] = min(max(cur, min_v), max_v)
            return True
    return False


def _render_add_product(zid: str, items_df: pd.DataFrame, watchlist: dict) -> None:
    if items_df is None or items_df.empty or "item_id" not in items_df.columns:
        st.warning("No product catalog available to pick from.")
        return

    # A widget's session_state key can't be assigned after that widget is
    # instantiated this run (Streamlit raises StreamlitAPIException) — reset
    # the picker back to "—" at the START of the run *following* a successful
    # add, before the selectbox below is created, not right after adding.
    if st.session_state.pop("_comm_wl_add_reset", False):
        st.session_state["comm_wl_add"] = "—"

    opts_df = (
        items_df[["item_id", "item_name"]]
        .dropna(subset=["item_id"])
        .drop_duplicates()
        .assign(item_id=lambda d: d["item_id"].astype(str))
        .sort_values("item_name")
    )
    label_map = {f"{r.item_id} - {r.item_name}": r.item_id for r in opts_df.itertuples()}
    watched_set = set(watchlist.keys())
    available_labels = [lbl for lbl, code in label_map.items() if code not in watched_set]
    at_cap = len(watchlist) >= comm.MAX_WATCHLIST_ITEMS

    pick = st.selectbox("Add a product", ["—"] + available_labels, key="comm_wl_add", disabled=at_cap)

    today = pd.Timestamp.today().normalize()
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        add_cutoff = st.date_input(
            "Cutoff date", value=today.date(), max_value=today.date(), key="comm_wl_add_cutoff",
        )
    cutoff_ts = pd.Timestamp(add_cutoff)
    min_back = comm.min_back_to(cutoff_ts).date()
    max_back = (cutoff_ts - pd.Timedelta(days=1)).date()
    just_clamped = _clamp_session_date("comm_wl_add_backto", min_back, max_back)
    with c2:
        back_kwargs = dict(min_value=min_back, max_value=max_back, key="comm_wl_add_backto")
        if not just_clamped:
            back_kwargs["value"] = min(max((today - pd.DateOffset(months=1)).date(), min_back), max_back)
        add_back_to = st.date_input("Back to (≤ 6mo before cutoff)", **back_kwargs)
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Add", key="comm_wl_add_btn", disabled=at_cap or pick == "—"):
            ok = comm.add_product(zid, label_map[pick], pd.Timestamp(add_cutoff), pd.Timestamp(add_back_to))
            if ok:
                st.session_state["_comm_wl_add_reset"] = True  # picked up at the top on the next run
                st.rerun()
            else:
                st.error("Could not add — check the dates and try again.")

    if at_cap:
        st.caption(f"⚠️ Watchlist is at the {comm.MAX_WATCHLIST_ITEMS}-item cap — remove one to add another.")


def _render_tracked_item_editor(zid: str, itemcode: str, itemname: str, dates: dict) -> None:
    today = pd.Timestamp.today().normalize()
    default_back = (today - pd.DateOffset(months=1)).date()
    try:
        saved_cutoff = pd.Timestamp(dates["cutoff"]).date()
        saved_back_to = pd.Timestamp(dates["back_to"]).date()
    except Exception:
        saved_cutoff, saved_back_to = today.date(), default_back

    with st.container(border=True):
        h1, h2 = st.columns([5, 1])
        with h1:
            st.markdown(f"**`{itemcode}`** — {itemname or '(name unavailable)'}")
        with h2:
            if st.button("🗑 Remove", key=f"comm_wl_rm_{itemcode}"):
                comm.remove_product(zid, itemcode)
                st.rerun()

        cutoff_key, back_key = f"comm_wl_cutoff_{itemcode}", f"comm_wl_backto_{itemcode}"
        d1, d2, d3 = st.columns([2, 2, 1])
        with d1:
            cutoff_in = st.date_input(
                "Cutoff date", value=saved_cutoff, max_value=today.date(), key=cutoff_key,
            )
        cutoff_ts = pd.Timestamp(cutoff_in)
        min_back = comm.min_back_to(cutoff_ts).date()
        max_back = (cutoff_ts - pd.Timedelta(days=1)).date()
        just_clamped = _clamp_session_date(back_key, min_back, max_back)
        with d2:
            back_kwargs = dict(min_value=min_back, max_value=max_back, key=back_key)
            if not just_clamped:
                back_kwargs["value"] = min(max(saved_back_to, min_back), max_back)
            back_in = st.date_input("Back to (≤ 6mo before cutoff)", **back_kwargs)
        with d3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Save", key=f"comm_wl_save_{itemcode}"):
                ok = comm.update_product_dates(zid, itemcode, pd.Timestamp(cutoff_in), pd.Timestamp(back_in))
                if ok:
                    st.success("Saved.", icon="✅")
                else:
                    st.error("Could not save — check the dates.")


def _render_watchlist_editor(zid: str, items_df: pd.DataFrame, watchlist: dict) -> None:
    _render_add_product(zid, items_df, watchlist)

    if not watchlist:
        return

    name_lookup = {}
    if items_df is not None and not items_df.empty and "item_id" in items_df.columns:
        name_lookup = dict(zip(items_df["item_id"].astype(str), items_df["item_name"]))

    st.write("**Currently tracked:**")
    for itemcode, dates in watchlist.items():
        _render_tracked_item_editor(zid, itemcode, name_lookup.get(itemcode), dates)


def _comparisons_to_long_df(rows: list) -> pd.DataFrame:
    """Raw-numeric long format (one row per product per period) for CSV export."""
    out = []
    for entry in rows:
        periods = [
            ("Before", entry["before"], entry["back_to"], entry["cutoff"]),
            ("Before Avg (Prorated)", entry["before_avg_prorated"], entry["back_to"], entry["cutoff"]),
            ("After", entry["after"], entry["cutoff"], entry["as_of"]),
        ]
        for period, metrics, wstart, wend in periods:
            out.append({
                "Item Code": entry["itemcode"], "Item Name": entry["itemname"],
                "Item Group": entry["itemgroup"], "Period": period,
                "Window Start": wstart.date(), "Window End": wend.date(),
                "Qty Sold": metrics["qty_sold"], "Sales Revenue": metrics["sales_revenue"],
                "Qty Returned": metrics["qty_returned"], "Net Qty": metrics["net_qty"],
                "Net Revenue": metrics["net_revenue"],
            })
    return pd.DataFrame(out)


def _render_product_tracking(zid: str, read_only: bool = False) -> None:
    st.subheader("📋 Product Tracking")
    items_df = _load_final_items(str(zid))
    watchlist = comm.load_watchlist(zid)

    # Setup (admin Commissions page, read_only=False) vs. results (Target
    # Management's "💰 Commission Results", read_only=True) — confirmed
    # 2026-09-20: "admin has the ear to set up, but target management you
    # can see the results." The admin page shows only the watchlist editor,
    # never the before/after data; Target Management shows only the
    # consolidated table, never editing controls — regardless of viewer role.
    if not read_only:
        st.caption(
            "Pick which products to track and their own cutoff/back-to dates (up to 6 "
            f"months back) — up to {comm.MAX_WATCHLIST_ITEMS} items, scoped to the active "
            "business (ZID). Results show in Target Management → 💰 Commission Results."
        )
        if st.session_state.get("user_role") == "admin":
            with st.expander(
                f"⚙️ Manage Watchlist ({len(watchlist)}/{comm.MAX_WATCHLIST_ITEMS})",
                expanded=not watchlist,
            ):
                _render_watchlist_editor(zid, items_df, watchlist)
        else:
            st.info("Setup is admin-only.")
        return

    st.caption(
        "Before/After comparison for each tracked product — 'After' runs from that "
        "product's own cutoff date to today."
    )
    if not watchlist:
        st.info("No products are being tracked yet. Ask an admin to add some via the Commissions page.")
        return

    today = pd.Timestamp.today().normalize()
    with st.spinner("Loading sales & returns…"):
        sales_df = _load_sales_daily(str(zid))
        returns_df = _load_returns_full(str(zid))

    comparisons = comm.build_product_comparisons(sales_df, returns_df, items_df, watchlist, today)
    if not comparisons:
        st.info("No data for the tracked products.")
        return

    commission_shared.render_before_after_table(comparisons, key_suffix=f"_pt_{zid}")

    long_df = _comparisons_to_long_df(comparisons)
    st.download_button(
        label="⬇ Download CSV (all products, Before + After)",
        data=long_df.to_csv(index=False).encode("utf-8"),
        file_name=f"product_tracking_{zid}_{today:%Y_%m_%d}.csv",
        mime="text/csv",
        key=f"dl_product_tracking{'_ro' if read_only else ''}",
    )


# ── Placeholders — not yet built, see commission_tracking_design.md ────────

def _render_placeholder(section: str, note: str = "") -> None:
    st.subheader(section)
    st.info(
        f"🚧 Not built yet. Fully specified in **commission_tracking_design.md** — "
        f"read that section before building it, don't guess at the shape here."
        + (f"\n\n{note}" if note else "")
    )


def _render_best_performer(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    commission_best_performer_view.render(zid, read_only=read_only, key_suffix=key_suffix)


def _render_customer_best_performer(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    commission_customer_best_performer_view.render(zid, read_only=read_only, key_suffix=key_suffix)


def _render_highest_product_sales(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    commission_rankings_view.render(zid, read_only=read_only, key_suffix=key_suffix)


def _render_app_usage(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    commission_app_usage_view.render(zid, read_only=read_only, key_suffix=key_suffix)


def _render_campaign_payout(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    commission_campaigns_view.render(zid, read_only=read_only, key_suffix=key_suffix)


def _render_individual_target(zid: str, read_only: bool = False) -> None:
    _render_placeholder(
        "🎯 B.2 — Individual Target Achievement",
        "Buildable now. One unconfirmed assumption: whether the 100001/100000 "
        "company-target gate also applies here.",
    )


def _render_customer_acquisition(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    commission_customer_acquisition_view.render(zid, read_only=read_only, key_suffix=key_suffix)


def _sections(read_only: bool, key_suffix: str) -> dict:
    """One entry per commission_tracking_design.md item — every section
    renderer takes `read_only` (Product Tracking's watchlist editor / B.1/3/4's
    roster+create+delete controls all hide when True, regardless of the
    viewer's actual role) so the exact same dict can back both the admin
    Commissions page (read_only=False) and Target Management's manager-facing
    "no setup, just results" Commission Results view (read_only=True, see
    render_section_picker below). A not-yet-built section's placeholder
    ignores read_only — there's nothing to hide yet."""
    return {
        "📋 Product Tracking": lambda zid: _render_product_tracking(zid, read_only=read_only),
        "🏆 Best Performer":
            lambda zid: _render_best_performer(zid, read_only=read_only, key_suffix=key_suffix),
        "🏆 Best Performer (Customers)":
            lambda zid: _render_customer_best_performer(zid, read_only=read_only, key_suffix=key_suffix),
        "📦 Highest Product Sales":
            lambda zid: _render_highest_product_sales(zid, read_only=read_only, key_suffix=key_suffix),
        "📱 App Usage":
            lambda zid: _render_app_usage(zid, read_only=read_only, key_suffix=key_suffix),
        "🎯 Product / Stock Clearance / Slow-Moving Campaign":
            lambda zid: _render_campaign_payout(zid, read_only=read_only, key_suffix=key_suffix),
        "🎯 Individual Target Achievement": _render_individual_target,
        "🆕 Customer Acquisition":
            lambda zid: _render_customer_acquisition(zid, read_only=read_only, key_suffix=key_suffix),
    }


def render_section_picker(zid: str, read_only: bool = False, key_suffix: str = "") -> None:
    """The shared "pick a commission type, see that section" picker — mounted
    on the admin Commissions page (read_only=False) and, per explicit ask
    2026-09-20, on Target Management's "🎯 Commission Results" mode
    (read_only=True) so managers can check results without admin setup
    access. Covers every section in commission_tracking_design.md, built or
    not — a not-yet-built one shows the same placeholder either place."""
    sections = _sections(read_only, key_suffix)
    section = st.selectbox("Commission Campaign", list(sections.keys()), key=f"comm_section{key_suffix}")
    usage_log.log_view("Commissions", section)

    st.divider()
    sections[section](zid)


def display_commissions_page(current_page: str, zid: str) -> None:
    st.title("💰 Sales Commissions & Incentive Tracking")
    st.caption(
        "Design doc: `commission_tracking_design.md` (status: agreed design). "
        "Sections below are built one at a time, discussed before coding."
    )
    render_section_picker(zid, read_only=False)
