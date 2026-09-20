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
from views import commission_campaigns_view
from views.marketing import _load_final_items  # noqa: F401 — re-exported; avoids duplicate cache


# ── Cached loaders ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_sales_daily(zid: str) -> pd.DataFrame:
    df = Analytics("sales_daily_item", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_returns_daily(zid: str) -> pd.DataFrame:
    df = Analytics("returns_daily_item", zid=zid, filters={}).data
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


def _fmt_metric(v, signed: bool = False) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:+,.0f}" if signed else f"{v:,.0f}"


def _render_comparison_table(entry: dict) -> None:
    """One product's own Before/After block — a separate table per product,
    since each one has its own date range and they can't share columns."""
    itemcode = entry["itemcode"]
    itemname = entry["itemname"] or "(name unavailable)"
    itemgroup = entry["itemgroup"] or "—"
    stock = entry["current_stock"]
    before, after = entry["before"], entry["after"]
    change = {k: after[k] - before[k] for k in before}

    with st.container(border=True):
        st.markdown(f"#### `{itemcode}` — {itemname}")
        stock_txt = _fmt_metric(stock)
        st.caption(f"Item Group: {itemgroup} · Current Stock: {stock_txt}")
        st.caption(
            f"Before: **{entry['back_to'].date()} → {entry['cutoff'].date()}**  |  "
            f"After: **{entry['cutoff'].date()} → {entry['as_of'].date()}**"
        )

        metric_cols = ["Qty Sold", "Sales Revenue", "Qty Returned", "Net Qty"]
        key_map = {"Qty Sold": "qty_sold", "Sales Revenue": "sales_revenue",
                   "Qty Returned": "qty_returned", "Net Qty": "net_qty"}
        rows = []
        for label, metrics, signed in [("Before", before, False), ("After", after, False),
                                        ("Change (After − Before)", change, True)]:
            row = {"Period": label}
            for col in metric_cols:
                row[col] = _fmt_metric(metrics[key_map[col]], signed)
            rows.append(row)
        df = pd.DataFrame(rows)

        def _hl_change(row):
            if row["Period"].startswith("Change"):
                return ["font-weight: bold; background-color: rgba(127,127,127,0.15)"] * len(row)
            return [""] * len(row)

        try:
            st.dataframe(df.style.apply(_hl_change, axis=1), width="stretch", hide_index=True)
        except Exception:
            st.dataframe(df, width="stretch", hide_index=True)


def _comparisons_to_long_df(rows: list) -> pd.DataFrame:
    """Raw-numeric long format (one row per product per period) for CSV export."""
    out = []
    for entry in rows:
        for period, metrics in [("Before", entry["before"]), ("After", entry["after"])]:
            out.append({
                "Item Code": entry["itemcode"], "Item Name": entry["itemname"],
                "Item Group": entry["itemgroup"], "Period": period,
                "Window Start": (entry["back_to"] if period == "Before" else entry["cutoff"]).date(),
                "Window End": (entry["cutoff"] if period == "Before" else entry["as_of"]).date(),
                "Qty Sold": metrics["qty_sold"], "Sales Revenue": metrics["sales_revenue"],
                "Qty Returned": metrics["qty_returned"], "Net Qty": metrics["net_qty"],
            })
    return pd.DataFrame(out)


def _render_product_tracking(zid: str) -> None:
    st.subheader("📋 Product Tracking")
    st.caption(
        "A small hand-picked watchlist of products, each with its own Before/After "
        "comparison — pick a cutoff date and how far back to compare (up to 6 months); "
        f"'After' runs from the cutoff to today. Admin-edited, up to "
        f"{comm.MAX_WATCHLIST_ITEMS} items, scoped to the active business (ZID)."
    )

    items_df = _load_final_items(str(zid))
    watchlist = comm.load_watchlist(zid)

    is_admin = st.session_state.get("user_role") == "admin"
    if is_admin:
        with st.expander(
            f"⚙️ Manage Watchlist ({len(watchlist)}/{comm.MAX_WATCHLIST_ITEMS})",
            expanded=not watchlist,
        ):
            _render_watchlist_editor(zid, items_df, watchlist)
        watchlist = comm.load_watchlist(zid)  # pick up any add/remove/save just made
    elif not watchlist:
        st.info("No products are being tracked yet. Ask an admin to add some via ⚙️ Manage Watchlist.")

    if not watchlist:
        return

    today = pd.Timestamp.today().normalize()
    with st.spinner("Loading sales & returns…"):
        sales_df = _load_sales_daily(str(zid))
        returns_df = _load_returns_daily(str(zid))

    comparisons = comm.build_product_comparisons(sales_df, returns_df, items_df, watchlist, today)
    if not comparisons:
        st.info("No data for the tracked products.")
        return

    for entry in comparisons:
        _render_comparison_table(entry)

    long_df = _comparisons_to_long_df(comparisons)
    st.download_button(
        label="⬇ Download CSV (all products, Before + After)",
        data=long_df.to_csv(index=False).encode("utf-8"),
        file_name=f"product_tracking_{zid}_{today:%Y_%m_%d}.csv",
        mime="text/csv",
        key="dl_product_tracking",
    )


# ── Placeholders — not yet built, see commission_tracking_design.md ────────

def _render_placeholder(section: str, note: str = "") -> None:
    st.subheader(section)
    st.info(
        f"🚧 Not built yet. Fully specified in **commission_tracking_design.md** — "
        f"read that section before building it, don't guess at the shape here."
        + (f"\n\n{note}" if note else "")
    )


def _render_best_performer(zid: str) -> None:
    _render_placeholder(
        "🏆 A.1 — Best Performer",
        "One open point: the 3-month averaging method (per-month average vs. pooled "
        "totals) isn't decided yet — confirm with the user before building.",
    )


def _render_highest_product_sales(zid: str) -> None:
    _render_placeholder("📦 A.2 — Highest Product Sales")


def _render_best_app_user(zid: str) -> None:
    _render_placeholder(
        "📱 A.4 — Best App User",
        "Deferred by design — waiting on the user to supply the real list of app "
        "data-hit locations to count. Do not build against the guess in the doc.",
    )


def _render_campaign_payout(zid: str) -> None:
    commission_campaigns_view.render(zid)


def _render_individual_target(zid: str) -> None:
    _render_placeholder(
        "🎯 B.2 — Individual Target Achievement",
        "Buildable now. One unconfirmed assumption: whether the 100001/100000 "
        "company-target gate also applies here.",
    )


def _render_new_customer(zid: str) -> None:
    _render_placeholder(
        "🆕 B.5 — New Customer Creation",
        "Not scoped yet — needs real design work with the user before building "
        "(existing customer-flow logic doesn't fit; first-ever-sale detection "
        "per customer is genuinely new logic).",
    )


_SECTIONS = {
    "📋 Product Tracking": _render_product_tracking,
    "🏆 Best Performer": _render_best_performer,
    "📦 Highest Product Sales": _render_highest_product_sales,
    "📱 Best App User": _render_best_app_user,
    "🎯 Product / Stock Clearance / Slow-Moving Campaign": _render_campaign_payout,
    "🎯 Individual Target Achievement": _render_individual_target,
    "🆕 New Customer Creation": _render_new_customer,
}


def display_commissions_page(current_page: str, zid: str) -> None:
    st.title("💰 Sales Commissions & Incentive Tracking")
    st.caption(
        "Design doc: `commission_tracking_design.md` (status: agreed design). "
        "Sections below are built one at a time, discussed before coding."
    )

    section = st.selectbox("Commission Campaign", list(_SECTIONS.keys()), key="comm_section")
    usage_log.log_view("Commissions", section)

    st.divider()
    _SECTIONS[section](zid)
