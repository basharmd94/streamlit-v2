from __future__ import annotations

import pandas as pd
import streamlit as st
from processing import next_month_target as nmt

from views._tm_shared import (
    _load_sales_window,
    _load_returns_window,
    _load_final_items,
    _load_purchase_open_combined,
)


# ── Next Month Target ──────────────────────────────────────────────────────────

_SALES_ONLY_NMT_ZIDS = ("100000", "100005")  # GI, Zepto — no shipment/stock tracking in this tool


def _render_next_month_target(zid):
    st.subheader("🔮 Next Month Target Estimate")
    st.caption("📝 Methodology — see Note [1] at the bottom of the page.")
    sales_only = str(zid) in _SALES_ONLY_NMT_ZIDS
    note3 = None  # only set on the stock-capped path (shipment picker below)

    if sales_only:
        note1 = (
            "Estimates next month's sales range purely from each item's historical net sales "
            "(qty/amt, after returns) over the trailing 12 completed months — low/median/high are "
            "the lowest non-zero, median (non-zero), and highest active months. This business has "
            "no stock or shipment tracking wired into this tool, so the estimate is **not** capped "
            "by stock on hand."
        )
    else:
        note1 = (
            "Estimates an achievable sales range for next month from current stock plus an optional "
            "incoming shipment, bounded by each item's lowest non-zero, median (non-zero), and "
            "highest net sales (qty/amt, after returns) over the trailing 12 completed months. "
            "Months with zero net sales are ignored when picking the 'low'/'median'/average figures. "
            "Current stock is a snapshot as of today, so it's first projected forward to next "
            "month's start by subtracting the expected sales for the rest of *this* month "
            "(trailing-12-month average active-month qty ÷ days in this month × days remaining)."
        )

    today = pd.Timestamp.today().normalize()
    hist_start, hist_end = nmt.trailing_12mo_window(today)
    next_month_start = pd.Timestamp(today.year, today.month, 1) + pd.DateOffset(months=1)
    next_month_end = next_month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    years = tuple(sorted({int(hist_start.year), int(hist_end.year)}))

    with st.spinner("Loading trailing 12-month sales history…"):
        sales_12mo = _load_sales_window(str(zid), years)
        returns_12mo = _load_returns_window(str(zid), years)

    monthly_net = nmt.compute_item_monthly_net(sales_12mo, returns_12mo, hist_start, hist_end)
    low_high = nmt.compute_item_low_high(monthly_net)

    st.markdown(
        f"**Historical window:** {hist_start.strftime('%b %Y')} – {hist_end.strftime('%b %Y')} "
        f"(12 completed months) &nbsp;|&nbsp; **Target month:** {next_month_start.strftime('%b %Y')}"
    )

    if sales_only:
        item_names = nmt.extract_item_names(sales_12mo)
        estimate_df = nmt.build_sales_only_estimate(low_high, item_names)
    else:
        with st.spinner("Loading stock and open shipments…"):
            stock_df = _load_final_items(str(zid))
            purchase_df = _load_purchase_open_combined()
        open_shipments = nmt.get_open_shipments(purchase_df)

        col1, col2 = st.columns(2)
        with col1:
            option_labels = {
                f"{row.shipmentname}  (ZID {row.zid})": (row.zid, row.shipmentname)
                for row in open_shipments.itertuples()
            }
            note3 = (
                "Select both the 100001 and 100009 entries if they're the same physical "
                "shipment split across books."
            )
            sel_labels = st.multiselect(
                "Incoming Shipment(s) (open, not yet received) — see Note [2] below",
                options=list(option_labels.keys()),
                key=f"nmt_shipments_{zid}",
            )
        with col2:
            entry_date = st.date_input(
                "Estimated Date of Entry (applies to all selected shipments)",
                value=next_month_start.date(), key=f"nmt_entry_date_{zid}",
            )

        sel_selections = [option_labels[lbl] for lbl in sel_labels]
        shipment_items = None
        incoming_fraction = 1.0
        if sel_selections:
            names_str = ", ".join(f"{name} (ZID {z})" for z, name in sel_selections)
            entry_ts = pd.Timestamp(entry_date)
            if entry_ts <= next_month_end:
                shipment_items = nmt.get_shipment_items(purchase_df, sel_selections)
                incoming_fraction, days_remaining, days_in_month = nmt.incoming_proration(
                    entry_ts, next_month_start, next_month_end
                )
                if incoming_fraction >= 1.0:
                    st.success(
                        f"Including shipment(s) **{names_str}** "
                        f"({len(shipment_items)} item(s) after cross-ZID code mapping) — "
                        f"arrives {entry_date}, on/before {next_month_start.strftime('%b %Y')} starts "
                        f"— full month available."
                    )
                else:
                    st.success(
                        f"Including shipment(s) **{names_str}** "
                        f"({len(shipment_items)} item(s) after cross-ZID code mapping) — "
                        f"arrives {entry_date}, leaving {days_remaining} of {days_in_month} days in "
                        f"{next_month_start.strftime('%b %Y')} ({incoming_fraction:.0%}). Incoming "
                        f"quantities are scaled down to that fraction before capping the estimate."
                    )
            else:
                st.warning(
                    f"Selected shipment(s) **{names_str}** are estimated to arrive {entry_date}, "
                    f"after {next_month_start.strftime('%b %Y')} ends — excluded from this estimate."
                )

        estimate_df = nmt.build_next_month_estimate(
            stock_df, low_high, shipment_items, incoming_fraction, today=today
        )

    if estimate_df.empty:
        if sales_only:
            st.info("No items with sales history were found in the trailing 12-month window.")
        else:
            st.info("No items with available stock (current + incoming) and sales history were found.")
        return

    total_low = float(estimate_df["est_low_amt"].sum())
    total_median = float(estimate_df["est_median_amt"].sum())
    total_high = float(estimate_df["est_high_amt"].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimated Low", f"{total_low:,.0f}")
    m2.metric("Estimated Median", f"{total_median:,.0f}")
    m3.metric("Estimated High", f"{total_high:,.0f}")
    m4.metric("Items Counted", f"{len(estimate_df):,}")

    _col_map = {
        "itemcode": "Item Code", "itemname": "Item Name", "itemgroup": "Item Group",
        "current_stock": "Current Stock", "projected_stock": "Projected Stock (Month Start)",
        "incoming_qty": "Incoming Qty", "available_qty": "Available Qty",
        "low_qty": "12M Low Qty", "median_qty": "12M Median Qty", "high_qty": "12M High Qty",
        "avg_qty": "12M Avg Qty (active)", "avg_price": "Avg Price",
        "est_low_qty": "Est. Low Qty", "est_median_qty": "Est. Median Qty", "est_high_qty": "Est. High Qty",
        "est_low_amt": "Est. Low Amt", "est_median_amt": "Est. Median Amt", "est_high_amt": "Est. High Amt",
    }
    disp = estimate_df.rename(columns=_col_map)
    cols_order = [c for c in _col_map.values() if c in disp.columns]
    disp = disp[cols_order]

    st.dataframe(
        disp.style.format({
            "Current Stock": "{:,.0f}", "Projected Stock (Month Start)": "{:,.0f}",
            "Incoming Qty": "{:,.0f}", "Available Qty": "{:,.0f}",
            "12M Low Qty": "{:,.0f}", "12M Median Qty": "{:,.0f}", "12M High Qty": "{:,.0f}",
            "12M Avg Qty (active)": "{:,.2f}", "Avg Price": "{:,.2f}",
            "Est. Low Qty": "{:,.0f}", "Est. Median Qty": "{:,.0f}", "Est. High Qty": "{:,.0f}",
            "Est. Low Amt": "{:,.0f}", "Est. Median Amt": "{:,.0f}", "Est. High Amt": "{:,.0f}",
        }, na_rep="—"),
        width="stretch", hide_index=True,
    )

    # ── Salesman × Area performance ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("👤 Salesman × Area Performance")

    st.caption("📝 Methodology — see Note [3] at the bottom of the page.")

    salesman_capped = None  # only populated (and used) on the stock-capped path
    if sales_only:
        note2 = (
            "For each salesman, accumulates net sales amount (after returns) across every product "
            "they sold within each area, then takes the lowest non-zero, median (non-zero), and "
            "highest month from that combined monthly total over the same trailing 12-month window."
        )
        sp_area_monthly = nmt.compute_salesman_area_monthly_net(sales_12mo, returns_12mo, hist_start, hist_end)
        sp_area_lh = nmt.compute_salesman_area_low_high(sp_area_monthly)
        sp_area_lh = sp_area_lh[sp_area_lh["high_amt"] > 0].copy()
    else:
        note2 = (
            "Each item's already stock-capped Low/Median/High estimate (the same figures shown in "
            "the table above) is split across the salesmen who sell it by their historical share of "
            "that item's net qty — so combined salesman totals reconstruct, never exceed, the product "
            "table's own totals. Areas split a salesman's allocation by their historical mix across areas."
        )
        item_spid_monthly = nmt.compute_item_spid_monthly_net(sales_12mo, returns_12mo, hist_start, hist_end)
        item_spid_lh = nmt.compute_item_spid_low_high(item_spid_monthly)
        salesman_capped = nmt.allocate_item_estimate_by_salesman(item_spid_lh, estimate_df)

        item_spid_area_totals = nmt.compute_item_spid_area_totals(sales_12mo, returns_12mo, hist_start, hist_end)
        sp_area_lh = nmt.allocate_salesman_amt_by_area(salesman_capped, item_spid_area_totals)
        sp_area_lh = sp_area_lh[sp_area_lh["high_amt"] > 0].copy()

    if sp_area_lh.empty:
        st.info("No salesman sales history found in the trailing 12-month window.")
    else:
        sp_names = nmt.extract_salesman_names(sales_12mo)
        sp_area_disp = sp_area_lh.merge(sp_names, on="spid", how="left")
        sp_area_disp["spname"] = sp_area_disp["spname"].fillna("")
        sp_area_disp = sp_area_disp.sort_values("high_amt", ascending=False).reset_index(drop=True)

        sa1, sa2, sa3 = st.columns(3)
        sa1.metric("Total Low", f"{sp_area_disp['low_amt'].sum():,.0f}")
        sa2.metric("Total Median", f"{sp_area_disp['median_amt'].sum():,.0f}")
        sa3.metric("Total High", f"{sp_area_disp['high_amt'].sum():,.0f}")

        sp_area_col_map = {
            "spid": "Salesman Code", "spname": "Salesman Name", "area": "Area",
            "low_amt": "Low Amount", "median_amt": "Median Amount", "high_amt": "High Amount",
        }
        sp_area_view = sp_area_disp.rename(columns=sp_area_col_map)
        sp_area_view = sp_area_view[[c for c in sp_area_col_map.values() if c in sp_area_view.columns]]

        st.dataframe(
            sp_area_view.style.format({
                "Low Amount": "{:,.0f}", "Median Amount": "{:,.0f}", "High Amount": "{:,.0f}",
            }, na_rep="—"),
            width="stretch", hide_index=True,
        )

        # ── Per-salesman product breakdown ─────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Salesman Product Breakdown")

        sp_names_active = sp_names[sp_names["spid"].isin(sp_area_disp["spid"])].copy()
        sp_options_map = {
            f"{row.spname} ({row.spid})": row.spid
            for row in sp_names_active.sort_values("spname").itertuples()
        }
        sel_sp_label = st.selectbox(
            "Select Salesman", ["— Select —"] + list(sp_options_map.keys()), key=f"nmt_sp_drilldown_{zid}"
        )

        if sel_sp_label != "— Select —":
            sel_spid = sp_options_map[sel_sp_label]

            if sales_only:
                sales_for_sp = (
                    sales_12mo[sales_12mo["spid"].astype(str) == sel_spid]
                    if sales_12mo is not None and not sales_12mo.empty and "spid" in sales_12mo.columns
                    else pd.DataFrame()
                )
                returns_for_sp = (
                    returns_12mo[returns_12mo["spid"].astype(str) == sel_spid]
                    if returns_12mo is not None and not returns_12mo.empty and "spid" in returns_12mo.columns
                    else pd.DataFrame()
                )
                monthly_net_sp = nmt.compute_item_monthly_net(sales_for_sp, returns_for_sp, hist_start, hist_end)
                low_high_sp = nmt.compute_item_low_high(monthly_net_sp)
                item_names_sp = nmt.extract_item_names(sales_for_sp)
                drill_df = nmt.build_sales_only_estimate(low_high_sp, item_names_sp)
            else:
                # Reuse the same stock-capped allocation as the area table above —
                # this salesman's own low/median/high per item, already capped at
                # their fair share of each item's available stock.
                drill_df = salesman_capped[salesman_capped["spid"] == sel_spid].copy()
                drill_df = drill_df[drill_df["est_high_qty"] > 0].copy()
                if not drill_df.empty:
                    item_names_sp = nmt.extract_item_names(sales_12mo)
                    drill_df = drill_df.merge(item_names_sp, on="itemcode", how="left")
                    drill_df["itemname"] = drill_df["itemname"].fillna("")
                    drill_df["itemgroup"] = drill_df["itemgroup"].fillna("")
                    drill_df = drill_df.sort_values("est_high_amt", ascending=False).reset_index(drop=True)

            if drill_df.empty:
                st.info("No product sales history found for this salesman in the trailing 12-month window.")
            else:
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Estimated Low", f"{drill_df['est_low_amt'].sum():,.0f}")
                d2.metric("Estimated Median", f"{drill_df['est_median_amt'].sum():,.0f}")
                d3.metric("Estimated High", f"{drill_df['est_high_amt'].sum():,.0f}")
                d4.metric("Items Counted", f"{len(drill_df):,}")

                drill_col_map = {
                    "itemcode": "Item Code", "itemname": "Item Name", "itemgroup": "Item Group",
                    "low_qty": "12M Low Qty", "median_qty": "12M Median Qty", "high_qty": "12M High Qty",
                    "avg_price": "Avg Price",
                    "est_low_amt": "Est. Low Amt", "est_median_amt": "Est. Median Amt",
                    "est_high_amt": "Est. High Amt",
                }
                drill_disp = drill_df.rename(columns=drill_col_map)
                drill_disp = drill_disp[[c for c in drill_col_map.values() if c in drill_disp.columns]]

                st.dataframe(
                    drill_disp.style.format({
                        "12M Low Qty": "{:,.0f}", "12M Median Qty": "{:,.0f}", "12M High Qty": "{:,.0f}",
                        "Avg Price": "{:,.2f}",
                        "Est. Low Amt": "{:,.0f}", "Est. Median Amt": "{:,.0f}", "Est. High Amt": "{:,.0f}",
                    }, na_rep="—"),
                    width="stretch", hide_index=True,
                )

    # ── Notes ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 Notes")
    st.markdown(f"**[1] Next Month Target Estimate** — {note1}")
    if note3:
        st.markdown(f"**[2] Incoming Shipment(s)** — {note3}")
    st.markdown(f"**[3] Salesman × Area Performance** — {note2}")


