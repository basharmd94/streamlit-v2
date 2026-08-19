# views/returns_registry.py
# Returns Registry — customer returns salesmen log directly into the mobile
# Ordering app, still pending approval (opcrn.xstatuscrn = '1-Open'). Table 1:
# one row per return header (date, customer, total BDT, reason). Table 2:
# product line items (opcdt) for whichever customer is selected via the
# filter above Table 1.

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.analytics import Analytics


@st.cache_data(show_spinner=False, ttl=300)
def _load_returns_registry(zid: str) -> pd.DataFrame:
    df = Analytics("returns_registry", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=300)
def _load_returns_registry_items(zid: str) -> pd.DataFrame:
    df = Analytics("returns_registry_items", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


def _render_returns_registry(zid: str) -> None:
    st.subheader("↩️ Returns Registry")
    st.caption(
        "Customer returns salesmen have logged into the mobile Ordering app, "
        "still pending approval (status: 1-Open). Sorted by date, latest first."
    )

    df = _load_returns_registry(str(zid))
    if df.empty:
        st.info("No open (unreconciled) returns found for this business.")
        return

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ── Filters ────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        sp_opts = sorted(df["spid"].dropna().unique().tolist())
        sp_name_map = df.drop_duplicates("spid").set_index("spid")["spname"].to_dict()
        sel_sp = st.multiselect(
            "Salesman (Emp Code)", sp_opts,
            format_func=lambda x: f"{x} — {sp_name_map.get(x, '')}",
            key="rr_sp",
        )
    with f2:
        # Single-select: this filter both narrows Table 1 AND drives Table 2
        # (the products list below only makes sense for one customer at a time).
        cus_opts_df = (
            df[["cusid", "cusname"]].drop_duplicates("cusid").sort_values("cusname")
        )
        cus_labels = {
            f"{r['cusname']} ({r['cusid']})": r["cusid"] for _, r in cus_opts_df.iterrows()
        }
        sel_cus_label = st.selectbox(
            "Customer", ["— All Customers —"] + list(cus_labels.keys()), key="rr_cus",
        )
        sel_cus = cus_labels.get(sel_cus_label)
    with f3:
        dates = df["date"].dt.date.dropna()
        date_range = None
        if not dates.empty:
            date_range = st.date_input(
                "Date (range)", value=(dates.min(), dates.max()), key="rr_daterange",
            )

    disp = df.copy()
    if sel_sp:
        disp = disp[disp["spid"].isin(sel_sp)]
    if sel_cus:
        disp = disp[disp["cusid"] == sel_cus]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        disp = disp[(disp["date"].dt.date >= start) & (disp["date"].dt.date <= end)]

    disp = disp.sort_values("date", ascending=False).reset_index(drop=True)

    st.caption(f"**{len(disp):,}** open return(s)")

    show = disp.rename(columns={
        "crnnum":    "Return #",
        "date":      "Date",
        "cusid":     "Cust Code",
        "cusname":   "Customer",
        "spid":      "Emp Code",
        "spname":    "Salesman",
        "reason":    "Reason",
        "total_amt": "Total (BDT)",
    })
    show_cols = ["Date", "Return #", "Cust Code", "Customer", "Emp Code", "Salesman", "Total (BDT)", "Reason"]
    show = show[[c for c in show_cols if c in show.columns]]

    st.dataframe(
        show,
        column_config={
            "Date":        st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Total (BDT)": st.column_config.NumberColumn("Total (BDT)", format="%.2f"),
        },
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "⬇ Download CSV",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name=f"returns_registry_{zid}.csv",
        mime="text/csv",
        key="rr_dl",
    )

    # ── Table 2: product line items for the customer selected above ─────────
    st.markdown("---")
    st.markdown("#### 📦 Products Being Returned")

    if not sel_cus:
        st.info("Select a customer above (Customer filter) to see their returned products.")
        return

    cust_crns = disp["crnnum"].unique().tolist()
    if not cust_crns:
        st.info("No open returns match the current filters for this customer.")
        return

    items_df = _load_returns_registry_items(str(zid))
    if items_df.empty:
        st.info("No product line items found.")
        return

    cust_items = items_df[items_df["crnnum"].isin(cust_crns)].copy()
    if cust_items.empty:
        st.info("No product line items found for this customer's open return(s).")
        return

    # Attach return date/reason for context when a customer has multiple open returns.
    cust_items = cust_items.merge(
        disp[["crnnum", "date", "reason"]], on="crnnum", how="left"
    )

    items_show = cust_items.rename(columns={
        "crnnum":   "Return #",
        "date":     "Date",
        "itemcode": "Item Code",
        "itemname": "Item Name",
        "qty":      "Qty",
        "rate":     "Rate",
        "lineamt":  "Line Amount",
        "reason":   "Reason",
    })
    items_cols = ["Return #", "Date", "Item Code", "Item Name", "Qty", "Rate", "Line Amount", "Reason"]
    items_show = items_show[[c for c in items_cols if c in items_show.columns]]
    items_show = items_show.sort_values(["Return #"]).reset_index(drop=True)

    cust_name = sel_cus_label.split(" (")[0]
    st.caption(f"**{len(items_show):,}** product line(s) for **{cust_name}**")
    st.dataframe(
        items_show,
        column_config={
            "Date":        st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Qty":         st.column_config.NumberColumn("Qty", format="%.2f"),
            "Rate":        st.column_config.NumberColumn("Rate", format="%.2f"),
            "Line Amount": st.column_config.NumberColumn("Line Amount", format="%.2f"),
        },
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "⬇ Download Products CSV",
        data=items_show.to_csv(index=False).encode("utf-8"),
        file_name=f"returns_registry_products_{zid}_{sel_cus}.csv",
        mime="text/csv",
        key="rr_items_dl",
    )
