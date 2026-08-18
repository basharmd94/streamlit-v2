# views/glpmt_shared.py
# Shared "App Collections" panel — payments salesmen enter directly into the
# mobile Ordering app (table: glpmt), staged pending reconciliation into the
# real GL ledger. Same panel is mounted in both Collection Analysis and
# Target Management, so this module is the single source of truth for it.

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.analytics import Analytics


@st.cache_data(show_spinner=False, ttl=300)
def _load_glpmt(zid: str) -> pd.DataFrame:
    df = Analytics("glpmt", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


def render_glpmt_panel(zid: str, key_suffix: str = "") -> None:
    """Filters: Salesman (emp code), Customer, Date of Entry (range).
    Always sorted by date of entry, latest first — per spec, not user-toggleable."""
    st.subheader("📲 App Collections")
    st.caption(
        "Payments salesmen have entered directly into the mobile Ordering app — "
        "staged here pending reconciliation into the ERP ledger. "
        "Sorted by date of entry, latest first."
    )

    df = _load_glpmt(str(zid))
    if df.empty:
        st.info("No app-entered payments found for this business.")
        return

    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["paydate"]     = pd.to_datetime(df["paydate"], errors="coerce")

    # ── Filters ────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns(3)
    with f1:
        sp_opts = sorted(df["spid"].dropna().unique().tolist())
        sp_name_map = df.drop_duplicates("spid").set_index("spid")["spname"].to_dict()
        sel_sp = st.multiselect(
            "Salesman (Emp Code)",
            sp_opts,
            format_func=lambda x: f"{x} — {sp_name_map.get(x, '')}",
            key=f"glpmt_sp{key_suffix}",
        )
    with f2:
        cus_opts = sorted(df["cusid"].dropna().unique().tolist())
        cus_name_map = df.drop_duplicates("cusid").set_index("cusid")["cusname"].to_dict()
        sel_cus = st.multiselect(
            "Customer",
            cus_opts,
            format_func=lambda x: f"{x} — {cus_name_map.get(x, '')}",
            key=f"glpmt_cus{key_suffix}",
        )
    with f3:
        entry_dates = df["entry_time"].dt.date.dropna()
        date_range = None
        if not entry_dates.empty:
            date_range = st.date_input(
                "Date of Entry (range)",
                value=(entry_dates.min(), entry_dates.max()),
                key=f"glpmt_daterange{key_suffix}",
            )

    disp = df.copy()
    if sel_sp:
        disp = disp[disp["spid"].isin(sel_sp)]
    if sel_cus:
        disp = disp[disp["cusid"].isin(sel_cus)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        disp = disp[
            (disp["entry_time"].dt.date >= start) & (disp["entry_time"].dt.date <= end)
        ]

    # Sorted by date of entry, latest first — non-negotiable per spec.
    disp = disp.sort_values("entry_time", ascending=False).reset_index(drop=True)

    st.caption(f"**{len(disp):,}** payment entr{'y' if len(disp) == 1 else 'ies'}")

    show = disp.rename(columns={
        "pmtnum":     "Payment #",
        "spid":       "Emp Code",
        "spname":     "Salesman",
        "cusid":      "Cust Code",
        "cusname":    "Customer",
        "paydate":    "Payment Date",
        "payamt":     "Amount",
        "paytype":    "Type",
        "bankdetail": "Bank Detail",
        "paystatus":  "Status",
        "remarks":    "Remarks",
        "entry_time": "Date of Entry",
    })
    show_cols = [
        "Date of Entry", "Payment #", "Emp Code", "Salesman",
        "Cust Code", "Customer", "Payment Date", "Amount",
        "Type", "Bank Detail", "Status", "Remarks",
    ]
    show = show[[c for c in show_cols if c in show.columns]]

    st.dataframe(
        show,
        column_config={
            "Date of Entry": st.column_config.DatetimeColumn("Date of Entry", format="YYYY-MM-DD HH:mm"),
            "Payment Date":  st.column_config.DateColumn("Payment Date", format="YYYY-MM-DD"),
            "Amount":        st.column_config.NumberColumn("Amount", format="%.2f"),
        },
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "⬇ Download CSV",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name=f"app_collections_{zid}.csv",
        mime="text/csv",
        key=f"glpmt_dl{key_suffix}",
    )
