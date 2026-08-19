# views/feedback.py
# Feedback — market-level feedback salesmen log via the mobile Ordering app.
# One underlying table (`feedback`), split into four category views: Customer,
# Product, Delivery Issue, Collection Issue. A single entry can carry more than
# one tag at once (e.g. a product issue tied to a specific customer), so it may
# legitimately appear in more than one of the four tables below.

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.analytics import Analytics

_CATEGORIES = {
    "👤 Customer Feedback": {
        "mask": lambda df: df["cusid"].notna() & (df["cusid"] != ""),
        "id_cols": ["cusid", "cusname"],
        "id_rename": {"cusid": "Cust Code", "cusname": "Customer"},
        "empty_msg": "No customer feedback found for this business.",
    },
    "📦 Product Feedback": {
        "mask": lambda df: df["itemcode"].notna() & (df["itemcode"] != ""),
        "id_cols": ["itemcode", "itemname"],
        "id_rename": {"itemcode": "Item Code", "itemname": "Item Name"},
        "empty_msg": "No product feedback found for this business.",
    },
    "🚚 Delivery Issue": {
        "mask": lambda df: df["is_delivery_issue"].fillna(False) == True,  # noqa: E712
        "id_cols": [],
        "id_rename": {},
        "empty_msg": "No delivery issues found for this business.",
    },
    "💰 Collection Issue": {
        "mask": lambda df: df["is_collection_issue"].fillna(False) == True,  # noqa: E712
        "id_cols": [],
        "id_rename": {},
        "empty_msg": "No collection issues found for this business.",
    },
}


@st.cache_data(show_spinner=False, ttl=300)
def _load_feedback(zid: str) -> pd.DataFrame:
    df = Analytics("feedback", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


def _render_feedback(zid: str) -> None:
    st.subheader("💬 Feedback")
    st.caption(
        "Market-level feedback salesmen log via the mobile Ordering app — about "
        "a customer, a product, a delivery issue, or a collection issue. A single "
        "entry can belong to more than one category (e.g. a product issue tied to "
        "a specific customer), so it may appear in more than one table below."
    )

    df = _load_feedback(str(zid))
    if df.empty:
        st.info("No feedback entries found for this business.")
        return

    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    cat_label = st.radio(
        "Feedback Type", list(_CATEGORIES.keys()), horizontal=True, key="fb_category",
    )
    cat = _CATEGORIES[cat_label]

    cat_df = df[cat["mask"](df)].copy()
    if cat_df.empty:
        st.info(cat["empty_msg"])
        return

    # ── Filters ────────────────────────────────────────────────────────────
    f1, f2 = st.columns(2)
    with f1:
        sp_opts = sorted(s for s in cat_df["spid"].dropna().unique().tolist() if s)
        sp_name_map = cat_df.drop_duplicates("spid").set_index("spid")["spname"].to_dict()
        sel_sp = st.multiselect(
            "Salesman (Emp Code)", sp_opts,
            format_func=lambda x: f"{x} — {sp_name_map.get(x, '')}",
            key="fb_sp",
        )
    with f2:
        dates = cat_df["created_at"].dt.date.dropna()
        date_range = None
        if not dates.empty:
            date_range = st.date_input(
                "Date (range)", value=(dates.min(), dates.max()), key="fb_daterange",
            )

    disp = cat_df.copy()
    if sel_sp:
        disp = disp[disp["spid"].isin(sel_sp)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        disp = disp[(disp["created_at"].dt.date >= start) & (disp["created_at"].dt.date <= end)]

    disp = disp.sort_values("created_at", ascending=False).reset_index(drop=True)

    label_noun = cat_label.split(" ", 1)[1]
    st.caption(f"**{len(disp):,}** {label_noun.lower()} entr{'y' if len(disp) == 1 else 'ies'}")

    rename = {
        "created_at": "Date", "spid": "Emp Code", "spname": "Salesman",
        "description": "Feedback",
        **cat["id_rename"],
    }
    show_cols = ["created_at", "spid", "spname"] + cat["id_cols"] + ["description"]
    show = disp[[c for c in show_cols if c in disp.columns]].rename(columns=rename)

    st.dataframe(
        show,
        column_config={
            "Date":     st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
            "Feedback": st.column_config.TextColumn("Feedback", width="large"),
        },
        width="stretch",
        hide_index=True,
    )

    file_slug = label_noun.strip().lower().replace(" ", "_")
    st.download_button(
        "⬇ Download CSV",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name=f"feedback_{file_slug}_{zid}.csv",
        mime="text/csv",
        key="fb_dl",
    )
