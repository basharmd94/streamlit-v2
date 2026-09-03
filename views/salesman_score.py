from __future__ import annotations

import calendar

import pandas as pd
import streamlit as st
from processing import common, salesman_score as ssc

from views._tm_shared import (
    _get_target,
    _load_ar_ledger_clean,
)


# ── Salesman Score ──────────────────────────────────────────────────────────────

# ZID peer map for consolidated scoring (100001 ↔ 100000 share the same field sales team)
_PEER_ZID_PROJ = {
    "100001": ("100000", "GI Corporation"),
    "100000": ("100001", "GULSHAN TRADING"),
}


@st.cache_data(show_spinner=False, ttl=86400)
def _load_score_sales(zid: str, year: int) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("sales", zid=zid, filters={"year": [year]}).data
    if df is None or df.empty:
        return pd.DataFrame()
    result = common.data_copy_add_columns(df.copy())
    return result[0] if result else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=86400)
def _load_score_returns(zid: str, year: int) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("return", zid=zid, filters={"year": [year]}).data
    if df is None or df.empty:
        return pd.DataFrame()
    result = common.data_copy_add_columns(df.copy())
    return result[0] if result else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=86400)
def _load_score_collection(zid: str, year: int) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("collection", zid=zid, filters={"year": [year]}).data
    return df if df is not None else pd.DataFrame()


def _render_salesman_score(sales_df: pd.DataFrame, returns_df: pd.DataFrame, zid, collection_df: pd.DataFrame = None):
    """
    Composite 0-100 performance score per salesman for one selected month —
    the real current month, or one of the 2 months before it, plus a Salesman
    ID column and 3 AR balance columns (selected month + the 2 before it,
    via salesman_due's FIFO trickledown — matches Collection Analysis ->
    Salesman Due -> main due report exactly). Sorted ascending by score
    (lowest/most-attention-needed first).

    Every performance metric (Sales, Collection, Products Sold, Customers
    Visited) is scoped to the selected month only — no 3-month window is used
    for scoring. The balance columns are the one intentional exception: they
    track AR aging (debt accumulated over the 2 months before the selected
    one), which is a different kind of signal than "this month's activity."

    Only shows columns that feed the score (directly or as the displayed
    preview of a value the score function recomputes internally) — see
    processing/salesman_score.compute_salesman_scores for the formula.
    """
    st.subheader("🎯 Salesman Score")
    today = pd.Timestamp.today().normalize()

    month_opts = ssc.month_choices(today)
    sel_label = st.selectbox("Month", [m[0] for m in month_opts], index=0, key="tm_score_month")
    sel_year, sel_month = next((y, m) for (lbl, y, m) in month_opts if lbl == sel_label)
    is_current = ssc.is_real_current_month(sel_year, sel_month, today)

    # ── Consolidation toggle (100001 ↔ 100000 only) ───────────────────────────
    peer_config = _PEER_ZID_PROJ.get(str(zid))
    if peer_config:
        scope_mode = st.radio(
            "Scope",
            [f"ZID {zid} only", "Consolidated (100001 + 100000)"],
            horizontal=True,
            key="tm_score_scope",
        )
        is_consolidated = scope_mode.startswith("Consolidated")
    else:
        is_consolidated = False

    if "date" not in sales_df.columns or "final_sales" not in sales_df.columns:
        st.warning("Required columns missing.")
        return

    df = sales_df.copy()
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")

    mo_start = pd.Timestamp(sel_year, sel_month, 1)
    mo_end_full = pd.Timestamp(sel_year, sel_month, calendar.monthrange(sel_year, sel_month)[1])
    mo_end = today if is_current else mo_end_full

    mo_data = df[(df["_dt"] >= mo_start) & (df["_dt"] <= mo_end)]

    # ── Returns for the selected month, per salesman ──────────────────────────
    ret_by_sp: dict = {}
    if returns_df is not None and not returns_df.empty and "treturnamt" in returns_df.columns:
        _r = returns_df.copy()
        _r["_dt"] = pd.to_datetime(_r["date"], errors="coerce")
        _r_mo = _r[(_r["_dt"] >= mo_start) & (_r["_dt"] <= mo_end)]
        if "spid" in _r_mo.columns:
            ret_by_sp = _r_mo.groupby(_r_mo["spid"].astype(str))["treturnamt"].sum().astype(float).to_dict()

    # ── Collection for the selected month, per salesman ───────────────────────
    coll_by_sp: dict = {}
    if collection_df is not None and not collection_df.empty and "value" in collection_df.columns:
        _c = collection_df.copy()
        _c["spid"] = _c["spid"].astype(str)
        _c["year"] = pd.to_numeric(_c["year"], errors="coerce")
        _c["month"] = pd.to_numeric(_c["month"], errors="coerce")
        coll_by_sp = (
            _c[(_c["year"] == sel_year) & (_c["month"] == sel_month)]
            .groupby("spid")["value"].sum().astype(float).to_dict()
        )

    # ── AR balances: selected month + the 2 before it, from the same FIFO ─────
    # trickledown methodology as Collection Analysis -> Salesman Due -> main
    # due report, so the numbers match that report exactly.
    proj = st.session_state.get("proj")
    with st.spinner("Loading AR ledger…"):
        ar_clean = _load_ar_ledger_clean(str(zid), proj)

    # ── Consolidation data loading ────────────────────────────────────────────
    if is_consolidated and peer_config:
        other_zid, other_proj = peer_config
        with st.spinner(f"Loading ZID {other_zid} data for consolidation…"):
            _o_sales = _load_score_sales(other_zid, sel_year)
            _o_ret   = _load_score_returns(other_zid, sel_year)
            _o_coll  = _load_score_collection(other_zid, sel_year)
            _o_ar    = _load_ar_ledger_clean(other_zid, other_proj)

        if not _o_sales.empty and "date" in _o_sales.columns:
            _os = _o_sales.copy()
            _os["_dt"] = pd.to_datetime(_os["date"], errors="coerce")
            other_mo = _os[(_os["_dt"] >= mo_start) & (_os["_dt"] <= mo_end)]
            mo_data = pd.concat([mo_data, other_mo], ignore_index=True)

        if not _o_ret.empty and "treturnamt" in _o_ret.columns and "date" in _o_ret.columns:
            _r2 = _o_ret.copy()
            _r2["_dt"] = pd.to_datetime(_r2["date"], errors="coerce")
            _r2_mo = _r2[(_r2["_dt"] >= mo_start) & (_r2["_dt"] <= mo_end)]
            if "spid" in _r2_mo.columns:
                for sp, v in _r2_mo.groupby(_r2_mo["spid"].astype(str))["treturnamt"].sum().items():
                    ret_by_sp[sp] = ret_by_sp.get(sp, 0.0) + float(v)

        if not _o_coll.empty and "value" in _o_coll.columns:
            _c2 = _o_coll.copy()
            _c2["spid"]  = _c2["spid"].astype(str)
            _c2["year"]  = pd.to_numeric(_c2["year"],  errors="coerce")
            _c2["month"] = pd.to_numeric(_c2["month"], errors="coerce")
            for sp, v in (_c2[(_c2["year"] == sel_year) & (_c2["month"] == sel_month)]
                           .groupby("spid")["value"].sum().items()):
                coll_by_sp[sp] = coll_by_sp.get(sp, 0.0) + float(v)

        if not _o_ar.empty:
            ar_clean = pd.concat([ar_clean, _o_ar], ignore_index=True)

        if not _o_sales.empty and "spid" in _o_sales.columns:
            sp_extra = _o_sales[["spid", "spname"]].dropna().drop_duplicates()
            sp_list = pd.concat(
                [df[["spid", "spname"]].dropna().drop_duplicates(), sp_extra],
                ignore_index=True,
            ).drop_duplicates("spid").sort_values("spname").reset_index(drop=True)
        else:
            sp_list = df[["spid", "spname"]].dropna().drop_duplicates().sort_values("spname")
    else:
        sp_list = df[["spid", "spname"]].dropna().drop_duplicates().sort_values("spname")

    cur = pd.Timestamp(sel_year, sel_month, 1)
    bal_months = [
        ((cur - pd.DateOffset(months=i)).strftime("%b %Y"),
         int((cur - pd.DateOffset(months=i)).year),
         int((cur - pd.DateOffset(months=i)).month))
        for i in (2, 1, 0)
    ]
    oldest_label, mid_label, newest_label = [b[0] for b in bal_months]
    bal_table = ssc.compute_salesman_balances_trickledown(ar_clean, months_back=5)

    def _bal_lookup(spid: str, y: int, m: int) -> float:
        col = f"{y}_{m:02d}"
        if bal_table.empty or col not in bal_table.columns or spid not in bal_table.index:
            return 0.0
        return float(bal_table.loc[spid, col])

    bal_by_month = {label: (y, m) for label, y, m in bal_months}

    # ── Build per-salesman rows ────────────────────────────────────────────────

    rows = []
    for _, sp_row in sp_list.iterrows():
        spid = str(sp_row["spid"])
        spname = sp_row["spname"]

        sp_mo = mo_data[mo_data["spid"].astype(str) == spid]

        sales = float(sp_mo["final_sales"].sum())
        ret = round(ret_by_sp.get(spid, 0.0), 0)
        net_sales = round(sales - ret, 0)

        if is_consolidated:
            target = (
                float(_get_target("100001", spid, sel_year, sel_month) or 0.0)
                + float(_get_target("100000", spid, sel_year, sel_month) or 0.0)
            )
        else:
            target = float(_get_target(zid, spid, sel_year, sel_month) or 0.0)
        pct_tgt = round(net_sales / target * 100, 1) if target > 0 else None

        coll = round(float(coll_by_sp.get(spid, 0.0)), 0)
        pct_coll = round(coll / net_sales * 100, 1) if net_sales > 0 else None

        uc_mo = int(sp_mo["cusid"].nunique()) if "cusid" in sp_mo.columns else 0
        up_mo = int(sp_mo["itemcode"].nunique()) if "itemcode" in sp_mo.columns else 0

        bal_oldest = _bal_lookup(spid, *bal_by_month[oldest_label])
        bal_mid = _bal_lookup(spid, *bal_by_month[mid_label])
        bal_newest = _bal_lookup(spid, *bal_by_month[newest_label])

        rows.append({
            "spid": spid,
            "Salesman ID": spid,
            "Salesman": spname,
            "Target": target,
            "Sales": round(sales, 0),
            "Return": ret,
            "Net Sales": net_sales,
            "% vs Target": pct_tgt,
            "Collection": coll,
            "% Collection": pct_coll,
            "Customers Visited": uc_mo,
            "Products Sold": up_mo,
            f"Balance ({oldest_label})": round(bal_oldest, 0),
            f"Balance ({mid_label})": round(bal_mid, 0),
            f"Balance ({newest_label})": round(bal_newest, 0),
            # scoring inputs — not displayed directly under these keys
            "target": target, "sales": sales, "net_sales": net_sales, "coll": coll,
            "uniq_prods": up_mo, "uniq_cust": uc_mo,
            "balance_recent2": bal_oldest + bal_mid,
            "balance_this_month": bal_newest,
        })

    if not rows:
        st.info("No salesmen found for this selection.")
        return

    scored = ssc.compute_salesman_scores(pd.DataFrame(rows))

    bal_cols = [f"Balance ({oldest_label})", f"Balance ({mid_label})", f"Balance ({newest_label})"]
    display_cols = [
        "Salesman ID", "Salesman", "Target", "Sales", "Return", "Net Sales", "% vs Target",
        "Collection", "% Collection", "Customers Visited", "Products Sold",
        *bal_cols, "score",
    ]
    t = scored[display_cols].rename(columns={"score": "Score"}).reset_index(drop=True)
    has_target_mask = scored["has_target"].reset_index(drop=True)

    fmt = {
        "Target": "{:,.0f}", "Sales": "{:,.0f}", "Return": "{:,.0f}", "Net Sales": "{:,.0f}",
        "% vs Target": lambda v: f"{v:.1f}%" if v is not None else "—",
        "Collection": "{:,.0f}",
        "% Collection": lambda v: f"{v:.1f}%" if v is not None else "—",
        "Customers Visited": "{:,.0f}",
        "Products Sold": "{:,.0f}",
        **{c: "{:,.0f}" for c in bal_cols},
        "Score": "{:.1f}",
    }

    def _row_style(row):
        if not has_target_mask.iloc[row.name]:
            return ["background-color: #F8D7DA"] * len(row)
        return [""] * len(row)

    try:
        styled = t.style.format(fmt, na_rep="—").apply(_row_style, axis=1)
        st.dataframe(styled, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(t, width="stretch", hide_index=True)

    newest_asof = "as of today" if is_current else f"as of {mo_end_full.strftime('%b %d')}"
    st.caption(
        f"**Score** (0–100, sorted lowest first): 45% Sales vs Target + 45% Collection vs Net Sales "
        f"(both capped at 100%) + 5% Products Sold + 5% Customers Visited (both for {sel_label} only) "
        f"— the last two scored relative to the top salesman in this table — minus up to 20% in negative points, "
        f"also peer-relative: 6 pts for Return vs Sales %, 12 pts for the combined "
        f"{oldest_label} + {mid_label} balance, 2 pts for the {newest_label} balance. "
        f"Rows highlighted red have no target set for {sel_label} — scored 0 on that 45% component. "
        f"Balance columns are point-in-time snapshots: {newest_label} is {newest_asof}, "
        f"{mid_label} and {oldest_label} are as of their own month-end.  \n"
        f"ℹ️ **% Collection** (column & score component) = Collection ÷ Net Sales, straight percentage — "
        f"same formula as All Salesmen Overview's % Collection column now (both previously used "
        f"gross Sales, and this one also previously had a 1.02 buffer)."
    )

    st.download_button(
        "⬇ Download Salesman Score CSV",
        t.to_csv(index=False).encode("utf-8"),
        file_name=f"salesman_score_{'consolidated' if is_consolidated else zid}_{sel_year}_{sel_month:02d}.csv",
        mime="text/csv",
        key="dl_salesman_score",
    )


