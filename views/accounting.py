import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple
from core.analytics import Analytics
from processing import common
from utils.utils import timed


# ---------- Cached loaders (simple table pulls) ----------
@st.cache_data(show_spinner=False, ttl=86400)
def _load_cacus(zid: str) -> pd.DataFrame:
    df = Analytics("cacus_simple", zid=zid, filters={"zid": (str(zid),)}).data
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=86400)
def _load_gldetail(zid: str) -> pd.DataFrame:
    project = st.session_state.proj
    filters = {"zid": (str(zid),)}
    df = Analytics("gldetail_simple", zid=zid, project=project, filters=filters).data
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=86400)
def _load_glheader(zid: str) -> pd.DataFrame:
    df = Analytics("glheader_simple", zid=zid, filters={"zid": (str(zid),)}).data
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=86400)
def _load_glmst(zid: str) -> pd.DataFrame:
    df = Analytics("glmst_simple", zid=zid, filters={"zid": (str(zid),)}).data
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=86400)
def _load_casup(zid: str) -> pd.DataFrame:
    df = Analytics("casup_simple", zid=zid, filters={"zid": (str(zid),)}).data
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _compute_ar_balances(zid: str, year: int, month: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cacus = _load_cacus(zid)
    gld   = _load_gldetail(zid)
    glh   = _load_glheader(zid)
    glm   = _load_glmst(zid)

    if any(df.empty for df in (cacus, gld, glh, glm)):
        return pd.DataFrame(), pd.DataFrame()

    # Normalize types for safe merges/filters
    for df, col in ((gld, "voucher"), (glh, "voucher"),
                    (gld, "ac_code"), (glm, "ac_code"),
                    (gld, "ac_sub"), (cacus, "cusid")):
        df[col] = df[col].astype(str)

    # Enrich detail with date parts and account info
    lines = gld.merge(glh[["zid","voucher","date","year","month"]], on=["zid","voucher"], how="left")
    lines = lines.merge(glm[["zid","ac_code","ac_type","ac_name"]], on=["zid","ac_code"], how="left")

    # Keep AR-related lines (Asset + linked to a customer)
    lines = lines[lines["ac_code"] == "01030001"]

    # IGNORE all OB vouchers globally (closing, period, trail)
    vup = lines["voucher"].astype(str).str.upper()
    lines = lines[~vup.str.startswith("OB--", na=False)].copy()

    # Attach customer names (inner join = keep only valid customers in lines)
    lines = lines.merge(cacus[["zid","cusid","cusname"]],
                        left_on=["zid","ac_sub"], right_on=["zid","cusid"], how="inner")

    # Build masks and helper columns once (faster than lambda per-group)
    yr = int(year); mo = int(month)
    y  = lines["year"].astype(int)
    m  = lines["month"].astype(int)
    val = lines["value"].astype(float)

    asof_cond   = (y < yr) | ((y == yr) & (m <= mo))
    period_cond = (y == yr) & (m == mo)

    lines["val_asof"]       = np.where(asof_cond,   val, 0.0)
    lines["val_period"]     = np.where(period_cond, val, 0.0)
    lines["tx_in_period"]   = np.where(period_cond, 1,   0)

    # Aggregate by customer
    agg = (lines.groupby(["cusid","cusname"], as_index=False)
                 .agg(closing_balance=("val_asof","sum"),
                      month_movement=("val_period","sum"),
                      tx_count_in_month=("tx_in_period","sum")))
    agg["had_tx_in_month"] = agg["tx_count_in_month"] > 0
    agg = agg.drop(columns=["tx_count_in_month"])

    # Include all customers (even if no lines after OB removal)
    all_customers = cacus[["cusid","cusname"]].drop_duplicates()
    out = (all_customers.merge(agg, on=["cusid","cusname"], how="left")
                      .fillna({"closing_balance":0.0, "month_movement":0.0, "had_tx_in_month":False}))

    # Sort by magnitude of closing balance
    out = out.sort_values("closing_balance", key=lambda s: s.abs(), ascending=False)

    # Trail up to as-of (also without OB lines)
    trail_asof = lines[asof_cond].copy().sort_values(["date","voucher","ac_code"])

    return out, trail_asof

def _compute_ap_balances(zid: str, year: int, month: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    casup = _load_casup(zid)
    gld   = _load_gldetail(zid)    # existing helper
    glh   = _load_glheader(zid)    # existing helper
    glm   = _load_glmst(zid)       # existing helper

    if any(df.empty for df in (casup, gld, glh, glm)):
        return pd.DataFrame(), pd.DataFrame()

    # Normalize types for safe merges/filters
    for df, col in ((gld, "voucher"), (glh, "voucher"),
                    (gld, "ac_code"), (glm, "ac_code"),
                    (gld, "ac_sub"), (casup, "supid")):
        df[col] = df[col].astype(str)

    # Enrich detail
    lines = gld.merge(glh[["zid","voucher","date","year","month"]], on=["zid","voucher"], how="left")
    lines = lines.merge(glm[["zid","ac_code","ac_type","ac_name"]], on=["zid","ac_code"], how="left")

    # Keep AP-related lines: Liability accounts + supplier-linked subledger
    lines = lines[lines["ac_code"].isin(["09030001", "09030004"])]

    # Ignore all OB vouchers
    vup = lines["voucher"].str.upper()
    lines = lines[~vup.str.startswith("OB--", na=False)].copy()

    # Attach supplier names (inner join = only valid suppliers)
    lines = lines.merge(casup[["zid","supid","supname"]],
                        left_on=["zid","ac_sub"], right_on=["zid","supid"], how="inner")

    yr, mo = int(year), int(month)
    y = lines["year"].astype(int)
    m = lines["month"].astype(int)
    val = lines["value"].astype(float)

    asof_cond   = (y < yr) | ((y == yr) & (m <= mo))
    period_cond = (y == yr) & (m == mo)

    lines["val_asof"]     = np.where(asof_cond,   val, 0.0)
    lines["val_period"]   = np.where(period_cond, val, 0.0)
    lines["tx_in_period"] = np.where(period_cond, 1,   0)

    # Aggregate by supplier
    agg = (lines.groupby(["supid","supname"], as_index=False)
                 .agg(closing_balance=("val_asof","sum"),
                      month_movement=("val_period","sum"),
                      tx_count_in_month=("tx_in_period","sum")))
    agg["had_tx_in_month"] = agg["tx_count_in_month"] > 0
    agg = agg.drop(columns=["tx_count_in_month"])

    # Include all suppliers (even with no lines after OB removal)
    all_sup = casup[["supid","supname"]].drop_duplicates()
    out = (all_sup.merge(agg, on=["supid","supname"], how="left")
                 .fillna({"closing_balance":0.0, "month_movement":0.0, "had_tx_in_month":False}))

    # Sort by magnitude of closing balance
    out = out.sort_values("closing_balance", key=lambda s: s.abs(), ascending=False)

    # Trail up to as-of (OB ignored)
    trail_asof = lines[asof_cond].copy().sort_values(["date","voucher","ac_code"])

    return out, trail_asof

@st.cache_data(show_spinner=False, ttl=86400)
def _ledger_accounts_by_type(zid: str, ac_type: str) -> pd.DataFrame:
    glm = _load_glmst(zid)
    if glm.empty:
        return pd.DataFrame()
    df = glm[glm["ac_type"] == ac_type].copy()
    df["label"] = df["ac_code"].astype(str) + " — " + df["ac_name"].astype(str)
    return df.sort_values("ac_code")

@st.cache_data(show_spinner=False, ttl=86400)
def _compute_ledger(zid: str, ac_type: str, ac_codes: list[str], year: int, month: int,
                    mode: str, ignore_ob: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (voucher_summary, line_table)
    - voucher_summary columns: date, voucher, amount, ob_amount, lines
    - line_table columns: date, voucher, ac_code, ac_name, ac_lv1, ac_lv2, amount, ob_amount, is_ob
    """
    gld = _load_gldetail(zid)
    glh = _load_glheader(zid)
    glm = _load_glmst(zid)

    if any(df.empty for df in (gld, glh, glm)):
        return pd.DataFrame(), pd.DataFrame()

    # normalize dtypes
    for df, col in ((gld,"voucher"), (glh,"voucher"), (gld,"ac_code"), (glm,"ac_code")):
        df[col] = df[col].astype(str)

    # enrich lines with date parts and account meta
    lines = gld.merge(glh[["zid","voucher","date","year","month"]],
                      on=["zid","voucher"], how="left")
    lines = lines.merge(glm[["zid","ac_code","ac_name","ac_type","ac_lv1","ac_lv2"]],
                        on=["zid","ac_code"], how="left")

    # filter by type and selected accounts
    lines = lines[lines["ac_type"] == ac_type]
    if ac_codes:
        lines = lines[lines["ac_code"].isin([str(c) for c in ac_codes])]

    if lines.empty:
        return pd.DataFrame(), pd.DataFrame()

    # compute period masks
    yr, mo = int(year), int(month)
    y  = lines["year"].astype(int)
    m  = lines["month"].astype(int)
    v  = lines["value"].astype(float)

    if mode == "Income (this month)":
        mask = (y == yr) & (m == mo)
    else:  # "Balance (as-of)"
        mask = (y < yr) | ((y == yr) & (m <= mo))

    # OB identification (case-insensitive)
    is_ob = lines["voucher"].astype(str).str.upper().str.startswith("OB--", na=False)
    lines = lines[mask].copy()
    lines["is_ob"] = is_ob[mask].values
    lines["ob_amount"] = np.where(lines["is_ob"], v[mask].values, 0.0)

    # amount column respects ignore_ob toggle; OB always shown in ob_amount
    if ignore_ob:
        lines["amount"] = np.where(lines["is_ob"], 0.0, v[mask].values)
    else:
        lines["amount"] = v[mask].values

    # build outputs (narration intentionally omitted)
    line_table = (lines[["date","voucher","ac_code","ac_name","ac_lv1","ac_lv2","amount","ob_amount","is_ob"]]
                        .sort_values(["date","voucher","ac_code"])
                        .reset_index(drop=True))

    return line_table

@timed
def display_accounting_analysis_main(current_page, zid: str):
    st.title("Accounting Analysis")

    tab_ar, tab_ap, tab_ledger = st.tabs([
        "🧾 AR Analysis",
        "📄 AP Analysis",
        "📘 Ledger Entries"
    ])

    # ───────────────────────────── AR Analysis ────────────────────────────────
    with tab_ar:
        c1, c2, _ = st.columns([1,1,2])
        with c1:
            year = st.number_input("Year", min_value=2018, max_value=2035, value=pd.Timestamp.today().year, step=1)
        with c2:
            month = st.number_input("Month", min_value=1, max_value=12, value=pd.Timestamp.today().month, step=1)

        if st.button("Load AR", type="primary"):
            st.session_state["_ar_loaded"] = True

        if st.session_state.get("_ar_loaded"):
            summary, trail_asof = _compute_ar_balances(zid, int(year), int(month))
            if summary.empty:
                st.info("No AR balances found for the selected period.")
            else:
                st.caption("AR balances as of selected month-end (OB included automatically)")
                total_ar = float(pd.to_numeric(summary["closing_balance"], errors="coerce").sum())
                st.caption(
                    f"AR balances as of selected month-end — OB vouchers (OB--) are ignored in all calculations and trails. "
                    f"Total closing: **{total_ar:,.2f}**"
                )
                st.dataframe(summary, use_container_width=True, height=440)
                st.write(common.create_download_link(summary,"ar_balances.xlsx"), unsafe_allow_html=True)

                # Drill-down for a single customer (up to month-end)
                st.markdown("### Customer Trail (up to selected month)")
                pick_id = st.selectbox(
                    "Choose a Customer ID:",
                    options=["—"] + summary["cusid"].astype(str).tolist(),
                    index=0
                )
                if pick_id and pick_id != "—":
                    cust_trail = trail_asof[trail_asof["cusid"].astype(str) == str(pick_id)]
                    if cust_trail.empty:
                        st.info("No GL lines up to the selected month for this customer.")
                    else:
                        st.dataframe(
                            cust_trail[["date","voucher","ac_code","ac_name","value"]]
                            .reset_index(drop=True),
                            use_container_width=True, height=420
                        )
                        st.write(common.create_download_link(cust_trail,"ar_trail.xlsx"), unsafe_allow_html=True)

    # ───────────────────────────── AP Analysis (placeholder) ──────────────────
    with tab_ap:
        c1, c2, _ = st.columns([1,1,2])
        with c1:
            ap_year = st.number_input("Year", min_value=2018, max_value=2035,
                                    value=pd.Timestamp.today().year, step=1, key="ap_year")
        with c2:
            ap_month = st.number_input("Month", min_value=1, max_value=12,
                                    value=pd.Timestamp.today().month, step=1, key="ap_month")

        if st.button("Load AP", type="primary"):
            st.session_state["_ap_loaded"] = True

        if st.session_state.get("_ap_loaded"):
            ap_summary, ap_trail_asof = _compute_ap_balances(zid, int(ap_year), int(ap_month))
            if ap_summary.empty:
                st.info("No AP balances found for the selected period.")
            else:
                st.caption("AP balances as of selected month-end — OB vouchers (OB--) are ignored in all calculations and trails.")
                total_ap = float(pd.to_numeric(ap_summary["closing_balance"], errors="coerce").sum())
                st.caption(
                    f"AP balances as of selected month-end — OB vouchers (OB--) are ignored in all calculations and trails. "
                    f"Total closing: **{total_ap:,.2f}**"
                )
                st.dataframe(ap_summary, use_container_width=True, height=440)
                st.write(common.create_download_link(ap_summary,"ap_balances.xlsx"), unsafe_allow_html=True)

                st.markdown("### Supplier Trail (up to selected month)")
                pick_sup = st.selectbox(
                    "Choose a Supplier ID:",
                    options=["—"] + ap_summary["supid"].astype(str).tolist(),
                    index=0
                )
                if pick_sup and pick_sup != "—":
                    sup_trail = ap_trail_asof[ap_trail_asof["supid"].astype(str) == str(pick_sup)]
                    if sup_trail.empty:
                        st.info("No GL lines up to the selected month for this supplier.")
                    else:
                        st.dataframe(
                            sup_trail[["date","voucher","ac_code","ac_name","value"]]
                            .reset_index(drop=True),
                            use_container_width=True, height=420
                        )
                        st.write(common.create_download_link(sup_trail,"ap_trail.xlsx"), unsafe_allow_html=True)

    # ───────────────────────── Ledger Entries (placeholder) ───────────────────
    with tab_ledger:
        colL, colR = st.columns([2,3], gap="large")
        with colL:
            st.subheader("Filters")

            mode = st.radio(
                "Computation",
                options=["Income (this month)", "Balance (as-of)"],
                index=0,
                help="Income: only the picked month. Balance: up to & including the picked month."
            )

            ac_choices = ["Income", "Expenditure"] if mode == "Income (this month)" else ["Asset", "Liability"]

            if st.session_state.get("ledger_prev_mode") != mode:
                st.session_state["ledger_prev_mode"] = mode
                st.session_state.pop("ledger_ac_type", None)
                st.session_state.pop("ledger_accounts", None)  # if you keyed your multiselect

            c1, c2 = st.columns(2)
            with c1:
                year = st.number_input("Year", min_value=2018, max_value=2035,
                                        value=pd.Timestamp.today().year, step=1, key="led_year")
            with c2:
                month = st.number_input("Month", min_value=1, max_value=12,
                                        value=pd.Timestamp.today().month, step=1, key="led_month")

            ignore_ob = st.toggle("Ignore OB (OB--)", value=True,
                                    help="If ON, OB amounts are excluded from totals. OB values still appear in a separate column.")

            # Account type → account picker
            ac_type = st.selectbox("Account Type", ac_choices, index=0, key="ledger_ac_type")
            acc_df  = _ledger_accounts_by_type(zid, ac_type)
            acc_labels = acc_df["label"].tolist()
            label2code = dict(zip(acc_df["label"], acc_df["ac_code"]))

            picked_labels = st.multiselect("Accounts (ac_code — ac_name)", options=acc_labels, placeholder="Pick one or more (empty = all in type)")
            ac_codes = [label2code[l] for l in picked_labels]

            go = st.button("Load ledger", type="primary")

        with colR:
            if go:
                lines = _compute_ledger(zid, ac_type, ac_codes, int(year), int(month), mode, bool(ignore_ob))
                if lines.empty:
                    st.info("No postings for the selected filters.")
                else:
                    st.caption("Line entries (no narration)")
                    st.dataframe(lines, use_container_width=True, height=420)
                    st.write(common.create_download_link(lines,"ledger_lines.xlsx"), unsafe_allow_html=True)
