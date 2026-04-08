import calendar
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from processing import ar_analysis as ar_proc
from utils.utils import timed


def _format_month_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename YYYY_MM columns to 'Mon-YYYY' labels."""
    rename_map = {}
    for col in df.columns:
        if (
            isinstance(col, str)
            and len(col) == 7
            and col[:4].isdigit()
            and col[4] == "_"
            and col[5:].isdigit()
        ):
            year, month = int(col[:4]), int(col[5:])
            rename_map[col] = f"{calendar.month_abbr[month]}-{year}"
    return df.rename(columns=rename_map)


@timed
def display_ar_analysis_page(current_page, zid, data_dict):
    """Main entry point for the AR Analysis page."""

    st.title("AR Analysis")

    # ── Session-state reads ──────────────────────────────────────────────────
    till_date = st.session_state.get("ar_till_date", date.today() - timedelta(days=2))
    selected_sp_list = st.session_state.get("ar_salesmen", [])

    # ── Raw data ─────────────────────────────────────────────────────────────
    raw_df = data_dict.get("ar_raw", pd.DataFrame())

    if raw_df is None or (isinstance(raw_df, pd.DataFrame) and raw_df.empty):
        st.info(
            "No AR data loaded. Use the sidebar to select filters and click **Load Data**."
        )
        return

    # Parse salesman codes from "CODE - Name" strings
    if selected_sp_list:
        sp_codes = [s.split(" - ")[0].strip() for s in selected_sp_list]
    else:
        sp_codes = []

    # ── Prepare cleaned DataFrame ────────────────────────────────────────────
    with st.spinner("Preparing customer ledger…"):
        df_clean, missing_diag_df = ar_proc.prep_customer_df(raw_df)

    if df_clean is None or df_clean.empty:
        st.warning("No data remained after ledger preparation.")
        return

    # ── Report mode selector ─────────────────────────────────────────────────
    report_mode = st.radio(
        "Report Mode",
        [
            "Latest Sale & Collection",
            "FIFO Trickle-Down Balances",
            "Salesman Due Summary",
            "Missing Customers",
        ],
        horizontal=True,
    )

    st.markdown("---")

    # ────────────────────────────────────────────────────────────────────────
    # Report: Latest Sale & Collection
    # ────────────────────────────────────────────────────────────────────────
    if report_mode == "Latest Sale & Collection":
        with st.spinner("Building Latest Sale & Collection report…"):
            report_df = ar_proc.make_latest_sale_collection_report(df_clean)

        if report_df.empty:
            st.warning("No data to display.")
            return

        # Apply salesman filter
        if sp_codes:
            report_df = report_df[
                report_df["Salesman Code"].astype(str).isin(sp_codes)
            ]

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Customers", len(report_df))
        col2.metric(
            "Total Balance",
            f"{report_df['Current Balance'].sum():,.0f}",
        )
        col3.metric(
            "Unique Salesmen",
            report_df["Salesman Code"].nunique(),
        )

        st.dataframe(report_df, use_container_width=True, hide_index=True)

    # ────────────────────────────────────────────────────────────────────────
    # Report: FIFO Trickle-Down Balances
    # ────────────────────────────────────────────────────────────────────────
    elif report_mode == "FIFO Trickle-Down Balances":
        months_back = st.slider("Months to Show", 1, 12, 4)

        with st.spinner("Building FIFO Trickle-Down Balances…"):
            report_cc = ar_proc.build_trickledown_balances(
                df_clean,
                till_date=till_date,
                months_back=months_back,
            )

        if report_cc is None or report_cc.empty:
            st.warning("No data to display.")
            return

        # Apply salesman filter (keep all rows when no filter selected)
        if sp_codes:
            report_cc = report_cc[
                report_cc["xsp"].astype(str).isin(sp_codes)
            ]

        # Format month columns for display
        display_cc = _format_month_cols(report_cc)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Customers", len(display_cc))
        if "total_due" in display_cc.columns:
            col2.metric("Total Due", f"{display_cc['total_due'].sum():,.0f}")
        col3.metric("Unique Salesmen", display_cc["xsp"].nunique() if "xsp" in display_cc.columns else 0)

        st.dataframe(display_cc, use_container_width=True, hide_index=True)

    # ────────────────────────────────────────────────────────────────────────
    # Report: Salesman Due Summary
    # ────────────────────────────────────────────────────────────────────────
    elif report_mode == "Salesman Due Summary":
        months_back = st.slider("Months to Show", 1, 12, 4)

        with st.spinner("Building Salesman Due Summary…"):
            report_cc = ar_proc.build_trickledown_balances(
                df_clean,
                till_date=till_date,
                months_back=months_back,
            )

        if report_cc is None or report_cc.empty:
            st.warning("No data to display.")
            return

        summary = ar_proc.build_main_due_report(report_cc)

        if summary is None or summary.empty:
            st.warning("No data to display.")
            return

        # Apply salesman filter – keep the "Total" row always
        if sp_codes:
            summary = summary[
                summary["xsp"].astype(str).isin(sp_codes)
                | (summary["xsp"].astype(str) == "Total")
            ]

        # Format month columns for display
        display_summary = _format_month_cols(summary)

        # Summary metrics
        total_row = display_summary[display_summary["xsp"].astype(str) == "Total"]
        col1, col2 = st.columns(2)
        col1.metric(
            "Salesman Count",
            len(display_summary) - (1 if not total_row.empty else 0),
        )
        if "total_due" in display_summary.columns and not total_row.empty:
            col2.metric(
                "Grand Total Due",
                f"{total_row['total_due'].values[0]:,.0f}",
            )

        st.dataframe(display_summary, use_container_width=True, hide_index=True)

    # ────────────────────────────────────────────────────────────────────────
    # Report: Missing Customers
    # ────────────────────────────────────────────────────────────────────────
    elif report_mode == "Missing Customers":
        with st.spinner("Computing FIFO balances to check missing customers…"):
            report_df = ar_proc.make_latest_sale_collection_report(df_clean)
            report_cc = ar_proc.build_trickledown_balances(df_clean, till_date=till_date)

        missing_df = ar_proc.find_missing_customers(report_df, report_cc)

        # Apply salesman filter
        if sp_codes and "Salesman Code" in missing_df.columns:
            missing_df = missing_df[
                missing_df["Salesman Code"].astype(str).isin(sp_codes)
            ]

        if missing_df.empty:
            st.success("No missing customers found.")
        else:
            st.metric("Missing Customers", len(missing_df))
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
