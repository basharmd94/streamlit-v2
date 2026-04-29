import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

from processing import common
from processing import daily_sales as ds
from utils.utils import timed


def _parse_id_name_selections(selections: list) -> list:
    """Extract ID codes from 'id - name' format strings."""
    codes = []
    for sel in selections:
        parts = sel.split(" - ", 1)
        codes.append(parts[0].strip())
    return codes


def _make_download_button(df: pd.DataFrame, label: str, filename: str, key: str):
    """Render a st.download_button for a DataFrame as Excel."""
    buf = BytesIO()
    df.reset_index().to_excel(buf, index=False)
    buf.seek(0)
    st.download_button(
        label=label,
        data=buf,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
    )


def _cap_dataframe(df: pd.DataFrame, cap: int = 50_000, key: str = "") -> pd.DataFrame:
    """Cap at cap rows; show info banner if truncated."""
    if len(df) > cap:
        st.info(f"⚠️ Displaying first {cap:,} of {len(df):,} rows.")
        return df.iloc[:cap]
    return df


@timed
def display_daily_sales_page(current_page, zid, data_dict):
    st.title("Daily Sales Analysis")

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    sales_raw      = data_dict.get("sales",      pd.DataFrame())
    returns_raw    = data_dict.get("return",      pd.DataFrame())
    collection_raw = data_dict.get("collection", pd.DataFrame())

    if sales_raw.empty and returns_raw.empty:
        st.warning("No sales data loaded. Please load data from the sidebar.")
        return

    # Apply common column transformations
    sales_df, returns_df = common.data_copy_add_columns(sales_raw, returns_raw)
    if not collection_raw.empty:
        (collection_df,) = common.data_copy_add_columns(collection_raw)
    else:
        collection_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Date window (end date from session state, start computed from it)
    # ------------------------------------------------------------------
    end_date   = st.session_state.get("daily_sales_end_date", ds.default_end_date())
    start_date = ds.default_start_date(end_date)

    st.caption(f"Analysis window: **{start_date}** to **{end_date}** (3 months)")

    # ------------------------------------------------------------------
    # Parse sidebar filter selections stored by app.py navigation
    # ------------------------------------------------------------------
    sp_selections    = st.session_state.get("daily_sales_sp",    [])
    item_selections  = st.session_state.get("daily_sales_item",  [])
    group_selections = st.session_state.get("daily_sales_group", [])

    sp_codes    = _parse_id_name_selections(sp_selections)
    item_codes  = _parse_id_name_selections(item_selections)
    item_groups = list(group_selections)

    # ------------------------------------------------------------------
    # Analysis mode selector
    # ------------------------------------------------------------------
    analysis_mode = st.radio(
        "Choose Analysis Mode:",
        ["Overview", "Comparison"],
        horizontal=True,
        key="ds_analysis_mode",
    )

    # ================================================================
    # OVERVIEW
    # ================================================================
    if analysis_mode == "Overview":

        # Section 1 — Daily bar chart
        st.subheader("📈 Select Plot Type")
        try:
            selected_plot = st.selectbox(
                "Choose metric to plot:",
                ["Net Sales", "Net Returns", "Net Collections"],
                key="ds_plot_select",
            )
            daily_df = ds.compute_daily_metric(
                sales_df, returns_df, collection_df,
                selected_plot, start_date, end_date,
                sp_codes, item_codes, item_groups,
            )
            if daily_df.empty:
                st.info("No data available for the selected metric and date window.")
            else:
                fig = px.bar(
                    daily_df,
                    x="date_label",
                    y="value",
                    title=f"{selected_plot} — Daily ({start_date} to {end_date})",
                    labels={"date_label": "Date", "value": selected_plot},
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("⚠️ Unable to generate report. Please reselect your options — the current combination may lack sufficient data for valid reporting.")

        # Section 2 — Period totals
        st.subheader("📊 Period Totals (3-Month Window)")
        try:
            totals = ds.compute_period_totals(
                sales_df, returns_df, collection_df,
                start_date, end_date, sp_codes, item_codes, item_groups,
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Net Sales",        f"{totals['net_sales']:,.0f}")
            with col2:
                st.metric("Net Returns",      f"{totals['net_returns']:,.0f}")
            with col3:
                st.metric("Net Collections",  f"{totals['net_collections']:,.0f}")
        except Exception:
            st.warning("⚠️ Unable to generate report. Please reselect your options — the current combination may lack sufficient data for valid reporting.")

        # Section 3 — Daily Pivot Table
        st.subheader("📋 Daily Pivot Table")
        try:
            col1, col2 = st.columns(2)
            with col1:
                pivot_entity = st.selectbox(
                    "Entity",
                    ["Salesman", "Product", "Product Group"],
                    key="ds_pivot_entity",
                )
            with col2:
                pivot_metric = st.selectbox(
                    "Metric",
                    ["Net Sales", "Net Returns", "Net Collections"],
                    key="ds_pivot_metric",
                )

            pivot_df = ds.compute_daily_pivot(
                sales_df, returns_df, collection_df,
                pivot_entity, pivot_metric, start_date, end_date,
                sp_codes=sp_codes, item_codes=item_codes, item_groups=item_groups,
            )

            # Show "breakdown unavailable" notice only when entity breakdown truly isn't possible
            if pivot_metric == "Net Collections":
                collection_has_sp = (
                    not collection_df.empty and 'spname' in collection_df.columns
                )
                if pivot_entity != "Salesman" or not collection_has_sp:
                    st.info(
                        "ℹ️ Entity breakdown is not available for Collections with the selected "
                        "entity — showing daily totals (column 'All')."
                    )

            if pivot_df.empty:
                st.info("No data available for the selected combination.")
            else:
                pivot_capped = _cap_dataframe(pivot_df, key="pivot_cap")
                st.dataframe(pivot_capped, use_container_width=True)
                _make_download_button(pivot_df, "⬇ Download Pivot", "daily_pivot.xlsx", "dl_daily_pivot")
        except Exception:
            st.warning("⚠️ Unable to generate report. Please reselect your options — the current combination may lack sufficient data for valid reporting.")

        # Section 4 — Moving Average Benchmark Table
        st.subheader("📐 Moving Average Benchmark Table")
        try:
            col1, col2 = st.columns(2)
            with col1:
                ma_entity = st.selectbox(
                    "Entity",
                    ["Salesman", "Product", "Product Group"],
                    key="ds_ma_entity",
                )
            with col2:
                # Net Collections is only meaningful for Salesman entity (requires spname)
                ma_metric_options = ["Net Sales", "Net Returns"]
                if ma_entity == "Salesman":
                    ma_metric_options.append("Net Collections")
                ma_metric = st.selectbox(
                    "Metric",
                    ma_metric_options,
                    key="ds_ma_metric",
                )

            ma_df = ds.compute_moving_avg_table(
                sales_df, returns_df, ma_entity, ma_metric, end_date, collection_df,
                sp_codes=sp_codes, item_codes=item_codes, item_groups=item_groups,
            )

            if ma_df.empty:
                st.info("No data available for the selected combination.")
            else:
                st.dataframe(ma_df, use_container_width=True)
                _make_download_button(ma_df, "⬇ Download MA Table", "daily_ma.xlsx", "dl_daily_ma")
        except Exception:
            st.warning("⚠️ Unable to generate report. Please reselect your options — the current combination may lack sufficient data for valid reporting.")

    # ================================================================
    # COMPARISON
    # ================================================================
    elif analysis_mode == "Comparison":

        dimension_column_map = {
            "Salesman":      ("spid",      "spname"),
            "Product":       ("itemcode",  "itemname"),
            "Product Group": ("itemgroup", None),
        }

        try:
            compare_type = st.selectbox(
                "Compare Across",
                ["Year-over-Year (YOY)", "Month vs Month"],
                key="ds_compare_type",
            )

            # ---- Year-over-Year ----
            if compare_type == "Year-over-Year (YOY)":
                col1, col2 = st.columns(2)
                with col1:
                    compare_by = st.selectbox(
                        "Compare By",
                        ["Salesman", "Product", "Product Group"],
                        key="ds_yoy_compare_by",
                    )
                with col2:
                    yoy_metric = st.selectbox(
                        "Metric",
                        ["Net Sales", "Net Returns", "Net Collections"],
                        key="ds_yoy_metric",
                    )

                code_col, name_col = dimension_column_map[compare_by]

                # Build entity option list
                if (name_col and name_col in sales_df.columns and code_col in sales_df.columns):
                    sub = sales_df[[code_col, name_col]].dropna().drop_duplicates()
                    display_options = sorted(
                        (sub[code_col].astype(str) + " - " + sub[name_col].astype(str)).tolist()
                    )
                elif code_col in sales_df.columns:
                    display_options = sorted(sales_df[code_col].dropna().unique().astype(str).tolist())
                else:
                    display_options = []

                # Pre-populate from sidebar filter selections
                if compare_by == "Salesman":
                    yoy_sidebar_defaults = [o for o in display_options if o.split(" - ")[0] in sp_codes]
                elif compare_by == "Product":
                    yoy_sidebar_defaults = [o for o in display_options if o.split(" - ")[0] in item_codes]
                elif compare_by == "Product Group":
                    yoy_sidebar_defaults = [o for o in display_options if o in item_groups]
                else:
                    yoy_sidebar_defaults = []

                selected_display = st.multiselect(
                    f"Select {compare_by} (leave empty to aggregate all)",
                    options=display_options,
                    default=yoy_sidebar_defaults,
                    key="ds_yoy_entity",
                )
                selected_codes = [s.split(" - ")[0] for s in selected_display]

                yoy_df = ds.compute_yoy_daily(
                    sales_df, returns_df, collection_df,
                    yoy_metric, end_date, code_col, selected_codes,
                )

                if yoy_df.empty:
                    st.info("No data available for the selected combination.")
                else:
                    fig = px.bar(
                        yoy_df,
                        x="date_label",
                        y="value",
                        color="period",
                        barmode="group",
                        title=f"{yoy_metric} — Year-over-Year Daily Comparison",
                        labels={"date_label": "MM-DD", "value": yoy_metric, "period": "Period"},
                        category_orders={"date_label": sorted(yoy_df["date_label"].unique().tolist())},
                    )
                    # Force categorical axis so Plotly doesn't reinterpret MM-DD strings as dates
                    fig.update_xaxes(type="category", tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 📋 Corresponding Data")
                    yoy_pivot = (
                        yoy_df.pivot_table(
                            index="date_label", columns="period", values="value", aggfunc="sum"
                        )
                        .fillna(0)
                        .round(2)
                    )
                    st.dataframe(_cap_dataframe(yoy_pivot, key="yoy_cap"), use_container_width=True)
                    _make_download_button(yoy_pivot, "⬇ Download YOY Data", "yoy_daily.xlsx", "dl_yoy_daily")

            # ---- Month vs Month ----
            elif compare_type == "Month vs Month":
                # Build month labels from sales in the 3-month window
                s_window = ds.apply_date_window(sales_df, start_date, end_date)
                if not s_window.empty and 'date' in s_window.columns:
                    s_tmp = s_window.copy()
                    s_tmp['_dt'] = pd.to_datetime(s_tmp['date'], errors='coerce')
                    s_tmp['month_label'] = (
                        s_tmp['_dt'].dt.strftime('%m') + '-' + s_tmp['_dt'].dt.strftime('%Y')
                    )
                    month_options = sorted(s_tmp['month_label'].dropna().unique().tolist())
                else:
                    month_options = []

                selected_months = st.multiselect(
                    "Select Months to Compare",
                    options=month_options,
                    default=month_options[:3] if month_options else [],
                    key="ds_mom_months",
                )

                col1, col2 = st.columns(2)
                with col1:
                    mom_compare_by = st.selectbox(
                        "Compare By",
                        ["Salesman", "Product", "Product Group"],
                        key="ds_mom_compare_by",
                    )
                with col2:
                    mom_metric = st.selectbox(
                        "Metric",
                        ["Net Sales", "Net Returns", "Net Collections"],
                        key="ds_mom_metric",
                    )

                mom_code_col, mom_name_col = dimension_column_map[mom_compare_by]

                if (mom_name_col and mom_name_col in sales_df.columns and mom_code_col in sales_df.columns):
                    sub = sales_df[[mom_code_col, mom_name_col]].dropna().drop_duplicates()
                    mom_display_options = sorted(
                        (sub[mom_code_col].astype(str) + " - " + sub[mom_name_col].astype(str)).tolist()
                    )
                elif mom_code_col in sales_df.columns:
                    mom_display_options = sorted(
                        sales_df[mom_code_col].dropna().unique().astype(str).tolist()
                    )
                else:
                    mom_display_options = []

                # Pre-populate from sidebar filter selections
                if mom_compare_by == "Salesman":
                    mom_sidebar_defaults = [o for o in mom_display_options if o.split(" - ")[0] in sp_codes]
                elif mom_compare_by == "Product":
                    mom_sidebar_defaults = [o for o in mom_display_options if o.split(" - ")[0] in item_codes]
                elif mom_compare_by == "Product Group":
                    mom_sidebar_defaults = [o for o in mom_display_options if o in item_groups]
                else:
                    mom_sidebar_defaults = []

                mom_selected_display = st.multiselect(
                    f"Select {mom_compare_by} (leave empty to aggregate all)",
                    options=mom_display_options,
                    default=mom_sidebar_defaults,
                    key="ds_mom_entity",
                )
                mom_selected_codes = [s.split(" - ")[0] for s in mom_selected_display]

                if selected_months:
                    mom_df = ds.compute_mom_daily(
                        sales_df, returns_df, collection_df,
                        mom_metric, selected_months, mom_code_col, mom_selected_codes,
                    )

                    if mom_df.empty:
                        st.info("No data available for the selected combination.")
                    else:
                        fig = px.bar(
                            mom_df,
                            x="day",
                            y="value",
                            color="month_label",
                            barmode="group",
                            title=f"{mom_metric} — Month vs Month Daily Comparison",
                            labels={
                                "day": "Day of Month",
                                "value": mom_metric,
                                "month_label": "Month",
                            },
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("### 📋 Corresponding Data")
                        mom_pivot = (
                            mom_df.pivot_table(
                                index="day", columns="month_label", values="value", aggfunc="sum"
                            )
                            .fillna(0)
                            .round(2)
                        )
                        st.dataframe(_cap_dataframe(mom_pivot, key="mom_cap"), use_container_width=True)
                        _make_download_button(
                            mom_pivot, "⬇ Download MoM Data", "mom_daily.xlsx", "dl_mom_daily"
                        )
                else:
                    st.info("Select at least one month to compare.")

        except Exception:
            st.warning("⚠️ Unable to generate report. Please reselect your options — the current combination may lack sufficient data for valid reporting.")

