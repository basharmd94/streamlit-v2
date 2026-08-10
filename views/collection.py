import streamlit as st
import pandas as pd
import calendar
from datetime import datetime
from core.analytics import Analytics
from processing import common, collection, salesman_due
from utils.utils import timed
from views.call_log_shared import render_call_log_readonly as _render_call_log_readonly

# ── ZID-scope constants ────────────────────────────────────────────────────────
# Scope toggle is shown only when the active ZID is one of the two shared-team entities.
_DUAL_ZIDS  = frozenset({"100001", "100000"})
_OTHER_ZID  = {"100001": "100000", "100000": "100001"}
_SCOPE_OPTS = ["Default ZID", "100001 + 100000"]


@st.cache_data(ttl=3600, show_spinner="Loading combined ZID data...")
def _load_other_zid_collection(other_zid: str, project: str,
                                years: tuple, months: tuple) -> tuple:
    """Load sales/returns/collection/AR for the partner ZID (100001↔100000).

    Returns (sales_df, return_df, collection_df, ar_df) — all may be empty DataFrames.
    Cached per (other_zid, project, years, months) so a single DB round-trip is
    shared across CP, Customer Ledger, and Order Analytics in the same session.
    """
    f: dict = {}
    if years:
        f["year"] = list(years)
    if months:
        f["month"] = list(months)
    s  = Analytics("sales",          zid=other_zid, filters=f).data
    r  = Analytics("return",         zid=other_zid, filters=f).data
    c  = Analytics("collection",     zid=other_zid, filters=f).data
    ar = Analytics("ar_due_ledger",  zid=other_zid, project=project, filters={}).data
    return (
        s  if s  is not None else pd.DataFrame(),
        r  if r  is not None else pd.DataFrame(),
        c  if c  is not None else pd.DataFrame(),
        ar if ar is not None else pd.DataFrame(),
    )


def _get_combined_collection(zid, project, filtered_data_s, filtered_data_r, filtered_data_c):
    """Concat current + partner ZID's sales/returns/collection; tag with 'zid' column.

    Returns (s_combined, r_combined, c_combined, ar_other, other_zid_str).
    Does NOT mutate the incoming DataFrames.
    """
    other = _OTHER_ZID[str(zid)]
    _yrs  = tuple(sorted(filtered_data_s["year"].dropna().unique().astype(int).tolist()))  \
            if not filtered_data_s.empty else ()
    _mos  = tuple(sorted(filtered_data_s["month"].dropna().unique().astype(int).tolist())) \
            if not filtered_data_s.empty else ()

    s2, r2, c2, ar2 = _load_other_zid_collection(other, project, _yrs, _mos)

    # Enrich partner data through the same pipeline as the primary data
    if not (c2.empty and s2.empty):
        c2_e, s2_e, r2_e = common.data_copy_add_columns(c2, s2, r2)
        c2_e = common.enrich_collection_with_sales_info(c2_e, s2_e)
    else:
        s2_e, r2_e, c2_e = s2, r2, c2

    # Tag — copy to avoid mutating top-level vars used by other sections
    def _tag(df, tag):
        d = df.copy(); d["zid"] = tag; return d

    s_comb = pd.concat([_tag(filtered_data_s, str(zid)), _tag(s2_e, other)], ignore_index=True)
    r_comb = pd.concat([_tag(filtered_data_r, str(zid)), _tag(r2_e, other)], ignore_index=True)
    c_comb = pd.concat([_tag(filtered_data_c, str(zid)), _tag(c2_e, other)], ignore_index=True)

    return s_comb, r_comb, c_comb, ar2, other


def _render_cust_ledger_body(sales_df, returns_df, collection_df, ar_df, key_suffix: str = ""):
    """Render both ledger sub-sections (Sales/Returns/Collections + AR) for one ZID."""
    # ── Sales / Returns / Collections ledger ─────────────────────────────────
    try:
        _g = ["date", "year", "month", "cusid", "cusname", "DOM", "DOW"]
        s_g  = sales_df.groupby(_g).final_sales.sum().reset_index()
        r_g  = returns_df.groupby(_g).treturnamt.sum().reset_index()
        c_g  = collection_df.groupby(_g).value.sum().reset_index()

        _, _, _, comb = collection.average_days_to_collection(s_g, r_g, c_g)
        comb = comb[["year", "month", "cusid", "cusname", "date",
                     "final_sales", "treturnamt", "value"]]

        cus_opts = (
            c_g[["cusid", "cusname"]].drop_duplicates().sort_values("cusname")
        )
        cus_opts["label"] = cus_opts["cusid"].astype(str) + " - " + cus_opts["cusname"]

        if not cus_opts.empty:
            sel = st.selectbox(
                "Select Customer",
                options=cus_opts["label"].tolist(),
                key=f"cl_src_customer_select{key_suffix}",
            )
            sel_cusid = sel.split(" - ")[0]
            ledger = comb[comb["cusid"] == sel_cusid].sort_values("date")
            st.markdown("**Customer Ledger (Sales / Returns / Collections)**")
            st.write(ledger.drop(columns=["cusid", "cusname"]))
    except Exception as _e:
        st.warning("⚠️ Unable to load customer sales/collection ledger.")
        st.caption(f"Details: {_e}")

    st.divider()

    # ── AR Customer Ledger ────────────────────────────────────────────────────
    st.subheader("AR Customer Ledger")
    if ar_df is None or ar_df.empty:
        st.info("No AR data available.")
        return
    ar_opts = (
        ar_df[["cusid", "cusname"]].drop_duplicates().sort_values("cusname")
        .assign(option=lambda d: d["cusid"].astype(str) + " - " + d["cusname"])
    )
    ar_sel = st.selectbox(
        "Select customer", ar_opts["option"].tolist(),
        key=f"cl_ar_select{key_suffix}",
    )
    ar_cusid = ar_sel.split(" - ")[0]
    ar_ledger = ar_df[ar_df["cusid"] == ar_cusid].sort_values("date").copy()
    ar_ledger["ending_balance"] = ar_ledger["value"].cumsum()
    st.dataframe(ar_ledger)


@st.cache_data(ttl=3600, show_spinner="Building Salesman Due report...")
def _load_salesman_due_reports(zid: str, project: str) -> dict:
    """Load + build the Salesman Due reports for one business (zid/project)."""
    ar_df = Analytics("ar_due_ledger", zid=zid, project=project, filters={}).data
    if ar_df is None or ar_df.empty:
        return {}

    cacus_df = Analytics("cacus_master", zid=zid, filters={}).data
    if cacus_df is None:
        cacus_df = pd.DataFrame(columns=["cusid", "cusname", "xcity", "xstate"])

    prmst_df = Analytics("prmst_simple", zid=zid, filters={}).data
    if prmst_df is None:
        prmst_df = pd.DataFrame(columns=["spid", "spname"])

    market_split = str(zid) in ("100000", "100001")
    return salesman_due.build_salesman_due_reports(ar_df, cacus_df, prmst_df, market_split)


@timed
def display_collection_analysis_page(current_page, zid, project, data_dict):
    st.sidebar.title("Overall Collection Analysis")

    #collection using sales, returns and collection separately
    filtered_data_c, filtered_data_s,filtered_data_r = common.data_copy_add_columns(data_dict['collection'],data_dict['sales'], data_dict['return'])
    filtered_data_c = common.enrich_collection_with_sales_info(filtered_data_c, filtered_data_s)
    analysis_mode = st.radio("Choose Analysis Mode:",["Overview","Comparison","Distributions","Descriptive Stats","Metric Comparison","CP",
        # "CPA",   # temporarily muted — uncomment to restore
        "Customer Ledger","Salesman Due","📈 Order Analytics"],horizontal=True)

    #collection using glheader and details.
    filtered_data_ar = data_dict['ar']
    filtered_data_ar = filtered_data_ar[filtered_data_ar['project'] == project]
    filtered_data_ar = common.add_voucher_type_ar(filtered_data_ar)
    # st.write(common.create_download_link(filtered_data_ar,"filtered_data_ar.xlsx"), unsafe_allow_html=True)


    if analysis_mode == "Overview":
        st.subheader("📈 Select Plot Type")
        plot_options = [
            'Net Sales',
            'Net Returns',
            'Net Discounts',
            'Collections'
        ]

        selected_plot = st.selectbox("Choose a plot to display:", plot_options)

        # Display selected plot
        if selected_plot == 'Net Sales':
            collection.plot_net(filtered_data_s, filtered_data_r, ['year', 'month'], 'final_sales', 'treturnamt', 'Net Sales (filtered)', current_page)
        elif selected_plot == 'Net Returns':
            collection.plot_total_returns(filtered_data_r, current_page)
        elif selected_plot == 'Net Discounts':
            collection.plot_total_discounts(filtered_data_s, current_page)
        elif selected_plot == 'Collections':
            collection.plot_collection(filtered_data_c, current_page)

        # Summary stats
        summary_stats = collection.calculate_summary_statistics(filtered_data_c,filtered_data_s, filtered_data_r)
        collection.display_summary_statistics(summary_stats)

        # Expandable section for pivot tables
        collection.display_entity_metric_pivot(filtered_data_c, filtered_data_s, filtered_data_r, current_page)

    elif analysis_mode == "Comparison":
        all_years = sorted(filtered_data_c["year"].dropna().unique().astype(int).tolist())
        compare_type = st.selectbox("Compare Across", ["Year-over-Year (YOY)", "Month vs Month"])
        if compare_type == "Year-over-Year (YOY)":
            st.subheader("📅 Year-over-Year (YOY) Comparison")
            selected_years = st.multiselect("Select Years", all_years, default=all_years)
            granularity = st.selectbox("Group By", ["Monthly", "Daily", "Day of Week", "Day of Month"])
            if granularity == "Monthly":
                all_months = sorted(filtered_data_c["month"].dropna().unique().astype(int).tolist())
                month_map = {i: calendar.month_abbr[i] for i in all_months}
                selected_months = st.multiselect("Select Months", [month_map[m] for m in all_months], default=[month_map[m] for m in all_months])
            elif granularity == "Daily":
                latest_year = max(all_years) if selected_years else datetime.now().year
                default_start = datetime(latest_year, 1, 1)
                default_end = datetime(latest_year, 1, 30)
                start_date, end_date = st.date_input("Select Date Range", value=[default_start, default_end])
            elif granularity == "Day of Week":
                average_or_total_YOY_DOW = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
            elif granularity == "Day of Month":
                all_months = sorted(filtered_data_c["month"].dropna().unique().astype(int).tolist())
                month_map = {i: calendar.month_abbr[i] for i in all_months}
                selected_months = st.multiselect("Select Months", [month_map[m] for m in all_months], default=[month_map[m] for m in all_months])
                average_or_total_YOY_DOM = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
        elif compare_type == "Month vs Month":
            st.subheader("📅 Month vs Month Comparison")
            granularity = st.selectbox("Group By", ["Monthly", "Day of Week", "Day of Month"])
            filtered_data_c["month_label"] = (filtered_data_c["month"].astype(int).apply(lambda x: f"{x:02d}") + "-" + filtered_data_c["year"].astype(str))
            month_options = sorted(filtered_data_c["month_label"].dropna().unique().tolist())
            selected_months = st.multiselect("Select Months to Compare", options=month_options, default=month_options[:3])
            if granularity == 'Day of Week':
                average_or_total_MOM_DOW = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
            elif granularity == 'Day of Month':
                average_or_total_MOM_DOM = st.radio("Aggregation", ["Total", "Average"], horizontal=True)
                day_options = list(range(1, 32))
                selected_dom_days = st.multiselect("Select Days (leave empty to include all days)", options=day_options)

        compare_by = st.selectbox("Compare By", ["Salesman", "Customer", "Product", "Product Group", "Area"])

        if compare_by in ["Product", "Product Group"]:
            metric_options = ["Net Sales", "Total Returns", "Total Discounts", "Net Units Sold"]
        else:
            metric_options = ["Net Sales", "Total Returns", "Total Discounts", "Net Units Sold", "Collection"]

        metric = st.selectbox("Metric", metric_options)

        dimension_column_map = {
            "Salesman":      ("spid",      "spname"),
            "Customer":      ("cusid",     "cusname"),
            "Product":       ("itemcode",  "itemname"),
            "Product Group": ("itemgroup", None),
            "Area":          ("area",      None),
        }
        code_col, name_col = dimension_column_map[compare_by]

        if name_col and name_col in filtered_data_s.columns and code_col in filtered_data_s.columns:
            filtered_sub = filtered_data_s[[code_col, name_col]].dropna().drop_duplicates()
            filtered_sub["combined"] = filtered_sub[code_col].astype(str) + " - " + filtered_sub[name_col].astype(str)
            display_options = sorted(filtered_sub["combined"].tolist())
        elif code_col in filtered_data_s.columns:
            display_options = sorted(filtered_data_s[code_col].dropna().unique().astype(str).tolist())
        else:
            display_options = []

        is_mom = compare_type == "Month vs Month"
        if is_mom:
            selected_display = st.multiselect(
                f"Select {compare_by}(s) to Compare",
                options=display_options, default=display_options[:3], max_selections=7,
                key="comp_entity_select",
            )
            selected_codes = [x.split(" - ")[0] for x in selected_display] if name_col else selected_display
        else:
            selected_display = st.selectbox(
                f"Select {compare_by} to Filter (leave as '(All)' to aggregate)",
                options=["(All)"] + display_options,
                key="comp_entity_select",
            )
            selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []

        if compare_type == "Year-over-Year (YOY)" and granularity == "Monthly":
            collection.plot_yoy_monthly_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                selected_month_names=selected_months
            )
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Daily":
            collection.plot_yoy_daily_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                start_date=start_date,
                end_date=end_date
            )
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Day of Week":
            collection.plot_yoy_dow_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                average_or_total=average_or_total_YOY_DOW
            )
        elif compare_type == "Year-over-Year (YOY)" and granularity == "Day of Month":
            collection.plot_yoy_dom_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_years=selected_years,
                selected_month_names=selected_months,
                average_or_total=average_or_total_YOY_DOM
            )
        elif compare_type == "Month vs Month" and granularity == "Monthly":
            collection.plot_month_vs_month_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                name_col=name_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_months=selected_months
            )
        elif compare_type == "Month vs Month" and granularity == "Day of Week":
            collection.plot_month_vs_month_dow_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                name_col=name_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_months=selected_months,
                aggregation_type=average_or_total_MOM_DOW
            )
        elif compare_type == "Month vs Month" and granularity == "Day of Month":
            collection.plot_month_vs_month_dom_comparison(
                filtered_data_c=filtered_data_c,
                filtered_data_s=filtered_data_s,
                filtered_data_r=filtered_data_r,
                code_col=code_col,
                name_col=name_col,
                selected_codes=selected_codes,
                metric=metric,
                selected_months=selected_months,
                aggregation_type=average_or_total_MOM_DOM,
                selected_days=selected_dom_days
            )
        else:
            st.warning("Please select at least one year and month.")

    elif analysis_mode == "Distributions":
        st.subheader("📊 Distribution Analysis")

        # Step 1: Metric and Group Choice
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_metric = st.selectbox("Metric", [
                "Net Sales",
                "Total Returns",
                "Total Discounts",
                "Collection"
            ], key="dist_metric")

        with col2:
            # Dynamically adjust group options
            if selected_metric == "Collection":
                group_options = ["Salesman", "Customer", "Area"]
            else:
                group_options = ["Customer", "Product", "Salesman", "Area", "Product Group"]

            selected_group = st.selectbox("Group By", group_options, key="dist_group")

        with col3:
            min_date = filtered_data_c["date"].min()
            max_date = filtered_data_c["date"].max()
            start_date, end_date = st.date_input("Date Range", value=(min_date, max_date))

        # Step 2: Value Range and Bins
        col_min, col_max, col_bins, col_button = st.columns(4)
        with col_min:
            value_min = st.number_input("Min Value (optional)", value=None, placeholder="e.g. 1000", key="dist_min")
        with col_max:
            value_max = st.number_input("Max Value (optional)", value=None, placeholder="e.g. 50000", key="dist_max")
        with col_bins:
            nbins = st.number_input("Number of Bins", min_value=1, max_value=500, value=100, key="dist_bins")
        with col_button:
            st.markdown("<br>", unsafe_allow_html=True)
            run_distribution = st.button("Generate Distribution", key="dist_button")

        # Step 3: Run Distribution on Button Click
        if run_distribution:
            filtered_data_d = filtered_data_s[
                (filtered_data_s["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_s["date"] <= pd.to_datetime(end_date))
            ]
            filtered_data_r_d = filtered_data_r[
                (filtered_data_r["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_r["date"] <= pd.to_datetime(end_date))
            ]
            filtered_data_c_d = filtered_data_c[
                (filtered_data_c["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_c["date"] <= pd.to_datetime(end_date))
            ]

            collection.plot_distribution_analysis(
                filtered_data_s=filtered_data_d,
                filtered_data_r=filtered_data_r_d,
                filtered_data_c=filtered_data_c_d,  # Pass collection data also
                metric=selected_metric,
                group_by=selected_group,
                value_min=value_min,
                value_max=value_max,
                nbins=nbins
            )

    elif analysis_mode == "Descriptive Stats":
        st.subheader("📐 Summary Statistics")

        # Date Range Selector
        min_date = filtered_data_c["date"].min()
        max_date = filtered_data_c["date"].max()
        start_date, end_date = st.date_input("Select Date Range", value=(min_date, max_date))

        group_by = st.selectbox("Group By", [
            "Customer", "Product", "Salesman", "Area", "Product Group",
        ])

        if st.button("Generate Summary Statistics"):
            # Filter datasets by Date
            filtered_data_d = filtered_data_s[
                (filtered_data_s["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_s["date"] <= pd.to_datetime(end_date))
            ]
            filtered_data_r_d = filtered_data_r[
                (filtered_data_r["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_r["date"] <= pd.to_datetime(end_date))
            ]
            filtered_data_c_d = filtered_data_c[
                (filtered_data_c["date"] >= pd.to_datetime(start_date)) &
                (filtered_data_c["date"] <= pd.to_datetime(end_date))
            ]

            # Compute and display
            stats_df = collection.generate_descriptive_statistics(filtered_data_d, filtered_data_r_d, filtered_data_c_d, group_by)
            st.markdown("### 📋 Summary Table")
            st.dataframe(stats_df, use_container_width=True)

    elif analysis_mode == "Metric Comparison":
        st.subheader("Compare Metrics")

        # 1. Compare By Entity
        compare_by = st.selectbox("Compare By", ["Salesman", "Customer", "Area"])

        # 2. Metric Choices based on Compare By
        full_metric_choices = [
            "Net Sales", "Total Returns", "Total Discounts", "Collection"
        ]

        # Restrict Collection to only allowed entities
        if compare_by in ["Product", "Product Group"]:
            metric_choices = [
                "Net Sales", "Total Returns", "Total Discounts"
            ]
        else:
            metric_choices = full_metric_choices

        # 3. Metric A and B Selection
        col1, col2 = st.columns(2)
        with col1:
            metric_x = st.selectbox("Metric A (Base)", metric_choices, index=0)
        with col2:
            metric_y = st.selectbox("Metric B (Compared Against A)", metric_choices, index=3)

        # 4. Year and Month Filters
        all_years = sorted(filtered_data_s["year"].dropna().unique().astype(int).tolist())
        all_months = sorted(filtered_data_s["month"].dropna().unique().astype(int).tolist())
        month_map = {i: calendar.month_abbr[i] for i in all_months}

        selected_years = st.multiselect("Select Years", all_years, default=all_years)
        selected_months = st.multiselect("Select Months", [month_map[m] for m in all_months], default=[month_map[m] for m in all_months])

        # 5. Entity Code Selection
        dimension_column_map = {
            "Product": ("itemcode", "itemname"),
            "Product Group": ("itemgroup", None),
            "Customer": ("cusid", "cusname"),
            "Salesman": ("spid", "spname"),
            "Area": ("area", None)
        }
        code_col, name_col = dimension_column_map[compare_by]

        if name_col:
            filtered_sub = filtered_data_s[[code_col, name_col]].dropna().drop_duplicates()
            filtered_sub["combined"] = filtered_sub[code_col].astype(str) + " - " + filtered_sub[name_col].astype(str)
            display_options = sorted(filtered_sub["combined"].tolist())
        else:
            display_options = sorted(filtered_data_s[code_col].dropna().unique().tolist())

        selected_display = st.selectbox(f"Select {compare_by} to View", options=["(All)"] + display_options)
        selected_codes = [selected_display.split(" - ")[0]] if selected_display != "(All)" else []

        # 6. Call the Plot Function
        collection.plot_metric_comparison_monthly(
            filtered_data_c=filtered_data_c,
            filtered_data_s=filtered_data_s,
            filtered_data_r=filtered_data_r,
            code_col=code_col,
            selected_codes=selected_codes,
            metric_x=metric_x,
            metric_y=metric_y,
            selected_years=selected_years,
            selected_month_names=selected_months
        )

    elif analysis_mode == "CP":
        st.subheader("Collection Performance")

        if str(zid) in _DUAL_ZIDS:
            _cp_scope = st.radio("Scope", _SCOPE_OPTS, horizontal=True, key="cp_zid_scope")
        else:
            _cp_scope = _SCOPE_OPTS[0]

        if _cp_scope == _SCOPE_OPTS[1]:
            sales_df, returns_df, collection_df, _, _ = _get_combined_collection(
                zid, project, filtered_data_s, filtered_data_r, filtered_data_c
            )
        else:
            sales_df      = filtered_data_s
            returns_df    = filtered_data_r
            collection_df = filtered_data_c

        try:
            #filtered options for collection, - shows the filtering options for sales, returns and collections and outputs the dataset after the filter
            sales_df, returns_df, collection_df = collection.filtered_options_for_collection(sales_df, returns_df, collection_df)

            # Early check: warn and stop if any core dataset is empty after filtering
            if collection_df.empty or sales_df.empty:
                st.warning(
                    "⚠️ The selected filters returned no data. "
                    "Please reselect your options — the current combination may lack sufficient data for valid reporting."
                )
                st.stop()

            sales_df = sales_df.groupby(['date', 'year', 'month', 'cusid', 'cusname', 'DOM', 'DOW']).final_sales.sum().reset_index()
            returns_df = returns_df.groupby(['date', 'year', 'month', 'cusid', 'cusname', 'DOM', 'DOW']).treturnamt.sum().reset_index()
            collection_df = collection_df.groupby(['date', 'year', 'month', 'cusid', 'cusname', 'DOM', 'DOW']).value.sum().reset_index()

            # Compute average days and other metrics
            avg_days, pivot_df, avg_days_between, combined_df = collection.average_days_to_collection(sales_df, returns_df, collection_df)

            # Check computed outputs are usable
            if avg_days is None or (isinstance(avg_days, pd.DataFrame) and avg_days.empty):
                st.warning(
                    "⚠️ Not enough data to compute collection metrics. "
                    "Please reselect your options — the current combination may lack sufficient data for valid reporting."
                )
                st.stop()

            summary_df = collection.customer_segmentation_by_collection_days(avg_days)

            # Display metrics in order:
            # ① Avg Days to Collection, ② Avg Days Between, ③ Collection Days by Year/Month, ④ Customer Segmentation
            for title, df in [
                ("Average Days to Collection", avg_days),
                ("Average Days Between Collections", avg_days_between),
                ("Collection Days by Year/Month", pivot_df),
                ("Customer Segmentation by Days to Collection", summary_df),
            ]:
                st.markdown(title)
                st.write(df)

            # Timeframe / DOW / DOM comparison in expander, defaulting to Monthly
            with st.expander("📊 Sales/Collection Comparison by Timeframe", expanded=False):
                timeframe = st.selectbox('Select Time Range', ['Daily', 'Monthly', 'Yearly'], index=1)
                grouped_df, grouped_df_DOM, grouped_df_DOW = collection.get_grouped_df_collection(sales_df, returns_df, collection_df, timeframe)

                for title, (df, x_axis) in {
                        f"Comparison of Sales/Collection {timeframe}": (grouped_df, 'timeframe'),
                        "Comparison of Sales/Collection DOM": (grouped_df_DOM, 'DOM'),
                        "Comparison of Sales/Collection DOW": (grouped_df_DOW, 'DOW')
                }.items():
                    st.markdown(title)
                    df = df.rename(columns={'value': 'value_collection'})
                    st.write(df)

        except Exception as _cp_err:
            st.warning(
                "⚠️ Unable to generate Collection Performance report. "
                "Please reselect your options — the current combination may lack sufficient data for valid reporting."
            )
            st.caption(f"Details: {_cp_err}")
            # long_df = df.melt(id_vars=[x_axis], value_vars=['final_sales', 'treturnamt', 'value_collection'], var_name='category', value_name='value')
            # common_v.plot_bar_chart(data=long_df, x_axis=x_axis, y_axis='value', color='category', title=f'{x_axis} Collection Analysis')

    # ── CPA temporarily muted ─────────────────────────────────────────────────
    # elif analysis_mode == "CPA":
    #     st.subheader("Collection Performance Advanced")
    #
    #     performance_analysis_type = st.radio(
    #         "Choose Analysis Type:",
    #         ["Order Timeliness Metrics", "Payment Timeliness", "Composite Scoring"],
    #         horizontal=True,
    #     )
    #
    #     if performance_analysis_type == "Order Timeliness Metrics":
    #         order_df, avg_df, std_df = collection.compute_order_frequency_metrics(filtered_data_ar)
    #         with st.expander("📦 Order Frequency (Count per Year)", expanded=True):
    #             st.dataframe(order_df)
    #         with st.expander("📅 Average Interval Between Orders (Days)", expanded=True):
    #             st.write("Average number of days between consecutive orders.")
    #             st.dataframe(avg_df)
    #         with st.expander("📉 Std. Dev. of Interval Between Orders (Days)", expanded=True):
    #             st.write("Standard deviation of inter-order intervals (low sd ⇒ regular orders).")
    #             st.dataframe(std_df)
    #     elif performance_analysis_type == "Payment Timeliness":
    #         collection.display_payment_timeliness_page(filtered_data_ar)
    #     elif performance_analysis_type == "Composite Scoring":
    #         collection.display_composite_scoring_page(filtered_data_ar)

    elif analysis_mode == "Customer Ledger":
        st.subheader("Customer Ledger")

        if str(zid) in _DUAL_ZIDS:
            _cl_scope = st.radio("Scope", _SCOPE_OPTS, horizontal=True, key="cl_zid_scope")
        else:
            _cl_scope = _SCOPE_OPTS[0]

        if _cl_scope == _SCOPE_OPTS[1]:
            # ── Combined: two separate selectbox panels, one per ZID ─────────
            other = _OTHER_ZID[str(zid)]
            _yrs = tuple(sorted(filtered_data_s["year"].dropna().unique().astype(int).tolist())) \
                   if not filtered_data_s.empty else ()
            _mos = tuple(sorted(filtered_data_s["month"].dropna().unique().astype(int).tolist())) \
                   if not filtered_data_s.empty else ()
            _s2, _r2, _c2, _ar2 = _load_other_zid_collection(other, project, _yrs, _mos)
            if not (_c2.empty and _s2.empty):
                _c2_e, _s2_e, _r2_e = common.data_copy_add_columns(_c2, _s2, _r2)
                _c2_e = common.enrich_collection_with_sales_info(_c2_e, _s2_e)
            else:
                _s2_e, _r2_e, _c2_e = _s2, _r2, _c2

            _ar2_f = (
                _ar2[_ar2["project"] == project].copy()
                if (_ar2 is not None and not _ar2.empty and "project" in _ar2.columns)
                else pd.DataFrame()
            )

            for _lzid, _ls, _lr, _lc, _lar in [
                (str(zid), filtered_data_s, filtered_data_r, filtered_data_c, filtered_data_ar),
                (other,    _s2_e,           _r2_e,           _c2_e,           _ar2_f),
            ]:
                st.markdown(f"#### ZID {_lzid}")
                _render_cust_ledger_body(_ls, _lr, _lc, _lar, key_suffix=f"_{_lzid}")
                st.divider()
        else:
            _render_cust_ledger_body(
                filtered_data_s, filtered_data_r, filtered_data_c,
                filtered_data_ar, key_suffix="",
            )

    elif analysis_mode == "Salesman Due":
        st.subheader("Salesman Due")
        st.caption("Salesman-wise customer due report, ported from the HM_36 daily due scripts.")

        sub_report = st.radio(
            "Select Report:",
            ["Main Due Report", "Latest Sale & Collection", "Customer Credit Trickle-down", "Missing Customers"],
            horizontal=True,
            key="salesman_due_sub_report",
        )

        if str(zid) in _DUAL_ZIDS:
            _sd_scope = st.radio("Scope", _SCOPE_OPTS, horizontal=True, key="sd_zid_scope")
        else:
            _sd_scope = _SCOPE_OPTS[0]

        try:
            reports = _load_salesman_due_reports(str(zid), project)
            if _sd_scope == _SCOPE_OPTS[1]:
                _other = _OTHER_ZID[str(zid)]
                reports2 = _load_salesman_due_reports(_other, project)
            else:
                reports2 = {}
        except Exception as _sd_err:
            st.warning("⚠️ Unable to load Salesman Due report.")
            st.caption(f"Details: {_sd_err}")
            reports = {}
            reports2 = {}

        if not reports:
            st.info("No AR customer due data found for this business.")
        else:
            report_map = {
                "Main Due Report": "main_due_report",
                "Latest Sale & Collection": "report_df",
                "Customer Credit Trickle-down": "report_cc_with_names",
                "Missing Customers": "missing_customers_df",
            }
            report_key = report_map[sub_report]

            if _sd_scope == _SCOPE_OPTS[1] and reports2:
                # ── Combined mode ─────────────────────────────────────────────
                if sub_report == "Main Due Report":
                    _main_mode = st.radio(
                        "Combined view",
                        ["Stacked (by ZID)", "Merged Total"],
                        horizontal=True,
                        key="sd_main_combined_mode",
                    )
                    if _main_mode == "Stacked (by ZID)":
                        m1 = reports["main_due_report"].copy()
                        m2 = reports2["main_due_report"].copy()
                        m1.insert(0, "ZID", str(zid))
                        m2.insert(0, "ZID", _other)
                        df_sd = pd.concat([m1, m2], ignore_index=True)
                    else:
                        # Merge: rebuild from combined customer-credit data
                        try:
                            cc_comb = pd.concat(
                                [reports["report_cc_with_names"],
                                 reports2["report_cc_with_names"]],
                                ignore_index=True,
                            )
                            rsc = salesman_due.group_by_salesman_area_city(cc_comb)
                            df_sd = salesman_due.build_main_due_report_market(rsc)
                        except Exception as _merge_err:
                            st.warning("⚠️ Could not build merged total — showing stacked instead.")
                            st.caption(f"Details: {_merge_err}")
                            m1 = reports["main_due_report"].copy();  m1.insert(0, "ZID", str(zid))
                            m2 = reports2["main_due_report"].copy(); m2.insert(0, "ZID", _other)
                            df_sd = pd.concat([m1, m2], ignore_index=True)
                else:
                    # For all other sub-reports: concat + prepend ZID column
                    d1 = reports[report_key].copy();  d1.insert(0, "ZID", str(zid))
                    d2 = reports2[report_key].copy(); d2.insert(0, "ZID", _other)
                    df_sd = pd.concat([d1, d2], ignore_index=True)
            else:
                df_sd = reports[report_key]

            if sub_report == "Latest Sale & Collection" and "Salesman Code" in df_sd.columns:
                sp_opts = (
                    df_sd[["Salesman Code", "Salesman Name"]]
                    .dropna(subset=["Salesman Code"])
                    .drop_duplicates()
                    .sort_values("Salesman Code")
                )
                sp_opts["label"] = (
                    sp_opts["Salesman Code"].astype(str) + " - " + sp_opts["Salesman Name"].astype(str)
                )
                sel_sp_label = st.selectbox(
                    "Filter by Salesman (leave empty for all)",
                    options=[""] + sp_opts["label"].tolist(),
                    key="salesman_due_latest_sp_filter",
                )
                if sel_sp_label:
                    sel_sp_code = sel_sp_label.split(" - ")[0]
                    df_sd = df_sd[df_sd["Salesman Code"].astype(str) == sel_sp_code].copy()

            # ── Days-since columns for Latest Sale & Collection ───────────────
            if sub_report == "Latest Sale & Collection":
                df_sd = df_sd.copy()
                _today = pd.Timestamp.today().normalize()

                if "Sales Date" in df_sd.columns:
                    df_sd["Sales Date"] = pd.to_datetime(df_sd["Sales Date"], errors="coerce")
                    df_sd["Days Since Sale"] = (
                        (_today - df_sd["Sales Date"]).dt.days
                        .where(df_sd["Sales Date"].notna())
                        .astype("Int64")
                    )

                if "Latest Collection Date" in df_sd.columns:
                    df_sd["Latest Collection Date"] = pd.to_datetime(
                        df_sd["Latest Collection Date"], errors="coerce"
                    )
                    df_sd["Days Since Collection"] = (
                        (_today - df_sd["Latest Collection Date"]).dt.days
                        .where(df_sd["Latest Collection Date"].notna())
                        .astype("Int64")
                    )

                # Reorder: insert each days-since column immediately after its date column
                _cols = list(df_sd.columns)
                for _date_col, _days_col in [
                    ("Sales Date", "Days Since Sale"),
                    ("Latest Collection Date", "Days Since Collection"),
                ]:
                    if _days_col in _cols and _date_col in _cols:
                        _cols.remove(_days_col)
                        _cols.insert(_cols.index(_date_col) + 1, _days_col)
                df_sd = df_sd[_cols]

            st.caption(f"{len(df_sd):,} rows")
            if len(df_sd) > 50_000:
                st.info("Showing first 50,000 rows. Download the CSV for full data.")
                st.dataframe(df_sd.head(50_000))
            else:
                st.dataframe(df_sd)

            st.download_button(
                "Download CSV",
                data=df_sd.to_csv(index=False).encode("utf-8"),
                file_name=f"salesman_due_{report_key}_{zid}.csv",
                mime="text/csv",
                key=f"salesman_due_download_{report_key}",
            )

            # ── Call log viewer (Latest Sale & Collection only) ───────────────
            if sub_report == "Latest Sale & Collection" and not df_sd.empty:
                st.markdown("---")
                st.markdown("#### 📞 Call Log Lookup")
                st.caption("View calls logged by customer support for any customer in this list.")

                _has_name = "Customer Name" in df_sd.columns
                _cus_opts_df = (
                    df_sd[["Customer Code", "Customer Name"] if _has_name else ["Customer Code"]]
                    .drop_duplicates("Customer Code")
                )
                _cus_labels = {
                    (
                        f"{row['Customer Name']} ({row['Customer Code']})"
                        if _has_name
                        else str(row["Customer Code"])
                    ): str(row["Customer Code"])
                    for _, row in _cus_opts_df.iterrows()
                }
                _cus_sel = st.selectbox(
                    "Select customer",
                    ["— pick a customer —"] + sorted(_cus_labels.keys()),
                    key="sd_calllog_cus",
                )
                if _cus_sel and _cus_sel != "— pick a customer —":
                    _cusid    = _cus_labels[_cus_sel]
                    _cus_name = _cus_sel.split(" (")[0] if "(" in _cus_sel else _cus_sel
                    _render_call_log_readonly(cusid=_cusid, customer_name=_cus_name)

    elif analysis_mode == "📈 Order Analytics":
        st.subheader("📈 Order Analytics")
        st.caption("Filters below apply across all sub-sections. Empty = no filter applied.")

        if str(zid) in _DUAL_ZIDS:
            _oa_scope = st.radio("Scope", _SCOPE_OPTS, horizontal=True, key="oa_zid_scope")
        else:
            _oa_scope = _SCOPE_OPTS[0]

        if _oa_scope == _SCOPE_OPTS[1]:
            _, _, _oa_c, _, _ = _get_combined_collection(
                zid, project, filtered_data_s, filtered_data_r, filtered_data_c
            )
            _oa_zid_col = "zid"
        else:
            _oa_c       = filtered_data_c
            _oa_zid_col = None

        with st.expander("🔍 Entity Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                sel_areas = st.multiselect("Area",
                    sorted(_oa_c["area"].dropna().unique().tolist()), key="coa_areas")
            with col2:
                sel_salesmen = st.multiselect("Salesman",
                    sorted(_oa_c["spname"].dropna().unique().tolist()), key="coa_salesmen")
            with col3:
                sel_customers = st.multiselect("Customer",
                    sorted(_oa_c["cusname"].dropna().unique().tolist()), key="coa_customers")

        sub_mode = st.radio(
            "Sub-section",
            ["Collection Size Distribution", "Rolling Average"],
            horizontal=True, key="coa_sub",
        )

        if sub_mode == "Collection Size Distribution":
            col1, col2, col3 = st.columns(3)
            with col1:
                value_min = st.number_input("Min Value (optional)", value=None, placeholder="e.g. 1000", key="coa_min")
            with col2:
                value_max = st.number_input("Max Value (optional)", value=None, placeholder="e.g. 100000", key="coa_max")
            with col3:
                nbins = st.number_input("Number of Bins", min_value=5, max_value=500, value=50, key="coa_bins")

            collection.plot_collection_size_distribution(
                _oa_c,
                sel_areas, sel_salesmen, sel_customers,
                value_min, value_max, nbins,
                zid_col=_oa_zid_col,
            )
            with st.expander("📖 How to read this chart"):
                st.markdown("""
**What you are looking at:** Each bar represents how many collection vouchers fall within a given payment value range. Only positive-value vouchers (actual payments received) are included; adjustments and reversals are excluded.

**What to look for:**
- **Where the bulk sits** — if most vouchers are in the 1K–10K range, customers are making many small payments rather than settling large dues at once. This increases collection visits but reduces credit risk concentration.
- **Large-value tail** — a few very large collection vouchers may represent bulk settlements after long overdue periods, or simply large accounts paying their monthly statement. Check whether these correlate with large outstanding dues in the AR analysis.
- **Comparing collection size to order size** — if your typical collection voucher is significantly smaller than your typical order, customers are consistently paying in instalments. This is a working-capital signal: the gap between invoicing and full recovery is wider than it appears.
- **Filtering by salesman** — collection is often the salesman's responsibility. A salesman whose collection distribution skews smaller than peers may be collecting partial payments or avoiding difficult conversations about overdue balances.
- **Filtering by area** — areas with very small average collection sizes relative to their sales volume may have a cultural or logistics barrier to cash collection that needs a different approach (e.g. mobile banking, more frequent visits).
                """)

        elif sub_mode == "Rolling Average":
            ra_windows = st.multiselect("Rolling Windows (days)", [5, 10, 30, 60],
                                        default=[10, 30], key="coa_ra_windows")
            if ra_windows:
                collection.plot_rolling_collection_average(
                    _oa_c,
                    sel_areas, sel_salesmen, sel_customers,
                    ra_windows,
                    zid_col=_oa_zid_col,
                )
                with st.expander("📖 How to read this chart"):
                    st.markdown("""
**What you are looking at:** The grey bars are daily total collection (sum of all payment vouchers received that day). The coloured lines are rolling averages over your selected windows.

**What to look for:**
- **Collection rhythm** — unlike sales, which spike at order creation, collection often has a predictable rhythm tied to your credit terms (e.g. 30-day, 45-day). A rolling average that lags the sales rolling average by approximately your credit term is healthy and expected.
- **Collection dropping faster than sales** — if the collection rolling average falls while the sales average holds steady, receivables are building up. This is a cash-flow warning sign even if the P&L looks fine.
- **End-of-month spikes in daily bars** — many businesses see collection concentrate in the last week of the month as customers settle before month-end. If this is absent, it may mean customers are not being asked to pay on a schedule.
- **Short-window vs. long-window divergence** — a 10-day rolling average running consistently below the 30-day average means recent collection is slower than the period average. This often reflects a backlog building or a key collector being absent.
- **Comparing collection to sales rolling average** — hold both charts side by side (or apply the same filters). A growing gap between sales and collection trends is the clearest operational signal that credit control needs attention.
                    """)
            else:
                st.info("Select at least one rolling window.")
