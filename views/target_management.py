import calendar
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from processing import common, target_management as tm, buying_pattern as bp
from utils.utils import timed


# ── JSON data paths ────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "data"
_TARGETS_FILE = _DATA_DIR / "targets.json"
_HOLIDAYS_FILE = _DATA_DIR / "public_holidays.json"


def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _save_json(path: Path, data: dict):
    _DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _target_key(zid, spid: str, year: int, month: int) -> str:
    return f"{zid}_{spid}_{year}-{month:02d}"


def _get_target(zid, spid: str, year: int, month: int):
    return _load_json(_TARGETS_FILE).get(_target_key(zid, spid, year, month))


def _save_target(zid, spid: str, year: int, month: int, value: float):
    data = _load_json(_TARGETS_FILE)
    data[_target_key(zid, spid, year, month)] = value
    _save_json(_TARGETS_FILE, data)


def _get_holidays() -> set:
    """Return all saved public holidays as a set of 'YYYY-MM-DD' strings."""
    return set(_load_json(_HOLIDAYS_FILE).get("holidays", []))


def _toggle_holiday(date_str: str, add: bool):
    data = _load_json(_HOLIDAYS_FILE)
    holidays = set(data.get("holidays", []))
    if add:
        holidays.add(date_str)
    else:
        holidays.discard(date_str)
    data["holidays"] = sorted(holidays)
    _save_json(_HOLIDAYS_FILE, data)


def _is_working_day(d, holidays: set) -> bool:
    """Mon–Thu and Sat–Sun are working days; Friday and public holidays are off."""
    return d.weekday() != 4 and d.strftime("%Y-%m-%d") not in holidays


def _count_working_days(start_d, end_d, holidays: set) -> int:
    count = 0
    cur = start_d
    while cur <= end_d:
        if _is_working_day(cur, holidays):
            count += 1
        cur += timedelta(days=1)
    return count


# ── Filter helpers ─────────────────────────────────────────────────────────────

def _sp_opts(df: pd.DataFrame) -> list:
    if not {"spid", "spname"}.issubset(df.columns):
        return []
    tmp = df[["spid", "spname"]].dropna().drop_duplicates().sort_values("spname")
    return (tmp["spid"].astype(str) + " - " + tmp["spname"].astype(str)).tolist()


def _cus_opts(df: pd.DataFrame) -> list:
    if not {"cusid", "cusname"}.issubset(df.columns):
        return []
    tmp = df[["cusid", "cusname"]].dropna().drop_duplicates().sort_values("cusname")
    return (tmp["cusid"].astype(str) + " - " + tmp["cusname"].astype(str)).tolist()


def _item_opts(df: pd.DataFrame) -> list:
    if not {"itemcode", "itemname"}.issubset(df.columns):
        return []
    tmp = df[["itemcode", "itemname"]].dropna().drop_duplicates().sort_values("itemname")
    return (tmp["itemcode"].astype(str) + " - " + tmp["itemname"].astype(str)).tolist()


def _codes(selection: list) -> list:
    """Extract the code part (before first ' - ') from code+name selections."""
    return [v.split(" - ", 1)[0].strip() for v in selection]


def _filter_code(df: pd.DataFrame, col: str, codes: list) -> pd.DataFrame:
    if not codes or col not in df.columns:
        return df
    return df[df[col].astype(str).isin([str(c) for c in codes])]


# ── Display helpers ────────────────────────────────────────────────────────────

def _current_month_label() -> str:
    now = pd.Timestamp.today()
    return f"{calendar.month_abbr[now.month]}-{str(now.year)[-2:]}"


def _month_cols(df: pd.DataFrame, id_cols: list) -> list:
    return [c for c in df.columns if c not in id_cols and c != "Total"]


def _with_totals_row(df: pd.DataFrame, id_cols: list) -> pd.DataFrame:
    if df.empty:
        return df
    totals = {}
    for c in df.columns:
        if c == id_cols[0]:
            totals[c] = "TOTAL"
        elif c in id_cols:
            totals[c] = ""
        else:
            totals[c] = df[c].sum() if pd.api.types.is_numeric_dtype(df[c]) else ""
    return pd.concat([df, pd.DataFrame([totals])], ignore_index=True)


def _styled(df: pd.DataFrame, id_cols: list, current_col: str):
    mcols = _month_cols(df, id_cols)
    numeric_cols = mcols + (["Total"] if "Total" in df.columns else [])

    def _hl(col):
        if col.name == current_col:
            return ["background-color: #FFF3CD; font-weight: bold"] * len(col)
        if col.name == "Total":
            return ["font-weight: bold"] * len(col)
        return [""] * len(col)

    fmt = {c: "{:,.0f}" for c in numeric_cols if c in df.columns}
    return df.style.apply(_hl, axis=0).format(fmt, na_rep="-")


def _render_table(
    df: pd.DataFrame,
    id_cols: list,
    current_col: str,
    dl_key: str,
    dl_filename: str,
):
    """Render a pivot table — no row cap, with TOTAL row and CSV download."""
    if df.empty:
        st.info("No data.")
        return
    row_count = len(df)
    show_df = _with_totals_row(df, id_cols)
    st.caption(f"{row_count:,} rows")
    try:
        st.dataframe(_styled(show_df, id_cols, current_col), use_container_width=True, height=480)
    except Exception:
        st.dataframe(show_df, use_container_width=True, height=480)
    st.download_button(
        label=f"⬇ Download CSV ({row_count:,} rows)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=dl_filename,
        mime="text/csv",
        key=f"dl_{dl_key}",
    )


def _render_not_ordered_table(
    df: pd.DataFrame,
    id_cols: list,
    current_col: str,
    pending_cusids: set,
    dl_key: str,
    dl_filename: str,
):
    """Render the not-ordered pivot with green highlight for rows with pending opmob orders."""
    if df.empty:
        st.info("No data.")
        return
    row_count = len(df)
    show_df = _with_totals_row(df, id_cols)
    st.caption(f"{row_count:,} rows")

    mcols = _month_cols(show_df, id_cols)
    numeric_cols = mcols + (["Total"] if "Total" in show_df.columns else [])

    def _style_func(d):
        styles = pd.DataFrame("", index=d.index, columns=d.columns)
        if current_col in d.columns:
            styles[current_col] = "background-color: #FFF3CD; font-weight: bold"
        if "Total" in d.columns:
            styles["Total"] = "font-weight: bold"
        if "Pending Order" in d.columns and pending_cusids:
            pending_mask = d["Pending Order"] == "✓"
            if pending_mask.any():
                for col in d.columns:
                    styles.loc[pending_mask, col] = "background-color: #D4EDDA"
                styles.loc[pending_mask, "Pending Order"] = (
                    "background-color: #198754; color: white; font-weight: bold; text-align: center"
                )
        return styles

    fmt = {c: "{:,.0f}" for c in numeric_cols if c in show_df.columns}
    try:
        st.dataframe(
            show_df.style.apply(_style_func, axis=None).format(fmt, na_rep="-"),
            use_container_width=True,
            height=480,
        )
    except Exception:
        st.dataframe(show_df, use_container_width=True, height=480)

    st.download_button(
        label=f"⬇ Download CSV ({row_count:,} rows)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=dl_filename,
        mime="text/csv",
        key=f"dl_{dl_key}",
    )


# ── Cacus directory loader ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_cacus_directory(zid: str) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("cacus_directory", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


# ── Opmob pending orders loader ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_opmob_pending(zid: str) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("opmob_pending", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


# ── Buying Pattern section ────────────────────────────────────────────────────

def _bp_styled(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def _priority_bg(val):
        try:
            v = float(val)
            if v >= 4.0: return "background-color: #FFCCCC; font-weight: bold"
            if v >= 2.8: return "background-color: #FFF3CD; font-weight: bold"
            return "background-color: #D4EDDA; font-weight: bold"
        except Exception:
            return ""

    def _trend_color(val):
        s = str(val)
        if "Growing"  in s: return "color: green; font-weight: bold"
        if "Declining" in s: return "color: red;   font-weight: bold"
        if "Flat"      in s: return "color: grey"
        return ""

    styler = df.style
    if "Visit Priority Score" in df.columns:
        try:
            styler = styler.map(_priority_bg, subset=["Visit Priority Score"])
        except AttributeError:
            styler = styler.applymap(_priority_bg, subset=["Visit Priority Score"])
    if "Trend" in df.columns:
        try:
            styler = styler.map(_trend_color, subset=["Trend"])
        except AttributeError:
            styler = styler.applymap(_trend_color, subset=["Trend"])
    if "Total Sales" in df.columns:
        styler = styler.format({"Total Sales": "{:,.0f}"})
    return styler


def _render_buying_pattern(bp_df: pd.DataFrame, is_any_filter: bool):
    """Render the buying pattern expander section."""
    with st.expander("📊 Buying Pattern Analysis", expanded=False):
        if bp_df.empty:
            st.info("No data to analyse for the current selection.")
            return

        ctrl1 = st.columns([3, 1])
        with ctrl1[0]:
            sort_col_map = {
                "Visit Priority Score": "priority_score",
                "Recency (months)":     "months_since_last",
                "Total Sales":          "total_sales",
                "Frequency":            "active_months",
                "Trend":                "_trend_score",
            }
            sort_label = st.selectbox(
                "Sort by", list(sort_col_map.keys()), key="bp_sort"
            )
        with ctrl1[1]:
            sort_desc = st.checkbox("Descending", value=True, key="bp_desc")

        ctrl2 = st.columns(4)
        all_tiers   = ["🥇 Platinum", "🥈 Gold", "🥉 Silver", "Bronze"]
        all_trends  = ["📈 Growing", "➡ Flat", "📉 Declining", "Insufficient data", "—"]
        with ctrl2[0]:
            tier_sel   = st.multiselect("Spend Tier", all_tiers,  default=all_tiers,  key="bp_tier")
        with ctrl2[1]:
            trend_sel  = st.multiselect("Trend",      all_trends, default=all_trends, key="bp_trend")
        with ctrl2[2]:
            rec_max    = st.slider("Max recency (months)", 1, 18, 18, key="bp_rec")
        with ctrl2[3]:
            single_only = st.checkbox("Single-product buyers only", key="bp_single")

        priority_band_sel = st.radio(
            "Visit Priority",
            ["🔴 This Week", "🟡 This Month", "🟢 All"],
            index=2,
            horizontal=True,
            key="bp_band",
        )

        filt = bp_df.copy()
        if tier_sel:
            filt = filt[filt["spend_tier"].isin(tier_sel)]
        if trend_sel:
            filt = filt[filt["trend"].isin(trend_sel)]
        filt = filt[filt["months_since_last"] <= rec_max]
        if single_only:
            filt = filt[filt["product_count"] == 1]
        if priority_band_sel == "🔴 This Week":
            filt = filt[filt["priority_score"] >= 4.0]
        elif priority_band_sel == "🟡 This Month":
            filt = filt[(filt["priority_score"] >= 2.8) & (filt["priority_score"] < 4.0)]

        sort_internal = sort_col_map[sort_label]
        if sort_internal in filt.columns:
            filt = filt.sort_values(sort_internal, ascending=not sort_desc)

        display_col_map = {
            "spname":           "Salesman",
            "cusid":            "Cust. Code",
            "cusname":          "Customer",
            "cusmobile":        "Mobile",
            "area":             "Area",
            "total_sales":      "Total Sales",
            "spend_tier":       "Tier",
            "months_since_last":"Recency (months)",
            "freq_display":     "Active months",
            "trend":            "Trend",
            "peak_months":      "Usual buying months",
            "product_count":    "Products bought",
            "priority_score":   "Visit Priority Score",
        }
        keep = [c for c in display_col_map if c in filt.columns]
        display_df = filt[keep].rename(columns=display_col_map).reset_index(drop=True)

        st.caption(f"{len(display_df):,} customers")

        try:
            st.dataframe(_bp_styled(display_df), use_container_width=True, height=520)
        except Exception:
            st.dataframe(display_df, use_container_width=True, height=520)

        st.download_button(
            label=f"⬇ Download CSV ({len(display_df):,} rows)",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="buying_pattern.csv",
            mime="text/csv",
            key="dl_bp",
        )


# ── Metric cards ───────────────────────────────────────────────────────────────

def _render_metric_cards(
    sp_sales: pd.DataFrame,
    opmob_df: pd.DataFrame,
    sel_spid: str,
    zid,
    all_sales: pd.DataFrame = None,
):
    """Render the performance metric cards for the selected salesman."""
    today = pd.Timestamp.today().normalize()
    holidays = _get_holidays()

    # ── Base data prep ────────────────────────────────────────────────────────
    last5_dates = set((today - pd.Timedelta(days=i)).date() for i in range(1, 6))
    last5_sorted = sorted(last5_dates, reverse=True)

    sp = sp_sales.copy()
    has_date = "date" in sp.columns
    if has_date:
        sp["_dt"] = pd.to_datetime(sp["date"], errors="coerce")
        sp["_d"]  = sp["_dt"].dt.date

    # ── Last 3 complete months (computed first — needed for mini table too) ───
    mo_start_cur = pd.Timestamp(today.year, today.month, 1)
    mo_start_3mo = mo_start_cur - pd.DateOffset(months=3)
    end_3mo      = mo_start_cur - pd.Timedelta(days=1)

    total_3mo = 0.0
    daily_avg_3mo = 0.0
    wd_3mo = 0
    last3 = pd.DataFrame()
    if has_date:
        last3 = sp[(sp["_dt"] >= mo_start_3mo) & (sp["_dt"] <= end_3mo)].copy()
        total_3mo = float(last3["final_sales"].sum())
        wd_3mo = _count_working_days(mo_start_3mo.date(), end_3mo.date(), holidays)
        daily_avg_3mo = total_3mo / wd_3mo if wd_3mo > 0 else 0.0

    monthly_avg_3mo = total_3mo / 3

    # Per-area 3-month daily average  (total area sales / working days)
    area_3mda: dict = {}
    if has_date and not last3.empty and "area" in last3.columns:
        area_totals = last3.groupby("area")["final_sales"].sum()
        area_3mda = (area_totals / wd_3mo if wd_3mo > 0 else area_totals * 0).to_dict()

    # ── Unique customer metrics (last 3 months) ───────────────────────────────
    total_uc_3mo   = 0
    avg_daily_uc   = 0.0
    if has_date and not last3.empty and "cusid" in last3.columns:
        total_uc_3mo = int(last3["cusid"].nunique())
        daily_uc_series = last3.groupby("_d")["cusid"].nunique()
        avg_daily_uc = float(daily_uc_series.mean()) if not daily_uc_series.empty else 0.0

    # ── Unique product metrics (last 3 months) ────────────────────────────────
    total_up_3mo  = 0
    avg_daily_up  = 0.0
    if has_date and not last3.empty and "itemcode" in last3.columns:
        total_up_3mo = int(last3["itemcode"].nunique())
        daily_up_series = last3.groupby("_d")["itemcode"].nunique()
        avg_daily_up = float(daily_up_series.mean()) if not daily_up_series.empty else 0.0

    # ── ZID-wide unique products (last 3 months) — for comparison ─────────────
    zid_up_3mo = 0
    if all_sales is not None and not all_sales.empty and "itemcode" in all_sales.columns:
        _all = all_sales.copy()
        if "date" in _all.columns:
            _all["_dt"] = pd.to_datetime(_all["date"], errors="coerce")
            _all3 = _all[(_all["_dt"] >= mo_start_3mo) & (_all["_dt"] <= end_3mo)]
            zid_up_3mo = int(_all3["itemcode"].nunique())

    # ── Last-5-days mini table (one row per date × area) ─────────────────────
    mini_rows = []
    if has_date:
        last5_sp = sp[sp["_d"].isin(last5_dates)].copy()

        # cusid → area lookup (for mapping opmob orders to areas)
        cusid_area_map: dict = {}
        if "cusid" in sp.columns and "area" in sp.columns:
            cusid_area_map = (
                sp[["cusid", "area"]].dropna()
                .drop_duplicates("cusid")
                .set_index("cusid")["area"]
                .to_dict()
            )

        # opmob pending by (date, area)
        opmob_area_day: dict = {}
        if not opmob_df.empty and "order_date" in opmob_df.columns:
            o = opmob_df[opmob_df["spid"].astype(str) == str(sel_spid)].copy()
            o["_d"]   = pd.to_datetime(o["order_date"], errors="coerce").dt.date
            o["area"] = o["cusid"].astype(str).map(cusid_area_map)
            o5 = o[o["_d"].isin(last5_dates)]
            if not o5.empty and "linetotal" in o5.columns:
                opmob_area_day = (
                    o5.groupby(["_d", "area"])["linetotal"].sum().to_dict()
                )

        if not last5_sp.empty and "area" in last5_sp.columns:
            grp = (
                last5_sp.dropna(subset=["area"])
                .groupby(["_d", "area"])
                .agg(do=("final_sales", "sum"), uc=("cusid", "nunique"))
                .reset_index()
                .sort_values(["_d", "area"], ascending=[False, True])
            )
            for _, row in grp.iterrows():
                mini_rows.append({
                    "Date":  row["_d"].strftime("%d %b"),
                    "Area":  row["area"],
                    "DO":    row["do"],
                    "Pend":  opmob_area_day.get((row["_d"], row["area"]), 0),
                    "3MDA":  area_3mda.get(row["area"], 0),
                    "UC":    int(row["uc"]),
                })

    # ── MTD + remaining days ──────────────────────────────────────────────────
    mtd_sales = 0.0
    if has_date:
        mtd_sales = float(sp[sp["_dt"] >= mo_start_cur]["final_sales"].sum())

    import calendar as _cal
    last_day_num = _cal.monthrange(today.year, today.month)[1]
    month_end    = date(today.year, today.month, last_day_num)
    tomorrow     = (today + pd.Timedelta(days=1)).date()
    remaining_wd = _count_working_days(tomorrow, month_end, holidays) if tomorrow <= month_end else 0

    # ── Month options for target selector ─────────────────────────────────────
    month_options = []
    for i in range(12):
        d = today + pd.DateOffset(months=i)
        label = f"{calendar.month_abbr[int(d.month)]} {int(d.year)}"
        month_options.append((label, int(d.year), int(d.month)))

    # ── Layout ────────────────────────────────────────────────────────────────
    st.markdown("---")

    # ── Row 1: Mini table full width ──────────────────────────────────────────
    st.markdown("**📅 Last 5 Days**")
    if mini_rows:
        df5 = pd.DataFrame(mini_rows)
        fmt = {"DO": "{:,.0f}", "Pend": "{:,.0f}", "3MDA": "{:,.0f}"}
        try:
            tbl_height = min(35 * len(df5) + 38, 320)
            st.dataframe(
                df5.style.format(fmt),
                use_container_width=True,
                hide_index=True,
                height=tbl_height,
            )
        except Exception:
            st.dataframe(df5, use_container_width=True, hide_index=True)
    else:
        st.info("No sales in the last 5 days.")

    st.markdown(" ")

    # ── Row 2: Target controls — fully horizontal ─────────────────────────────
    st.markdown("**🎯 Monthly Target**")
    t_cols = st.columns([1.5, 1.5, 0.7, 1.5, 1.5, 2])

    with t_cols[0]:
        sel_mo_label = st.selectbox(
            "Month",
            [m[0] for m in month_options],
            key=f"tm_target_month_{sel_spid}",
        )
    sel_mo_year, sel_mo_month = next(
        (m[1], m[2]) for m in month_options if m[0] == sel_mo_label
    )
    is_current_month = (sel_mo_year == today.year and sel_mo_month == today.month)
    saved_target = _get_target(zid, sel_spid, sel_mo_year, sel_mo_month)

    with t_cols[1]:
        target_val = st.number_input(
            "Target",
            min_value=0.0,
            value=float(saved_target) if saved_target is not None else 0.0,
            step=1000.0,
            format="%.0f",
            key=f"tm_target_{sel_spid}_{sel_mo_year}_{sel_mo_month}",
        )

    with t_cols[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save", key=f"tm_save_{sel_spid}"):
            _save_target(zid, sel_spid, sel_mo_year, sel_mo_month, target_val)
            st.toast(f"Target saved for {sel_mo_label}!", icon="✅")

    if is_current_month:
        with t_cols[3]:
            st.metric("MTD Sales", f"{mtd_sales:,.0f}")
        with t_cols[4]:
            if target_val > 0:
                gap = target_val - mtd_sales
                if remaining_wd > 0:
                    daily_req = gap / remaining_wd
                    icon = (
                        "🔴" if daily_req > daily_avg_3mo * 1.2
                        else "🟡" if daily_req > daily_avg_3mo
                        else "🟢"
                    )
                    st.metric(
                        "Daily Required",
                        f"{icon} {daily_req:,.0f}",
                        delta=f"{remaining_wd} days left",
                        delta_color="off",
                    )
                else:
                    pct = (mtd_sales - target_val) / target_val * 100
                    label = "Above target 🟢" if pct >= 0 else "Below target 🔴"
                    st.metric("vs Target", f"{pct:+.1f}%", delta=label, delta_color="off")
    else:
        with t_cols[3]:
            st.caption("MTD & daily required shown for current month only.")

    st.markdown(" ")

    # ── Row 3: Summary metrics — 7 equal columns ──────────────────────────────
    m_cols = st.columns(7)
    with m_cols[0]:
        st.metric("📊 Daily Avg Sales", f"{daily_avg_3mo:,.0f}", delta="last 3 months", delta_color="off")
    with m_cols[1]:
        st.metric("📈 Monthly Avg Sales", f"{monthly_avg_3mo:,.0f}", delta="last 3 months", delta_color="off")
    with m_cols[2]:
        st.metric("👥 Unique Customers", f"{total_uc_3mo:,}", delta="last 3 months", delta_color="off")
    with m_cols[3]:
        st.metric("👤 Avg Daily Customers", f"{avg_daily_uc:,.1f}", delta="last 3 months", delta_color="off")
    with m_cols[4]:
        st.metric("📦 Unique Products", f"{total_up_3mo:,}", delta="last 3 months", delta_color="off")
    with m_cols[5]:
        st.metric("🗂️ Avg Daily Products", f"{avg_daily_up:,.1f}", delta="last 3 months", delta_color="off")
    with m_cols[6]:
        _zid_delta = f"of {zid_up_3mo} ZID total" if zid_up_3mo > 0 else "last 3 months"
        st.metric("🏢 ZID Unique Products", f"{zid_up_3mo:,}", delta=_zid_delta, delta_color="off")

    # Public holidays management (expander below cards — covers full year)
    with st.expander("🗓 Manage Public Holidays", expanded=False):
        all_holidays = sorted(_get_holidays())

        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            hol_range = st.date_input(
                "Select a day or drag to pick a range (any month, any year)",
                value=(),
                key="tm_new_hol",
            )
        with h_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add", key="tm_add_hol"):
                if hol_range:
                    if isinstance(hol_range, (list, tuple)) and len(hol_range) == 2:
                        cur_h, end_h = hol_range
                        while cur_h <= end_h:
                            _toggle_holiday(str(cur_h), add=True)
                            cur_h += timedelta(days=1)
                    else:
                        single = hol_range[0] if isinstance(hol_range, (list, tuple)) else hol_range
                        _toggle_holiday(str(single), add=True)
                    st.rerun()

        if all_holidays:
            # Group by year-month for readability
            from itertools import groupby
            def _ym(d): return d[:7]  # "YYYY-MM"
            st.write(f"**{len(all_holidays)} holiday(s) saved across all months:**")
            for ym, group in groupby(all_holidays, key=_ym):
                yr, mo = int(ym[:4]), int(ym[5:])
                st.markdown(f"*{calendar.month_abbr[mo]} {yr}*")
                for h in group:
                    hc1, hc2 = st.columns([4, 1])
                    with hc1:
                        st.write(h)
                    with hc2:
                        if st.button("✖", key=f"rm_hol_{h}"):
                            _toggle_holiday(h, add=False)
                            st.rerun()
        else:
            st.info("No public holidays saved yet.")

    st.markdown("---")


# ── Main view ─────────────────────────────────────────────────────────────────

@timed
def display_target_management_page(current_page, zid, data_dict):
    st.title("Target Management")

    raw_sales   = data_dict.get("sales",  pd.DataFrame())
    raw_returns = data_dict.get("return", pd.DataFrame())

    if raw_sales.empty:
        st.warning("No sales data available for the selected filters.")
        return

    sales_df, returns_df = common.data_copy_add_columns(raw_sales, raw_returns)

    if "final_sales" not in sales_df.columns:
        st.error("Could not compute net sales — 'final_sales' column missing.")
        return

    current_col = _current_month_label()

    # ── Filters: salesman (single), customer, area (cascading) ────────────────
    fcols = st.columns(3)

    with fcols[0]:
        sp_opts = _sp_opts(sales_df)
        sel_sp_raw = st.selectbox(
            "Salesman *(required)",
            [None] + sp_opts,
            format_func=lambda x: "— select a salesman —" if x is None else x,
            key="tm_sp",
        )

    if not sel_sp_raw:
        st.info("👆 Select a salesman to view reports.")
        return

    sel_spid  = _codes([sel_sp_raw])[0]
    sel_spids = [sel_spid]

    # Filter by salesman
    f_sp     = _filter_code(sales_df,   "spid", sel_spids)
    f_sp_ret = _filter_code(returns_df, "spid", sel_spids)

    # Load pending opmob orders and filter to this salesman
    opmob_df = _load_opmob_pending(str(zid))
    if not opmob_df.empty:
        opmob_df = opmob_df[opmob_df["spid"].astype(str).isin([str(s) for s in sel_spids])].copy()
    pending_cusids = set(opmob_df["cusid"].astype(str).unique()) if not opmob_df.empty else set()

    # Customer options cascade from salesman selection
    with fcols[1]:
        sel_cus_raw = st.multiselect("Customer", _cus_opts(f_sp), key="tm_cus")

    sel_cusids = _codes(sel_cus_raw)
    f_sp_cus   = _filter_code(f_sp,     "cusid", sel_cusids) if sel_cusids else f_sp
    f_sp_cus_r = _filter_code(f_sp_ret, "cusid", sel_cusids) if sel_cusids else f_sp_ret

    # Area options cascade from salesman + customer selection
    with fcols[2]:
        area_opts = sorted(f_sp_cus["area"].dropna().unique().tolist())
        sel_area  = st.multiselect("Area", area_opts, key="tm_area")

    f_final   = f_sp_cus[f_sp_cus["area"].isin(sel_area)]     if sel_area else f_sp_cus
    f_final_r = f_sp_cus_r[f_sp_cus_r["area"].isin(sel_area)] if sel_area and "area" in f_sp_cus_r.columns else f_sp_cus_r

    # ── Metric cards (uses full salesman data, not customer/area filtered) ─────
    _render_metric_cards(f_sp, opmob_df, sel_spid, zid, all_sales=sales_df)

    # ── Customer-wise pivot ───────────────────────────────────────────────────
    try:
        pivot = tm.build_customer_wise_monthly(f_final, f_final_r)
    except Exception as e:
        st.warning("Unable to build report. Please adjust your filters.")
        st.caption(f"Details: {e}")
        return

    if pivot.empty:
        st.warning("No data for the current selection.")
        return

    id_raw      = ["spid", "spname", "cusid", "cusname", "cusmobile", "area"]
    month_col_list = [c for c in pivot.columns if c not in id_raw and c != "Total"]

    rename_map = {
        "spname":    "Salesman",
        "cusid":     "Cust. Code",
        "cusname":   "Customer",
        "cusmobile": "Mobile",
        "area":      "Area",
    }
    display_pivot = (
        pivot
        .drop(columns=["spid"], errors="ignore")
        .rename(columns=rename_map)
    )
    id_cols_display = [rename_map.get(c, c) for c in id_raw if c != "spid"]

    # Split on running month
    if current_col in display_pivot.columns:
        not_ordered = display_pivot[display_pivot[current_col] == 0].copy()
        ordered     = display_pivot[display_pivot[current_col]  > 0].copy()
    else:
        not_ordered = display_pivot.copy()
        ordered     = pd.DataFrame(columns=display_pivot.columns)

    st.subheader(f"🔴 Not Ordered — {current_col}")

    # Add pending order indicator column
    if pending_cusids and not not_ordered.empty:
        not_ordered["Pending Order"] = not_ordered["Cust. Code"].astype(str).apply(
            lambda x: "✓" if x in pending_cusids else ""
        )
        not_ordered_id_cols = id_cols_display + ["Pending Order"]
    else:
        not_ordered_id_cols = id_cols_display

    _render_not_ordered_table(
        not_ordered, not_ordered_id_cols, current_col, pending_cusids,
        "not_ordered", f"not_ordered_{current_col}.csv",
    )

    # Expander: pending order product breakdown
    if pending_cusids and not not_ordered.empty:
        not_ordered_cusids = set(not_ordered["Cust. Code"].astype(str).unique())
        pending_in_not_ordered = pending_cusids & not_ordered_cusids
        if pending_in_not_ordered and not opmob_df.empty:
            pending_detail = opmob_df[
                opmob_df["cusid"].astype(str).isin(pending_in_not_ordered)
            ].copy()
            with st.expander(
                f"📋 Pending opmob Orders — {len(pending_in_not_ordered)} customer(s)", expanded=False
            ):
                detail_display = (
                    pending_detail[["cusname", "cusid", "itemcode", "itemname", "linetotal"]]
                    .rename(columns={
                        "cusname":   "Customer",
                        "cusid":     "Cust. Code",
                        "itemcode":  "Item Code",
                        "itemname":  "Item",
                        "linetotal": "Line Total",
                    })
                    .reset_index(drop=True)
                )
                try:
                    st.dataframe(
                        detail_display.style.format({"Line Total": "{:,.2f}"}, na_rep="-"),
                        use_container_width=True,
                    )
                except Exception:
                    st.dataframe(detail_display, use_container_width=True)
                st.download_button(
                    label=f"⬇ Download CSV ({len(detail_display):,} rows)",
                    data=detail_display.to_csv(index=False).encode("utf-8"),
                    file_name=f"pending_orders_{current_col}.csv",
                    mime="text/csv",
                    key="dl_pending_orders",
                )

    st.markdown(" ")
    st.subheader(f"✅ Ordered — {current_col}")
    _render_table(ordered, id_cols_display, current_col, "ordered", f"ordered_{current_col}.csv")

    st.markdown("---")

    # ── Customer-Product sub-section ──────────────────────────────────────────
    st.subheader("📦 Customer-Product Breakdown")
    st.caption("Scoped to salesman / customer / area selections above.")

    scols = st.columns(2)
    with scols[0]:
        sel_sec_cus_raw  = st.multiselect("Customer", _cus_opts(f_final), key="tm_sec_cus")

    sel_sec_cusids = _codes(sel_sec_cus_raw)
    fs2 = _filter_code(f_final,   "cusid", sel_sec_cusids) if sel_sec_cusids else f_final
    fr2 = _filter_code(f_final_r, "cusid", sel_sec_cusids) if sel_sec_cusids else f_final_r

    with scols[1]:
        sel_sec_item_raw = st.multiselect("Product", _item_opts(fs2), key="tm_sec_item")

    sel_sec_itemcodes = _codes(sel_sec_item_raw)
    fs2 = _filter_code(fs2, "itemcode", sel_sec_itemcodes) if sel_sec_itemcodes else fs2
    fr2 = _filter_code(fr2, "itemcode", sel_sec_itemcodes) if sel_sec_itemcodes else fr2

    try:
        prod_pivot = tm.build_customer_product_monthly(fs2, fr2)
    except Exception as e:
        st.warning("Unable to build product breakdown.")
        st.caption(f"Details: {e}")
        prod_pivot = pd.DataFrame()

    if not prod_pivot.empty:
        prod_rename = {
            "spname":    "Salesman",
            "cusid":     "Cust. Code",
            "cusname":   "Customer",
            "cusmobile": "Mobile",
            "area":      "Area",
            "itemcode":  "Item Code",
            "itemname":  "Item",
        }
        prod_id_raw = ["spname", "cusid", "cusname", "cusmobile", "area", "itemcode", "itemname"]
        prod_display = (
            prod_pivot
            .drop(columns=["spid"], errors="ignore")
            .rename(columns=prod_rename)
        )
        prod_id_cols = [prod_rename.get(c, c) for c in prod_id_raw]
        _render_table(prod_display, prod_id_cols, current_col, "prod_breakdown", "customer_product_breakdown.csv")
    else:
        st.info("No product-level data for the current selection.")

    st.markdown("---")

    # ── No-sales customers from cacus ─────────────────────────────────────────
    st.subheader("🚫 Customers with No Sales")
    st.caption("Select an area to see customers in the directory with zero sales in the loaded period.")

    cacus_df = _load_cacus_directory(str(zid))

    if cacus_df.empty:
        st.warning("Customer directory not available.")
    else:
        # Use the same area options as the top filter (salesman's territory)
        sel_cacus_area = st.multiselect("Filter by Area", area_opts, key="tm_cacus_area")

        if not sel_cacus_area:
            st.info("Select one or more areas above to see customers with no sales.")
        else:
            cacus_filtered = cacus_df[cacus_df["area"].isin(sel_cacus_area)].copy()
            sold_cusids    = set(sales_df["cusid"].astype(str).unique())
            no_sales_df    = cacus_filtered[
                ~cacus_filtered["cusid"].astype(str).isin(sold_cusids)
            ].copy()

            no_sales_display = (
                no_sales_df
                .rename(columns={"cusname": "Customer", "cusmobile": "Mobile", "area": "Area"})
                [["Customer", "Mobile", "Area"]]
                .reset_index(drop=True)
            )
            st.caption(f"{len(no_sales_display):,} customers with no sales in the selected area(s)")

            if no_sales_display.empty:
                st.success("All customers in the selected area(s) have sales in this period.")
            else:
                st.dataframe(no_sales_display, use_container_width=True)
                st.download_button(
                    label=f"⬇ Download CSV ({len(no_sales_display):,} rows)",
                    data=no_sales_display.to_csv(index=False).encode("utf-8"),
                    file_name="no_sales_customers.csv",
                    mime="text/csv",
                    key="dl_no_sales",
                )

    st.markdown("---")

    # ── Buying Pattern Analysis ───────────────────────────────────────────────
    try:
        bp_df = bp.compute_buying_pattern(
            pivot_df   = pivot,
            sales_df   = f_final,
            id_cols    = [c for c in id_raw if c in pivot.columns],
            month_cols = month_col_list,
        )
    except Exception as e:
        bp_df = pd.DataFrame()
        st.caption(f"Buying pattern error: {e}")

    _render_buying_pattern(bp_df, is_any_filter=bool(sel_spid))
