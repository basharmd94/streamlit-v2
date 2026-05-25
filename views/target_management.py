import calendar
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from processing import common, target_management as tm, buying_pattern as bp, daily_sales as ds
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


def _prune_targets():
    """
    Silently remove target entries older than 24 rolling months.
    Runs on every page load — no prompt, no notification.
    """
    data = _load_json(_TARGETS_FILE)
    if not data:
        return
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=24)
    cutoff_ts = pd.Timestamp(cutoff.year, cutoff.month, 1)
    pruned = {}
    for key, val in data.items():
        try:
            # key format: {zid}_{spid}_{YYYY}-{MM}
            date_part = key.rsplit("_", 1)[-1]   # "YYYY-MM"
            y, m = int(date_part[:4]), int(date_part[5:7])
            if pd.Timestamp(y, m, 1) >= cutoff_ts:
                pruned[key] = val
        except Exception:
            pruned[key] = val  # keep unparseable entries
    if len(pruned) != len(data):
        _save_json(_TARGETS_FILE, pruned)


def _prune_holidays():
    """
    Silently remove holidays from calendar years older than (current_year - 1).
    Keeps exactly 2 calendar years: previous year and current year.
    Runs on every page load — no prompt.
    """
    data = _load_json(_HOLIDAYS_FILE)
    if not data:
        return
    keep_from = pd.Timestamp.today().year - 1
    holidays = data.get("holidays", [])
    pruned = [h for h in holidays if int(h[:4]) >= keep_from]
    if len(pruned) != len(holidays):
        data["holidays"] = sorted(pruned)
        _save_json(_HOLIDAYS_FILE, data)


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


# ── Inventory default warehouses (mirrors views/inventory.py) ─────────────────

_INV_DEFAULT_WAREHOUSES = [
    "Finished Goods Store Packaging",
    "HMBR -Main Store (4th Floor)",
    "HMBR -W5 (MirerBaazar 2nd Floor)",
    "HMBR -W7 (2) (MirerBaazar 3rd Floor)",
    "HMBR -W7 (MirerBaazar 3rd Floor)",
    "HMBR Showroom (Mirer Bazar)",
    "Raw Material Store Packaging",
]

# Per-ZID default item groups for inventory coverage.
# When defined, only these groups are shown and prefix exclusions (Z/RAW/M)
# are skipped. ZIDs not listed here fall back to the prefix exclusions.
_INV_DEFAULT_ITEMGROUPS_BY_ZID = {
    "100000": [
        "",          # items with no/empty itemgroup value
        "Chemical Item",
        "Steel Item",
        "Plastic Item",
        "Thread Tape Item",
        "Multiplug",
        "Drain Cover",
        "Paint Roller Item",
    ],
    "100005": [
        "Industrial & Household",
        "Marble Cleaner",
        "Multisurface Cleaner",
        "Laundry Detergent",
        "Steel Scrubber",
    ],
}


@st.cache_data(show_spinner=False, ttl=86400)
def _load_inv_stock_summed(zid: str, cutoff_year: int, cutoff_month: int) -> pd.DataFrame:
    """
    Load final stock summed at item level (warehouse column OFF).
    Applies the default warehouse list, excludes items whose code/name
    starts with 'Z' or 'RAW', and keeps only items with final_qty > 50.
    Cutoff is the last snapshot at or before cutoff_year/cutoff_month.
    """
    from core.analytics import Analytics

    def _effective_zids(primary: str) -> list:
        return [primary, "100009"] if primary == "100001" else [primary]

    frames = []
    for z in _effective_zids(str(zid)):
        try:
            df = Analytics("stock", zid=z, filters={"zid": (str(z),)}).data
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df.assign(_src_zid=str(z)))   # tag source zid
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()

    inv = pd.concat(frames, ignore_index=True)

    # Normalize types
    for col in ["year", "month"]:
        if col in inv.columns:
            inv[col] = pd.to_numeric(inv[col], errors="coerce")
    for c in ["warehouse", "itemcode", "itemname", "itemgroup"]:
        if c in inv.columns:
            inv[c] = inv[c].astype(str)
    if "stockqty" in inv.columns:
        inv["stockqty"] = pd.to_numeric(inv["stockqty"], errors="coerce").fillna(0.0)

    # Apply cutoff
    inv["ym"] = inv["year"].fillna(0).astype(int) * 100 + inv["month"].fillna(0).astype(int)
    inv = inv[inv["ym"] <= (cutoff_year * 100 + cutoff_month)]

    if inv.empty:
        return pd.DataFrame()

    # Filter to default warehouses (intersect with available)
    if "warehouse" in inv.columns:
        available_wh = set(inv["warehouse"].unique())
        wh_filter = [w for w in _INV_DEFAULT_WAREHOUSES if w in available_wh]
        if wh_filter:
            inv = inv[inv["warehouse"].isin(wh_filter)]

    # Sum by itemcode ONLY — do NOT include itemname/itemgroup in the groupby.
    # The same itemcode can appear in both zids with different caitem names/groups
    # (100009 xdrawing maps to a 100001 code but keeps 100009's xdesc).
    # Grouping by name/group would produce two rows instead of one summed row.
    agg_qty = (
        inv.groupby("itemcode", as_index=False)
           .agg(final_qty=("stockqty", "sum"))
    )

    # Name/group lookup — prefer primary zid's caitem so the displayed name
    # matches what the main company's inventory shows.
    _meta = (
        inv[["itemcode", "itemname", "itemgroup", "_src_zid"]]
        .drop_duplicates()
        .sort_values("_src_zid", key=lambda s: s.map(lambda x: 0 if x == str(zid) else 1))
        .drop_duplicates("itemcode", keep="first")
        [["itemcode", "itemname", "itemgroup"]]
    )
    agg = agg_qty.merge(_meta, on="itemcode", how="left")

    # Apply item group filter or prefix exclusions depending on ZID
    default_groups = _INV_DEFAULT_ITEMGROUPS_BY_ZID.get(str(zid))
    if default_groups and "itemgroup" in agg.columns:
        # ZID has explicit item groups defined — filter to those only.
        # "" in the list also catches null/NaN itemgroup values.
        named = [g for g in default_groups if g != ""]
        include_blank = "" in default_groups
        mask = agg["itemgroup"].isin(named)
        if include_blank:
            mask |= agg["itemgroup"].isna() | (agg["itemgroup"].str.strip() == "")
        agg = agg[mask]
    else:
        # No explicit groups defined — exclude items starting with Z, RAW, or M
        name_up = agg["itemname"].str.upper()
        code_up = agg["itemcode"].str.upper()
        exclude = (
            name_up.str.startswith("Z")   | name_up.str.startswith("RAW") | name_up.str.startswith("M") |
            code_up.str.startswith("Z")   | code_up.str.startswith("RAW") | code_up.str.startswith("M")
        )
        agg = agg[~exclude]

    # Exclude zero-stock items
    agg = agg[agg["final_qty"] >= 1]

    return agg.reset_index(drop=True)


# ── Cacus directory loader ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=86400)
def _load_cacus_directory(zid: str) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("cacus_directory", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


# ── Opmob pending orders loader ───────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=86400)
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
    sp_returns: pd.DataFrame = None,
):
    """Render the performance metric cards for the selected salesman."""
    today = pd.Timestamp.today().normalize()
    holidays = _get_holidays()

    # ── Base data prep ────────────────────────────────────────────────────────
    sp = sp_sales.copy()
    has_date = "date" in sp.columns
    if has_date:
        sp["_dt"] = pd.to_datetime(sp["date"], errors="coerce")
        sp["_d"]  = sp["_dt"].dt.date

    # ── Last 3 complete months ────────────────────────────────────────────────
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

    # ── MTD sales + returns ───────────────────────────────────────────────────
    mtd_sales = 0.0
    if has_date:
        mtd_sales = float(sp[sp["_dt"] >= mo_start_cur]["final_sales"].sum())

    mtd_return = 0.0
    if sp_returns is not None and not sp_returns.empty and "treturnamt" in sp_returns.columns:
        _r = sp_returns.copy()
        _r["_dt"] = pd.to_datetime(_r["date"], errors="coerce")
        mtd_return = float(_r[_r["_dt"] >= mo_start_cur]["treturnamt"].sum())

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

    # ── Target controls — fully horizontal ───────────────────────────────────
    st.markdown("**🎯 Monthly Target**")
    t_cols = st.columns([1.5, 1.5, 0.7, 1.3, 1.3, 1.3, 1.5])

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
            st.metric("MTD Return", f"{mtd_return:,.0f}")
        with t_cols[5]:
            net_sales = mtd_sales - mtd_return
            st.metric("Net Sales", f"{net_sales:,.0f}")
        with t_cols[6]:
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
                    pct = ((mtd_sales - mtd_return) - target_val) / target_val * 100
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

    st.markdown("---")


# ── All-Salesmen Overview ──────────────────────────────────────────────────────

def _render_overview(sales_df: pd.DataFrame, returns_df: pd.DataFrame, opmob_all: pd.DataFrame, zid):
    """
    Two-table overview for the currently selected ZID:
      Table 1 — one row per salesman: target/MTD/daily-required/3-month metrics
      Table 2 — one row per salesman × date × area for the current month:
                 sales, unique customers, unique products, pending opmob total
    """
    today        = pd.Timestamp.today().normalize()
    holidays     = _get_holidays()
    cur_year     = today.year
    cur_month    = today.month
    mo_start_cur = pd.Timestamp(cur_year, cur_month, 1)
    mo_start_3mo = mo_start_cur - pd.DateOffset(months=3)
    end_3mo      = mo_start_cur - pd.Timedelta(days=1)
    month_end    = pd.Timestamp(cur_year, cur_month,
                                calendar.monthrange(cur_year, cur_month)[1])

    wd_3mo       = _count_working_days(mo_start_3mo.date(), end_3mo.date(), holidays)
    remaining_wd = _count_working_days(today.date(), month_end.date(), holidays)

    if "date" not in sales_df.columns or "final_sales" not in sales_df.columns:
        st.warning("Required columns missing.")
        return

    df = sales_df.copy()
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["_d"]  = df["_dt"].dt.date

    last3   = df[(df["_dt"] >= mo_start_3mo) & (df["_dt"] <= end_3mo)]
    mtd_all = df[(df["_dt"] >= mo_start_cur) & (df["_dt"] <= today)]

    # ── MTD returns per salesman ───────────────────────────────────────────────
    mtd_ret_by_sp: dict = {}   # spid -> mtd treturnamt
    if returns_df is not None and not returns_df.empty and "treturnamt" in returns_df.columns:
        _r = returns_df.copy()
        _r["_dt"] = pd.to_datetime(_r["date"], errors="coerce")
        _r_mtd = _r[(_r["_dt"] >= mo_start_cur) & (_r["_dt"] <= today)]
        if "spid" in _r_mtd.columns:
            mtd_ret_by_sp = _r_mtd.groupby(_r_mtd["spid"].astype(str))["treturnamt"].sum().to_dict()

    # ── ZID-wide unique products (3 months) ───────────────────────────────────
    zid_up = int(last3["itemcode"].nunique()) if "itemcode" in last3.columns else 0

    # ── opmob pending per salesman × area ────────────────────────────────────
    pend_sp_area: dict = {}   # (spid, area) -> total pending
    if not opmob_all.empty:
        cusid_area = (
            df[["cusid", "area"]].dropna().drop_duplicates("cusid")
            .set_index("cusid")["area"].to_dict()
        )
        ob = opmob_all.copy()
        ob["area"] = ob["cusid"].astype(str).map(cusid_area)
        if "spid" in ob.columns and "linetotal" in ob.columns:
            for (sp, ar), grp in ob.dropna(subset=["area"]).groupby(["spid", "area"]):
                pend_sp_area[(str(sp), str(ar))] = float(grp["linetotal"].sum())

    # ── Table 1: salesman summary ─────────────────────────────────────────────
    st.subheader("📋 Salesman Summary — Current Month")
    sp_list = (
        df[["spid", "spname"]].dropna().drop_duplicates()
        .sort_values("spname")
    )
    rows1 = []
    for _, sp_row in sp_list.iterrows():
        spid   = str(sp_row["spid"])
        spname = sp_row["spname"]

        sp3   = last3[last3["spid"].astype(str) == spid]
        sp_mtd = mtd_all[mtd_all["spid"].astype(str) == spid]

        total_3mo   = float(sp3["final_sales"].sum())
        daily_avg   = round(total_3mo / wd_3mo, 0)  if wd_3mo > 0 else 0.0
        monthly_avg = round(total_3mo / 3, 0)
        mtd_sales   = float(sp_mtd["final_sales"].sum())

        mtd_ret   = round(mtd_ret_by_sp.get(spid, 0.0), 0)
        net_sales = round(mtd_sales - mtd_ret, 0)

        target    = _get_target(zid, spid, cur_year, cur_month) or 0.0
        gap       = target - mtd_sales
        daily_req = round(gap / remaining_wd, 0) if remaining_wd > 0 and target > 0 else 0.0
        pct_tgt   = round(net_sales / target * 100, 1) if target > 0 else None

        uc_3mo   = int(sp3["cusid"].nunique())    if "cusid"    in sp3.columns else 0
        up_3mo   = int(sp3["itemcode"].nunique()) if "itemcode" in sp3.columns else 0
        daily_uc = round(float(sp3.groupby("_d")["cusid"].nunique().mean()),    1) if not sp3.empty and "cusid"    in sp3.columns else 0.0
        daily_up = round(float(sp3.groupby("_d")["itemcode"].nunique().mean()), 1) if not sp3.empty and "itemcode" in sp3.columns else 0.0

        rows1.append({
            "Salesman":         spname,
            "Target":           target,
            "MTD Sales":        round(mtd_sales, 0),
            "MTD Return":       mtd_ret,
            "Net Sales":        net_sales,
            "% vs Target":      pct_tgt,
            "Days Left":        remaining_wd,
            "Daily Required":   daily_req,
            "Daily Avg (3M)":   daily_avg,
            "Monthly Avg (3M)": monthly_avg,
            "Uniq Cust (3M)":   uc_3mo,
            "Avg Daily Cust":   daily_uc,
            "Uniq Prods (3M)":  up_3mo,
            "Avg Daily Prods":  daily_up,
            "ZID Uniq Prods":   zid_up,
        })

    if rows1:
        t1 = pd.DataFrame(rows1).sort_values("MTD Sales", ascending=False).reset_index(drop=True)

        def _style_t1(df):
            styled = df.style.format({
                "Target":           "{:,.0f}",
                "MTD Sales":        "{:,.0f}",
                "MTD Return":       "{:,.0f}",
                "Net Sales":        "{:,.0f}",
                "% vs Target":      lambda v: f"{v:.1f}%" if v is not None else "—",
                "Days Left":        "{:,.0f}",
                "Daily Required":   "{:,.0f}",
                "Daily Avg (3M)":   "{:,.0f}",
                "Monthly Avg (3M)": "{:,.0f}",
                "Uniq Cust (3M)":   "{:,.0f}",
                "Avg Daily Cust":   "{:.1f}",
                "Uniq Prods (3M)":  "{:,.0f}",
                "Avg Daily Prods":  "{:.1f}",
                "ZID Uniq Prods":   "{:,.0f}",
            }, na_rep="—")

            def _col_pct(col):
                out = []
                for v in col:
                    if v is None:
                        out.append("")
                    elif v >= 100:
                        out.append("background-color: #D4EDDA; color: #155724")
                    elif v >= 75:
                        out.append("background-color: #FFF3CD; color: #856404")
                    else:
                        out.append("background-color: #F8D7DA; color: #721C24")
                return out

            if "% vs Target" in df.columns:
                styled = styled.apply(_col_pct, subset=["% vs Target"])
            return styled

        try:
            st.dataframe(_style_t1(t1), use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(t1, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇ Download Summary CSV",
            t1.to_csv(index=False).encode("utf-8"),
            file_name=f"summary_{zid}_{cur_year}_{cur_month:02d}.csv",
            mime="text/csv",
            key="dl_ov_summary",
        )

    # ── Prior 3 months — one expander each ───────────────────────────────────
    st.markdown("---")
    st.subheader("📆 Previous 3 Months")
    for _i in range(1, 4):
        _prior = mo_start_cur - pd.DateOffset(months=_i)
        _render_prior_month_section(
            df, returns_df, zid, int(_prior.year), int(_prior.month), holidays
        )


# ── Prior-month salesman performance section ─────────────────────────────────

def _render_prior_month_section(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    zid,
    year: int,
    month: int,
    holidays: set,
):
    """
    Render one prior month's salesman performance inside an expander.
    Columns match the current-month summary table, with Days Left / Daily Required
    fixed at 0 and Sales showing the full-month total.
    Monthly Avg (3M) = average of the 3 months immediately before this month.
    """
    import calendar as _cal

    mo_start  = pd.Timestamp(year, month, 1)
    last_day  = _cal.monthrange(year, month)[1]
    mo_end    = pd.Timestamp(year, month, last_day)
    mo_label  = f"{_cal.month_abbr[month]} {year}"

    # Working days in this month
    wd_month = _count_working_days(mo_start.date(), mo_end.date(), holidays)

    # 3-month lookback window (months immediately before this month)
    m3_end   = mo_start - pd.Timedelta(days=1)
    m3_start = mo_start - pd.DateOffset(months=3)

    df = sales_df.copy()
    if "_dt" not in df.columns:
        df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    if "_d" not in df.columns:
        df["_d"] = df["_dt"].dt.date

    mo_data = df[(df["_dt"] >= mo_start) & (df["_dt"] <= mo_end)]
    m3_data = df[(df["_dt"] >= m3_start) & (df["_dt"] < mo_start)]

    # ── Returns for this month per salesman ───────────────────────────────────
    ret_by_sp: dict = {}
    if returns_df is not None and not returns_df.empty and "treturnamt" in returns_df.columns:
        _r = returns_df.copy()
        _r["_dt"] = pd.to_datetime(_r["date"], errors="coerce")
        _r_mo = _r[(_r["_dt"] >= mo_start) & (_r["_dt"] <= mo_end)]
        if "spid" in _r_mo.columns:
            ret_by_sp = _r_mo.groupby(_r_mo["spid"].astype(str))["treturnamt"].sum().to_dict()

    # ZID-wide unique products for this month
    zid_up = int(mo_data["itemcode"].nunique()) if "itemcode" in mo_data.columns else 0

    sp_list = (
        df[["spid", "spname"]].dropna().drop_duplicates()
        .sort_values("spname")
    )

    rows = []
    for _, sp_row in sp_list.iterrows():
        spid   = str(sp_row["spid"])
        spname = sp_row["spname"]

        sp_mo = mo_data[mo_data["spid"].astype(str) == spid]
        sp_m3 = m3_data[m3_data["spid"].astype(str) == spid]

        sales      = float(sp_mo["final_sales"].sum())
        ret        = round(ret_by_sp.get(spid, 0.0), 0)
        net_sales  = round(sales - ret, 0)
        target     = float(_get_target(zid, spid, year, month) or 0.0)
        pct_tgt    = round(net_sales / target * 100, 1) if target > 0 else 0.0
        daily_avg  = round(sales / wd_month, 0) if wd_month > 0 else 0.0
        monthly_avg_3m = round(float(sp_m3["final_sales"].sum()) / 3, 0)

        uc = int(sp_mo["cusid"].nunique())    if "cusid"    in sp_mo.columns else 0
        up = int(sp_mo["itemcode"].nunique()) if "itemcode" in sp_mo.columns else 0

        avg_daily_uc = (
            round(float(sp_mo.groupby("_d")["cusid"].nunique().mean()), 1)
            if not sp_mo.empty and "cusid" in sp_mo.columns else 0.0
        )
        avg_daily_up = (
            round(float(sp_mo.groupby("_d")["itemcode"].nunique().mean()), 1)
            if not sp_mo.empty and "itemcode" in sp_mo.columns else 0.0
        )

        rows.append({
            "Salesman":         spname,
            "Target":           target,
            "Sales":            round(sales, 0),
            "Return":           ret,
            "Net Sales":        net_sales,
            "% vs Target":      pct_tgt,
            "Days Left":        0,
            "Daily Required":   0,
            "Daily Avg":        daily_avg,
            "Monthly Avg (3M)": monthly_avg_3m,
            "Uniq Cust":        uc,
            "Avg Daily Cust":   avg_daily_uc,
            "Uniq Prods":       up,
            "Avg Daily Prods":  avg_daily_up,
            "ZID Uniq Prods":   zid_up,
        })

    with st.expander(f"📅 {mo_label}", expanded=False):
        if not rows:
            st.info("No data available for this month.")
            return

        t = (
            pd.DataFrame(rows)
            .sort_values("Sales", ascending=False)
            .reset_index(drop=True)
        )

        def _style_prior(df_inner):
            styled = df_inner.style.format(
                {
                    "Target":           "{:,.0f}",
                    "Sales":            "{:,.0f}",
                    "Return":           "{:,.0f}",
                    "Net Sales":        "{:,.0f}",
                    "% vs Target":      "{:.1f}%",
                    "Days Left":        "{:,.0f}",
                    "Daily Required":   "{:,.0f}",
                    "Daily Avg":        "{:,.0f}",
                    "Monthly Avg (3M)": "{:,.0f}",
                    "Uniq Cust":        "{:,.0f}",
                    "Avg Daily Cust":   "{:.1f}",
                    "Uniq Prods":       "{:,.0f}",
                    "Avg Daily Prods":  "{:.1f}",
                    "ZID Uniq Prods":   "{:,.0f}",
                },
                na_rep="—",
            )

            def _col_pct(col):
                out = []
                for v in col:
                    if not v:
                        out.append("")
                    elif v >= 100:
                        out.append("background-color: #D4EDDA; color: #155724")
                    elif v >= 75:
                        out.append("background-color: #FFF3CD; color: #856404")
                    else:
                        out.append("background-color: #F8D7DA; color: #721C24")
                return out

            if "% vs Target" in df_inner.columns:
                styled = styled.apply(_col_pct, subset=["% vs Target"])
            return styled

        try:
            st.dataframe(_style_prior(t), use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(t, use_container_width=True, hide_index=True)

        st.download_button(
            f"⬇ Download {mo_label} CSV",
            t.to_csv(index=False).encode("utf-8"),
            file_name=f"summary_{zid}_{year}_{month:02d}.csv",
            mime="text/csv",
            key=f"dl_prior_{year}_{month:02d}",
        )


# ── Salesman daily breakdown (current month) ──────────────────────────────────

def _render_sp_daily_breakdown(
    sp_sales: pd.DataFrame,
    opmob_df: pd.DataFrame,
    sel_spid: str,
    zid,
):
    """
    Daily breakdown for the selected salesman for the current month:
    one row per date × area with Sales, Pending opmob, Uniq Cust, Uniq Prods.
    """
    today        = pd.Timestamp.today().normalize()
    cur_year     = today.year
    cur_month    = today.month
    mo_start_cur = pd.Timestamp(cur_year, cur_month, 1)

    st.markdown("---")
    st.subheader("📅 Daily Breakdown — Current Month")

    if "date" not in sp_sales.columns or sp_sales.empty:
        st.info("No sales data available for the daily breakdown.")
        return

    df = sp_sales.copy()
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["_d"]  = df["_dt"].dt.date
    mtd = df[(df["_dt"] >= mo_start_cur) & (df["_dt"] <= today)]

    if mtd.empty or "area" not in mtd.columns:
        st.info("No current-month sales data available for this salesman.")
        return

    # cusid → area lookup for pending opmob mapping
    cusid_area_map: dict = {}
    if "cusid" in df.columns and "area" in df.columns:
        cusid_area_map = (
            df[["cusid", "area"]].dropna()
            .drop_duplicates("cusid")
            .set_index("cusid")["area"]
            .to_dict()
        )

    # pending opmob by area for this salesman
    pend_area: dict = {}
    if not opmob_df.empty and "cusid" in opmob_df.columns and "linetotal" in opmob_df.columns:
        ob = opmob_df.copy()
        ob["area"] = ob["cusid"].astype(str).map(cusid_area_map)
        for ar, grp in ob.dropna(subset=["area"]).groupby("area"):
            pend_area[str(ar)] = float(grp["linetotal"].sum())

    grp = (
        mtd.dropna(subset=["area"])
        .groupby(["_d", "area"])
        .agg(
            Sales       =("final_sales", "sum"),
            uniq_cust   =("cusid",       pd.Series.nunique),
            uniq_prods  =("itemcode",    pd.Series.nunique),
        )
        .reset_index()
        .rename(columns={"_d": "Date", "area": "Area",
                         "uniq_cust": "Uniq Cust", "uniq_prods": "Uniq Prods"})
        .sort_values(["Date", "Area"], ascending=[False, True])
    )

    grp["Pending"] = grp["Area"].apply(lambda a: pend_area.get(str(a), 0.0))

    t = grp[["Date", "Area", "Sales", "Pending", "Uniq Cust", "Uniq Prods"]].reset_index(drop=True)

    try:
        st.dataframe(
            t.style.format({
                "Sales":      "{:,.0f}",
                "Pending":    "{:,.0f}",
                "Uniq Cust":  "{:,.0f}",
                "Uniq Prods": "{:,.0f}",
            }, na_rep="—"),
            use_container_width=True,
            hide_index=True,
            height=min(35 * len(t) + 60, 520),
        )
    except Exception:
        st.dataframe(t, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download Daily Breakdown CSV",
        t.to_csv(index=False).encode("utf-8"),
        file_name=f"daily_{sel_spid}_{cur_year}_{cur_month:02d}.csv",
        mime="text/csv",
        key="dl_sp_daily",
    )


# ── Inventory coverage vs this month's sales ──────────────────────────────────

def _render_inventory_coverage(sp_sales: pd.DataFrame, zid: str):
    """
    Compare what the salesman sold this month against stock available at the
    end of the previous month (warehouse-summed, qty >= 1, no Z/RAW/M items).

    🟢 Green       — in inventory AND sold this month
    🟣 Purple      — NOT sold this month BUT sold to these customers in the
                     loaded timeline (historical); shown even at 0 stock
    🔴 Red         — in inventory but never sold historically, not this month
    🔵 Blue        — sold this month but NOT in the inventory list
    """
    st.markdown("---")
    st.subheader("🗂️ Inventory Coverage — This Month vs Prior-Month Stock")

    today = pd.Timestamp.today()
    cur_year, cur_month = today.year, today.month

    # Cutoff = end of previous month
    prev = today - pd.DateOffset(months=1)
    cutoff_year, cutoff_month = int(prev.year), int(prev.month)
    mo_start = pd.Timestamp(cur_year, cur_month, 1)

    st.caption(
        f"Stock cutoff: **{cutoff_year}-{cutoff_month:02d}** · "
        f"Excludes items starting with **Z**, **RAW**, or **M** · "
        f"Default warehouses only (warehouse toggle OFF)"
    )

    # ── Guard ─────────────────────────────────────────────────────────────────
    if sp_sales.empty or "date" not in sp_sales.columns:
        st.info("No sales data available.")
        return

    sp = sp_sales.copy()
    sp["_dt"] = pd.to_datetime(sp["date"], errors="coerce")

    # ── Full item name lookup from all sales history ───────────────────────────
    name_lookup: dict = {}   # {itemcode: itemname}
    if "itemcode" in sp.columns and "itemname" in sp.columns:
        name_lookup = (
            sp[["itemcode", "itemname"]]
            .dropna(subset=["itemcode"])
            .drop_duplicates("itemcode")
            .assign(itemcode=lambda d: d["itemcode"].astype(str),
                    itemname=lambda d: d["itemname"].astype(str))
            .set_index("itemcode")["itemname"]
            .to_dict()
        )

    group_lookup: dict = {}  # {itemcode: itemgroup}
    if "itemcode" in sp.columns and "itemgroup" in sp.columns:
        group_lookup = (
            sp[["itemcode", "itemgroup"]]
            .dropna(subset=["itemcode"])
            .drop_duplicates("itemcode")
            .assign(itemcode=lambda d: d["itemcode"].astype(str),
                    itemgroup=lambda d: d["itemgroup"].astype(str))
            .set_index("itemcode")["itemgroup"]
            .to_dict()
        )

    # ── Products sold THIS month ───────────────────────────────────────────────
    sold_this_month = sp[sp["_dt"] >= mo_start]
    sold_codes: set = set()
    if not sold_this_month.empty and "itemcode" in sold_this_month.columns:
        sold_codes = set(sold_this_month["itemcode"].dropna().astype(str).unique())

    # ── Products sold in ANY PREVIOUS month (historical) ──────────────────────
    sold_prev = sp[sp["_dt"] < mo_start]
    historical_codes: set = set()
    if not sold_prev.empty and "itemcode" in sold_prev.columns:
        historical_codes = set(sold_prev["itemcode"].dropna().astype(str).unique())
    # Green takes full precedence — purple pool never contains current-month items
    purple_codes = historical_codes - sold_codes

    # ── Inventory stock at prior-month cutoff ──────────────────────────────────
    with st.spinner("Loading inventory data…"):
        inv_df = _load_inv_stock_summed(str(zid), cutoff_year, cutoff_month)

    if inv_df.empty:
        st.warning(
            f"No inventory data found for cutoff {cutoff_year}-{cutoff_month:02d}. "
            "The stock query may not have returned results for this period."
        )
        return

    inv_items: set = set(inv_df["itemcode"].astype(str).unique())
    # Build quick lookup for inv rows
    inv_row_map = {
        str(r["itemcode"]): r
        for _, r in inv_df.iterrows()
    }

    # ── Build combined table ───────────────────────────────────────────────────
    rows = []

    # Pass 1 — inventory items (qty >= 1)
    for code, inv_row in inv_row_map.items():
        sold_tm  = code in sold_codes
        prev_sold = code in purple_codes   # purple_codes already excludes sold_codes
        rows.append({
            "This Month":    "✅ Sold" if sold_tm else "❌ Not Sold",
            "Prev. Sold":    "🟣" if prev_sold else "",
            "Item Code":     code,
            "Item Name":     str(inv_row["itemname"]),
            "Item Group":    str(inv_row["itemgroup"]),
            "Stock Qty":     inv_row["final_qty"],
        })

    # Pass 2 — historically sold items with 0/no inventory stock
    for code in purple_codes:
        if code not in inv_items:
            rows.append({
                "This Month":  "❌ Not Sold",
                "Prev. Sold":  "🟣",
                "Item Code":   code,
                "Item Name":   name_lookup.get(code, "—"),
                "Item Group":  group_lookup.get(code, "—"),
                "Stock Qty":   0,
            })

    # Pass 3 — sold this month but not in inventory list
    for code in sold_codes:
        if code not in inv_items:
            rows.append({
                "This Month":  "✅ Sold",
                "Prev. Sold":  "🟣" if code in historical_codes else "",
                "Item Code":   code,
                "Item Name":   name_lookup.get(code, "—"),
                "Item Group":  group_lookup.get(code, "—"),
                "Stock Qty":   None,
            })

    if not rows:
        st.info("No data to display.")
        return

    result = pd.DataFrame(rows)
    # Sort: sold this month first, then not sold; within each group alphabetically
    result = (
        result.assign(_sort=result["This Month"].map({"✅ Sold": 0, "❌ Not Sold": 1}))
              .sort_values(["_sort", "Item Name"])
              .drop(columns=["_sort"])
              .reset_index(drop=True)
    )

    # ── Summary metrics ────────────────────────────────────────────────────────
    n_sold    = int((result["This Month"] == "✅ Sold").sum())
    n_unsold  = int((result["This Month"] == "❌ Not Sold").sum())
    n_prev    = int((result["Prev. Sold"] == "🟣").sum())
    n_missing = int(
        ((result["This Month"] == "✅ Sold") & result["Stock Qty"].isna()).sum()
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("✅ Sold This Month",   n_sold,    f"of {len(inv_df)} stocked items")
    mc2.metric("❌ Not Sold",          n_unsold,  "stocked but not sold this month")
    mc3.metric("🟣 Previously Sold",  n_prev,    "have prior history (any month)")
    mc4.metric("⚠️ Not in Stock",     n_missing, "sold but absent from stock list")

    # ── Colour-coded table ─────────────────────────────────────────────────────
    # Row color driven by "This Month" only — "Prev. Sold" column is a quiet indicator
    _ROW_BG = {
        "✅ Sold":    ("background-color: #D4EDDA", "color: #155724"),
        "❌ Not Sold": ("background-color: #F8D7DA", "color: #721C24"),
    }

    def _colour(row):
        bg, fg = _ROW_BG.get(row["This Month"], ("", ""))
        base = f"{bg}; {fg}"
        styles = [base] * len(row)
        # "Prev. Sold" cell gets no extra styling — neutral against the row bg
        return styles

    fmt = {"Stock Qty": lambda v: f"{v:,.0f}" if v is not None and pd.notna(v) else "—"}

    try:
        styled = result.style.apply(_colour, axis=1).format(fmt, na_rep="—")
        st.dataframe(styled, use_container_width=True, height=520, hide_index=True)
    except Exception:
        st.dataframe(result, use_container_width=True, height=520, hide_index=True)

    st.download_button(
        label=f"⬇ Download Coverage CSV ({len(result):,} rows)",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name=f"inv_coverage_{cur_year}_{cur_month:02d}_{zid}.csv",
        mime="text/csv",
        key="dl_inv_coverage",
    )


# ── Main view ─────────────────────────────────────────────────────────────────

@timed
def display_target_management_page(current_page, zid, data_dict):
    st.title("Target Management")

    # ── Maintenance: prune stale JSON entries on every load ───────────────────
    _prune_targets()
    _prune_holidays()

    # ── Holiday warning: prompt if no holidays entered for current year ────────
    _cur_year = pd.Timestamp.today().year
    _cur_year_holidays = [h for h in _get_holidays() if h.startswith(str(_cur_year))]
    if not _cur_year_holidays:
        st.warning(
            f"⚠️ No public holidays have been entered for **{_cur_year}**. "
            "Working-day calculations (daily averages, daily required, days left) "
            "will not account for public holidays until you add them. "
            "Please open the **🗓 Manage Public Holidays** panel above and add this year's holidays."
        )

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

    # ── View mode radio ───────────────────────────────────────────────────────
    _view_mode = st.radio(
        "View",
        ["👤 Individual Salesman", "📊 All Salesmen Overview", "📈 Moving Average"],
        horizontal=True,
        key="tm_view_mode",
    )

    # ── Public holidays management (always accessible) ───────────────────────
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

    if _view_mode == "📊 All Salesmen Overview":
        opmob_all = _load_opmob_pending(str(zid))
        _render_overview(sales_df, returns_df, opmob_all, zid)
        return

    if _view_mode == "📈 Moving Average":
        from datetime import date as _date
        st.subheader("📈 Moving Average Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            ma_entity = st.selectbox(
                "Entity", ["Salesman", "Product", "Product Group"], key="tm_ma_entity"
            )
        with col2:
            ma_metric = st.selectbox(
                "Metric", ["Net Sales", "Net Returns"], key="tm_ma_metric"
            )
        with col3:
            ma_end_date = st.date_input(
                "End Date", value=_date.today(), key="tm_ma_end_date"
            )

        try:
            ma_df = ds.compute_moving_avg_table(
                sales_df=sales_df,
                returns_df=returns_df,
                entity=ma_entity,
                metric=ma_metric,
                end_date=ma_end_date,
                collection_df=None,
            )
            if ma_df is not None and not ma_df.empty:
                st.dataframe(ma_df, use_container_width=True)
            else:
                st.info("No moving average data available for the selected filters.")
        except Exception as _ma_err:
            st.warning("Unable to compute moving average.")
            st.caption(f"Details: {_ma_err}")
        return

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
    _render_metric_cards(f_sp, opmob_df, sel_spid, zid, all_sales=sales_df, sp_returns=f_sp_ret)

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

    # ── Daily Breakdown for selected salesman ────────────────────────────────
    _render_sp_daily_breakdown(f_sp, opmob_df, sel_spid, zid)

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

    # ── Inventory Coverage ────────────────────────────────────────────────────
    _render_inventory_coverage(f_sp, str(zid))
