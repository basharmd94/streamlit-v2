import calendar
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from processing import (
    common, target_management as tm, buying_pattern as bp, daily_sales as ds,
    next_month_target as nmt, salesman_due as sd, salesman_score as ssc,
)
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

def _format_unquoted_dict(d: dict) -> str:
    """Pretty-prints a dict as {key: value, ...} with no quotes around keys or
    string values — for copy/paste into systems that don't want JSON quoting."""
    lines = [f"  {k}: {v}" for k, v in d.items()]
    return "{\n" + ",\n".join(lines) + "\n}"


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


# ── Final items view loader ───────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_final_items(zid: str) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("final_items_view", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_opspprc(zid: str) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("opspprc", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


# ── Opmob pending orders loader ───────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=86400)
def _load_opmob_pending(zid: str) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("opmob_pending", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_opmob_all(zid: str) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("opmob_all", zid=zid, filters={}).data
    return df if df is not None else pd.DataFrame()


# ── Next Month Target loaders ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def _load_sales_window(zid: str, years: tuple) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("sales", zid=zid, filters={"year": list(years)}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_returns_window(zid: str, years: tuple) -> pd.DataFrame:
    from core.analytics import Analytics
    df = Analytics("return", zid=zid, filters={"year": list(years)}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_purchase_open_combined() -> pd.DataFrame:
    """Open IP purchase shipments for 100001 (HMBR import) + 100009 (Gulshan
    Packaging) combined — they keep separate PO books for the same physical
    shipments, so both need to be in play for Next Month Target's incoming
    shipment picker.
    """
    from core.analytics import Analytics
    df = Analytics("purchase", zid=["100001", "100009"], filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_ar_ledger_clean(zid: str, proj: str) -> pd.DataFrame:
    """Cleaned AR ledger (per-row running_balance per customer, salesman code
    bfilled/ffilled) for the Salesman Score tab's point-in-time balance
    snapshots — same source/cleanup as Collection Analysis's Salesman Due
    report, via salesman_due.prep_ar_ledger.
    """
    from core.analytics import Analytics
    ar_df = Analytics("ar_due_ledger", zid=zid, project=proj, filters={}).data
    if ar_df is None or ar_df.empty:
        return pd.DataFrame()
    return sd.prep_ar_ledger(ar_df)


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
            "whatsapp":         "WhatsApp Number",
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
    sp_returns: pd.DataFrame = None,
    collection_df: pd.DataFrame = None,
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

    # ── MTD sales + returns ───────────────────────────────────────────────────
    mtd_df = pd.DataFrame()
    mtd_sales = 0.0
    mtd_unique_products = 0
    mtd_unique_customers = 0
    if has_date:
        mtd_df = sp[sp["_dt"] >= mo_start_cur]
        mtd_sales = float(mtd_df["final_sales"].sum())
        mtd_unique_products = int(mtd_df["itemcode"].nunique()) if "itemcode" in mtd_df.columns else 0
        mtd_unique_customers = int(mtd_df["cusid"].nunique()) if "cusid" in mtd_df.columns else 0

    mtd_return = 0.0
    if sp_returns is not None and not sp_returns.empty and "treturnamt" in sp_returns.columns:
        _r = sp_returns.copy()
        _r["_dt"] = pd.to_datetime(_r["date"], errors="coerce")
        mtd_return = float(_r[_r["_dt"] >= mo_start_cur]["treturnamt"].sum())

    net_sales = mtd_sales - mtd_return

    # ── MTD collection for this salesman, current calendar month ─────────────
    mtd_collection = 0.0
    if (
        collection_df is not None and not collection_df.empty
        and "value" in collection_df.columns and "spid" in collection_df.columns
    ):
        _c = collection_df.copy()
        _c["spid"]  = _c["spid"].astype(str)
        _c["year"]  = pd.to_numeric(_c["year"],  errors="coerce")
        _c["month"] = pd.to_numeric(_c["month"], errors="coerce")
        _c_mtd = _c[
            (_c["spid"] == str(sel_spid)) & (_c["year"] == today.year) & (_c["month"] == today.month)
        ]
        mtd_collection = float(_c_mtd["value"].sum())

    import calendar as _cal
    last_day_num = _cal.monthrange(today.year, today.month)[1]
    month_end    = date(today.year, today.month, last_day_num)
    tomorrow     = (today + pd.Timedelta(days=1)).date()
    remaining_wd = _count_working_days(tomorrow, month_end, holidays) if tomorrow <= month_end else 0

    # ── Month options for target selector ─────────────────────────────────────
    # Show full current year: Jan <current_year> → Dec <current_year>
    month_options = []
    current_year = today.year
    for m in range(1, 13):
        label = f"{calendar.month_abbr[m]} {current_year}"
        month_options.append((label, current_year, m))

    # ── Layout ────────────────────────────────────────────────────────────────
    st.markdown("---")

    # ── Target controls — fully horizontal ───────────────────────────────────
    st.markdown("**🎯 Monthly Target**")
    t_cols = st.columns([1.5, 1.5, 0.7, 1.3, 1.3, 1.3, 1.5])

    with t_cols[0]:
        _default_mo_idx = next(
            (i for i, m in enumerate(month_options) if m[1] == today.year and m[2] == today.month),
            0,
        )
        sel_mo_label = st.selectbox(
            "Month",
            [m[0] for m in month_options],
            index=_default_mo_idx,
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

    # ── Row 2: MTD performance — collection, activity, % vs target, 3M avg ───
    if is_current_month:
        pct_mtd_vs_target = round(net_sales / target_val * 100, 1) if target_val > 0 else None
        pct_coll_vs_mtd_sales = round(mtd_collection / (1.02 * mtd_sales) * 100, 1) if mtd_sales > 0 else None

        p_cols = st.columns(6)
        with p_cols[0]:
            st.metric("💰 Collection (MTD)", f"{mtd_collection:,.0f}")
        with p_cols[1]:
            st.metric("📦 Products Sold (MTD)", f"{mtd_unique_products:,}")
        with p_cols[2]:
            st.metric("👥 Customers Visited (MTD)", f"{mtd_unique_customers:,}")
        with p_cols[3]:
            st.metric("🎯 % MTD Sales vs Target", f"{pct_mtd_vs_target:.1f}%" if pct_mtd_vs_target is not None else "—")
        with p_cols[4]:
            st.metric("💧 % Collection vs MTD Sales", f"{pct_coll_vs_mtd_sales:.1f}%" if pct_coll_vs_mtd_sales is not None else "—")
        with p_cols[5]:
            st.metric("📈 Monthly Avg Sales (3M)", f"{monthly_avg_3mo:,.0f}", delta="last 3 months", delta_color="off")

    st.markdown("---")


# ── All-Salesmen Overview ──────────────────────────────────────────────────────

def _render_overview(sales_df: pd.DataFrame, returns_df: pd.DataFrame, opmob_all: pd.DataFrame, zid, collection_df: pd.DataFrame = None):
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

    remaining_wd   = _count_working_days(today.date(), month_end.date(), holidays)
    wd_elapsed     = max(1, _count_working_days(mo_start_cur.date(), today.date(), holidays))
    total_wd_month = _count_working_days(mo_start_cur.date(), month_end.date(), holidays)

    if "date" not in sales_df.columns or "final_sales" not in sales_df.columns:
        st.warning("Required columns missing.")
        return

    df = sales_df.copy()
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["_d"]  = df["_dt"].dt.date

    last3   = df[(df["_dt"] >= mo_start_3mo) & (df["_dt"] <= end_3mo)]
    mtd_all = df[(df["_dt"] >= mo_start_cur) & (df["_dt"] <= today)]

    # Warn if historical data is missing — Monthly Avg (3M) will show as 0 without it
    if last3.empty:
        st.warning(
            f"⚠️ No data found for the 3-month lookback window "
            f"({mo_start_3mo.strftime('%b %Y')} – {end_3mo.strftime('%b %Y')}). "
            "**Monthly Avg (3M)** will show 0. "
            "Please load at least 3 prior months in the sidebar filters."
        )

    # ── MTD returns per salesman ───────────────────────────────────────────────
    mtd_ret_by_sp: dict = {}   # spid -> mtd treturnamt
    if returns_df is not None and not returns_df.empty and "treturnamt" in returns_df.columns:
        _r = returns_df.copy()
        _r["_dt"] = pd.to_datetime(_r["date"], errors="coerce")
        _r_mtd = _r[(_r["_dt"] >= mo_start_cur) & (_r["_dt"] <= today)]
        if "spid" in _r_mtd.columns:
            mtd_ret_by_sp = _r_mtd.groupby(_r_mtd["spid"].astype(str))["treturnamt"].sum().to_dict()

    # ── Collection per salesman × year × month ────────────────────────────────
    coll_by_sp: dict = {}  # (spid, year, month) -> collection value
    if collection_df is not None and not collection_df.empty and "value" in collection_df.columns:
        _c = collection_df.copy()
        _c["spid"]  = _c["spid"].astype(str)
        _c["year"]  = pd.to_numeric(_c["year"],  errors="coerce")
        _c["month"] = pd.to_numeric(_c["month"], errors="coerce")
        coll_by_sp = _c.groupby(["spid", "year", "month"])["value"].sum().to_dict()

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
        monthly_avg = round(total_3mo / 3, 0)
        mtd_sales   = float(sp_mtd["final_sales"].sum())

        mtd_ret   = round(mtd_ret_by_sp.get(spid, 0.0), 0)
        net_sales = round(mtd_sales - mtd_ret, 0)

        target    = _get_target(zid, spid, cur_year, cur_month) or 0.0
        gap       = target - mtd_sales
        daily_req = round(gap / remaining_wd, 0) if remaining_wd > 0 and target > 0 else 0.0
        pct_tgt   = round(net_sales / target * 100, 1) if target > 0 else None

        mtd_coll  = round(float(coll_by_sp.get((spid, cur_year, cur_month), 0.0)), 0)
        pct_coll  = round(mtd_coll / (1.02 * mtd_sales) * 100, 1) if mtd_sales > 0 else None

        mtd_up = int(sp_mtd["itemcode"].nunique()) if "itemcode" in sp_mtd.columns else 0
        mtd_uc = int(sp_mtd["cusid"].nunique())    if "cusid"    in sp_mtd.columns else 0

        rows1.append({
            "Salesman":                 spname,
            "Target":                   target,
            "MTD Sales":                round(mtd_sales, 0),
            "MTD Return":               mtd_ret,
            "Net Sales":                net_sales,
            "% vs Target":              pct_tgt,
            "MTD Collection":           mtd_coll,
            "% Collection":             pct_coll,
            "Products Sold (MTD)":      mtd_up,
            "Customers Visited (MTD)":  mtd_uc,
            "Days Left":                remaining_wd,
            "Daily Required":           daily_req,
            "Monthly Avg (3M)":         monthly_avg,
            "Daily Avg (MTD)":          round(mtd_sales / wd_elapsed, 0),
            "Predicted Total":          round((mtd_sales / wd_elapsed) * total_wd_month, 0),
            "Predicted vs Target":      round((mtd_sales / wd_elapsed) * total_wd_month - target, 0) if target > 0 else None,
            "Collection Gap":           round(target * 1.02 - mtd_coll, 0) if target > 0 else None,
        })

    if rows1:
        t1 = pd.DataFrame(rows1).sort_values("MTD Sales", ascending=False).reset_index(drop=True)

        def _style_t1(df):
            styled = df.style.format({
                "Target":                   "{:,.0f}",
                "MTD Sales":                "{:,.0f}",
                "MTD Return":               "{:,.0f}",
                "Net Sales":                "{:,.0f}",
                "% vs Target":              lambda v: f"{v:.1f}%" if v is not None else "—",
                "MTD Collection":           "{:,.0f}",
                "% Collection":             lambda v: f"{v:.1f}%" if v is not None else "—",
                "Products Sold (MTD)":      "{:,.0f}",
                "Customers Visited (MTD)":  "{:,.0f}",
                "Days Left":                "{:,.0f}",
                "Daily Required":           "{:,.0f}",
                "Monthly Avg (3M)":         "{:,.0f}",
                "Daily Avg (MTD)":          "{:,.0f}",
                "Predicted Total":          "{:,.0f}",
                "Predicted vs Target":      lambda v: f"{v:+,.0f}" if v is not None else "—",
                "Collection Gap":           lambda v: f"{v:,.0f}" if v is not None else "—",
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
            if "% Collection" in df.columns:
                styled = styled.apply(_col_pct, subset=["% Collection"])
            return styled

        try:
            st.dataframe(_style_t1(t1), use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(t1, use_container_width=True, hide_index=True)

        _3m_period = f"{mo_start_3mo.strftime('%b %Y')} – {end_3mo.strftime('%b %Y')}"
        st.caption(
            f"**Monthly Avg (3M)** = total sales in prior 3 months ÷ 3 &nbsp;|&nbsp; "
            f"3M window: {_3m_period} &nbsp;|&nbsp; "
            f"see the **3 Month Averages** tab for the full 3-month breakdown per salesman.",
            unsafe_allow_html=True,
        )

        st.download_button(
            "⬇ Download Summary CSV",
            t1.to_csv(index=False).encode("utf-8"),
            file_name=f"summary_{zid}_{cur_year}_{cur_month:02d}.csv",
            mime="text/csv",
            key="dl_ov_summary",
        )

        # ── Raw data export — pick a salesman, get their row as copy/paste text ──
        with st.expander("📋 Raw Data Export (copy/paste)", expanded=False):
            sel_export_sp = st.selectbox(
                "Select Salesman", t1["Salesman"].tolist(), key="tm_ov_export_sp"
            )
            if sel_export_sp:
                export_row = t1[t1["Salesman"] == sel_export_sp].iloc[0].to_dict()
                st.code(_format_unquoted_dict(export_row), language=None)

    # ── Prior 3 months — one expander each ───────────────────────────────────
    st.markdown("---")
    st.subheader("📆 Previous 3 Months")
    for _i in range(1, 4):
        _prior = mo_start_cur - pd.DateOffset(months=_i)
        _render_prior_month_section(
            df, returns_df, zid, int(_prior.year), int(_prior.month), holidays, collection_df
        )


# ── Prior-month salesman performance section ─────────────────────────────────

def _render_prior_month_section(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    zid,
    year: int,
    month: int,
    holidays: set,
    collection_df: pd.DataFrame = None,
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

    # ── Collection per salesman for this month ────────────────────────────────
    prior_coll_by_sp: dict = {}
    if collection_df is not None and not collection_df.empty and "value" in collection_df.columns:
        _c = collection_df.copy()
        _c["spid"]  = _c["spid"].astype(str)
        _c["year"]  = pd.to_numeric(_c["year"],  errors="coerce")
        _c["month"] = pd.to_numeric(_c["month"], errors="coerce")
        _mo_c = _c[(_c["year"] == year) & (_c["month"] == month)]
        prior_coll_by_sp = _mo_c.groupby("spid")["value"].sum().to_dict()

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

        coll       = round(float(prior_coll_by_sp.get(spid, 0.0)), 0)
        pct_coll   = round(coll / (1.02 * sales) * 100, 1) if sales > 0 else 0.0

        uc = int(sp_mo["cusid"].nunique())    if "cusid"    in sp_mo.columns else 0
        up = int(sp_mo["itemcode"].nunique()) if "itemcode" in sp_mo.columns else 0

        rows.append({
            "Salesman":                spname,
            "Target":                  target,
            "Sales":                   round(sales, 0),
            "Return":                  ret,
            "Net Sales":               net_sales,
            "% vs Target":             pct_tgt,
            "Collection":              coll,
            "% Collection":            pct_coll,
            "Products Sold":           up,
            "Customers Visited":       uc,
            "Days Left":               0,
            "Daily Required":          0,
            "Daily Avg":               daily_avg,
            "Monthly Avg (3M)":        monthly_avg_3m,
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
                    "Target":              "{:,.0f}",
                    "Sales":               "{:,.0f}",
                    "Return":              "{:,.0f}",
                    "Net Sales":           "{:,.0f}",
                    "% vs Target":         "{:.1f}%",
                    "Collection":          "{:,.0f}",
                    "% Collection":        "{:.1f}%",
                    "Products Sold":       "{:,.0f}",
                    "Customers Visited":   "{:,.0f}",
                    "Days Left":           "{:,.0f}",
                    "Daily Required":      "{:,.0f}",
                    "Daily Avg":           "{:,.0f}",
                    "Monthly Avg (3M)":    "{:,.0f}",
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
            if "% Collection" in df_inner.columns:
                styled = styled.apply(_col_pct, subset=["% Collection"])
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
            ret_by_sp = _r_mo.groupby(_r_mo["spid"].astype(str))["treturnamt"].sum().to_dict()

    # ── Collection for the selected month, per salesman ───────────────────────
    coll_by_sp: dict = {}
    if collection_df is not None and not collection_df.empty and "value" in collection_df.columns:
        _c = collection_df.copy()
        _c["spid"] = _c["spid"].astype(str)
        _c["year"] = pd.to_numeric(_c["year"], errors="coerce")
        _c["month"] = pd.to_numeric(_c["month"], errors="coerce")
        coll_by_sp = (
            _c[(_c["year"] == sel_year) & (_c["month"] == sel_month)]
            .groupby("spid")["value"].sum().to_dict()
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
        pct_coll = round(coll / (1.02 * sales) * 100, 1) if sales > 0 else None

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
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(t, use_container_width=True, hide_index=True)

    newest_asof = "as of today" if is_current else f"as of {mo_end_full.strftime('%b %d')}"
    st.caption(
        f"**Score** (0–100, sorted lowest first): 45% Sales vs Target + 45% Collection vs Sales "
        f"(both capped at 100%) + 5% Products Sold + 5% Customers Visited (both for {sel_label} only) "
        f"— the last two scored relative to the top salesman in this table — minus up to 20% in negative points, "
        f"also peer-relative: 6 pts for Return vs Sales %, 12 pts for the combined "
        f"{oldest_label} + {mid_label} balance, 2 pts for the {newest_label} balance. "
        f"Rows highlighted red have no target set for {sel_label} — scored 0 on that 45% component. "
        f"Balance columns are point-in-time snapshots: {newest_label} is {newest_asof}, "
        f"{mid_label} and {oldest_label} are as of their own month-end."
    )

    st.download_button(
        "⬇ Download Salesman Score CSV",
        t.to_csv(index=False).encode("utf-8"),
        file_name=f"salesman_score_{'consolidated' if is_consolidated else zid}_{sel_year}_{sel_month:02d}.csv",
        mime="text/csv",
        key="dl_salesman_score",
    )


# ── 3 Month Averages ──────────────────────────────────────────────────────────

def _render_three_month_averages(sales_df: pd.DataFrame, returns_df: pd.DataFrame, zid):
    """
    Per-salesman 3-month averages for the last 3 calendar months INCLUDING the
    current month (this month to-date + the 2 complete months before it):
    Daily/Monthly Avg Sales, Unique Customers/Products + their daily averages,
    and ZID-wide Unique Products for comparison.

    Also hosts Moving Average Analysis underneath (folded in from the former
    standalone "Moving Average" tab).
    """
    st.subheader("📊 3 Month Averages")
    today = pd.Timestamp.today().normalize()
    holidays = _get_holidays()

    mo_start_cur = pd.Timestamp(today.year, today.month, 1)
    mo_start_3mo = mo_start_cur - pd.DateOffset(months=2)
    end_3mo      = today
    wd_3mo       = _count_working_days(mo_start_3mo.date(), end_3mo.date(), holidays)

    if "date" not in sales_df.columns or "final_sales" not in sales_df.columns:
        st.warning("Required columns missing.")
        return

    df = sales_df.copy()
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["_d"]  = df["_dt"].dt.date

    last3 = df[(df["_dt"] >= mo_start_3mo) & (df["_dt"] <= end_3mo)]

    if last3.empty:
        st.warning(
            f"⚠️ No data found for the 3-month window "
            f"({mo_start_3mo.strftime('%b %Y')} – {end_3mo.strftime('%b %Y')}). "
            "Please load at least 3 months (incl. this one) in the sidebar filters."
        )
        return

    zid_up = int(last3["itemcode"].nunique()) if "itemcode" in last3.columns else 0

    sp_list = df[["spid", "spname"]].dropna().drop_duplicates().sort_values("spname")

    rows = []
    for _, sp_row in sp_list.iterrows():
        spid   = str(sp_row["spid"])
        spname = sp_row["spname"]
        sp3 = last3[last3["spid"].astype(str) == spid]

        total_3mo   = float(sp3["final_sales"].sum())
        daily_avg   = round(total_3mo / wd_3mo, 0) if wd_3mo > 0 else 0.0
        monthly_avg = round(total_3mo / 3, 0)

        uc_3mo   = int(sp3["cusid"].nunique())    if "cusid"    in sp3.columns else 0
        up_3mo   = int(sp3["itemcode"].nunique()) if "itemcode" in sp3.columns else 0
        daily_uc = round(float(sp3.groupby("_d")["cusid"].nunique().mean()),    1) if not sp3.empty and "cusid"    in sp3.columns else 0.0
        daily_up = round(float(sp3.groupby("_d")["itemcode"].nunique().mean()), 1) if not sp3.empty and "itemcode" in sp3.columns else 0.0

        rows.append({
            "Salesman ID":              spid,
            "Salesman":                 spname,
            "Daily Avg Sales (3M)":     daily_avg,
            "Monthly Avg Sales (3M)":   monthly_avg,
            "Unique Customers (3M)":    uc_3mo,
            "Avg Daily Customers (3M)": daily_uc,
            "Unique Products (3M)":     up_3mo,
            "Avg Daily Products (3M)":  daily_up,
            "ZID Unique Products (3M)": zid_up,
        })

    if not rows:
        st.info("No salesmen found for this selection.")
        return

    t = pd.DataFrame(rows).sort_values("Monthly Avg Sales (3M)", ascending=False).reset_index(drop=True)

    _3m_period = (f"{mo_start_3mo.strftime('%b %Y')} – {end_3mo.strftime('%b %Y')}"
                  f" ({wd_3mo} working days)")
    st.caption(f"3M window: {_3m_period}")

    try:
        styled = t.style.format({
            "Daily Avg Sales (3M)":     "{:,.0f}",
            "Monthly Avg Sales (3M)":   "{:,.0f}",
            "Unique Customers (3M)":    "{:,.0f}",
            "Avg Daily Customers (3M)": "{:.1f}",
            "Unique Products (3M)":     "{:,.0f}",
            "Avg Daily Products (3M)":  "{:.1f}",
            "ZID Unique Products (3M)": "{:,.0f}",
        }, na_rep="—")
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(t, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download 3 Month Averages CSV",
        t.to_csv(index=False).encode("utf-8"),
        file_name=f"three_month_averages_{zid}.csv",
        mime="text/csv",
        key="dl_3m_avg",
    )

    # ── Moving Average Analysis (folded in from the former standalone tab) ────
    st.markdown("---")
    st.subheader("📈 Moving Average Analysis")

    from datetime import date as _date
    ma_col1, ma_col2, ma_col3 = st.columns(3)
    with ma_col1:
        ma_entity = st.selectbox(
            "Entity", ["Salesman", "Product", "Product Group"], key="tm_ma_entity"
        )
    with ma_col2:
        ma_metric = st.selectbox(
            "Metric", ["Net Sales", "Net Returns"], key="tm_ma_metric"
        )
    with ma_col3:
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


# ── Collection Details tab — salesman + month picker, total shown outside ─────

def _render_collection_details_tab(
    sales_df: pd.DataFrame,
    zid,
    collection_df: pd.DataFrame,
    return_df: pd.DataFrame,
):
    """
    Voucher-level collection rows (RCT/CRCT/BRCT/STJV/JV--/ADJV) for one
    salesman and one month, plus a unified day-book table beneath it showing
    all transaction types (DOs, returns, collections) in separate amount columns.
    """
    st.subheader("🧾 SR Trn")
    today = pd.Timestamp.today().normalize()

    month_opts = ssc.month_choices(today)
    sel_label = st.selectbox(
        "Month", [m[0] for m in month_opts], index=0, key="tm_coll_details_month"
    )
    sel_year, sel_month = next((y, m) for (lbl, y, m) in month_opts if lbl == sel_label)

    sp_opts = _sp_opts(sales_df)
    sel_sp_raw = st.selectbox(
        "Salesman",
        [None] + sp_opts,
        format_func=lambda x: "— select a salesman —" if x is None else x,
        key="tm_coll_details_sp",
    )
    if not sel_sp_raw:
        st.info("👆 Select a salesman to view collection details.")
        return
    sel_spid = _codes([sel_sp_raw])[0]

    # ── Helper: normalise a dataframe's spid/year/month columns ──────────────
    def _prep(df):
        df = df.copy()
        df["spid"]  = df["spid"].astype(str)
        df["year"]  = pd.to_numeric(df["year"],  errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        return df

    def _sp_month_filter(df):
        return df[(df["spid"] == str(sel_spid)) & (df["year"] == sel_year) & (df["month"] == sel_month)]

    # ── Collections ───────────────────────────────────────────────────────────
    if collection_df is None or collection_df.empty or not {"spid","year","month","glvoucher","date","cusid","value"}.issubset(collection_df.columns):
        st.info("No collection data loaded.")
        coll_detail = pd.DataFrame()
    else:
        c = _prep(collection_df)
        coll_detail = _sp_month_filter(c).copy()

    total = float(coll_detail["value"].sum()) if not coll_detail.empty else 0.0
    st.metric(f"💰 Total Collection — {sel_label}", f"{total:,.0f}")

    if not coll_detail.empty:
        coll_detail["date"] = pd.to_datetime(coll_detail["date"], errors="coerce")
        coll_cols = ["date", "glvoucher", "cusid"] + (["cusname"] if "cusname" in coll_detail.columns else []) + ["value"]
        t = (
            coll_detail[coll_cols]
            .rename(columns={"date": "Date", "glvoucher": "Transaction Code",
                              "cusid": "Customer Code", "cusname": "Customer", "value": "Amount"})
            .sort_values("Date", ascending=False)
            .reset_index(drop=True)
        )
        st.caption(f"{len(t):,} collection transaction(s)")
        try:
            st.dataframe(
                t.style.format({"Date": "{:%Y-%m-%d}", "Amount": "{:,.0f}"}, na_rep="—"),
                use_container_width=True, hide_index=True,
            )
        except Exception:
            st.dataframe(t, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇ Download Collection CSV",
            t.to_csv(index=False).encode("utf-8"),
            file_name=f"collection_detail_{sel_spid}_{sel_year}_{sel_month:02d}.csv",
            mime="text/csv",
            key="dl_collection_details_tab",
        )
    else:
        st.info("No collection transactions for this salesman in this month.")

    # ── Unified day-book: Sales (DO) + Returns + Collections + Mobile Orders ──
    st.markdown("---")
    st.markdown("#### 📋 All Transactions")

    rows = []

    _COLS = ["Date","Voucher","Cust Code","Customer","Sales","Return","Collection","Mobile Order"]

    # Sales (DOs) — one row per voucher per customer
    if sales_df is not None and not sales_df.empty and {"spid","year","month","voucher","date","cusid","cusname","altsales"}.issubset(sales_df.columns):
        s = _prep(sales_df)
        s_filt = _sp_month_filter(s)
        if not s_filt.empty:
            s_grp = (
                s_filt.groupby(["date","voucher","cusid","cusname"], as_index=False)
                      .agg(Sales=("altsales","sum"))
            )
            s_grp["date"] = pd.to_datetime(s_grp["date"], errors="coerce")
            s_grp = s_grp.rename(columns={"voucher":"Voucher","date":"Date","cusid":"Cust Code","cusname":"Customer"})
            s_grp["Return"]       = None
            s_grp["Collection"]   = None
            s_grp["Mobile Order"] = None
            rows.append(s_grp[_COLS])

    # Returns — one row per return voucher per customer
    if return_df is not None and not return_df.empty and {"spid","year","month","revoucher","date","cusid","cusname","treturnamt"}.issubset(return_df.columns):
        r = _prep(return_df)
        r_filt = _sp_month_filter(r)
        if not r_filt.empty:
            r_grp = (
                r_filt.groupby(["date","revoucher","cusid","cusname"], as_index=False)
                      .agg(Return=("treturnamt","sum"))
            )
            r_grp["date"] = pd.to_datetime(r_grp["date"], errors="coerce")
            r_grp = r_grp.rename(columns={"revoucher":"Voucher","date":"Date","cusid":"Cust Code","cusname":"Customer"})
            r_grp["Sales"]        = None
            r_grp["Collection"]   = None
            r_grp["Mobile Order"] = None
            rows.append(r_grp[_COLS])

    # Collections
    if not coll_detail.empty:
        has_cusname = "cusname" in coll_detail.columns
        c_cols = ["date","glvoucher","cusid"] + (["cusname"] if has_cusname else [])
        c_txn = coll_detail[c_cols + ["value"]].copy()
        if not has_cusname:
            c_txn["cusname"] = None
        c_txn = c_txn.rename(columns={"date":"Date","glvoucher":"Voucher","cusid":"Cust Code","cusname":"Customer","value":"Collection"})
        c_txn["Sales"]        = None
        c_txn["Return"]       = None
        c_txn["Mobile Order"] = None
        rows.append(c_txn[_COLS])

    # Mobile orders (opmob) — one row per order number per customer
    mob_df = _load_opmob_all(str(zid))
    if not mob_df.empty and {"spid","year","month","mob_voucher","date","cusid","cusname","mob_total"}.issubset(mob_df.columns):
        m = _prep(mob_df)
        m_filt = _sp_month_filter(m)
        if not m_filt.empty:
            m_filt = m_filt.copy()
            m_filt["date"] = pd.to_datetime(m_filt["date"], errors="coerce")
            m_filt = m_filt.rename(columns={"mob_voucher":"Voucher","date":"Date","cusid":"Cust Code","cusname":"Customer","mob_total":"Mobile Order"})
            m_filt["Sales"]      = None
            m_filt["Return"]     = None
            m_filt["Collection"] = None
            rows.append(m_filt[_COLS])

    if not rows:
        st.info("No transactions found for this salesman and month.")
        return

    txn = pd.concat(rows, ignore_index=True)
    txn["Date"] = pd.to_datetime(txn["Date"], errors="coerce")
    txn = txn.sort_values(["Date","Voucher"]).reset_index(drop=True)

    st.caption(f"{len(txn):,} transaction row(s) — DOs, Returns, Collections, Mobile Orders")
    try:
        st.dataframe(
            txn.style.format(
                {"Date": "{:%Y-%m-%d}", "Sales": "{:,.0f}", "Return": "{:,.0f}",
                 "Collection": "{:,.0f}", "Mobile Order": "{:,.0f}"},
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.dataframe(txn, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download All Transactions CSV",
        txn.to_csv(index=False).encode("utf-8"),
        file_name=f"all_txn_{sel_spid}_{sel_year}_{sel_month:02d}.csv",
        mime="text/csv",
        key="dl_all_txn_tab",
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


# ── Next Month Target ──────────────────────────────────────────────────────────

_SALES_ONLY_NMT_ZIDS = ("100000", "100005")  # GI, Zepto — no shipment/stock tracking in this tool


def _render_next_month_target(zid):
    st.subheader("🔮 Next Month Target Estimate")
    st.caption("📝 Methodology — see Note [1] at the bottom of the page.")
    sales_only = str(zid) in _SALES_ONLY_NMT_ZIDS
    note3 = None  # only set on the stock-capped path (shipment picker below)

    if sales_only:
        note1 = (
            "Estimates next month's sales range purely from each item's historical net sales "
            "(qty/amt, after returns) over the trailing 12 completed months — low/median/high are "
            "the lowest non-zero, median (non-zero), and highest active months. This business has "
            "no stock or shipment tracking wired into this tool, so the estimate is **not** capped "
            "by stock on hand."
        )
    else:
        note1 = (
            "Estimates an achievable sales range for next month from current stock plus an optional "
            "incoming shipment, bounded by each item's lowest non-zero, median (non-zero), and "
            "highest net sales (qty/amt, after returns) over the trailing 12 completed months. "
            "Months with zero net sales are ignored when picking the 'low'/'median'/average figures. "
            "Current stock is a snapshot as of today, so it's first projected forward to next "
            "month's start by subtracting the expected sales for the rest of *this* month "
            "(trailing-12-month average active-month qty ÷ days in this month × days remaining)."
        )

    today = pd.Timestamp.today().normalize()
    hist_start, hist_end = nmt.trailing_12mo_window(today)
    next_month_start = pd.Timestamp(today.year, today.month, 1) + pd.DateOffset(months=1)
    next_month_end = next_month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
    years = tuple(sorted({int(hist_start.year), int(hist_end.year)}))

    with st.spinner("Loading trailing 12-month sales history…"):
        sales_12mo = _load_sales_window(str(zid), years)
        returns_12mo = _load_returns_window(str(zid), years)

    monthly_net = nmt.compute_item_monthly_net(sales_12mo, returns_12mo, hist_start, hist_end)
    low_high = nmt.compute_item_low_high(monthly_net)

    st.markdown(
        f"**Historical window:** {hist_start.strftime('%b %Y')} – {hist_end.strftime('%b %Y')} "
        f"(12 completed months) &nbsp;|&nbsp; **Target month:** {next_month_start.strftime('%b %Y')}"
    )

    if sales_only:
        item_names = nmt.extract_item_names(sales_12mo)
        estimate_df = nmt.build_sales_only_estimate(low_high, item_names)
    else:
        with st.spinner("Loading stock and open shipments…"):
            stock_df = _load_final_items(str(zid))
            purchase_df = _load_purchase_open_combined()
        open_shipments = nmt.get_open_shipments(purchase_df)

        col1, col2 = st.columns(2)
        with col1:
            option_labels = {
                f"{row.shipmentname}  (ZID {row.zid})": (row.zid, row.shipmentname)
                for row in open_shipments.itertuples()
            }
            note3 = (
                "Select both the 100001 and 100009 entries if they're the same physical "
                "shipment split across books."
            )
            sel_labels = st.multiselect(
                "Incoming Shipment(s) (open, not yet received) — see Note [2] below",
                options=list(option_labels.keys()),
                key=f"nmt_shipments_{zid}",
            )
        with col2:
            entry_date = st.date_input(
                "Estimated Date of Entry (applies to all selected shipments)",
                value=next_month_start.date(), key=f"nmt_entry_date_{zid}",
            )

        sel_selections = [option_labels[lbl] for lbl in sel_labels]
        shipment_items = None
        incoming_fraction = 1.0
        if sel_selections:
            names_str = ", ".join(f"{name} (ZID {z})" for z, name in sel_selections)
            entry_ts = pd.Timestamp(entry_date)
            if entry_ts <= next_month_end:
                shipment_items = nmt.get_shipment_items(purchase_df, sel_selections)
                incoming_fraction, days_remaining, days_in_month = nmt.incoming_proration(
                    entry_ts, next_month_start, next_month_end
                )
                if incoming_fraction >= 1.0:
                    st.success(
                        f"Including shipment(s) **{names_str}** "
                        f"({len(shipment_items)} item(s) after cross-ZID code mapping) — "
                        f"arrives {entry_date}, on/before {next_month_start.strftime('%b %Y')} starts "
                        f"— full month available."
                    )
                else:
                    st.success(
                        f"Including shipment(s) **{names_str}** "
                        f"({len(shipment_items)} item(s) after cross-ZID code mapping) — "
                        f"arrives {entry_date}, leaving {days_remaining} of {days_in_month} days in "
                        f"{next_month_start.strftime('%b %Y')} ({incoming_fraction:.0%}). Incoming "
                        f"quantities are scaled down to that fraction before capping the estimate."
                    )
            else:
                st.warning(
                    f"Selected shipment(s) **{names_str}** are estimated to arrive {entry_date}, "
                    f"after {next_month_start.strftime('%b %Y')} ends — excluded from this estimate."
                )

        estimate_df = nmt.build_next_month_estimate(
            stock_df, low_high, shipment_items, incoming_fraction, today=today
        )

    if estimate_df.empty:
        if sales_only:
            st.info("No items with sales history were found in the trailing 12-month window.")
        else:
            st.info("No items with available stock (current + incoming) and sales history were found.")
        return

    total_low = float(estimate_df["est_low_amt"].sum())
    total_median = float(estimate_df["est_median_amt"].sum())
    total_high = float(estimate_df["est_high_amt"].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimated Low", f"{total_low:,.0f}")
    m2.metric("Estimated Median", f"{total_median:,.0f}")
    m3.metric("Estimated High", f"{total_high:,.0f}")
    m4.metric("Items Counted", f"{len(estimate_df):,}")

    _col_map = {
        "itemcode": "Item Code", "itemname": "Item Name", "itemgroup": "Item Group",
        "current_stock": "Current Stock", "projected_stock": "Projected Stock (Month Start)",
        "incoming_qty": "Incoming Qty", "available_qty": "Available Qty",
        "low_qty": "12M Low Qty", "median_qty": "12M Median Qty", "high_qty": "12M High Qty",
        "avg_qty": "12M Avg Qty (active)", "avg_price": "Avg Price",
        "est_low_qty": "Est. Low Qty", "est_median_qty": "Est. Median Qty", "est_high_qty": "Est. High Qty",
        "est_low_amt": "Est. Low Amt", "est_median_amt": "Est. Median Amt", "est_high_amt": "Est. High Amt",
    }
    disp = estimate_df.rename(columns=_col_map)
    cols_order = [c for c in _col_map.values() if c in disp.columns]
    disp = disp[cols_order]

    st.dataframe(
        disp.style.format({
            "Current Stock": "{:,.0f}", "Projected Stock (Month Start)": "{:,.0f}",
            "Incoming Qty": "{:,.0f}", "Available Qty": "{:,.0f}",
            "12M Low Qty": "{:,.0f}", "12M Median Qty": "{:,.0f}", "12M High Qty": "{:,.0f}",
            "12M Avg Qty (active)": "{:,.2f}", "Avg Price": "{:,.2f}",
            "Est. Low Qty": "{:,.0f}", "Est. Median Qty": "{:,.0f}", "Est. High Qty": "{:,.0f}",
            "Est. Low Amt": "{:,.0f}", "Est. Median Amt": "{:,.0f}", "Est. High Amt": "{:,.0f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    # ── Salesman × Area performance ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("👤 Salesman × Area Performance")

    st.caption("📝 Methodology — see Note [3] at the bottom of the page.")

    salesman_capped = None  # only populated (and used) on the stock-capped path
    if sales_only:
        note2 = (
            "For each salesman, accumulates net sales amount (after returns) across every product "
            "they sold within each area, then takes the lowest non-zero, median (non-zero), and "
            "highest month from that combined monthly total over the same trailing 12-month window."
        )
        sp_area_monthly = nmt.compute_salesman_area_monthly_net(sales_12mo, returns_12mo, hist_start, hist_end)
        sp_area_lh = nmt.compute_salesman_area_low_high(sp_area_monthly)
        sp_area_lh = sp_area_lh[sp_area_lh["high_amt"] > 0].copy()
    else:
        note2 = (
            "Each item's already stock-capped Low/Median/High estimate (the same figures shown in "
            "the table above) is split across the salesmen who sell it by their historical share of "
            "that item's net qty — so combined salesman totals reconstruct, never exceed, the product "
            "table's own totals. Areas split a salesman's allocation by their historical mix across areas."
        )
        item_spid_monthly = nmt.compute_item_spid_monthly_net(sales_12mo, returns_12mo, hist_start, hist_end)
        item_spid_lh = nmt.compute_item_spid_low_high(item_spid_monthly)
        salesman_capped = nmt.allocate_item_estimate_by_salesman(item_spid_lh, estimate_df)

        item_spid_area_totals = nmt.compute_item_spid_area_totals(sales_12mo, returns_12mo, hist_start, hist_end)
        sp_area_lh = nmt.allocate_salesman_amt_by_area(salesman_capped, item_spid_area_totals)
        sp_area_lh = sp_area_lh[sp_area_lh["high_amt"] > 0].copy()

    if sp_area_lh.empty:
        st.info("No salesman sales history found in the trailing 12-month window.")
    else:
        sp_names = nmt.extract_salesman_names(sales_12mo)
        sp_area_disp = sp_area_lh.merge(sp_names, on="spid", how="left")
        sp_area_disp["spname"] = sp_area_disp["spname"].fillna("")
        sp_area_disp = sp_area_disp.sort_values("high_amt", ascending=False).reset_index(drop=True)

        sa1, sa2, sa3 = st.columns(3)
        sa1.metric("Total Low", f"{sp_area_disp['low_amt'].sum():,.0f}")
        sa2.metric("Total Median", f"{sp_area_disp['median_amt'].sum():,.0f}")
        sa3.metric("Total High", f"{sp_area_disp['high_amt'].sum():,.0f}")

        sp_area_col_map = {
            "spid": "Salesman Code", "spname": "Salesman Name", "area": "Area",
            "low_amt": "Low Amount", "median_amt": "Median Amount", "high_amt": "High Amount",
        }
        sp_area_view = sp_area_disp.rename(columns=sp_area_col_map)
        sp_area_view = sp_area_view[[c for c in sp_area_col_map.values() if c in sp_area_view.columns]]

        st.dataframe(
            sp_area_view.style.format({
                "Low Amount": "{:,.0f}", "Median Amount": "{:,.0f}", "High Amount": "{:,.0f}",
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )

        # ── Per-salesman product breakdown ─────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Salesman Product Breakdown")

        sp_names_active = sp_names[sp_names["spid"].isin(sp_area_disp["spid"])].copy()
        sp_options_map = {
            f"{row.spname} ({row.spid})": row.spid
            for row in sp_names_active.sort_values("spname").itertuples()
        }
        sel_sp_label = st.selectbox(
            "Select Salesman", ["— Select —"] + list(sp_options_map.keys()), key=f"nmt_sp_drilldown_{zid}"
        )

        if sel_sp_label != "— Select —":
            sel_spid = sp_options_map[sel_sp_label]

            if sales_only:
                sales_for_sp = (
                    sales_12mo[sales_12mo["spid"].astype(str) == sel_spid]
                    if sales_12mo is not None and not sales_12mo.empty and "spid" in sales_12mo.columns
                    else pd.DataFrame()
                )
                returns_for_sp = (
                    returns_12mo[returns_12mo["spid"].astype(str) == sel_spid]
                    if returns_12mo is not None and not returns_12mo.empty and "spid" in returns_12mo.columns
                    else pd.DataFrame()
                )
                monthly_net_sp = nmt.compute_item_monthly_net(sales_for_sp, returns_for_sp, hist_start, hist_end)
                low_high_sp = nmt.compute_item_low_high(monthly_net_sp)
                item_names_sp = nmt.extract_item_names(sales_for_sp)
                drill_df = nmt.build_sales_only_estimate(low_high_sp, item_names_sp)
            else:
                # Reuse the same stock-capped allocation as the area table above —
                # this salesman's own low/median/high per item, already capped at
                # their fair share of each item's available stock.
                drill_df = salesman_capped[salesman_capped["spid"] == sel_spid].copy()
                drill_df = drill_df[drill_df["est_high_qty"] > 0].copy()
                if not drill_df.empty:
                    item_names_sp = nmt.extract_item_names(sales_12mo)
                    drill_df = drill_df.merge(item_names_sp, on="itemcode", how="left")
                    drill_df["itemname"] = drill_df["itemname"].fillna("")
                    drill_df["itemgroup"] = drill_df["itemgroup"].fillna("")
                    drill_df = drill_df.sort_values("est_high_amt", ascending=False).reset_index(drop=True)

            if drill_df.empty:
                st.info("No product sales history found for this salesman in the trailing 12-month window.")
            else:
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Estimated Low", f"{drill_df['est_low_amt'].sum():,.0f}")
                d2.metric("Estimated Median", f"{drill_df['est_median_amt'].sum():,.0f}")
                d3.metric("Estimated High", f"{drill_df['est_high_amt'].sum():,.0f}")
                d4.metric("Items Counted", f"{len(drill_df):,}")

                drill_col_map = {
                    "itemcode": "Item Code", "itemname": "Item Name", "itemgroup": "Item Group",
                    "low_qty": "12M Low Qty", "median_qty": "12M Median Qty", "high_qty": "12M High Qty",
                    "avg_price": "Avg Price",
                    "est_low_amt": "Est. Low Amt", "est_median_amt": "Est. Median Amt",
                    "est_high_amt": "Est. High Amt",
                }
                drill_disp = drill_df.rename(columns=drill_col_map)
                drill_disp = drill_disp[[c for c in drill_col_map.values() if c in drill_disp.columns]]

                st.dataframe(
                    drill_disp.style.format({
                        "12M Low Qty": "{:,.0f}", "12M Median Qty": "{:,.0f}", "12M High Qty": "{:,.0f}",
                        "Avg Price": "{:,.2f}",
                        "Est. Low Amt": "{:,.0f}", "Est. Median Amt": "{:,.0f}", "Est. High Amt": "{:,.0f}",
                    }, na_rep="—"),
                    use_container_width=True, hide_index=True,
                )

    # ── Notes ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 Notes")
    st.markdown(f"**[1] Next Month Target Estimate** — {note1}")
    if note3:
        st.markdown(f"**[2] Incoming Shipment(s)** — {note3}")
    st.markdown(f"**[3] Salesman × Area Performance** — {note2}")


# ── Field Tracking ────────────────────────────────────────────────────────────

_TRACK_COLORS = [
    [0,   116, 217],
    [185,  66, 252],
    [255,  65,  54],
    [46,  204,  64],
    [1,   255, 112],
]
_ORDER_COLOR   = [255, 165,   0]
_CHECKIN_COLOR = [  0, 200, 100]


@st.cache_data(show_spinner=False, ttl=600)
def _load_tracking_salesmen(zid: str) -> pd.DataFrame:
    from core.db import get_dataframe
    from core import queries
    sql, params = queries.get_field_tracking_salesmen(int(zid))
    df = get_dataframe(sql, params)
    return df if df is not None else pd.DataFrame()


def _render_field_tracking(zid):
    try:
        import pydeck as pdk
    except ImportError:
        st.error("pydeck is not installed. Run: pip install pydeck")
        return

    st.subheader("🗺️ Field Tracking")

    sp_df = _load_tracking_salesmen(str(zid))
    if sp_df.empty:
        st.info("No salesmen with location records found for this entity.")
        return

    sp_labels = (sp_df["username"] + " — " + sp_df["display_name"]).tolist()
    sp_map    = dict(zip(sp_labels, sp_df["username"].tolist()))

    col1, col2 = st.columns([3, 1])
    with col1:
        sel_labels = st.multiselect("Salesman", sp_labels, default=sp_labels[:1], key="ft_sp")
    with col2:
        sel_date = st.date_input("Date", value=pd.Timestamp.today().date(), key="ft_date")

    if not sel_labels:
        st.info("Select at least one salesman.")
        return

    from core.db import get_dataframe
    from core import queries

    date_str   = str(sel_date)
    path_data  = []
    point_data = []
    order_data = []
    all_coords = []
    stats      = []

    for i, label in enumerate(sel_labels):
        username  = sp_map[label]
        sp_name   = label.split(" — ", 1)[-1]
        color     = _TRACK_COLORS[i % len(_TRACK_COLORS)]

        # ── GPS track ─────────────────────────────────────────────────────────
        sql, params = queries.get_location_track(username, date_str)
        track_df    = get_dataframe(sql, params)

        n_pings = 0
        if track_df is not None and not track_df.empty:
            coords = list(zip(
                track_df["longitude"].astype(float),
                track_df["latitude"].astype(float),
            ))
            all_coords.extend(coords)
            n_pings = len(coords)

            path_data.append({"path": coords, "color": color})

            for _, row in track_df.iterrows():
                ts_val  = row.get("ts")
                ts_str  = pd.to_datetime(ts_val).strftime("%H:%M") if pd.notna(ts_val) else ""
                addr    = str(row.get("formatted_address") or "").strip()
                is_ci   = bool(row.get("is_check_in"))
                is_mock = bool(row.get("is_mock_location"))
                tip     = f"{sp_name}  {ts_str}"
                if is_ci:
                    tip += "  [CHECK-IN]"
                if is_mock:
                    tip += "  ⚠ MOCK"
                if addr:
                    tip += f"\n{addr}"
                point_data.append({
                    "coordinates": [float(row["longitude"]), float(row["latitude"])],
                    "color":  _CHECKIN_COLOR if is_ci else color,
                    "radius": 35 if is_ci else 18,
                    "tooltip": tip,
                })

        # ── Order locations ───────────────────────────────────────────────────
        sql2, params2 = queries.get_opmob_order_locations(int(zid), username, date_str)
        ord_df        = get_dataframe(sql2, params2)

        n_orders = 0
        if ord_df is not None and not ord_df.empty:
            for _, row in ord_df.iterrows():
                lat = float(row["lat"] or 0)
                lon = float(row["lon"] or 0)
                if lat == 0 and lon == 0:
                    continue
                all_coords.append((lon, lat))
                n_orders += 1
                order_data.append({
                    "coordinates": [lon, lat],
                    "color":  _ORDER_COLOR,
                    "radius": 45,
                    "tooltip": (
                        f"Order: {row['order_num']}\n"
                        f"Customer: {row['cusname']}\n"
                        f"Status: {row['status']}\n"
                        f"Total: {int(row['total'] or 0):,}"
                    ),
                })

        stats.append(f"**{sp_name}**: {n_pings} pings · {n_orders} orders")

    if not all_coords:
        st.info(f"No location data for {sel_date.strftime('%d %b %Y')}.")
        return

    # ── Auto-center + zoom ────────────────────────────────────────────────────
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    span = max(max(lats) - min(lats), max(lons) - min(lons))
    zoom = 14 if span < 0.01 else (12 if span < 0.05 else (10 if span < 0.2 else 8))

    # ── pydeck layers ─────────────────────────────────────────────────────────
    layers = []
    if path_data:
        layers.append(pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_color="color",
            width_min_pixels=3,
            width_max_pixels=5,
            pickable=False,
        ))
    if point_data:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=point_data,
            get_position="coordinates",
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
            opacity=0.85,
        ))
    if order_data:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=order_data,
            get_position="coordinates",
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            auto_highlight=True,
            opacity=0.9,
        ))

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=(min(lats) + max(lats)) / 2,
            longitude=(min(lons) + max(lons)) / 2,
            zoom=zoom,
            pitch=0,
        ),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"text": "{tooltip}"},
    )
    st.pydeck_chart(deck, use_container_width=True)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = []
    for i, label in enumerate(sel_labels):
        c = _TRACK_COLORS[i % len(_TRACK_COLORS)]
        hex_c = "#{:02x}{:02x}{:02x}".format(*c)
        legend_items.append(f"<span style='color:{hex_c}'>●</span> {label.split(' — ',1)[-1]}")
    legend_items.append("<span style='color:#ffa500'>●</span> Order location")
    legend_items.append("<span style='color:#00c864'>●</span> Check-in")
    st.markdown("  &nbsp;·&nbsp;  ".join(legend_items), unsafe_allow_html=True)
    st.caption("  ·  ".join(stats))


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
        ["👤 Individual Salesman", "📊 All Salesmen Overview", "🎯 Salesman Score",
         "📊 3 Month Averages", "🧾 SR Trn",
         "📦 Current Stock", "🔮 Next Month Target", "🗺️ Field Tracking"],
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
        _render_overview(sales_df, returns_df, opmob_all, zid,
                         collection_df=data_dict.get("collection", pd.DataFrame()))
        return

    if _view_mode == "🎯 Salesman Score":
        _render_salesman_score(sales_df, returns_df, zid,
                                collection_df=data_dict.get("collection", pd.DataFrame()))
        return

    if _view_mode == "📊 3 Month Averages":
        _render_three_month_averages(sales_df, returns_df, zid)
        return

    if _view_mode == "🧾 SR Trn":
        _render_collection_details_tab(
            sales_df,
            zid,
            data_dict.get("collection", pd.DataFrame()),
            returns_df if returns_df is not None else pd.DataFrame(),
        )
        return

    if _view_mode == "📦 Current Stock":
        st.subheader("📦 Current Stock")
        with st.spinner("Loading stock data…"):
            stock_df  = _load_final_items(str(zid))
            wh_df     = _load_opspprc(str(zid))

        if stock_df.empty:
            st.warning("No stock data available from final_items_view for this entity.")
        else:
            # Merge wholesale price info if available
            if not wh_df.empty:
                wh_df = wh_df[["item_id", "wh_qty", "wh_price"]].copy()
                wh_df["item_id"] = wh_df["item_id"].astype(str)
                stock_df["item_id"] = stock_df["item_id"].astype(str)
                stock_df = stock_df.merge(wh_df, on="item_id", how="left")

            _col_map = {
                "item_id":    "Item ID",
                "item_name":  "Item Name",
                "item_group": "Item Group",
                "stock":      "Stock",
                "wh_qty":     "WH Qty",
                "wh_price":   "WH Price",
            }
            disp = (
                stock_df
                .rename(columns=_col_map)
                [[c for c in _col_map.values() if c in stock_df.rename(columns=_col_map).columns]]
                .reset_index(drop=True)
            )

            # Search filter
            _search = st.text_input("🔍 Search by Item Name or Group", key="tm_stock_search")
            if _search:
                _mask = (
                    disp["Item Name"].str.contains(_search, case=False, na=False) |
                    disp["Item Group"].str.contains(_search, case=False, na=False)
                )
                disp = disp[_mask].reset_index(drop=True)

            st.caption(f"{len(disp):,} items  —  WH Qty = minimum qty for wholesale price; WH Price = std price minus discount")
            fmt = {"Stock": "{:,.0f}"}
            if "WH Qty" in disp.columns:
                fmt["WH Qty"]   = "{:,.0f}"
                fmt["WH Price"] = "{:,.2f}"
            st.dataframe(
                disp.style.format(fmt, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "⬇ Download CSV",
                disp.to_csv(index=False).encode("utf-8"),
                file_name=f"current_stock_{zid}.csv",
                mime="text/csv",
                key="dl_current_stock",
            )
        return

    if _view_mode == "🔮 Next Month Target":
        _render_next_month_target(zid)
        return

    if _view_mode == "🗺️ Field Tracking":
        _render_field_tracking(zid)
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
    collection_df_all = data_dict.get("collection", pd.DataFrame())
    _render_metric_cards(
        f_sp, opmob_df, sel_spid, zid, sp_returns=f_sp_ret,
        collection_df=collection_df_all,
    )

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

    id_raw      = ["spid", "spname", "cusid", "cusname", "cusmobile", "whatsapp", "area"]
    month_col_list = [c for c in pivot.columns if c not in id_raw and c != "Total"]

    rename_map = {
        "spname":    "Salesman",
        "cusid":     "Cust. Code",
        "cusname":   "Customer",
        "cusmobile": "Mobile",
        "whatsapp":  "WhatsApp Number",
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
            "whatsapp":  "WhatsApp Number",
            "area":      "Area",
            "itemcode":  "Item Code",
            "itemname":  "Item",
        }
        prod_id_raw = ["spname", "cusid", "cusname", "cusmobile", "whatsapp", "area", "itemcode", "itemname"]
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
                .rename(columns={"cusname": "Customer", "cusmobile": "Mobile",
                                  "whatsapp": "WhatsApp Number", "area": "Area"})
                [["Customer", "Mobile", "WhatsApp Number", "Area"]]
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

    # ── Daily Breakdown for selected salesman (collapsed by default) ──────────
    st.markdown("---")
    with st.expander("📅 Daily Breakdown — Current Month", expanded=False):
        _render_sp_daily_breakdown(f_sp, opmob_df, sel_spid, zid)

    # ── Buying Pattern Analysis — commented out for now ───────────────────────
    # try:
    #     bp_df = bp.compute_buying_pattern(
    #         pivot_df   = pivot,
    #         sales_df   = f_final,
    #         id_cols    = [c for c in id_raw if c in pivot.columns],
    #         month_cols = month_col_list,
    #     )
    # except Exception as e:
    #     bp_df = pd.DataFrame()
    #     st.caption(f"Buying pattern error: {e}")
    #
    # _render_buying_pattern(bp_df, is_any_filter=bool(sel_spid))

    # ── Inventory Coverage (collapsed; loads only on demand — can be slow) ─────
    st.markdown("---")
    with st.expander("🗂️ Inventory Coverage — This Month vs Prior-Month Stock", expanded=False):
        gen_key = f"tm_inv_cov_ready_{sel_spid}"
        if st.button("▶ Generate Inventory Coverage", key=f"tm_gen_inv_cov_{sel_spid}"):
            st.session_state[gen_key] = True
        if st.session_state.get(gen_key):
            _render_inventory_coverage(f_sp, str(zid))
        else:
            st.info("Click **Generate Inventory Coverage** above to load this section — it can take a while.")
