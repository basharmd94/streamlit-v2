from __future__ import annotations

import calendar
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from processing import salesman_due as sd
from views.marketing import _load_final_items  # noqa: F401 — re-exported; avoids duplicate cache




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


