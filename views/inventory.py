import streamlit as st
import pandas as pd
import numpy as np
import calendar
from core.analytics import Analytics
from processing import common
from utils.utils import timed


def _effective_zids(primary_zid: str) -> list[str]:
    """
    If 100001 is chosen, auto-include 100009 (packaging) as well.
    Otherwise, just use the primary zid.
    """
    p = str(primary_zid)
    return [p, "100009"] if p == "100001" else [p]

@st.cache_data(show_spinner=False)
def _load_stock_flow(zid: str) -> pd.DataFrame:
    zids = _effective_zids(zid)  # keep your 100001 → also 100009 rule
    frames = []
    for z in zids:
        try:
            df = Analytics("stock_flow", zid=z, filters={"zid": (str(z),)}).data
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df.assign(_src_zid=str(z)))
        except Exception as e:
            st.error(f"Error loading stock_flow for zid={z}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_product_inventory(zid: str) -> pd.DataFrame:
    """
    Loads monthly product-level inventory transactions from `stock` joined to `caitem`.
    Your SQL now uses: WHERE stock.zid = (%s)
    We therefore call once per effective zid and concatenate.
    The SQL already returns itemcode with packcode CASE logic applied.
    """
    zids = _effective_zids(zid)
    frames = []
    for z in zids:
        try:
            # IMPORTANT: Single parameter tuple (z,) to match "= (%s)"
            df = Analytics("stock", zid=z, filters={"zid": (str(z),)}).data
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df.assign(_src_zid=str(z)))
        except Exception as e:
            st.error(f"Error loading product inventory for zid={z}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_inventory_value(zid: str) -> pd.DataFrame:
    """
    Loads warehouse-level monthly stock value snapshots from `stock_value`.
    The SQL uses: WHERE zid = (%s)
    We will mirror the product logic and include 100009 when the primary zid is 100001,
    so that packaging warehouses are reflected in warehouse snapshot/value too.
    """
    zids = _effective_zids(zid)
    frames = []
    for z in zids:
        try:
            df = Analytics("stock_value", zid=z, filters={"zid": (str(z),)}).data
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df.assign(_src_zid=str(z)))
        except Exception as e:
            st.error(f"Error loading stock_value for zid={z}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

@timed
def display_inventory_analysis_main(current_page, zid: str):
    st.title("Inventory Analysis")

    inv_df = _load_product_inventory(zid)
    val_df = _load_inventory_value(zid)

    if inv_df is None or inv_df.empty:
        st.info("No product inventory data found for the selected company (zid).")
        return

    # --- Normalize dtypes ---
    if "zid" in inv_df.columns:
        inv_df["zid"] = inv_df["zid"].astype(str)
    for col in ["year", "month"]:
        if col in inv_df.columns:
            inv_df[col] = pd.to_numeric(inv_df[col], errors="coerce").astype("Int64")
    for c in ["warehouse", "itemgroup", "itemcode", "itemname"]:
        if c in inv_df.columns:
            inv_df[c] = inv_df[c].astype(str)
    if "stockqty" in inv_df.columns:
        inv_df["stockqty"] = pd.to_numeric(inv_df["stockqty"], errors="coerce").fillna(0.0)
    if "stockvalue" in inv_df.columns:
        inv_df["stockvalue"] = pd.to_numeric(inv_df["stockvalue"], errors="coerce").fillna(0.0)

    # Helper year-month key for comparisons
    inv_df["ym"] = inv_df["year"].fillna(0).astype(int) * 100 + inv_df["month"].fillna(0).astype(int)
    inv_df["year_month"] = inv_df.apply(
        lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}"
        if pd.notna(r['year']) and pd.notna(r['month']) else None, axis=1
    )

    # --- Build filter options from data ---
    years = sorted(inv_df["year"].dropna().astype(int).unique().tolist())
    months = list(range(1, 13))
    warehouses = sorted(inv_df["warehouse"].dropna().unique().tolist())
    itemgroups = sorted(inv_df["itemgroup"].dropna().unique().tolist())

    # Product selector label: "code — name"
    inv_df["product_label"] = inv_df.apply(
        lambda r: f"{r['itemcode']} — {r['itemname']}" if pd.notna(r.get("itemname")) else str(r['itemcode']),
        axis=1
    )
    products = (
        inv_df[["itemcode", "product_label"]]
        .drop_duplicates()
        .sort_values("itemcode")["product_label"]
        .tolist()
    )

    # --- UI: Filters ---
    st.subheader("Filters")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])

    with c1:
        year_sel = st.selectbox("Year (cutoff)", years, index=len(years) - 1 if years else 0, key="inv_year")
    with c2:
        month_sel = st.selectbox(
            "Month (cutoff)", months,
            index=(pd.Timestamp.today().month - 1),
            format_func=lambda m: calendar.month_abbr[m],
            key="inv_month"
        )
    with c3:
        use_latest = st.toggle(
            "Use Max Year–Month", value=False,
            help="If ON, uses the latest available period in data (considering auto-included packaging zid when applicable)."
        )
    with c4:
        st.caption("🔎 Item code already applies packcode CASE logic in SQL (no extra toggle needed).")

    wh_sel = st.multiselect("Warehouse(s)", options=warehouses, default=warehouses)
    ig_sel = st.multiselect("Item Group(s)", options=itemgroups, default=itemgroups)

    prod_sel_labels = st.multiselect("Product(s)", options=products, default=[],
                                     placeholder="Type to search code/name…")
    label_to_code = dict(inv_df[["product_label", "itemcode"]].drop_duplicates().values.tolist())
    prod_codes = [label_to_code.get(lbl) for lbl in prod_sel_labels]

    # Determine cutoff based on toggle (on merged dataset)
    if use_latest:
        latest_row = inv_df.loc[inv_df["ym"].idxmax()]
        cutoff_year, cutoff_month = int(latest_row["year"]), int(latest_row["month"])
    else:
        cutoff_year, cutoff_month = int(year_sel), int(month_sel)
    cutoff_ym = cutoff_year * 100 + cutoff_month
    st.markdown(f"**Cutoff:** {cutoff_year}-{cutoff_month:02d}")

    # Apply filters
    df_f = inv_df.copy()
    if wh_sel:
        df_f = df_f[df_f["warehouse"].isin(wh_sel)]
    if ig_sel:
        df_f = df_f[df_f["itemgroup"].isin(ig_sel)]
    if prod_codes:
        df_f = df_f[df_f["itemcode"].isin(prod_codes)]
    df_f = df_f[df_f["ym"] <= cutoff_ym]  # up to & including cutoff

    if df_f.empty:
        st.warning("No rows match the current filters.")
        return

    # ------ Report 1: Product Inventory Ledger (Monthly Qty) ------
    st.subheader("1) Product Inventory Ledger (Monthly Qty)")
    ledger_cols = ["year", "month", "warehouse", "itemcode", "itemname", "itemgroup", "stockqty"]
    # Keep zid in the ledger to trace packaging vs primary if desired; not grouped by zid when running totals
    if "zid" in df_f.columns and "zid" not in ledger_cols:
        ledger_cols = ["zid"] + ledger_cols

    ledger = (
        df_f[ledger_cols]
        .sort_values(["itemcode", "warehouse", "year", "month"])
        .reset_index(drop=True)
    )
    # Running cumulative qty per itemcode×warehouse across combined zids
    ledger["running_qty"] = ledger.groupby(["itemcode", "warehouse"])["stockqty"].cumsum()

    _DISPLAY_LIMIT = 50_000
    if len(ledger) > _DISPLAY_LIMIT:
        st.info(
            f"Showing first {_DISPLAY_LIMIT:,} of {len(ledger):,} rows. "
            f"Use the download button below for the full dataset."
        )
        st.dataframe(ledger.head(_DISPLAY_LIMIT), use_container_width=True, height=420)
    else:
        st.dataframe(ledger, use_container_width=True, height=420)

    _csv_ledger = ledger.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇ Download Ledger CSV ({len(ledger):,} rows)",
        data=_csv_ledger,
        file_name="inventory_ledger.csv",
        mime="text/csv",
        key="dl_ledger",
    )

    # ------ Report 2: Final Stock by Product & Warehouse (Qty & Value as-of cutoff) ------
    st.subheader("2) Final Stock — Qty & Value (as of cutoff)")
    final = (
        df_f
        .groupby(["warehouse", "itemcode", "itemname", "itemgroup"], as_index=False)
        .agg(final_qty=("stockqty", "sum"),
             final_value=("stockvalue", "sum"))
        .sort_values(["warehouse", "itemcode"])
        .reset_index(drop=True)
    )
    st.dataframe(final, use_container_width=True, height=420)
    st.download_button(
        label=f"⬇ Download Final Stock CSV ({len(final):,} rows)",
        data=final.to_csv(index=False).encode("utf-8"),
        file_name="final_stock_by_product.csv",
        mime="text/csv",
        key="dl_final_stock",
    )

    # ------ Report 3: Warehouse Value Transactions per Month (flow up to cutoff) ------
    st.subheader("3) Warehouse Value Transactions (Monthly, up to cutoff)")
    flow = (
        df_f
        .groupby(["year", "month", "warehouse"], as_index=False)
        .agg(value_txn=("stockvalue", "sum"))
        .sort_values(["year", "month", "warehouse"])
    )
    flow["year_month"] = flow.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}", axis=1)
    flow = flow[["year_month", "warehouse", "value_txn"]]
    st.dataframe(flow, use_container_width=True, height=360)
    st.download_button(
        label=f"⬇ Download Warehouse Flow CSV ({len(flow):,} rows)",
        data=flow.to_csv(index=False).encode("utf-8"),
        file_name="warehouse_value_flow.csv",
        mime="text/csv",
        key="dl_wh_flow",
    )

   # ------ Report 4: Warehouse Ending Stock Value (as of cutoff) ------
    st.subheader("4) Warehouse Ending Stock Value (as of cutoff)")

    # df_f already contains all rows up to the cutoff (and only selected warehouses/products/groups)
    warehouse_ending = (
        df_f.groupby(["warehouse"], as_index=False)
            .agg(ending_value=("stockvalue", "sum"))
            .sort_values("warehouse")
            .reset_index(drop=True)
    )

    st.dataframe(warehouse_ending, use_container_width=True, height=300)
    st.download_button(
        label=f"⬇ Download Warehouse Ending Value CSV ({len(warehouse_ending):,} rows)",
        data=warehouse_ending.to_csv(index=False).encode("utf-8"),
        file_name="warehouse_ending_stock_value.csv",
        mime="text/csv",
        key="dl_wh_ending",
    )

   # ------ Report 5: Movement Analysis (Fast / Slow / Stagnant) ------
    st.subheader("5) Movement Analysis (Fast / Slow / Stagnant)")

    flow_df = _load_stock_flow(zid)

    if flow_df is None or flow_df.empty:
        st.info("No stock_flow data found for the effective zids.")
    else:
        import numpy as np

        # ---- 1) Normalize types ----
        for col in ["year", "month"]:
            if col in flow_df.columns:
                flow_df[col] = pd.to_numeric(flow_df[col], errors="coerce")
        for c in ["warehouse", "itemcode"]:
            if c in flow_df.columns:
                flow_df[c] = flow_df[c].astype(str)
        for c in ["qty_in","qty_out","net_qty","val_in","val_out","net_val"]:
            if c in flow_df.columns:
                flow_df[c] = pd.to_numeric(flow_df[c], errors="coerce").fillna(0.0)

        # ---- 2) Make stock_flow pack-aware so keys align with Report 2 ----
        # We mirror the same CASE logic used in your stock SQL.
        # Load caitem for effective zids and build a raw->display mapping.
        try:
            zids_eff = _effective_zids(zid)
            caitem_frames = []
            for z in zids_eff:
                try:
                    ci = Analytics("caitem", zid=z, filters={"zid": (str(z),)}).data
                    if isinstance(ci, pd.DataFrame) and not ci.empty:
                        caitem_frames.append(ci[["itemcode", "itemname", "itemgroup", "packcode"]].copy())
                except Exception as _e:
                    pass
            if caitem_frames:
                caitem_map = pd.concat(caitem_frames, ignore_index=True).drop_duplicates()
                # compute display_code = packcode (when valid) else raw itemcode
                def _choose_code(row):
                    pk = str(row.get("packcode") or "").strip()
                    if pk and pk.upper() != "NO" and not pk.startswith("KH"):
                        return pk
                    return str(row.get("itemcode"))
                caitem_map["display_code"] = caitem_map.apply(_choose_code, axis=1)
                # Raw->display mapping
                raw_to_display = dict(zip(caitem_map["itemcode"].astype(str), caitem_map["display_code"]))
                # Apply mapping to flow_df item codes
                flow_df["itemcode"] = flow_df["itemcode"].map(raw_to_display).fillna(flow_df["itemcode"]).astype(str)
                # Build a lookup of display_code -> (name, group)
                disp_lookup = (
                    caitem_map[["display_code","itemname","itemgroup"]]
                    .drop_duplicates()
                    .rename(columns={"display_code":"itemcode"})
                )
            else:
                disp_lookup = inv_df[["itemcode","itemname","itemgroup"]].drop_duplicates()
        except Exception:
            # Fallback: use metadata from inv_df if caitem load fails
            disp_lookup = inv_df[["itemcode","itemname","itemgroup"]].drop_duplicates()

        # Attach itemname/itemgroup after mapping
        flow_df = flow_df.merge(disp_lookup, on="itemcode", how="left")

        # ---- 3) Apply the same user filters (warehouse / itemgroup / product) ----
        if wh_sel:
            flow_df = flow_df[flow_df["warehouse"].isin(wh_sel)]
        if ig_sel:
            flow_df = flow_df[flow_df["itemgroup"].isin(ig_sel)]
        if prod_codes:
            flow_df = flow_df[flow_df["itemcode"].isin(prod_codes)]

        # If nothing remains, still show balances with zeros
        # Compute balances base from df_f (already filtered and <= cutoff)
        if "final" not in locals():
            final = (
                df_f.groupby(["warehouse","itemcode","itemname","itemgroup"], as_index=False)
                    .agg(final_qty=("stockqty","sum"),
                        final_value=("stockvalue","sum"))
            )
        base = final.rename(columns={"final_qty":"ending_qty","final_value":"ending_value"})

        if flow_df.empty:
            # No movement rows after filters: show base with zero movement
            movement = base.copy()
            for c in ["qty_in_K","qty_out_K","abs_qty_K","active_months_K","months_since_move","f2s_qty_K"]:
                movement[c] = 0.0
            movement["movement_class"] = np.where(movement["ending_qty"] > 0, "STAGNANT", "NORMAL")

            out_cols = [
                "warehouse","itemgroup","itemcode","itemname",
                "abs_qty_K","qty_in_K","qty_out_K","active_months_K","months_since_move",
                "ending_qty","ending_value","f2s_qty_K","movement_class"
            ]
            movement = movement[out_cols].sort_values(
                ["warehouse","movement_class","abs_qty_K"], ascending=[True, True, False]
            )
            st.dataframe(movement, use_container_width=True, height=420)
            st.download_button(
                label=f"⬇ Download Movement Analysis CSV ({len(movement):,} rows)",
                data=movement.to_csv(index=False).encode("utf-8"),
                file_name="movement_analysis_0m.csv",
                mime="text/csv",
                key="dl_movement_0m",
            )
        else:
            # ---- 4) Timing model: history to cutoff + trailing window K months ----
            y = pd.to_numeric(flow_df["year"], errors="coerce").fillna(0).astype("int64")
            m = pd.to_numeric(flow_df["month"], errors="coerce").fillna(0).astype("int64")
            flow_df["mi"] = y * 12 + m

            cutoff_mi = int(cutoff_year) * 12 + int(cutoff_month)
            flow_all = flow_df[flow_df["mi"] <= cutoff_mi]

            # Trailing window size (months)
            K = st.slider("Trailing window (months) for movement ranking", 3, 12, 6,
                        help="Used for movement intensity (in/out) metrics only.")
            window = flow_all[flow_all["mi"] >= (cutoff_mi - (K - 1))]

            # ---- 5) Aggregate movement over window (INTENSITY) ----
            grp = ["warehouse","itemcode","itemname","itemgroup"]
            window_agg = (
                window.groupby(grp, as_index=False)
                    .agg(qty_in_K=("qty_in","sum"),
                        qty_out_K=("qty_out","sum"),
                        active_months_K=("net_qty", lambda s: (s != 0).sum()))
            )
            window_agg["abs_qty_K"] = window_agg["qty_in_K"] + window_agg["qty_out_K"]

            # ---- 6) Last movement across full history to cutoff (not just window) ----
            moved = flow_all.loc[(flow_all["qty_in"] > 0) | (flow_all["qty_out"] > 0)]
            last_move = (
                moved.groupby(grp, as_index=False)["mi"].max()
                    .rename(columns={"mi":"last_move_mi"})
            )

            # ---- 7) LEFT-JOIN movement onto balances base ----
            agg = (base
                .merge(window_agg, on=grp, how="left")
                .merge(last_move,  on=grp, how="left"))

            # Fill zeros for missing movement; ∞ months for never-moved
            for c in ["qty_in_K","qty_out_K","abs_qty_K","active_months_K"]:
                if c in agg.columns:
                    agg[c] = agg[c].fillna(0.0)
            agg["months_since_move"] = cutoff_mi - agg["last_move_mi"]
            agg["months_since_move"] = agg["months_since_move"].where(agg["last_move_mi"].notna(), np.inf)

            # ---- 8) Ratios & classification ----
            agg["f2s_qty_K"] = (agg["abs_qty_K"] / agg["ending_qty"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Percentiles per (warehouse,itemgroup)
            ref_75 = agg.groupby(["warehouse","itemgroup"])["abs_qty_K"].transform(lambda s: s.quantile(0.75) if len(s) else 0.0)
            low_25 = agg.groupby(["warehouse","itemgroup"])["abs_qty_K"].transform(lambda s: s.quantile(0.25) if len(s) else 0.0)

            agg["movement_class"] = np.where(
                (agg["abs_qty_K"] >= ref_75) & (agg["months_since_move"] <= 1), "FAST",
                np.where(
                    (agg["active_months_K"] == 0) & (agg["ending_qty"] > 0), "STAGNANT",
                    np.where(
                        (agg["abs_qty_K"] <= low_25) | (agg["months_since_move"] >= 3), "SLOW",
                        "NORMAL"
                    )
                )
            )

            # ---- 9) Output ----
            out_cols = [
                "warehouse","itemgroup","itemcode","itemname",
                "abs_qty_K","qty_in_K","qty_out_K","active_months_K","months_since_move",
                "ending_qty","ending_value","f2s_qty_K","movement_class"
            ]
            movement = agg[out_cols].sort_values(
                ["warehouse","movement_class","abs_qty_K"], ascending=[True, True, False]
            )
            st.dataframe(movement, use_container_width=True, height=420)
            st.download_button(
                label=f"⬇ Download Movement Analysis CSV ({len(movement):,} rows)",
                data=movement.to_csv(index=False).encode("utf-8"),
                file_name=f"movement_analysis_{K}m.csv",
                mime="text/csv",
                key="dl_movement_Km",
            )
