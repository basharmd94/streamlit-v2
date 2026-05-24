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

@st.cache_data(show_spinner=False, ttl=86400)
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

@st.cache_data(show_spinner=False, ttl=86400)
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

@st.cache_data(show_spinner=False, ttl=86400)
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

_DEFAULT_WAREHOUSES = [
    "Finished Goods Store Packaging",
    "HMBR -Main Store (4th Floor)",
    "HMBR -W5 (MirerBaazar 2nd Floor)",
    "HMBR -W7 (2) (MirerBaazar 3rd Floor)",
    "HMBR -W7 (MirerBaazar 3rd Floor)",
    "HMBR Showroom (Mirer Bazar)",
    "Raw Material Store Packaging",
]

_DEFAULT_ITEMGROUPS = [
    "Sanitary",
    "RAW Material Packaging",
    "RAW Material GPK",
    "Packaging Item (P)",
    "Packaging Item",
    "Karigor Item",
    "Industrial & Household",
    "Household Product",
    "Hardware",
    "Import Item",
    "Furniture Fittings",
    "Finished Goods Packaging",
]



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
    # Exclude garbage year values (e.g. 2102 from data-entry errors in the DB)
    # that would push the default cutoff far into the future and empty the window.
    _current_year = pd.Timestamp.today().year
    years = sorted(
        y for y in inv_df["year"].dropna().astype(int).unique()
        if 2000 <= y <= _current_year + 1
    )
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
    st.caption("🔎 Item code already applies packcode CASE logic in SQL (no extra toggle needed).")

    year_sel = st.selectbox("Year (cutoff)", years, index=len(years) - 1 if years else 0, key="inv_year")
    month_sel = st.selectbox(
        "Month (cutoff)", months,
        index=(pd.Timestamp.today().month - 1),
        format_func=lambda m: calendar.month_abbr[m],
        key="inv_month"
    )
    use_latest = st.toggle(
        "Use Max Year–Month", value=False,
        help="If ON, uses the latest available period in data (considering auto-included packaging zid when applicable)."
    )

    _default_wh = [w for w in _DEFAULT_WAREHOUSES if w in warehouses]
    wh_sel = st.multiselect("Warehouse(s)", options=warehouses, default=_default_wh if _default_wh else warehouses)

    _default_ig = [g for g in _DEFAULT_ITEMGROUPS if g in itemgroups]
    ig_sel = st.multiselect("Item Group(s)", options=itemgroups, default=_default_ig if _default_ig else itemgroups)

    label_to_code = dict(inv_df[["product_label", "itemcode"]].drop_duplicates().values.tolist())

    # Determine cutoff based on toggle (on merged dataset)
    if use_latest:
        latest_row = inv_df.loc[inv_df["ym"].idxmax()]
        cutoff_year, cutoff_month = int(latest_row["year"]), int(latest_row["month"])
    else:
        cutoff_year, cutoff_month = int(year_sel), int(month_sel)
    cutoff_ym = cutoff_year * 100 + cutoff_month
    st.markdown(f"**Cutoff:** {cutoff_year}-{cutoff_month:02d}")

    # Apply base filters (without product filter — applied per-report)
    df_f = inv_df.copy()
    if wh_sel:
        df_f = df_f[df_f["warehouse"].isin(wh_sel)]
    if ig_sel:
        df_f = df_f[df_f["itemgroup"].isin(ig_sel)]
    df_f = df_f[df_f["ym"] <= cutoff_ym]

    if df_f.empty:
        st.warning("No rows match the current filters.")
        return

    # ------ Product Inventory Ledger (gated by button) ------
    st.subheader("Product Inventory Ledger")
    prod_sel_labels = st.multiselect("Product(s)", options=products, default=[],
                                     placeholder="Type to search code/name…", key="inv_prod_sel")
    prod_codes = [label_to_code.get(lbl) for lbl in prod_sel_labels]

    if st.button("Generate Product Ledger", type="primary", key="inv_gen_ledger"):
        st.session_state["_inv_ledger_ready"] = True

    if st.session_state.get("_inv_ledger_ready"):
        df_ledger = df_f.copy()
        if prod_codes:
            df_ledger = df_ledger[df_ledger["itemcode"].isin(prod_codes)]

        ledger_cols = ["year", "month", "warehouse", "itemcode", "itemname", "itemgroup", "stockqty"]
        if "zid" in df_ledger.columns:
            ledger_cols = ["zid"] + ledger_cols

        ledger = (
            df_ledger[ledger_cols]
            .sort_values(["itemcode", "warehouse", "year", "month"])
            .reset_index(drop=True)
        )
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

        st.download_button(
            label=f"⬇ Download Ledger CSV ({len(ledger):,} rows)",
            data=ledger.to_csv(index=False).encode("utf-8"),
            file_name="inventory_ledger.csv",
            mime="text/csv",
            key="dl_ledger",
        )

    # ------ Final Stock by Product & Warehouse (Qty & Value as-of cutoff) ------
    st.subheader("Final Stock — Qty & Value (as of cutoff)")
    _tog1, _tog2 = st.columns(2)
    with _tog1:
        show_warehouse_col = st.toggle(
            "Show by Warehouse",
            value=False,
            key="inv_show_wh",
            help="ON: one row per warehouse × item.",
        )
    with _tog2:
        show_zid_col = st.toggle(
            "Show by ZID",
            value=False,
            key="inv_show_zid",
            help="ON: split rows by source ZID (e.g. 100001 vs 100009). Both toggles OFF = fully summed totals.",
        )

    # Build item name/group lookup — prefer primary zid so cross-zid totals
    # display the main company's caitem name rather than the packaging name.
    _has_src = "_src_zid" in df_f.columns
    if _has_src:
        _meta = (
            df_f[["itemcode", "itemname", "itemgroup", "_src_zid"]]
            .drop_duplicates()
            .sort_values("_src_zid", key=lambda s: s.map(lambda x: 0 if x == str(zid) else 1))
            .drop_duplicates("itemcode", keep="first")
            [["itemcode", "itemname", "itemgroup"]]
        )
    else:
        _meta = df_f[["itemcode", "itemname", "itemgroup"]].drop_duplicates("itemcode", keep="first")

    # Always compute warehouse × itemcode aggregation for Movement Analysis below.
    # Group by itemcode only (NOT itemname/itemgroup) so cross-zid rows for the
    # same effective product are summed together, then merge preferred names back.
    final = (
        df_f.groupby(["warehouse", "itemcode"], as_index=False)
            .agg(final_qty=("stockqty", "sum"), final_value=("stockvalue", "sum"))
            .merge(_meta, on="itemcode", how="left")
            [["warehouse", "itemcode", "itemname", "itemgroup", "final_qty", "final_value"]]
            .sort_values(["warehouse", "itemcode"])
            .reset_index(drop=True)
    )

    # Determine groupby keys and build the display frame
    _grp = []
    if show_zid_col and _has_src:
        _grp.append("_src_zid")
    if show_warehouse_col:
        _grp.append("warehouse")
    _grp.append("itemcode")

    if show_zid_col and _has_src:
        # Per-zid split — include each zid's own itemname/itemgroup in the groupby
        # so names reflect the caitem of that zid.
        final_display = (
            df_f.groupby(_grp + ["itemname", "itemgroup"], as_index=False)
                .agg(final_qty=("stockqty", "sum"), final_value=("stockvalue", "sum"))
                .rename(columns={"_src_zid": "ZID"})
                .sort_values(
                    ["ZID"] + (["warehouse"] if show_warehouse_col else []) + ["itemcode"]
                )
                .reset_index(drop=True)
        )
    else:
        # Sum across zids — aggregate by itemcode (+ warehouse if toggled),
        # then merge preferred item names back.
        agg = (
            df_f.groupby(_grp, as_index=False)
                .agg(final_qty=("stockqty", "sum"), final_value=("stockvalue", "sum"))
                .merge(_meta, on="itemcode", how="left")
        )
        _col_order = (
            (["warehouse"] if show_warehouse_col else [])
            + ["itemcode", "itemname", "itemgroup", "final_qty", "final_value"]
        )
        final_display = (
            agg[[c for c in _col_order if c in agg.columns]]
            .sort_values(["warehouse", "itemcode"] if show_warehouse_col else ["itemcode"])
            .reset_index(drop=True)
        )

    # Generic total row — label goes in whichever column comes first
    _first = final_display.columns[0]
    _tot = {c: "" for c in final_display.columns}
    _tot[_first]        = "─── TOTAL ───"
    _tot["final_qty"]   = final_display["final_qty"].sum()
    _tot["final_value"] = final_display["final_value"].sum()
    final_with_total = pd.concat([final_display, pd.DataFrame([_tot])], ignore_index=True)
    st.dataframe(final_with_total, use_container_width=True, height=420)
    st.download_button(
        label=f"⬇ Download Final Stock CSV ({len(final_display):,} rows)",
        data=final_display.to_csv(index=False).encode("utf-8"),
        file_name="final_stock_by_product.csv",
        mime="text/csv",
        key="dl_final_stock",
    )

    # ------ Warehouse Ending Stock Value (gated by button) ------
    st.subheader("Warehouse Ending Stock Value")
    if st.button("Generate Warehouse Ending Value", type="primary", key="inv_gen_wh_end"):
        st.session_state["_inv_wh_end_ready"] = True

    if st.session_state.get("_inv_wh_end_ready"):
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

    # ------ Movement Analysis (gated by button) ------
    st.subheader("Movement Analysis (Fast / Slow / Stagnant)")
    if st.button("Generate Movement Analysis", type="primary", key="inv_gen_movement"):
        st.session_state["_inv_movement_ready"] = True

    if st.session_state.get("_inv_movement_ready"):
        flow_df = _load_stock_flow(zid)

        if flow_df is None or flow_df.empty:
            st.info("No stock_flow data found for the effective zids.")
        else:
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

            # ---- 2) Apply the same user filters (warehouse / product) ----
            if wh_sel:
                flow_df = flow_df[flow_df["warehouse"].isin(wh_sel)]
            if prod_codes:
                flow_df = flow_df[flow_df["itemcode"].isin(prod_codes)]

            base = final.rename(columns={"final_qty": "ending_qty", "final_value": "ending_value"})

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
                # ---- 3) Timing model: history to cutoff + trailing window K months ----
                y = pd.to_numeric(flow_df["year"], errors="coerce").fillna(0).astype("int64")
                m = pd.to_numeric(flow_df["month"], errors="coerce").fillna(0).astype("int64")
                flow_df["mi"] = y * 12 + m

                cutoff_mi = int(cutoff_year) * 12 + int(cutoff_month)
                flow_all = flow_df[flow_df["mi"] <= cutoff_mi]

                K = st.slider("Trailing window (months) for movement ranking", 3, 12, 6,
                              help="Used for movement intensity (in/out) metrics only.")
                window = flow_all[flow_all["mi"] >= (cutoff_mi - (K - 1))]

                # ---- 4) Aggregate movement over window (INTENSITY) ----
                grp_key = ["warehouse", "itemcode"]
                window_agg = (
                    window.groupby(grp_key, as_index=False)
                          .agg(qty_in_K=("qty_in","sum"),
                               qty_out_K=("qty_out","sum"),
                               active_months_K=("net_qty", lambda s: (s != 0).sum()))
                )
                window_agg["abs_qty_K"] = window_agg["qty_in_K"] + window_agg["qty_out_K"]

                # ---- 5) Last movement across full history to cutoff ----
                moved = flow_all.loc[(flow_all["qty_in"] > 0) | (flow_all["qty_out"] > 0)]
                last_move = (
                    moved.groupby(grp_key, as_index=False)["mi"].max()
                         .rename(columns={"mi":"last_move_mi"})
                )

                # ---- 6) LEFT-JOIN movement onto balances base ----
                agg = (base
                       .merge(window_agg, on=grp_key, how="left")
                       .merge(last_move,  on=grp_key, how="left"))

                for c in ["qty_in_K","qty_out_K","abs_qty_K","active_months_K"]:
                    if c in agg.columns:
                        agg[c] = agg[c].fillna(0.0)
                agg["months_since_move"] = cutoff_mi - agg["last_move_mi"]
                agg["months_since_move"] = agg["months_since_move"].where(agg["last_move_mi"].notna(), np.inf)

                # ---- 7) Ratios & classification ----
                agg["f2s_qty_K"] = (agg["abs_qty_K"] / agg["ending_qty"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

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

                # ---- 8) Output ----
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
