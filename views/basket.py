import streamlit as st
import pandas as pd
import numpy as np
import calendar
from core.analytics import Analytics, basket_prepare
from utils.utils import timed


# -----------------------------
# Basket Analysis helper functions
# -----------------------------

def _table_writeup(title: str, columns: dict[str, str], toggles: list[str] | None = None):
    """
    Render a consistent explanation block under a table.
    columns: {"colname": "meaning", ...}
    toggles: ["Toggle X: meaning", ...]
    """
    st.markdown(f"**{title} — Column Guide**")
    st.markdown("\n".join([f"- `{k}`: {v}" for k, v in columns.items()]))

    if toggles:
        st.markdown("**Controls / Options**")
        st.markdown("\n".join([f"- {t}" for t in toggles]))

def _ensure_datetime(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce")

def _month_name(m: int) -> str:
    try:
        return calendar.month_abbr[int(m)]
    except Exception:
        return str(m)

def _build_order_sets(df: pd.DataFrame, order_id_col: str, key_col: str) -> dict:
    """
    order_id -> set(keys)
    Presence-based (unique keys per order).
    """
    tmp = df[[order_id_col, key_col]].dropna()
    grouped = tmp.groupby(order_id_col)[key_col].agg(lambda x: set(x.astype(str).unique()))
    return grouped.to_dict()

def _anchor_orders(order_sets: dict, anchors: set[str], mode: str) -> set:
    """
    mode: 'ALL' (intersection) or 'ANY' (union)
    """
    if not anchors:
        return set()
    out = set()
    if mode == "ALL":
        for oid, s in order_sets.items():
            if anchors.issubset(s):
                out.add(oid)
    else:
        for oid, s in order_sets.items():
            if s.intersection(anchors):
                out.add(oid)
    return out

def _basket_recommendations_items(df: pd.DataFrame,order_id_col: str,item_col: str,qty_col: str,value_col: str,anchor_items: list[str],anchor_mode: str = "ALL",top_n: int = 200,) -> pd.DataFrame:
    """
    Return ranked recommended items (excluding anchor items) with support/confidence/lift and qty/value extras.
    """
    if df.empty or not anchor_items:
        return pd.DataFrame()

    work = df.copy()
    work[item_col] = work[item_col].astype(str)
    work[order_id_col] = work[order_id_col].astype(str)

    order_sets = _build_order_sets(work, order_id_col, item_col)
    total_orders = len(order_sets)
    if total_orders == 0:
        return pd.DataFrame()

    anchors = set([str(x).split(" - ")[0] for x in anchor_items])
    anchor_orders = _anchor_orders(order_sets, anchors, anchor_mode)
    anchor_orders_n = len(anchor_orders)
    if anchor_orders_n == 0:
        return pd.DataFrame()

    # Item presence counts across all orders
    exploded = []
    for oid, items in order_sets.items():
        for it in items:
            exploded.append((oid, it))
    exp_df = pd.DataFrame(exploded, columns=[order_id_col, item_col])

    item_order_counts = exp_df.groupby(item_col)[order_id_col].nunique()

    # Co-occur counts within anchor orders
    anchor_items_rows = exp_df[exp_df[order_id_col].isin(anchor_orders)]
    co_counts = anchor_items_rows.groupby(item_col)[order_id_col].nunique()

    # Build result (exclude anchors)
    candidates = co_counts.index.difference(pd.Index(list(anchors)))
    if len(candidates) == 0:
        return pd.DataFrame()

    base = pd.DataFrame({
        "itemcode": candidates,
        "orders_with_anchor_and_item": co_counts.loc[candidates].values,
        "orders_with_item": item_order_counts.reindex(candidates).fillna(0).astype(int).values
    })

    base["anchor_orders"] = anchor_orders_n
    base["total_orders"] = total_orders
    base["support"] = base["orders_with_anchor_and_item"] / total_orders
    base["confidence"] = base["orders_with_anchor_and_item"] / anchor_orders_n
    base["item_support"] = base["orders_with_item"] / total_orders
    base["lift"] = np.where(base["item_support"] > 0, base["confidence"] / base["item_support"], np.nan)

    # Qty / value extras inside anchor orders for candidate items
    in_anchor = work[work[order_id_col].isin(anchor_orders)]
    cand_mask = in_anchor[item_col].isin(base["itemcode"])
    in_anchor = in_anchor.loc[cand_mask, [order_id_col, item_col, qty_col, value_col]]

    # Coerce numeric
    in_anchor[qty_col] = pd.to_numeric(in_anchor[qty_col], errors="coerce").fillna(0.0)
    in_anchor[value_col] = pd.to_numeric(in_anchor[value_col], errors="coerce").fillna(0.0)

    agg = in_anchor.groupby(item_col).agg(
        total_qty_when_cooccurs=(qty_col, "sum"),
        avg_qty_when_cooccurs=(qty_col, "mean"),
        total_value_when_cooccurs=(value_col, "sum"),
        avg_value_when_cooccurs=(value_col, "mean"),
    ).reset_index().rename(columns={item_col: "itemcode"})

    base = base.merge(agg, on="itemcode", how="left")

    # Add itemname/itemgroup if available
    meta_cols = [c for c in ["itemname", "itemgroup"] if c in work.columns]
    if meta_cols:
        meta = (
            work[[item_col] + meta_cols]
            .dropna()
            .drop_duplicates(subset=[item_col])
            .rename(columns={item_col: "itemcode"})
        )
        base = base.merge(meta, on="itemcode", how="left")

    # Rank
    base = base.sort_values(["lift", "confidence", "support"], ascending=[False, False, False]).head(top_n)
    return base.reset_index(drop=True)

def _best_months_for_pairs(df: pd.DataFrame,order_id_col: str,month_col: str,item_col: str,anchor_orders: set,itemcodes: list[str],top_k_months: int = 2) -> dict:
    """
    For each itemcode, find the 'best months' based on co-occur frequency within anchor orders.
    Returns dict itemcode -> "Jan, Feb"
    """
    if df.empty or not anchor_orders or not itemcodes:
        return {}

    work = df[[order_id_col, month_col, item_col]].copy()
    work[order_id_col] = work[order_id_col].astype(str)
    work[item_col] = work[item_col].astype(str)
    work = work[work[order_id_col].isin(anchor_orders)]
    work[month_col] = pd.to_numeric(work[month_col], errors="coerce").astype("Int64")

    out = {}
    for code in itemcodes:
        sub = work[work[item_col] == str(code)]
        if sub.empty:
            out[str(code)] = ""
            continue
        cnt = sub.groupby(month_col)[order_id_col].nunique().sort_values(ascending=False)
        months = [m for m in cnt.index.tolist() if pd.notna(m)]
        months = months[:top_k_months]
        out[str(code)] = ", ".join(_month_name(int(m)) for m in months)
    return out

def _basket_recommendations_groups(df: pd.DataFrame,order_id_col: str,group_col: str,qty_col: str,value_col: str,anchor_groups: list[str],anchor_mode: str = "ALL",top_n: int = 200) -> pd.DataFrame:
    """
    Group-to-group basket recommendations.
    """
    if df.empty or not anchor_groups:
        return pd.DataFrame()

    work = df.copy()
    work[group_col] = work[group_col].fillna("").astype(str)
    work[order_id_col] = work[order_id_col].astype(str)

    order_sets = _build_order_sets(work, order_id_col, group_col)
    total_orders = len(order_sets)
    if total_orders == 0:
        return pd.DataFrame()

    anchors = set([str(x) for x in anchor_groups if str(x).strip() != ""])
    anchor_orders = _anchor_orders(order_sets, anchors, anchor_mode)
    anchor_orders_n = len(anchor_orders)
    if anchor_orders_n == 0:
        return pd.DataFrame()

    # Presence counts
    exploded = []
    for oid, groups in order_sets.items():
        for g in groups:
            if g != "":
                exploded.append((oid, g))
    exp_df = pd.DataFrame(exploded, columns=[order_id_col, group_col])
    group_order_counts = exp_df.groupby(group_col)[order_id_col].nunique()

    anchor_rows = exp_df[exp_df[order_id_col].isin(anchor_orders)]
    co_counts = anchor_rows.groupby(group_col)[order_id_col].nunique()

    candidates = co_counts.index.difference(pd.Index(list(anchors)))
    if len(candidates) == 0:
        return pd.DataFrame()

    base = pd.DataFrame({
        "recommended_group": candidates,
        "orders_with_anchor_and_group": co_counts.loc[candidates].values,
        "orders_with_group": group_order_counts.reindex(candidates).fillna(0).astype(int).values
    })
    base["anchor_orders"] = anchor_orders_n
    base["total_orders"] = total_orders
    base["support"] = base["orders_with_anchor_and_group"] / total_orders
    base["confidence"] = base["orders_with_anchor_and_group"] / anchor_orders_n
    base["group_support"] = base["orders_with_group"] / total_orders
    base["lift"] = np.where(base["group_support"] > 0, base["confidence"] / base["group_support"], np.nan)

    # qty/value extras for recommended groups within anchor orders
    in_anchor = work[work[order_id_col].isin(anchor_orders)]
    in_anchor = in_anchor[in_anchor[group_col].isin(base["recommended_group"])]
    in_anchor[qty_col] = pd.to_numeric(in_anchor[qty_col], errors="coerce").fillna(0.0)
    in_anchor[value_col] = pd.to_numeric(in_anchor[value_col], errors="coerce").fillna(0.0)

    agg = in_anchor.groupby(group_col).agg(
        total_qty_when_cooccurs=(qty_col, "sum"),
        total_value_when_cooccurs=(value_col, "sum"),
    ).reset_index().rename(columns={group_col: "recommended_group"})

    base = base.merge(agg, on="recommended_group", how="left")
    base = base.sort_values(["lift", "confidence", "support"], ascending=[False, False, False]).head(top_n)
    return base.reset_index(drop=True)

def display_basket_analysis_page(current_page, zid: str, data_dict: dict, selected_filters: dict):
    # ── Basket Analysis is temporarily disabled ──
    st.info("Basket Analysis is currently unavailable.")

def _display_basket_analysis_page_disabled(current_page, zid: str, data_dict: dict, selected_filters: dict):
    st.header("Basket Analysis")

    # Prepare merged datasets (Basket-only merge rule) using navbar filters
    prep = basket_prepare(str(zid), selected_filters or {})
    scope_zids = prep.get("scope_zids", [str(zid)])
    sales_df = prep.get("sales", pd.DataFrame())
    purchase_df = prep.get("purchase", pd.DataFrame())

    # Ensure expected types / columns
    if sales_df is None or sales_df.empty:
        st.warning("No sales data available for Basket Analysis with the selected filters.")
        return

    # Build order_id (zid + DO number)
    sales_df = sales_df.copy()

    # Compute final_sales (xdtwotax - xdtdisc) consistent with all other views
    if "altsales" in sales_df.columns and "proddiscount" in sales_df.columns:
        sales_df["final_sales"] = (
            pd.to_numeric(sales_df["altsales"], errors="coerce").fillna(0)
            - pd.to_numeric(sales_df["proddiscount"], errors="coerce").fillna(0)
        )
    sales_df["zid"] = sales_df["zid"].astype(str)
    sales_df["voucher"] = sales_df["voucher"].astype(str)
    sales_df["order_id"] = sales_df["zid"] + "-" + sales_df["voucher"]

    # Parse dates
    if "date" in sales_df.columns:
        sales_df["date"] = _ensure_datetime(sales_df["date"])
    if "month" in sales_df.columns:
        sales_df["month"] = pd.to_numeric(sales_df["month"], errors="coerce").fillna(0).astype(int)

    # -----------------------------
    # Customer Master Audit (top)
    # -----------------------------
    if scope_zids == ["100000", "100001"]:
        with st.expander("Customer Master Audit (100000 vs 100001)", expanded=True):
            c0 = prep.get("cacus_100000", pd.DataFrame()).copy()
            c1 = prep.get("cacus_100001", pd.DataFrame()).copy()

            if c0.empty or c1.empty:
                st.warning("Customer master data not available for audit.")
            else:
                c0["cusid"] = c0["cusid"].astype(str)
                c1["cusid"] = c1["cusid"].astype(str)

                overlap = set(c0["cusid"]).intersection(set(c1["cusid"]))
                only_0 = set(c0["cusid"]) - set(c1["cusid"])
                only_1 = set(c1["cusid"]) - set(c0["cusid"])

                st.write({
                    "overlap_customers": len(overlap),
                    "only_in_100000": len(only_0),
                    "only_in_100001": len(only_1),
                })

                # Compare all shared columns except zid
                cols0 = [c for c in c0.columns if c != "zid"]
                cols1 = [c for c in c1.columns if c != "zid"]
                common_cols = sorted(set(cols0).intersection(set(cols1)))
                if "cusid" not in common_cols:
                    st.error("cacus_simple must include 'cusid' for audit.")
                else:
                    left = c0[common_cols].rename(columns={c: f"{c}_100000" for c in common_cols if c != "cusid"})
                    right = c1[common_cols].rename(columns={c: f"{c}_100001" for c in common_cols if c != "cusid"})

                    merged = left.merge(right, on="cusid", how="inner")

                    mismatch_counts = {}
                    mismatch_mask = pd.Series(False, index=merged.index)
                    for c in common_cols:
                        if c == "cusid":
                            continue
                        a = merged[f"{c}_100000"].astype(str)
                        b = merged[f"{c}_100001"].astype(str)
                        msk = a.ne(b)
                        mismatch_counts[c] = int(msk.sum())
                        mismatch_mask = mismatch_mask | msk

                    st.subheader("Mismatch counts by column")
                    st.dataframe(pd.DataFrame([mismatch_counts]).T.rename(columns={0: "mismatch_count"}))

                    only_mismatches = st.checkbox("Show only mismatches", value=True, key="basket_cacus_only_mismatch")
                    audit_df = merged[mismatch_mask] if only_mismatches else merged
                    st.dataframe(audit_df)

    # -----------------------------
    # Main perspective selector
    # -----------------------------
    perspective = st.radio("Select Perspective", ["Sales Perspective", "Purchase Perspective"], horizontal=True)

    # ==========================================================
    # SALES PERSPECTIVE
    # ==========================================================
    if perspective == "Sales Perspective":
        tabA, tabB, tabC, tabD = st.tabs(["Product(s) Basket", "Customer Pattern", "Area Pattern", "Product Group Basket"])

        # ---- Tab A: Product(s) basket ----
        with tabA:
            st.subheader("Product(s) Basket")

            # Build product options from the (already filtered) sales_df
            prod_meta = (
                sales_df[["itemcode", "itemname"]]
                .dropna()
                .drop_duplicates()
                .sort_values(["itemcode", "itemname"])
            )
            prod_options = (prod_meta["itemcode"].astype(str) + " - " + prod_meta["itemname"].astype(str)).tolist()

            anchor_products = st.multiselect("Select Anchor Products (multi)", prod_options)
            anchor_mode = st.radio("Anchor Rule", ["ALL (intersection)", "ANY (union)"], index=0, horizontal=True)
            anchor_mode_key = "ALL" if anchor_mode.startswith("ALL") else "ANY"

            rank_by = st.selectbox("Rank by", ["lift", "confidence", "support"], index=0)
            show_seasonality = st.toggle("Show seasonality (best months)", value=True)

            if not anchor_products:
                st.info("Select at least one anchor product to see recommendations.")
            else:
                rec = _basket_recommendations_items(
                    df=sales_df,
                    order_id_col="order_id",
                    item_col="itemcode",
                    qty_col="quantity",
                    value_col="final_sales",
                    anchor_items=anchor_products,
                    anchor_mode=anchor_mode_key,
                    top_n=200,
                )

                if rec.empty:
                    st.warning("No anchor orders found (try switching ALL/ANY or widening filters).")
                else:
                    # Add best months (optional)
                    if show_seasonality and "month" in sales_df.columns:
                        # Recompute anchor orders for month calc
                        order_sets = _build_order_sets(sales_df, "order_id", "itemcode")
                        anchors = set([str(x).split(" - ")[0] for x in anchor_products])
                        anchor_orders = _anchor_orders(order_sets, anchors, anchor_mode_key)
                        best = _best_months_for_pairs(
                            df=sales_df,
                            order_id_col="order_id",
                            month_col="month",
                            item_col="itemcode",
                            anchor_orders=anchor_orders,
                            itemcodes=rec["itemcode"].astype(str).head(50).tolist(),
                            top_k_months=2,
                        )
                        rec["best_months"] = rec["itemcode"].astype(str).map(best).fillna("")

                    # Final sort by chosen metric
                    rec = rec.sort_values(rank_by, ascending=False).reset_index(drop=True)

                    # Show KPIs
                    st.write({
                        "anchor_mode": anchor_mode_key,
                        "recommended_items": int(rec.shape[0]),
                    })
                    st.dataframe(rec)
                    _table_writeup(
                        "Product Basket Recommendations",
                        columns={
                            "itemcode": "Product code of the recommended item.",
                            "itemname": "Product name of the recommended item.",
                            "itemgroup": "Product group (itemgroup2) of the recommended item.",
                            "orders_with_anchor": "Number of orders (DO) that contain the selected anchor set.",
                            "orders_with_item": "Number of orders (DO) that contain the recommended item.",
                            "orders_with_anchor_and_item": "Number of orders that contain both the anchor set and the recommended item.",
                            "support": "Share of all orders that contain both anchor and recommended item.",
                            "confidence": "Likelihood the recommended item appears given the anchor order set.",
                            "lift": "Strength above baseline: confidence divided by the overall probability of the item.",
                            "avg_qty_when_cooccurs": "Average quantity of the recommended item within anchor orders where it appears.",
                            "total_qty_when_cooccurs": "Total quantity of the recommended item across anchor orders where it appears.",
                            "avg_value_when_cooccurs": "Average final_sales of the recommended item within anchor orders where it appears.",
                            "total_value_when_cooccurs": "Total final_sales of the recommended item across anchor orders where it appears.",
                            "best_months": "Months where co-occurrence is strongest (if seasonality toggle is ON).",
                        },
                        toggles=[
                            "Anchor Rule: **ALL (intersection)** means orders must contain every selected anchor product. **ANY (union)** means orders may contain at least one anchor product.",
                            "Rank by: Sorts results by lift / confidence / support.",
                            "Show seasonality: Adds best-months signal based on co-occurrence by month."
                        ]
                    )

        # ---- Tab B: Customer pattern ----
        with tabB:
            st.subheader("Customer Pattern")

            if "cusid" not in sales_df.columns or "cusname" not in sales_df.columns:
                st.warning("Customer fields not available in sales data.")
            else:
                cust_meta = (
                    sales_df[["cusid", "cusname"]]
                    .dropna()
                    .drop_duplicates()
                    .sort_values(["cusid", "cusname"])
                )
                cust_options = (cust_meta["cusid"].astype(str) + " - " + cust_meta["cusname"].astype(str)).tolist()
                selected_customer = st.selectbox("Select Customer", [""] + cust_options, index=0)

                if not selected_customer:
                    st.info("Select a customer to see patterns.")
                else:
                    cusid = selected_customer.split(" - ")[0].strip()
                    cdf = sales_df[sales_df["cusid"].astype(str) == cusid].copy()
                    if cdf.empty:
                        st.warning("No sales for this customer in the selected filters.")
                    else:
                        cdf["date"] = _ensure_datetime(cdf["date"])
                        cdf["dow"] = cdf["date"].dt.day_name()
                        cdf["dom"] = cdf["date"].dt.day

                        def dom_bucket(d):
                            if pd.isna(d):
                                return ""
                            d = int(d)
                            if d <= 5:
                                return "01-05"
                            if d <= 10:
                                return "06-10"
                            if d <= 15:
                                return "11-15"
                            if d <= 20:
                                return "16-20"
                            if d <= 25:
                                return "21-25"
                            return "26-31"

                        cdf["dom_bucket"] = cdf["dom"].apply(dom_bucket)

                        # presence by order_id
                        def top_products(group_col):
                            tmp = cdf.copy()
                            tmp["itemcode"] = tmp["itemcode"].astype(str)
                            orders = tmp.groupby([group_col, "itemcode"])["order_id"].nunique().reset_index(name="order_count")
                            qty = tmp.groupby([group_col, "itemcode"])["quantity"].sum().reset_index(name="total_qty")
                            val = tmp.groupby([group_col, "itemcode"])["final_sales"].sum().reset_index(name="total_value")
                            out = orders.merge(qty, on=[group_col, "itemcode"], how="left").merge(val, on=[group_col, "itemcode"], how="left")
                            meta = tmp[["itemcode", "itemname", "itemgroup"]].drop_duplicates(subset=["itemcode"])
                            out = out.merge(meta, on="itemcode", how="left")
                            return out.sort_values([group_col, "order_count"], ascending=[True, False])

                        st.write({
                            "customer_orders": int(cdf["order_id"].nunique()),
                            "customer_total_value": float(pd.to_numeric(cdf["final_sales"], errors="coerce").fillna(0).sum()),
                            "customer_total_qty": float(pd.to_numeric(cdf["quantity"], errors="coerce").fillna(0).sum()),
                        })

                        st.markdown("### By Month")
                        month_tbl = top_products("month")
                        st.dataframe(month_tbl, use_container_width=True)
                        _table_writeup(
                            "Customer Pattern — By Month",
                            columns={
                                "month": "Calendar month (1–12) of the sales date.",
                                "itemcode": "Product code.",
                                "itemname": "Product name.",
                                "itemgroup": "Product group (itemgroup2).",
                                "order_count": "Number of distinct orders (DO) in which the product appears for the selected month.",
                                "total_qty": "Total quantity sold for the product in the selected month.",
                                "total_value": "Total sales value (final_sales) for the product in the selected month."
                            }
                        )

                        st.markdown("### By Day of Week")
                        dow_tbl = top_products("dow")
                        st.dataframe(dow_tbl, use_container_width=True)
                        _table_writeup(
                            "Customer Pattern — By Day of Week",
                            columns={
                                "dow": "Day name (Monday–Sunday) derived from the sales date.",
                                "itemcode": "Product code.",
                                "itemname": "Product name.",
                                "itemgroup": "Product group (itemgroup2).",
                                "order_count": "Number of distinct orders (DO) in which the product appears for the selected day of week.",
                                "total_qty": "Total quantity sold for the product on that day of week.",
                                "total_value": "Total sales value (final_sales) for the product on that day of week."
                            }
                        )

                        st.markdown("### By Day-of-Month Bucket")
                        dom_tbl = top_products("dom_bucket")
                        st.dataframe(dom_tbl, use_container_width=True)
                        _table_writeup(
                            "Customer Pattern — By Day-of-Month Bucket",
                            columns={
                                "dom_bucket": "Day-of-month grouped as 01–05, 06–10, 11–15, 16–20, 21–25, 26–31.",
                                "itemcode": "Product code.",
                                "itemname": "Product name.",
                                "itemgroup": "Product group (itemgroup2).",
                                "order_count": "Number of distinct orders (DO) in which the product appears for the selected day bucket.",
                                "total_qty": "Total quantity sold for the product in that day bucket.",
                                "total_value": "Total sales value (final_sales) for the product in that day bucket."
                            }
                        )


        # ---- Tab C: Area pattern ----
        with tabC:
            st.subheader("Area Pattern (cuscity)")

            area_col = "area" if "area" in sales_df.columns else None
            if not area_col:
                st.warning("Area field not available in sales data.")
            else:
                areas = sorted([a for a in sales_df[area_col].dropna().astype(str).unique().tolist() if a.strip() != ""])
                selected_area = st.selectbox("Select Area", [""] + areas, index=0)
                if not selected_area:
                    st.info("Select an area to see top products.")
                else:
                    adf = sales_df[sales_df[area_col].astype(str) == selected_area].copy()
                    if adf.empty:
                        st.warning("No sales in this area for the selected filters.")
                    else:
                        top = (
                            adf.groupby(["itemcode", "itemname", "itemgroup"])["order_id"]
                            .nunique()
                            .reset_index(name="order_count")
                        )
                        qty = adf.groupby(["itemcode"])["quantity"].sum().reset_index(name="total_qty")
                        val = adf.groupby(["itemcode"])["final_sales"].sum().reset_index(name="total_value")
                        top = top.merge(qty, on="itemcode", how="left").merge(val, on="itemcode", how="left")
                        top = top.sort_values("order_count", ascending=False)
                        st.dataframe(top)
                        _table_writeup(
                            "Area Pattern — Top Products",
                            columns={
                                "area": "Customer city/area (cacus.cuscity) for the selected area filter.",
                                "itemcode": "Product code.",
                                "itemname": "Product name.",
                                "itemgroup": "Product group (itemgroup2).",
                                "order_count": "Number of distinct orders (DO numbers) in which the product appears within the selected area and time range.",
                                "total_qty": "Total quantity sold for the product within the selected area and time range.",
                                "total_value": "Total sales value (final_sales) for the product within the selected area and time range."
                            },
                            toggles=[
                                "Area dropdown: Filters the analysis to customers whose `cuscity` matches the chosen area.",
                                "All results reflect the global navbar filters (date range, salesman, customer, product filters)."
                            ]
                        )


        # ---- Tab D: Product group basket ----
        with tabD:
            st.subheader("Product Group Basket")

            if "itemgroup" not in sales_df.columns:
                st.warning("itemgroup not available in sales data.")
            else:
                groups = sorted([g for g in sales_df["itemgroup"].dropna().astype(str).unique().tolist() if g.strip() != ""])
                anchor_groups = st.multiselect("Select Anchor Product Groups (multi)", groups)

                anchor_mode_g = st.radio("Anchor Rule (Groups)", ["ALL (intersection)", "ANY (union)"], index=0, horizontal=True)
                anchor_mode_g_key = "ALL" if anchor_mode_g.startswith("ALL") else "ANY"
                rank_by_g = st.selectbox("Rank groups by", ["lift", "confidence", "support"], index=0, key="rank_by_group")

                if not anchor_groups:
                    st.info("Select at least one product group.")
                else:
                    grec = _basket_recommendations_groups(
                        df=sales_df,
                        order_id_col="order_id",
                        group_col="itemgroup",
                        qty_col="quantity",
                        value_col="final_sales",
                        anchor_groups=anchor_groups,
                        anchor_mode=anchor_mode_g_key,
                        top_n=200,
                    )
                    if grec.empty:
                        st.warning("No anchor orders found for these groups (try ANY or widen filters).")
                    else:
                        grec = grec.sort_values(rank_by_g, ascending=False).reset_index(drop=True)
                        st.dataframe(grec)
                    _table_writeup(
                        "Product Group Basket — Co-occurring Groups",
                        columns={
                        "recommended_group": "A product group (itemgroup2) that frequently appears in the same order as the selected anchor group(s).",
                        "orders_with_anchor": "Number of distinct orders (DO numbers) that contain the anchor product group set.",
                        "orders_with_group": "Number of distinct orders that contain the recommended group.",
                        "orders_with_anchor_and_group": "Number of distinct orders that contain both the anchor group set and the recommended group.",
                        "support": "Share of all orders that contain both the anchor groups and the recommended group.",
                        "confidence": "Likelihood the recommended group appears given the anchor group set.",
                        "lift": "Strength above baseline: confidence divided by the overall probability of the recommended group appearing in any order.",
                        "total_qty_in_anchor_orders": "Total quantity of items belonging to the recommended group within anchor orders where the group appears.",
                        "total_value_in_anchor_orders": "Total final_sales of items belonging to the recommended group within anchor orders where the group appears.",
                        },
                        toggles=[
                            "Anchor Rule: **ALL (intersection)** means the order must contain at least one item from every selected anchor group. **ANY (union)** means the order must contain at least one item from any selected anchor group.",
                            "Rank by: Sorts group recommendations by lift / confidence / support.",
                            "Presence-based: Group co-occurrence is based on whether a group appears in an order at least once, not the quantity."
                        ]
                    )

    # ==========================================================
    # PURCHASE PERSPECTIVE
    # ==========================================================
    else:
        if purchase_df is None or purchase_df.empty:
            st.warning("No purchase data available for this business scope.")
            return

        # --- Enforce navbar timeline on purchase_df ---
        purchase_df = purchase_df.copy()
        purchase_df["combinedate"] = pd.to_datetime(purchase_df["combinedate"], errors="coerce")

        # Prefer filtering by navbar date range if available
        date_from = selected_filters.get("date_from") or selected_filters.get("start_date")
        date_to   = selected_filters.get("date_to") or selected_filters.get("end_date")

        if date_from and date_to:
            d0 = pd.to_datetime(date_from)
            d1 = pd.to_datetime(date_to)
            purchase_df = purchase_df[(purchase_df["combinedate"] >= d0) & (purchase_df["combinedate"] <= d1)]
        else:
            # If navbar uses years selection instead of explicit dates
            years = selected_filters.get("years") or selected_filters.get("year_list")
            if years:
                years_set = set(int(y) for y in years)
                purchase_df = purchase_df[purchase_df["combinedate"].dt.year.isin(years_set)]

        if str(zid) in ("100000", "100001"):
            purchase_df = purchase_df[
                purchase_df["zid"].astype(str).isin(["100001", "100009"])
            ]
        p = purchase_df.copy()
        p["povoucher"] = p["povoucher"].astype(str)
        p["shipmentname"] = p["shipmentname"].astype(str)
        p["status"] = p["status"].astype(str)
        p["combinedate"] = _ensure_datetime(p["combinedate"])

        # Only Received shipments (phase 1)
        p_received = p[p["status"] == "5-Received"].copy()

        if p_received.empty:
            st.warning("No 'Received' shipments found (status = 5-Received).")
            return

        # --------------------------------------------
        # Build grouped shipment metadata (merge 100001 + 100009 by shipmentname + ship_date)
        # --------------------------------------------
        ship_src = p_received.copy()
        ship_src["zid"] = ship_src["zid"].astype(str)
        ship_src["povoucher"] = ship_src["povoucher"].astype(str)
        ship_src["shipmentname"] = ship_src["shipmentname"].fillna("").astype(str)
        ship_src["combinedate"] = _ensure_datetime(ship_src["combinedate"])

        # Day-level grouping date
        ship_src["ship_date"] = ship_src["combinedate"].dt.date

        # Group key: (shipmentname, ship_date)
        ship_meta = (
            ship_src.groupby(["shipmentname", "ship_date"], as_index=False)
            .agg(
                zids=("zid", lambda s: " + ".join(sorted(set(s.astype(str))))),
                povouchers=("povoucher", lambda s: " + ".join(sorted(set(s.astype(str))))),
                combinedate_min=("combinedate", "min"),
            )
            .sort_values(["combinedate_min", "shipmentname"])
        )

        # Label: "100001 + 100009 | IP--A + IP--B - NAME,YYYY-MM-DD"
        ship_meta["ship_label"] = (
            ship_meta["zids"]
            + " | "
            + ship_meta["povouchers"]
            + " - "
            + ship_meta["shipmentname"]
            + ","
            + ship_meta["ship_date"].astype(str)
        )

        ship_options = ship_meta["ship_label"].tolist()

        selected_ship = st.selectbox(
            "Select Shipment (Merged by shipmentname)",
            [""] + ship_options,
            index=0
        )

        window_days = st.selectbox(
            "Before/After Window (days)",
            [7, 14, 30, 45, 60],
            index=2
        )

        # Resolve selection back to underlying purchase rows
        selected_shipment_df = None
        if selected_ship:
            chosen = ship_meta[ship_meta["ship_label"] == selected_ship]
            if not chosen.empty:
                chosen_name = chosen.iloc[0]["shipmentname"]
                chosen_date = chosen.iloc[0]["ship_date"]

                selected_shipment_df = ship_src[
                    (ship_src["shipmentname"] == chosen_name)
                    & (ship_src["ship_date"] == chosen_date)
                ].copy()


        if not selected_ship:
            st.info("Select a shipment to run the before/after and basket analysis.")
            return

        # Use parsed zid + povoucher selection (prevents collisions across businesses)
        if selected_shipment_df is None or selected_shipment_df.empty:
            st.warning("Shipment not found in Received dataset.")
            return

        # Display label for debug: merged POV list
        pov = " + ".join(sorted(set(selected_shipment_df["povoucher"].astype(str))))
        # Shipment date = earliest combinedate among the merged shipment rows
        combinedate = selected_shipment_df["combinedate"].min()

        # Shipment items = all itemcodes in the merged shipment rows
        ship_items = set(selected_shipment_df["itemcode"].astype(str).unique().tolist())



        # Build before/after windows
        start_before = combinedate - pd.Timedelta(days=int(window_days))
        end_before = combinedate
        start_after = combinedate
        end_after = combinedate + pd.Timedelta(days=int(window_days))

        # Filter sales for before/after
        s = sales_df.copy()
        s["date"] = _ensure_datetime(s["date"])
        before = s[(s["date"] >= start_before) & (s["date"] < end_before)].copy()
        after = s[(s["date"] >= start_after) & (s["date"] <= end_after)].copy()

        st.write({
            "shipment_povoucher": pov,
            "shipment_date_combinedate": str(combinedate.date()) if pd.notna(combinedate) else "",
            "window_days": int(window_days),
            "shipment_items_count": len(ship_items),
        })

        viewA, viewB = st.tabs(["Shipment Impact (Before vs After)", "Shipment Basket + Extra Uplift"])

        # ---- View A ----
        with viewA:
            st.subheader("Shipment Impact: Item-level Before vs After")

            def item_summary(df_in: pd.DataFrame) -> pd.DataFrame:
                if df_in.empty:
                    return pd.DataFrame(columns=["itemcode", "itemname", "qty", "value"])
                tmp = df_in.copy()
                tmp["quantity"] = pd.to_numeric(tmp["quantity"], errors="coerce").fillna(0.0)
                tmp["final_sales"] = pd.to_numeric(tmp["final_sales"], errors="coerce").fillna(0.0)
                out = tmp.groupby(["itemcode", "itemname"], as_index=False).agg(
                    qty=("quantity", "sum"),
                    value=("final_sales", "sum"),
                )
                return out

            bsum = item_summary(before)
            asum = item_summary(after)

            merged = bsum.merge(asum, on=["itemcode", "itemname"], how="outer", suffixes=("_before", "_after")).fillna(0.0)
            merged["delta_qty"] = merged["qty_after"] - merged["qty_before"]
            merged["delta_value"] = merged["value_after"] - merged["value_before"]
            merged["pct_qty_change"] = np.where(merged["qty_before"] > 0, merged["delta_qty"] / merged["qty_before"], np.nan)
            merged["pct_value_change"] = np.where(merged["value_before"] > 0, merged["delta_value"] / merged["value_before"], np.nan)

            # Shipment items table
            # Shipment items table (include ALL shipment items, even if sales are 0)
            ship_base = selected_shipment_df[["itemcode", "itemname"]].copy()
            ship_base["itemcode"] = ship_base["itemcode"].astype(str)
            ship_base["itemname"] = ship_base["itemname"].fillna("").astype(str)
            ship_base = ship_base.drop_duplicates(subset=["itemcode"])

            merged_work = merged.copy()
            merged_work["itemcode"] = merged_work["itemcode"].astype(str)

            ship_tbl = ship_base.merge(
                merged_work,
                on="itemcode",
                how="left",
                suffixes=("_ship", "")
            )

            # Prefer shipment itemname if sales-side itemname is missing
            if "itemname_ship" in ship_tbl.columns and "itemname" in ship_tbl.columns:
                ship_tbl["itemname"] = ship_tbl["itemname"].fillna(ship_tbl["itemname_ship"])
                ship_tbl = ship_tbl.drop(columns=["itemname_ship"], errors="ignore")

            # Fill numeric sales fields with 0
            for c in ["qty_before", "value_before", "qty_after", "value_after", "delta_qty", "delta_value"]:
                if c in ship_tbl.columns:
                    ship_tbl[c] = pd.to_numeric(ship_tbl[c], errors="coerce").fillna(0.0)

            # Recompute pct fields safely
            ship_tbl["pct_qty_change"] = np.where(
                ship_tbl["qty_before"] > 0,
                ship_tbl["delta_qty"] / ship_tbl["qty_before"],
                np.nan
            )
            ship_tbl["pct_value_change"] = np.where(
                ship_tbl["value_before"] > 0,
                ship_tbl["delta_value"] / ship_tbl["value_before"],
                np.nan
            )

            ship_tbl = ship_tbl.sort_values("delta_value", ascending=False)

            st.markdown("**Shipment items (before vs after)**")
            st.dataframe(ship_tbl, use_container_width=True)


            # Non-shipment uplift table
            non_tbl = merged[~merged["itemcode"].astype(str).isin(ship_items)].copy()
            non_tbl = non_tbl.sort_values("delta_value", ascending=False)
            st.markdown("**Non-shipment items uplift (before vs after)**")
            st.dataframe(non_tbl)

            _table_writeup(
                "Shipment Impact — Shipment Items (Before vs After)",
                columns={
                    "itemcode": "Item code for the shipment line (packcode-based mapping already applied where relevant).",
                    "itemname": "Item name.",
                    "qty_before": "Total sales quantity of this item in the **before window** (combinedate − N days to combinedate).",
                    "qty_after": "Total sales quantity of this item in the **after window** (combinedate to combinedate + N days).",
                    "delta_qty": "qty_after − qty_before.",
                    "pct_qty_change": "Percentage change in quantity from before to after. Blank/NA if qty_before is zero.",
                    "value_before": "Total sales value (final_sales) of this item in the before window.",
                    "value_after": "Total sales value (final_sales) of this item in the after window.",
                    "delta_value": "value_after − value_before.",
                    "pct_value_change": "Percentage change in value from before to after. Blank/NA if value_before is zero.",
                },
                toggles=[
                    "Shipment dropdown: Selects a single shipment (povoucher − shipmentname).",
                    "Window (days): Controls the before/after period. Applied symmetrically around `combinedate`.",
                    "Received-only: This analysis is valid only for shipments marked as Received."
                ]
            )


        # ---- View B ----
        with viewB:
            st.subheader("Shipment Basket (after window) + Extra Products Uplift")

            # Find after-window orders that contain any shipment item
            if after.empty:
                st.warning("No sales in the after window.")
            else:
                # Build order sets in after
                after_work = after.copy()
                after_work["itemcode"] = after_work["itemcode"].astype(str)
                order_sets = _build_order_sets(after_work, "order_id", "itemcode")

                shipment_orders = set()
                for oid, items in order_sets.items():
                    if items.intersection(ship_items):
                        shipment_orders.add(oid)

                if not shipment_orders:
                    st.warning("No after-window orders contain shipment items.")
                else:
                    # Build recommendations among non-shipment items, anchored on "shipment item present"
                    # We'll pass anchor_mode ANY with anchor items = shipment items (because condition is any shipment item present)
                    rec = _basket_recommendations_items(
                        df=after_work,
                        order_id_col="order_id",
                        item_col="itemcode",
                        qty_col="quantity",
                        value_col="final_sales",
                        anchor_items=list(ship_items),
                        anchor_mode="ANY",
                        top_n=200,
                    )
                    if rec.empty:
                        st.warning("No extra-product basket results found.")
                    else:
                        # Limit to non-shipment items explicitly
                        rec = rec[~rec["itemcode"].astype(str).isin(ship_items)].copy()
                        st.markdown("**Extra products (not in shipment) most likely to sell with shipment items**")
                        st.dataframe(rec)

                        # Extra products uplift table (top K)
                        top_k = st.slider("Top extra products to compare (before vs after)", min_value=10, max_value=60, value=30, step=5)
                        top_items = rec["itemcode"].astype(str).head(int(top_k)).tolist()

                        def uplift_for(items: list[str]) -> pd.DataFrame:
                            if not items:
                                return pd.DataFrame()
                            b = before[before["itemcode"].astype(str).isin(items)].copy()
                            a = after[after["itemcode"].astype(str).isin(items)].copy()

                            bsum = item_summary(b).rename(columns={"qty": "qty_before", "value": "value_before"})
                            asum = item_summary(a).rename(columns={"qty": "qty_after", "value": "value_after"})
                            u = bsum.merge(asum, on=["itemcode", "itemname"], how="outer").fillna(0.0)
                            u["delta_qty"] = u["qty_after"] - u["qty_before"]
                            u["delta_value"] = u["value_after"] - u["value_before"]
                            u["pct_qty_change"] = np.where(u["qty_before"] > 0, u["delta_qty"] / u["qty_before"], np.nan)
                            u["pct_value_change"] = np.where(u["value_before"] > 0, u["delta_value"] / u["value_before"], np.nan)
                            return u

                        uplift_tbl = uplift_for(top_items)
                        if uplift_tbl.empty:
                            st.warning("No uplift data for the selected extra products in the before/after windows.")
                        else:
                            # Add lift/confidence context from basket rec
                            ctx = rec[["itemcode", "support", "confidence", "lift"]].copy()
                            uplift_tbl = uplift_tbl.merge(ctx, on="itemcode", how="left")
                            uplift_tbl = uplift_tbl.sort_values("delta_value", ascending=False)
                            st.markdown("**Extra products uplift (before vs after) with basket context**")
                            st.dataframe(uplift_tbl)
                            _table_writeup(
                                "Shipment Impact — Non-shipment Items (Before vs After)",
                                columns={
                                    "itemcode": "Item code for products **not** included in the selected shipment.",
                                    "itemname": "Item name.",
                                    "itemgroup": "Product group (itemgroup2).",
                                    "qty_before": "Total quantity sold in the before window.",
                                    "qty_after": "Total quantity sold in the after window.",
                                    "delta_qty": "qty_after − qty_before.",
                                    "pct_qty_change": "Percentage change in quantity from before to after. Blank/NA if qty_before is zero.",
                                    "value_before": "Total sales value (final_sales) in the before window.",
                                    "value_after": "Total sales value (final_sales) in the after window.",
                                    "delta_value": "value_after − value_before.",
                                    "pct_value_change": "Percentage change in value from before to after. Blank/NA if value_before is zero.",
                                },
                                toggles=[
                                    "Ranking (if present): Sorts by delta_value / delta_qty / pct change to highlight the strongest changes.",
                                    "This table is the main test of the hypothesis: whether non-shipment products increase after the shipment arrives."
                                ]
                            )
