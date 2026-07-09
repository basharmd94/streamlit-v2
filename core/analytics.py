import time

from core.db import get_data
from core import queries
from processing import common
import streamlit as st
import pandas as pd
from utils.loggin_config import LogManager

# core/analytics.py


class Analytics:
    def __init__(self, table_name, zid=None, project=None, filters=None):
        """
        Fetches and stores a DataFrame in self.data by:
          1) Injecting zid (and optionally project) into filters
          2) Calling the appropriate SQL builder which returns (query, params)
          3) Executing via core.db.get_data(query, *params)
        """
        _start = time.perf_counter()
        try:
            self._load(table_name, zid, project, filters)
        finally:
            elapsed = time.perf_counter() - _start
            LogManager.logger.info(f"Analytics({table_name}) took {elapsed:.4f} s")

    def _load(self, table_name, zid=None, project=None, filters=None):
        self.data = None
        filters = filters.copy() if filters else {}

        zid_list = (
            [str(z) for z in zid]
            if isinstance(zid, (list, tuple, set))
            else [str(zid)]
        )

        if table_name in ("purchase", "stock_movement", "purchase_batches"):
            if len(zid_list) == 1 and zid_list[0] == "100001":
                zid_list.append("100009")

            if len(zid_list) == 1:
                zid_list.append(zid_list[0])

            if len(zid_list) > 2:
                zid_list = zid_list[:2]

        filters["zid"] = tuple(zid_list)
        filters["project"] = project

        query_map = {
            "sales":            queries.get_sales_data,
            "sales_7day":       queries.get_sales_7day,
            "return":           queries.get_return_data,
            "stock":            queries.get_product_inventory_data,
            "stock_value":      queries.get_inventory_value_data,
            "stock_flow":       queries.get_stock_flow_data,
            "purchase":         queries.get_purchase_data,
            "collection":             queries.get_collection_data,
            "collection_period_opts": queries.get_collection_period_opts,
            "collection_entity_opts": queries.get_collection_entity_opts,
            "sales_period_opts":      queries.get_sales_period_opts,
            "sales_entity_opts":      queries.get_sales_entity_opts,
            "purchase_batches":       queries.get_purchase_batches,
            "gl_overhead_daily":      queries.get_gl_overhead_daily,
            "sales_daily_item":       queries.get_sales_daily_item,
            "returns_daily_item":     queries.get_returns_daily_item,
            "ar":               queries.get_ar_data,
            "payments":         queries.get_payment_data,
            "cacus_simple":     queries.get_cacus_simple,
            "cacus_directory":  queries.get_cacus_directory,
            "gldetail_simple":  queries.get_gldetail_simple,
            "glheader_simple":  queries.get_glheader_simple,
            "glmst_simple":     queries.get_glmst_simple,
            "casup_simple":     queries.get_casup_simple,
            "stock_movement":   queries.get_stock_movement_data,
            "caitem":           queries.get_caitem_data,
            "opmob_pending":    queries.get_opmob_pending,
            "opmob_all":        queries.get_opmob_all_data,
            "final_items_view": queries.get_final_items_view,
            "opspprc":          queries.get_opspprc_data,
            "ar_due_ledger":    queries.get_ar_due_ledger,
            "cacus_master":     queries.get_cacus_master,
            "prmst_simple":     queries.get_prmst_simple,
            "mo_header":        queries.get_mo_header_data,
            "mo_detail":        queries.get_mo_detail_data,
            "admin_expense_monthly":    queries.get_admin_expense_monthly,
            "latest_sales_collection":  queries.get_latest_sales_collection,
        }

        query_func = query_map.get(table_name)
        if not query_func:
            st.error(f"No query builder for table {table_name}")
            return

        result = query_func(filters)
        if isinstance(result, tuple) and len(result) == 2:
            sql, params = result
        else:
            sql = result
            if table_name == "purchase":
                params = list(filters["zid"])
            else:
                params = [filters["zid"][0]]

        data, cols = get_data(sql, *params)
        if data is None or cols is None:
            st.error("Error fetching data. Check filters/SQL.")
        else:
            self.data = common.to_dataframe(data, cols)


# -----------------------------
# Basket Analysis helpers
# -----------------------------

def basket_scope_zids(selected_zid: str) -> list[str]:
    """Basket-only business merge rule."""
    z = str(selected_zid)
    if z in ("100000", "100001"):
        return ["100000", "100001"]
    return [z]


@st.cache_data(show_spinner=False, ttl=86400)
def basket_load_sales(scope_zids: tuple[str, ...], filters: dict) -> pd.DataFrame:
    dfs = []
    for z in scope_zids:
        df = Analytics("sales", zid=z, filters=filters).data
        if df is not None and not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


@st.cache_data(show_spinner=False, ttl=86400)
def basket_load_purchase_for_basket(selected_zid: str) -> pd.DataFrame:
    z = str(selected_zid)

    def _load_pair(a: str, b: str) -> pd.DataFrame:
        df = Analytics("purchase", zid=[a, b], filters={}).data
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    if z in ("100000", "100001"):
        return _load_pair("100001", "100009")

    return _load_pair(z, z)


@st.cache_data(show_spinner=False, ttl=86400)
def basket_load_cacus(zid: str) -> pd.DataFrame:
    df = Analytics("cacus_simple", zid=str(zid), filters={}).data
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=86400)
def basket_prepare(selected_zid: str, filters: dict) -> dict:
    scope = basket_scope_zids(selected_zid)
    scope_tuple = tuple(scope)
    sales_df = basket_load_sales(scope_tuple, filters or {})
    purchase_df = basket_load_purchase_for_basket(str(selected_zid))

    out = {
        "scope_zids": scope,
        "sales": sales_df,
        "purchase": purchase_df,
    }

    if scope == ["100000", "100001"]:
        out["cacus_100000"] = basket_load_cacus("100000")
        out["cacus_100001"] = basket_load_cacus("100001")

    return out
