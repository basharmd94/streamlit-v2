from db import db_utils, sql_scripts
from modules.data_process_files import common
import streamlit as st
import pandas as pd

# modules/analytics.py


class Analytics:
    def __init__(self, table_name, zid=None, project=None, filters=None):
        """
        Fetches and stores a DataFrame in self.data by:
          1) Injecting zid (and optionally project) into filters
          2) Calling the appropriate SQL builder which returns (query, params)
          3) Executing via db_utils.get_data(query, *params)
        """
        # Ensure the attribute always exists to avoid AttributeError in callers
        self.data = None
        filters = filters.copy() if filters else {}

        # ✨ Cast every ZID to *string* right here
        zid_list = (
            [str(z) for z in zid]                    # list / tuple / set
            if isinstance(zid, (list, tuple, set))   
            else [str(zid)]                          # single value
        )

        if table_name in ("purchase","stock_movement"):
            # Only add 100009 when the scope is *exactly* just 100001
            if len(zid_list) == 1 and zid_list[0] == "100001":
                zid_list.append("100009")

            # Pad to two values if still length-1
            if len(zid_list) == 1:
                zid_list.append(zid_list[0])

            # Safety: never allow more than 2, because purchase SQL uses 2 placeholders
            if len(zid_list) > 2:
                zid_list = zid_list[:2]

        filters["zid"] = tuple(zid_list)
        filters["project"] = project

        # 2) Map table names to builder functions
        query_map = {
            "sales":      sql_scripts.get_sales_data,
            "return":     sql_scripts.get_return_data,
            "stock":      sql_scripts.get_product_inventory_data,
            "stock_value":sql_scripts.get_inventory_value_data,
            "stock_flow": sql_scripts.get_stock_flow_data,
            "purchase":   sql_scripts.get_purchase_data,
            "collection": sql_scripts.get_collection_data,
            "ar":         sql_scripts.get_ar_data,
            "payments":   sql_scripts.get_payment_data,
            # New Entries 27/09/2025
            "cacus_simple":     sql_scripts.get_cacus_simple,
            "gldetail_simple":  sql_scripts.get_gldetail_simple,
            "glheader_simple":  sql_scripts.get_glheader_simple,
            "glmst_simple":     sql_scripts.get_glmst_simple,
            "casup_simple":     sql_scripts.get_casup_simple,
            # New Entries 28/02/2026
            "stock_movement":   sql_scripts.get_stock_movement_data,
        }

        query_func = query_map.get(table_name)
        if not query_func:
            st.error(f"No query builder for table {table_name}")
            return

        # 3) Build SQL + params
        result = query_func(filters)
        if isinstance(result, tuple) and len(result) == 2:
            sql, params = result
        else:
            sql = result

            # ─── NEW: choose param style by table ───────────────────────
            if table_name == "purchase":
                
                params = list(filters["zid"])        # e.g. ['100001', '100009']
            else:
                # one placeholder “… = %s” → need one scalar
                params = [filters["zid"][0]]         # e.g. ['100001']

        # 4) Execute and load into DataFrame
        data, cols = db_utils.get_data(sql, *params)
        if data is None or cols is None:
            st.error("Error fetching data. Check filters/SQL.")
        else:
            self.data = common.to_dataframe(data, cols)

# -----------------------------
# Basket Analysis helpers (Basket-only; does NOT modify existing pipelines)
# -----------------------------

def basket_scope_zids(selected_zid: str) -> list[str]:
    """Basket-only business merge rule."""
    z = str(selected_zid)
    if z in ("100000", "100001"):
        return ["100000", "100001"]
    return [z]


@st.cache_data(show_spinner=False)
def basket_load_sales(scope_zids: tuple[str, ...], filters: dict) -> 'common.pd.DataFrame':  # type: ignore
    """
    Load sales for one or two ZIDs and concatenate.
    Uses the existing Analytics('sales', zid=..., filters=...) call per-ZID (sales SQL is zid = %s).
    """
    dfs = []
    for z in scope_zids:
        df = Analytics("sales", zid=z, filters=filters).data
        if df is not None and not df.empty:
            dfs.append(df)
    if not dfs:
        import pandas as pd
        return pd.DataFrame()
    import pandas as pd
    return pd.concat(dfs, ignore_index=True)

@st.cache_data(show_spinner=False)
def basket_load_purchase_for_basket(selected_zid: str) -> "pd.DataFrame":
    """
    Basket Purchase loader:
      - For merged GI/Gulshan analysis, purchases must come ONLY from 100001 + 100009
      - 100000 purchases are not relevant (not international)
      - For Zepto (100005), keep its own purchase scope (100005 only) unless you later define otherwise.
    """

    z = str(selected_zid)

    def _load_pair(a: str, b: str) -> pd.DataFrame:
        df = Analytics("purchase", zid=[a, b], filters={}).data
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    if z in ("100000", "100001"):
        # Always: only Gulshan + Packaging Imports
        return _load_pair("100001", "100009")

    # Zepto or others: default to itself
    return _load_pair(z, z)

@st.cache_data(show_spinner=False)
def basket_load_cacus(zid: str) -> 'common.pd.DataFrame':  # type: ignore
    """Load simple customer master for a single zid."""
    df = Analytics("cacus_simple", zid=str(zid), filters={}).data
    import pandas as pd
    return df if df is not None else pd.DataFrame()


@st.cache_data(show_spinner=False)
def basket_prepare(selected_zid: str, filters: dict) -> dict:
    """
    Prepare all basket-only datasets using the navbar-selected filters.
    Returns:
      - sales_df: merged sales for the basket scope (100000+100001 or single zid)
      - purchase_df: purchase data for the same scope
      - scope_zids: list[str]
      - cacus_100000 / cacus_100001 (only when scope is merged)
    """
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


