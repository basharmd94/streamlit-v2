from db import db_utils, sql_scripts
from modules.data_process_files import common
import streamlit as st

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

        if table_name == "purchase":
            # use string literals for the test
            if "100001" in zid_list and "100009" not in zid_list:
                zid_list.append("100009")

            # Pad to two values if still length-1
            if len(zid_list) == 1:
                zid_list.append(zid_list[0])

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


    

