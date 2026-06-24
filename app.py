import streamlit as st
from core.analytics import Analytics
from views import sales, margin, collection, basket, purchase, financial, accounting, inventory, target_management as target_mgmt_view, manufacturing
from views.home import display_home_page
import pandas as pd
from io import BytesIO
from datetime import datetime, date, timedelta
from utils.loggin_config import LogManager
from utils.utils import timed
from auth import auth


@st.cache_data(show_spinner=False, ttl=86400)
def _load_raw(tables: tuple[str], zid: str) -> pd.DataFrame:
    """
    Load and cache the full combined DataFrame for the given tables + ZID
    with NO filters.  Called once; both pass-1 and pass-2 of the sidebar
    reuse this cache entry so only one DB round-trip ever happens.
    """
    combined_df = None
    for table in tables:
        df = Analytics(table, zid=zid, filters={}).data
        if df is not None:
            combined_df = df.copy() if combined_df is None else pd.concat(
                [combined_df, df], ignore_index=True
            )
    return combined_df if combined_df is not None else pd.DataFrame()


@timed
@st.cache_data(ttl=86400)
def load_filter_options(tables: tuple[str], zid: str, filter_columns: list[str],
                        pre_filters: tuple = ()):
    """
    pre_filters: tuple of (key, tuple_of_values) pairs applied as SQL filters
    before building the option lists.  Hashable so st.cache_data works.
    Example: (("year", (2024, 2025)), ("month", (1, 2, 3)))

    Internally reuses the _load_raw cache so no second DB query is needed
    for pass-2 (entity options filtered by year/month).
    """
    # Get the full cached DataFrame — zero DB cost on cache hit
    combined_df = _load_raw(tables, zid)
    if combined_df is None or combined_df.empty:
        return {}

    # Apply pre_filters in pandas instead of re-querying the DB
    if pre_filters:
        for key, values in pre_filters:
            if values and key in combined_df.columns:
                try:
                    int_vals = [int(float(v)) for v in values]
                    combined_df = combined_df[
                        combined_df[key].apply(lambda x: int(float(x)) if pd.notna(x) else -1)
                        .isin(int_vals)
                    ]
                except Exception:
                    combined_df = combined_df[combined_df[key].isin(values)]

    if combined_df.empty:
        return {}

    filter_options = {}
    cols_set = set(combined_df.columns)

    for col in filter_columns:
        if col not in cols_set:
            continue

        # 🔍 For these three, build "id - name" so Streamlit search works by either
        if col == "spname" and {"spid", "spname"} <= cols_set:
            tmp = (
                combined_df[["spid", "spname"]]
                .dropna()
                .drop_duplicates()
                .sort_values(["spid", "spname"])
            )
            values = (tmp["spid"].astype(str) + " - " + tmp["spname"].astype(str)).tolist()

        elif col == "cusname" and {"cusid", "cusname"} <= cols_set:
            tmp = (
                combined_df[["cusid", "cusname"]]
                .dropna()
                .drop_duplicates()
                .sort_values(["cusid", "cusname"])
            )
            values = (tmp["cusid"].astype(str) + " - " + tmp["cusname"].astype(str)).tolist()

        elif col == "itemname" and {"itemcode", "itemname"} <= cols_set:
            tmp = (
                combined_df[["itemcode", "itemname"]]
                .dropna()
                .drop_duplicates()
                .sort_values(["itemcode", "itemname"])
            )
            values = (tmp["itemcode"].astype(str) + " - " + tmp["itemname"].astype(str)).tolist()

        else:
            values = combined_df[col].dropna().unique().tolist()
            # Normalize year/month to integers to avoid decimals like 2024.0
            if col in ("year", "month"):
                try:
                    values = sorted({int(float(v)) for v in values})
                except Exception:
                    values = sorted(values)
            else:
                values = sorted(values)

        filter_options[col] = list(values)

    return filter_options

@st.cache_data(show_spinner=False, ttl=86400)
def load_employee_options(zid: str) -> list[str]:
    """Load salesman options for the AR Analysis sidebar."""
    from core.db import get_dataframe
    df = get_dataframe(
        "SELECT spid::text AS spid, spname FROM employee WHERE zid = %s ORDER BY spname",
        (zid,)
    )
    if df is not None and not df.empty:
        return (df["spid"].astype(str) + " - " + df["spname"].astype(str)).tolist()
    return []


@st.cache_data(show_spinner=False, ttl=86400)
def _load_zid_dict() -> dict:
    """Load business display names from the database business table."""
    from core.db import get_dataframe
    from core import queries
    sql, params = queries.get_all_businesses()
    df = get_dataframe(sql, params)
    if df is not None and not df.empty:
        return dict(zip(df["zid"].astype(str), df["org"].astype(str)))
    # Fallback: return empty dict (app will show raw ZID keys)
    return {}

@timed
@st.cache_data(show_spinner=False, ttl=86400)
def process_data(zid: str,filters: dict, tables: tuple[str], page: str = None) -> dict:
    data_dict = {}
    for table in tables:
        df = Analytics(table, zid=zid, filters=filters).data
        # Ensure every table key exists; default to empty DataFrame if fetch failed
        data_dict[table] = df if df is not None else pd.DataFrame()
    return data_dict


@timed
@st.cache_data(show_spinner=False, ttl=86400)
def load_purchase_data(zid: str, project: str) -> dict:
    """Cached loader for Purchase Analysis (same caching pattern as process_data —
    previously this loop had no caching at all, re-pulling everything on every click).

    GL tables are only fetched here when the selected zid is 100001. For any other
    zid, purchase_analysis() always overrides glheader/gldetail/glmst with 100001's
    data anyway (the overhead/profitability pools are always sourced from trading),
    so fetching them here for a different zid would just be discarded unused.
    """
    purchase_tables = ["sales", "return", "purchase", "stock", "stock_movement"]
    if str(zid) == "100001":
        purchase_tables += ["glheader_simple", "gldetail_simple", "glmst_simple"]

    data_dict = {}
    for table in purchase_tables:
        df = Analytics(table, zid=zid, project=project, filters={}).data
        data_dict[table] = df if df is not None else pd.DataFrame()
    return data_dict

def create_multi_download_buttons(data_dict: dict):
    st.sidebar.markdown("### 📥 Download Filtered Data")
    for table_name, df in data_dict.items():
        if df is not None and not df.empty:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer) as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            buffer.seek(0)
            st.sidebar.download_button(
                label=f"Download {table_name.capitalize()} Data",
                data=buffer,
                file_name=f"{table_name.lower()}_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.sidebar.caption(f"⚠️ No data for **{table_name}**")

class BaseApp:
    def __init__(self):
        st.set_page_config(
            page_title="Business Analysis",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Hide Streamlit's default menu and footer
        st.markdown("""
            <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                .stDeployButton {display:none;}
                
                /* Reduce padding at the top of the page */
                .main > div {
                    padding-top: 1rem;
                }
                
                /* Style the sidebar */
                .css-1d391kg {
                    padding-top: 1rem;
                }
                
                /* Style sidebar text */
                .css-163ttbj {
                    font-size: 14px;
                }
                
                /* Style the company info header */
                .company-header {
                    padding: 1rem;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    margin-bottom: 2rem;
                    border: 1px solid #e9ecef;
                }
                
                .company-header h3 {
                    margin: 0;
                    color: #2c3e50;
                    font-size: 1.2rem;
                }
                
                /* Add space after sidebar selections */
                .element-container:has(div.row-widget.stSelectbox) {
                    margin-bottom: 1rem;
                }
            </style>
        """, unsafe_allow_html=True)
        
        self.title = "Business Data Analysis"
        auth.init_auth()

                
        # Initialize session state variables
        if 'zid' not in st.session_state:
            st.session_state.zid = '100001'  # Default zid
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Home'
        # Keep instance attribute in sync with session state
        self.current_page = st.session_state.current_page

    @timed
    def run(self):
        if not st.session_state.authenticated:
            auth.render_login_page()
        else:
            self.navigation()
            # Add logout button in sidebar
            st.sidebar.markdown("---")
            if st.sidebar.button("🔄 Refresh All Data", key="refresh_all_data", help="Clear all cached data and reload from the database"):
                st.cache_data.clear()
                st.rerun()
            if st.sidebar.button("Logout", key='logout_button'):
                auth.logout()
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()

    @timed
    def home(self):
        display_home_page(st.session_state.username)

    def call_if_data_loaded(self, display_fn):
        data_loaded_for = st.session_state.get("ready_to_load_page")
        if (
            st.session_state.get("ready_to_load")
            and st.session_state.get("last_data_dict")
            and data_loaded_for == self.current_page
        ):
            display_fn(st.session_state.last_data_dict)
        else:
            st.info("⬅️ Please select your filters and click **🔄 Load Data** to begin analysis.")

    @timed
    def navigation(self):
        menu = [
            "Home",
            "Overall Sales Analysis",
            "Customer Data View",
            "Overall Margin Analysis",
            "Collection Analysis",
            "Purchase Analysis",
            "Basket Analysis",
            "Financial Statements",
            "Target Management",
            "Accounting Analysis",
            "Inventory Analysis",
            "Manufacturing Analysis",
        ]

         # Filter menu based on user's role
        authorized_menu = [page for page in menu if auth.check_page_access(page)]
        
        if not authorized_menu:
            st.error("You don't have access to any pages. Please contact your administrator.")
            return

        _ALLOWED_ZIDS = {'100000', '100001', '100005', '100009'}
        zid_dict = {k: v for k, v in _load_zid_dict().items() if k in _ALLOWED_ZIDS}

        project_dict = {
            '100000': 'GI Corporation',
            '100001': 'GULSHAN TRADING',
            '100005': 'Zepto Chemicals'
        }

        # Use the page_selector widget key directly — it's updated by Streamlit before
        # the script re-runs, so there's no one-render delay vs st.session_state.current_page.
        _on_financials = st.session_state.get('page_selector') == 'Financial Statements'
        selected_zid = st.sidebar.selectbox(
            'Select Business (ZID):',
            list(zid_dict.keys()),
            format_func=lambda x: zid_dict[x],
            disabled=_on_financials,
        )
        st.session_state.zid = selected_zid
        st.session_state.proj = project_dict.get(selected_zid, zid_dict.get(selected_zid, selected_zid))

        # Display the company info in a styled header
        st.markdown(f"""
            <div class="company-header">
                <h3>📊 {zid_dict[selected_zid]} (ID: {selected_zid})</h3>
            </div>
        """, unsafe_allow_html=True)

        # Navigation in sidebar
        st.session_state.current_page = st.sidebar.selectbox(
            "Menu",
            authorized_menu,
            key='page_selector'
        )
        # Mirror the selected page into the instance attribute
        self.current_page = st.session_state.current_page

        self.page_data_map = {
            "Overall Sales Analysis": ("sales", "return"),
            "Overall Margin Analysis": ("sales", "return"),
            "Collection Analysis": ("collection","sales","return","ar"),
            "Basket Analysis": ("sales", "return", "purchase", "cacus_simple"),
            "Purchase Analysis": ("sales", "purchase", "stock"),
            "Customer Data View": ("sales", "return"),
            "Target Management":  ("sales", "return", "collection")
        }

        if self.current_page in self.page_data_map and self.current_page != "Purchase Analysis":
            tables = self.page_data_map[self.current_page]

            st.sidebar.title("Filters")

            # ── All other pages — two-pass sidebar ──────────────────────
            if True:
                # Pass 1: year / month options (no entity pre-filter)
                base_opts = load_filter_options(tables, st.session_state.zid, ['year', 'month'])
                current_year = datetime.now().year
                all_years = sorted(base_opts.get("year", []))
                valid_years = [y for y in all_years if int(y) <= current_year]
                selected_years  = st.sidebar.multiselect("Select Year",  valid_years,
                                      default=valid_years[-2:] if len(valid_years) >= 2 else valid_years)
                selected_months = st.sidebar.multiselect("Select Month", base_opts.get("month", []))

                # Pass 2: entity options scoped to the selected years/months
                pre_filters_tuple = tuple(
                    (k, tuple(int(v) for v in vals))
                    for k, vals in [("year", selected_years), ("month", selected_months)]
                    if vals
                )
                if self.current_page == "Collection Analysis":
                    entity_cols = ['spname', 'cusname', 'area']
                else:
                    entity_cols = ['spname', 'cusname', 'itemname', 'area', 'itemgroup']

                entity_opts = load_filter_options(tables, st.session_state.zid,
                                                  entity_cols, pre_filters_tuple)

                selected_salesmen  = st.sidebar.multiselect("Select Salesman",       entity_opts.get("spname",    []))
                selected_customers = st.sidebar.multiselect("Select Customer",        entity_opts.get("cusname",   []))
                selected_areas     = st.sidebar.multiselect("Select Area",            entity_opts.get("area",      []))

                if self.current_page == "Collection Analysis":
                    selected_products = []
                    selected_groups   = []
                    selected_filters  = {
                        "year":    [int(x) for x in selected_years],
                        "month":   [int(x) for x in selected_months],
                        "spname":  selected_salesmen,
                        "cusname": selected_customers,
                        "area":    selected_areas,
                    }
                else:
                    selected_products = st.sidebar.multiselect("Select Product",       entity_opts.get("itemname",  []))
                    selected_groups   = st.sidebar.multiselect("Select Product Group", entity_opts.get("itemgroup", []))
                    selected_filters  = {
                        "year":      [int(x) for x in selected_years],
                        "month":     [int(x) for x in selected_months],
                        "spname":    selected_salesmen,
                        "cusname":   selected_customers,
                        "itemname":  selected_products,
                        "area":      selected_areas,
                        "itemgroup": selected_groups,
                    }

                if st.sidebar.button("🔄 Load Data"):
                    st.session_state.ready_to_load      = True
                    st.session_state.ready_to_load_page = self.current_page
                    st.session_state.last_filters       = selected_filters
                    st.session_state.last_data_dict     = process_data(
                        zid=st.session_state.zid, filters=selected_filters, tables=tables
                    )
            
        elif self.current_page == "Purchase Analysis":
            selected_filters = {}
            st.session_state.last_data_dict = {}

            st.sidebar.title("Purchase Loader")

            if st.sidebar.button("🔄 Load Purchase Data"):
                # We need returns + GL tables for batch profitability + overhead pools
                st.session_state.purchase_data_dict = load_purchase_data(
                    str(st.session_state.zid), st.session_state.proj
                )
                st.session_state.purchase_ready = True

            if st.session_state.get("purchase_ready"):
                self.purchase_analysis(st.session_state.purchase_data_dict)
            else:
                st.sidebar.info("Press **Load Purchase Data** to fetch data")
                # Render an empty page so layout doesn’t break
                st.write("⬅ Use the sidebar to load purchase data")
            return  # ⬅ prevent the main router from running twice


        if self.current_page == "Home":
            self.home()
        elif self.current_page == "Overall Sales Analysis":
            self.call_if_data_loaded(self.overall_sales_analysis)
        elif self.current_page == "Customer Data View":
            self.call_if_data_loaded(self.customer_data_view)
        elif self.current_page == "Overall Margin Analysis":
            self.call_if_data_loaded(self.overall_margin_analysis)
        elif self.current_page == "Purchase Analysis":
            self.purchase_analysis()
        elif self.current_page == "Collection Analysis":
            self.call_if_data_loaded(self.collection_analysis)
        elif self.current_page == "Basket Analysis":
            self.call_if_data_loaded(self.basket_analysis)
        elif self.current_page == "Financial Statements":
            self.financials()
        elif self.current_page == "Accounting Analysis":
            self.accounting_analysis()
        elif self.current_page == "Inventory Analysis":
            self.inventory_analysis()
        elif self.current_page == "Manufacturing Analysis":
            self.manufacturing_analysis()
        elif self.current_page == "Target Management":
            self.call_if_data_loaded(self.target_management_analysis)

    @timed
    def overall_sales_analysis(self, data_dict):
        sales.display_overall_sales_analysis_page(self.current_page, st.session_state.zid, data_dict)

    @timed
    def customer_data_view(self, data_dict):
        sales.display_customer_data_view_page(current_page=self.current_page, zid=st.session_state.zid, data_dict=data_dict)

    @timed
    def overall_margin_analysis(self, data_dict):
        margin.display_margin_analysis_page(self.current_page, st.session_state.zid, data_dict)

    @timed
    def purchase_analysis(self, data_dict):

        # --- Force GL tables for overhead explorer to always come from trading (100001) ---
        if str(st.session_state.zid) != "100001":
            gl_100001 = Analytics("glheader_simple", zid="100001", filters={}).data
            gd_100001 = Analytics("gldetail_simple", zid="100001", filters={}).data
            gm_100001 = Analytics("glmst_simple",   zid="100001", filters={}).data

            data_dict["glheader_simple"] = gl_100001 if gl_100001 is not None else pd.DataFrame()
            data_dict["gldetail_simple"] = gd_100001 if gd_100001 is not None else pd.DataFrame()
            data_dict["glmst_simple"]    = gm_100001 if gm_100001 is not None else pd.DataFrame()

        # --- Ensure stock_movement exists (load base zid if missing/empty) ---
        zid_str = str(st.session_state.zid)

        base_sm = data_dict.get("stock_movement")
        if base_sm is None or (isinstance(base_sm, pd.DataFrame) and base_sm.empty) or (not isinstance(base_sm, pd.DataFrame)):
            base_sm = Analytics("stock_movement", zid=zid_str, filters={}).data
            if base_sm is None:
                base_sm = pd.DataFrame()

        # --- Ensure stock_movement contains BOTH 100001 + 100009 for packcode-linked items ---
        if zid_str in ("100001", "100009"):
            other_zid = "100009" if zid_str == "100001" else "100001"

            other_sm = Analytics("stock_movement", zid=other_zid, filters={}).data
            if other_sm is None:
                other_sm = pd.DataFrame()

            data_dict["stock_movement"] = (
                pd.concat([base_sm, other_sm], ignore_index=True)
                .drop_duplicates()
            )
        else:
            # non-linked zids: just use base
            data_dict["stock_movement"] = base_sm

        # NOTE: we are intentionally NOT loading/using data_dict["stock"] anymore

        # Now render the view AFTER data_dict has what the view expects
        purchase.display_purchase_analysis_page(self.current_page, st.session_state.zid, data_dict)

    @timed
    def collection_analysis(self, data_dict):
        collection.display_collection_analysis_page(self.current_page, st.session_state.zid, st.session_state.proj, data_dict)

    @timed
    def basket_analysis(self, data_dict):
        basket.display_basket_analysis_page(self.current_page, st.session_state.zid, data_dict, st.session_state.get("last_filters", {}))

    @timed
    def financials(self):
        financial.display_financial_statements(self.current_page, st.session_state.zid)

    @timed
    def accounting_analysis(self):
        accounting.display_accounting_analysis_main(self.current_page, st.session_state.zid)

    @timed
    def inventory_analysis(self):
        inventory.display_inventory_analysis_main(self.current_page, st.session_state.zid)

    @timed
    def manufacturing_analysis(self):
        manufacturing.display_manufacturing_analysis_page(self.current_page, st.session_state.zid)

    @timed
    def target_management_analysis(self, data_dict):
        target_mgmt_view.display_target_management_page(self.current_page, st.session_state.zid, data_dict)

if __name__ == "__main__":
    app = BaseApp()
    app.run()

