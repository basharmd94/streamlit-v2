import streamlit as st
from modules.analytics import Analytics
from modules import views
import pandas as pd
from io import BytesIO
from datetime import datetime
from utils.loggin_config import LogManager
from utils.utils import timed
from auth import auth_utils


@timed
@st.cache_data
def load_filter_options(tables: tuple[str], zid: str, filter_columns: list[str]):
    combined_df = None
    for table in tables:
        df = Analytics(table, zid=zid, filters={}).data
        if df is not None:
            if combined_df is None:
                combined_df = df.copy()
            else:
                combined_df = pd.concat([combined_df, df], ignore_index=True)
    if combined_df is None or combined_df.empty:
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

@timed
@st.cache_data(show_spinner=False)
def process_data(zid: str,filters: dict, tables: tuple[str], page: str = None) -> dict:
    data_dict = {}
    for table in tables:
        df = Analytics(table, zid=zid, filters=filters).data
        # Ensure every table key exists; default to empty DataFrame if fetch failed
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
        auth_utils.init_auth()

                
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
            auth_utils.render_login_page()
        else:
            self.navigation()
            # Add logout button in sidebar
            st.sidebar.markdown("---")
            if st.sidebar.button("Logout", key='logout_button'):
                auth_utils.logout()
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()

    @timed
    def home(self):
        st.write(f"Welcome to the Business Data Analysis App, {st.session_state.username}!")

    def call_if_data_loaded(self, display_fn):
        if st.session_state.get("ready_to_load") and st.session_state.get("last_data_dict"):
            display_fn(st.session_state.last_data_dict)
        else:
            st.info("Please select your filters and click '🔄 Load Data' to begin analysis.")

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
            "Manufacturing Analysis",
            "Accounting Analysis",
            "Inventory Analysis"
        ]

         # Filter menu based on user's role
        authorized_menu = [page for page in menu if auth_utils.check_page_access(page)]
        
        if not authorized_menu:
            st.error("You don't have access to any pages. Please contact your administrator.")
            return

        zid_dict = {
            '100000': 'GI Corporation',
            '100001': 'Gulshan Trading',
            '100005': 'Zepto Chemicals'
        }

        project_dict = {
            '100000': 'GI Corporation',
            '100001': 'GULSHAN TRADING',
            '100005': 'Zepto Chemicals'
        }

        selected_zid = st.sidebar.selectbox('Select Business (ZID):', list(zid_dict.keys()), format_func=lambda x: zid_dict[x])
        st.session_state.zid = selected_zid
        st.session_state.proj = project_dict[selected_zid]

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
            "Basket Analysis": ("sales", "return"),
            "Purchase Analysis": ("sales", "purchase", "stock"),
            "Customer Data View": ("sales", "return")
        }

        if self.current_page in self.page_data_map and self.current_page != "Purchase Analysis":
            tables = self.page_data_map[self.current_page]

            # Dynamically define filter columns
            if self.current_page == "Collection Analysis":
                filter_columns = ['year', 'month', 'cusname', 'area']  # Only these 4
            else:
                filter_columns = ['year', 'month', 'spname', 'cusname', 'itemname', 'area', 'itemgroup']

            filter_options = load_filter_options(tables,st.session_state.zid, filter_columns)

            st.sidebar.title("Filters")
            current_year = datetime.now().year
            all_years = sorted(filter_options.get("year", []))
            valid_years = [y for y in all_years if int(y) <= current_year]
            selected_years = st.sidebar.multiselect("Select Year", valid_years, default=valid_years[-2:] if len(valid_years) >= 2 else valid_years)
            selected_months = st.sidebar.multiselect("Select Month", filter_options.get("month", []))
            selected_salesmen = st.sidebar.multiselect("Select Salesman", filter_options.get("spname", []))
            selected_customers = st.sidebar.multiselect("Select Customer", filter_options.get("cusname", []))
            selected_products = st.sidebar.multiselect("Select Product", filter_options.get("itemname", []))
            selected_areas = st.sidebar.multiselect("Select Area", filter_options.get("area", []))
            selected_groups = st.sidebar.multiselect("Select Product Group", filter_options.get("itemgroup", []))

            if self.current_page == "Collection Analysis":
                selected_filters = {
                    "year": [int(x) for x in selected_years],
                    "month": [int(x) for x in selected_months],
                    "cusname": selected_customers,
                    "area": selected_areas
                }
            else:
                selected_filters = {
                    "year": [int(x) for x in selected_years],
                    "month": [int(x) for x in selected_months],
                    "spname": selected_salesmen,
                    "cusname": selected_customers,
                    "itemname": selected_products,
                    "area": selected_areas,
                    "itemgroup": selected_groups
                }
            
            if st.sidebar.button("🔄 Load Data"):
                st.session_state.ready_to_load = True
                st.session_state.last_data_dict = process_data(zid=st.session_state.zid, filters=selected_filters, tables=tables)
            
        elif self.current_page == "Purchase Analysis":
            selected_filters = {}
            st.session_state.last_data_dict = {}

            st.sidebar.title("Purchase Loader")

            if st.sidebar.button("🔄 Load Purchase Data"):
                purchase_tables = ("sales", "purchase", "stock")
                # If you decide to add purchase-specific filters, build them here
                st.session_state.purchase_data_dict = process_data(zid=st.session_state.zid, filters={}, tables=purchase_tables)
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

    @timed
    def overall_sales_analysis(self, data_dict):
        views.display_overall_sales_analysis_page(self.current_page, st.session_state.zid, data_dict)
    
    @timed
    def customer_data_view(self, data_dict):
        views.display_customer_data_view_page(current_page=self.current_page, zid=st.session_state.zid, data_dict=data_dict)

    @timed
    def overall_margin_analysis(self, data_dict):
        views.display_margin_analysis_page(self.current_page, st.session_state.zid, data_dict)

    @timed
    def purchase_analysis(self, data_dict):
        views.display_purchase_analysis_page(self.current_page, st.session_state.zid, data_dict)

    @timed
    def collection_analysis(self, data_dict):
        views.display_collection_analysis_page(self.current_page, st.session_state.zid, st.session_state.proj, data_dict)

    @timed
    def basket_analysis(self, data_dict=None):
        views.display_basket_analysis_page(self.current_page, st.session_state.zid)

    @timed
    def financials(self):
        views.display_financial_statements(self.current_page, st.session_state.zid)

    @timed
    def accounting_analysis(self):
        views.display_accounting_analysis_main(self.current_page, st.session_state.zid)

    @timed
    def inventory_analysis(self):
        views.display_inventory_analysis_main(self.current_page, st.session_state.zid)

if __name__ == "__main__":
    app = BaseApp()
    app.run()

