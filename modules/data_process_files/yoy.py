import streamlit as st
import pandas as pd
from modules.data_process_files import common
pd.set_option('display.float_format', '{:.2f}'.format)

def display_pivot_tables(filtered_data, filtered_data_r, current_page):
    """
    Display pivoted tables for various categories in the Streamlit app.

    Args:
    - filtered_data: Filtered sales data.
    - filtered_data_r: Filtered returns data.
    - current_page: Currently selected page or filter in the Streamlit app.
    """
    pivot_args_list = [
    {'title': 'Gross Margin by Salesman', 'args': {'valuesales': 'gross_margin', 'valuereturn': 'treturnamt', 'index': ['spid', 'spname'], 'column': ['year', 'month']}},
    {'title': 'Gross Margin by Area', 'args': {'valuesales': 'gross_margin', 'valuereturn': 'treturnamt', 'index': 'area', 'column': ['year', 'month']}},
    {'title': 'Gross Margin by Customer', 'args': {'valuesales': 'gross_margin', 'valuereturn': 'treturnamt', 'index': ['cusid', 'cusname'], 'column': ['year', 'month']}},
    {'title': 'Gross Margin by Product', 'args': {'valuesales': 'gross_margin', 'valuereturn': 'treturnamt', 'index': ['itemcode', 'itemname'], 'column': ['year', 'month']}},
    {'title': 'Gross Margin by Product Group', 'args': {'valuesales': 'gross_margin', 'valuereturn': 'treturnamt', 'index': 'itemgroup', 'column': ['year', 'month']}},
    {'title': 'Net Sales by Salesman', 'args': {'valuesales': 'totalsales', 'valuereturn': 'treturnamt', 'index': ['spid', 'spname'], 'column': ['year', 'month']}},
    {'title': 'Net Sales by Area', 'args': {'valuesales': 'totalsales', 'valuereturn': 'treturnamt', 'index': 'area', 'column': ['year', 'month']}},
    {'title': 'Net Sales by Customer', 'args': {'valuesales': 'totalsales', 'valuereturn': 'treturnamt', 'index': ['cusid', 'cusname'], 'column': ['year', 'month']}},
    {'title': 'Net Sales by Product', 'args': {'valuesales': 'totalsales', 'valuereturn': 'treturnamt', 'index': ['itemcode', 'itemname'], 'column': ['year', 'month']}},
    {'title': 'Quantity Sold per Product', 'args': {'valuesales': 'quantity', 'valuereturn': 'returnqty', 'index': ['itemcode', 'itemname'], 'column': ['year', 'month']}},
    {'title': 'Net Sales by Product Group', 'args': {'valuesales': 'totalsales', 'valuereturn': 'treturnamt', 'index': 'itemgroup', 'column': ['year', 'month']}}
    ]   

    for pivot in pivot_args_list:
        pivot_table = common.net_pivot(filtered_data, filtered_data_r, pivot['args'], current_page)
        st.markdown(pivot['title'])
        st.write(pivot_table)
        st.markdown(common.create_download_link(pivot_table), unsafe_allow_html=True)

