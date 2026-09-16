import streamlit as st

from processing import usage_log


def display_home_page(username=None):
    usage_log.log_view("Home")
    st.write(f"Welcome to the Business Data Analysis App{', ' + username if username else ''}!")
