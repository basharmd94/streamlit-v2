import streamlit as st

def display_home_page(username=None):
    st.write(f"Welcome to the Business Data Analysis App{', ' + username if username else ''}!")
