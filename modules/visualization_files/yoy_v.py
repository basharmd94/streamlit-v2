import plotly.graph_objects as go
import streamlit as st
from modules.data_process_files import common

def plot_yoy(data1, data2, xaxis, color, yaxis1, yaxis2, bartitle, current_page):
    grouped_data, yaxis = common.net_sales_vertical(data1, data2, xaxis, yaxis1, yaxis2, current_page)

    unique_years = grouped_data[color].unique()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    fig = go.Figure()

    for year in unique_years:
        subset = grouped_data[grouped_data[color] == year]
        fig.add_trace(go.Bar(
            x=subset['month'],
            y=subset[yaxis],
            name=str(year)
        ))

    fig.update_layout(
        title=bartitle, 
        xaxis_title="Month", 
        yaxis_title="Net Sales", 
        barmode='group',
        xaxis=dict(tickvals=month_order, ticktext=month_order)  # This line sets the x-ticks explicitly
    )
    
    st.plotly_chart(fig, use_container_width=True)
