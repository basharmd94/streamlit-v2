import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
from processing import common

def plot_histogram(data_dict, y_axis_title):
    x = list(data_dict.keys())
    y = [val[0] for val in data_dict.values()]

    fig = go.Figure(data=[go.Bar(x=x, y=y)])

    fig.update_layout(title_text='Histogram of Values Over Time',
                      xaxis_title='Timeline',
                      yaxis_title=y_axis_title)

    st.plotly_chart(fig, use_container_width=True)

def plot_bar_chart(data, x_axis, y_axis, color=None, title=""):
    """
    Create an interactive bar chart using plotly.graph_objects.

    Parameters:
    - data: DataFrame containing the data to be plotted.
    - x_axis: The column in the dataframe to be used for the x-axis.
    - y_axis: The column in the dataframe to be used for the y-axis.
    - color: The column in the dataframe to be used for bar colors (for grouped bars).
    - title: The title of the chart.
    """

    if color:
        unique_colors = data[color].unique()
        traces = []

        for col in unique_colors:
            subset = data[data[color] == col]
            traces.append(go.Bar(x=subset[x_axis], y=subset[y_axis], name=str(col)))

        layout = go.Layout(title=title, barmode='group')
        fig = go.Figure(data=traces, layout=layout)

    else:
        fig = go.Figure(data=[go.Bar(x=data[x_axis], y=data[y_axis])])
        fig.update_layout(title_text=title)

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)