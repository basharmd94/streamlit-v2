import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
from modules.data_process_files import common

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

    """
    Convert month name or string to its numeric representation.
    
    Parameters:
    - month: Month name or string representation
    
    Returns:
    - Numeric month (1-12)
    """
    month_mapping = {
        'January': 1, 'Jan': 1,
        'February': 2, 'Feb': 2,
        'March': 3, 'Mar': 3,
        'April': 4, 'Apr': 4,
        'May': 5,
        'June': 6, 'Jun': 6,
        'July': 7, 'Jul': 7,
        'August': 8, 'Aug': 8,
        'September': 9, 'Sep': 9,
        'October': 10, 'Oct': 10,
        'November': 11, 'Nov': 11,
        'December': 12, 'Dec': 12
    }
    
    # If it's already a number, convert to int
    if isinstance(month, (int, float)):
        return int(month)
    
    # If it's a string, look up in mapping
    if isinstance(month, str):
        # Try exact match first
        if month in month_mapping:
            return month_mapping[month]
        
        # Try case-insensitive match
        month_lower = month.lower()
        for key, value in month_mapping.items():
            if key.lower() == month_lower:
                return value
    
    # If no match found, raise an error
    raise ValueError(f"Could not convert month: {month}")