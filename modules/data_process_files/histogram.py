import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from modules.data_process_files import common

def process_and_print(processed_data,selected_column,selected_metric):
    df_print = processed_data.reset_index()
    st.markdown(f"{selected_column}" + " with " + f"{selected_metric}")
    st.dataframe(df_print, width=800)
    st.markdown(common.create_download_link(df_print), unsafe_allow_html=True)

def visualize_histogram(data, selected_column, selected_metric):
    
    metric_column_mapping = {
        'Counts':'None',
        'Voucher Count':'voucher',
        'Voucher':'voucher',
        'Customer': 'cusid',
        'Products': 'itemcode',
        'Sales': 'totalsales',
        'Returns': 'treturnamt',
        'Cost': 'cost',
        'Quantity': 'quantity',
        'Margin': 'gross_margin'
    }

    mapped_column = metric_column_mapping[selected_column]
    mapped_metric = metric_column_mapping[selected_metric]

    # Handle the more specific condition first
    if selected_column in ['Sales','Returns','Cost','Margin','Quantity'] and selected_metric == 'Voucher':
        processed_data = data.groupby(mapped_metric)[mapped_column].sum()
        process_and_print(processed_data,selected_column,selected_metric)
    # Handle the next specific condition
    elif selected_column in ['Sales','Returns','Cost','Margin'] and selected_metric == 'Distribution':
        processed_data = data[mapped_column].value_counts()
        process_and_print(processed_data,selected_column,selected_metric)
    # Then handle the more general conditions
    elif selected_metric in ['Voucher','Sales','Returns','Margin','Quantity','Cost']:
        processed_data = data.groupby(mapped_column)[mapped_metric].sum()
        process_and_print(processed_data,selected_column,selected_metric)
    elif selected_metric == 'Voucher Count':
        processed_data = data.groupby(mapped_column)[mapped_metric].nunique()
        process_and_print(processed_data,selected_column,selected_metric)
    elif selected_metric == 'Counts':
        processed_data = data[mapped_column].value_counts()
        process_and_print(processed_data,selected_column,selected_metric)
    else:
        processed_data = data[mapped_column]
        process_and_print(processed_data,selected_column,selected_metric)
    
    bin_width = (processed_data.max() - processed_data.min()) / 100  # Creating approximately 50 bins
    bin_width = max(bin_width,1)
    bins = [i for i in range(0, int(processed_data.max()) + int(bin_width), int(bin_width))]
    # Visualization
    # Create the figure
    fig = go.Figure(data=[go.Histogram(x=processed_data, xbins = dict(start=processed_data.min(), end=processed_data.max(), size=bin_width),marker_color='lightcoral')])
    fig.update_layout(title_text=f'Histogram of {selected_column} - {selected_metric}',
                      xaxis_title=f"{selected_column}",
                      yaxis_title="Frequency")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment processed_data into these bins and count the occurrences
    bin_counts = pd.cut(processed_data, bins=bins, right=False, include_lowest=True).value_counts().sort_index()
    bin_counts_df = pd.DataFrame({"Bin Range": bin_counts.index.astype(str),"Count": bin_counts.values})
    
    # Remove the `[` and `)` from the Bin Range column using literal strings
    bin_counts_df['Bin Range'] = bin_counts_df['Bin Range'].str.replace('[', '', regex=False).str.replace(')', '', regex=False)

    # Split the Bin Range column into two separate columns: Start and End
    bin_counts_df[['Start', 'End']] = bin_counts_df['Bin Range'].str.split(',', expand=True)

    # Drop the original 'Bin Range' column
    bin_counts_df = bin_counts_df.drop('Bin Range', axis=1)

    # Reorder columns for clarity
    bin_counts_df = bin_counts_df[['Start', 'End', 'Count']]

    # Display the DataFrame using st.dataframe instead of st.write
    st.dataframe(bin_counts_df, use_container_width=True)
    st.markdown(common.create_download_link(bin_counts_df), unsafe_allow_html=True)