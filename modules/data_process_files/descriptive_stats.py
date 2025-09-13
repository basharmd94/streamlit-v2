import streamlit as st
import pandas as pd
from modules.visualization_files import common_v
from modules.data_process_files import common

def show_stats(variance,std_dev,minimum,maximum,IQR,skew,kurt):
    cols = st.columns(7)

    with cols[0]:
        st.write('Variance: ',variance)

    with cols[1]:
        st.write('Standard Deviation: ',std_dev)

    with cols[2]:
        st.write('Mininmum: ',minimum)

    with cols[3]:
        st.write('Maximum: ',maximum)

    with cols[4]:
        st.write('IQR: ',IQR)
    
    with cols[5]:
        st.write('Skewness: ',skew)

    with cols[6]:
        st.write('Kurtosis: ',kurt)

    
def process_and_visualize_v3(data, pective, metric, metric2, timing, stats):

    full_group_mapping = {
        'Year/Month/Date':['year', 'month', 'DOM'],
        'Year/Month/Day':['year','month','DOW'],
        'Year/Month':['year', 'month'],
        'Year/Date':['year','DOM'],
        'Year/Day':['year','DOW'],
        'Month/Date':['month','DOM'],
        'Month/Day':['month','DOW'],
        'Month':['month'],
        'Year':['year'],
        'Date':['DOM'],
        'Day':['DOW']
    }

    aggregation_group_mapping = {
        'Year/Month/Date':['year', 'month', 'DOM'],
        'Year/Month/Day':['year','month','DOW'],
        'Year/Month':['year', 'month'],
        'Year/Date':['year','DOM'],
        'Year/Day':['year','DOW'],
        'Month/Date':['month','DOM'],
        'Month/Day':['month','DOW'],
        'Month':['month'],
        'Year':['year'],
        'Date':['DOM'],
        'Day':['DOW']
    }

    if metric in ['totalsales', 'treturnamount', 'gross_margin']:
        grouped_data = common.make_aggregates(data, full_group_mapping[timing], metric)
    else:
        grouped_data = common.find_unique_overtime(data, full_group_mapping[timing], metric)
        grouped_data = common.make_aggregates(grouped_data, aggregation_group_mapping[pective], metric)

    variance,std_dev,minimum,maximum,IQR,skew,kurt = common.find_stats(grouped_data,metric)

    # Apply stats (mean/median) if specified
    if stats == "Mean":
        grouped_data = common.find_mean(grouped_data, aggregation_group_mapping[pective], metric)
    elif stats == "Median":
        grouped_data = common.find_median(grouped_data, aggregation_group_mapping[pective], metric)
    else:
        pass

    # Ensure month order is maintained if 'month' column exists
    if 'month' in grouped_data.columns:
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        grouped_data['month'] = pd.Categorical(grouped_data['month'], categories=month_order, ordered=True)
        grouped_data = grouped_data.sort_values(by='month')
    
    # Visualize data
    common_v.plot_bar_chart(grouped_data, aggregation_group_mapping[pective][-1], metric, aggregation_group_mapping[pective][0] if len(aggregation_group_mapping[pective]) > 1 else None, f'Analysis of {metric2}, on a {timing} basis,{pective},{stats}')
    st.markdown(f"Analysis of {metric2}, on a {timing} basis,{pective},{stats}")
    st.write(grouped_data, use_container_width=True)

    show_stats(variance,std_dev,minimum,maximum,IQR,skew,kurt)
