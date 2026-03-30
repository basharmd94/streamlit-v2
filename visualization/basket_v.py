import pandas as pd
import streamlit as st
import plotly.graph_objects as go

def market_basket_heatmap(salesbasket_df):
    # Combine unique items from 'Item of Interest' and 'Item 1' to 'Item 5'
    all_unique_items = pd.concat([
        salesbasket_df['Item of Interest'],
        salesbasket_df['Item 1'],
        salesbasket_df['Item 2'],
        salesbasket_df['Item 3'],
        salesbasket_df['Item 4'],
        salesbasket_df['Item 5']
    ]).unique()

    # Create the support matrix with combined unique items as index and columns
    combined_support_matrix = pd.DataFrame(index=all_unique_items, columns=all_unique_items, data=0)

    # Populate the matrix with support values, skipping NaN values
    for index, row in salesbasket_df.iterrows():
        item_of_interest = row['Item of Interest']
        for i in range(1, 6):
            associated_item = row[f'Item {i}']
            support_value = row[f'% Support {i}']
            
            # Check for NaN values and skip them
            if pd.notna(associated_item) and pd.notna(support_value):
                combined_support_matrix.at[item_of_interest, associated_item] = support_value

    # Select a subset of the data: the top 50 items with the highest average support values
    top_items = combined_support_matrix.mean(axis=1).nlargest(100).index

    # Subset the matrix for visualization
    subset_matrix = combined_support_matrix.loc[top_items, top_items]

    # Create an interactive heatmap using plotly
    trace = go.Heatmap(
        z=subset_matrix.values,
        x=subset_matrix.columns.tolist(),
        y=subset_matrix.index.tolist(),
        colorscale='YlGnBu',
        hoverinfo='z'
    )

    layout = go.Layout(
        title="Interactive Market Basket Analysis Heatmap (Top 100 Items)",
        xaxis_title="Item",
        yaxis_title="Item of Interest",
        yaxis=dict(autorange="reversed")
    )

    fig = go.Figure(data=[trace], layout=layout)

    # Display the heatmap in Streamlit
    st.plotly_chart(fig)

