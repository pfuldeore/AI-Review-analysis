import pandas as pd
import streamlit as st

def load_and_preview_data(uploaded_file):
    """Load the dataset and display a preview."""
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    return df