import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("Batch Sentiment Analysis for Customer Feedback")
st.markdown("Upload customer feedback for your product and analyze the sentiment distribution.")

# Simple test to ensure the app runs
st.write("App is running successfully!")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset:")
    st.dataframe(df.head())
