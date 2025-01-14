import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Header
st.title("Customer Feedback Analysis")
st.markdown("Upload your customer feedback CSV file to analyze sentiments.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display basic information
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Display basic statistics
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Total number of entries: {len(df)}")
            st.write(f"Number of columns: {len(df.columns)}")
            
        with col2:
            st.write("Columns in dataset:")
            st.write(", ".join(df.columns))

    except Exception as e:
        st.error(f"Error reading the file: {str(e)}")
        
# Add footer
st.markdown("---")
st.markdown("### Instructions:")
st.markdown("1. Prepare your CSV file with customer feedback")
st.markdown("2. Upload the file using the file uploader above")
st.markdown("3. View the basic statistics and data preview")
