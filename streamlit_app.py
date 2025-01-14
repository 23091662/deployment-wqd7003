import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import plotly.express as px

# Initialize sentiment analyzer
@st.cache_resource
def load_model():
    """Load the sentiment analysis model"""
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Page configuration
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Header
st.title("Customer Feedback Analysis")
st.markdown("Upload your customer feedback CSV file to analyze sentiments.")

# Load model
with st.spinner('Loading sentiment analysis model...'):
    sentiment_analyzer = load_model()

if sentiment_analyzer is None:
    st.error("Failed to load the sentiment analysis model. Please try again later.")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display basic information
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Column selection
        text_column = st.selectbox(
            "Select the column containing the feedback text:",
            df.columns
        )

        if st.button("Analyze Sentiments"):
            with st.spinner('Analyzing sentiments...'):
                # Process feedback in batches to avoid memory issues
                batch_size = 32
                sentiments = []
                scores = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(0, len(df), batch_size):
                    batch = df[text_column][i:i+batch_size].tolist()
                    results = sentiment_analyzer(batch)
                    
                    for result in results:
                        sentiments.append(result['label'])
                        scores.append(result['score'])
                    
                    # Update progress
                    progress = (i + batch_size) / len(df)
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f'Processed {min(i + batch_size, len(df))} of {len(df)} entries')

                # Add results to dataframe
                df['Sentiment'] = sentiments
                df['Confidence'] = scores
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Display results
                st.subheader("Analysis Results")
                st.dataframe(df)

                # Create visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment distribution pie chart
                    sentiment_counts = df['Sentiment'].value_counts()
                    fig1 = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title='Sentiment Distribution'
                    )
                    st.plotly_chart(fig1)

                with col2:
                    # Confidence distribution histogram
                    fig2 = px.histogram(
                        df,
                        x='Confidence',
                        title='Confidence Score Distribution',
                        nbins=20
                    )
                    st.plotly_chart(fig2)

                # Download button for results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
        
# Add footer
st.markdown("---")
st.markdown("""
### Instructions:
1. Prepare your CSV file with customer feedback
2. Upload the file using the file uploader above
3. Select the column containing the feedback text
4. Click 'Analyze Sentiments' to process the data
5. View the results and download the analysis
""")

# Add information about the model
st.sidebar.markdown("""
### About the Analysis
This app uses DistilBERT model fine-tuned for sentiment analysis. The sentiments are classified as:
- POSITIVE: Positive feedback
- NEGATIVE: Negative feedback

The confidence score indicates how certain the model is about its prediction.
""")
