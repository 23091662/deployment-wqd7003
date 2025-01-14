import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob

# Page configuration
st.set_page_config(page_title="Customer Feedback Analysis", layout="wide")

# Header
st.title("Customer Feedback Analysis")
st.markdown("Upload your customer feedback CSV file to analyze sentiments.")

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    try:
        # Create TextBlob object
        blob = TextBlob(str(text))
        # Get sentiment polarity (-1 to 1)
        polarity = blob.sentiment.polarity
        
        # Classify sentiment
        if polarity > 0.1:
            return 'positive', polarity
        elif polarity < -0.1:
            return 'negative', polarity
        else:
            return 'neutral', polarity
    except:
        return 'neutral', 0.0

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
                # Process feedback
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                sentiments = []
                polarity_scores = []
                
                # Process each text
                for idx, text in enumerate(df[text_column]):
                    sentiment, polarity = analyze_sentiment(text)
                    sentiments.append(sentiment)
                    polarity_scores.append(polarity)
                    
                    # Update progress
                    progress = (idx + 1) / len(df)
                    progress_bar.progress(progress)
                    status_text.text(f'Processed {idx + 1} of {len(df)} entries')

                # Add results to dataframe
                df['Sentiment'] = sentiments
                df['Sentiment_Score'] = polarity_scores
                
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
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    # Score distribution histogram
                    fig2 = px.histogram(
                        df,
                        x='Sentiment_Score',
                        title='Sentiment Score Distribution',
                        nbins=20
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Summary statistics
                st.subheader("Summary Statistics")
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    positive_pct = (df['Sentiment'] == 'positive').mean() * 100
                    st.metric("Positive Feedback", f"{positive_pct:.1f}%")
                
                with col4:
                    neutral_pct = (df['Sentiment'] == 'neutral').mean() * 100
                    st.metric("Neutral Feedback", f"{neutral_pct:.1f}%")
                
                with col5:
                    negative_pct = (df['Sentiment'] == 'negative').mean() * 100
                    st.metric("Negative Feedback", f"{negative_pct:.1f}%")

                # Download results
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

# Add information about the analysis
st.sidebar.markdown("""
### About the Analysis
This app uses TextBlob for sentiment analysis. The analysis provides:
- Sentiment (Positive/Neutral/Negative)
- Sentiment Score (-1 to +1)
  - Positive scores indicate positive sentiment
  - Negative scores indicate negative sentiment
  - Scores near zero indicate neutral sentiment
""")
