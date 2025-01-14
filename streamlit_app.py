import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Page config
st.set_page_config(
    page_title="Batch Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the model and tokenizer"""
    tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    return tokenizer, model

# Load model and tokenizer
tokenizer, model = load_model()

def predict_batch_sentiment(texts):
    """Predict sentiment for a batch of texts"""
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        results.append({"text": text, "sentiment": sentiment_map[predicted_class]})
    return results

# UI elements
st.title("Batch Sentiment Analysis for Customer Feedback")
st.markdown("Upload customer feedback for your product and analyze the sentiment distribution.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    if "comments" not in df.columns:
        st.error("The uploaded CSV must have a 'comments' column.")
    else:
        # Display uploaded data
        st.write("Uploaded Dataset:")
        st.dataframe(df.head())

        # Analyze sentiments
        if st.button("Analyze Sentiments"):
            with st.spinner('Analyzing sentiments...'):
                comments = df["comments"].tolist()
                try:
                    # Process comments
                    results = predict_batch_sentiment(comments)
                    sentiments = [result["sentiment"] for result in results]

                    # Add sentiments to DataFrame
                    df["Sentiment"] = sentiments
                    
                    # Create tabs for different views
                    tab1, tab2 = st.tabs(["Results", "Distribution"])
                    
                    with tab1:
                        st.write("Analysis Results:")
                        st.dataframe(df)
                        
                        # Option to download results
                        st.download_button(
                            label="Download Results",
                            data=df.to_csv(index=False),
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )

                    with tab2:
                        # Calculate sentiment percentages
                        sentiment_counts = df["Sentiment"].value_counts(normalize=True) * 100
                        sentiment_counts = sentiment_counts.to_dict()

                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Display sentiment percentages
                            st.write("Sentiment Distribution (%):")
                            for sentiment, percentage in sentiment_counts.items():
                                st.write(f"{sentiment.capitalize()}: {percentage:.2f}%")

                        with col2:
                            # Plot sentiment distribution
                            fig, ax = plt.subplots()
                            colors = ['#ff9999', '#66b3ff', '#99ff99']
                            ax.pie(
                                sentiment_counts.values(),
                                labels=sentiment_counts.keys(),
                                autopct="%1.1f%%",
                                startangle=90,
                                colors=colors
                            )
                            ax.axis("equal")
                            st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error during analysis: {e}")
