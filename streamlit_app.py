import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Page config
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")
st.title("Batch Sentiment Analysis for Customer Feedback")
st.markdown("Upload customer feedback for your product and analyze the sentiment distribution.")

@st.cache_resource
def load_model():
    """Load model and tokenizer"""
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Initialize the model
try:
    sentiment_pipeline = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

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
            try:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process comments in batches
                results = []
                total = len(df)
                
                for i, text in enumerate(df["comments"]):
                    # Update progress
                    progress = (i + 1) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing comment {i+1} of {total}")
                    
                    # Analyze sentiment
                    sentiment_result = sentiment_pipeline(text)[0]
                    score = sentiment_result['score']
                    
                    # Convert score to sentiment category
                    if score <= 0.3:
                        sentiment = "negative"
                    elif score <= 0.6:
                        sentiment = "neutral"
                    else:
                        sentiment = "positive"
                    
                    results.append(sentiment)

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Add results to DataFrame
                df["Sentiment"] = results
                
                # Display results
                st.write("Analysis Results:")
                st.dataframe(df)

                # Calculate and display sentiment distribution
                sentiment_counts = df["Sentiment"].value_counts(normalize=True) * 100
                
                # Create two columns for distribution display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Sentiment Distribution (%):")
                    for sentiment, percentage in sentiment_counts.items():
                        st.write(f"{sentiment.capitalize()}: {percentage:.2f}%")

                with col2:
                    # Plot sentiment distribution
                    fig, ax = plt.subplots()
                    ax.pie(
                        sentiment_counts.values(),
                        labels=sentiment_counts.keys(),
                        autopct="%1.1f%%",
                        startangle=90
                    )
                    ax.axis("equal")
                    st.pyplot(fig)

                # Download results
                st.download_button(
                    label="Download Results",
                    data=df.to_csv(index=False),
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

# Add footer with instructions
st.markdown("---")
st.markdown("""
### How to use:
1. Upload a CSV file containing a 'comments' column
2. Click 'Analyze Sentiments' to process the data
3. View the results and download the analyzed data
""")
