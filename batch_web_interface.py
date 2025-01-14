import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

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

        # Send comments to the backend for sentiment analysis
        if st.button("Analyze Sentiments"):
            comments = df["comments"].tolist()
            try:
                # Send data to the new FastAPI for analysis
                response = requests.post(
                    "http://127.0.0.1:8001/predict_batch",
                    json={"texts": comments}
                )
                if response.status_code == 200:
                    # Process API results
                    results = response.json()["results"]
                    sentiments = [result["sentiment"] for result in results]

                    # Add sentiments to DataFrame
                    df["Sentiment"] = sentiments
                    st.write("Analysis Results:")
                    st.dataframe(df)

                    # Calculate sentiment percentages
                    sentiment_counts = df["Sentiment"].value_counts(normalize=True) * 100
                    sentiment_counts = sentiment_counts.to_dict()

                    # Display sentiment percentages
                    st.write("Sentiment Distribution (%):")
                    for sentiment, percentage in sentiment_counts.items():
                        st.write(f"{sentiment.capitalize()}: {percentage:.2f}%")

                    # Plot sentiment distribution
                    fig, ax = plt.subplots()
                    ax.pie(
                        sentiment_counts.values(),
                        labels=sentiment_counts.keys(),
                        autopct="%1.1f%%",
                        startangle=90
                    )
                    ax.axis("equal")  # Equal aspect ratio for pie chart
                    st.pyplot(fig)

                    # Option to download results
                    st.download_button(
                        label="Download Results",
                        data=df.to_csv(index=False),
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("Error in API response.")
            except Exception as e:
                st.error(f"Connection error: {e}")
