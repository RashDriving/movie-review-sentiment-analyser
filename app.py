import streamlit as st
from sentiment_model import predict_sentiment

st.set_page_config(page_title="Movie Sentiment Analyzer")

st.title("🎬 Movie Review Sentiment Analyzer")

st.write("Enter a movie review below to analyze its sentiment.")

review = st.text_area("Movie Review")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        sentiment = predict_sentiment(review)
        st.subheader(f"Sentiment: {sentiment}")
