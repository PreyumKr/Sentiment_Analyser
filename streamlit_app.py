import nltk
import pandas as pd
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import os
import logging
import torch
# from concurrent.futures import ThreadPoolExecutor


# Suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Suppress Python warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Check Files Required by NLTK
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Creating Page Metadata
st.set_page_config(page_title="Review Sentiment Analysis", layout='wide')

st.title("Review Sentiment Analysis App")
st.write("Analyse sentiment of review text using VADER and RoBERTa Models")

# Sidebar Settings to show Graphs and Select Models
st.sidebar.header("Settings")
models_to_use = st.sidebar.multiselect("Select Models", ["VADER", "RoBERTa"], default=["VADER", "RoBERTa"])
show_graph = st.sidebar.checkbox("Show Sentiment Distribution Graphs", value=True)


# Load Models with caching
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_roberta():
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model


# Load Selected Model and toggle graphs
sia_vader = load_vader() if "VADER" in models_to_use else None
reberta_data = load_roberta() if "RoBERTa" in models_to_use else None
roberta_tokenizer, sia_roberta = reberta_data if reberta_data else (None, None)

# Taking Text Input
st.subheader("Enter Review Text")
placeholder = "Type your review text here..."
text_input = st.text_area("Input Text", placeholder=placeholder, height=120)

def analyze_vader_parallel(text):
    if sia_vader:
        return sia_vader.polarity_scores(text)
    return None
def analyze_roberta_parallel(text):
    if sia_roberta and roberta_tokenizer:
        tokens = roberta_tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = sia_roberta(**tokens)
        scores = output.logits.softmax(dim=1).numpy()[0]
        return scores
    return None

# Analyze Button
if st.button("Analyze Sentiment"):
    if not text_input.strip():
        st.warning("Please enter some review text first to analyze!")
    else:
        st.subheader("Sentiment Analysis Results")
        
        # Create columns for results
        col1, col2 = st.columns(2)

        # with ThreadPoolExecutor(max_workers=2) as executor:
        #     vader_future = executor.submit(analyze_vader_parallel, text_input)
        #     roberta_future = executor.submit(analyze_roberta_parallel, text_input)

        #     roberta_scores = roberta_future.result()

        # VADER Analysis
        if "VADER" in models_to_use and sia_vader:
            vader_scores = sia_vader.polarity_scores(text_input)
            # vader_scores = vader_future.result()
            
            col1.write("### VADER Sentiment Scores")
            col1.write(f"Positive: {vader_scores['pos']:.2f}")
            col1.write(f"Neutral: {vader_scores['neu']:.2f}")
            col1.write(f"Negative: {vader_scores['neg']:.2f}")

            # Analysis Label
            compound_score = vader_scores['compound']
            if compound_score >= 0.05:
                vader_sentiment = "Positive 😊️"
            elif compound_score <= -0.05:
                vader_sentiment = "Negative 😡️"
            else:
                vader_sentiment = "Neutral 😐️"

            col1.write(f"Overall Sentiment: {vader_sentiment}")
            
            # Visualize VADER Sentiment Distribution
            if show_graph:
                vader_df = pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Score': [vader_scores['pos'], vader_scores['neu'], vader_scores['neg']]
                })
                col1.bar_chart(vader_df.set_index('Sentiment'))

        # RoBERTa Analysis
        if "RoBERTa" in models_to_use and sia_roberta:
            if roberta_tokenizer and sia_roberta:
                tokens = roberta_tokenizer(text_input, return_tensors='pt')
                with torch.no_grad():
                    output = sia_roberta(**tokens)
                scores = output.logits.softmax(dim=1).numpy()[0]
                # scores = roberta_future.result()
                roberta_sentiment = ["Negative 😡️", "Neutral 😐️", "Positive 😊️"][scores.argmax()]
                col2.write("### RoBERTa Sentiment Scores")
                col2.write(f"Positive: {scores[2]:.2f}")
                col2.write(f"Neutral: {scores[1]:.2f}")
                col2.write(f"Negative: {scores[0]:.2f}")
                col2.write(f"Overall Sentiment: {roberta_sentiment}")

                # Visualize RoBERTa Sentiment Distribution
                if show_graph:
                    roberta_df = pd.DataFrame({
                        'Sentiment': ['Positive', 'Neutral', 'Negative'],
                        'Score': scores[::-1]
                    })
                    col2.bar_chart(roberta_df.set_index('Sentiment'))