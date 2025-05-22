import streamlit as st
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


import spacy
import subprocess
import sys
# Load SpaCy English model
# nlp = spacy.load("en_core_web_sm")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If not installed, download the model
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ----- Streamlit UI -----
st.set_page_config(page_title="News Topic Modeling", layout="centered")
st.title("📰 News Topic Modeling ")
st.write("Paste multiple news articles below (separate each with **two newlines**) and discover the top topics using Latent Dirichlet Allocation (LDA).")
st.sidebar.title("TOPIC MODELING FOR NEWS ARTICLES")
st.sidebar.info('''This project takes multiple news articles as input.

It uses Natural Language Processing (NLP) techniques to preprocess and analyze the text.

The main goal is to extract and display the main topics from these articles using Latent Dirichlet Allocation (LDA) — a popular topic modeling algorithm.

The topics are shown as concise keywords summarizing each article’s main content''')
st.sidebar.info('''The project is done by 
                
PRITHIVIRAJAN.M
                
PRAGADEESH.T
                
PRATHIPAN.V''')

# ----- Input -----
articles_input = st.text_area("📝 Enter news articles (separate by two newlines):", height=300)
num_words = st.number_input("🔢 Number of words per topic summary:", min_value=1, max_value=30, value=1)

# ----- Preprocessing with SpaCy -----
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha and len(token) > 2
    ]
    return " ".join(tokens)

# ----- LDA Execution -----
if st.button("🚀 Run Topic Modeling"):
    raw_articles = [a.strip() for a in articles_input.split('\n\n') if a.strip()]

    if not raw_articles:
        st.warning("⚠️ Please enter at least one article.")
    else:
        with st.spinner("🔍 Processing articles..."):
            cleaned_articles = [preprocess(article) for article in raw_articles]

            vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(cleaned_articles)

            lda = LatentDirichletAllocation(n_components=3, random_state=42)
            lda.fit(tfidf_matrix)

            feature_names = vectorizer.get_feature_names_out()

            st.subheader("🧠 Topics per Article:")
            for idx, article in enumerate(cleaned_articles):
                topic_distribution = lda.transform(tfidf_matrix[idx])
                top_topic_idx = topic_distribution.argmax()
                top_word_indices = lda.components_[top_topic_idx].argsort()[-num_words:][::-1]
                top_words = [feature_names[i] for i in top_word_indices]
                st.markdown(f"**Article {idx+1} Topic:** {' '.join(top_words)}")
