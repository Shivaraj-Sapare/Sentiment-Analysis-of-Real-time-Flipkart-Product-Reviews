# =========================================================
# Flipkart Review Sentiment Analysis + Pain Point Detection
# Dataset: YONEX MAVIS 350 (Badminton Shuttle)
# =========================================================

import os
import re
import pickle
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Detect Streamlit
RUNNING_STREAMLIT = False
try:
    import streamlit as st
    RUNNING_STREAMLIT = True
except:
    pass

stop_words = set(stopwords.words('english'))

# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(words)

# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
def train_model():

    print("Loading dataset...")

    df = pd.read_csv("reviews_data_dump/reviews_badminton/data.csv")
    df.columns = df.columns.str.strip()

    print("Columns found:", df.columns)

    df['Review text'] = df['Review text'].fillna("")
    df['Review Title'] = df['Review Title'].fillna("")

    # Combine title + review
    df["full_review"] = df["Review Title"] + " " + df["Review text"]

    # Create sentiment label
    df['sentiment'] = df['Ratings'].apply(lambda x: "positive" if x >= 4 else "negative")

    # Clean text
    print("Cleaning text...")
    df['clean_review'] = df['full_review'].apply(clean_text)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_review'])
    y = df['sentiment']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    print("Training model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluate
    pred = model.predict(X_test)

    print("\n=========== MODEL RESULT ===========")
    print(classification_report(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred, pos_label="positive"))
    print("===================================\n")

    # Save
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    print("Model & Vectorizer saved successfully!")

# ---------------------------------------------------------
# CUSTOMER PAIN POINT ANALYSIS
# ---------------------------------------------------------
def get_negative_insights():

    df = pd.read_csv("reviews_data_dump/reviews_badminton/data.csv")
    df.columns = df.columns.str.strip()

    df['Review text'] = df['Review text'].fillna("")
    df['Review Title'] = df['Review Title'].fillna("")

    df["full_review"] = df["Review Title"] + " " + df["Review text"]
    df['sentiment'] = df['Ratings'].apply(lambda x: "positive" if x >= 4 else "negative")

    df['clean_review'] = df['full_review'].apply(clean_text)

    negative_reviews = df[df['sentiment']=="negative"]['clean_review']

    words = " ".join(negative_reviews).split()
    common_words = Counter(words).most_common(15)

    return common_words

# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------
def run_streamlit():

    st.title("🏸 Flipkart Shuttle Review Sentiment Analyzer")

    if not os.path.exists("model.pkl"):
        st.warning("Training model first time...")
        train_model()

    model = pickle.load(open("model.pkl","rb"))
    vectorizer = pickle.load(open("vectorizer.pkl","rb"))

    # Prediction section
    st.subheader("Enter a Review")
    review = st.text_area("Type customer review here")

    if st.button("Predict Sentiment"):

        review_clean = clean_text(review)
        vec = vectorizer.transform([review_clean])
        pred = model.predict(vec)[0]

        if pred == "positive":
            st.success("😊 Positive Review")
        else:
            st.error("😠 Negative Review")

    # Pain Point Section
    st.markdown("---")
    st.subheader("🔍 Common Customer Complaints")

    if st.button("Show Pain Points"):
        insights = get_negative_insights()

        for word, count in insights:
            st.write(f"{word} : {count} complaints")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    if RUNNING_STREAMLIT:
        run_streamlit()
    else:
        train_model()