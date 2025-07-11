import streamlit as st
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Load Saved Models ---
@st.cache_resource
def load_models():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Load TfidfVectorizer
    encoder = joblib.load("encoder.pkl")
    clf_lr = joblib.load("logistic_regression.pkl")
    clf_rf = joblib.load("random_forest.pkl")
    return vectorizer, encoder, clf_lr, clf_rf

# Load models
vectorizer, encoder, clf_lr, clf_rf = load_models()

# --- Preprocess Function ---
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def preprocess(text):
    # Clean the text
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    # Remove stopwords and stem
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# --- Streamlit App UI ---
st.title("📰 News Classifier")
st.write("Enter a news article text below to classify it into categories.")

# Text input
user_input = st.text_area("Enter News Text", height=200)

# Select classifier
classifier_choice = st.selectbox("Choose Classifier:", ["Logistic Regression", "Random Forest"])

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and vectorize the input
        cleaned_text = preprocess(user_input)
        vector = vectorizer.transform([cleaned_text])

        # Predict
        if classifier_choice == "Logistic Regression":
            prediction = clf_lr.predict(vector)
        else:
            prediction = clf_rf.predict(vector)

        # Decode predicted label
        predicted_category = encoder.inverse_transform(prediction)[0]

        st.success(f"Predicted Category: **{predicted_category}**")
