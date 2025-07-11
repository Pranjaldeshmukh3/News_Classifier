import streamlit as st
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


# --- Load Saved Models ---
@st.cache_resource
def load_models():
    w2v_model = Word2Vec.load("word2vec_model.bin")
    encoder = joblib.load("encoder.pkl")
    clf_lr = joblib.load("logistic_regression.pkl")
    clf_rf = joblib.load("random_forest.pkl")
    return w2v_model, encoder, clf_lr, clf_rf




w2v_model, encoder, clf_lr, clf_rf = load_models()

# --- Preprocess Function ---
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def preprocess(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
