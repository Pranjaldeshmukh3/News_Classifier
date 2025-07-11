import streamlit as st
import numpy as np
import joblib
from gensim.models import Word2Vec

# --- Load Saved Models ---
@st.cache_resource
def load_models():
    w2v_model = Word2Vec.load("word2vec_model.bin")
    encoder = joblib.load("label_encoder.pkl")
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
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# --- Vectorize Function ---
def vectorize(sentence):
    words = sentence.split()
    word_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if not word_vecs:
        return np.zeros(w2v_model.vector_size)
    return np.mean(word_vecs, axis=0).reshape(1, -1)

# --- Streamlit UI (same as earlier) ---
# ... (keep the UI code I shared above unchanged)


# --- Streamlit UI ---
st.set_page_config(page_title="News Classifier", page_icon="📰", layout="centered")

st.title("News Article Classifier")
st.markdown(
    """
    This app predicts the category of BBC news articles using **Word2Vec embeddings** and ML models.
    - **Enter text** in the box below and select your preferred classifier.
    - Get predictions with **Logistic Regression** or **Random Forest**.
    """
)

# User Input
user_text = st.text_area("Enter a BBC news headline/article:", height=150, placeholder="Type your news article here...")
model_choice = st.selectbox("Choose a Classifier:", ("Logistic Regression", "Random Forest"))

if st.button("🔮 Predict"):
    if not user_text.strip():
        st.warning("Please enter some text to classify.")
    else:
        # Preprocess and Vectorize user input
        cleaned_text = preprocess(user_text)
        user_vec = vectorize(cleaned_text).reshape(1, -1)

        # Make Prediction
        if model_choice == "Logistic Regression":
            pred_encoded = clf_lr.predict(user_vec)[0]
        else:
            pred_encoded = clf_rf.predict(user_vec)[0]

        pred_label = encoder.inverse_transform([pred_encoded])[0]
        st.success(f"**Predicted Category:** `{pred_label.upper()}`")

# --- Show Confusion Matrix ---
if st.checkbox("Show Confusion Matrices"):
    st.markdown("### Confusion Matrix (Test Set)")
    y_pred_lr = clf_lr.predict(X_test)
    y_pred_rf = clf_rf.predict(X_test)

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    class_names = encoder.classes_

    # Plot side-by-side confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Greens", ax=axes[0])
    axes[0].set_title("Logistic Regression")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].xaxis.set_ticklabels(class_names, rotation=45)
    axes[0].yaxis.set_ticklabels(class_names, rotation=0)

    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_title("Random Forest")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].xaxis.set_ticklabels(class_names, rotation=45)
    axes[1].yaxis.set_ticklabels(class_names, rotation=0)

    plt.tight_layout()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("[GitHub Repository](https://github.com/) | Made with using Streamlit")
