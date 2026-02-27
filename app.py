import streamlit as st
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("mental_health.csv")

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Clean dataset
df["clean_text"] = df["text"].apply(preprocess)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Streamlit UI
st.title("🧠 Mental Health Sentiment Detection App")

st.write("Enter a sentence and the model will predict whether it expresses positive or negative sentiment.")

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip() != "":
        clean = preprocess(user_input)
        vector = vectorizer.transform([clean])
        prediction = model.predict(vector)[0]

        st.success(f"Prediction: {prediction}")

        if prediction == "negative":
            st.warning("The text reflects possible emotional distress.")
        else:
            st.info("The text reflects a positive emotional state.")
    else:
        st.error("Please enter some text.")