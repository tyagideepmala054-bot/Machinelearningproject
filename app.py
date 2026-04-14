import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Movie Review Sentiment Analysis 🎬")

text = st.text_area("Enter your review")

if st.button("Predict"):
    if text.strip() != "":
        vec = vectorizer.transform([text])
        result = model.predict(vec)
        st.success(f"Sentiment: {result[0]}")
    else:
        st.warning("Please enter some text")