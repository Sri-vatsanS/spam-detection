import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Title
st.title("📩 Spam Message Detector")

# Text input
message = st.text_area("Enter your message:")

# Predict button
if st.button("Predict"):
    data = vectorizer.transform([message])
    prediction = model.predict(data)
    result = "❌ Spam" if prediction[0] == 1 else "✅ Ham"
    st.success(f"This message is: {result}")
