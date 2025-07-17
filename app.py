import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer (using pickle instead of joblib)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App title
st.title("🔍 Udaan – Job Recommender")

# Description
st.markdown("Enter your skills, interests, or background and get personalized job recommendations.")

# User input
user_input = st.text_area("💬 Describe your background (e.g., 'Python, statistics, machine learning'):")

# On submit
if st.button("Get Recommendation"):
    if user_input.strip() == "":
        st.warning("Please enter something about your background.")
    else:
        # Vectorize input
        X_input = vectorizer.transform([user_input])
        
        # Predict job title
        prediction = model.predict(X_input)[0]
        
        # Output
        st.success(f"🎯 Recommended Job Title: **{prediction}**")
