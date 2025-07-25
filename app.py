import streamlit as st
import joblib
import numpy as np

# Load your trained XGBoost model
model = joblib.load("xgboost_fraud_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Predict whether a credit card transaction is **Fraudulent** or **Legitimate** based on transaction details.")

st.subheader("Enter Transaction Details:")

# Create inputs for Time, V1-V28, and Amount
time = st.number_input("⏱️ Time", help="Time elapsed between this transaction and the first transaction in the dataset.")
features = []

for i in range(1, 29):
    value = st.number_input(f"V{i}", key=f"v{i}")
    features.append(value)

amount = st.number_input("💰 Transaction Amount", help="Transaction amount in USD.")

# Combine all features in the correct order
input_data = np.array([[time] + features + [amount]])

if st.button("🔍 Predict Transaction"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probability of fraud

    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"🚨 Fraudulent Transaction Detected!\n\nConfidence: **{probability:.2%}**")
    else:
        st.success(f"✅ Legitimate Transaction\n\nConfidence: **{(1 - probability):.2%}**")
