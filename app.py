import streamlit as st
import joblib
import numpy as np

# Load the trained XGBoost model
model = joblib.load("xgboost_fraud_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Predict whether a credit card transaction is **Fraudulent** or **Legitimate**.")

st.subheader("Enter Transaction Details:")

# Input for Time (rounded integer)
time = st.number_input("⏱️ Time (in seconds)", min_value=0, step=1, help="Time since the first transaction.")

# Input for Amount
amount = st.number_input("💰 Amount (USD)", min_value=0.0, help="Transaction amount.")

# ---- Create dummy values for V1 to V28 (either zeros or average values)
# You can update this to realistic sample values if needed
v_features = [0.0] * 28  # all V1–V28 set to 0.0 for now

# Final input data: Time + V1–V28 + Amount
input_data = np.array([[time] + v_features + [amount]])

# Prediction button
if st.button("🔍 Predict Transaction"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # fraud probability

    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"🚨 Fraudulent Transaction Detected!\n\nConfidence: **{probability:.2%}**")
    else:
        st.success(f"✅ Legitimate Transaction\n\nConfidence: **{(1 - probability):.2%}**")
