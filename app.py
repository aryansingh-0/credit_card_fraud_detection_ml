import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("xgboost_fraud_model.pkl")

# Page Config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# Header
st.title("💳 Credit Card Fraud Detection")
st.markdown("Predict whether a transaction is **Fraudulent** or **Legitimate** using an XGBoost model.")

st.markdown("---")
st.subheader("🔢 Enter Transaction Details")

# Layout for 28 features + Amount
cols = st.columns(3)
features = []

for i in range(1, 29):
    with cols[(i - 1) % 3]:
        val = st.number_input(f"V{i}", format="%.6f", key=f"v{i}")
        features.append(val)

# Transaction Amount
Amount = st.number_input("Transaction Amount ($)", format="%.2f")
features.append(Amount)

# Convert input to NumPy array
input_data = np.array([features])

# Predict Button
st.markdown("---")
if st.button("🔍 Predict Fraud"):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"🚨 Fraudulent Transaction Detected!")
        st.markdown(f"**Confidence:** {proba:.2%}")
    else:
        st.success("✅ Legitimate Transaction")
        st.markdown(f"**Confidence:** {(1 - proba):.2%}")
