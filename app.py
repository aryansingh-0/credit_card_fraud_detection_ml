import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("xgboost_fraud_model.pkl")

st.set_page_config(page_title="Fraud Detector", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detector")
st.markdown("### Enter transaction details to check if it's Fraudulent or Legitimate.")

st.markdown("---")

# Input fields (Top correlated features only)
col1, col2 = st.columns(2)

with col1:
    V14 = st.number_input("V14 (e.g., -2.5)", value=0.0, format="%.4f")
    V17 = st.number_input("V17 (e.g., 2.3)", value=0.0, format="%.4f")
with col2:
    V12 = st.number_input("V12 (e.g., -1.2)", value=0.0, format="%.4f")

# Input as array
input_data = np.array([[V14, V17, V12]])

# Predict Button
if st.button("🔍 Predict Fraud"):
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"🚨 **Fraudulent Transaction Detected!**")
        st.markdown(f"**Confidence:** `{prob:.2%}`")
    else:
        st.success(f"✅ **Legitimate Transaction**")
        st.markdown(f"**Confidence:** `{1 - prob:.2%}`")
