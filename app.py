import streamlit as st
import joblib
import numpy as np

# Load the trained XGBoost model
model = joblib.load("xgboost_fraud_model.pkl")

# Streamlit page setup
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection App")
st.markdown("Enter transaction details to predict whether it's **Fraudulent** or **Legitimate**.")

# === Input Section === #
st.subheader("🔢 Input Transaction Details")

# Initialize input list
features = []

# Add Time
features.append(st.number_input("⏱️ Time", format="%.2f"))

# Columns for V1–V28 for better layout
v_cols = st.columns(3)
for i in range(1, 29):
    with v_cols[(i - 1) % 3]:
        val = st.number_input(f"V{i}", format="%.6f", key=f"V{i}")
        features.append(val)

# Add Amount
features.append(st.number_input("💰 Transaction Amount", format="%.2f"))

# Convert to NumPy array
input_data = np.array([features])

# === Predict Button === #
st.markdown("---")
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[0][1]

        st.markdown("---")
        if prediction[0] == 1:
            st.error(f"🚨 Fraudulent Transaction Detected!")
            st.markdown(f"**Confidence:** {proba:.2%}")
        else:
            st.success("✅ Legitimate Transaction")
            st.markdown(f"**Confidence:** {(1 - proba):.2%}")

    except Exception as e:
        st.error("⚠️ An error occurred during prediction.")
        st.text(str(e))
