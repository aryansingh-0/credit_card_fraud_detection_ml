import streamlit as st
import joblib
import numpy as np

# Load the trained XGBoost model
model = joblib.load("xgboost_fraud_model.pkl")

# Mean values for V1 to V28 from your dataset (use actual values from your dataset)
mean_values = {
    "V1": 1.168e-15, "V2": 3.417e-16, "V3": -1.380e-15, "V4": 2.074e-15,
    "V5": 9.604e-16, "V6": 1.487e-15, "V7": -5.556e-16, "V8": 1.213e-16,
    "V9": -2.406e-15, "V10": 1.249e-15, "V11": 1.571e-15, "V12": -2.065e-15,
    "V13": -2.001e-15, "V14": -7.165e-16, "V15": -2.440e-16, "V16": -1.261e-15,
    "V17": 7.825e-16, "V18": 5.411e-16, "V19": 1.196e-15, "V20": -2.299e-16,
    "V21": 1.654e-16, "V22": -3.569e-16, "V23": 2.579e-16, "V24": 4.473e-15,
    "V25": 5.341e-16, "V26": 1.683e-15, "V27": -3.660e-16, "V28": -1.227e-16
}

# Streamlit App UI
st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")
st.title("💳 Credit Card Fraud Detector")
st.markdown("Enter just **Time** and **Amount** — we'll use mean values for the rest!")

# User inputs
time = st.number_input("⏱️ Time", value=94813.86, step=100.0, format="%.2f")
amount = st.number_input("💰 Amount", value=88.35, step=10.0, format="%.2f")

# Construct input vector
features = [time]
features.extend(mean_values.values())  # V1 to V28 mean values
features.append(amount)

# Convert to NumPy array
input_data = np.array([features])

# Predict button
if st.button("🔍 Predict Fraud"):
    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"🚨 Fraudulent Transaction Detected! Confidence: {prob:.2%}")
        else:
            st.success(f"✅ Legitimate Transaction. Confidence: {(1 - prob):.2%}")

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.text(str(e))
