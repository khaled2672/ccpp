import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===========================
# Load Models and Scaler
# ===========================
@st.cache_resource
def load_models():
    rf = joblib.load("rf_model.joblib")
    xgb = joblib.load("xgb_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return rf, xgb, scaler

rf_model, xgb_model, scaler = load_models()

# ===========================
# Streamlit Sidebar
# ===========================
st.sidebar.title("⚙️ Power Prediction Input")
st.sidebar.markdown("Adjust parameters below:")

ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("Ambient Relative Humidity (%)", 10.0, 100.0, 50.0)
pressure = st.sidebar.slider("Ambient Pressure (mbar)", 799.0, 1035.0, 900.0)
vacuum = st.sidebar.slider("Exhaust Vacuum (cmHg)", 3.0, 12.0, 7.5)
weight = st.sidebar.slider("Model Weight (RF vs XGB)", 0.0, 1.0, 0.5)

# ===========================
# Prepare Features
# ===========================
features = np.array([[ambient_temp, humidity, pressure, vacuum]])
scaled = scaler.transform(features)

# ===========================
# Make Predictions
# ===========================
rf_pred = rf_model.predict(scaled)[0]
xgb_pred = xgb_model.predict(scaled)[0]
ensemble_pred = weight * rf_pred + (1 - weight) * xgb_pred

# ===========================
# Display Results
# ===========================
st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict output power (in MW) using ambient and operational conditions.")

col1, col2, col3 = st.columns(3)
col1.metric("Random Forest", f"{rf_pred:.2f} MW")
col2.metric("XGBoost", f"{xgb_pred:.2f} MW")
col3.metric("Ensemble (Weight: {weight:.2f})", f"{ensemble_pred:.2f} MW", delta=f"{ensemble_pred - (rf_pred + xgb_pred)/2:.2f} vs avg")
