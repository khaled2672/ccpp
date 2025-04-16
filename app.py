import streamlit as st
import numpy as np
import joblib

# Load models and scaler
try:
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")  # Ensure scaler is loaded
except FileNotFoundError as e:
    st.error(f"❌ Missing file: {e}")
    st.stop()

# Optimal ensemble weight (from optimization, this can be dynamic or static)
best_w = 0.75  # Example, update this dynamically if you want

# Safe prediction function with error handling
def ensemble_predict(inputs_scaled):
    try:
        # Get predictions from both models
        rf_pred = rf_model.predict(inputs_scaled)
        xgb_pred = xgb_model.predict(inputs_scaled)
        st.write("✅ RF Prediction:", rf_pred)
        st.write("✅ XGB Prediction:", xgb_pred)

        # Ensure we're returning a single predicted power value
        ensemble_pred = best_w * rf_pred + (1 - best_w) * xgb_pred
        return ensemble_pred
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        return [None]

# UI
st.title("⚡ Gas Turbine Power Output Prediction")
st.markdown("Adjust the inputs to simulate power output:")

# User input sliders
ambient_temp = st.slider("Ambient Temperature (°C)", 10.0, 40.0, 25.0)
ambient_rh = st.slider("Ambient Relative Humidity (%)", 10.0, 100.0, 60.0)
ambient_pressure = st.slider("Ambient Pressure (mbar)", 990.0, 1035_
