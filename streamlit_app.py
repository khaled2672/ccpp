import streamlit as st
import numpy as np
import joblib

# Load models and scaler
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("standard_scaler.pkl")

# Load ensemble weight
try:
    with open("best_weight.txt", "r") as f:
        best_w = float(f.read().strip())
except FileNotFoundError:
    st.warning("⚠️ best_weight.txt not found. Using default ensemble weight.")
    best_w = 0.5

st.title("🔋 Power Prediction Dashboard")

st.markdown("Enter the ambient conditions below to predict **Total Power Output (MW)** using an ensemble model.")

# Input fields
ambient_temp = st.number_input("Ambient Temperature (°C)", min_value=15.0, max_value=40.0, value=25.0)
relative_humidity = st.number_input("Ambient Relative Humidity (%)", min_value=20.0, max_value=90.0, value=50.0)
ambient_pressure = st.number_input("Ambient Pressure (millibar)", min_value=797.0, max_value=801.0, value=800.0)
exhaust_vacuum = st.number_input("Exhaust Vacuum (cm Hg)", min_value=3.0, max_value=12.0, value=5.0)

# Create feature vector
features = np.array([[ambient_temp, relative_humidity, ambient_pressure, exhaust_vacuum]])
features_scaled = scaler.transform(features)

# Make predictions
rf_pred = rf_model.predict(features_scaled)
xgb_pred = xgb_model.predict(features_scaled)
final_pred = best_w * rf_pred + (1 - best_w) * xgb_pred

# Show prediction
st.subheader("⚡ Predicted Power Output")
st.success(f"{final_pred[0]:.4f} MW")
