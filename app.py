import streamlit as st
import numpy as np
import joblib

# Load models and scaler
try:
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")
    st.stop()

# Optimal ensemble weight (update if needed)
best_w = 1.0  # 100% RF, 0% XGB — change if using a mix

# Prediction function
def ensemble_predict(inputs_scaled):
    rf_pred = rf_model.predict(inputs_scaled)
    xgb_pred = xgb_model.predict(inputs_scaled)
    return best_w * rf_pred + (1 - best_w) * xgb_pred

# Streamlit UI
st.title("⚡ Gas Turbine Power Output Prediction")

st.markdown("Adjust the sliders to simulate conditions:")

# Input sliders
ambient_temp = st.slider("Ambient Temperature (°C)", 10.0, 40.0, 25.0)
ambient_rh = st.slider("Ambient Relative Humidity (%)", 10.0, 100.0, 60.0)
ambient_pressure = st.slider("Ambient Pressure (mbar)", 990.0, 1035.0, 1013.0)
exhaust_vacuum = st.slider("Exhaust Vacuum (cm Hg)", 3.0, 12.0, 8.0)

# User input array
user_input = np.array([[ambient_temp, ambient_rh, ambient_pressure, exhaust_vacuum]])

try:
    # Scale input
    user_input_scaled = scaler.transform(user_input)

    # Debug info (optional, comment out later)
    st.write("🔍 Scaled Input:", user_input_scaled)

    # Predictions
    rf_pred = rf_model.predict(user_input_scaled)
    xgb_pred = xgb_model.predict(user_input_scaled)
    st.write("🌲 RF Prediction:", rf_pred)
    st.write("📦 XGB Prediction:", xgb_pred)

    predicted_power = ensemble_predict(user_input_scaled)[0]

    # Output
    st.subheader("🔋 Predicted Power Output")
    st.metric("MW", f"{predicted_power:.3f}")

except Exception as e:
    st.error(f"⚠️ Prediction failed: {e}")
