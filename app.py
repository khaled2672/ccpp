import streamlit as st
import numpy as np
import joblib

# Load models and scaler
try:
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    st.error(f"❌ Missing file: {e}")
    st.stop()

# Optimal ensemble weight (from optimization)
best_w = 1.0  # 100% RF

# Safe prediction function
def ensemble_predict(inputs_scaled):
    try:
        rf_pred = rf_model.predict(inputs_scaled)
        xgb_pred = xgb_model.predict(inputs_scaled)
        st.write("✅ RF Prediction:", rf_pred)
        st.write("✅ XGB Prediction:", xgb_pred)
        return best_w * rf_pred + (1 - best_w) * xgb_pred
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        return [None]

# UI
st.title("⚡ Gas Turbine Power Output Prediction")
st.markdown("Adjust the inputs to simulate power output:")

# User input sliders
ambient_temp = st.slider("Ambient Temperature (°C)", 10.0, 40.0, 25.0)
ambient_rh = st.slider("Ambient Relative Humidity (%)", 10.0, 100.0, 60.0)
ambient_pressure = st.slider("Ambient Pressure (mbar)", 990.0, 1035.0, 1013.0)
exhaust_vacuum = st.slider("Exhaust Vacuum (cm Hg)", 3.0, 12.0, 8.0)

# Create input array
user_input = np.array([[ambient_temp, ambient_rh, ambient_pressure, exhaust_vacuum]])

try:
    # Scale the input
    user_input_scaled = scaler.transform(user_input)
    st.write("🔍 Scaled Input:", user_input_scaled)

    # Predict
    prediction = ensemble_predict(user_input_scaled)
    if prediction[0] is not None:
        st.subheader("🔋 Predicted Power Output")
        st.metric("MW", f"{prediction[0]:.3f}")
    else:
        st.warning("⚠️ Model did not return a valid prediction.")

except Exception as e:
    st.error(f"💥 Unexpected error: {e}")
