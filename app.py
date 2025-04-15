import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model artifacts
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Ensemble weight
best_w = 1.0  # 100% RF as per your latest result

# Prediction function
def ensemble_predict(inputs_scaled):
    rf_pred = rf_model.predict(inputs_scaled)
    xgb_pred = xgb_model.predict(inputs_scaled)
    return best_w * rf_pred + (1 - best_w) * xgb_pred

# Streamlit UI
st.title("⚡ Gas Turbine Power Output Prediction")

st.markdown("Adjust the input features below:")

ambient_temp = st.slider("Ambient Temperature (°C)", 10.0, 40.0, 25.0)
ambient_rh = st.slider("Ambient Relative Humidity (%)", 10.0, 100.0, 60.0)
ambient_pressure = st.slider("Ambient Pressure (mbar)", 990.0, 1035.0, 1013.0)
exhaust_vacuum = st.slider("Exhaust Vacuum (cm Hg)", 3.0, 12.0, 8.0)

# Combine features into array
user_input = np.array([[ambient_temp, ambient_rh, ambient_pressure, exhaust_vacuum]])
user_input_scaled = scaler.transform(user_input)

# Predict
predicted_power = ensemble_predict(user_input_scaled)[0]

st.subheader("🔋 Predicted Power Output")
st.metric("MW", f"{predicted_power:.3f}")
