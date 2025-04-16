# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Gas Turbine Prediction", page_icon="⚡")

# 🔹 Load models
try:
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    FEATURES = joblib.load("features.pkl")
    with open("best_weight.txt", "r") as f:
        best_w = float(f.read().strip())
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.title("⚡ Gas Turbine Power Output Prediction")

# 🔹 User Input
st.sidebar.header("Input Parameters")
temp = st.sidebar.slider("🌡️ Ambient Temperature (°C)", 10.0, 40.0, 25.0)
humidity = st.sidebar.slider("💧 Relative Humidity (%)", 10.0, 100.0, 60.0)
pressure = st.sidebar.slider("🌬️ Pressure (mbar)", 990.0, 1035.0, 1013.0)
vacuum = st.sidebar.slider("🌀 Exhaust Vacuum (cm Hg)", 3.0, 12.0, 8.0)

# 🔹 Format input to match training feature order
raw_input = {
    'Ambient Temperature': temp,
    'Ambient Relative Humidity': humidity,
    'Ambient Pressure': pressure,
    'Exhaust Vacuum': vacuum,
}
input_df = pd.DataFrame([raw_input])[FEATURES]
scaled_input = scaler.transform(input_df)

# 🔹 Predict
rf_pred = rf_model.predict(scaled_input)
xgb_pred = xgb_model.predict(scaled_input)
ensemble_pred = best_w * rf_pred + (1 - best_w) * xgb_pred

# 🔹 Display
st.subheader("🔋 Predicted Power Output (MW)")
st.metric("Ensemble Prediction", f"{ensemble_pred[0]:.3f}")

with st.expander("🔍 Model Details"):
    st.write(f"• Random Forest: `{rf_pred[0]:.3f}` MW")
    st.write(f"• XGBoost: `{xgb_pred[0]:.3f}` MW")
    st.write(f"• Ensemble Weights → RF: `{best_w:.2f}`, XGB: `{1 - best_w:.2f}`")
