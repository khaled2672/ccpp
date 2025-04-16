import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Gas Turbine Power Output", page_icon="⚡")

# Load model components
try:
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    FEATURES = joblib.load("features.pkl")
    with open("best_weight.txt", "r") as f:
        best_w = float(f.read().strip())
except Exception as e:
    st.error(f"❌ Missing or broken file: {e}")
    st.stop()

# UI: Input sliders
st.title("⚡ Gas Turbine Power Output Prediction")
st.sidebar.header("🧾 Input Parameters")

ambient_temp = st.sidebar.slider("🌡️ Ambient Temperature (°C)", 10.0, 40.0, 25.0)
ambient_rh = st.sidebar.slider("💧 Relative Humidity (%)", 10.0, 100.0, 60.0)
ambient_pressure = st.sidebar.slider("🌬️ Pressure (mbar)", 990.0, 1035.0, 1013.0)
exhaust_vacuum = st.sidebar.slider("🌀 Exhaust Vacuum (cm Hg)", 3.0, 12.0, 8.0)

# Form the input dict
raw_input = {
    "Ambient Temperature": ambient_temp,
    "Ambient Relative Humidity": ambient_rh,
    "Ambient Pressure": ambient_pressure,
    "Exhaust Vacuum": exhaust_vacuum,
}

# Ensure order matches training features
input_df = pd.DataFrame([raw_input])[FEATURES]

# Scale input
input_scaled = scaler.transform(input_df)

# Make predictions
rf_pred = rf_model.predict(input_scaled)[0]
xgb_pred = xgb_model.predict(input_scaled)[0]
ensemble_pred = best_w * rf_pred + (1 - best_w) * xgb_pred

# Output
st.subheader("🔋 Predicted Power Output (MW)")
st.metric("Ensemble Model", f"{ensemble_pred:.3f}")
st.caption("Prediction is based on a weighted average of Random Forest and XGBoost models.")

with st.expander("🔎 Detailed Model Breakdown"):
    st.write(f"• Random Forest Prediction: `{rf_pred:.3f}` MW")
    st.write(f"• XGBoost Prediction: `{xgb_pred:.3f}` MW")
    st.write(f"• Ensemble Weights → RF: `{best_w:.2f}`, XGB: `{1 - best_w:.2f}`")
