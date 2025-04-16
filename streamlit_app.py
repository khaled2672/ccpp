import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and tools
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# Load ensemble weight
with open("best_weight.txt", "r") as f:
    best_w = float(f.read().strip())

# App UI
st.set_page_config(page_title="Gas Turbine Power Prediction", page_icon="🔥")
st.title("⚡ Gas Turbine Power Output Prediction")

st.sidebar.header("🔧 Input Parameters")
ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", 10.0, 40.0, 25.0)
humidity = st.sidebar.slider("Relative Humidity (%)", 10.0, 100.0, 60.0)
pressure = st.sidebar.slider("Ambient Pressure (mbar)", 990.0, 1035.0, 1013.0)
vacuum = st.sidebar.slider("Exhaust Vacuum (cm Hg)", 3.0, 12.0, 8.0)

# Prepare input in correct order
input_data = pd.DataFrame([[ambient_temp, humidity, pressure, vacuum]], columns=features)

# Scale input
scaled_input = scaler.transform(input_data)

# Make predictions
rf_pred = rf_model.predict(scaled_input)
xgb_pred = xgb_model.predict(scaled_input)
ensemble_pred = best_w * rf_pred + (1 - best_w) * xgb_pred

# Show result
st.subheader("🔋 Predicted Power Output (MW)")
st.metric("Ensemble Prediction", f"{ensemble_pred[0]:.3f}")

with st.expander("📊 Model Details"):
    st.write(f"• Random Forest: `{rf_pred[0]:.3f}` MW")
    st.write(f"• XGBoost: `{xgb_pred[0]:.3f}` MW")
    st.write(f"• Ensemble Weights → RF: `{best_w:.2f}`, XGB: `{1-best_w:.2f}`")
