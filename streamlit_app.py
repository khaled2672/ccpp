import streamlit as st
import numpy as np
import pandas as pd
from utils import load_models, ensemble_predict

st.set_page_config(page_title="🔌 Power Predictor", layout="centered")

rf, xgb, scaler, best_w = load_models()

st.title("🔌 Power Prediction App")

# User Inputs
temp = st.slider("Ambient Temperature (°C)", 15.0, 40.0, 25.78)
humidity = st.slider("Ambient Relative Humidity (%)", 20.0, 100.0, 60.17)
pressure = st.slider("Ambient Pressure (mmHg)", 795.0, 805.0, 798.98)
vacuum = st.slider("Exhaust Vacuum (inHg)", 3.0, 12.0, 10.43)

input_data = np.array([[temp, humidity, pressure, vacuum]])
pred_power = ensemble_predict(rf, xgb, scaler, best_w, input_data)[0]

st.metric(label="⚡ Predicted Power Output (MW)", value=f"{pred_power:.4f}")
