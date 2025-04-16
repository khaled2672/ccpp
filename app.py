import streamlit as st
import numpy as np
import pandas as pd
import joblib

# 🔹 Set page config
st.set_page_config(page_title="Gas Turbine Power Prediction", page_icon="⚡")

# 🔹 Load models, scaler, feature list, and ensemble weight
try:
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    FEATURES = joblib.load("features.pkl")
    FEATURES = list(FEATURES)  # Ensure it's a list of strings
    with open("best_weight.txt", "r") as f:
        best_w = float(f.read().strip())
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# 🔹 Page title
st.title("⚡ Gas Turbine Power Output Prediction")

# 🔹 Sidebar: User Inputs
st.sidebar.header("🛠️ Input Parameters")
temp = st.sidebar.slider("🌡️ Ambient Temperature (°C)", 10.0, 40.0, 25.0)
humidity = st.sidebar.slider("💧 Relative Humidity (%)", 10.0, 100.0, 60.0)
pressure = st.sidebar.slider("🌬️ Ambient Pressure (mbar)", 990.0, 1035.0, 1013.0)
vacuum = st.sidebar.slider("🌀 Exhaust Vacuum (cm Hg)", 3.0, 12.0, 8.0)

# 🔹 Prepare input
raw_input = {
    'Ambient Temperature': temp,
    'Ambient Relative Humidity': humidity,
    'Ambient Pressure': pressure,
    'Exhaust Vacuum': vacuum
}

input_df = pd.DataFrame([raw_input])[FEATURES]
scaled_input = scaler.transform(input_df)

# 🔹 Predictions
rf_pred = rf_model.predict(scaled_input)
xgb_pred = xgb_model.predict(scaled_input)
ensemble_pred = best_w * rf_pred + (1 - best_w) * xgb_pred

# 🔹 Display Results
st.subheader("🔋 Predicted Power Output (MW)")
st.metric("⚡ Ensemble Prediction", f"{ensemble_pred[0]:.3f}")

with st.expander("📊 Model Details"):
    st.write(f"• Random Forest Prediction: `{rf_pred[0]:.3f}` MW")
    st.write(f"• XGBoost Prediction: `{xgb_pred[0]:.3f}` MW")
    st.write(f"• Ensemble Weights → RF: `{best_w:.2f}`, XGB: `{1 - best_w:.2f}`")
joblib.dump(X.columns.tolist(), "features.pkl")  # 👈 Add this line


