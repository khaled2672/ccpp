import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Gas Turbine Power Output", page_icon="⚡")

# Load models and scaler
try:
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    st.error(f"❌ Missing model file: {e}")
    st.stop()

# Load best ensemble weight
try:
    with open("best_weight.txt", "r") as f:
        best_w = float(f.read().strip())
except FileNotFoundError:
    best_w = 0.5  # default fallback weight
    st.warning("⚠️ 'best_weight.txt' not found. Using default weight: 0.5")

# App title
st.title("⚡ Gas Turbine Power Output Prediction")
st.markdown("Use the sliders below to simulate how environmental factors impact the turbine's power output.")

# Sidebar Inputs
st.sidebar.header("🧾 Input Parameters")
ambient_temp = st.sidebar.slider("🌡️ Ambient Temperature (°C)", 10.0, 40.0, 25.0)
ambient_rh = st.sidebar.slider("💧 Relative Humidity (%)", 10.0, 100.0, 60.0)
ambient_pressure = st.sidebar.slider("🌬️ Pressure (mbar)", 990.0, 1035.0, 1013.0)
exhaust_vacuum = st.sidebar.slider("🌀 Exhaust Vacuum (cm Hg)", 3.0, 12.0, 8.0)

user_input = np.array([[ambient_temp, ambient_rh, ambient_pressure, exhaust_vacuum]])

try:
    # Scale input
    user_input_scaled = scaler.transform(user_input)

    # Predictions
    rf_pred = rf_model.predict(user_input_scaled)
    xgb_pred = xgb_model.predict(user_input_scaled)
    ensemble_pred = best_w * rf_pred + (1 - best_w) * xgb_pred

    # Output
    st.subheader("🔋 Predicted Power Output (MW)")
    st.metric("Ensemble Model", f"{ensemble_pred[0]:.3f}")
    st.caption("Prediction is based on a weighted average of Random Forest and XGBoost models.")

    with st.expander("🔎 Detailed Model Breakdown"):
        st.write(f"• Random Forest Prediction: `{rf_pred[0]:.3f}` MW")
        st.write(f"• XGBoost Prediction: `{xgb_pred[0]:.3f}` MW")
        st.write(f"• Ensemble Weight → RF: `{best_w:.2f}` | XGB: `{1 - best_w:.2f}`")

except Exception as e:
    st.error(f"⚠️ Error during prediction: {e}")
