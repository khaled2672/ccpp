import streamlit as st
import numpy as np
import joblib

# Load models and scaler
try:
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")

    with open("best_weight.txt", "r") as f:
        best_w = float(f.read().strip())

except FileNotFoundError as e:
    st.error(f"❌ Missing file: {e}")
    st.stop()

st.title("⚡ Gas Turbine Power Output Prediction")
st.markdown("Adjust the inputs to simulate predicted power output from your trained ensemble model.")

# User Inputs
ambient_temp = st.slider("🌡️ Ambient Temperature (°C)", 10.0, 40.0, 25.0)
ambient_rh = st.slider("💧 Ambient Relative Humidity (%)", 10.0, 100.0, 60.0)
ambient_pressure = st.slider("🌬️ Ambient Pressure (mbar)", 990.0, 1035.0, 1013.0)
exhaust_vacuum = st.slider("🌀 Exhaust Vacuum (cm Hg)", 3.0, 12.0, 8.0)

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
    st.metric("Ensemble", f"{ensemble_pred[0]:.3f}")
    st.write("🧠 Model Predictions:")
    st.write(f"• Random Forest: {rf_pred[0]:.3f} MW")
    st.write(f"• XGBoost: {xgb_pred[0]:.3f} MW")
    st.write(f"• Ensemble Weight → RF: {best_w:.2f} | XGB: {1 - best_w:.2f}")

except Exception as e:
    st.error(f"⚠️ Error during prediction: {e}")
with open("best_weight.txt", "r") as f:
