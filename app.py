import streamlit as st
import numpy as np
import joblib

# Load models and scaler
rf_model = joblib.load("rf_model.pkl")  # Ensure the model is saved as 'rf_model.pkl'
xgb_model = joblib.load("xgb_model.pkl")  # Ensure the model is saved as 'xgb_model.pkl'
scaler = joblib.load("scaler.pkl")  # Ensure the scaler is saved as 'scaler.pkl'

# Set optimal ensemble weight (from previous optimization)
best_w = 1.0  # Update this if you've optimized this weight

# Prediction function: weighted ensemble of Random Forest and XGBoost
def ensemble_predict(inputs_scaled):
    rf_pred = rf_model.predict(inputs_scaled)
    xgb_pred = xgb_model.predict(inputs_scaled)
    return best_w * rf_pred + (1 - best_w) * xgb_pred

# Streamlit UI
st.title("⚡ Gas Turbine Power Output Prediction")

st.markdown("Use the sliders to set input conditions:")

# Inputs from user (sliders for features)
ambient_temp = st.slider("Ambient Temperature (°C)", 10.0, 40.0, 25.0)
ambient_rh = st.slider("Ambient Relative Humidity (%)", 10.0, 100.0, 60.0)
ambient_pressure = st.slider("Ambient Pressure (mbar)", 990.0, 1035.0, 1013.0)
exhaust_vacuum = st.slider("Exhaust Vacuum (cm Hg)", 3.0, 12.0, 8.0)

# Prepare the user input into an array for model prediction
user_input = np.array([[ambient_temp, ambient_rh, ambient_pressure, exhaust_vacuum]])

# Scale the input
user_input_scaled = scaler.transform(user_input)

# Get the prediction
predicted_power = ensemble_predict(user_input_scaled)[0]

# Display the result in Streamlit
st.subheader("🔋 Predicted Power Output")
st.metric("MW", f"{predicted_power:.3f}")
streamlit run app.py
