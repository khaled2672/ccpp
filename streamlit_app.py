import streamlit as st
import numpy as np
import joblib

# Load the pre-trained models and scaler
try:
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError as e:
    st.error(f"❌ Missing file: {e}")
    st.stop()

# Define optimal ensemble weight (from your optimization)
best_w = 1.0  # 100% RF (You can adjust this if needed)

# Prediction function
def ensemble_predict(inputs_scaled):
    try:
        rf_pred = rf_model.predict(inputs_scaled)
        xgb_pred = xgb_model.predict(inputs_scaled)
        return best_w * rf_pred + (1 - best_w) * xgb_pred
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        return [None]

# UI for user input (Streamlit interface)
st.title("⚡ Gas Turbine Power Output Prediction")

# Sliders for user input
ambient_temp = st.slider("Ambient Temperature (°C)", 10.0, 40.0, 25.0)
ambient_rh = st.slider("Ambient Relative Humidity (%)", 10.0, 100.0, 60.0)
ambient_pressure = st.slider("Ambient Pressure (mbar)", 990.0, 1035.0, 1013.0)
exhaust_vacuum = st.slider("Exhaust Vacuum (cm Hg)", 3.0, 12.0, 8.0)

# Create input array for prediction
user_input = np.array([[ambient_temp, ambient_rh, ambient_pressure, exhaust_vacuum]])

# Scale the input and make prediction
user_input_scaled = scaler.transform(user_input)
prediction = ensemble_predict(user_input_scaled)
# Check input scaling and prediction
user_input_scaled = scaler.transform(user_input)
st.write(f"Scaled input: {user_input_scaled}")  # Output the scaled inputs
prediction = ensemble_predict(user_input_scaled)
st.write(f"Prediction: {prediction}")  # Output the prediction result

if prediction[0] is not None:
    st.subheader("🔋 Predicted Power Output")
    st.metric("MW", f"{prediction[0]:.3f}")
else:
    st.warning("⚠️ Model did not return a valid prediction.")
rf_model = joblib.load("/path/to/your/model/rf_model.pkl")
xgb_model = joblib.load("/path/to/your/model/xgb_model.pkl")
scaler = joblib.load("/path/to/your/model/scaler.pkl")

