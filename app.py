import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load models and scaler
@st.cache_resource
def load_models():
    rf_model = joblib.load('rf_model.joblib')
    xgb_model = joblib.load('xgb_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # Load best ensemble weight
    with open("best_weight.txt", "r") as f:
        best_w = float(f.read())
    
    return rf_model, xgb_model, scaler, best_w

rf_model, xgb_model, scaler, best_w = load_models()

# App title and description
st.title("Power Plant Output Predictor")
st.markdown("""
This app predicts the power output of a combined cycle power plant based on ambient conditions.
The model uses an ensemble of Random Forest and XGBoost algorithms.
""")

# Sidebar with user inputs
st.sidebar.header("Input Parameters")

def user_input_features():
    ambient_temp = st.sidebar.slider('Ambient Temperature (°C)', 1.8, 37.1, 20.0)
    ambient_humidity = st.sidebar.slider('Ambient Relative Humidity (%)', 25.0, 100.0, 60.0)
    ambient_pressure = st.sidebar.slider('Ambient Pressure (mbar)', 992.9, 1033.3, 1013.0)
    exhaust_vacuum = st.sidebar.slider('Exhaust Vacuum (cmHg)', 25.4, 81.6, 50.0)
    
    data = {
        'Ambient Temperature': ambient_temp,
        'Ambient Relative Humidity': ambient_humidity,
        'Ambient Pressure': ambient_pressure,
        'Exhaust Vacuum': exhaust_vacuum
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display input parameters
st.subheader('Input Parameters')
st.write(input_df)

# Preprocess input and make prediction
def predict_power(input_data):
    # Scale the input
    scaled_input = scaler.transform(input_data)
    
    # Make predictions
    rf_pred = rf_model.predict(scaled_input)[0]
    xgb_pred = xgb_model.predict(scaled_input)[0]
    
    # Ensemble prediction
    ensemble_pred = best_w * rf_pred + (1 - best_w) * xgb_pred
    
    return {
        'Random Forest': rf_pred,
        'XGBoost': xgb_pred,
        'Ensemble': ensemble_pred
    }

# Make predictions immediately when inputs change
predictions = predict_power(input_df)

# Display prediction results
st.subheader('Prediction Results')
st.metric(label="Ensemble Predicted Power Output", value=f"{predictions['Ensemble']:.2f} MW")

st.write("Individual Model Predictions:")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Random Forest", value=f"{predictions['Random Forest']:.2f} MW")
with col2:
    st.metric(label="XGBoost", value=f"{predictions['XGBoost']:.2f} MW")

# Show ensemble weight
st.write(f"Ensemble Weight: Random Forest ({best_w:.2f}), XGBoost ({(1-best_w):.2f})")

# Add some explanations
st.markdown("""
### Model Information
- **Random Forest**: 100 trees with max depth of 20
- **XGBoost**: 300 trees with max depth of 9, learning rate 0.1
- **Ensemble**: Weighted combination of both models
""")
