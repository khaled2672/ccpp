import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load models and scaler
rf_model = joblib.load('rf_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

# Feature bounds for UI
feature_bounds = {
    'Ambient Temperature': [20.0, 30.0],
    'Ambient Relative Humidity': [40.0, 70.0],
    'Ambient Pressure': [999.0, 1035.0],
    'Exhaust Vacuum': [3.5, 8.0],
    'Weight': [0.0, 1.0]
}

# Sidebar UI
st.sidebar.title("⚙️ Input Settings")
inputs = {}
for feature, (low, high) in feature_bounds.items():
    default = (low + high) / 2
    inputs[feature] = st.sidebar.slider(feature, low, high, default)

# Prepare input for prediction
feature_names = list(feature_bounds.keys())[:-1]
input_features = np.array([inputs[f] for f in feature_names]).reshape(1, -1)
input_weight = inputs['Weight']

# Scale features
scaled_features = scaler.transform(input_features)

# Predict with both models
rf_pred = rf_model.predict(scaled_features)[0]
xgb_pred = xgb_model.predict(scaled_features)[0]
ensemble_pred = input_weight * rf_pred + (1 - input_weight) * xgb_pred

# Show results
st.title("🔋 CCPP Power Prediction")
st.markdown("This app predicts the power output of a Combined Cycle Power Plant based on ambient conditions and blends Random Forest & XGBoost models for better accuracy.")

st.subheader("🔢 Model Predictions")
st.write(f"Random Forest Prediction: {rf_pred:.2f} MW")
st.write(f"XGBoost Prediction: {xgb_pred:.2f} MW")
st.write(f"Ensemble Prediction (Weight {input_weight:.2f}): {ensemble_pred:.2f} MW")

# Visualization
st.subheader("📈 Feature Importance")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
rf_importance = pd.Series(rf_model.feature_importances_, index=feature_names)
xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_names)
rf_importance.plot(kind='barh', ax=ax1, title='Random Forest')
xgb_importance.plot(kind='barh', ax=ax2, title='XGBoost', color='salmon')
st.pyplot(fig)

st.markdown("---")
st.caption("Developed using Streamlit and optimized with Particle Swarm Optimization (PSO)")
