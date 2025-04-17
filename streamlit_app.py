import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO

# Set page config
st.set_page_config(page_title="Power Plant Optimization", layout="wide", page_icon="⚡")

# Load models (ensure they are loaded before using them)
@st.cache_resource
def load_models():
    try:
        return {
            'rf_model': joblib.load("random_forest_model.joblib"),
            'xgb_model': joblib.load("xgboost_model.joblib"),
            'scaler': joblib.load("scaler.joblib"),
            'best_weight': np.load("best_weight.npy").item()
        }
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

models = load_models()  # Ensure models are loaded

# Define the feature bounds for PSO
feature_bounds = {
    'Ambient Temperature': [16.0, 38.0],
    'Relative Humidity': [20.0, 90.0],
    'Ambient Pressure': [797.0, 801.0],
    'Exhaust Vacuum': [3.0, 12.0],
}

# PSO objective function
def objective_function(x):
    preds = []
    for i in range(x.shape[0]):
        input_features = x[i, :-1]
        w = np.clip(x[i, -1], 0, 1)
        scaled_input = models['scaler'].transform([input_features])
        rf_pred = models['rf_model'].predict(scaled_input)[0]
        xgb_pred = models['xgb_model'].predict(scaled_input)[0]
        ensemble_pred = w * rf_pred + (1 - w) * xgb_pred
        preds.append(-ensemble_pred)  # PSO minimizes, so negate
    return np.array(preds)

# Optimization trigger button
if st.button("🔍 Optimize Inputs for Max Power"):
    st.info("Running optimization... please wait ⏳")
    
    lb = [b[0] for b in feature_bounds.values()] + [0.0]
    ub = [b[1] for b in feature_bounds.values()] + [1.0]
    bounds = (lb, ub)

    # Initialize PSO optimizer
    optimizer = GlobalBestPSO(
        n_particles=30,
        dimensions=5,
        options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
        bounds=bounds
    )

    # Run the optimizer
    cost, pos = optimizer.optimize(objective_function, iters=50)

    # Extract optimized values
    best_features = pos[:-1]
    best_weight = pos[-1]
    scaled_input = models['scaler'].transform([best_features])
    rf_pred = models['rf_model'].predict(scaled_input)[0]
    xgb_pred = models['xgb_model'].predict(scaled_input)[0]
    final_pred = best_weight * rf_pred + (1 - best_weight) * xgb_pred

    # Display results
    st.success(f"⚡ Max Predicted Power: {final_pred:.2f} MW")
    st.subheader("🔧 Optimal Input Settings:")
    for name, val in zip(feature_bounds.keys(), best_features):
        st.write(f"- {name}: {val:.2f}")
    st.write(f"⚖️ Optimized Ensemble Weight: RF = {best_weight:.2f}, XGB = {1 - best_weight:.2f}")
