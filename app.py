import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pyswarms.single.global_best import GlobalBestPSO

# Page config
st.set_page_config(page_title="Gas Turbine Power Prediction", layout="wide")

# Load models and transformers
try:
    rf_model = joblib.load("random_forest_model.joblib")
    xgb_model = joblib.load("xgboost_model.joblib")
    minmax_scaler = joblib.load("minmax_scaler.joblib")
    standard_scaler = joblib.load("standard_scaler.joblib")
    poly = joblib.load("poly_transformer.joblib")
    best_weight = joblib.load("best_ensemble_weight.joblib")
except Exception as e:
    st.error(f"Model/component loading failed: {str(e)}")
    st.stop()

# Title
st.title("⚡ Gas Turbine Power Output Prediction")
st.markdown("Predict power output from ambient sensor inputs and optimize operating conditions.")

# Tabs
tabs = st.tabs(["🔍 Single Prediction", "📂 Predict from CSV"])

# === Tab 1: Single Prediction ===
with tabs[0]:
    st.subheader("Input Ambient Sensor Readings")

    # User inputs for ambient conditions
    T = st.number_input("Ambient Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    RH = st.number_input("Ambient Relative Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    AP = st.number_input("Ambient Pressure (mbar)", min_value=500.0, max_value=1100.0, value=1010.0)
    EV = st.number_input("Exhaust Vacuum (cm Hg)", min_value=0.0, max_value=10.0, value=4.5)

    # Prediction button
    if st.button("Predict Power Output"):
        features = np.array([[T, RH, AP, EV]])

        # Feature transformation and scaling
        poly_features = poly.transform(features)
        scaled = minmax_scaler.transform(poly_features)
        final_input = standard_scaler.transform(scaled)

        # Model predictions
        rf_pred = rf_model.predict(final_input)[0]
        xgb_pred = xgb_model.predict(final_input)[0]
        ensemble_pred = best_weight * rf_pred + (1 - best_weight) * xgb_pred

        # Display result
        st.success(f"🔋 Predicted Power Output: {ensemble_pred:.2f} MW")

# === Tab 2: Batch Prediction ===
with tabs[1]:
    st.subheader("Upload CSV with Sensor Data")
    st.markdown("CSV must contain columns: **T, RH, AP, V** (in that order)")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            # Read CSV data
            df = pd.read_csv(uploaded_file)
            required_cols = ["T", "RH", "AP", "V"]
            if not all(col in df.columns for col in required_cols):
                st.error("CSV must contain columns: T, RH, AP, V")
            else:
                # Transform features and make predictions
                features = df[required_cols].values
                poly_features = poly.transform(features)
                scaled = minmax_scaler.transform(poly_features)
                final_input = standard_scaler.transform(scaled)

                # Get predictions from models
                rf_preds = rf_model.predict(final_input)
                xgb_preds = xgb_model.predict(final_input)
                ensemble_preds = best_weight * rf_preds + (1 - best_weight) * xgb_preds

                # Add predictions to dataframe
                df["Predicted Power (MW)"] = ensemble_preds
                st.dataframe(df)

                # CSV download button
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")

# === PSO Optimization ===
st.subheader("🧠 Optimal Conditions via PSO")
st.markdown("Find ambient settings that **maximize** predicted power output using Particle Swarm Optimization.")

# Define PSO bounds based on features
pso_bounds = {
    'Ambient Temperature': [15.0, 35.0],
    'Ambient Relative Humidity': [20.0, 80.0],
    'Ambient Pressure': [798.0, 802.0],
    'Exhaust Vacuum': [3.5, 7.0],
}
lb = np.array([v[0] for v in pso_bounds.values()])
ub = np.array([v[1] for v in pso_bounds.values()])
pso_feature_names = list(pso_bounds.keys())

def objective_function(x):
    """
    Objective function for PSO that predicts power using the ensemble model.
    It returns the negative of the predicted power (for maximization).
    """
    preds = []
    for row in x:
        raw = row.reshape(1, -1)
        poly_f = poly.transform(raw)
        minmax_f = minmax_scaler.transform(poly_f)
        final_f = standard_scaler.transform(minmax_f)

        rf = rf_model.predict(final_f)[0]
        xgb = xgb_model.predict(final_f)[0]
        y = best_weight * rf + (1 - best_weight) * xgb
        preds.append(-y)  # Negative for maximization (since PSO minimizes the objective function)
    
    return np.array(preds)

if st.button("🚀 Run PSO to Optimize"):
    with st.spinner("Running Particle Swarm Optimization..."):
        # PSO optimization
        optimizer = GlobalBestPSO(
            n_particles=30,  # Number of particles
            dimensions=len(lb),  # Number of features to optimize
            options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},  # PSO hyperparameters
            bounds=(lb, ub)  # Bounds for each feature
        )

        # Run optimization
        cost, pos = optimizer.optimize(objective_function, iters=100)

        # Get the optimal power prediction
        poly_f = poly.transform(pos.reshape(1, -1))
        minmax_f = minmax_scaler.transform(poly_f)
        final_f = standard_scaler.transform(minmax_f)

        # Get final predictions
        rf = rf_model.predict(final_f)[0]
        xgb = xgb_model.predict(final_f)[0]
        max_power = best_weight * rf + (1 - best_weight) * xgb

    # Display results
    st.success(f"🎯 Optimal Power Output: {max_power:.2f} MW")
    st.markdown("### 🌡️ Optimal Ambient Settings")
    for k, v in zip(pso_feature_names, pos):
        st.markdown(f"**{k}**: {v:.2f}")
