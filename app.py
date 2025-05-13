import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Gas Turbine Power Prediction", layout="wide")

# Try to import pyswarms
try:
    import pyswarms
    from pyswarms.single.global_best import GlobalBestPSO
    pso_installed = True
except ModuleNotFoundError:
    pso_installed = False
    st.error("⚠️ `pyswarms` is not installed. PSO optimization will be unavailable.")

# Load models and transformers
try:
    rf_model = joblib.load("random_forest_model.joblib")
    xgb_model = joblib.load("xgboost_model.joblib")
    minmax_scaler = joblib.load("minmax_scaler.joblib")
    standard_scaler = joblib.load("standard_scaler.joblib")
    poly = joblib.load("poly_transformer.joblib")
    best_weight = joblib.load("best_ensemble_weight.joblib")
except Exception as e:
    st.error(f"❌ Model/component loading failed: {str(e)}")
    st.stop()

st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output from ambient sensor inputs and optimize operating conditions.")

tabs = st.tabs(["🔍 Single Prediction", "📂 Predict from CSV"])

# === Helper for preprocessing ===
def preprocess_input(features):
    poly_features = poly.transform(features)
    scaled = minmax_scaler.transform(poly_features)
    return standard_scaler.transform(scaled)

# === Tab 1: Manual input ===
with tabs[0]:
    st.subheader("Input Ambient Sensor Readings")

    T = st.number_input("Ambient Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    RH = st.number_input("Ambient Relative Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    AP = st.number_input("Ambient Pressure (mbar)", min_value=500.0, max_value=1100.0, value=1010.0)
    EV = st.number_input("Exhaust Vacuum (cm Hg)", min_value=0.0, max_value=10.0, value=4.5)

    if st.button("Predict Power Output"):
        features = np.array([[T, RH, AP, EV]])
        final_input = preprocess_input(features)

        rf_pred = rf_model.predict(final_input)[0]
        xgb_pred = xgb_model.predict(final_input)[0]
        ensemble_pred = best_weight * rf_pred + (1 - best_weight) * xgb_pred

        st.success(f"🔋 Predicted Power Output: {ensemble_pred:.2f} MW")

# === Tab 2: CSV Upload ===
with tabs[1]:
    st.subheader("Batch Prediction from CSV")
    st.markdown("Upload a CSV file containing ambient conditions. Required columns (with flexible naming):")

    column_mappings = {
        "T": ["Ambient Temperature", "Temperature", "Temp", "Amb Temp", "Ambient_Temperature", "AT", "T"],
        "RH": ["Relative Humidity", "Ambient Relative Humidity", "Humidity", "Rel Humidity", "Humidity (%)", "RH"],
        "AP": ["Ambient Pressure", "Pressure", "Amb Pressure", "Pressure (mbar)", "AP"],
        "V": ["Exhaust Vacuum", "Vacuum", "Exhaust Vac", "Vacuum (cmHg)", "EV", "V"],
    }

    def map_columns(df):
        rename_map = {}
        for standard, aliases in column_mappings.items():
            for alias in aliases:
                if alias in df.columns:
                    rename_map[alias] = standard
                    break
        return df.rename(columns=rename_map)

    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df = map_columns(df)
            required_cols = ["T", "RH", "AP", "V"]

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns after mapping: {', '.join(missing_cols)}")
            else:
                if df[required_cols].isnull().sum().sum() > 0:
                    st.warning("⚠️ Missing values detected. Dropping rows with missing data.")
                    df.dropna(subset=required_cols, inplace=True)

                inputs = preprocess_input(df[required_cols].values)
                rf_preds = rf_model.predict(inputs)
                xgb_preds = xgb_model.predict(inputs)
                ensemble_preds = best_weight * rf_preds + (1 - best_weight) * xgb_preds

                df["Predicted Power (MW)"] = ensemble_preds
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# === PSO Optimization ===
if pso_installed:
    st.subheader("🧠 Optimal Conditions via PSO")
    st.markdown("Find ambient settings that **maximize** predicted power output using Particle Swarm Optimization.")

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
        preds = []
        for row in x:
            final_input = preprocess_input(row.reshape(1, -1))
            rf = rf_model.predict(final_input)[0]
            xgb = xgb_model.predict(final_input)[0]
            y = best_weight * rf + (1 - best_weight) * xgb
            preds.append(-y)
        return np.array(preds)

    if st.button("🚀 Run PSO to Optimize"):
        with st.spinner("Running Particle Swarm Optimization..."):
            optimizer = GlobalBestPSO(
                n_particles=30,
                dimensions=len(lb),
                options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
                bounds=(lb, ub)
            )
            cost, pos = optimizer.optimize(objective_function, iters=100)

            final_input = preprocess_input(pos.reshape(1, -1))
            rf = rf_model.predict(final_input)[0]
            xgb = xgb_model.predict(final_input)[0]
            max_power = best_weight * rf + (1 - best_weight) * xgb

        st.success(f"🎯 Optimal Power Output: {max_power:.2f} MW")
        st.markdown("### 🌡️ Optimal Ambient Settings")
        for k, v in zip(pso_feature_names, pos):
            st.markdown(f"**{k}**: {v:.2f}")
else:
    st.warning("⚠️ PSO optimization requires the `pyswarms` package. Please install it to enable this feature.")
