import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
with st.sidebar:
    st.subheader("How to Use")
    st.markdown("""
    1. Adjust sliders to set plant conditions  
    2. View the predicted power output  
    3. Compare models using the toggle  
    """)
# Load models and scaler
rf_model = joblib.load('rf_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

# Feature bounds for UI
feature_bounds = {
    'Ambient Temperature': [0.0, 50.0],
    'Ambient Relative Humidity': [10.0, 100.0],
    'Ambient Pressure': [799.0, 1035.0],
    'Exhaust Vacuum': [3.0, 12.0],
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
st.write(f"**Random Forest Prediction:** {rf_pred:.2f} MW")
st.write(f"**XGBoost Prediction:** {xgb_pred:.2f} MW")
st.write(f"**Ensemble Prediction (Weight {input_weight:.2f}):** {ensemble_pred:.2f} MW")

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

# Column mapping function
def map_columns(df):
    """Map user-uploaded CSV columns to the required features."""
    column_mapping = {
        "Ambient Temperature": ["Ambient Temperature", "Temperature", "Temp", "Amb Temp", "Ambient_Temperature"],
        "Ambient Relative Humidity": ["Relative Humidity", "Ambient Relative Humidity", "Humidity", "Rel Humidity", "Humidity (%)"],
        "Ambient Pressure": ["Ambient Pressure", "Pressure", "Amb Pressure", "Pressure (mbar)"],
        "Exhaust Vacuum": ["Exhaust Vacuum", "Vacuum", "Exhaust Vac", "Vacuum (cmHg)"]
    }

    mapped_columns = {}
    for target, possible_names in column_mapping.items():
        for name in possible_names:
            if name in df.columns:
                mapped_columns[target] = name
                break

    if len(mapped_columns) < 4:
        missing_cols = [col for col in column_mapping.keys() if col not in mapped_columns]
        st.error(f"Missing columns: {', '.join(missing_cols)}. Please upload a file with the required columns.")
        return None

    df = df.rename(columns=mapped_columns)
    return df

# Batch Prediction with CSV Upload
st.subheader("📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload input data (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data", df.head())

    df_processed = map_columns(df)
    if df_processed is not None:
        st.write("✅ Dataset Columns Mapped Successfully")

        features = df_processed[["Ambient Temperature", "Ambient Relative Humidity", "Ambient Pressure", "Exhaust Vacuum"]]
        scaled = scaler.transform(features)
        rf_preds = rf_model.predict(scaled)
        xgb_preds = xgb_model.predict(scaled)

        final_preds = input_weight * rf_preds + (1 - input_weight) * xgb_preds
        df_processed['Predicted Power (MW)'] = final_preds

        st.write("⚡ Predictions", df_processed)

        csv = df_processed.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results as CSV", data=csv, file_name="predicted_power.csv", mime='text/csv')
"""
            <style>
            body {
                background-color: #0e1117;
                color: #f1f1f1;
            }
            .stApp {
                background-color: #0e1117;
            }
            .css-1d391kg, .css-1cpxqw2 {
                color: #f1f1f1 !important;
            }
            .css-1v3fvcr {
                background-color: #262730 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: #ffffff;
                color: #000000;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

set_theme(dark_mode)
