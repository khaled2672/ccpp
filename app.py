import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ========== Sidebar Instructions ==========
with st.sidebar:
    st.subheader("How to Use")
    st.markdown("""
    1. Adjust sliders to set plant conditions  
    2. View the predicted power output  
    3. Upload a CSV file for batch predictions  
    """)

# ========== Load Models ==========
rf_model = joblib.load('rf_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

# ========== Feature Bounds ==========
feature_bounds = {
    'Ambient Temperature': [0.0, 50.0],
    'Ambient Relative Humidity': [10.0, 100.0],
    'Ambient Pressure': [799.0, 1035.0],
    'Exhaust Vacuum': [3.0, 12.0],
    'Weight': [0.0, 1.0]
}

# ========== Sidebar UI ==========
st.sidebar.title("⚙️ Input Settings")
inputs = {}
for feature, (low, high) in feature_bounds.items():
    default = (low + high) / 2
    inputs[feature] = st.sidebar.slider(feature, low, high, default)

# ========== Single Prediction ==========
feature_names = list(feature_bounds.keys())[:-1]
input_features = np.array([inputs[f] for f in feature_names]).reshape(1, -1)
input_weight = inputs['Weight']
scaled_features = scaler.transform(input_features)

rf_pred = rf_model.predict(scaled_features)[0]
xgb_pred = xgb_model.predict(scaled_features)[0]
ensemble_pred = input_weight * rf_pred + (1 - input_weight) * xgb_pred

# ========== Display Predictions ==========
st.title("🔋 CCPP Power Prediction App")
st.markdown("Predict the power output of a Combined Cycle Power Plant using ambient conditions and an ensemble of machine learning models.")

st.subheader("🔢 Single Prediction")
st.write(f"**Random Forest Prediction:** {rf_pred:.2f} MW")
st.write(f"**XGBoost Prediction:** {xgb_pred:.2f} MW")
st.write(f"**Ensemble Prediction (Weight {input_weight:.2f}):** {ensemble_pred:.2f} MW")


# ========== Column Mapping for CSV Upload ==========
def map_columns(df):
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

# ========== Batch Prediction ==========
st.subheader("📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV file with ambient conditions", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data", df.head())

    df_mapped = map_columns(df)
    if df_mapped is not None:
        st.success("✅ Dataset Columns Mapped Successfully")

        features = df_mapped[["Ambient Temperature", "Ambient Relative Humidity", "Ambient Pressure", "Exhaust Vacuum"]]
        scaled = scaler.transform(features)
        rf_preds = rf_model.predict(scaled)
        xgb_preds = xgb_model.predict(scaled)

        weight = inputs['Weight']
        final_preds = weight * rf_preds + (1 - weight) * xgb_preds

        df_mapped['Predicted Power (MW)'] = final_preds
        st.write("⚡ Predictions", df_mapped)

        csv = df_mapped.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results as CSV", data=csv, file_name="predicted_power.csv", mime='text/csv') 
        
