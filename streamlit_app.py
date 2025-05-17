import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Theme configuration (only matplotlib plots)
def set_theme(dark):
    plt.style.use('dark_background' if dark else 'default')

# Cache model loading
@st.cache_resource
def load_models():
    try:
        return (
            joblib.load('rf_model.joblib'),
            joblib.load('xgb_model.joblib'),
            joblib.load('scaler.joblib')
        )
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# Case-insensitive column mapping
def map_columns(df):
    column_mapping = {
        "Ambient Temperature (°C)": ["Ambient Temperature", "Temperature", "Temp", "Amb Temp", "Ambient_Temperature", "AT"],
        "Ambient Relative Humidity (%)": ["Relative Humidity", "Ambient Relative Humidity", "Humidity", "Rel Humidity", "Humidity (%)", "RH"],
        "Ambient Pressure (mbar)": ["Ambient Pressure", "Pressure", "Amb Pressure", "Pressure (mbar)", "AP"],
        "Exhaust Vacuum (cmHg)": ["Exhaust Vacuum", "Vacuum", "Exhaust Vac", "Vacuum (cmHg)", "EV"]
    }
    mapped_columns = {}
    columns_lower = {col.lower(): col for col in df.columns}
    for target, aliases in column_mapping.items():
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower in columns_lower:
                mapped_columns[target] = columns_lower[alias_lower]
                break
    return mapped_columns

@st.cache_data
def generate_example_csv():
    example_data = {
        "Temperature (°C)": [25.0, 30.0, 27.5],
        "Humidity (%)": [60.0, 65.0, 62.5],
        "Pressure (mbar)": [1010.0, 1005.0, 1007.5],
        "Vacuum (cmHg)": [5.0, 6.0, 5.5]
    }
    return pd.DataFrame(example_data).to_csv(index=False)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Feature bounds
feature_bounds = {
    'Ambient Temperature': [0.0, 50.0],
    'Ambient Relative Humidity': [10.0, 100.0],
    'Ambient Pressure': [799.0, 1035.0],
    'Exhaust Vacuum': [3.0, 12.0],
}

# Initialize slider values in session state
for feature, (low, high) in feature_bounds.items():
    if f"slider_{feature}" not in st.session_state:
        st.session_state[f"slider_{feature}"] = (low + high) / 2

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("⚙️ CCPP Power Predictor")
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    set_theme(st.session_state.dark_mode)

    st.subheader("How to Use")
    st.markdown("""
    1. Adjust sliders to set plant conditions  
    2. View the predicted power output  
    3. Upload CSV for batch predictions  
    4. Best weight used: 65% RF + 35% XGB
    """)

    with st.spinner("Loading models..."):
        rf_model, xgb_model, scaler = load_models()

    st.subheader("Input Parameters")
    for feature, (low, high) in feature_bounds.items():
        st.session_state[f"slider_{feature}"] = st.slider(
            feature, low, high, st.session_state[f"slider_{feature}"]
        )

    if st.button("🔄 Reset to Defaults"):
        for feature, (low, high) in feature_bounds.items():
            st.session_state[f"slider_{feature}"] = (low + high) / 2
        st.experimental_rerun()

# ========== MAIN CONTENT ==========
st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output using ambient conditions with an ensemble of Random Forest & XGBoost models.")

# Set best static weights
weights = {'rf': 0.65, 'xgb': 0.35}

# Prepare input for prediction
feature_names = list(feature_bounds.keys())
input_features = np.array([st.session_state[f"slider_{f}"] for f in feature_names]).reshape(1, -1)

with st.spinner("Making predictions..."):
    try:
        scaled_features = scaler.transform(input_features)
        rf_pred = rf_model.predict(scaled_features)[0]
        xgb_pred = xgb_model.predict(scaled_features)[0]
        ensemble_pred = weights['rf'] * rf_pred + weights['xgb'] * xgb_pred
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

st.subheader("🔢 Model Predictions")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Random Forest", f"{rf_pred:.2f} MW")
with col2:
    st.metric("XGBoost", f"{xgb_pred:.2f} MW")
with col3:
    st.metric(f"Ensemble", f"{ensemble_pred:.2f} MW")

# ========== BATCH PREDICTION ==========
st.subheader("📂 Batch Prediction")
st.markdown("Upload a CSV file with multiple records to get predictions for all of them at once.")

st.download_button(
    "⬇️ Download Example CSV",
    data=generate_example_csv(),
    file_name="ccpp_example_input.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader("Upload your input data (CSV format)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("Uploaded file is empty")
            st.stop()

        st.success("File uploaded successfully!")
        with st.expander("View uploaded data"):
            st.dataframe(df.head())

        mapped_columns = map_columns(df)
        if len(mapped_columns) < 4:
            missing_cols = [col for col in feature_names if col not in mapped_columns]
            st.error(f"Missing or unmatched columns: {', '.join(missing_cols)}")
            st.stop()

        df_processed = df.rename(columns=mapped_columns)
        missing_cols = [col for col in feature_names if col not in df_processed.columns]
        if missing_cols:
            st.error(f"Missing columns after mapping: {', '.join(missing_cols)}")
            st.stop()

        with st.spinner("Processing data..."):
            features = df_processed[feature_names]
            scaled = scaler.transform(features)
            rf_preds = rf_model.predict(scaled)
            xgb_preds = xgb_model.predict(scaled)
            final_preds = weights['rf'] * rf_preds + weights['xgb'] * xgb_preds

            results = df_processed.copy()
            results['RF_Prediction (MW)'] = rf_preds
            results['XGB_Prediction (MW)'] = xgb_preds
            results['Ensemble_Prediction (MW)'] = final_preds

            st.success("Predictions completed!")

            def color_positive_green(val):
                return 'color: green' if val > final_preds.mean() else 'color: red'

            st.dataframe(results.style.format({
                'RF_Prediction (MW)': '{:.2f}',
                'XGB_Prediction (MW)': '{:.2f}',
                'Ensemble_Prediction (MW)': '{:.2f}'
            }).applymap(color_positive_green, subset=['Ensemble_Prediction (MW)']))

            st.download_button(
                "⬇️ Download Full Results",
                data=results.to_csv(index=False).encode(),
                file_name="ccpp_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.caption("Developed with Streamlit | Ensemble Model: Random Forest (65%) + XGBoost (35%)")
