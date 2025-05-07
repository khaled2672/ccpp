import streamlit as st
import numpy as np 
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ========== THEME ==========
def set_theme(dark):
    plt.style.use('dark_background' if dark else 'default')
    if dark:
        st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #f1f1f1; }
        </style>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background-color: #ffffff; color: #000000; }
        </style>""", unsafe_allow_html=True)

# ========== CACHING ==========
@st.cache_resource
def load_models():
    return (
        joblib.load("rf_model.joblib"),
        joblib.load("xgb_model.joblib"),
        joblib.load("scaler.joblib")
    )

@st.cache_data
def generate_example_csv():
    data = {
        "Temperature (°C)": [25.0, 30.0, 27.5],
        "Humidity (%)": [60.0, 65.0, 62.5],
        "Pressure (mbar)": [1010.0, 1005.0, 1007.5],
        "Vacuum (cmHg)": [5.0, 6.0, 5.5]
    }
    return pd.DataFrame(data).to_csv(index=False)

# ========== COLUMN MAPPING ==========
def map_columns(df):
    column_mapping = {
        "Ambient Temperature": ["Temperature", "Ambient Temperature", "Temp", "AT"],
        "Ambient Relative Humidity": ["Humidity", "Relative Humidity", "RH"],
        "Ambient Pressure": ["Pressure", "Ambient Pressure", "AP"],
        "Exhaust Vacuum": ["Vacuum", "Exhaust Vacuum", "EV"]
    }
    mapped = {}
    for key, names in column_mapping.items():
        for name in names:
            if name in df.columns:
                mapped[key] = name
                break
    return mapped

# ========== SESSION INIT ==========
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'reset' not in st.session_state:
    st.session_state.reset = False

# ========== FEATURE SETUP ==========
FEATURE_BOUNDS = {
    'Ambient Temperature': [0.0, 50.0],
    'Ambient Relative Humidity': [10.0, 100.0],
    'Ambient Pressure': [799.0, 1035.0],
    'Exhaust Vacuum': [3.0, 12.0],
    'Model Weight (RF vs XGB)': [0.0, 1.0]
}
DEFAULTS = {k: (v[0]+v[1])/2 for k,v in FEATURE_BOUNDS.items()}

# Handle reset before slider creation
if st.session_state.reset:
    for feature in FEATURE_BOUNDS:
        st.session_state[f"slider_{feature}"] = DEFAULTS[feature]
    st.session_state.reset = False

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("⚙️ CCPP Power Predictor")
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    set_theme(st.session_state.dark_mode)
    st.markdown("1. Adjust sliders\n2. View predictions\n3. Upload CSV\n4. Compare models")

    rf_model, xgb_model, scaler = load_models()

    # Input sliders
    st.subheader("Input Parameters")
    inputs = {}
    for feature, (low, high) in FEATURE_BOUNDS.items():
        key = f"slider_{feature}"
        default_val = st.session_state.get(key, DEFAULTS[feature])
        inputs[feature] = st.slider(feature, low, high, default_val, key=key)

    # Reset button
    if st.button("🔄 Reset to Defaults"):
        st.session_state.reset = True
        st.rerun()

# ========== MAIN ==========
st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output using ambient conditions with Random Forest & XGBoost.")

# Prepare input
input_keys = list(FEATURE_BOUNDS.keys())[:-1]  # exclude weight
X_input = np.array([inputs[f] for f in input_keys]).reshape(1, -1)
model_weight = inputs['Model Weight (RF vs XGB)']

# Prediction
with st.spinner("Predicting..."):
    scaled = scaler.transform(X_input)
    rf_pred = rf_model.predict(scaled)[0]
    xgb_pred = xgb_model.predict(scaled)[0]
    ensemble = model_weight * rf_pred + (1 - model_weight) * xgb_pred

# Show output
st.subheader("🔢 Model Predictions")
col1, col2, col3 = st.columns(3)
col1.metric("Random Forest", f"{rf_pred:.2f} MW")
col2.metric("XGBoost", f"{xgb_pred:.2f} MW")
col3.metric(f"Ensemble ({model_weight:.2f})", f"{ensemble:.2f} MW", 
            delta=f"{(ensemble - ((rf_pred + xgb_pred)/2)):.2f} vs avg")

# ========== CSV Upload ==========
st.subheader("📂 Batch Prediction")
st.download_button("⬇️ Download Example CSV", generate_example_csv(), file_name="example.csv", mime="text/csv")

uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully.")
    st.dataframe(df.head())

    mapped = map_columns(df)
    if len(mapped) < 4:
        st.error("Missing columns in uploaded file.")
        st.stop()

    df_renamed = df.rename(columns=mapped)
    X_batch = df_renamed[input_keys]
    try:
        scaled_batch = scaler.transform(X_batch)
        rf_preds = rf_model.predict(scaled_batch)
        xgb_preds = xgb_model.predict(scaled_batch)
        ensemble_preds = model_weight * rf_preds + (1 - model_weight) * xgb_preds
        df_result = df.copy()
        df_result["RF_Prediction"] = rf_preds
        df_result["XGB_Prediction"] = xgb_preds
        df_result["Ensemble_Prediction"] = ensemble_preds
        st.success("Batch predictions complete.")
        st.dataframe(df_result.head())

        csv_result = df_result.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results", csv_result, file_name="ccpp_results.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# ========== FOOTER ==========
st.markdown("---")
st.caption(f"Model Weights — RF: {model_weight:.0%}, XGB: {(1-model_weight):.0%}")
