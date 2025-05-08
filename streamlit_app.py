import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# === Theme and Background ===
def set_theme(dark):
    background_url = "https://images.unsplash.com/photo-1581090700227-1e8eaff7ef72"  # Replace with your own hosted image URL if needed
    text_color = "#f1f1f1" if dark else "#000000"
    bg_color = "#0e1117" if dark else "#ffffff"

    background_style = f"""
    <style>
    .stApp {{
        background-image: url('{background_url}');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        color: {text_color};
    }}
    .css-1d391kg, .css-1cpxqw2, .st-b7, .st-b8, .st-b9 {{
        color: {text_color} !important;
    }}
    .css-1v3fvcr {{
        background-color: {'#262730' if dark else '#f0f2f6'} !important;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)
    plt.style.use('dark_background' if dark else 'default')

# === Model Loading ===
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

# === Generate Example CSV ===
@st.cache_data
def generate_example_csv():
    example_data = {
        "Temperature (°C)": [25.0, 30.0, 27.5],
        "Humidity (%)": [60.0, 65.0, 62.5],
        "Pressure (mbar)": [1010.0, 1005.0, 1007.5],
        "Vacuum (cmHg)": [5.0, 6.0, 5.5]
    }
    return pd.DataFrame(example_data).to_csv(index=False)

# === Column Mapping ===
def map_columns(df):
    column_mapping = {
        "Ambient Temperature (°C)": ["Ambient Temperature", "Temperature", "Temp", "Amb Temp", "Ambient_Temperature", "AT"],
        "Ambient Relative Humidity (%)": ["Relative Humidity", "Ambient Relative Humidity", "Humidity", "Rel Humidity", "Humidity (%)", "RH"],
        "Ambient Pressure (mbar)": ["Ambient Pressure", "Pressure", "Amb Pressure", "Pressure (mbar)", "AP"],
        "Exhaust Vacuum (cmHg)": ["Exhaust Vacuum", "Vacuum", "Exhaust Vac", "Vacuum (cmHg)", "EV"]
    }
    mapped_columns = {}
    for target, possible_names in column_mapping.items():
        for name in possible_names:
            if name in df.columns:
                mapped_columns[target] = name
                break
    return mapped_columns

# === Session Initialization ===
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# === SIDEBAR ===
with st.sidebar:
    st.title("⚙️ CCPP Power Predictor")
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    set_theme(st.session_state.dark_mode)

    st.subheader("How to Use")
    st.markdown("""
    1. Adjust sliders to set plant conditions  
    2. View the predicted power output  
    3. Compare models using the toggle  
    4. Upload CSV for batch predictions
    """)

    with st.spinner("Loading models..."):
        rf_model, xgb_model, scaler = load_models()

    feature_bounds = {
        'Ambient Temperature': [0.0, 50.0],
        'Ambient Relative Humidity': [10.0, 100.0],
        'Ambient Pressure': [799.0, 1035.0],
        'Exhaust Vacuum': [3.0, 12.0],
        'Model Weight (RF vs XGB)': [0.0, 1.0]
    }

    st.subheader("Input Parameters")
    inputs = {}
    for feature, (low, high) in feature_bounds.items():
        default = (low + high) / 2
        inputs[feature] = st.slider(
            feature, low, high, default,
            help=f"Adjust {feature} between {low} and {high}"
        )

    if st.button("🔄 Reset to Defaults"):
        for feature in inputs:
            inputs[feature] = (feature_bounds[feature][0] + feature_bounds[feature][1]) / 2

# === MAIN CONTENT ===
st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output using ambient conditions with an ensemble of Random Forest & XGBoost models.")

# === Prediction Logic ===
feature_names = list(feature_bounds.keys())[:-1]
input_features = np.array([inputs[f] for f in feature_names]).reshape(1, -1)
input_weight = inputs['Model Weight (RF vs XGB)']

with st.spinner("Making predictions..."):
    try:
        scaled_features = scaler.transform(input_features)
        rf_pred = rf_model.predict(scaled_features)[0]
        xgb_pred = xgb_model.predict(scaled_features)[0]
        ensemble_pred = input_weight * rf_pred + (1 - input_weight) * xgb_pred
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

# === Display Results ===
st.subheader("🔢 Model Predictions")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Random Forest", f"{rf_pred:.2f} MW")
with col2:
    st.metric("XGBoost", f"{xgb_pred:.2f} MW")
with col3:
    st.metric(
        f"Ensemble (Weight: {input_weight:.2f})",
        f"{ensemble_pred:.2f} MW",
        delta=f"{(ensemble_pred - (rf_pred + xgb_pred)/2):.2f} vs avg"
    )

# === Batch Prediction ===
st.subheader("📂 Batch Prediction")
st.markdown("Upload a CSV file with multiple records to get predictions for all of them at once.")

st.download_button(
    "⬇️ Download Example CSV",
    data=generate_example_csv(),
    file_name="ccpp_example_input.csv",
    mime="text/csv",
    help="Example file with the expected format"
)

uploaded_file = st.file_uploader(
    "Upload your input data (CSV format)",
    type=["csv"],
    help="CSV should contain columns for temperature, humidity, pressure, and vacuum"
)

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
            st.error(f"Could not find columns for: {', '.join(missing_cols)}")
            st.stop()

        df_processed = df.rename(columns=mapped_columns)
        required_cols = feature_names
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        if missing_cols:
            st.error(f"Missing columns after mapping: {', '.join(missing_cols)}")
            st.stop()

        with st.spinner("Processing data..."):
            features = df_processed[required_cols]
            scaled = scaler.transform(features)
            rf_preds = rf_model.predict(scaled)
            xgb_preds = xgb_model.predict(scaled)
            final_preds = input_weight * rf_preds + (1 - input_weight) * xgb_preds

            results = df_processed.copy()
            results['RF_Prediction (MW)'] = rf_preds
            results['XGB_Prediction (MW)'] = xgb_preds
            results['Ensemble_Prediction (MW)'] = final_preds

            st.success("Predictions completed!")
            st.dataframe(results.style.format({
                'RF_Prediction (MW)': '{:.2f}',
                'XGB_Prediction (MW)': '{:.2f}',
                'Ensemble_Prediction (MW)': '{:.2f}'
            }))

            st.download_button(
                "⬇️ Download Full Results",
                data=results.to_csv(index=False).encode(),
                file_name="ccpp_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# === Footer ===
st.markdown("---")
st.caption(f"""
Developed with Streamlit | Optimized with Particle Swarm Optimization (PSO)  
Model weights: Random Forest ({input_weight*100:.0f}%), XGBoost ({(1-input_weight)*100:.0f}%)
""")
