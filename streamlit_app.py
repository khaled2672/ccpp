import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ================= THEME =================
def set_theme(dark):
    plt.style.use('dark_background' if dark else 'default')
    if dark:
        st.markdown("""
            <style>
            .stApp {
                background-color: #0e1117;
                color: #f1f1f1;
            }
            .css-1d391kg, .css-1cpxqw2 {
                color: #f1f1f1 !important;
            }
            .css-1v3fvcr {
                background-color: #262730 !important;
            }
            .st-b7, .st-b8, .st-b9 {
                color: #f1f1f1 !important;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp {
                background-color: #ffffff;
                color: #000000;
            }
            </style>
        """, unsafe_allow_html=True)

# =============== CACHING ===============
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

@st.cache_data
def generate_example_csv():
    example_data = {
        "Temperature (°C)": [25.0, 30.0, 27.5],
        "Humidity (%)": [60.0, 65.0, 62.5],
        "Pressure (mbar)": [1010.0, 1005.0, 1007.5],
        "Vacuum (cmHg)": [5.0, 6.0, 5.5]
    }
    return pd.DataFrame(example_data).to_csv(index=False)

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

# ================= STATE INIT ================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# =============== SIDEBAR ==================
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

    # Feature bounds
    feature_bounds = {
        "Ambient Temperature (°C)": [0.0, 50.0],
        "Ambient Relative Humidity (%)": [10.0, 100.0],
        "Ambient Pressure (mbar)": [799.0, 1035.0],
        "Exhaust Vacuum (cmHg)": [3.0, 12.0],
        "Model Weight (RF vs XGB)": [0.0, 1.0]
    }

    # Initialize feature values if not already
    for feature, (low, high) in feature_bounds.items():
        if feature not in st.session_state:
            st.session_state[feature] = (low + high) / 2

    # Sliders
    st.subheader("Input Parameters")
    for feature, (low, high) in feature_bounds.items():
        st.session_state[feature] = st.slider(
            feature, low, high, st.session_state[feature],
            key=feature,
            help=f"Adjust {feature} between {low} and {high}"
        )

    # Reset button
    if st.button("🔄 Reset to Defaults"):
        for feature, (low, high) in feature_bounds.items():
            st.session_state[feature] = (low + high) / 2
        st.experimental_rerun()

# ============== MAIN CONTENT ===============
st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output using ambient conditions with an ensemble of Random Forest & XGBoost models.")

# Prepare input
feature_names = list(feature_bounds.keys())[:-1]
input_features = np.array([
    st.session_state[feature] for feature in feature_names
]).reshape(1, -1)
input_weight = st.session_state["Model Weight (RF vs XGB)"]

# Make predictions
with st.spinner("Making predictions..."):
    try:
        scaled_features = scaler.transform(input_features)
        rf_pred = rf_model.predict(scaled_features)[0]
        xgb_pred = xgb_model.predict(scaled_features)[0]
        ensemble_pred = input_weight * rf_pred + (1 - input_weight) * xgb_pred
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

# Show results
st.subheader("🔢 Model Predictions")
col1, col2, col3 = st.columns(3)
col1.metric("Random Forest", f"{rf_pred:.2f} MW")
col2.metric("XGBoost", f"{xgb_pred:.2f} MW")
col3.metric(
    f"Ensemble (Weight: {input_weight:.2f})",
    f"{ensemble_pred:.2f} MW",
    delta=f"{(ensemble_pred - (rf_pred + xgb_pred)/2):.2f} vs avg"
)

# ============== BATCH UPLOAD ===============
st.subheader("📂 Batch Prediction")
st.markdown("Upload a CSV file with multiple records to get predictions for all of them at once.")

st.download_button(
    "⬇️ Download Example CSV",
    data=generate_example_csv(),
    file_name="ccpp_example_input.csv",
    mime="text/csv",
    help="Example file with the expected format"
)

uploaded_file = st.file_uploader("Upload your input data (CSV format)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("Uploaded file is empty.")
            st.stop()

        st.success("File uploaded successfully!")
        with st.expander("View uploaded data"):
            st.dataframe(df.head())

        mapped_columns = map_columns(df)
        if len(mapped_columns) < 4:
            missing = [col for col in feature_names if col not in mapped_columns]
            st.error(f"Could not find columns for: {', '.join(missing)}")
            st.stop()

        df = df.rename(columns=mapped_columns)
        required_cols = feature_names
        if any(col not in df.columns for col in required_cols):
            st.error(f"Missing required columns: {', '.join(required_cols)}")
            st.stop()

        with st.spinner("Processing batch..."):
            scaled = scaler.transform(df[required_cols])
            rf_preds = rf_model.predict(scaled)
            xgb_preds = xgb_model.predict(scaled)
            final_preds = input_weight * rf_preds + (1 - input_weight) * xgb_preds

            results = df.copy()
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

# ============== FOOTER ==============
st.markdown("---")
st.caption("""
Developed with Streamlit | Optimized with Particle Swarm Optimization (PSO)  
Model weights: Random Forest ({:.0f}%), XGBoost ({:.0f}%)
""".format(input_weight * 100, (1 - input_weight) * 100))
