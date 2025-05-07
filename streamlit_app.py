import streamlit as st
import numpy as np 
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ========== Theme Configuration ==========
def set_theme(dark):
    plt.style.use('dark_background' if dark else 'default')
    if dark:
        st.markdown(""" 
        <style>
        .stApp { background-color: #0e1117; color: #f1f1f1; }
        .css-1d391kg, .css-1cpxqw2 { color: #f1f1f1 !important; }
        .css-1v3fvcr { background-color: #262730 !important; }
        .st-b7, .st-b8, .st-b9 { color: #f1f1f1 !important; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background-color: #ffffff; color: #000000; }
        </style>
        """, unsafe_allow_html=True)

# ========== Caching ==========
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

# ========== Session State ==========
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ========== Sidebar ==========
with st.sidebar:
    st.title("⚙️ CCPP Power Predictor")
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    set_theme(st.session_state.dark_mode)

    st.subheader("How to Use")
    st.markdown("""
    1. Adjust sliders for plant inputs  
    2. View predicted power output  
    3. Upload CSV for batch predictions
    """)

    with st.spinner("Loading models..."):
        rf_model, xgb_model, scaler = load_models()

    st.subheader("Input Parameters")
    ambient_temp = st.slider("Ambient Temperature", 0.0, 50.0, 25.0)
    humidity = st.slider("Ambient Relative Humidity", 10.0, 100.0, 55.0)
    pressure = st.slider("Ambient Pressure", 799.0, 1035.0, 917.0)
    vacuum = st.slider("Exhaust Vacuum", 3.0, 12.0, 7.5)
    weight = st.slider("Model Weight (RF vs XGB)", 0.0, 1.0, 0.5)

# ========== Main Area ==========
st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output using ambient conditions with an ensemble of Random Forest & XGBoost models.")

# ========== Single Prediction ==========
try:
    input_features = np.array([[ambient_temp, humidity, pressure, vacuum]])
    scaled_features = scaler.transform(input_features)

    rf_pred = rf_model.predict(scaled_features)[0]
    xgb_pred = xgb_model.predict(scaled_features)[0]
    ensemble_pred = weight * rf_pred + (1 - weight) * xgb_pred
except Exception as e:
    st.error(f"Prediction error: {str(e)}")
    st.stop()

# ========== Display Predictions ==========
st.subheader("🔢 Model Predictions")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Random Forest", f"{rf_pred:.2f} MW")
with col2:
    st.metric("XGBoost", f"{xgb_pred:.2f} MW")
with col3:
    st.metric("Ensemble", f"{ensemble_pred:.2f} MW", delta=f"{(ensemble_pred - (rf_pred + xgb_pred)/2):.2f} vs avg")

# ========== Batch Prediction ==========
st.subheader("📂 Batch Prediction")

st.download_button(
    "⬇️ Download Example CSV",
    data=generate_example_csv(),
    file_name="ccpp_example_input.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("Uploaded file is empty")
            st.stop()

        st.success("File uploaded successfully!")
        st.dataframe(df.head())

        mapped = map_columns(df)
        if len(mapped) < 4:
            missing = [col for col in ["Ambient Temperature (°C)", "Ambient Relative Humidity (%)", "Ambient Pressure (mbar)", "Exhaust Vacuum (cmHg)"] if col not in mapped]
            st.error(f"Missing columns: {', '.join(missing)}")
            st.stop()

        df_processed = df.rename(columns=mapped)
        feature_order = list(mapped.keys())
        features = df_processed[feature_order]

        scaled = scaler.transform(features)
        rf_preds = rf_model.predict(scaled)
        xgb_preds = xgb_model.predict(scaled)
        ensemble_preds = weight * rf_preds + (1 - weight) * xgb_preds

        results = df_processed.copy()
        results['RF_Prediction (MW)'] = rf_preds
        results['XGB_Prediction (MW)'] = xgb_preds
        results['Ensemble_Prediction (MW)'] = ensemble_preds

        st.dataframe(results.style.format({
            'RF_Prediction (MW)': '{:.2f}',
            'XGB_Prediction (MW)': '{:.2f}',
            'Ensemble_Prediction (MW)': '{:.2f}'
        }))

        st.download_button(
            "⬇️ Download Results",
            data=results.to_csv(index=False).encode(),
            file_name="ccpp_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")

# ========== Footer ==========
st.markdown("---")
st.caption(f"Developed with Streamlit | Random Forest: {weight*100:.0f}% | XGBoost: {(1-weight)*100:.0f}%")
