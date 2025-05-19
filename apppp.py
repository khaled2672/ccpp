import streamlit as st 
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import StringIO

def set_theme(dark):
    background_url = "https://img.freepik.com/free-photo/view-nuclear-power-plant-with-towers-letting-out-steam-from-process_23-2150957658.jpg"
    if dark:
        st.markdown(f"""
            <style>
            .stApp {{
                background: url('{background_url}') no-repeat center center fixed;
                background-size: cover;
                color: #FFFFFF;
            }}
            .main .block-container {{
                background-color: rgba(0, 0, 0, 0.75);
                padding: 2rem;
                border-radius: 12px;
            }}
            [data-testid="stSidebar"] > div:first-child {{
                background-color: #111111 !important;
                color: #FFFFFF;
            }}
            .css-1d391kg, .css-1cpxqw2, .st-b7, .st-b8, .st-b9 {{
                color: #FFFFFF !important;
            }}
            .stDownloadButton, .stButton>button {{
                background-color: #FF0000 !important;
                color: #FFFFFF !important;
                border: 1px solid #FFFFFF !important;
            }}
            .stDownloadButton:hover, .stButton>button:hover {{
                background-color: #cc0000 !important;
            }}
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <style>
            .stApp {{
                background: url('{background_url}') no-repeat center center fixed;
                background-size: cover;
                color: #000000;
            }}
            .main .block-container {{
                background-color: rgba(255, 255, 255, 0.85);
                padding: 2rem;
                border-radius: 12px;
            }}
            [data-testid="stSidebar"] > div:first-child {{
                background-color: #FFFFFF !important;
                color: #000000;
            }}
            .css-1d391kg, .css-1cpxqw2, .st-b7, .st-b8, .st-b9 {{
                color: #000000 !important;
            }}
            .stDownloadButton, .stButton>button {{
                background-color: #FF0000 !important;
                color: #FFFFFF !important;
                border: none !important;
            }}
            .stDownloadButton:hover, .stButton>button:hover {{
                background-color: #cc0000 !important;
            }}
            </style>
        """, unsafe_allow_html=True)

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

def map_columns(df):
    column_mapping = {
        "Ambient Temperature": ["Ambient Temperature", "Temperature", "Temp", "Amb Temp", "Ambient_Temperature", "AT", "Temperature (°C)"],
        "Ambient Relative Humidity": ["Relative Humidity", "Ambient Relative Humidity", "Humidity", "Rel Humidity", "Humidity (%)", "RH"],
        "Ambient Pressure": ["Ambient Pressure", "Pressure", "Amb Pressure", "Pressure (mbar)", "AP"],
        "Exhaust Vacuum": ["Exhaust Vacuum", "Vacuum", "Exhaust Vac", "Vacuum (cmHg)", "EV"]
    }
    mapped_columns = {}
    for target, possible_names in column_mapping.items():
        for name in possible_names:
            if name in df.columns:
                mapped_columns[target] = name
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

with st.sidebar:
    st.title("⚙️ CCPP Power Predictor")
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    set_theme(st.session_state.dark_mode)
    st.subheader("How to Use")
    st.markdown("""
    1. Adjust sliders to set plant conditions  
    2. View the predicted power output  
    3. Compare models  
    4. Upload CSV for batch predictions  
    """)
    with st.spinner("Loading models..."):
        rf_model, xgb_model, scaler = load_models()

    feature_bounds = {
        'Ambient Temperature': [0.0, 50.0],
        'Ambient Relative Humidity': [10.0, 100.0],
        'Ambient Pressure': [799.0, 1035.0],
        'Exhaust Vacuum': [3.0, 12.0]
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

st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output using ambient conditions with an ensemble of Random Forest & XGBoost models.")

feature_names = list(feature_bounds.keys())
input_features = np.array([inputs[f] for f in feature_names]).reshape(1, -1)
input_weight = 0.65

with st.spinner("Making predictions..."):
    try:
        scaled_features = scaler.transform(input_features)
        rf_pred = rf_model.predict(scaled_features)[0]
        xgb_pred = xgb_model.predict(scaled_features)[0]
        ensemble_pred = input_weight * rf_pred + (1 - input_weight) * xgb_pred
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

text_color = "#FFFFFF" if st.session_state.dark_mode else "#000000"

st.subheader("🔢 Model Predictions")
col1, col2, col3 = st.columns(3)
for col, title, value in zip(
    [col1, col2, col3],
    ["Random Forest", "XGBoost", "Ensemble (65% RF / 35% XGB)"],
    [rf_pred, xgb_pred, ensemble_pred]
):
    col.markdown(
        f""" <div style="
             background-color: rgba(30, 30, 30, 0.7) if st.session_state.dark_mode else rgba(240, 240, 240, 0.8);
             padding: 1.5rem;
             border-radius: 10px;
             box-shadow: 0 4px 8px rgba(0,0,0,0.1);
             text-align: center;
         "> <h3 style="margin-top: 0; color: {text_color};">{title}</h3> <h2 style="color: {text_color};">{value:.2f} MW</h2> </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.subheader("📁 Batch Prediction via CSV Upload")

uploaded_file = st.file_uploader("Upload CSV file with plant conditions", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df.head())

        mapped_cols = map_columns(df)

        if len(mapped_cols) < len(feature_bounds):
            st.warning("Some required columns are missing or could not be mapped automatically. Please check the CSV headers.")
        else:
            pred_features = df[[mapped_cols[feat] for feat in feature_names]]
            pred_features.columns = feature_names

            scaled_batch_features = scaler.transform(pred_features)

            rf_preds = rf_model.predict(scaled_batch_features)
            xgb_preds = xgb_model.predict(scaled_batch_features)
            final_preds = input_weight * rf_preds + (1 - input_weight) * xgb_preds

            df["Random Forest Prediction (MW)"] = rf_preds
            df["XGBoost Prediction (MW)"] = xgb_preds
            df["Ensemble Prediction (MW)"] = final_preds

            st.success(f"Predicted {len(final_preds)} records successfully!")
            st.dataframe(df)

            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv_buffer.getvalue(),
                file_name="ccpp_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
else:
    st.info("Upload a CSV file to run batch predictions. You can download an example file below.")
    st.download_button(
        label="Download Example CSV",
        data=generate_example_csv(),
        file_name="example_ccpp_data.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("""
Developed with Streamlit | Optimized with Particle Swarm Optimization (PSO)  
Model weights: Random Forest (65%), XGBoost (35%)
""")
