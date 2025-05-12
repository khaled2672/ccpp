import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import StringIO

# Theme configuration with background images
def set_theme(dark):
    plt.style.use('dark_background' if dark else 'default')
    if dark:
        st.markdown(
            """ <style>
            .stApp {
            background-image: url("https://img.freepik.com/free-photo/view-nuclear-power-plant-with-towers-letting-out-steam-from-process_23-2150957658.jpg?t=st=1746689462~exp=1746693062~hmac=71da5c1edb4e4c2bd79eda912f889934c4d11e1aeea35a5106d1bd18e53a89b4&w=1380");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #f1f1f1;
            }
            /* Dark overlay for better readability */
            .stApp\:before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 0, 0, 0.75);
            z-index: -1;
            }
            /* Main content area */
            .main .block-container {
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2rem;
            border-radius: 10px;
            backdrop-filter: blur(4px);
            }
            /* Sidebar */
            [data-testid="stSidebar"] > div\:first-child {
            background-color: rgba(0, 0, 0, 0.8) !important;
            color: #ffffff ;
            backdrop-filter: blur(4px);
            }
            /* Text colors */
            .css-1d391kg, .css-1cpxqw2, .st-b7, .st-b8, .st-b9 {
            color: #f1f1f1 !important;
            }
            /* Widget styling */
            .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj {
            background-color: rgba(30, 30, 30, 0.7) !important;
            }
            /* Button styling */
            .stDownloadButton, .stButton>button {
            background-color: #4a8af4 !important;
            color: black !important;
            border: white !important;
            }
            .stDownloadButton\:hover, .stButton>button\:hover {
            background-color: #f5f6f7 !important;
            } </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """ <style>
            .stApp {
            background-image: url("https://img.freepik.com/free-photo/view-nuclear-power-plant-with-towers-letting-out-steam-from-process_23-2150957658.jpg?t=st=1746689462~exp=1746693062~hmac=71da5c1edb4e4c2bd79eda912f889934c4d11e1aeea35a5106d1bd18e53a89b4&w=1380");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #333333;
            }
            /* Light overlay for better readability */
            .stApp\:before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.75);
            z-index: -1;
            }
            /* Main content area */
            .main .block-container {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 2rem;
            border-radius: 10px;
            backdrop-filter: blur(4px);
            }
            /* Sidebar */
            [data-testid="stSidebar"] > div\:first-child {
            background-color: rgba(255, 255, 255, 0.85) !important;
            backdrop-filter: blur(4px);
            }
            /* Text colors */
            .css-1d391kg, .css-1cpxqw2, .st-b7, .st-b8, .st-b9 {
            color: #ffffff !important;
            }
            /* Widget styling */
            .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj {
            background-color: rgba(240, 240, 240, 0.8) !important;
            }
            /* Button styling */
            .stDownloadButton, .stButton>button {
            background-color: #4a8af4 !important;
            color: white !important;
            border: none !important;
            }
            .stDownloadButton\:hover, .stButton>button\:hover {
            background-color: #3a7ae4 !important;
            } </style>
            """,
            unsafe_allow_html=True
        )

# Cache resources for better performance
@st.cache_resource
def load_models():
    """Load models and scaler with caching"""
    try:
        return (
            joblib.load('rf_model.joblib'),
            joblib.load('xgb_model.joblib'),
            joblib.load('scaler.joblib')
        )
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# Column mapping function
def map_columns(df):
    """Map user-uploaded CSV columns to the required features."""
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

# Generate example CSV data
@st.cache_data
def generate_example_csv():
    """Generate example CSV data for download"""
    example_data = {
        "Temperature (°C)": [25.0, 30.0, 27.5],
        "Humidity (%)": [60.0, 65.0, 62.5],
        "Pressure (mbar)": [1010.0, 1005.0, 1007.5],
        "Vacuum (cmHg)": [5.0, 6.0, 5.5]
    }
    return pd.DataFrame(example_data).to_csv(index=False)

# Initialize session state for theme persistence
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("⚙️ CCPP Power Predictor")

    # Dark mode toggle
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    set_theme(st.session_state.dark_mode)

    st.subheader("How to Use")
    st.markdown("""
    1. Adjust sliders to set plant conditions  
    2. View the predicted power output  
    3. Compare models using the toggle  
    4. Upload CSV for batch predictions
    """)

    # Load models
    with st.spinner("Loading models..."):
        rf_model, xgb_model, scaler = load_models()

    # Feature bounds for UI
    feature_bounds = {
        'Ambient Temperature': [0.0, 50.0],
        'Ambient Relative Humidity': [10.0, 100.0],
        'Ambient Pressure': [799.0, 1035.0],
        'Exhaust Vacuum': [3.0, 12.0],
        'Model Weight (RF vs XGB)': [0.0, 1.0]
    }

    # Input sliders
    st.subheader("Input Parameters")
    inputs = {}
    for feature, (low, high) in feature_bounds.items():
        default = (low + high) / 2
        inputs[feature] = st.slider(
            feature, low, high, default,
            help=f"Adjust {feature} between {low} and {high}"
        )

    # Reset button
    if st.button("🔄 Reset to Defaults"):
        for feature in inputs:
            inputs[feature] = (feature_bounds[feature][0] + feature_bounds[feature][1]) / 2

# ========== MAIN CONTENT ==========
st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output using ambient conditions with an ensemble of Random Forest & XGBoost models.")

# Prepare input for prediction
feature_names = list(inputs.keys())[:4]  # Make sure only the first 4 features are passed
input_data = np.array([inputs[feature] for feature in feature_names]).reshape(1, -1)

# Predict using both models
st.subheader("Predictions")
rf_pred = rf_model.predict(input_data)
xgb_pred = xgb_model.predict(input_data)

# Show predictions
st.write(f"Power Output (RF Model): {rf_pred[0]:.2f} MW")
st.write(f"Power Output (XGB Model): {xgb_pred[0]:.2f} MW")

# ========== CSV Upload & Batch Prediction ==========
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        mapped_columns = map_columns(df)
        df = df.rename(columns=mapped_columns)

        # Ensure we only use the 4 expected features
        features = df[list(mapped_columns.keys())].values
        features_scaled = scaler.transform(features)

        rf_preds = rf_model.predict(features_scaled)
        xgb_preds = xgb_model.predict(features_scaled)

        st.write("Batch Prediction Results:")
        df['RF_Prediction'] = rf_preds
        df['XGB_Prediction'] = xgb_preds
        st.write(df)

    except Exception as e:
        st.error(f"Error processing the CSV file: {str(e)}")

# ========== Example CSV Download ==========
st.download_button(
    label="Download Example CSV",
    data=generate_example_csv(),
    file_name="example_data.csv",
    mime="text/csv"
)
