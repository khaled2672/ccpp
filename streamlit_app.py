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
                background-image: url("https://img.freepik.com/free-photo/view-nuclear-power-plant-with-towers-letting-out-steam-from-process_23-2150957658.jpg");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
                color: #f1f1f1;
            }
            /* Dark overlay for better readability */
            .stApp:before {
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
            [data-testid="stSidebar"] > div:first-child {
                background-color: rgba(0, 0, 0, 0.8) !important;
                color: #ffffff;
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
            .stDownloadButton:hover, .stButton>button:hover {
                background-color: #f5f6f7 !important;
            } </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """ <style>
            .stApp {
                background-image: url("https://img.freepik.com/free-photo/view-nuclear-power-plant-with-towers-letting-out-steam-from-process_23-2150957658.jpg");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
                color: #333333;
            }
            /* Light overlay for better readability */
            .stApp:before {
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
            [data-testid="stSidebar"] > div:first-child {
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
            .stDownloadButton:hover, .stButton>button:hover {
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
        "Ambient Temperature": ["Ambient Temperature", "Temperature", "Temp", "Amb Temp", "Ambient_Temperature", "AT", "Temperature (°C)"],
        "Ambient Relative Humidity": ["Relative Humidity", "Ambient Relative Humidity", "Humidity", "Rel Humidity", "Humidity (%)", "RH", "Humidity (%)"],
        "Ambient Pressure": ["Ambient Pressure", "Pressure", "Amb Pressure", "Pressure (mbar)", "AP", "Pressure (mbar)"],
        "Exhaust Vacuum": ["Exhaust Vacuum", "Vacuum", "Exhaust Vac", "Vacuum (cmHg)", "EV", "Vacuum (cmHg)"]
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
    3. Compare models  
    4. Upload CSV for batch predictions  
    5. Ensemble weights fixed: 65% RF / 35% XGB
    """)

    # Load models
    with st.spinner("Loading models..."):
        rf_model, xgb_model, scaler = load_models()

    # Feature bounds for UI
    feature_bounds = {
        'Ambient Temperature': [0.0, 50.0],
        'Ambient Relative Humidity': [10.0, 100.0],
        'Ambient Pressure': [799.0, 1035.0],
        'Exhaust Vacuum': [3.0, 12.0]
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
feature_names = list(feature_bounds.keys())
input_features = np.array([inputs[f] for f in feature_names]).reshape(1, -1)

# Fixed model weight
input_weight = 0.65

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

# Display results in cards
st.subheader("🔢 Model Predictions")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f""" <div style="
             background-color: {'rgba(30, 30, 30, 0.7)' if st.session_state.dark_mode else 'rgba(240, 240, 240, 0.8)'};
             padding: 1.5rem;
             border-radius: 10px;
             box-shadow: 0 4px 8px rgba(0,0,0,0.1);
             text-align: center;
         "> <h3 style="margin-top: 0;">Random Forest</h3> <h2 style="color: {'#4a8af4' if st.session_state.dark_mode else '#2a6fdb'};">{rf_pred:.2f} MW</h2> </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f""" <div style="
             background-color: {'rgba(30, 30, 30, 0.7)' if st.session_state.dark_mode else 'rgba(240, 240, 240, 0.8)'};
             padding: 1.5rem;
             border-radius: 10px;
             box-shadow: 0 4px 8px rgba(0,0,0,0.1);
             text-align: center;
         "> <h3 style="margin-top: 0;">XGBoost</h3> <h2 style="color: {'#4a8af4' if st.session_state.dark_mode else '#2a6fdb'};">{xgb_pred:.2f} MW</h2> </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f""" <div style="
             background-color: {'rgba(30, 30, 30, 0.7)' if st.session_state.dark_mode else 'rgba(240, 240, 240, 0.8)'};
             padding: 1.5rem;
             border-radius: 10px;
             box-shadow: 0 4px 8px rgba(0,0,0,0.1);
             text-align: center;
         "> <h3 style="margin-top: 0;">Ensemble (Weight: 65% RF / 35% XGB)</h3> <h2 style="color: {'#4a8af4' if st.session_state.dark_mode else '#2a6fdb'};">{ensemble_pred:.2f} MW</h2> <p style="margin-bottom: 0; font-size: 0.9rem;">{(ensemble_pred - (rf_pred + xgb_pred)/2):.2f} vs avg</p> </div>
        """,
        unsafe_allow_html=True
    )

# Batch prediction upload
st.markdown("---")
st.subheader("📁 Batch Prediction via CSV Upload")

uploaded_file = st.file_uploader("Upload CSV file with plant conditions", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df.head())

        # Map columns
        mapped_cols = map_columns(df)

        if len(mapped_cols) < len(feature_bounds):
            st.warning("Some required columns are missing or could not be mapped automatically. Please check the CSV headers.")
        else:
            # Extract features for prediction
            pred_features = df[[mapped_cols[feat] for feat in feature_names]]
            pred_features.columns = feature_names  # rename columns to model feature names

            # Scale features
            scaled_batch_features = scaler.transform(pred_features)

            # Predict
            rf_preds = rf_model.predict(scaled_batch_features)
            xgb_preds = xgb_model.predict(scaled_batch_features)
            final_preds = input_weight * rf_preds + (1 - input_weight) * xgb_preds

            # Add predictions to dataframe
            df["Random Forest Prediction (MW)"] = rf_preds
            df["XGBoost Prediction (MW)"] = xgb_preds
            df["Ensemble Prediction (MW)"] = final_preds

            st.success(f"Predicted {len(final_preds)} records successfully!")

            # Show predictions
            st.dataframe(df)

            # Download results
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
    # Provide example CSV for user
    st.info("Upload a CSV file to run batch predictions. You can download an example file below.")
    st.download_button(
        label="Download Example CSV",
        data=generate_example_csv(),
        file_name="example_ccpp_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("""
Developed with Streamlit | Optimized with Particle Swarm Optimization (PSO)  
Model weights: Random Forest (65%), XGBoost (35%)
""")
