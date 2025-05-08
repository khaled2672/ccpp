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
            """
            <style>
            .stApp {
                background-image: url("https://media.istockphoto.com/id/2163868131/photo/gas-turbine-electrical-power-plant-at-industrial-estate-business-gas-turbine-electricity.jpg?s=1024x1024&w=is&k=20&c=3QUBbQ5N1SRHdraIQdaSqi-VY4MIoPYagmnZxlSSoUo=");
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
                color: white !important;
                border: none !important;
            }
            .stDownloadButton:hover, .stButton>button:hover {
                background-color: #3a7ae4 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-image: url("https://www.enka.com/wp-content/uploads/freshizer/015ac6b46b92087855b6591cfe219b8c_IZMIR_010-800-400-c.jpg");
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
            }
            </style>
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
feature_names = list(feature_bounds.keys())[:-1]  # Exclude weight
input_features = np.array([inputs[f] for f in feature_names]).reshape(1, -1)
input_weight = inputs['Model Weight (RF vs XGB)']

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
        f"""
        <div style="
            background-color: {'rgba(30, 30, 30, 0.7)' if st.session_state.dark_mode else 'rgba(240, 240, 240, 0.8)'};
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
        ">
            <h3 style="margin-top: 0;">Random Forest</h3>
            <h2 style="color: {'#4a8af4' if st.session_state.dark_mode else '#2a6fdb'};">{rf_pred:.2f} MW</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""
        <div style="
            background-color: {'rgba(30, 30, 30, 0.7)' if st.session_state.dark_mode else 'rgba(240, 240, 240, 0.8)'};
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
        ">
            <h3 style="margin-top: 0;">XGBoost</h3>
            <h2 style="color: {'#4a8af4' if st.session_state.dark_mode else '#2a6fdb'};">{xgb_pred:.2f} MW</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f"""
        <div style="
            background-color: {'rgba(30, 30, 30, 0.7)' if st.session_state.dark_mode else 'rgba(240, 240, 240, 0.8)'};
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
        ">
            <h3 style="margin-top: 0;">Ensemble (Weight: {input_weight:.2f})</h3>
            <h2 style="color: {'#4a8af4' if st.session_state.dark_mode else '#2a6fdb'};">{ensemble_pred:.2f} MW</h2>
            <p style="margin-bottom: 0; font-size: 0.9rem;">{(ensemble_pred - (rf_pred + xgb_pred)/2):.2f} vs avg</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Batch Prediction with CSV Upload
st.subheader("📂 Batch Prediction")
st.markdown("Upload a CSV file with multiple records to get predictions for all of them at once.")

# Example CSV download
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
        
        # Column mapping
        mapped_columns = map_columns(df)
        if len(mapped_columns) < 4:
            missing_cols = [col for col in feature_names if col not in mapped_columns]
            st.error(f"Could not find columns for: {', '.join(missing_cols)}")
            st.stop()
            
        df_processed = df.rename(columns=mapped_columns)
        required_cols = feature_names  # From feature_bounds
        
        # Check for missing columns after mapping
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        if missing_cols:
            st.error(f"Missing columns after mapping: {', '.join(missing_cols)}")
            st.stop()
            
        # Process data
        with st.spinner("Processing data..."):
            features = df_processed[required_cols]
            try:
                scaled = scaler.transform(features)
                rf_preds = rf_model.predict(scaled)
                xgb_preds = xgb_model.predict(scaled)
                final_preds = input_weight * rf_preds + (1 - input_weight) * xgb_preds
                
                results = df_processed.copy()
                results['RF_Prediction (MW)'] = rf_preds
                results['XGB_Prediction (MW)'] = xgb_preds
                results['Ensemble_Prediction (MW)'] = final_preds
                
                st.success("Predictions completed!")
                
                # Display results with conditional formatting
                def color_positive_green(val):
                    color = 'green' if val > (rf_preds.mean() + xgb_preds.mean())/2 else 'red'
                    return f'color: {color}'
                
                st.dataframe(results.style.format({
                    'RF_Prediction (MW)': '{:.2f}',
                    'XGB_Prediction (MW)': '{:.2f}',
                    'Ensemble_Prediction (MW)': '{:.2f}'
                }).applymap(color_positive_green, subset=['Ensemble_Prediction (MW)']))
                
                # Download results
                csv = results.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Full Results",
                    data=csv,
                    file_name="ccpp_predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.caption("""
Developed with Streamlit | Optimized with Particle Swarm Optimization (PSO)  
Model weights: Random Forest ({:.0f}%), XGBoost ({:.0f}%)
""".format(input_weight*100, (1-input_weight)*100))
