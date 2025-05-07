import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import StringIO

# Theme configuration
def set_theme(dark):
    plt.style.use('dark_background' if dark else 'default')
    if dark:
        st.markdown(
            """
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
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #ffffff;
                color: #000000;
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

    # Input sliders and buttons
    st.subheader("Input Parameters")
    inputs = {}
    for feature, (low, high) in feature_bounds.items():
        default = (low + high) / 2

        if feature == 'Model Weight (RF vs XGB)':
            if 'model_weight' not in st.session_state:
                st.session_state.model_weight = default

            col_minus, col_val, col_plus = st.columns([1, 2, 1])

            with col_minus:
                if st.button("➖", key="decrease_weight"):
                    st.session_state.model_weight = max(low, st.session_state.model_weight - 0.1)

            with col_val:
                st.session_state.model_weight = st.number_input(
                    feature, min_value=low, max_value=high,
                    value=st.session_state.model_weight, step=0.1,
                    key="model_weight_input"
                )

            with col_plus:
                if st.button("➕", key="increase_weight"):
                    st.session_state.model_weight = min(high, st.session_state.model_weight + 0.1)

            inputs[feature] = st.session_state.model_weight
        else:
            inputs[feature] = st.slider(
                feature, low, high, default,
                help=f"Adjust {feature} between {low} and {high}"
            )

    # Reset button
    if st.button("🔄 Reset to Defaults"):
        for feature in feature_bounds:
            if feature == 'Model Weight (RF vs XGB)':
                st.session_state.model_weight = (feature_bounds[feature][0] + feature_bounds[feature][1]) / 2
                inputs[feature] = st.session_state.model_weight
            else:
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

# Display results
st.subheader("🔢 Model Predictions")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Random Forest", f"{rf_pred:.2f} MW", delta_color="off")
with col2:
    st.metric("XGBoost", f"{xgb_pred:.2f} MW", delta_color="off")
with col3:
    st.metric(
        f"Ensemble (Weight: {input_weight:.2f})", 
        f"{ensemble_pred:.2f} MW",
        delta=f"{(ensemble_pred - (rf_pred + xgb_pred)/2):.2f} vs avg"
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
                
                # Display results
                st.dataframe(results.style.format({
                    'RF_Prediction (MW)': '{:.2f}',
                    'XGB_Prediction (MW)': '{:.2f}',
                    'Ensemble_Prediction (MW)': '{:.2f}'
                }))
                
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
