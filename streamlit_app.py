import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import os

# Constants
FEATURE_BOUNDS = {
    'Ambient Temperature (°C)': [0.0, 50.0],
    'Ambient Relative Humidity (%)': [10.0, 100.0],
    'Ambient Pressure (mbar)': [799.0, 1035.0],
    'Exhaust Vacuum (cmHg)': [3.0, 12.0],
    'Model Weight (RF vs XGB)': [0.0, 1.0]
}

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
    model_files = {
        'rf_model': 'rf_model.joblib',
        'xgb_model': 'xgb_model.joblib',
        'scaler': 'scaler.joblib'
    }
    
    loaded_models = {}
    for name, file in model_files.items():
        try:
            if not os.path.exists(file):
                st.error(f"Model file not found: {file}")
                st.stop()
            loaded_models[name] = joblib.load(file)
        except Exception as e:
            st.error(f"Error loading {name}: {str(e)}")
            st.stop()
    
    return loaded_models['rf_model'], loaded_models['xgb_model'], loaded_models['scaler']

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

    # Input sliders
    st.subheader("Input Parameters")
    inputs = {}
    for feature, (low, high) in FEATURE_BOUNDS.items():
        default = (low + high) / 2
        step = 0.1 if feature in ['Ambient Temperature (°C)', 'Model Weight (RF vs XGB)'] else 1.0
        inputs[feature] = st.slider(
            feature, low, high, default, step,
            help=f"Adjust {feature} between {low} and {high}",
            key=f"slider_{feature}"  # Unique key for each slider
        )

    # Reset button
    if st.button("🔄 Reset to Defaults"):
        for feature in FEATURE_BOUNDS:
            st.session_state[f"slider_{feature}"] = (FEATURE_BOUNDS[feature][0] + FEATURE_BOUNDS[feature][1]) / 2
        st.rerun()

# ========== MAIN CONTENT ==========
st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output using ambient conditions with an ensemble of Random Forest & XGBoost models.")

# Prepare input for prediction
feature_names = list(FEATURE_BOUNDS.keys())[:-1]  # Exclude weight
input_features = np.array([inputs[f] for f in feature_names]).reshape(1, -1)
input_weight = inputs['Model Weight (RF vs XGB)']

# Make predictions - this will run on every slider change
with st.spinner("Calculating predictions..."):
    try:
        scaled_features = scaler.transform(input_features)
        rf_pred = rf_model.predict(scaled_features)[0]
        xgb_pred = xgb_model.predict(scaled_features)[0]
        ensemble_pred = input_weight * rf_pred + (1 - input_weight) * xgb_pred
        
        # Store predictions in session state
        st.session_state['predictions'] = {
            'rf': rf_pred,
            'xgb': xgb_pred,
            'ensemble': ensemble_pred
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

# Display results - will update reactively
st.subheader("🔢 Model Predictions")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Random Forest", 
             f"{st.session_state['predictions']['rf']:.2f} MW", 
             delta_color="off")
with col2:
    st.metric("XGBoost", 
             f"{st.session_state['predictions']['xgb']:.2f} MW", 
             delta_color="off")
with col3:
    st.metric(
        f"Ensemble (Weight: {input_weight:.2f})", 
        f"{st.session_state['predictions']['ensemble']:.2f} MW",
        delta=f"{(st.session_state['predictions']['ensemble'] - (st.session_state['predictions']['rf'] + st.session_state['predictions']['xgb'])/2):.2f} vs avg"
    )

# Visualization
st.subheader("📊 Prediction Comparison")
fig, ax = plt.subplots(figsize=(10, 5))
models = ['Random Forest', 'XGBoost', 'Ensemble']
values = [
    st.session_state['predictions']['rf'],
    st.session_state['predictions']['xgb'], 
    st.session_state['predictions']['ensemble']
]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.bar(models, values, color=colors)
ax.set_ylabel('Power Output (MW)')
ax.set_title('Model Predictions Comparison')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom')

st.pyplot(fig)

# Rest of your code (CSV upload, batch prediction etc.) remains the same...
