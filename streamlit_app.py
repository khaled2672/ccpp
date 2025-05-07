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
    """Configure light/dark theme with custom styles"""
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
            .st-cb { background-color: #1a1a2e; }
            .st-cg { color: #f1f1f1 !important; }
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
            .st-cb { background-color: #f0f2f6; }
            </style>
            """,
            unsafe_allow_html=True
        )

# Cache resources for better performance
@st.cache_resource
def load_models():
    """Load models and scaler with caching and validation"""
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

# Column mapping function
def map_columns(df):
    """Map user-uploaded CSV columns to the required features with fuzzy matching"""
    column_mapping = {
        "Ambient Temperature (°C)": ["temperature", "ambient temp", "temp", "at", "amb_temp"],
        "Ambient Relative Humidity (%)": ["humidity", "relative humidity", "rh", "ambient humidity"],
        "Ambient Pressure (mbar)": ["pressure", "ambient pressure", "ap", "amb_press"],
        "Exhaust Vacuum (cmHg)": ["vacuum", "exhaust vacuum", "ev", "exh_vac"]
    }

    mapped_columns = {}
    for target, possible_names in column_mapping.items():
        for name in possible_names:
            # Case insensitive matching
            matching_cols = [col for col in df.columns if name.lower() in col.lower()]
            if matching_cols:
                mapped_columns[target] = matching_cols[0]
                break

    return mapped_columns

# Generate example CSV data
@st.cache_data
def generate_example_csv():
    """Generate example CSV data with realistic ranges"""
    example_data = {
        "Temperature": np.random.uniform(0, 50, 100),
        "Humidity": np.random.uniform(10, 100, 100),
        "Pressure": np.random.uniform(990, 1035, 100),
        "Vacuum": np.random.uniform(3, 12, 100)
    }
    return pd.DataFrame(example_data).to_csv(index=False)

# Visualization functions
def plot_predictions_comparison(rf_pred, xgb_pred, ensemble_pred):
    """Create comparison bar plot of model predictions"""
    fig, ax = plt.subplots(figsize=(8, 4))
    models = ['Random Forest', 'XGBoost', 'Ensemble']
    values = [rf_pred, xgb_pred, ensemble_pred]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars = ax.bar(models, values, color=colors)
    ax.set_ylabel('Power Output (MW)')
    ax.set_title('Model Predictions Comparison')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    return fig

def plot_feature_importance(models, feature_names):
    """Plot feature importance for both models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # RF Feature Importance
    rf_importance = models[0].feature_importances_
    sns.barplot(x=rf_importance, y=feature_names, ax=ax1, palette='Blues_d')
    ax1.set_title('Random Forest Feature Importance')
    
    # XGB Feature Importance
    xgb_importance = models[1].feature_importances_
    sns.barplot(x=xgb_importance, y=feature_names, ax=ax2, palette='Oranges_d')
    ax2.set_title('XGBoost Feature Importance')
    
    plt.tight_layout()
    return fig

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
            help=f"Adjust {feature} between {low} and {high}"
        )

    # Reset button
    if st.button("🔄 Reset to Defaults"):
        for feature in inputs:
            inputs[feature] = (FEATURE_BOUNDS[feature][0] + FEATURE_BOUNDS[feature][1]) / 2

# ========== MAIN CONTENT ==========
st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output using ambient conditions with an ensemble of Random Forest & XGBoost models.")

# Prepare input for prediction
feature_names = list(FEATURE_BOUNDS.keys())[:-1]  # Exclude weight
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

# Visualizations
st.subheader("📊 Prediction Visualizations")
tab1, tab2 = st.tabs(["Prediction Comparison", "Feature Importance"])

with tab1:
    st.pyplot(plot_predictions_comparison(rf_pred, xgb_pred, ensemble_pred))
    
with tab2:
    st.pyplot(plot_feature_importance([rf_model, xgb_model], feature_names))

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
            st.write(f"Shape: {df.shape}")
            
            # Basic statistics
            st.subheader("Data Statistics")
            st.dataframe(df.describe())
        
        # Column mapping
        mapped_columns = map_columns(df)
        st.info(f"Detected columns: {', '.join(mapped_columns.values())}")
        
        if len(mapped_columns) < 4:
            missing_cols = [col for col in feature_names if col not in mapped_columns]
            st.error(f"Could not find columns for: {', '.join(missing_cols)}")
            st.stop()
            
        df_processed = df.rename(columns=mapped_columns)
        required_cols = feature_names  # From FEATURE_BOUNDS
        
        # Check for missing columns after mapping
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        if missing_cols:
            st.error(f"Missing columns after mapping: {', '.join(missing_cols)}")
            st.stop()
            
        # Data validation
        invalid_ranges = []
        for col, (min_val, max_val) in FEATURE_BOUNDS.items():
            if col in df_processed.columns:
                if (df_processed[col] < min_val).any() or (df_processed[col] > max_val).any():
                    invalid_ranges.append(col)
        
        if invalid_ranges:
            st.warning(f"Values outside expected range detected in: {', '.join(invalid_ranges)}")
            
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
                
                st.success(f"Predictions completed for {len(results)} records!")
                
                # Display results
                st.dataframe(results.style.format({
                    'RF_Prediction (MW)': '{:.2f}',
                    'XGB_Prediction (MW)': '{:.2f}',
                    'Ensemble_Prediction (MW)': '{:.2f}'
                }).background_gradient(cmap='Blues' if st.session_state.dark_mode else 'YlOrBr', subset=['Ensemble_Prediction (MW)']))
                
                # Visualize predictions distribution
                st.subheader("Predictions Distribution")
                fig, ax = plt.subplots()
                sns.histplot(results['Ensemble_Prediction (MW)'], kde=True, ax=ax)
                ax.set_xlabel('Power Output (MW)')
                st.pyplot(fig)
                
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
Developed with Streamlit | Model weights: Random Forest ({:.0f}%), XGBoost ({:.0f}%)
""".format(input_weight*100, (1-input_weight)*100))
