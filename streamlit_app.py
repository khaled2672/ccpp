import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import StringIO

# ========== THEME CONFIGURATION ==========
def set_theme(dark):
    """Configure light/dark theme with plot styling"""
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

# ========== MODEL LOADING ==========
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

# ========== COLUMN MAPPING ==========
def map_columns(df):
    """Flexibly map user-uploaded CSV columns to required features with units"""
    column_mapping = {
        "Ambient Temperature (°C)": ["temperature", "temp", "at", "ambient_temp", "ambient temperature", "t", "atmospheric temperature"],
        "Ambient Relative Humidity (%)": ["humidity", "rh", "relative_humidity", "ambient humidity", "humidity (%)", "h", "relative humidity"],
        "Ambient Pressure (mbar)": ["pressure", "ap", "ambient_pressure", "amb pressure", "pressure (mbar)", "p", "atmospheric pressure"],
        "Exhaust Vacuum (cmHg)": ["vacuum", "ev", "exhaust_vac", "exhaust_vacuum", "vacuum (cmhg)", "v", "exhaust pressure"]
    }

    mapped_columns = {}
    df_columns_lower = [col.lower().strip() for col in df.columns]
    
    for display_name, possible_names in column_mapping.items():
        found = False
        for name in possible_names:
            if name in df_columns_lower:
                original_col = df.columns[df_columns_lower.index(name)]
                mapped_columns[display_name] = original_col
                found = True
                break
        
        if not found:
            st.warning(f"⚠️ Could not find column matching: {display_name}")
    
    return mapped_columns

# ========== EXAMPLE CSV GENERATION ==========
@st.cache_data
def generate_example_csv():
    """Generate example CSV data for download"""
    example_data = {
        "Temperature": [25.0, 30.0, 27.5],
        "Humidity (%)": [60.0, 65.0, 62.5],
        "Pressure": [1010.0, 1005.0, 1007.5],
        "Vacuum": [5.0, 6.0, 5.5]
    }
    return pd.DataFrame(example_data).to_csv(index=False)

# ========== INITIALIZATION ==========
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("⚙️ CCPP Power Predictor")
    
    # Theme toggle
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    set_theme(st.session_state.dark_mode)
    
    # Instructions
    st.subheader("How to Use")
    st.markdown("""
    1. Adjust sliders to set plant conditions  
    2. View predicted power output  
    3. Compare model performance  
    4. Upload CSV for batch predictions
    """)
    
    # Model loading
    with st.spinner("Loading models..."):
        rf_model, xgb_model, scaler = load_models()

    # Input parameters with units
    feature_bounds = {
        'Ambient Temperature (°C)': [0.0, 50.0],
        'Ambient Relative Humidity (%)': [10.0, 100.0],
        'Ambient Pressure (mbar)': [799.0, 1035.0],
        'Exhaust Vacuum (cmHg)': [3.0, 12.0],
        'Model Weight (RF vs XGB)': [0.0, 1.0]
    }

    st.subheader("Input Parameters")
    inputs = {}
    for feature, (low, high) in feature_bounds.items():
        default = (low + high) / 2
        inputs[feature] = st.slider(
            feature, low, high, default,
            help=f"Recommended range: {low} to {high}"
        )

    if st.button("🔄 Reset to Defaults"):
        for feature in inputs:
            inputs[feature] = (feature_bounds[feature][0] + feature_bounds[feature][1]) / 2

# ========== MAIN INTERFACE ==========
st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict net hourly electrical energy output using ambient conditions")

# Single prediction
feature_names = list(feature_bounds.keys())[:-1]  # Exclude weight
input_features = np.array([inputs[f] for f in feature_names]).reshape(1, -1)
input_weight = inputs['Model Weight (RF vs XGB)']

with st.spinner("Calculating predictions..."):
    try:
        scaled_features = scaler.transform(input_features)
        rf_pred = rf_model.predict(scaled_features)[0]
        xgb_pred = xgb_model.predict(scaled_features)[0]
        ensemble_pred = input_weight * rf_pred + (1 - input_weight) * xgb_pred
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

# Prediction results
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

# Feature importance visualization
st.subheader("📈 Feature Importance")
try:
    display_names = [
        "Ambient Temperature (°C)",
        "Ambient Relative Humidity (%)",
        "Ambient Pressure (mbar)",
        "Exhaust Vacuum (cmHg)"
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Random Forest importance
    rf_importance = pd.Series(rf_model.feature_importances_, index=display_names)
    rf_importance.plot(kind='barh', ax=ax1, title='Random Forest', color='#1f77b4')
    ax1.set_xlabel("Importance Score")
    
    # XGBoost importance
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=display_names)
    xgb_importance.plot(kind='barh', ax=ax2, title='XGBoost', color='#ff7f0e')
    ax2.set_xlabel("Importance Score")
    
    # Add value labels
    for ax in [ax1, ax2]:
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f'{width:.2f}', 
                       (width * 1.02, p.get_y() + p.get_height()/2.),
                       ha='left', va='center')
    
    fig.tight_layout()
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not generate feature importance plots: {str(e)}")

# ========== BATCH PREDICTION ==========
st.subheader("📂 Batch Prediction")
st.markdown("Upload a CSV file containing multiple records for prediction")

# Example CSV download
with st.expander("🛠️ CSV Format Requirements"):
    st.markdown("""
    **Required columns (case insensitive):**
    - Temperature (e.g., 'temp', 'ambient_temp', 'AT')
    - Humidity (e.g., 'humidity', 'RH', 'relative_humidity')
    - Pressure (e.g., 'pressure', 'AP', 'ambient_pressure')
    - Vacuum (e.g., 'vacuum', 'EV', 'exhaust_vacuum')
    
    **Example CSV:**
    """)
    st.download_button(
        "⬇️ Download Example CSV",
        data=generate_example_csv(),
        file_name="ccpp_example_input.csv",
        mime="text/csv"
    )

uploaded_file = st.file_uploader(
    "Drag & drop your CSV file here",
    type=["csv"],
    help="CSV should contain columns for temperature, humidity, pressure and vacuum"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("❌ Uploaded file is empty")
            st.stop()
            
        st.success("✅ File uploaded successfully")
        
        with st.expander("👀 View uploaded data"):
            st.dataframe(df.head())
        
        # Column mapping
        mapped_columns = map_columns(df)
        
        # Check required columns
        required_columns = [
            "Ambient Temperature (°C)",
            "Ambient Relative Humidity (%)",
            "Ambient Pressure (mbar)",
            "Exhaust Vacuum (cmHg)"
        ]
        
        missing_cols = [col for col in required_columns if col not in mapped_columns]
        if missing_cols:
            st.error("❌ Missing required columns in uploaded file")
            st.markdown("**Could not find these required columns:**")
            for col in missing_cols:
                st.markdown(f"- {col}")
            
            st.markdown("**Please check your CSV includes columns with these names or similar:**")
            st.code("""
            Temperature (e.g., temp, AT, ambient_temp)\n
            Humidity (e.g., humidity, RH, relative_humidity)\n
            Pressure (e.g., pressure, AP, ambient_pressure)\n
            Vacuum (e.g., vacuum, EV, exhaust_vacuum)
            """)
            st.stop()
            
        # Process data
        with st.spinner("⚙️ Processing data..."):
            try:
                # Rename columns to standard names
                df_processed = df.rename(columns=mapped_columns)
                
                # Prepare features
                features = df_processed[required_columns]
                scaled = scaler.transform(features)
                
                # Make predictions
                rf_preds = rf_model.predict(scaled)
                xgb_preds = xgb_model.predict(scaled)
                final_preds = input_weight * rf_preds + (1 - input_weight) * xgb_preds
                
                # Prepare results
                results = df_processed.copy()
                results['RF_Prediction (MW)'] = rf_preds
                results['XGB_Prediction (MW)'] = xgb_preds
                results['Ensemble_Prediction (MW)'] = final_preds
                
                st.success(f"✅ Successfully processed {len(results)} records")
                
                # Show results
                st.dataframe(results.style.format({
                    'RF_Prediction (MW)': '{:.2f}',
                    'XGB_Prediction (MW)': '{:.2f}',
                    'Ensemble_Prediction (MW)': '{:.2f}'
                }))
                
                # Download button
                csv = results.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ Download Predictions",
                    data=csv,
                    file_name="ccpp_predictions.csv",
                    mime="text/csv",
                    help="Download all predictions as CSV"
                )
                
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
                
    except Exception as e:
        st.error(f"❌ File loading error: {str(e)}")

# Footer
st.markdown("---")
st.caption("""
Developed with Streamlit | Model weights: RF {:.0f}% / XGB {:.0f}%
""".format(input_weight*100, (1-input_weight)*100))
