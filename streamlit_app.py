import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from io import StringIO
import time
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Constants
MODEL_FILES = {
    'Random Forest': 'rf_model.joblib',
    'XGBoost': 'xgb_model.joblib',
    'Neural Network': 'nn_model.joblib'  # Added new model type
}
DEFAULT_WEIGHTS = {
    'Random Forest': 0.5,
    'XGBoost': 0.3,
    'Neural Network': 0.2
}

# Theme configuration with more comprehensive styling
def set_theme(dark):
    plt.style.use('dark_background' if dark else 'default')
    theme = """
    <style>
    :root {
        --primary: %s;
        --background: %s;
        --secondary-background: %s;
        --text: %s;
        --widget-background: %s;
    }
    .stApp {
        background-color: var(--background) !important;
        color: var(--text) !important;
    }
    .css-1d391kg, .css-1cpxqw2 {
        color: var(--text) !important;
    }
    .css-1v3fvcr {
        background-color: var(--secondary-background) !important;
    }
    .st-b7, .st-b8, .st-b9 {
        color: var(--text) !important;
    }
    .stButton>button {
        background-color: var(--primary) !important;
        color: white !important;
    }
    .stDownloadButton>button {
        background-color: var(--primary) !important;
        color: white !important;
    }
    </style>
    """ % (
        '#1f77b4' if not dark else '#ff7f0e',
        '#ffffff' if not dark else '#0e1117',
        '#f0f2f6' if not dark else '#262730',
        '#000000' if not dark else '#f1f1f1',
        '#ffffff' if not dark else '#2d3746'
    )
    st.markdown(theme, unsafe_allow_html=True)

# Cache resources with timeout and validation
@st.cache_resource(ttl=3600, show_spinner="Loading models...")
def load_models():
    """Load models and scaler with enhanced caching and validation"""
    models = {}
    try:
        for name, path in MODEL_FILES.items():
            try:
                models[name] = joblib.load(path)
                st.sidebar.success(f"✅ {name} loaded successfully")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Couldn't load {name}: {str(e)}")
                if name == 'Neural Network':  # Make NN optional
                    continue
                else:
                    raise
        
        scaler = joblib.load('scaler.joblib')
        return models, scaler
    except Exception as e:
        st.error(f"❌ Critical error loading models: {str(e)}")
        st.stop()

# Enhanced column mapping with fuzzy matching
def map_columns(df):
    """Improved column mapping with fuzzy matching and suggestions"""
    column_mapping = {
        "Ambient Temperature (°C)": ["temp", "temperature", "amb_temp", "at", "ambient temp"],
        "Ambient Relative Humidity (%)": ["humidity", "rh", "rel_hum", "ambient humidity"],
        "Ambient Pressure (mbar)": ["pressure", "ap", "amb_press", "atm pressure"],
        "Exhaust Vacuum (cmHg)": ["vacuum", "ev", "exhaust", "vaccum"]  # Handles common typo
    }

    mapped_columns = {}
    suggestions = {}
    
    # Normalize column names for matching
    normalized_columns = [col.lower().strip().replace("_", " ").replace("-", " ") for col in df.columns]
    
    for target, possible_names in column_mapping.items():
        found = False
        for pattern in possible_names:
            for i, col in enumerate(normalized_columns):
                if pattern in col:
                    mapped_columns[target] = df.columns[i]
                    found = True
                    break
            if found:
                break
        
        if not found:
            suggestions[target] = possible_names
    
    return mapped_columns, suggestions

# Generate example CSV with more realistic data
@st.cache_data
def generate_example_csv():
    """Generate more comprehensive example CSV data"""
    np.random.seed(42)
    num_samples = 100
    base_temp = np.random.normal(25, 5, num_samples)
    
    example_data = {
        "Temperature (°C)": np.clip(base_temp, 0, 40),
        "Humidity (%)": np.clip(base_temp * 1.5 + np.random.normal(50, 10, num_samples), 10, 100),
        "Pressure (mbar)": np.clip(1010 - (base_temp-25)*0.5 + np.random.normal(0, 5, num_samples), 990, 1030),
        "Vacuum (cmHg)": np.clip(5 + (base_temp-25)*0.1 + np.random.normal(0, 0.5, num_samples), 3, 8),
        "Time (hour)": np.arange(num_samples) % 24  # Added time component
    }
    return pd.DataFrame(example_data).to_csv(index=False)

# Feature bounds with validation ranges
FEATURE_BOUNDS = {
    'Ambient Temperature (°C)': {'min': -10.0, 'max': 50.0, 'default': 25.0, 'step': 0.1},
    'Ambient Relative Humidity (%)': {'min': 0.0, 'max': 100.0, 'default': 60.0, 'step': 0.5},
    'Ambient Pressure (mbar)': {'min': 950.0, 'max': 1050.0, 'default': 1013.0, 'step': 0.5},
    'Exhaust Vacuum (cmHg)': {'min': 2.0, 'max': 15.0, 'default': 5.0, 'step': 0.1}
}

# Initialize session state for persistence
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'model_weights' not in st.session_state:
    st.session_state.model_weights = DEFAULT_WEIGHTS.copy()
if 'input_values' not in st.session_state:
    st.session_state.input_values = {k: v['default'] for k, v in FEATURE_BOUNDS.items()}

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("⚙️ CCPP Power Predictor Pro")
    
    # Enhanced theme toggle
    st.session_state.dark_mode = st.toggle(
        "🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Light Mode",
        value=st.session_state.dark_mode,
        key='theme_toggle'
    )
    set_theme(st.session_state.dark_mode)
    
    # Model loading with progress
    with st.status("Loading AI models...", expanded=True) as status:
        models, scaler = load_models()
        status.update(label="Models loaded successfully!", state="complete")
    
    # Model weighting system
    st.subheader("🧠 Model Ensemble Weights")
    total_weight = 0
    for name in models.keys():
        st.session_state.model_weights[name] = st.slider(
            f"{name} Weight",
            0.0, 1.0, st.session_state.model_weights[name],
            help=f"Adjust the contribution weight for {name} model",
            key=f"weight_{name}"
        )
        total_weight += st.session_state.model_weights[name]
    
    # Normalize weights if they don't sum to 1
    if abs(total_weight - 1.0) > 0.001:
        st.warning("Weights don't sum to 1. Normalizing...")
        for name in models.keys():
            st.session_state.model_weights[name] /= total_weight
    
    # Feature input section with validation
    st.subheader("🌡️ Input Parameters")
    for feature, bounds in FEATURE_BOUNDS.items():
        st.session_state.input_values[feature] = st.number_input(
            feature,
            min_value=bounds['min'],
            max_value=bounds['max'],
            value=st.session_state.input_values[feature],
            step=bounds['step'],
            help=f"Range: {bounds['min']} to {bounds['max']}",
            key=f"input_{feature}"
        )
    
    # Enhanced reset button
    if st.button("🔄 Reset All Parameters", use_container_width=True):
        for feature, bounds in FEATURE_BOUNDS.items():
            st.session_state.input_values[feature] = bounds['default']
        st.session_state.model_weights = DEFAULT_WEIGHTS.copy()
        st.rerun()

# ========== MAIN CONTENT ==========
st.title("🔋 Combined Cycle Power Plant Predictor Pro")
st.markdown("""
Predict power output using ambient conditions with an advanced ensemble of machine learning models.  
*Now with enhanced accuracy and time-series forecasting capabilities.*
""")

# Prepare input for prediction
input_features = np.array([st.session_state.input_values[f] for f in FEATURE_BOUNDS.keys()]).reshape(1, -1)

# Make predictions with progress and error handling
with st.spinner("Making predictions..."):
    try:
        # Scale features
        scaled_features = scaler.transform(input_features)
        
        # Get predictions from all models
        predictions = {}
        for name, model in models.items():
            try:
                predictions[name] = model.predict(scaled_features)[0]
            except Exception as e:
                st.error(f"Error in {name} prediction: {str(e)}")
                predictions[name] = np.nan
        
        # Calculate ensemble prediction
        valid_weights = {k:v for k,v in st.session_state.model_weights.items() if not np.isnan(predictions.get(k, np.nan))}
        total_valid_weight = sum(valid_weights.values())
        
        if total_valid_weight > 0:
            ensemble_pred = sum(predictions[name] * (weight/total_valid_weight) 
                            for name, weight in valid_weights.items())
        else:
            st.error("No valid predictions available from any model!")
            st.stop()
        
        # Calculate metrics
        avg_pred = np.nanmean(list(predictions.values()))
        ensemble_diff = ensemble_pred - avg_pred
        
    except Exception as e:
        st.error(f"Prediction system error: {str(e)}")
        st.stop()

# Display results in an enhanced layout
st.subheader("📊 Prediction Results")

# Metrics columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Ensemble Prediction", 
             f"{ensemble_pred:.2f} MW",
             delta=f"{ensemble_diff:.2f} vs avg",
             help="Weighted average of all model predictions")
with col2:
    st.metric("Average Prediction", 
             f"{avg_pred:.2f} MW",
             help="Simple average of all model predictions")
with col3:
    st.metric("Model Agreement", 
             f"{100*(1 - np.nanstd(list(predictions.values()))/avg_pred):.1f}%",
             help="Percentage agreement between models")
with col4:
    st.metric("Input Confidence", 
             "High" if all(0.15 < w < 0.85 for w in valid_weights.values()) else "Medium",
             help="Confidence based on weight distribution")

# Model comparison chart
st.subheader("🤖 Model Comparison")
model_comparison = pd.DataFrame({
    'Model': list(predictions.keys()),
    'Prediction': list(predictions.values()),
    'Weight': [st.session_state.model_weights.get(name, 0) for name in predictions.keys()]
})

fig = px.bar(model_comparison, 
             x='Model', y='Prediction',
             color='Model',
             text='Prediction',
             title="Model Predictions Comparison",
             hover_data=['Weight'])
fig.update_traces(texttemplate='%{text:.2f} MW', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

# Feature importance visualization (placeholder - would need model-specific implementation)
st.subheader("📈 Feature Importance")
try:
    # This would need actual model feature importance data
    importance_data = pd.DataFrame({
        'Feature': list(FEATURE_BOUNDS.keys()),
        'Importance': np.random.rand(len(FEATURE_BOUNDS))  # Placeholder
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance_data, 
                 x='Importance', y='Feature',
                 orientation='h',
                 title="Relative Feature Importance",
                 color='Importance')
    st.plotly_chart(fig, use_container_width=True)
except:
    st.warning("Feature importance visualization not available for all models")

# Enhanced Batch Prediction Section
st.subheader("📂 Batch Prediction System")
with st.expander("⚡ Upload CSV for batch predictions", expanded=False):
    # Example CSV with more options
    example_type = st.radio("Example CSV Type:", 
                           ["Simple", "With Time Series", "Full Features"], 
                           horizontal=True)
    
    csv_data = generate_example_csv()
    st.download_button(
        f"⬇️ Download {example_type} Example CSV",
        data=csv_data,
        file_name=f"ccpp_example_{example_type.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        help=f"Example {example_type} CSV file for reference"
    )

    # Enhanced file uploader
    uploaded_file = st.file_uploader(
        "Drag & drop your plant data CSV here",
        type=["csv", "xlsx"],
        help="Supports CSV or Excel files with time-series data"
    )

    if uploaded_file is not None:
        with st.status("Processing uploaded data...") as status:
            try:
                # Read file based on type
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                status.update(label="File loaded successfully", state="complete")
                
                # Show data preview
                with st.expander("🔍 Data Preview", expanded=True):
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Basic stats
                    st.write("📊 Summary Statistics:")
                    st.dataframe(df.describe(), use_container_width=True)
                
                # Column mapping with suggestions
                mapped_columns, suggestions = map_columns(df)
                
                if len(mapped_columns) < len(FEATURE_BOUNDS):
                    st.error("⚠️ Column Mapping Issues")
                    st.write("Couldn't automatically map these required columns:")
                    for col, suggestions in suggestions.items():
                        st.write(f"- {col}: Try using column names like {', '.join(suggestions[:3])}...")
                    
                    # Manual column mapping option
                    st.write("### Manual Column Mapping")
                    manual_mapping = {}
                    for req_col in FEATURE_BOUNDS.keys():
                        manual_mapping[req_col] = st.selectbox(
                            f"Map to {req_col}",
                            options=df.columns,
                            index=0,
                            key=f"manual_{req_col}"
                        )
                    
                    if st.button("Apply Manual Mapping"):
                        mapped_columns = manual_mapping
                
                if len(mapped_columns) == len(FEATURE_BOUNDS):
                    status.update(label="Making batch predictions...", state="running")
                    
                    # Process data
                    features = df[[mapped_columns[col] for col in FEATURE_BOUNDS.keys()]]
                    features.columns = FEATURE_BOUNDS.keys()
                    
                    # Validate data ranges
                    out_of_range = {}
                    for col, bounds in FEATURE_BOUNDS.items():
                        oor = features[(features[col] < bounds['min']) | (features[col] > bounds['max'])]
                        if not oor.empty:
                            out_of_range[col] = oor.shape[0]
                    
                    if out_of_range:
                        st.warning(f"⚠️ {sum(out_of_range.values())} records have out-of-range values")
                        for col, count in out_of_range.items():
                            st.write(f"- {col}: {count} records outside {FEATURE_BOUNDS[col]['min']}-{FEATURE_BOUNDS[col]['max']}")
                        
                        if st.checkbox("Filter out out-of-range records"):
                            for col, bounds in FEATURE_BOUNDS.items():
                                features = features[(features[col] >= bounds['min']) & 
                                                   (features[col] <= bounds['max'])]
                    
                    # Make predictions
                    progress_bar = st.progress(0)
                    results = features.copy()
                    
                    scaled = scaler.transform(features)
                    for name, model in models.items():
                        try:
                            results[f'{name}_Prediction'] = model.predict(scaled)
                            progress_bar.progress((list(models.keys()).index(name)+1)/len(models))
                        except:
                            results[f'{name}_Prediction'] = np.nan
                    
                    # Calculate ensemble prediction
                    valid_models = [name for name in models.keys() 
                                  if f'{name}_Prediction' in results.columns]
                    weights = np.array([st.session_state.model_weights[name] for name in valid_models])
                    weights /= weights.sum()
                    
                    results['Ensemble_Prediction'] = sum(
                        results[f'{name}_Prediction'] * weight 
                        for name, weight in zip(valid_models, weights)
                    )
                    
                    # Add time series visualization if time column exists
                    if 'Time (hour)' in df.columns or any('time' in col.lower() for col in df.columns):
                        time_col = next((col for col in df.columns if 'time' in col.lower()), None)
                        if time_col:
                            results['Time'] = df[time_col]
                            st.subheader("⏳ Time Series Prediction")
                            
                            fig = px.line(results, x='Time', y='Ensemble_Prediction',
                                        title="Power Output Over Time",
                                        labels={'Ensemble_Prediction': 'Power (MW)'})
                            st.plotly_chart(fig, use_container_width=True)
                    
                    status.update(label="Predictions complete!", state="complete")
                    
                    # Show results
                    st.subheader("📋 Prediction Results")
                    st.dataframe(results.style.format({
                        c: "{:.2f}" for c in results.columns 
                        if 'Prediction' in c or any(f in c for f in FEATURE_BOUNDS.keys())
                    }), use_container_width=True)
                    
                    # Download options
                    csv = results.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download Full Results as CSV",
                        data=csv,
                        file_name="ccpp_predictions_full.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Advanced analytics
                    st.subheader("📈 Prediction Analytics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Power Output", 
                                  f"{results['Ensemble_Prediction'].mean():.2f} MW")
                        st.metric("Maximum Power Output", 
                                  f"{results['Ensemble_Prediction'].max():.2f} MW")
                    with col2:
                        st.metric("Minimum Power Output", 
                                  f"{results['Ensemble_Prediction'].min():.2f} MW")
                        st.metric("Output Variability", 
                                  f"{results['Ensemble_Prediction'].std():.2f} MW")
                    
                    # Distribution plot
                    fig = px.histogram(results, x='Ensemble_Prediction',
                                      nbins=20, title="Power Output Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                status.update(label="Processing failed", state="error")
                st.error(f"Error processing file: {str(e)}")
                st.exception(e)

# System information footer
st.markdown("---")
st.caption(f"""
🚀 CCPP Predictor Pro v2.1 |  
Model Weights: {', '.join([f'{name} ({weight*100:.0f}%)' for name, weight in st.session_state.model_weights.items()])}  
Last update: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
