import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# 1. SET PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="Power Plant Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.subheader("How to Use")
    st.markdown("""
    1. Adjust sliders to set plant conditions  
    2. View the predicted power output  
    3. Compare models using the toggle  
    """)

# 2. Load Models (cached)
@st.cache_resource
def load_models():
    try:
        return {
            'rf_model': joblib.load('random_forest_model.joblib'),
            'xgb_model': joblib.load('xgboost_model.joblib'),
            'scaler': joblib.load('scaler.joblib'),
            'best_weight': np.load('best_weight.npy').item()
        }
    except FileNotFoundError as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

models = load_models()

# 3. Prediction Functions
def predict_power(features):
    """Make predictions from all models"""
    scaled_features = models['scaler'].transform([features])
    return {
        'rf': models['rf_model'].predict(scaled_features)[0],
        'xgb': models['xgb_model'].predict(scaled_features)[0],
        'ensemble': (models['best_weight'] * models['rf_model'].predict(scaled_features)[0] + 
                     (1 - models['best_weight']) * models['xgb_model'].predict(scaled_features)[0])
    }

# 4. Function to map CSV columns to required columns
def map_columns(df):
    """Map user-uploaded CSV columns to the required features."""
    column_mapping = {
        "Ambient Temperature": ["Ambient Temperature", "Temperature", "Temp", "Amb Temp", "Ambient_Temperature"],
        "Relative Humidity": ["Relative Humidity","Ambient Relative Humidity", "Humidity", "Rel Humidity", "Humidity (%)"],
        "Ambient Pressure": ["Ambient Pressure", "Pressure", "Amb Pressure", "Pressure (mbar)"],
        "Exhaust Vacuum": ["Exhaust Vacuum", "Vacuum", "Exhaust Vac", "Vacuum (cmHg)"]
    }

    mapped_columns = {}
    for target, possible_names in column_mapping.items():
        for name in possible_names:
            if name in df.columns:
                mapped_columns[target] = name
                break

    if len(mapped_columns) < 4:
        missing_cols = [col for col in column_mapping.keys() if col not in mapped_columns]
        st.error(f"Missing columns: {', '.join(missing_cols)}. Please upload a file with the required columns.")
        return None

    df = df.rename(columns=mapped_columns)
    return df

# 5. App Interface
st.title("⚡ Power Plant Performance Optimizer")

# Sidebar Controls
with st.sidebar:
    st.header("Control Panel")
    ambient_temp = st.slider("Ambient Temperature (°C)", 16.0, 38.0, 25.0)
    humidity = st.slider("Ambient Relative Humidity (%)", 20.0, 90.0, 60.0)
    pressure = st.slider("Ambient Pressure (mbar)", 797.0, 801.0, 799.0)
    exhaust_vacuum = st.slider("Exhaust Vacuum (cmHg)", 3.0, 12.0, 7.0)
    show_individual = st.checkbox("Show Individual Model Predictions", value=True)
    auto_optimize = st.checkbox("🔁 Auto-optimize ensemble weights (requires 'Actual Power' column)", value=False)

# Get predictions
current_features = [ambient_temp, humidity, pressure, exhaust_vacuum]
predictions = predict_power(current_features)

# Main Display
col1, col2 = st.columns(2)

with col1:
    st.metric("Optimal Power Prediction", 
             f"{predictions['ensemble']:.2f} MW",
             help="Combined prediction using both models")
    
    if show_individual:
        st.subheader("Individual Model Predictions")
        col1a, col1b = st.columns(2)
        with col1a:
            st.metric("Random Forest", f"{predictions['rf']:.2f} MW",
                     delta=f"{predictions['rf']-predictions['ensemble']:.2f} vs ensemble")
        with col1b:
            st.metric("XGBoost", f"{predictions['xgb']:.2f} MW",
                     delta=f"{predictions['xgb']-predictions['ensemble']:.2f} vs ensemble")

    st.subheader("Model Weights and Feature Importance")
    tab1, tab2 = st.tabs(["Model Weights", "Feature Importance"])

    with tab1:
        st.write(f"**Ensemble Weighting:** {models['best_weight']*100:.1f}% RF / {(1-models['best_weight'])*100:.1f}% XGB")
        fig1, ax1 = plt.subplots()
        ax1.pie([models['best_weight'], 1-models['best_weight']], 
               labels=['Random Forest', 'XGBoost'],
               autopct='%1.1f%%')
        st.pyplot(fig1)

    with tab2:
        fig2, ax2 = plt.subplots()
        pd.Series(models['rf_model'].feature_importances_,
                 index=['Temp', 'Humidity', 'Pressure', 'Vacuum']
                ).plot(kind='barh', ax=ax2, title='Random Forest')
        st.pyplot(fig2)

with col2:
    st.subheader("Prediction Comparison")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    models_data = {
        'Random Forest': predictions['rf'],
        'XGBoost': predictions['xgb'],
        'Ensemble': predictions['ensemble']
    }
    pd.Series(models_data).plot(kind='bar', ax=ax3, 
                              color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.set_ylabel("Power Output (MW)")
    plt.xticks(rotation=0)
    st.pyplot(fig3)

    st.subheader("Feature Correlations")
    corr = pd.DataFrame(np.random.randn(100, 5), 
                       columns=['Temp', 'Humidity', 'Pressure', 'Vacuum', 'Power']).corr()
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax4, cmap='coolwarm', center=0)
    st.pyplot(fig4)

# Function to optimize ensemble weight
def optimize_weight(rf_preds, xgb_preds, y_true):
    best_weight = 0
    lowest_mae = float('inf')
    for w in np.linspace(0, 1, 101):
        blended = w * rf_preds + (1 - w) * xgb_preds
        mae = mean_absolute_error(y_true, blended)
        if mae < lowest_mae:
            lowest_mae = mae
            best_weight = w
    return best_weight, lowest_mae

# 6. Batch Prediction with CSV Upload
st.subheader("📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload input data (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data", df.head())

    df_processed = map_columns(df)
    if df_processed is not None:
        st.write("✅ Dataset Columns Mapped Successfully")

        features = df_processed[["Ambient Temperature", "Ambient Relative Humidity", "Ambient Pressure", "Exhaust Vacuum"]]
        scaled = models['scaler'].transform(features)
        rf_preds = models['rf_model'].predict(scaled)
        xgb_preds = models['xgb_model'].predict(scaled)

        if auto_optimize and 'Actual Power' in df_processed.columns:
            y_true = df_processed['Actual Power'].values
            weight, best_mae = optimize_weight(rf_preds, xgb_preds, y_true)
            st.success(f"✅ Auto-optimized ensemble weight: {weight:.2f} RF / {1 - weight:.2f} XGB")
            st.write(f"📉 Best MAE: {best_mae:.2f} MW")
        else:
            weight = models['best_weight']
            if auto_optimize:
                st.warning("⚠️ 'Actual Power' column not found. Optimization skipped.")

        final_preds = weight * rf_preds + (1 - weight) * xgb_preds
        df_processed['Predicted Power (MW)'] = final_preds

        st.write("⚡ Predictions", df_processed)

        csv = df_processed.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results as CSV", data=csv, file_name="predicted_power.csv", mime='text/csv')
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
def set_theme(dark):
    if dark:
        st.markdown(
            """
            <style>
            body {
                background-color: #0e1117;
                color: #f1f1f1;
            }
            .stApp {
                background-color: #0e1117;
            }
            .css-1d391kg, .css-1cpxqw2 {
                color: #f1f1f1 !important;
            }
            .css-1v3fvcr {
                background-color: #262730 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            body {
                background-color: #ffffff;
                color: #000000;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

set_theme(dark_mode)
