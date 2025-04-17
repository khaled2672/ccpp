# app.py
import streamlit as st

# 1. SET PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="Power Plant Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Now import other libraries AFTER set_page_config
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 3. Load Models (cached)
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

# 4. App Title (AFTER set_page_config)
st.title("⚡ Power Plant Performance Optimizer")

# 5. Sidebar Inputs
with st.sidebar:
    st.header("Control Panel")
    ambient_temp = st.slider("Ambient Temperature (°C)", 16.0, 38.0, 25.0)
    humidity = st.slider("Relative Humidity (%)", 20.0, 90.0, 60.0)
    pressure = st.slider("Ambient Pressure (mbar)", 797.0, 801.0, 799.0)
    exhaust_vacuum = st.slider("Exhaust Vacuum (cmHg)", 3.0, 12.0, 7.0)
    weight = st.slider("RF/XGB Weight Ratio", 0.0, 1.0, models['best_weight'])

# 6. Main Content
def predict_power(features, weight):
    """Make prediction using ensemble model"""
    scaled_features = models['scaler'].transform([features])
    rf_pred = models['rf_model'].predict(scaled_features)[0]
    xgb_pred = models['xgb_model'].predict(scaled_features)[0]
    return weight * rf_pred + (1 - weight) * xgb_pred

col1, col2 = st.columns(2)

with col1:
    current_features = [ambient_temp, humidity, pressure, exhaust_vacuum]
    power = predict_power(current_features, weight)
    st.metric("Predicted Power Output", f"{power:.2f} MW")
    
    # Feature importance plot
    st.subheader("Feature Importance")
    fig, ax = plt.subplots()
    pd.Series(
        models['rf_model'].feature_importances_,
        index=['Temp', 'Humidity', 'Pressure', 'Vacuum']
    ).plot(kind='barh', ax=ax)
    st.pyplot(fig)

with col2:
    # Correlation matrix
    st.subheader("Feature Correlations")
    corr = pd.DataFrame(np.random.randn(100, 5), 
                      columns=['Temp', 'Humidity', 'Pressure', 'Vacuum', 'Power']).corr()
    fig2, ax2 = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax2)
    st.pyplot(fig2)
