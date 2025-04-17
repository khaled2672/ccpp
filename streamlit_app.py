# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SET PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="Power Plant Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
def predict_power(features, weight):
    """Make prediction using ensemble model"""
    scaled_features = models['scaler'].transform([features])
    rf_pred = models['rf_model'].predict(scaled_features)[0]
    xgb_pred = models['xgb_model'].predict(scaled_features)[0]
    return {
        'ensemble': weight * rf_pred + (1 - weight) * xgb_pred,
        'rf': rf_pred,
        'xgb': xgb_pred
    }

# 4. App Interface
st.title("⚡ Power Plant Performance Optimizer")

# Sidebar Controls
with st.sidebar:
    st.header("Control Panel")
    ambient_temp = st.slider("Ambient Temperature (°C)", 16.0, 38.0, 25.0)
    humidity = st.slider("Relative Humidity (%)", 20.0, 90.0, 60.0)
    pressure = st.slider("Ambient Pressure (mbar)", 797.0, 801.0, 799.0)
    exhaust_vacuum = st.slider("Exhaust Vacuum (cmHg)", 3.0, 12.0, 7.0)
    weight = st.slider("RF/XGB Weight Ratio", 0.0, 1.0, models['best_weight'])

# Main Content
current_features = [ambient_temp, humidity, pressure, exhaust_vacuum]
predictions = predict_power(current_features, weight)

# Prediction Display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Random Forest Prediction", f"{predictions['rf']:.2f} MW")
with col2:
    st.metric("XGBoost Prediction", f"{predictions['xgb']:.2f} MW")
with col3:
    st.metric("Ensemble Prediction", f"{predictions['ensemble']:.2f} MW", 
             delta=f"{(predictions['ensemble']-np.mean([predictions['rf'], predictions['xgb']])):.2f} vs average")

# Model Comparison Visualization
st.subheader("Model Comparison")
fig, ax = plt.subplots(figsize=(10, 4))
models_data = {
    'Random Forest': predictions['rf'],
    'XGBoost': predictions['xgb'],
    'Ensemble': predictions['ensemble']
}
pd.Series(models_data).plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel("Power Output (MW)")
ax.set_title("Prediction Comparison Across Models")
st.pyplot(fig)


# RF Importance
pd.Series(models['rf_model'].feature_importances_, 
         index=['Temp', 'Humidity', 'Pressure', 'Vacuum']
        plot(kind='barh', ax=ax1, title='Random Forest', color='#1f77b4')

# XGB Importance
pd.Series(models['xgb_model'].feature_importances_, 
         index=['Temp', 'Humidity', 'Pressure', 'Vacuum']
        ).plot(kind='barh', ax=ax2, title='XGBoost', color='#ff7f0e')

plt.tight_layout()
st.pyplot(fig2)

# Prediction Explanation
with st.expander("ℹ️ How these predictions were made"):
    st.markdown("""
    - **Random Forest**: Ensemble of decision trees averaging multiple predictions
    - **XGBoost**: Gradient boosted trees with sequential error correction
    - **Ensemble**: Weighted combination of both models (RF: {:.0f}%, XGB: {:.0f}%)
    
    Feature importance shows which parameters most affect each model's predictions.
    """.format(weight*100, (1-weight)*100))
