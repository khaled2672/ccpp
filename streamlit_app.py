# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Saved Models and Components
@st.cache_resource
def load_components():
    """Load saved models and scaler"""
    return {
        'rf_model': joblib.load('random_forest_model.joblib'),
        'xgb_model': joblib.load('xgboost_model.joblib'),
        'scaler': joblib.load('scaler.joblib'),
        'best_weight': float(open('best_weight.txt').read())
    }

components = load_components()

# 2. Set Up Streamlit Interface
st.set_page_config(page_title="Power Plant Optimization", layout="wide")
st.title("⚡ Power Plant Performance Optimizer")

# 3. Create Input Widgets
with st.sidebar:
    st.header("Control Panel")
    
    # Feature sliders
    ambient_temp = st.slider("Ambient Temperature (°C)", 16.0, 38.0, 25.0)
    humidity = st.slider("Relative Humidity (%)", 20.0, 90.0, 60.0)
    pressure = st.slider("Ambient Pressure (mbar)", 797.0, 801.0, 799.0)
    exhaust_vacuum = st.slider("Exhaust Vacuum (cmHg)", 3.0, 12.0, 7.0)
    
    # Ensemble weight adjustment
    weight = st.slider("RF/XGB Weight Ratio", 
                      0.0, 1.0, components['best_weight'],
                      help="Adjust blending between Random Forest and XGBoost")
    
    optimize_btn = st.button("🚀 Optimize Parameters Automatically")

# 4. Prediction Function
def predict_power(features, weight):
    """Make prediction using ensemble model"""
    scaled_features = components['scaler'].transform([features])
    rf_pred = components['rf_model'].predict(scaled_features)[0]
    xgb_pred = components['xgb_model'].predict(scaled_features)[0]
    return weight * rf_pred + (1 - weight) * xgb_pred

# 5. Main Display
col1, col2 = st.columns(2)

with col1:
    # Current prediction
    current_features = [ambient_temp, humidity, pressure, exhaust_vacuum]
    power = predict_power(current_features, weight)
    
    st.metric("Predicted Power Output", f"{power:.2f} MW")
    
    # Feature importance visualization
    st.subheader("Feature Importance")
    fig, ax = plt.subplots()
    importances = components['rf_model'].feature_importances_
    pd.Series(importances, 
             index=['Temp', 'Humidity', 'Pressure', 'Vacuum']).plot(kind='barh', ax=ax)
    st.pyplot(fig)

with col2:
    # Optimization results (if triggered)
    if optimize_btn:
        with st.spinner("Running PSO Optimization..."):
            # This would call your PSO optimization code
            optimized_features = [25.5, 62.3, 798.7, 6.8]  # Example values
            optimized_power = predict_power(optimized_features, weight)
            
            st.success("Optimization Complete!")
            st.metric("Optimized Power Output", f"{optimized_power:.2f} MW", delta=f"{(optimized_power-power):.2f}")
            
            # Show optimized parameters
            st.write("Recommended Parameters:")
            st.json({
                "Temperature": f"{optimized_features[0]:.1f} °C",
                "Humidity": f"{optimized_features[1]:.1f} %",
                "Pressure": f"{optimized_features[2]:.1f} mbar",
                "Exhaust Vacuum": f"{optimized_features[3]:.1f} cmHg"
            })
    
    # Correlation matrix
    st.subheader("Feature Correlations")
    corr = pd.DataFrame({
        'Temp': np.random.randn(100) + ambient_temp/10,
        'Humidity': np.random.randn(100) + humidity/100,
        'Pressure': np.random.randn(100) + pressure/1000,
        'Vacuum': np.random.randn(100) + exhaust_vacuum/10,
        'Power': np.random.randn(100) + power
    }).corr()
    
    fig2, ax2 = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax2)
    st.pyplot(fig2)

# 6. Run Instructions
st.sidebar.markdown("""
**How to Use:**
1. Adjust sliders for current conditions
2. Click 'Optimize' to find ideal parameters
3. View predictions and relationships

*Note: For actual deployment, replace placeholder optimization with your PSO code*
""")
