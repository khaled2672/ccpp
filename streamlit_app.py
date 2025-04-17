# app.py
import streamlit as st

# 1. Set page config FIRST
st.set_page_config(
    page_title="Power Plant Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Import libraries after page config
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 3. Load models and data
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('random_forest_model.joblib')
        xgb_model = joblib.load('xgboost_model.joblib')
        scaler = joblib.load('scaler.joblib')
        best_weight = np.load('best_weight.npy').item()
        return rf_model, xgb_model, scaler, best_weight
    except FileNotFoundError as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data.csv")
    except FileNotFoundError:
        st.warning("⚠️ data.csv not found. Showing placeholder correlations.")
        return pd.DataFrame(np.random.randn(100, 5), columns=['Temp', 'Humidity', 'Pressure', 'Vacuum', 'Power'])

rf_model, xgb_model, scaler, best_weight = load_models()
df = load_dataset()

# 4. App Title
st.title("⚡ Power Plant Performance Optimizer")

# 5. Sidebar Inputs
with st.sidebar:
    st.header("Control Panel")
    ambient_temp = st.slider("Ambient Temperature (°C)", 16.0, 38.0, 25.0)
    humidity = st.slider("Relative Humidity (%)", 20.0, 90.0, 60.0)
    pressure = st.slider("Ambient Pressure (mbar)", 797.0, 801.0, 799.0)
    exhaust_vacuum = st.slider("Exhaust Vacuum (cmHg)", 3.0, 12.0, 7.0)
    weight = st.slider("Ensemble Weight (0 = XGB, 1 = RF)", 0.0, 1.0, float(best_weight))

# 6. Prediction Logic
def predict_power(features, weight):
    scaled_features = scaler.transform([features])
    rf_pred = rf_model.predict(scaled_features)[0]
    xgb_pred = xgb_model.predict(scaled_features)[0]
    return weight * rf_pred + (1 - weight) * xgb_pred

current_features = [ambient_temp, humidity, pressure, exhaust_vacuum]
power_output = predict_power(current_features, weight)

# 7. Main Layout
col1, col2 = st.columns(2)

with col1:
    st.metric("⚡ Predicted Power Output", f"{power_output:.2f} MW")
    
    st.subheader("🎯 Feature Importance (RF)")
    fig, ax = plt.subplots()
    pd.Series(
        rf_model.feature_importances_,
        index=['Temp', 'Humidity', 'Pressure', 'Vacuum']
    ).plot(kind='barh', color='skyblue', ax=ax)
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)

with col2:
    st.subheader("📊 Feature Correlation Matrix")
    correlation_matrix = df.corr()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

# 8. Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit, Scikit-learn, XGBoost, and PSO.")
