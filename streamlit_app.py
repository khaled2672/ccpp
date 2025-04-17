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

# 4. App Interface
st.title("⚡ Power Plant Performance Optimizer")

# Sidebar Controls
with st.sidebar:
    st.header("Control Panel")
    ambient_temp = st.slider("Ambient Temperature (°C)", 16.0, 38.0, 25.0)
    humidity = st.slider("Relative Humidity (%)", 20.0, 90.0, 60.0)
    pressure = st.slider("Ambient Pressure (mbar)", 797.0, 801.0, 799.0)
    exhaust_vacuum = st.slider("Exhaust Vacuum (cmHg)", 3.0, 12.0, 7.0)
    show_individual = st.checkbox("Show Individual Model Predictions", value=True)

# Get predictions
current_features = [ambient_temp, humidity, pressure, exhaust_vacuum]
predictions = predict_power(current_features)

# Main Display
col1, col2 = st.columns(2)

with col1:
    # Primary prediction display
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
    
    # Feature importance
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
    # Model comparison visualization
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
    
    # Correlation matrix
    st.subheader("Feature Correlations")
    corr = pd.DataFrame(np.random.randn(100, 5), 
                       columns=['Temp', 'Humidity', 'Pressure', 'Vacuum', 'Power']).corr()
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax4, cmap='coolwarm', center=0)
    st.pyplot(fig4)

st.subheader("📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload input data (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data", df.head())

    if all(col in df.columns for col in ['Ambient Temperature', 'Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum']):
        scaled = models['scaler'].transform(df[['Ambient Temperature', 'Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum']])
        rf_preds = models['rf_model'].predict(scaled)
        xgb_preds = models['xgb_model'].predict(scaled)
        final_preds = weight * rf_preds + (1 - weight) * xgb_preds
        df['Predicted Power (MW)'] = final_preds
        st.write("⚡ Predictions", df)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results as CSV", data=csv, file_name="predicted_power.csv", mime='text/csv')
    else:
        st.error("CSV must contain: Ambient Temperature, Relative Humidity, Ambient Pressure, Exhaust Vacuum")
