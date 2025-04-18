import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from io import BytesIO

# 1. SET PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="Power Plant Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .header-text {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    .stSlider > div > div > div > div {
        background-color: #4f8bf9 !important;
    }
    .st-b7 {
        background-color: #4f8bf9 !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. Load Models (cached)
@st.cache_resource
def load_models():
    try:
        return {
            'rf_model': joblib.load('random_forest_model.joblib'),
            'xgb_model': joblib.load('xgboost_model.joblib'),
            'scaler': joblib.load('scaler.joblib'),
            'best_weight': np.load('best_weight.npy').item(),
            'feature_names': ['Ambient Temperature', 'Relative Humidity', 
                            'Ambient Pressure', 'Exhaust Vacuum']
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

def generate_shap_plot(model, features, feature_names):
    """Generate SHAP plot for model explanation"""
    explainer = shap.Explainer(model)
    shap_values = explainer(models['scaler'].transform([features]))
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.tight_layout()
    return fig

# 4. Column mapping with improved matching
def map_columns(df):
    """Map user-uploaded CSV columns to the required features with fuzzy matching"""
    column_mapping = {
        "Ambient Temperature": ["temp", "temperature", "amb_temp", "at", "ambient_temp"],
        "Relative Humidity": ["humidity", "rh", "rel_hum", "ambient_humidity"],
        "Ambient Pressure": ["pressure", "ap", "amb_press", "ambient_pressure"],
        "Exhaust Vacuum": ["vacuum", "ev", "exhaust_vac", "exhaust_vacuum"]
    }
    
    # Convert all column names to lowercase for case-insensitive matching
    df_columns_lower = [str(col).lower() for col in df.columns]
    mapped_columns = {}
    
    for target, possible_names in column_mapping.items():
        for pattern in possible_names:
            matches = [col for col in df_columns_lower if pattern in col]
            if matches:
                original_col_name = df.columns[df_columns_lower.index(matches[0])]
                mapped_columns[target] = original_col_name
                break
    
    if len(mapped_columns) < 4:
        missing_cols = [col for col in column_mapping.keys() if col not in mapped_columns]
        st.error(f"Could not automatically detect columns for: {', '.join(missing_cols)}")
        st.info("Please rename your columns to include these keywords: temperature, humidity, pressure, vacuum")
        return None
    
    return df.rename(columns=mapped_columns)

# 5. App Interface
st.title("⚡ Power Plant Performance Optimizer")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    st.subheader("Environmental Conditions")
    ambient_temp = st.slider("Ambient Temperature (°C)", 16.0, 38.0, 25.0, 0.1)
    humidity = st.slider("Relative Humidity (%)", 20.0, 90.0, 60.0, 0.1)
    pressure = st.slider("Ambient Pressure (mbar)", 990.0, 1030.0, 1013.0, 0.1)
    exhaust_vacuum = st.slider("Exhaust Vacuum (cmHg)", 25.0, 85.0, 50.0, 0.1)
    
    st.subheader("Display Options")
    show_individual = st.checkbox("Show Individual Model Predictions", value=True)
    show_shap = st.checkbox("Show Model Explanations (SHAP)", value=True)
    dark_mode = st.toggle("🌙 Dark Mode", value=False)

# Dark mode styling
if dark_mode:
    st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        .metric-card {
            background-color: #262730 !important;
            color: white !important;
        }
        .css-1v3fvcr {
            background-color: #262730 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Get predictions
current_features = [ambient_temp, humidity, pressure, exhaust_vacuum]
predictions = predict_power(current_features)

# Main Display
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("🏭 Optimal Power Prediction", 
             f"{predictions['ensemble']:.2f} MW",
             help="Combined prediction using both models")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if show_individual:
        st.subheader("🔍 Model Comparison")
        col1a, col1b = st.columns(2)
        with col1a:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🌲 Random Forest", f"{predictions['rf']:.2f} MW",
                     delta=f"{predictions['rf']-predictions['ensemble']:.2f} vs ensemble")
            st.markdown('</div>', unsafe_allow_html=True)
        with col1b:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🚀 XGBoost", f"{predictions['xgb']:.2f} MW",
                     delta=f"{predictions['xgb']-predictions['ensemble']:.2f} vs ensemble")
            st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📊 Model Composition")
    tab1, tab2 = st.tabs(["⚖️ Weights", "📈 Feature Importance"])
    
    with tab1:
        st.write(f"**Ensemble Weighting:** {models['best_weight']*100:.1f}% RF / {(1-models['best_weight'])*100:.1f}% XGB")
        fig1, ax1 = plt.subplots()
        ax1.pie([models['best_weight'], 1-models['best_weight']], 
               labels=['Random Forest', 'XGBoost'],
               autopct='%1.1f%%',
               colors=['#1f77b4', '#ff7f0e'])
        st.pyplot(fig1)
    
    with tab2:
        feature_imp = pd.DataFrame({
            'Feature': models['feature_names'],
            'Importance': models['rf_model'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig2, ax2 = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis', ax=ax2)
        ax2.set_title('Random Forest Feature Importance')
        st.pyplot(fig2)

with col2:
    st.subheader("📈 Prediction Visualization")
    
    # Interactive prediction plot
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    models_data = {
        'Random Forest': predictions['rf'],
        'XGBoost': predictions['xgb'],
        'Ensemble': predictions['ensemble']
    }
    pd.Series(models_data).plot(kind='bar', ax=ax3, 
                              color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.set_ylabel("Power Output (MW)")
    ax3.set_title("Model Predictions Comparison")
    plt.xticks(rotation=0)
    st.pyplot(fig3)
    
    if show_shap:
        st.subheader("🤖 Model Explanation (SHAP)")
        explanation_tabs = st.tabs(["Random Forest", "XGBoost"])
        
        with explanation_tabs[0]:
            st.pyplot(generate_shap_plot(models['rf_model'], current_features, models['feature_names']))
        
        with explanation_tabs[1]:
            st.pyplot(generate_shap_plot(models['xgb_model'], current_features, models['feature_names']))

# 6. Batch Prediction with CSV Upload
st.subheader("📂 Batch Prediction via CSV Upload")
uploaded_file = st.file_uploader("Upload your plant data (CSV format)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        with st.expander("👀 View Uploaded Data"):
            st.dataframe(df.head())
        
        df_processed = map_columns(df)
        if df_processed is not None:
            st.success("✅ Columns successfully mapped!")
            
            # Check if we have actual values for optimization
            has_actual = 'Actual Power' in df_processed.columns
            
            if has_actual:
                st.info("🔍 Actual power values detected - will calculate model accuracy")
            
            features = df_processed[models['feature_names']]
            scaled = models['scaler'].transform(features)
            rf_preds = models['rf_model'].predict(scaled)
            xgb_preds = models['xgb_model'].predict(scaled)
            
            # Use either optimized or default weights
            use_optimized = st.checkbox("Optimize weights based on actual values", value=has_actual, disabled=not has_actual)
            
            if use_optimized and has_actual:
                y_true = df_processed['Actual Power'].values
                best_w = 0.5
                best_score = 0
                
                # Find best weight
                for w in np.linspace(0, 1, 101):
                    blended = w * rf_preds + (1 - w) * xgb_preds
                    score = r2_score(y_true, blended)
                    if score > best_score:
                        best_score = score
                        best_w = w
                
                st.success(f"🎯 Optimized weight: {best_w:.2f} (R2: {best_score:.3f})")
                final_preds = best_w * rf_preds + (1 - best_w) * xgb_preds
            else:
                best_w = models['best_weight']
                final_preds = best_w * rf_preds + (1 - best_w) * xgb_preds
            
            # Add predictions to dataframe
            df_processed['Predicted Power (MW)'] = final_preds
            
            if has_actual:
                df_processed['Prediction Error'] = df_processed['Actual Power'] - df_processed['Predicted Power (MW)']
                mae = mean_absolute_error(df_processed['Actual Power'], df_processed['Predicted Power (MW)'])
                r2 = r2_score(df_processed['Actual Power'], df_processed['Predicted Power (MW)'])
                
                st.metric("📉 Mean Absolute Error", f"{mae:.2f} MW")
                st.metric("📊 R2 Score", f"{r2:.3f}")
            
            with st.expander("🔍 View Predictions"):
                st.dataframe(df_processed)
            
            # Download button
            csv = df_processed.to_csv(index=False).encode()
            st.download_button(
                label="⬇️ Download Predictions",
                data=csv,
                file_name="power_predictions.csv",
                mime="text/csv"
            )
            
            # Plot actual vs predicted if available
            if has_actual:
                st.subheader("📊 Actual vs Predicted")
                fig5, ax5 = plt.subplots()
                ax5.scatter(df_processed['Actual Power'], df_processed['Predicted Power (MW)'], alpha=0.5)
                ax5.plot([df_processed['Actual Power'].min(), df_processed['Actual Power'].max()],
                        [df_processed['Actual Power'].min(), df_processed['Actual Power'].max()], 
                        'r--')
                ax5.set_xlabel("Actual Power (MW)")
                ax5.set_ylabel("Predicted Power (MW)")
                st.pyplot(fig5)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: 0.8rem;
    color: #6c757d;
    text-align: center;
    padding: 10px;
}
</style>
<div class="footer">
    Power Plant Optimization Dashboard • Powered by Streamlit • Models: Random Forest & XGBoost
</div>
""", unsafe_allow_html=True)
