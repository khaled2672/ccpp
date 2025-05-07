import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="CCPP Power Output Prediction",
    page_icon="⚡",
    layout="wide"
)

# Title and description
st.title("⚡ Combined Cycle Power Plant (CCPP) Power Output Prediction")
st.write("""
This app predicts the electrical energy output of a combined cycle power plant 
based on ambient temperature, exhaust vacuum, ambient pressure, and relative humidity.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", 
                          ["Prediction", "Data Exploration", "Model Information"])

# Load dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"
    # In practice, you might want to download and extract this locally first
    # For this example, we'll use a sample approach
    data = pd.read_excel("Folds5x2_pp.xlsx")  # You should have this file or load from URL
    return data

try:
    df = load_data()
except:
    st.warning("Couldn't load the dataset. Using sample data.")
    # Sample data if loading fails
    data = {
        'AT': [14.96, 25.18, 5.11, 20.86, 10.82],
        'V': [41.76, 62.96, 39.40, 57.32, 37.50],
        'AP': [1024.07, 1020.04, 1012.16, 1010.24, 1003.19],
        'RH': [73.17, 59.08, 94.39, 76.64, 72.88],
        'PE': [463.26, 444.37, 488.56, 446.48, 473.90]
    }
    df = pd.DataFrame(data)

# Train model (or load pre-trained)
@st.cache_resource
def train_model():
    X = df[['AT', 'V', 'AP', 'RH']]
    y = df['PE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model()

# Prediction page
if options == "Prediction":
    st.header("Power Output Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        at = st.slider("Ambient Temperature (AT) in °C", 
                       min_value=0.0, max_value=40.0, value=25.0, step=0.1)
        v = st.slider("Exhaust Vacuum (V) in cm Hg", 
                     min_value=25.0, max_value=85.0, value=50.0, step=0.1)
    
    with col2:
        st.subheader("")
        ap = st.slider("Ambient Pressure (AP) in mbar", 
                      min_value=990.0, max_value=1040.0, value=1013.0, step=0.1)
        rh = st.slider("Relative Humidity (RH) in %", 
                      min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    
    if st.button("Predict Power Output"):
        input_data = [[at, v, ap, rh]]
        prediction = model.predict(input_data)[0]
        
        st.success(f"Predicted Electrical Power Output: **{prediction:.2f} MW**")
        
        # Show feature importance
        st.subheader("Feature Importance")
        feature_imp = pd.DataFrame({
            'Feature': ['AT', 'V', 'AP', 'RH'],
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
        ax.set_title("Feature Importance for Power Output Prediction")
        st.pyplot(fig)

# Data Exploration page
elif options == "Data Exploration":
    st.header("Data Exploration")
    
    st.subheader("Sample Data")
    st.write(df.head())
    
    st.subheader("Data Statistics")
    st.write(df.describe())
    
    st.subheader("Feature Distributions")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, col in enumerate(['AT', 'V', 'AP', 'RH']):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Correlation Matrix")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Model Information page
elif options == "Model Information":
    st.header("Model Information")
    
    st.subheader("Model Performance")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"- **Model Type**: Random Forest Regressor")
    st.write(f"- **Number of Trees**: 100")
    st.write(f"- **Mean Absolute Error (MAE)**: {mae:.2f} MW")
    st.write(f"- **R² Score**: {r2:.2f}")
    
    st.subheader("Actual vs Predicted Values")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("Actual Power Output (MW)")
    ax.set_ylabel("Predicted Power Output (MW)")
    ax.set_title("Actual vs Predicted Values")
    st.pyplot(fig)
    
    st.subheader("Dataset Information")
    st.write("""
    The dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011).
    The power plant consists of:
    - 3 gas turbines
    - 1 steam turbine
    - 4 heat recovery steam generators
    
    **Features**:
    - AT: Ambient Temperature (°C)
    - V: Exhaust Vacuum (cm Hg)
    - AP: Ambient Pressure (mbar)
    - RH: Relative Humidity (%)
    
    **Target**:
    - PE: Net hourly electrical energy output (MW)
    """)

# Footer
st.markdown("---")
st.markdown("""
*Note: This is a demo application for predicting power output of a Combined Cycle Power Plant.*
""")
