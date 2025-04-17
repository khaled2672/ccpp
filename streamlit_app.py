import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyswarms.single.global_best import GlobalBestPSO

# 1️⃣ Load data from a local CSV file
@st.cache
def load_data():
    # Load your dataset (make sure you have 'data.csv' in your project folder)
    df = pd.read_csv('data.csv')
    df = df[['Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum', 'Total Power']]
    return df

# 2️⃣ Set up feature columns and target column
selected_features = ['Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum']
target_column = 'Total Power'

# 3️⃣ Preprocess Data and Train Models
@st.cache(allow_output_mutation=True)
def preprocess_and_train(df):
    # Separate features and target
    X = df[selected_features]
    y = df[target_column]

    # Scale features
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)
    joblib.dump(feature_scaler, "feature_scaler.pkl")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # StandardScaler for model training
    standard_scaler = StandardScaler()
    X_train_scaled = standard_scaler.fit_transform(X_train)
    X_test_scaled = standard_scaler.transform(X_test)
    joblib.dump(standard_scaler, "standard_scaler.pkl")

    # Define models
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    xgb_model = XGBRegressor(n_estimators=300, max_depth=9, learning_rate=0.2, subsample=0.9, random_state=50, verbosity=0)

    # Train models
    rf_model.fit(X_train_scaled, y_train)
    xgb_model.fit(X_train_scaled, y_train)

    # Save models
    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(xgb_model, "xgb_model.pkl")

    return rf_model, xgb_model, X_test_scaled, y_test, feature_scaler, standard_scaler

# 4️⃣ Ensemble optimization function
def ensemble_predict(features, rf_model, xgb_model, best_w, feature_scaler, standard_scaler):
    # Scale the features using the preloaded scaler
    scaled_input = standard_scaler.transform(feature_scaler.transform(features.reshape(1, -1)))
    
    # Get predictions from both models
    rf_pred = rf_model.predict(scaled_input)
    xgb_pred = xgb_model.predict(scaled_input)
    
    # Weighted ensemble prediction
    return best_w * rf_pred + (1 - best_w) * xgb_pred

# 5️⃣ PSO optimization for ensemble weight
def pso_optimization(X_test_scaled, y_test, rf_model, xgb_model, feature_scaler):
    def objective_function(x):
        preds = []
        for i in range(x.shape[0]):
            input_features = x[i, :-1]  # Last element is the ensemble weight
            w = np.clip(x[i, -1], 0, 1)
            input_scaled = feature_scaler.transform(input_features.reshape(1, -1))
            scaled_input = standard_scaler.transform(input_scaled)
            rf_pred = rf_model.predict(scaled_input)
            xgb_pred = xgb_model.predict(scaled_input)
            ensemble_pred = w * rf_pred + (1 - w) * xgb_pred
            preds.append(-ensemble_pred)
        return np.ravel(preds)

    # Set bounds for PSO
    lb = [0.0] * 4 + [0.0]  # Min bounds for each feature and weight
    ub = [1.0] * 4 + [1.0]  # Max bounds for each feature and weight
    bounds = (lb, ub)

    # Run PSO optimization
    optimizer = GlobalBestPSO(n_particles=50, dimensions=5, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds=bounds)
    cost, pos = optimizer.optimize(objective_function, iters=100)

    # Return optimal weight
    optimal_weight = pos[-1]
    return optimal_weight

# 6️⃣ Streamlit UI
def run_app():
    # Load data and models
    df = load_data()
    rf_model, xgb_model, X_test_scaled, y_test, feature_scaler, standard_scaler = preprocess_and_train(df)

    # Find the best ensemble weight using PSO
    optimal_weight = pso_optimization(X_test_scaled, y_test, rf_model, xgb_model, feature_scaler)
    
    # Display result
    st.title("Power Prediction using Optimized Ensemble of RF & XGBoost")
    st.write(f"Optimized Ensemble Weight: {optimal_weight:.2f} RF / {1 - optimal_weight:.2f} XGB")
    
    # Input for prediction
    st.subheader("Enter Features for Prediction")
    ambient_temperature = st.slider('Ambient Temperature (°C)', 16.0, 37.0, 25.0)
    ambient_relative_humidity = st.slider('Ambient Relative Humidity (%)', 20.0, 90.0, 50.0)
    ambient_pressure = st.slider('Ambient Pressure (hPa)', 797.8, 800.1, 799.0)
    exhaust_vacuum = st.slider('Exhaust Vacuum (mmHg)', 3.0, 12.0, 7.0)

    # Make prediction
    input_features = np.array([ambient_temperature, ambient_relative_humidity, ambient_pressure, exhaust_vacuum])
    predicted_power = ensemble_predict(input_features, rf_model, xgb_model, optimal_weight, feature_scaler, standard_scaler)

    # Display predicted power
    st.write(f"Predicted Total Power (MW): {predicted_power[0]:.4f} MW")

    # Show model performance metrics
    st.subheader("Model Performance Metrics")
    rf_pred = rf_model.predict(X_test_scaled)
    xgb_pred = xgb_model.predict(X_test_scaled)
    rf_r2 = r2_score(y_test, rf_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)

    st.write(f"Random Forest R2: {rf_r2:.4f}")
    st.write(f"XGBoost R2: {xgb_r2:.4f}")

    # Show correlation matrix
    st.subheader("Feature Correlation Matrix")
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    st.pyplot()

# Run the app
if __name__ == "__main__":
    run_app()
