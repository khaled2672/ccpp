import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pyswarms.single.global_best import GlobalBestPSO
import shap

# Page configuration
st.set_page_config(
    page_title="CCPP Power Optimization",
    page_icon="⚡",
    layout="wide"
)

# Title and description
st.title("⚡ Combined Cycle Power Plant Power Optimization")
st.write("""
This app predicts and optimizes the electrical energy output of a combined cycle power plant 
using ensemble machine learning models and particle swarm optimization (PSO).
""")

# Load models and scaler
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('rf_model.joblib')
        xgb_model = joblib.load('xgb_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return rf_model, xgb_model, scaler
    except:
        st.error("Model files not found! Please ensure you have the trained models.")
        return None, None, None

rf_model, xgb_model, scaler = load_models()

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data.csv')  # Update with your actual data path
        return df
    except:
        st.warning("Couldn't load the dataset. Using sample data.")
        # Sample data if loading fails
        data = {
            'Ambient Temperature': [25.18, 20.86, 10.82, 14.96, 5.11],
            'Ambient Relative Humidity': [59.08, 76.64, 72.88, 73.17, 94.39],
            'Ambient Pressure': [1020.04, 1010.24, 1003.19, 1024.07, 1012.16],
            'Exhaust Vacuum': [62.96, 57.32, 37.50, 41.76, 39.40],
            'Total Power': [444.37, 446.48, 473.90, 463.26, 488.56]
        }
        return pd.DataFrame(data)

df = load_data()

# Feature bounds for PSO
feature_bounds = {
    'Ambient Temperature': [20.0, 30.0],
    'Ambient Relative Humidity': [40.0, 70.0],
    'Ambient Pressure': [799.0, 800.0],
    'Exhaust Vacuum': [4.5, 6.0]
}

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", 
                          ["Power Prediction", "Model Analysis", "Power Optimization"])

# Power Prediction Page
if options == "Power Prediction":
    st.header("Power Output Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        at = st.slider("Ambient Temperature (°C)", 
                      min_value=0.0, max_value=40.0, value=25.0, step=0.1)
        rh = st.slider("Relative Humidity (%)", 
                      min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    
    with col2:
        st.subheader("")
        ap = st.slider("Ambient Pressure (mbar)", 
                      min_value=990.0, max_value=1040.0, value=1013.0, step=0.1)
        ev = st.slider("Exhaust Vacuum (cm Hg)", 
                      min_value=25.0, max_value=85.0, value=50.0, step=0.1)
    
    if st.button("Predict Power Output"):
        if rf_model is None or xgb_model is None:
            st.error("Models not loaded. Cannot make predictions.")
        else:
            # Prepare input
            input_data = np.array([[at, rh, ap, ev]])
            scaled_input = scaler.transform(input_data)
            
            # Get predictions
            rf_pred = rf_model.predict(scaled_input)[0]
            xgb_pred = xgb_model.predict(scaled_input)[0]
            
            # Use optimal ensemble weight (could be loaded from file)
            ensemble_weight = 0.6  # Default - should be loaded from your optimization
            ensemble_pred = ensemble_weight * rf_pred + (1 - ensemble_weight) * xgb_pred
            
            # Display results
            st.success(f"""
            **Predicted Electrical Power Output:**
            - Random Forest: {rf_pred:.2f} MW
            - XGBoost: {xgb_pred:.2f} MW
            - Ensemble: {ensemble_pred:.2f} MW
            """)
            
            # Show feature importance
            st.subheader("Feature Importance")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            pd.Series(rf_model.feature_importances_, 
                     index=['AT', 'RH', 'AP', 'EV']).plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title("Random Forest Feature Importance")
            
            pd.Series(xgb_model.feature_importances_, 
                     index=['AT', 'RH', 'AP', 'EV']).plot(kind='bar', ax=ax2, color='salmon')
            ax2.set_title("XGBoost Feature Importance")
            
            st.pyplot(fig)

# Model Analysis Page
elif options == "Model Analysis":
    st.header("Model Performance Analysis")
    
    if rf_model is None or xgb_model is None:
        st.error("Models not loaded. Cannot show analysis.")
    else:
        # Generate predictions for the test set (in a real app, you'd load these)
        X = df[['Ambient Temperature', 'Ambient Relative Humidity', 
               'Ambient Pressure', 'Exhaust Vacuum']]
        y = df['Total Power']
        
        # Split data (for demo purposes - in practice use your actual test set)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        
        # Get predictions
        rf_preds = rf_model.predict(X_test_scaled)
        xgb_preds = xgb_model.predict(X_test_scaled)
        
        # Calculate metrics
        def get_metrics(y_true, y_pred):
            return {
                "R2": r2_score(y_true, y_pred),
                "MSE": mean_squared_error(y_true, y_pred),
                "MAE": mean_absolute_error(y_true, y_pred),
                "MBE": np.mean(y_true - y_pred)
            }
        
        results = {
            "Random Forest": get_metrics(y_test, rf_preds),
            "XGBoost": get_metrics(y_test, xgb_preds)
        }
        
        # Display metrics
        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame(results).T
        st.dataframe(metrics_df.style.format("{:.4f}"))
        
        # Plot actual vs predicted
        st.subheader("Actual vs Predicted Values")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.scatter(y_test, rf_preds, alpha=0.5)
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        ax1.set_xlabel("Actual Power (MW)")
        ax1.set_ylabel("Predicted Power (MW)")
        ax1.set_title("Random Forest")
        
        ax2.scatter(y_test, xgb_preds, alpha=0.5)
        ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        ax2.set_xlabel("Actual Power (MW)")
        ax2.set_ylabel("Predicted Power (MW)")
        ax2.set_title("XGBoost")
        
        st.pyplot(fig)
        
        # SHAP analysis
        st.subheader("SHAP Feature Importance")
        
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_scaled, 
                         feature_names=['AT', 'RH', 'AP', 'EV'], 
                         plot_type="bar", show=False)
        st.pyplot(fig)

# Power Optimization Page
elif options == "Power Optimization":
    st.header("Power Output Optimization")
    st.write("""
    Use Particle Swarm Optimization (PSO) to find the optimal operating conditions
    that maximize power output within specified constraints.
    """)
    
    # Let user adjust bounds
    st.subheader("Parameter Constraints")
    
    cols = st.columns(4)
    with cols[0]:
        at_min = st.number_input("Min Temperature (°C)", value=20.0)
        at_max = st.number_input("Max Temperature (°C)", value=30.0)
    with cols[1]:
        rh_min = st.number_input("Min Humidity (%)", value=40.0)
        rh_max = st.number_input("Max Humidity (%)", value=70.0)
    with cols[2]:
        ap_min = st.number_input("Min Pressure (mbar)", value=990.0)
        ap_max = st.number_input("Max Pressure (mbar)", value=1040.0)
    with cols[3]:
        ev_min = st.number_input("Min Exhaust Vacuum (cm Hg)", value=25.0)
        ev_max = st.number_input("Max Exhaust Vacuum (cm Hg)", value=85.0)
    
    # PSO parameters
    st.subheader("PSO Parameters")
    col1, col2 = st.columns(2)
    with col1:
        n_particles = st.slider("Number of particles", 10, 100, 30)
        max_iter = st.slider("Maximum iterations", 10, 200, 50)
    with col2:
        cognitive = st.slider("Cognitive parameter (c1)", 0.1, 2.0, 0.5)
        social = st.slider("Social parameter (c2)", 0.1, 2.0, 0.3)
    
    if st.button("Run Optimization"):
        if rf_model is None or xgb_model is None:
            st.error("Models not loaded. Cannot run optimization.")
        else:
            with st.spinner("Running PSO optimization..."):
                # Define bounds
                bounds_dict = {
                    'Ambient Temperature': [at_min, at_max],
                    'Ambient Relative Humidity': [rh_min, rh_max],
                    'Ambient Pressure': [ap_min, ap_max],
                    'Exhaust Vacuum': [ev_min, ev_max],
                    'Weight': [0.0, 1.0]
                }
                
                # Convert to arrays for PSO
                lb = np.array([v[0] for v in bounds_dict.values()])
                ub = np.array([v[1] for v in bounds_dict.values()])
                
                # Define objective function
                def objective_function(x):
                    features = x[:, :-1]
                    weights = x[:, -1].reshape(-1, 1)
                    
                    scaled_features = scaler.transform(features)
                    rf_pred = rf_model.predict(scaled_features).reshape(-1, 1)
                    xgb_pred = xgb_model.predict(scaled_features).reshape(-1, 1)
                    
                    ensemble_pred = weights * rf_pred + (1 - weights) * xgb_pred
                    return -ensemble_pred.flatten()  # Negative for maximization
                
                # Run PSO
                optimizer = GlobalBestPSO(
                    n_particles=n_particles,
                    dimensions=len(lb),
                    options={'c1': cognitive, 'c2': social, 'w': 0.9},
                    bounds=(lb, ub)
                
                cost, pos = optimizer.optimize(objective_function, iters=max_iter)
                
                # Get results
                optimal_features = pos[:-1]
                optimal_weight = pos[-1]
                
                scaled_input = scaler.transform(optimal_features.reshape(1, -1))
                rf_power = rf_model.predict(scaled_input)[0]
                xgb_power = xgb_model.predict(scaled_input)[0]
                optimal_power = optimal_weight * rf_power + (1 - optimal_weight) * xgb_power
                
                # Display results
                st.success("Optimization Complete!")
                st.subheader("Optimal Parameters")
                
                result_data = {
                    "Parameter": ['Ambient Temperature', 'Relative Humidity', 
                                'Ambient Pressure', 'Exhaust Vacuum', 'Ensemble Weight'],
                    "Optimal Value": [optimal_features[0], optimal_features[1],
                                    optimal_features[2], optimal_features[3], optimal_weight],
                    "Units": ['°C', '%', 'mbar', 'cm Hg', '']
                }
                
                st.table(pd.DataFrame(result_data))
                
                st.subheader("Predicted Power Output")
                st.write(f"""
                - Random Forest prediction: {rf_power:.2f} MW
                - XGBoost prediction: {xgb_power:.2f} MW
                - Ensemble prediction: {optimal_power:.2f} MW
                """)
                
                # Plot convergence
                st.subheader("PSO Convergence")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(optimizer.cost_history)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Best Cost (-Power)")
                ax.set_title("PSO Convergence History")
                st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
*Note: This is a demo application for power output prediction and optimization of a Combined Cycle Power Plant.*
""")
