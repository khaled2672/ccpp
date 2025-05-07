import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pyswarms.single.global_best import GlobalBestPSO
import shap

# Page configuration
st.set_page_config(
    page_title="CCPP Power Optimization",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Combined Cycle Power Plant Power Optimization")
st.write("""
This app predicts and optimizes the electrical energy output of a combined cycle power plant 
using ensemble machine learning models and particle swarm optimization (PSO).
""")

# Load models
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
        return pd.read_csv('data.csv')
    except:
        st.warning("Couldn't load the dataset. Using sample data.")
        return pd.DataFrame({
            'Ambient Temperature': [25.18, 20.86, 10.82, 14.96, 5.11],
            'Ambient Relative Humidity': [59.08, 76.64, 72.88, 73.17, 94.39],
            'Ambient Pressure': [1020.04, 1010.24, 1003.19, 1024.07, 1012.16],
            'Exhaust Vacuum': [62.96, 57.32, 37.50, 41.76, 39.40],
            'Total Power': [444.37, 446.48, 473.90, 463.26, 488.56]
        })

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", 
                        ["Power Prediction", "Model Analysis", "Power Optimization"])

# Power Prediction
if page == "Power Prediction":
    st.header("Power Output Prediction")

    col1, col2 = st.columns(2)
    with col1:
        at = st.slider("Ambient Temperature (°C)", 0.0, 40.0, 25.0)
        rh = st.slider("Relative Humidity (%)", 0.0, 100.0, 70.0)
    with col2:
        ap = st.slider("Ambient Pressure (mbar)", 990.0, 1040.0, 1013.0)
        ev = st.slider("Exhaust Vacuum (cm Hg)", 25.0, 85.0, 50.0)

    if st.button("Predict Power Output"):
        if None in [rf_model, xgb_model, scaler]:
            st.error("Models not loaded. Cannot predict.")
        else:
            input_data = np.array([[at, rh, ap, ev]])
            scaled_input = scaler.transform(input_data)
            rf_pred = rf_model.predict(scaled_input)[0]
            xgb_pred = xgb_model.predict(scaled_input)[0]
            weight = 0.6
            ensemble_pred = weight * rf_pred + (1 - weight) * xgb_pred

            st.success(f"""
            **Predicted Power Output:**
            - Random Forest: {rf_pred:.2f} MW
            - XGBoost: {xgb_pred:.2f} MW
            - Ensemble: {ensemble_pred:.2f} MW
            """)

            st.subheader("Feature Importance")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            pd.Series(rf_model.feature_importances_, ['AT', 'RH', 'AP', 'EV']).plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title("Random Forest")
            pd.Series(xgb_model.feature_importances_, ['AT', 'RH', 'AP', 'EV']).plot(kind='bar', ax=ax2, color='salmon')
            ax2.set_title("XGBoost")
            st.pyplot(fig)

# Model Analysis
elif page == "Model Analysis":
    st.header("Model Performance Analysis")
    if None in [rf_model, xgb_model, scaler]:
        st.error("Models not loaded.")
    else:
        X = df[['Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum']]
        y = df['Total Power']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = scaler.transform(X_test)

        rf_preds = rf_model.predict(X_test_scaled)
        xgb_preds = xgb_model.predict(X_test_scaled)

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

        st.subheader("Model Metrics")
        st.dataframe(pd.DataFrame(results).T.style.format("{:.4f}"))

        st.subheader("Actual vs Predicted")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(y_test, rf_preds, alpha=0.5)
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        ax1.set_title("Random Forest")
        ax2.scatter(y_test, xgb_preds, alpha=0.5)
        ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        ax2.set_title("XGBoost")
        st.pyplot(fig)

        st.subheader("SHAP Summary")
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test_scaled)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_scaled, feature_names=['AT', 'RH', 'AP', 'EV'], show=False)
        st.pyplot(fig)

# Power Optimization
elif page == "Power Optimization":
    st.header("Power Output Optimization with PSO")

    st.subheader("Parameter Bounds")
    cols = st.columns(4)
    with cols[0]:
        at_min = st.number_input("Min Temp (°C)", value=20.0)
        at_max = st.number_input("Max Temp (°C)", value=30.0)
    with cols[1]:
        rh_min = st.number_input("Min Humidity (%)", value=40.0)
        rh_max = st.number_input("Max Humidity (%)", value=70.0)
    with cols[2]:
        ap_min = st.number_input("Min Pressure", value=990.0)
        ap_max = st.number_input("Max Pressure", value=1040.0)
    with cols[3]:
        ev_min = st.number_input("Min Exhaust Vacuum", value=25.0)
        ev_max = st.number_input("Max Exhaust Vacuum", value=85.0)

    st.subheader("PSO Parameters")
    col1, col2 = st.columns(2)
    with col1:
        n_particles = st.slider("Particles", 10, 100, 30)
        max_iter = st.slider("Iterations", 10, 200, 50)
    with col2:
        cognitive = st.slider("Cognitive (c1)", 0.1, 2.0, 0.5)
        social = st.slider("Social (c2)", 0.1, 2.0, 0.3)

    if st.button("Run Optimization"):
        if None in [rf_model, xgb_model, scaler]:
            st.error("Models not loaded.")
        else:
            with st.spinner("Optimizing..."):
                bounds = {
                    'Ambient Temperature': [at_min, at_max],
                    'Ambient Relative Humidity': [rh_min, rh_max],
                    'Ambient Pressure': [ap_min, ap_max],
                    'Exhaust Vacuum': [ev_min, ev_max],
                    'Weight': [0.0, 1.0]
                }

                lb = np.array([v[0] for v in bounds.values()])
                ub = np.array([v[1] for v in bounds.values()])

                def objective(x):
                    features = x[:, :-1]
                    weights = x[:, -1].reshape(-1, 1)
                    scaled = scaler.transform(features)
                    rf = rf_model.predict(scaled).reshape(-1, 1)
                    xgb = xgb_model.predict(scaled).reshape(-1, 1)
                    ensemble = weights * rf + (1 - weights) * xgb
                    return -ensemble.flatten()

                optimizer = GlobalBestPSO(
                    n_particles=n_particles,
                    dimensions=len(lb),
                    options={'c1': cognitive, 'c2': social, 'w': 0.9},
                    bounds=(lb, ub)
                )

                cost, pos = optimizer.optimize(objective, iters=max_iter)
                opt_features = pos[:-1]
                opt_weight = pos[-1]
                scaled_input = scaler.transform(opt_features.reshape(1, -1))
                rf_out = rf_model.predict(scaled_input)[0]
                xgb_out = xgb_model.predict(scaled_input)[0]
                ensemble_out = opt_weight * rf_out + (1 - opt_weight) * xgb_out

                st.success("Optimization Complete!")
                st.write("### Optimal Inputs and Output")
                st.table(pd.DataFrame({
                    "Parameter": ['Ambient Temp', 'Humidity', 'Pressure', 'Exhaust Vac', 'Weight'],
                    "Value": list(opt_features) + [opt_weight],
                    "Units": ['°C', '%', 'mbar', 'cm Hg', '']
                }))

                st.write(f"""
                - RF Output: {rf_out:.2f} MW  
                - XGB Output: {xgb_out:.2f} MW  
                - Ensemble Output: {ensemble_out:.2f} MW
                """)

                st.subheader("Convergence Plot")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(optimizer.cost_history)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Cost (-Power)")
                st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("*This is a demo app for optimizing Combined Cycle Power Plant output.*")
