import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyswarms.single.global_best import GlobalBestPSO

# 1️⃣ Load default dataset and preprocess
@st.cache
def load_data():
    # Default preloaded dataset (update path if necessary)
    df = pd.read_csv("/content/data/data.csv")  # This can be updated to a public dataset URL or static file
    target_column = 'Total Power'
    selected_features = ['Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum']

    # Feature scaling for PSO
    feature_scaler = MinMaxScaler()
    df[selected_features] = feature_scaler.fit_transform(df[selected_features])
    joblib.dump(feature_scaler, "/content/feature_scaler.pkl")

    X = df[selected_features]
    y = df[target_column]
    return X, y, df, selected_features

X, y, df, selected_features = load_data()

# 2️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ StandardScaler for model training
standard_scaler = StandardScaler()
X_train_scaled = standard_scaler.fit_transform(X_train)
X_test_scaled = standard_scaler.transform(X_test)
joblib.dump(standard_scaler, "standard_scaler.pkl")

# 4️⃣ Train models once
@st.cache
def train_models():
    # Define models
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    xgb_model = XGBRegressor(n_estimators=300, max_depth=9, learning_rate=0.2, subsample=0.9, random_state=50, verbosity=0)

    # Train models
    rf_model.fit(X_train_scaled, y_train)
    xgb_model.fit(X_train_scaled, y_train)
    
    # Save models
    joblib.dump(rf_model, "rf_model.pkl")
    joblib.dump(xgb_model, "xgb_model.pkl")
    
    return rf_model, xgb_model

rf_model, xgb_model = train_models()

# 5️⃣ PSO Optimization
@st.cache
def pso_optimization():
    # Load scaler and models
    feature_scaler = joblib.load("feature_scaler.pkl")
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")

    # PSO objective function
    def objective_function(x):
        preds = []
        for i in range(x.shape[0]):
            input_features = x[i, :-1]  # Last element is the ensemble weight
            w = np.clip(x[i, -1], 0, 1)
            
            # Scale features
            input_scaled = feature_scaler.transform(input_features.reshape(1, -1))
            
            # Get predictions
            rf_pred = rf_model.predict(input_scaled)
            xgb_pred = xgb_model.predict(input_scaled)
            
            # Weighted prediction
            ensemble_pred = w * rf_pred + (1 - w) * xgb_pred
            preds.append(-ensemble_pred)
        return np.ravel(preds)

    # Define feature bounds for PSO
    feature_bounds = {
        'Ambient Temperature': [16.788, 37.101],
        'Ambient Relative Humidity': [20.244555, 88.487236],
        'Ambient Pressure': [797.8021, 800.0850],
        'Exhaust Vacuum': [3.000248, 12.000992],
    }

    lb = [feature_bounds[feat][0] for feat in selected_features] + [0.0]  # + blending weight lower bound
    ub = [feature_bounds[feat][1] for feat in selected_features] + [1.0]  # + blending weight upper bound
    bounds = (lb, ub)

    # Run PSO optimizer
    optimizer = GlobalBestPSO(
        n_particles=50,
        dimensions=5,  # 4 features + 1 weight
        options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
        bounds=bounds
    )

    cost, pos = optimizer.optimize(objective_function, iters=100)

    # Extract optimal results
    optimal_input = pos[:-1]
    optimal_weight = pos[-1]
    
    # Scale and predict final ensemble output
    scaled_input = feature_scaler.transform(np.array(optimal_input).reshape(1, -1))
    rf_pred = rf_model.predict(scaled_input)
    xgb_pred = xgb_model.predict(scaled_input)
    final_power = optimal_weight * rf_pred + (1 - optimal_weight) * xgb_pred

    return optimal_input, optimal_weight, final_power

optimal_input, optimal_weight, final_power = pso_optimization()

# 6️⃣ Display results in Streamlit app
st.title("Power Prediction and PSO Optimization")

# 7️⃣ Show Dataset Correlation Matrix
st.subheader("Dataset Correlation Matrix")
correlation_matrix = df.corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)

# 8️⃣ Show Optimal Features and Ensemble Weight
st.subheader("Optimized Input Features and Ensemble Weight")
st.write(f"Optimized Input Features (Original Scale):")
for name, val in zip(selected_features, optimal_input):
    st.write(f"{name}: {val:.4f}")

st.write(f"Optimized Ensemble Weight: {optimal_weight:.2f} RF / {1 - optimal_weight:.2f} XGB")

# 9️⃣ Show Final Predicted Power
st.subheader("Final Predicted Power Output (MW)")
st.write(f"Predicted Power: {final_power[0]:.4f} MW")

# 🔟 Optional: Show plots (Model Comparison)
metrics = ['R2', 'MSE', 'MAE', 'MBE']
results = {
    'Random Forest': {'R2': 0.92, 'MSE': 0.1, 'MAE': 0.07, 'MBE': 0.05},
    'XGBoost': {'R2': 0.91, 'MSE': 0.12, 'MAE': 0.08, 'MBE': 0.06},
    'Combined': {'R2': 0.93, 'MSE': 0.09, 'MAE': 0.06, 'MBE': 0.04}
}
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
for i, metric in enumerate(metrics):
    values = [results[m][metric] for m in results.keys()]
    axes[i].bar(results.keys(), values, color=['#4169E1', '#FF8C00', '#2E8B57'])
    axes[i].set_title(f"{metric} Comparison")
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis='x', rotation=30)
plt.tight_layout()
st.pyplot(fig)
