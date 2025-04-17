import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyswarms.single.global_best import GlobalBestPSO

# ------------------------------------------
# 1️⃣ Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')  # Make sure data.csv is in the same folder
    df = df[['Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum', 'Total Power']]
    return df

# ------------------------------------------
# 2️⃣ Preprocess and Train Once
@st.cache_resource
def preprocess_and_train(df):
    selected_features = ['Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Exhaust Vacuum']
    target_column = 'Total Power'

    X = df[selected_features]
    y = df[target_column]

    # Scale features using MinMaxScaler for PSO input
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)

    # Then standard scale for model training
    standard_scaler = StandardScaler()
    X_scaled_std = standard_scaler.fit_transform(X_scaled)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_std, y, test_size=0.2, random_state=42)

    # Train models
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    xgb = XGBRegressor(n_estimators=300, max_depth=9, learning_rate=0.2, subsample=0.9, random_state=50, verbosity=0)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    return rf, xgb, feature_scaler, standard_scaler, X_test, y_test

# ------------------------------------------
# 3️⃣ Predict using ensemble
def ensemble_predict(features, rf, xgb, weight, feature_scaler, standard_scaler):
    features_scaled = feature_scaler.transform(features.reshape(1, -1))
    features_std = standard_scaler.transform(features_scaled)
    rf_pred = rf.predict(features_std)
    xgb_pred = xgb.predict(features_std)
    return weight * rf_pred + (1 - weight) * xgb_pred

# ------------------------------------------
# 4️⃣ Optimize blending weight using PSO
def optimize_weight(rf, xgb, X_test, y_test):
    def objective_function(x):
        preds = []
        for i in range(x.shape[0]):
            w = np.clip(x[i][0], 0, 1)
            blended = w * rf.predict(X_test) + (1 - w) * xgb.predict(X_test)
            loss = mean_squared_error(y_test, blended)
            preds.append(loss)
        return np.array(preds)

    bounds = ([0.0], [1.0])
    optimizer = GlobalBestPSO(n_particles=20, dimensions=1, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds=bounds)
    best_loss, pos = optimizer.optimize(objective_function, iters=50, verbose=False)
    return pos[0]

# ------------------------------------------
# 5️⃣ Streamlit App UI
def main():
    st.title("⚡ Power Prediction with RF & XGBoost Ensemble")

    df = load_data()
    rf, xgb, feature_scaler, standard_scaler, X_test, y_test = preprocess_and_train(df)
    best_weight = optimize_weight(rf, xgb, X_test, y_test)

    st.markdown(f"### ✅ Optimal Ensemble Weight: {best_weight:.2f} (RF) / {1 - best_weight:.2f} (XGB)")

    # Inputs
    st.subheader("🔧 Input Features")
    temp = st.slider("Ambient Temperature (°C)", 16.0, 37.0, 25.0)
    humidity = st.slider("Ambient Relative Humidity (%)", 20.0, 90.0, 50.0)
    pressure = st.slider("Ambient Pressure (hPa)", 797.8, 800.1, 799.0)
    vacuum = st.slider("Exhaust Vacuum (mmHg)", 3.0, 12.0, 6.5)

    input_array = np.array([temp, humidity, pressure, vacuum])
    prediction = ensemble_predict(input_array, rf, xgb, best_weight, feature_scaler, standard_scaler)

    st.success(f"⚡ Predicted Total Power: {prediction[0]:.4f} MW")

    # Correlation heatmap
    st.subheader("📊 Feature Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# ------------------------------------------
if __name__ == "__main__":
    main()
