import pyswarms as ps
import streamlit as st
from pyswarms.single.global_best import GlobalBestPSO

def objective_function(x):
    preds = []
    for i in range(x.shape[0]):
        input_features = x[i, :-1]
        w = np.clip(x[i, -1], 0, 1)
        scaled_input = models['scaler'].transform([input_features])
        rf_pred = models['rf_model'].predict(scaled_input)[0]
        xgb_pred = models['xgb_model'].predict(scaled_input)[0]
        ensemble_pred = w * rf_pred + (1 - w) * xgb_pred
        preds.append(-ensemble_pred)  # negative for maximization
    return np.array(preds)

feature_bounds = {
    'Ambient Temperature': [16.0, 38.0],
    'Relative Humidity': [20.0, 90.0],
    'Ambient Pressure': [797.0, 801.0],
    'Exhaust Vacuum': [3.0, 12.0],
}
lb = [b[0] for b in feature_bounds.values()] + [0.0]
ub = [b[1] for b in feature_bounds.values()] + [1.0]
bounds = (lb, ub)

if st.button("🔍 Optimize Inputs for Max Power"):
    optimizer = GlobalBestPSO(
        n_particles=30,
        dimensions=5,
        options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
        bounds=bounds
    )
    cost, pos = optimizer.optimize(objective_function, iters=50)

    best_features = pos[:-1]
    best_weight = pos[-1]
    scaled_input = models['scaler'].transform([best_features])
    rf_pred = models['rf_model'].predict(scaled_input)[0]
    xgb_pred = models['xgb_model'].predict(scaled_input)[0]
    final_pred = best_weight * rf_pred + (1 - best_weight) * xgb_pred

    st.success(f"⚡ Max Predicted Power: {final_pred:.2f} MW")
    st.write("🔧 Optimal Inputs:")
    for name, val in zip(feature_bounds.keys(), best_features):
        st.write(f"- {name}: {val:.2f}")
    st.write(f"⚖️ Optimal Ensemble Weight: RF={best_weight:.2f}, XGB={1 - best_weight:.2f}")
