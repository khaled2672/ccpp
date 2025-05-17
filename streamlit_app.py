import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import StringIO

# Theme configuration

def set\_theme(dark):
plt.style.use('dark\_background' if dark else 'default')
\# (theme CSS remains unchanged — no changes needed here)
\# ...

# Cache model loading

@st.cache\_resource
def load\_models():
try:
return (
joblib.load('rf\_model.joblib'),
joblib.load('xgb\_model.joblib'),
joblib.load('scaler.joblib')
)
except Exception as e:
st.error(f"Error loading models: {str(e)}")
st.stop()

# Column mapping

def map\_columns(df):
column\_mapping = {
"Ambient Temperature (°C)": \["Ambient Temperature", "Temperature", "Temp", "Amb Temp", "Ambient\_Temperature", "AT"],
"Ambient Relative Humidity (%)": \["Relative Humidity", "Ambient Relative Humidity", "Humidity", "Rel Humidity", "Humidity (%)", "RH"],
"Ambient Pressure (mbar)": \["Ambient Pressure", "Pressure", "Amb Pressure", "Pressure (mbar)", "AP"],
"Exhaust Vacuum (cmHg)": \["Exhaust Vacuum", "Vacuum", "Exhaust Vac", "Vacuum (cmHg)", "EV"]
}
mapped\_columns = {}
for target, aliases in column\_mapping.items():
for name in aliases:
if name in df.columns:
mapped\_columns\[target] = name
break
return mapped\_columns

@st.cache\_data
def generate\_example\_csv():
example\_data = {
"Temperature (°C)": \[25.0, 30.0, 27.5],
"Humidity (%)": \[60.0, 65.0, 62.5],
"Pressure (mbar)": \[1010.0, 1005.0, 1007.5],
"Vacuum (cmHg)": \[5.0, 6.0, 5.5]
}
return pd.DataFrame(example\_data).to\_csv(index=False)

if 'dark\_mode' not in st.session\_state:
st.session\_state.dark\_mode = False

# ========== SIDEBAR ==========

with st.sidebar:
st.title("⚙️ CCPP Power Predictor")
st.session\_state.dark\_mode = st.toggle("🌙 Dark Mode", value=st.session\_state.dark\_mode)
set\_theme(st.session\_state.dark\_mode)

```
st.subheader("How to Use")
st.markdown("""
1. Adjust sliders to set plant conditions  
2. View the predicted power output  
3. Upload CSV for batch predictions  
4. Best weight used: 65% RF + 35% XGB
""")

with st.spinner("Loading models..."):
    rf_model, xgb_model, scaler = load_models()

feature_bounds = {
    'Ambient Temperature': [0.0, 50.0],
    'Ambient Relative Humidity': [10.0, 100.0],
    'Ambient Pressure': [799.0, 1035.0],
    'Exhaust Vacuum': [3.0, 12.0],
}

st.subheader("Input Parameters")
inputs = {}
for feature, (low, high) in feature_bounds.items():
    default = (low + high) / 2
    inputs[feature] = st.slider(feature, low, high, default)

if st.button("🔄 Reset to Defaults"):
    for feature in inputs:
        inputs[feature] = (feature_bounds[feature][0] + feature_bounds[feature][1]) / 2
```

# ========== MAIN CONTENT ==========

st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output using ambient conditions with an ensemble of Random Forest & XGBoost models.")

# Set best static weights

rf\_weight = 0.65
xgb\_weight = 0.35

# Prepare input for prediction

feature\_names = list(feature\_bounds.keys())
input\_features = np.array(\[inputs\[f] for f in feature\_names]).reshape(1, -1)

with st.spinner("Making predictions..."):
try:
scaled\_features = scaler.transform(input\_features)
rf\_pred = rf\_model.predict(scaled\_features)\[0]
xgb\_pred = xgb\_model.predict(scaled\_features)\[0]
ensemble\_pred = rf\_weight \* rf\_pred + xgb\_weight \* xgb\_pred
except Exception as e:
st.error(f"Prediction error: {str(e)}")
st.stop()

st.subheader("🔢 Model Predictions")
col1, col2, col3 = st.columns(3)
with col1:
st.metric("Random Forest", f"{rf\_pred:.2f} MW")
with col2:
st.metric("XGBoost", f"{xgb\_pred:.2f} MW")
with col3:
st.metric(f"Ensemble", f"{ensemble\_pred:.2f} MW")

# ========== BATCH PREDICTION ==========

st.subheader("📂 Batch Prediction")
st.markdown("Upload a CSV file with multiple records to get predictions for all of them at once.")

st.download\_button(
"⬇️ Download Example CSV",
data=generate\_example\_csv(),
file\_name="ccpp\_example\_input.csv",
mime="text/csv"
)

uploaded\_file = st.file\_uploader("Upload your input data (CSV format)", type=\["csv"])

if uploaded\_file is not None:
try:
df = pd.read\_csv(uploaded\_file)
if df.empty:
st.error("Uploaded file is empty")
st.stop()

```
    st.success("File uploaded successfully!")
    with st.expander("View uploaded data"):
        st.dataframe(df.head())

    mapped_columns = map_columns(df)
    if len(mapped_columns) < 4:
        missing_cols = [col for col in feature_names if col not in mapped_columns]
        st.error(f"Missing or unmatched columns: {', '.join(missing_cols)}")
        st.stop()

    df_processed = df.rename(columns=mapped_columns)
    missing_cols = [col for col in feature_names if col not in df_processed.columns]
    if missing_cols:
        st.error(f"Missing columns after mapping: {', '.join(missing_cols)}")
        st.stop()

    with st.spinner("Processing data..."):
        features = df_processed[feature_names]
        scaled = scaler.transform(features)
        rf_preds = rf_model.predict(scaled)
        xgb_preds = xgb_model.predict(scaled)
        final_preds = rf_weight * rf_preds + xgb_weight * xgb_preds

        results = df_processed.copy()
        results['RF_Prediction (MW)'] = rf_preds
        results['XGB_Prediction (MW)'] = xgb_preds
        results['Ensemble_Prediction (MW)'] = final_preds

        st.success("Predictions completed!")

        def color_positive_green(val):
            return 'color: green' if val > final_preds.mean() else 'color: red'

        st.dataframe(results.style.format({
            'RF_Prediction (MW)': '{:.2f}',
            'XGB_Prediction (MW)': '{:.2f}',
            'Ensemble_Prediction (MW)': '{:.2f}'
        }).applymap(color_positive_green, subset=['Ensemble_Prediction (MW)']))

        st.download_button(
            "⬇️ Download Full Results",
            data=results.to_csv(index=False).encode(),
            file_name="ccpp_predictions.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"Error processing file: {str(e)}")
```

# Footer

st.markdown("---")
st.caption("Developed with Streamlit | Ensemble Model: Random Forest (65%) + XGBoost (35%)")
