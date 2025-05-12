import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import StringIO

# Theme configuration with background images

def set\_theme(dark):
plt.style.use('dark\_background' if dark else 'default')
if dark:
st.markdown(
""" <style>
.stApp {
background-image: url("[https://img.freepik.com/free-photo/view-nuclear-power-plant-with-towers-letting-out-steam-from-process\_23-2150957658.jpg?t=st=1746689462\~exp=1746693062\~hmac=71da5c1edb4e4c2bd79eda912f889934c4d11e1aeea35a5106d1bd18e53a89b4\&w=1380](https://img.freepik.com/free-photo/view-nuclear-power-plant-with-towers-letting-out-steam-from-process_23-2150957658.jpg?t=st=1746689462~exp=1746693062~hmac=71da5c1edb4e4c2bd79eda912f889934c4d11e1aeea35a5106d1bd18e53a89b4&w=1380)");
background-size: cover;
background-attachment: fixed;
background-position: center;
color: #f1f1f1;
}
/\* Dark overlay for better readability */
.stApp\:before {
content: "";
position: absolute;
top: 0;
left: 0;
right: 0;
bottom: 0;
background-color: rgba(0, 0, 0, 0.75);
z-index: -1;
}
/* Main content area */
.main .block-container {
background-color: rgba(0, 0, 0, 0.7);
padding: 2rem;
border-radius: 10px;
backdrop-filter: blur(4px);
}
/* Sidebar */
\[data-testid="stSidebar"] > div\:first-child {
background-color: rgba(0, 0, 0, 0.8) !important;
color: #ffffff ;
backdrop-filter: blur(4px);
}
/* Text colors */
.css-1d391kg, .css-1cpxqw2, .st-b7, .st-b8, .st-b9 {
color: #f1f1f1 !important;
}
/* Widget styling */
.st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj {
background-color: rgba(30, 30, 30, 0.7) !important;
}
/* Button styling */
.stDownloadButton, .stButton>button {
background-color: #4a8af4 !important;
color: black !important;
border: white !important;
}
.stDownloadButton\:hover, .stButton>button\:hover {
background-color: #f5f6f7 !important;
} </style>
""",
unsafe\_allow\_html=True
)
else:
st.markdown(
""" <style>
.stApp {
background-image: url("[https://img.freepik.com/free-photo/view-nuclear-power-plant-with-towers-letting-out-steam-from-process\_23-2150957658.jpg?t=st=1746689462\~exp=1746693062\~hmac=71da5c1edb4e4c2bd79eda912f889934c4d11e1aeea35a5106d1bd18e53a89b4\&w=1380](https://img.freepik.com/free-photo/view-nuclear-power-plant-with-towers-letting-out-steam-from-process_23-2150957658.jpg?t=st=1746689462~exp=1746693062~hmac=71da5c1edb4e4c2bd79eda912f889934c4d11e1aeea35a5106d1bd18e53a89b4&w=1380)");
background-size: cover;
background-attachment: fixed;
background-position: center;
color: #333333;
}
/* Light overlay for better readability */
.stApp\:before {
content: "";
position: absolute;
top: 0;
left: 0;
right: 0;
bottom: 0;
background-color: rgba(255, 255, 255, 0.75);
z-index: -1;
}
/* Main content area */
.main .block-container {
background-color: rgba(255, 255, 255, 0.8);
padding: 2rem;
border-radius: 10px;
backdrop-filter: blur(4px);
}
/* Sidebar */
\[data-testid="stSidebar"] > div\:first-child {
background-color: rgba(255, 255, 255, 0.85) !important
backdrop-filter: blur(4px);
}
/* Text colors */
.css-1d391kg, .css-1cpxqw2, .st-b7, .st-b8, .st-b9 {
color: #ffffff !important;
}
/* Widget styling */
.st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj {
background-color: rgba(240, 240, 240, 0.8) !important;
}
/* Button styling \*/
.stDownloadButton, .stButton>button {
background-color: #4a8af4 !important;
color: white !important;
border: none !important;
}
.stDownloadButton\:hover, .stButton>button\:hover {
background-color: #3a7ae4 !important;
} </style>
""",
unsafe\_allow\_html=True
)

# Cache resources for better performance

@st.cache\_resource
def load\_models():
"""Load models and scaler with caching"""
try:
return (
joblib.load('rf\_model.joblib'),
joblib.load('xgb\_model.joblib'),
joblib.load('scaler.joblib')
)
except Exception as e:
st.error(f"Error loading models: {str(e)}")
st.stop()

# Column mapping function

def map\_columns(df):
"""Map user-uploaded CSV columns to the required features."""
column\_mapping = {
"Ambient Temperature (°C)": \["Ambient Temperature", "Temperature", "Temp", "Amb Temp", "Ambient\_Temperature", "AT"],
"Ambient Relative Humidity (%)": \["Relative Humidity", "Ambient Relative Humidity", "Humidity", "Rel Humidity", "Humidity (%)", "RH"],
"Ambient Pressure (mbar)": \["Ambient Pressure", "Pressure", "Amb Pressure", "Pressure (mbar)", "AP"],
"Exhaust Vacuum (cmHg)": \["Exhaust Vacuum", "Vacuum", "Exhaust Vac", "Vacuum (cmHg)", "EV"]
}

```
mapped_columns = {}
for target, possible_names in column_mapping.items():
    for name in possible_names:
        if name in df.columns:
            mapped_columns[target] = name
            break

return mapped_columns
```

# Generate example CSV data

@st.cache\_data
def generate\_example\_csv():
"""Generate example CSV data for download"""
example\_data = {
"Temperature (°C)": \[25.0, 30.0, 27.5],
"Humidity (%)": \[60.0, 65.0, 62.5],
"Pressure (mbar)": \[1010.0, 1005.0, 1007.5],
"Vacuum (cmHg)": \[5.0, 6.0, 5.5]
}
return pd.DataFrame(example\_data).to\_csv(index=False)

# Initialize session state for theme persistence

if 'dark\_mode' not in st.session\_state:
st.session\_state.dark\_mode = False

# ========== SIDEBAR ==========

with st.sidebar:
st.title("⚙️ CCPP Power Predictor")

```
# Dark mode toggle
st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
set_theme(st.session_state.dark_mode)

st.subheader("How to Use")
st.markdown("""
1. Adjust sliders to set plant conditions  
2. View the predicted power output  
3. Compare models using the toggle  
4. Upload CSV for batch predictions
""")

# Load models
with st.spinner("Loading models..."):
    rf_model, xgb_model, scaler = load_models()

# Feature bounds for UI
feature_bounds = {
    'Ambient Temperature': [0.0, 50.0],
    'Ambient Relative Humidity': [10.0, 100.0],
    'Ambient Pressure': [799.0, 1035.0],
    'Exhaust Vacuum': [3.0, 12.0],
    'Model Weight (RF vs XGB)': [0.0, 1.0]
}

# Input sliders
st.subheader("Input Parameters")
inputs = {}
for feature, (low, high) in feature_bounds.items():
    default = (low + high) / 2
    inputs[feature] = st.slider(
        feature, low, high, default,
        help=f"Adjust {feature} between {low} and {high}"
    )

# Reset button
if st.button("🔄 Reset to Defaults"):
    for feature in inputs:
        inputs[feature] = (feature_bounds[feature][0] + feature_bounds[feature][1]) / 2
```

# ========== MAIN CONTENT ==========

st.title("🔋 Combined Cycle Power Plant Predictor")
st.markdown("Predict power output using ambient conditions with an ensemble of Random Forest & XGBoost models.")

# Prepare input for prediction

feature\_names = list(feature\_bounds.keys())\[:-1]  # Exclude weight
input\_features = np.array(\[inputs\[f] for f in feature\_names]).reshape(1, -1)
input\_weight = inputs\['Model Weight (RF vs XGB)']

# Make predictions

with st.spinner("Making predictions..."):
try:
scaled\_features = scaler.transform(input\_features)
rf\_pred = rf\_model.predict(scaled\_features)\[0]
xgb\_pred = xgb\_model.predict(scaled\_features)\[0]
ensemble\_pred = input\_weight \* rf\_pred + (1 - input\_weight) \* xgb\_pred
except Exception as e:
st.error(f"Prediction error: {str(e)}")
st.stop()

# Display results in cards

st.subheader("🔢 Model Predictions")
col1, col2, col3 = st.columns(3)
with col1:
st.markdown(
f""" <div style="
      background-color: {'rgba(30, 30, 30, 0.7)' if st.session_state.dark_mode else 'rgba(240, 240, 240, 0.8)'};
      padding: 1.5rem;
      border-radius: 10px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.1);
      text-align: center;
  "> <h3 style="margin-top: 0;">Random Forest</h3> <h2 style="color: {'#4a8af4' if st.session_state.dark_mode else '#2a6fdb'};">{rf\_pred:.2f} MW</h2> </div>
""",
unsafe\_allow\_html=True
)
with col2:
st.markdown(
f""" <div style="
      background-color: {'rgba(30, 30, 30, 0.7)' if st.session_state.dark_mode else 'rgba(240, 240, 240, 0.8)'};
      padding: 1.5rem;
      border-radius: 10px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.1);
      text-align: center;
  "> <h3 style="margin-top: 0;">XGBoost</h3> <h2 style="color: {'#4a8af4' if st.session_state.dark_mode else '#2a6fdb'};">{xgb\_pred:.2f} MW</h2> </div>
""",
unsafe\_allow\_html=True
)
with col3:
st.markdown(
f""" <div style="
      background-color: {'rgba(30, 30, 30, 0.7)' if st.session_state.dark_mode else 'rgba(240, 240, 240, 0.8)'};
      padding: 1.5rem;
      border-radius: 10px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.1);
      text-align: center;
  "> <h3 style="margin-top: 0;">Ensemble (Weight: {input\_weight:.2f})</h3> <h2 style="color: {'#4a8af4' if st.session_state.dark_mode else '#2a6fdb'};">{ensemble\_pred:.2f} MW</h2> <p style="margin-bottom: 0; font-size: 0.9rem;">{(ensemble\_pred - (rf\_pred + xgb\_pred)/2):.2f} vs avg</p> </div>
""",
unsafe\_allow\_html=True
)

# Batch Prediction with CSV Upload

st.subheader("📂 Batch Prediction")
st.markdown("Upload a CSV file with multiple records to get predictions for all of them at once.")

# Example CSV download

st.download\_button(
"⬇️ Download Example CSV",
data=generate\_example\_csv(),
file\_name="ccpp\_example\_input.csv",
mime="text/csv",
help="Example file with the expected format"
)

uploaded\_file = st.file\_uploader(
"Upload your input data (CSV format)",
type=\["csv"],
help="CSV should contain columns for temperature, humidity, pressure, and vacuum"
)

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
    
    # Column mapping
    mapped_columns = map_columns(df)
    if len(mapped_columns) < 4:
        missing_cols = [col for col in feature_names if col not in mapped_columns]
        st.error(f"Could not find columns for: {', '.join(missing_cols)}")
        st.stop()
        
    df_processed = df.rename(columns=mapped_columns)
    required_cols = feature_names  # From feature_bounds
    
    # Check for missing columns after mapping
    missing_cols = [col for col in required_cols if col not in df_processed.columns]
    if missing_cols:
        st.error(f"Missing columns after mapping: {', '.join(missing_cols)}")
        st.stop()
        
    # Process data
    with st.spinner("Processing data..."):
        features = df_processed[required_cols]
        try:
            scaled = scaler.transform(features)
            rf_preds = rf_model.predict(scaled)
            xgb_preds = xgb_model.predict(scaled)
            final_preds = input_weight * rf_preds + (1 - input_weight) * xgb_preds
            
            results = df_processed.copy()
            results['RF_Prediction (MW)'] = rf_preds
            results['XGB_Prediction (MW)'] = xgb_preds
            results['Ensemble_Prediction (MW)'] = final_preds
            
            st.success("Predictions completed!")
            
            # Display results with conditional formatting
            def color_positive_green(val):
                color = 'green' if val > (rf_preds.mean() + xgb_preds.mean())/2 else 'red'
                return f'color: {color}'
            
            st.dataframe(results.style.format({
                'RF_Prediction (MW)': '{:.2f}',
                'XGB_Prediction (MW)': '{:.2f}',
                'Ensemble_Prediction (MW)': '{:.2f}'
            }).applymap(color_positive_green, subset=['Ensemble_Prediction (MW)']))
            
            # Download results
            csv = results.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Full Results",
                data=csv,
                file_name="ccpp_predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            
except Exception as e:
    st.error(f"Error processing file: {str(e)}")
```

# Footer

st.markdown("---")
st.caption("""
Developed with Streamlit | Optimized with Particle Swarm Optimization (PSO)
Model weights: Random Forest ({:.0f}%), XGBoost ({:.0f}%)
""".format(input\_weight\*100, (1-input\_weight)\*100)) File "/mount/src/ccpp/streamlit\_app.py", line 1
update this code import streamlit as st
^
SyntaxError: invalid syntax
