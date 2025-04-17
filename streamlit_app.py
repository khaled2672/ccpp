# 1. Load Models
@st.cache_resource
def load_models():
    try:
        return {
            'rf_model': joblib.load('random_forest_model.joblib'),
            'xgb_model': joblib.load('xgboost_model.joblib'),
            'scaler': joblib.load('scaler.joblib'),
            'best_weight': np.load('best_weight.npy').item()
        }
    except FileNotFoundError as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

models = load_models()

# 2. Preprocess the dataset
def preprocess_data(df):
    # Automatic column detection and mapping
    column_mapping = {
        "Ambient Temperature": ["Ambient Temperature", "Temperature", "Temp", "Amb Temp"],
        "Relative Humidity": ["Relative Humidity", "Humidity", "Rel Humidity", "Humidity (%)"],
        "Ambient Pressure": ["Ambient Pressure", "Pressure", "Amb Pressure", "Pressure (mbar)"],
        "Exhaust Vacuum": ["Exhaust Vacuum", "Vacuum", "Exhaust Vac", "Vacuum (cmHg)"]
    }

    # Try to find the columns based on column names or regex
    processed_columns = {}
    for target, possible_names in column_mapping.items():
        for name in possible_names:
            if name in df.columns:
                processed_columns[target] = name
                break

    if len(processed_columns) < 4:
        st.error("CSV file is missing required columns!")
        return None

    # Rename columns to match expected names
    df = df.rename(columns=processed_columns)
    return df

# 3. Upload and process the CSV
st.subheader("📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload input data (CSV format)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data", df.head())

    # Preprocess and map columns
    df_processed = preprocess_data(df)
    if df_processed is not None:
        st.write("✅ Dataset Columns Mapped Successfully")

        # Perform scaling and prediction
        features = df_processed[["Ambient Temperature", "Relative Humidity", "Ambient Pressure", "Exhaust Vacuum"]]
        scaled = models['scaler'].transform(features)
        rf_preds = models['rf_model'].predict(scaled)
        xgb_preds = models['xgb_model'].predict(scaled)

        # Apply ensemble model
        final_preds = models['best_weight'] * rf_preds + (1 - models['best_weight']) * xgb_preds
        df_processed['Predicted Power (MW)'] = final_preds

        st.write("⚡ Predictions", df_processed)

        # Download Button
        csv = df_processed.to_csv(index=False).encode()
        st.download_button("⬇️ Download Results as CSV", data=csv, file_name="predicted_power.csv", mime='text/csv')
