import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load and train the model (cache to avoid retraining)
@st.cache_data
def load_and_train_model():
    df = pd.read_csv('/content/data/data_with_fuel_consumption.csv')
    df = df.dropna(subset=['Total Power', 'Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Fuel_Consumption_MMBtu'])
    X = df[['Total Power', 'Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure']]
    y = df['Fuel_Consumption_MMBtu']
    model = LinearRegression()
    model.fit(X, y)
    return model

model = load_and_train_model()

def predict_fuel_consumption(power_output_mw, ambient_temp_c, ambient_humidity_pct, ambient_pressure_kpa, model):
    features = pd.DataFrame({
        'Total Power': [power_output_mw],
        'Ambient Temperature': [ambient_temp_c],
        'Ambient Relative Humidity': [ambient_humidity_pct],
        'Ambient Pressure': [ambient_pressure_kpa]
    })
    return model.predict(features)[0]

def calculate_fuel_cost(fuel_consumption_mmbtu, fuel_price_per_mmbtu):
    return fuel_consumption_mmbtu * fuel_price_per_mmbtu

def calculate_total_variable_cost(fuel_cost, variable_om_cost_per_mwh, power_output_mw):
    return fuel_cost + variable_om_cost_per_mwh * power_output_mw

def calculate_revenue(power_output_mw, market_price_per_mwh):
    return power_output_mw * market_price_per_mwh

def calculate_profitability_metrics(power_output_mw, total_variable_cost, revenue, market_price_per_mwh):
    profit_per_hour = revenue - total_variable_cost
    production_cost_per_mwh = total_variable_cost / power_output_mw if power_output_mw > 0 else float('nan')
    profit_per_mwh = market_price_per_mwh - production_cost_per_mwh if power_output_mw > 0 else float('nan')
    return profit_per_hour, profit_per_mwh

st.title("CCPP Economic Analysis Dashboard")

power_output = st.number_input("Power Output (MW)", min_value=0.0, value=100.0)
ambient_temp = st.number_input("Ambient Temperature (°C)", value=25.0)
ambient_humidity = st.number_input("Ambient Relative Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
ambient_pressure = st.number_input("Ambient Pressure (kPa)", value=101.3)

fuel_price = st.number_input("Fuel Price ($/MMBtu)", min_value=0.0, value=3.5)
variable_om_cost = st.number_input("Variable O&M Cost ($/MWh)", min_value=0.0, value=2.0)
market_price = st.number_input("Market Price ($/MWh)", min_value=0.0, value=50.0)

if st.button("Calculate Economics"):
    predicted_fuel_consumption = predict_fuel_consumption(power_output, ambient_temp, ambient_humidity, ambient_pressure, model)
    fuel_cost = calculate_fuel_cost(predicted_fuel_consumption, fuel_price)
    total_variable_cost = calculate_total_variable_cost(fuel_cost, variable_om_cost, power_output)
    revenue = calculate_revenue(power_output, market_price)
    profit_per_hour, profit_per_mwh = calculate_profitability_metrics(power_output, total_variable_cost, revenue, market_price)

    st.subheader("Results")
    st.write(f"**Predicted Fuel Consumption (MMBtu):** {predicted_fuel_consumption:.2f}")
    st.write(f"**Fuel Cost ($):** {fuel_cost:.2f}")
    st.write(f"**Total Variable Cost ($):** {total_variable_cost:.2f}")
    st.write(f"**Revenue ($):** {revenue:.2f}")
    st.write(f"**Profit per Hour ($):** {profit_per_hour:.2f}")
    st.write(f"**Profit per MWh ($/MWh):** {profit_per_mwh:.2f}")
