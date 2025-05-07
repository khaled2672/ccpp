import streamlit as st
import numpy as np

# Define a function to increase or decrease values
def adjust_value(value, delta=0.01, min_val=0.0, max_val=100.0):
    """Adjust the value within bounds."""
    new_value = value + delta
    new_value = max(min(new_value, max_val), min_val)
    return new_value

# Initialize session state for theme persistence
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("⚙️ CCPP Power Predictor")

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

# Feature bounds for UI
feature_bounds = {
    'Ambient Temperature': [0.0, 50.0],
    'Ambient Relative Humidity': [10.0, 100.0],
    'Ambient Pressure': [799.0, 1035.0],
    'Exhaust Vacuum': [3.0, 12.0],
    'Model Weight (RF vs XGB)': [0.0, 1.0]
}

# Input sliders with increment and decrement buttons
st.subheader("Input Parameters")
inputs = {}
for feature, (low, high) in feature_bounds.items():
    default = (low + high) / 2

    # Add buttons to adjust the value by 0.01
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button(f"➖ {feature}", key=f"{feature}_minus"):
            default = adjust_value(default, delta=-0.01, min_val=low, max_val=high)

    with col2:
        inputs[feature] = st.slider(
            feature, low, high, default,
            help=f"Adjust {feature} between {low} and {high}",
            key=feature
        )

    with col3:
        if st.button(f"➕ {feature}", key=f"{feature}_plus"):
            default = adjust_value(default, delta=0.01, min_val=low, max_val=high)

# Reset button
if st.button("🔄 Reset to Defaults"):
    for feature in inputs:
        inputs[feature] = (feature_bounds[feature][0] + feature_bounds[feature][1]) / 2
