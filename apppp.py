import io

@st.cache_data
def load_and_train_model(csv_data):
    # csv_data can be a filepath (str) or file-like object from uploader
    df = pd.read_csv(csv_data)
    df = df.dropna(subset=['Total Power', 'Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure', 'Fuel_Consumption_MMBtu'])
    X = df[['Total Power', 'Ambient Temperature', 'Ambient Relative Humidity', 'Ambient Pressure']]
    y = df['Fuel_Consumption_MMBtu']
    model = LinearRegression()
    model.fit(X, y)
    return model

st.title("CCPP Economic Analysis Dashboard")

uploaded_file = st.file_uploader("Upload CSV data file (optional)", type=['csv'])

if uploaded_file is not None:
    # Use uploaded file for training
    model = load_and_train_model(uploaded_file)
else:
    # Use default file path
    model = load_and_train_model('data_with_fuel_consumption.csv')

# Rest of your code here...
