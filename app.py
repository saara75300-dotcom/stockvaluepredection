%%writefile app.py
import streamlit as st
import pandas as pd

# Assuming model_logic.py exists in the same directory
# and contains the necessary functions.
# If you removed it, you'll need to generate it again.
# from model_logic import get_dummy_stock_data, train_linear_regression_model, predict_split_factor

# --- Dummy functions for demonstration if model_logic.py is not available --- 
# In a real scenario, these would come from model_logic.py
def get_dummy_stock_data():
    dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=100, freq='D'))
    split_factors = [1.0] * 100
    dummy_data = pd.DataFrame({
        'date': dates,
        'close': [np.random.rand() * 1000 for _ in range(100)],
        'high': [np.random.rand() * 1000 + 10 for _ in range(100)],
        'splitFactor': split_factors
    })
    return dummy_data

def train_linear_regression_model(data):
    # This is a placeholder. A real model would be trained here.
    class MockLinearRegression:
        def predict(self, X):
            # Always predict 1.0 for dummy model
            return np.array([1.0])
    return MockLinearRegression()

def predict_split_factor(model, day_of_year):
    return model.predict(pd.DataFrame([[day_of_year]], columns=['date']))[0]
# --- End of Dummy functions ---

import numpy as np # Ensure numpy is imported for dummy functions

st.set_page_config(layout="wide")

st.title('Stock Split Factor Predictor')

st.write('This app predicts the stock split factor using a simple Linear Regression model.')

# Load and train model (using dummy data for demonstration)
# If you have 'GOOG.csv' in the same directory, you can uncomment and use the line below:
# stock_data = pd.read_csv('GOOG.csv')
stock_data = get_dummy_stock_data()
trained_model = train_linear_regression_model(stock_data)

st.header("Predict Split Factor")
day_input = st.number_input('Enter a day of the year (1-366) to predict the split factor:', min_value=1, max_value=366, value=180)

if st.button('Predict'):
    if trained_model:
        predicted_value = predict_split_factor(trained_model, day_input)
        st.success(f'Predicted split factor for day {day_input}: **{predicted_value:.2f}**')
    else:
        st.error("Model not trained. Please ensure data is loaded correctly.")

st.sidebar.header('About')
st.sidebar.write('This is a demo Streamlit app integrating a simple Linear Regression model.')
st.sidebar.write('The model predicts a stock split factor based on the day of the year.')
st.sidebar.write('*(Using dummy data for demonstration purposes)*')
