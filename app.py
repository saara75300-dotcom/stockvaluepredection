import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(
    page_title="Stock Value Prediction",
    page_icon="📈",
    layout="centered"
)

# Load trained model
with open("stock_model.pkl", "rb") as file:
    model = pickle.load(file)

# UI Header
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>📊 Stock Value Prediction App</h1>",
    unsafe_allow_html=True
)

st.write("Predict future stock value using Machine Learning")

st.divider()

# Input Section
st.subheader("🔢 Enter Stock Details")

col1, col2 = st.columns(2)

with col1:
    open_price = st.number_input("Open Price", min_value=0.0)
    low_price = st.number_input("Low Price", min_value=0.0)

with col2:
    high_price = st.number_input("High Price", min_value=0.0)
    volume = st.number_input("Volume", min_value=0.0)

st.divider()

# Prediction
if st.button("📈 Predict Stock Value"):
    input_data = np.array([[open_price, high_price, low_price, volume]])
    prediction = model.predict(input_data)

    st.success(f"✅ Predicted Stock Value: ₹ {prediction[0]:.2f}")

# Footer
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)
