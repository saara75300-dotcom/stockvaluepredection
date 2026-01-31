import streamlit as st
import pickle
import numpy as np

# Load model
with open("stock_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("📊 Stock Value Prediction App")
st.write("Predict future stock value using Machine Learning")

st.subheader("Enter Stock Details")

open_price = st.number_input("Open Price", min_value=0.0, format="%.2f")
high_price = st.number_input("High Price", min_value=0.0, format="%.2f")
low_price = st.number_input("Low Price", min_value=0.0, format="%.2f")
volume = st.number_input("Volume", min_value=1.0, format="%.0f")

# 🔘 BUTTON CODE — INGATHA PODANUM
if st.button("📈 Predict Stock Value"):
    try:
        input_data = np.array([[open_price, high_price, low_price, volume]])
        prediction = model.predict(input_data)

        st.success(f"✅ Predicted Stock Value: ₹ {prediction[0]:.2f}")

    except Exception as e:
        st.error("❌ Error occurred while predicting")
