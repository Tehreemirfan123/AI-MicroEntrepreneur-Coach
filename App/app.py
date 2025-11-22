import os
import sys
import streamlit as st
import pandas as pd
import joblib
from utils.predict import prepare_input, predict_profit, predict_failure

# --- HTML Styling for the pointer cusor usage ---
st.markdown("""
    <style>
    div[data-baseweb="select"] * {
        cursor: pointer !important;
    }
    </style>
""", unsafe_allow_html=True) 

# Define project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_path = os.path.join(project_root, "Dataset_Generation", "micro_business_dataset_500.csv")
models_path = os.path.join(project_root, "Models_Files")
sys.path.append(project_root)

# Load dataset and models
df = pd.read_csv(dataset_path)

profit_model = joblib.load(os.path.join(models_path, "profit_model.pkl"))
failure_model = joblib.load(os.path.join(models_path, "failure_model.pkl"))
encoder = joblib.load(os.path.join(models_path, "encoder.pkl"))

# Streamlit UI
st.markdown("""
    <h1 style='text-align: center; font-style: italic; color: white;'>BizBuddy AI</h1>
    <h3 style='text-align: center; color:#cc4444;'>Welcome to BizBuddy AI!</h3>
    <p style='text-align: center; color: #EDF0DA; font-size:17px;'>
        Skill up. Start up. Smarter with AI.
    </p>
""", unsafe_allow_html=True)
st.write("Enter details to predict monthly profit and failure risk for your micro-business.")

city = st.selectbox("City", df['City'].unique())
business = st.selectbox("Business Type", df['Business'].unique())
product = st.selectbox("Product / Service", df['Product/Service'].unique())
marketing_channel = st.selectbox("Marketing Channel", df['Marketing_Channel'].unique())
startup_cost = st.number_input("Startup Cost (PKR)", min_value=0)
cost_per_unit = st.number_input("Cost per Unit (PKR)", min_value=0)
price_per_unit = st.number_input("Price per Unit (PKR)", min_value=0)

# Prediction Button
if st.button("Predict"):
    sample = {
        "Business": business,
        "City": city,
        "Product/Service": product,
        "Marketing_Channel": marketing_channel,
        "Startup_Cost_PKR": startup_cost,
        "Cost_per_Unit": cost_per_unit,
        "Price_per_Unit": price_per_unit
    }

    # Pass models and encoder
    profit = predict_profit(sample, profit_model, encoder)
    failure = predict_failure(sample, failure_model, encoder)
    
    # Prediction Results
    failure_percentage = round(failure * 100, 2)
    st.subheader("Prediction Results")
    st.write(f"**Estimated Monthly Profit:** {round(profit, 2)} PKR")
    st.write(f"**Failure Risk Probability:** {failure_percentage}%")
