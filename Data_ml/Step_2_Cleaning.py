# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model and scaler
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to map building states to numerical values
def encode_building_state(state):
    state_mapping = {
        'GOOD': 1,
        'AS_NEW': 2,
        'TO_RENOVATE': 3,
        'TO_BE_DONE_UP': 4,
        'JUST_RENOVATED': 5,
        'TO_RESTORE': 6,
        'UNKNOWN': 0
    }
    return state_mapping.get(state, 0)  # Default to 'UNKNOWN' if not found

# Function to map property types to numerical values
def encode_property_type(property_type):
    property_type_mapping = {
        'HOUSE': 0,
        'APARTMENT': 1
    }
    return property_type_mapping.get(property_type, -1)  # Default to -1 if not found

# Function to make predictions
def predict_price(features):
    # Scale the features
    scaled_features = scaler.transform([features])
    # Make prediction
    price_prediction = model.predict(scaled_features)
    return price_prediction[0]

# Streamlit app title
st.title("House Price Prediction")

# User input for features
st.sidebar.header("Input Features")

# User inputs
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10)
property_type_input = st.sidebar.selectbox("Property Type", options=["HOUSE", "APARTMENT"])
postal_code = st.sidebar.text_input("Postal Code")
facades = st.sidebar.number_input("Facades", min_value=0, max_value=10)
terrace = st.sidebar.number_input("Terrace (Yes=1, No=0)", min_value=0, max_value=1)
terrace_surface = st.sidebar.number_input("Terrace Surface (m²)")
building_state_input = st.sidebar.selectbox("Building State", options=["GOOD", "AS_NEW", "TO_RENOVATE", "TO_BE_DONE_UP", "JUST_RENOVATED", "TO_RESTORE", "UNKNOWN"])
garden = st.sidebar.number_input("Garden (Yes=1, No=0)", min_value=0, max_value=1)
garden_surface = st.sidebar.number_input("Garden Surface (m²)")
pool = st.sidebar.number_input("Pool (Yes=1, No=0)", min_value=0, max_value=1)
living_area = st.sidebar.number_input("Living Area (m²)")
surface_of_the_plot = st.sidebar.number_input("Surface of the Plot (m²)")
wealth_index = st.sidebar.number_input("Wealth Index", min_value=0.0)
density = st.sidebar.number_input("Density", min_value=0.0)

# Collect user input into a list and encode categorical variables
input_features = [
    bedrooms,
    encode_property_type(property_type_input),  # Encoding property type
    postal_code,  # Assuming postal_code is not used directly; consider encoding it if needed
    facades,
    terrace,
    terrace_surface,
    encode_building_state(building_state_input),  # Encoding building state
    garden,
    garden_surface,
    pool,
    living_area,
    surface_of_the_plot,
    wealth_index,
    density
]

# Make a prediction button
if st.sidebar.button("Predict Price"):
    predicted_price = predict_price(input_features)
    st.write(f"**Predicted Price:** ${predicted_price:,.2f}")
