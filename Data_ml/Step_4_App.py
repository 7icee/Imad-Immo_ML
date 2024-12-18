import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model, scaler, and additional data
model = joblib.load('Imad-Immo_ML/Data_ml/Step_4_Linear_Regression_Model.pkl')
scaler = joblib.load('Step_4_Scaler.pkl')
wealth_and_density_data = pd.read_excel('Step_4_Postal_Codes_Data.xlsx')

# Function to retrieve wealth index and density based on postal code
def get_wealth_and_density(postal_code):
    info = wealth_and_density_data[wealth_and_density_data['Postal_code'] == postal_code]
    if info.empty:
        raise ValueError(f"Postal code {postal_code} not found.")
    wealth_index = info['Wealth_Index'].values[0]
    density = info['Density'].values[0]
    return wealth_index, density

# Set up Streamlit with custom CSS
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏡",
    layout="wide"
)

st.markdown(
    """
    <style>
    div.block-container {
        padding-top: 4rem; /* Reduce space above the title */
    }
    .stApp {
        background-color: #bccbd4; /* Light gray background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add page title and description
st.title("🏡 Real Estate Price Predictor")
st.write("## Predict property prices using machine learning")

# Divide the input page into 3 columns
col1, col2, col3 = st.columns(3)

# Column 1 retrieving Building and Location information
with col1:
    st.markdown("#### Building & Location")
    property_type = st.selectbox("Property Type", ["Home", "Apartment"])
    postal_code = st.number_input("Postal Code", min_value=1000, max_value=9999, step=1)
    facades = st.number_input("Number of Facades", min_value=0, step=1)
    building_state = st.selectbox("Building State", [
        "Good", "As New", "To Renovate", "To Be Done Up",
        "Just Renovated", "To Restore", "Unknown"
    ])
    

# Column 2 retrieving Property Details
with col2:
    st.markdown("#### Property Details")
    living_area = st.number_input("Living Area (m²)", min_value=0, step=1)
    if property_type == "Home":
        plot_surface = st.number_input("Area of the Property (m²)", min_value=0, step=1)
    else:
        plot_surface = 0
        st.text_input("Area of the Property (m²)", value="N/A", disabled=True)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1)
    pool_option = st.selectbox("Is there a Pool?", ["No", "Yes"])
    pool = 1 if pool_option == "Yes" else 0

# Column 3 retrieving additional information
with col3:
    st.markdown("#### Additional information")
    terrace = st.selectbox("Is there a Terrace?", ["No", "Yes"])
    if terrace == "Yes":
        terrace_surface = st.number_input("Terrace Surface (m²)", min_value=0, step=1)
    else:
        terrace_surface = 0
        st.text_input("Terrace Surface (m²)", value="N/A", disabled=True)
    garden = st.selectbox("Is there a Garden?", ["No", "Yes"])
    if garden == "Yes":
        garden_surface = st.number_input("Garden Surface (m²)", min_value=0, step=1)
    else:
        garden_surface = 0
        st.text_input("Garden Surface (m²)", value="N/A", disabled=True)
    
#Predicition Price button creation
if st.button("Predict Price"):
    error_messages = []

    # Inputs error management
    if living_area < 10:
        error_messages.append("Please provide a value for the Living Area")
    
    if property_type == "Home" and plot_surface < 10:
        error_messages.append("Please provide a value for the Area of the Property")

    if terrace == "Yes" and terrace_surface < 1:
        error_messages.append("Please provide a value for the surface of the terrace")

    if garden == "Yes" and garden_surface < 1:
        error_messages.append("Please provide a value for the surface of the garden")

    if error_messages:
        for message in error_messages:
            st.error(message)
    else:
        try:
            # Retrieve wealth index and density based on postal code
            wealth_index, density = get_wealth_and_density(postal_code)

            # Complete input data
            input_data = [
                bedrooms,
                0 if property_type == "Home" else 1,
                postal_code,
                facades,
                1 if terrace == "Yes" else 0,
                terrace_surface,
                {"Good": 1, "As New": 2, "To Renovate": 3, "To Be Done Up": 4, "Just Renovated": 5, "To Restore": 6, "Unknown": 0}[building_state],
                1 if garden == "Yes" else 0,
                garden_surface,
                1 if pool else 0,
                living_area,
                plot_surface,
                wealth_index,
                density
            ]

            # Scale the input and make prediction
            scaled_input = scaler.transform([input_data])
            predicted_price = model.predict(scaled_input)

            # Display the predicted price
            st.success(f"Predicted Price: **${predicted_price[0]:,.2f}**")

        except ValueError as e:
            st.error(str(e))

# Footer
st.markdown("---")
st.write("### Powered by Imachine Learning 🚀")
