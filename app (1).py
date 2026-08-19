
import streamlit as st
import pandas as pd
import joblib

model_data = joblib.load("house_price_model.pkl")

preprocessor = model_data["preprocessor"]
model = model_data["model"]

st.title("🏠 House Price Prediction")

st.write("Enter the house details below.")

location = st.selectbox(
    "Location",
    ["Chennai", "Coimbatore", "Salem", "Madurai", "Trichy"]
)

property_type = st.selectbox(
    "Property Type",
    ["Apartment", "Villa", "Independent House"]
)

area = st.number_input(
    "Area (sqft)",
    min_value=500,
    max_value=5000,
    value=1500
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

house_age = st.number_input(
    "House Age",
    min_value=0,
    max_value=100,
    value=5
)

parking = st.number_input(
    "Parking",
    min_value=0,
    max_value=5,
    value=1
)

distance = st.number_input(
    "Distance to City (km)",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)

if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "Location": [location],
        "Property_Type": [property_type],
        "Area_sqft": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "House_Age": [house_age],
        "Parking": [parking],
        "Distance_to_City": [distance]
    })

    input_processed = preprocessor.transform(input_data)

    prediction = model.predict(input_processed)[0]

    st.success(
        f"Predicted House Price: ₹{prediction:,.0f}"
    )
