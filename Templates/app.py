import streamlit as st
import numpy as np
import pickle

# Load the trained model and preprocessor
knr = pickle.load(open('knr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #2e7d32;'>Crop Yield Prediction Per Country</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: #555;'>Enter the details below to predict crop yield (hg/ha)</h4>",
    unsafe_allow_html=True
)
st.write("---")

# Input fields for all required features
Year = st.number_input("Year", min_value=1900, max_value=2100, value=2013, step=1)
average_rain_fall_mm_per_year = st.number_input("Average Rainfall (mm per year)", min_value=0.0, value=0.0, step=0.1)
pesticides_tonnes = st.number_input("Pesticides (tonnes)", min_value=0.0, value=0.0, step=0.1)
avg_temp = st.number_input("Average Temperature (°C)", value=0.0, step=0.1)
Area = st.text_input("Area (Country)", placeholder="e.g. Albania")
Item = st.text_input("Item (Crop)", placeholder="e.g. Maize")

st.write("")

# Predict button
if st.button("Predict", type="primary", use_container_width=True):
    if not Area or not Item:
        st.error("Please enter both Area and Item before predicting.")
    else:
        features = np.array(
            [[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]],
            dtype=object
        )
        transformed_features = preprocessor.transform(features)
        prediction = knr.predict(transformed_features).reshape(1, -1)

        st.success(f"Predicted Yield: {prediction[0][0]} hg/ha")