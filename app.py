import numpy as np
import pickle
import pandas as pd
import streamlit as st
from datetime import date
import streamlit.components.v1 as components

# Load the trained model, LabelEncoders, and scaler
with open("xgb.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

with open("label_encoders.pkl", "rb") as enc_file:
    le_dict = pickle.load(enc_file)  # Dictionary of LabelEncoders

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)  # Trained Scaler


def predict_rainfall(input_data):
    # Convert inputs to DataFrame
    columns = [
        "Date", "Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation",
        "Sunshine", "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm",
        "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
        "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am",
        "Temp3pm", "RainToday"
    ]
    df = pd.DataFrame([input_data], columns=columns)

    # Scale numerical features
    df_scaled = scaler.transform(df)

    # Make prediction
    prediction = classifier.predict(df_scaled)
    return prediction


def main():
    st.title("Rainfall Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Rainfall Prediction ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input fields
    date_input = st.date_input("Date")
    location = st.selectbox("Location", le_dict["Location"].classes_)
    minTemp = st.number_input("Min Temperature", value=0.0)
    maxTemp = st.number_input("Max Temperature", value=0.0)
    rainfall = st.number_input("Rainfall", value=0.0)
    evaporation = st.number_input("Evaporation", value=0.0)
    sunshine = st.number_input("Sunshine", value=0.0)
    windGustDir = st.selectbox("Wind Gust Direction", le_dict["WindGustDir"].classes_)
    windGustSpeed = st.number_input("Wind Gust Speed", value=0.0)
    windDir9am = st.selectbox("Wind Direction 9AM", le_dict["WindDir9am"].classes_)
    windDir3pm = st.selectbox("Wind Direction 3PM", le_dict["WindDir3pm"].classes_)
    windSpeed9am = st.number_input("Wind Speed 9AM", value=0.0)
    windSpeed3pm = st.number_input("Wind Speed 3PM", value=0.0)
    humidity9am = st.number_input("Humidity 9AM", value=0.0)
    humidity3pm = st.number_input("Humidity 3PM", value=0.0)
    pressure9am = st.number_input("Pressure 9AM", value=0.0)
    pressure3pm = st.number_input("Pressure 3PM", value=0.0)
    cloud9am = st.number_input("Cloud 9AM", value=0.0)
    cloud3pm = st.number_input("Cloud 3PM", value=0.0)
    temp9am = st.number_input("Temperature 9AM", value=0.0)
    temp3pm = st.number_input("Temperature 3PM", value=0.0)
    rainToday = st.selectbox("Rain Today", le_dict["RainToday"].classes_)

    result = ""
    if st.button("Predict"):
        # Convert date to ordinal
        date_ordinal = date_input.toordinal()

        # Encode categorical inputs
        location_encoded = le_dict["Location"].transform([location])[0]
        windGustDir_encoded = le_dict["WindGustDir"].transform([windGustDir])[0]
        windDir9am_encoded = le_dict["WindDir9am"].transform([windDir9am])[0]
        windDir3pm_encoded = le_dict["WindDir3pm"].transform([windDir3pm])[0]
        rainToday_encoded = le_dict["RainToday"].transform([rainToday])[0]

        # Combine all inputs
        input_data = [
            date_ordinal, location_encoded, minTemp, maxTemp, rainfall, evaporation, sunshine,
            windGustDir_encoded, windGustSpeed, windDir9am_encoded, windDir3pm_encoded,
            windSpeed9am, windSpeed3pm, humidity9am, humidity3pm, pressure9am, pressure3pm,
            cloud9am, cloud3pm, temp9am, temp3pm, rainToday_encoded
        ]

        # Make prediction
        result = predict_rainfall(input_data)
        st.success(f'The prediction is: {result}')
        if(result==[0]):
            st.write("No Rainfall")
        else:
            st.write("Rainfall")
    if st.button("About"):
        st.text("This app predicts rainfall.")
        st.text("Built by sai_kiran_alikana")


if __name__ == '__main__':
    main()
