import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model
model = joblib.load('best_model_rf.pkl')

# Define the input fields
def get_user_input():
    age = st.slider('Age', 20, 100, 50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cigsPerDay = st.slider('Cigarettes Per Day', 0, 50, 10)
    totChol = st.slider('Total Cholesterol', 100, 400, 200)
    sysBP = st.slider('Systolic Blood Pressure', 90, 200, 120)
    diaBP = st.slider('Diastolic Blood Pressure', 60, 120, 80)
    BMI = st.slider('Body Mass Index', 10, 50, 25)
    heartRate = st.slider('Heart Rate', 50, 150, 70)
    glucose = st.slider('Glucose Level', 50, 200, 100)

    user_data = {
        'age': age,
        'cigsPerDay': cigsPerDay,
        'totChol': totChol,
        'sysBP': sysBP,
        'diaBP': diaBP,
        'BMI': BMI,
        'heartRate': heartRate,
        'glucose': glucose,
        'male_Male': 1 if sex == 'Male' else 0,
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
user_input = get_user_input()

# Predict CHD risk
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Display results
st.subheader('Prediction')
st.write('Risk of CHD' if prediction[0] else 'No Risk of CHD')

st.subheader('Prediction Probability')
st.write(prediction_proba)
