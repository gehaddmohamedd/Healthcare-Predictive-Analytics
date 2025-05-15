# import streamlit as st
import pandas as pd
# import numpy as np
import joblib

# Load the trained model
# model = joblib.load('heart_disease_model.pkl')

# st.title("Heart Disease Prediction")

# # Input fields
# age = st.slider("Age", 20, 100, 50)
# sex = st.selectbox("Sex", [0, 1])  # 0: female, 1: male
# cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
# thalach = st.slider("Max Heart Rate (thalach)", 70, 210, 150)
# exang = st.selectbox("Exercise-induced angina (exang)", [0, 1])
# oldpeak = st.number_input("ST depression (oldpeak)")
# ca_log = st.number_input("Number of major vessels (ca_log)")
# thal = st.selectbox("Thalassemia (thal)", [0, 1, 2])
# slope = st.selectbox("Slope", [0, 1, 2])
# sex = int(sex)

# # Predict button
# if st.button("Predict"):
#     input_data = np.array([[ca_log, thal, oldpeak, thalach, exang, cp, sex, slope]])
#     prediction = model.predict(input_data)
#     result = "Has Heart Disease" if prediction[0] == 1 else "No Heart Disease"
#     st.success(f"Prediction: {result}")
import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Set page config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")


# Title with emoji
st.title("Heart Disease Prediction App ❤️")
st.markdown("""
Welcome to the Heart Disease Prediction app.
Enter the patient data below to predict the risk of heart disease.
            
""")

# Collect user input
age = st.slider('Age', 18, 100, 50)

sex = st.selectbox('Sex', options=['Female', 'Male'])
sex = 1 if sex == 'Male' else 0

cp_map = {
    'Typical angina': 0,
    'Atypical angina': 1,
    'Non-anginal pain': 2,
    'Asymptomatic': 3
}
cp_input = st.selectbox('Chest Pain Type (cp)', options=list(cp_map.keys()))
cp = cp_map[cp_input]

trestbps = st.slider('Resting Blood Pressure (trestbps)', 80, 200, 120)
chol = st.slider('Serum Cholestoral (chol)', 100, 600, 240)

fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=['False', 'True'])
fbs = 1 if fbs == 'True' else 0

restecg_map = {
    'Normal': 0,
    'ST-T wave abnormality': 1,
    'Left ventricular hypertrophy': 2
}
restecg_input = st.selectbox('Resting ECG (restecg)', options=list(restecg_map.keys()))
restecg = restecg_map[restecg_input]

thalach = st.slider('Maximum Heart Rate (thalach)', 60, 220, 150)

exang = st.selectbox('Exercise Induced Angina (exang)', options=['No', 'Yes'])
exang = 1 if exang == 'Yes' else 0

oldpeak = st.slider('ST depression (oldpeak)', 0.0, 6.0, 1.0)

slope_map = {
    'Upsloping': 0,
    'Flat': 1,
    'Downsloping': 2
}
slope_input = st.selectbox('Slope of ST Segment (slope)', options=list(slope_map.keys()))
slope = slope_map[slope_input]

ca = st.slider('Number of Major Vessels (ca)', 0, 4, 0)

thal_map = {
    'Normal': 1,
    'Fixed defect': 2,
    'Reversible defect': 3
}
thal_input = st.selectbox('Thalassemia (thal)', options=list(thal_map.keys()))
thal = thal_map[thal_input]

# Create feature array
features = np.array([[ca, thal, oldpeak, thalach, exang, cp, sex, slope]])

# Prediction button
if st.button('Predict Heart Disease'):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("The patient is likely to have heart disease. 😡")
    else:
        st.success("The patient is not likely to have heart disease. 😊")

# Footer
st.markdown("""
---
*Made with ❤️*
""")
