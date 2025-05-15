# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle

# # Load the trained model and scaler
# try:
#     with open("model.pkl", "rb") as f:
#         model = pickle.load(f)
#     with open("scaler.pkl", "rb") as f:
#         scaler = pickle.load(f)
# except FileNotFoundError as e:
#     st.error(f"Error: {e}. Please make sure model.pkl and scaler.pkl are in the app directory.")
#     st.stop()

# def bp_status(row):
#     if row['ap_hi'] >= 140 or row['ap_lo'] >= 90:
#         return 'High'
#     elif (row['ap_hi'] >= 130) or (row['ap_lo'] >= 80):
#         return 'Elevated'
#     else:
#         return 'Normal'


# # Feature list (based on cardiovascular dataset processing)
# features = [
#     'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
#     'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'BMI'
# ]

# # Streamlit app
# st.title("Cardiovascular Disease Prediction")
# st.write("Fill in the patient data to predict the risk of cardiovascular disease.")

# input_data = {}

# with st.form("prediction_form"):
#     input_data['age'] = st.number_input("Age (in years)", min_value=1, max_value=120, value=50)*365
#     input_data['gender'] = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
#     input_data['height'] = st.number_input("Height (in cm)", min_value=100, max_value=250, value=165)
#     input_data['weight'] = st.number_input("Weight (in kg)", min_value=30.0, max_value=200.0, value=70.0)
#     input_data['ap_hi'] = st.number_input("Systolic BP (ap_hi)", value=120)
#     input_data['ap_lo'] = st.number_input("Diastolic BP (ap_lo)", value=80)
#     input_data['cholesterol'] = st.selectbox("Cholesterol", options=[1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x - 1])
#     input_data['gluc'] = st.selectbox("Glucose", options=[1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x - 1])
#     input_data['smoke'] = st.selectbox("Smokes?", options=[0, 1])
#     input_data['alco'] = st.selectbox("Alcohol intake?", options=[0, 1])
#     input_data['active'] = st.selectbox("Physically active?", options=[0, 1])
    

#     submitted = st.form_submit_button("Predict")

# if submitted:
#     try:
#         input_df = pd.DataFrame([input_data], columns=features)
#         height_m = input_data['height'] / 100
#         input_df['BMI'] = input_df['weight'] / (height_m ** 2)
#         input_df['bp_status'] = input_df.apply(bp_status, axis=1)
#         input_df['BMI_High'] = (input_df['BMI'] >= 25).astype(int)
#         continuous_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
#         input_df[continuous_features] = scaler.transform(input_df[continuous_features])
#         prediction = model.predict(input_df)[0]

#         if prediction == 1:
#             st.warning("High risk of cardiovascular disease.")
#         else:
#             st.success("Low risk of cardiovascular disease.")

#         st.write("Input Summary:")
#         st.json(input_data)
#     except Exception as e:
#         st.error(f"Prediction error: {str(e)}")

# # Info section
# st.markdown("""
# ### Model Info
# This model uses a Random Forest Classifier trained on a cardiovascular dataset.
# Inputs include age, BMI, blood pressure, cholesterol, lifestyle factors, etc.
# """)
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model and scaler
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Error: {e}. Please make sure model.pkl and scaler.pkl are in the app directory.")
    st.stop()

def bp_status(row):
    if row['ap_hi'] >= 140 or row['ap_lo'] >= 90:
        return 'High'
    elif (row['ap_hi'] >= 130) or (row['ap_lo'] >= 80):
        return 'Elevated'
    else:
        return 'Normal'

# Streamlit app
st.title("Cardiovascular Disease Prediction")
st.write("Fill in the patient data to predict the risk of cardiovascular disease.")

input_data = {}

with st.form("prediction_form"):
    input_data['age'] = st.number_input("Age (in years)", min_value=1, max_value=120, value=50)*365
    input_data['gender'] = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
    input_data['height'] = st.number_input("Height (in cm)", min_value=100, max_value=250, value=165)
    input_data['weight'] = st.number_input("Weight (in kg)", min_value=30.0, max_value=200.0, value=70.0)
    input_data['ap_hi'] = st.number_input("Systolic BP (ap_hi)", value=120)
    input_data['ap_lo'] = st.number_input("Diastolic BP (ap_lo)", value=80)
    input_data['cholesterol'] = st.selectbox("Cholesterol", options=[1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x - 1])
    input_data['gluc'] = st.selectbox("Glucose", options=[1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x - 1])
    input_data['smoke'] = st.selectbox("Smokes?", options=[0, 1])
    input_data['alco'] = st.selectbox("Alcohol intake?", options=[0, 1])
    input_data['active'] = st.selectbox("Physically active?", options=[0, 1])
    
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Calculate BMI
        height_m = input_data['height'] / 100
        input_df['BMI'] = input_df['weight'] / (height_m ** 2)
        
        # Determine BP status
        input_df['bp_status'] = input_df.apply(bp_status, axis=1)
        
        # Create BMI_High feature
        input_df['BMI_High'] = (input_df['BMI'] >= 25).astype(int)
        
        # One-hot encode categorical features
        categorical_cols = ['bp_status', 'cholesterol', 'gluc', 'gender']
        input_df = pd.get_dummies(input_df, columns=categorical_cols)
        

        model_features = model.feature_names_in_
        
        # Add missing columns with 0 and reorder to match model
        for feature in model_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
                
        input_df = input_df[model_features]
   
        # Scale continuous features
        continuous_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
        input_df[continuous_features] = scaler.transform(input_df[continuous_features])
        
        # Make prediction
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.warning("High risk of cardiovascular disease.")
        else:
            st.success("Low risk of cardiovascular disease.")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.error("Please check that all input values are valid.")

# Info section
st.markdown("""
### Model Info
This model uses a Random Forest Classifier trained on a cardiovascular dataset.
Inputs include age, BMI, blood pressure, cholesterol, lifestyle factors, etc.
""")