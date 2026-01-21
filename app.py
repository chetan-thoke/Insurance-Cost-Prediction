import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("models/best_insurance_model.pkl")

model = load_model()

st.title("Insurance Premium Prediction 💰")

age = st.number_input("Age", min_value=18, max_value=100)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)
children = st.number_input("Number of Children", min_value=0, max_value=5)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northwest", "northeast", "southwest", "southeast"])

if st.button("Predict Premium"):
    input_df = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex": [sex],
        "smoker": [smoker],
        "region": [region]
    })

    prediction = model.predict(input_df)
    st.success(f"Estimated Insurance Premium: ₹{prediction[0]:,.2f}")
