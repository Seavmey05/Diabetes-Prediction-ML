import streamlit as st
import requests

st.title("Diabetes Prediction System")

model = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Random Forest"]
)

gender = st.selectbox("Gender", ["Female", "Male"])
gender = 0 if gender == "Female" else 1

age = st.number_input("Age", min_value=1)

urea = st.number_input("Urea")

cr = st.number_input("Creatinine")

hba1c = st.number_input("HbA1c")

chol = st.number_input("Cholesterol")

tg = st.number_input("Triglycerides")

hdl = st.number_input("HDL")

ldl = st.number_input("LDL")

vldl = st.number_input("VLDL")

bmi = st.number_input("BMI")

if st.button("Predict"):

    patient = {
        "gender": gender,
        "age": age,
        "urea": urea,
        "cr": cr,
        "hba1c": hba1c,
        "chol": chol,
        "tg": tg,
        "hdl": hdl,
        "ldl": ldl,
        "vldl": vldl,
        "bmi": bmi
    }

    if model == "Logistic Regression":
        url = "http://127.0.0.1:8000/predict/logistic"
    else:
        url = "http://127.0.0.1:8000/predict/randomforest"

    response = requests.post(url, json=patient)

    if response.status_code == 200:
        prediction = response.json()["Prediction"]
        st.success(f"Prediction: {prediction}")
    else:
        st.error("Prediction failed. Check if FastAPI is running.")