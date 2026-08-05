from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load models
lr_model = joblib.load("logistic_model.joblib")
rf_model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("label_encoder.joblib")


class Patient(BaseModel):
    gender: int
    age: int
    urea: float
    cr: float
    hba1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float

@app.post("/predict/logistic")
def predict_logistic(data: Patient):

    features = np.array([[
        data.gender,
        data.age,
        data.urea,
        data.cr,
        data.hba1c,
        data.chol,
        data.tg,
        data.hdl,
        data.ldl,
        data.vldl,
        data.bmi
    ]])

    features = scaler.transform(features)

    prediction = lr_model.predict(features)

    label = encoder.inverse_transform(prediction)

    return {"Prediction": label[0]}

@app.post("/predict/randomforest")
def predict_randomforest(data: Patient):

    features = np.array([[
        data.gender,
        data.age,
        data.urea,
        data.cr,
        data.hba1c,
        data.chol,
        data.tg,
        data.hdl,
        data.ldl,
        data.vldl,
        data.bmi
    ]])

    prediction = rf_model.predict(features)

    label = encoder.inverse_transform(prediction)

    return {"Prediction": label[0]}

