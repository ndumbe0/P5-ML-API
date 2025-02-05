from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import joblib
import pandas as pd
import os
from pydantic import BaseModel
import numpy as np

# Initialize data directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load artifacts once at startup
try:
    model = joblib.load('sepsis_model.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
except FileNotFoundError:
    raise RuntimeError("Model artifacts not found. Train model first!")

app = FastAPI()

class PatientData(BaseModel):
    ID: str
    PRG: float
    PL: float
    PR: float
    SK: float
    TS: float
    M11: float
    BD2: float
    Age: int

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    return {"message": f"File {file.filename} uploaded successfully"}

@app.get("/patient/{patient_id}")
async def get_patient_data(patient_id: str):
    try:
        uploaded_data = pd.read_csv(os.path.join(DATA_DIR, "Paitients_Files_Train.csv"))
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="No data file found")
    
    patient = uploaded_data[uploaded_data['ID'] == patient_id]
    if patient.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return patient.iloc[0].to_dict()

@app.post("/predict/{patient_id}")
async def predict_sepsis(patient_id: str):
    try:
        uploaded_data = pd.read_csv(os.path.join(DATA_DIR, "Paitients_Files_Train.csv"))
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="No data file found")
    
    patient = uploaded_data[uploaded_data['ID'] == patient_id]
    if patient.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    input_data = patient.drop(['ID', 'Sepssis', 'Insurance'], axis=1)
    input_data = pd.get_dummies(input_data)
    
    # Ensure all training features are present
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    input_data = input_data[feature_columns]
    scaled_data = scaler.transform(input_data)
    
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]
    
    return {
        "patient_id": patient_id,
        "sepsis_prediction": "Positive" if prediction[0] else "Negative",
        "probability": float(probability)
    }