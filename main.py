from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from tempfile import NamedTemporaryFile
import os

app = FastAPI()

# Load model and scaler
MODEL_PATH = os.path.join(os.getcwd(), "sepsis_model.joblib")
SCALER_PATH = os.path.join(os.getcwd(), "scaler.joblib")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file not found. Please ensure they exist in the working directory.")

model = joblib.load('sepsis_model.joblib')
scaler = joblib.load('scaler.joblib')

# Store uploaded data globally
uploaded_data = None

class PatientID(BaseModel):
    patient_id: str

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global uploaded_data
    try:
        # Save uploaded file temporarily
        suffix = '.csv' if file.filename.endswith('.csv') else '.xlsx'
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Read file
        if suffix == '.csv':
            df = pd.read_csv(tmp_path)
        else:
            df = pd.read_excel(tmp_path)
        
        os.unlink(tmp_path)  # Delete temp file
        
        uploaded_data = df
        return {"message": "File uploaded successfully", "patient_count": len(df)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patient_ids/")
async def get_patient_ids():
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="Upload a file first")
    return {"patient_ids": uploaded_data['ID'].tolist()}

@app.post("/predict/")
async def predict_sepsis(patient_id: PatientID):
    global uploaded_data
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="Upload a file first")
    
    patient = uploaded_data[uploaded_data['ID'] == patient_id.patient_id]
    if patient.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Preprocess
    features = patient.drop(['ID'], axis=1)
    features = features.fillna(features.median())
    scaled_features = scaler.transform(features)
    
    # Predict
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)[0][1]
    
    return {
        "patient_id": patient_id.patient_id,
        "prediction": "Positive" if prediction[0] == 1 else "Negative",
        "probability": float(probability),
        "features": features.iloc[0].to_dict()
    }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head><title>Sepsis Prediction API</title></head>
        <body>
            <h1>Welcome to Sepsis Prediction API</h1>
            <p>Visit <a href="/docs">/docs</a> for interactive UI.</p>
        </body>
    </html>
    """