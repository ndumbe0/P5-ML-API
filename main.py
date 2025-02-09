from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib
import numpy as np
from tempfile import NamedTemporaryFile
import os
import webbrowser
import uvicorn
import threading
from typing import List
from pydantic import BaseModel

app = FastAPI()

# Load model and scaler
MODEL_PATH = os.path.join(os.getcwd(), "sepsis_model.joblib")
SCALER_PATH = os.path.join(os.getcwd(), "scaler.joblib")

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file not found. Please ensure they exist in the working directory.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Store uploaded data globally
uploaded_data = None

class PatientRequest(BaseModel):
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
        
        # Ensure sepsis column is handled if present
        if 'Sepssis' in df.columns:
            df = df.drop(columns=['Sepssis'])
        
        uploaded_data = df
        return {"message": "File uploaded successfully", "patient_count": len(df)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patient_ids/", response_model=List[str])
async def get_patient_ids():
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="Upload a file first")
    return uploaded_data['ID'].tolist()

@app.post("/predict/")
async def predict_sepsis(request: PatientRequest):
    global uploaded_data
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="Upload a file first")
    
    patient_id = request.patient_id
    patient = uploaded_data[uploaded_data['ID'] == patient_id]
    
    if patient.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    try:
        # Clean data - remove non-numeric columns
        numeric_cols = patient.select_dtypes(include=np.number).columns.tolist()
        features = patient[numeric_cols].copy()
        
        # Fill missing values with column medians
        features = features.fillna(features.median())
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Predict
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)[0][1]
        
        return {
            "patient_id": patient_id,
            "prediction": "Positive" if prediction[0] == 1 else "Negative",
            "probability": float(probability),
            "features": features.iloc[0].to_dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

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

def open_browser():
    import time
    time.sleep(3)  # Increased delay to ensure server starts
    webbrowser.open("http://127.0.0.1:8000/docs", new=0)  # new=0 opens in same tab

if __name__ == "__main__":
    # Prevent multiple browser tabs
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Thread(target=open_browser).start()
    
    # Run the FastAPI app
    uvicorn.run(app, host="127.0.0.1", port=8000)