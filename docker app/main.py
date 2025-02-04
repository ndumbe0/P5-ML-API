import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
from io import StringIO
import socket
import os

# Initialize data directory
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load and preprocess data
try:
    uploaded_data = pd.read_csv(os.path.join(DATA_DIR, "Paitients_Files_Train.csv"))
except FileNotFoundError:
    uploaded_data = None

data = pd.read_csv("F:\\school\\Azubi Africa\\P5-ML-API\\data\\Paitients_Files_Train.csv")
X = data.drop(['ID', 'Sepssis', 'Insurance'], axis=1)
y = data['Sepssis']

# Preprocessing pipeline
X = pd.get_dummies(X).fillna(X.mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save artifacts
joblib.dump(model, 'sepsis_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# FastAPI application
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
    global uploaded_data
    contents = await file.read()
    
    # Save file to persistent storage
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    
    uploaded_data = pd.read_csv(StringIO(str(contents, 'utf-8')))
    return {"message": f"File {file.filename} uploaded and saved successfully"}

@app.get("/files")
async def list_uploaded_files():
    return {"files": os.listdir(DATA_DIR)}

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)

@app.get("/patient/{patient_id}")
async def get_patient_data(patient_id: str):
    global uploaded_data
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    patient = uploaded_data[uploaded_data['ID'] == patient_id]
    if patient.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return patient.iloc[0].to_dict()

@app.post("/predict/{patient_id}")
async def predict_sepsis(patient_id: str):
    global uploaded_data
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    patient = uploaded_data[uploaded_data['ID'] == patient_id]
    if patient.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    model = joblib.load('sepsis_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    input_data = patient.drop(['ID', 'Sepssis', 'Insurance'], axis=1)
    input_data = pd.get_dummies(input_data).reindex(columns=X.columns, fill_value=0)
    
    prediction = model.predict(scaler.transform(input_data))
    probability = model.predict_proba(scaler.transform(input_data))[0][1]
    
    return {
        "patient_id": patient_id,
        "sepsis_prediction": "Positive" if prediction[0] else "Negative",
        "probability": float(probability)
    }

if __name__ == "__main__":
    import uvicorn
    import sys
    import nest_asyncio

    # Port conflict check
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sock.connect_ex(('localhost', 8000)) == 0:
        print(f"Port 8000 already in use. Terminate existing process first.")
        sys.exit(1)

    # Run with import string format
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        server_header=False
    )
