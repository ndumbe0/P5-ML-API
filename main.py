import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib
from io import StringIO

# Load and preprocess the data
data = pd.read_csv("F:\\school\\Azubi Africa\\P5-ML-API\\data\\Paitients_Files_Train.csv")
X = data.drop(['ID', 'Sepssis', 'Insurance'], axis=1)
y = data['Sepssis']

# Handle missing values and encode categorical variables
X = pd.get_dummies(X)
X = X.fillna(X.mean())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'sepsis_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Create FastAPI app
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

# Global variable to store uploaded data
uploaded_data = None

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global uploaded_data
    contents = await file.read()
    s = str(contents, 'utf-8')
    data = StringIO(s)
    uploaded_data = pd.read_csv(data)
    return {"message": "File uploaded successfully"}

@app.get("/patient/{patient_id}")
async def get_patient_data(patient_id: str):
    global uploaded_data
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    patient = uploaded_data[uploaded_data['ID'] == patient_id]
    if patient.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return {
        "ID": patient_id,
        "PRG": float(patient['PRG'].values[0]),
        "PL": float(patient['PL'].values[0]),
        "PR": float(patient['PR'].values[0]),
        "SK": float(patient['SK'].values[0]),
        "TS": float(patient['TS'].values[0]),
        "M11": float(patient['M11'].values[0]),
        "BD2": float(patient['BD2'].values[0]),
        "Age": int(patient['Age'].values[0])
    }

@app.post("/predict/{patient_id}")
async def predict_sepsis(patient_id: str):
    global uploaded_data
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    patient = uploaded_data[uploaded_data['ID'] == patient_id]
    if patient.empty:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Load the saved model and scaler
    model = joblib.load('sepsis_model.joblib')
    scaler = joblib.load('scaler.joblib')
    
    # Prepare input data
    input_data = patient.drop(['ID', 'Sepssis', 'Insurance'], axis=1)
    
    # Preprocess the input data
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1]
    
    return {
        "patient_id": patient_id,
        "sepsis_prediction": "Positive" if prediction[0] == 1 else "Negative",
        "probability": float(probability)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
