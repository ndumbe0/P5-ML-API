import os
import io
import json
import time
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib
import numpy as np
from dotenv import load_dotenv

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.getenv("API_KEY", "sk-sepsis-2024-dev-key")
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY", "")

if GEMINI_AVAILABLE and GOOGLE_AI_API_KEY:
    genai.configure(api_key=GOOGLE_AI_API_KEY)

app = FastAPI(
    title="Sepsis Prediction API",
    description="Advanced ML API for sepsis prediction with confidence scores and AI explanations",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sepsis_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

model = None
scaler = None
expected_features = None
model_metrics = None


def load_ml_assets():
    global model, scaler, expected_features, model_metrics
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        if os.path.exists(FEATURES_PATH):
            expected_features = joblib.load(FEATURES_PATH)
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                model_metrics = json.load(f)
        logger.info("ML assets loaded successfully")
        return model, scaler, expected_features
    except Exception as e:
        logger.error(f"Error loading ML assets: {e}")
        return None, None, None


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


class PatientData(BaseModel):
    PRG: float = Field(..., ge=0, description="Plasma glucose")
    PL: float = Field(..., ge=0, description="Blood pressure")
    PR: float = Field(..., ge=0, description="Diastolic blood pressure")
    SK: float = Field(..., ge=0, description="Skin thickness")
    TS: float = Field(..., ge=0, description="Insulin")
    M11: float = Field(..., ge=0, description="BMI")
    BD2: float = Field(..., ge=0, description="Diabetes pedigree function")
    Age: float = Field(..., ge=0, le=120, description="Age in years")
    Insurance: float = Field(..., ge=0, le=1, description="Insurance (0 or 1)")

    @validator('Insurance')
    def validate_insurance(cls, v):
        if v not in [0, 1]:
            raise ValueError('Insurance must be 0 or 1')
        return v


class BatchPredictionRequest(BaseModel):
    patients: List[PatientData]


class PredictionResponse(BaseModel):
    patient_id: Optional[str] = None
    prediction: str
    probability: float
    confidence: str
    features_used: Dict[str, Any]
    explanation: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    timestamp: str


@app.on_event("startup")
async def startup_event():
    global model, scaler, expected_features
    load_ml_assets()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_type = type(model).__name__ if model else None
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_type=model_type,
        timestamp=pd.Timestamp.now().isoformat()
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>Sepsis Prediction API</title></head>
        <body>
            <h1>Sepsis Prediction API v2.0</h1>
            <p>Status: <strong>Running</strong></p>
            <p>Documentation: <a href="/docs">/docs</a></p>
            <p>Health Check: <a href="/health">/health</a></p>
        </body>
    </html>
    """


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    patient: PatientData,
    x_api_key: str = Depends(verify_api_key),
    explain: bool = Query(False, description="Include Gemini explanation")
):
    if model is None or scaler is None or expected_features is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")

    try:
        input_data = pd.DataFrame([patient.dict()])
        for col in expected_features:
            if col not in input_data.columns:
                input_data[col] = 0
        input_df = input_data[expected_features]
        input_df = input_df.fillna(input_df.median())
        scaled_features = scaler.transform(input_df)
        prediction = model.predict(scaled_features)[0]
        probability = float(model.predict_proba(scaled_features)[0, 1])
        confidence = "High" if probability > 0.8 or probability < 0.2 else "Medium" if 0.3 < probability < 0.7 else "Low"
        result_label = "Positive" if prediction == 1 else "Negative"
        explanation = None
        if explain and GEMINI_AVAILABLE and GOOGLE_AI_API_KEY:
            try:
                explanation = await get_gemini_explanation(patient.dict(), probability, result_label)
            except Exception as e:
                logger.warning(f"Gemini explanation failed: {e}")
                explanation = "Explanation unavailable"
        return PredictionResponse(
            prediction=result_label,
            probability=round(probability, 4),
            confidence=confidence,
            features_used=patient.dict(),
            explanation=explanation
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(
    file: UploadFile = File(...),
    x_api_key: str = Depends(verify_api_key)
):
    if model is None or scaler is None or expected_features is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        content = await file.read()
        suffix = file.filename.lower()
        if suffix.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif suffix.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Only .csv or .xlsx files supported")

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        patient_ids = df["ID"].tolist() if "ID" in df.columns else [f"row_{i}" for i in range(len(df))]
        features_df = df.copy()
        cols_to_drop = ["ID", "Sepssis", "Sepsis"]
        for col in cols_to_drop:
            if col in features_df.columns:
                features_df = features_df.drop(columns=[col])
        for col in expected_features:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[expected_features]
        features_df = features_df.fillna(features_df.median())
        scaled_features = scaler.transform(features_df)
        predictions = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)[:, 1]
        results = []
        for i, p_id in enumerate(patient_ids):
            prob = float(probabilities[i])
            confidence = "High" if prob > 0.8 or prob < 0.2 else "Medium" if 0.3 < prob < 0.7 else "Low"
            results.append({
                "patient_id": str(p_id),
                "prediction": "Positive" if predictions[i] == 1 else "Negative",
                "probability": round(prob, 4),
                "confidence": confidence
            })
        return {"message": "Batch prediction successful", "count": len(results), "results": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model-info")
async def model_info(x_api_key: str = Depends(verify_api_key)):
    if model_metrics is None:
        raise HTTPException(status_code=404, detail="No model metrics available")
    return {
        "model_type": model_metrics.get("best_model"),
        "accuracy": model_metrics.get("accuracy"),
        "precision": model_metrics.get("precision"),
        "recall": model_metrics.get("recall"),
        "f1_score": model_metrics.get("f1"),
        "roc_auc": model_metrics.get("roc_auc"),
        "feature_count": model_metrics.get("feature_count"),
        "features": model_metrics.get("feature_columns"),
        "best_params": model_metrics.get("best_params"),
    }


async def get_gemini_explanation(features: dict, probability: float, prediction: str) -> str:
    if not GEMINI_AVAILABLE or not GOOGLE_AI_API_KEY:
        return "Gemini API not configured"
    try:
        model_gen = genai.GenerativeModel("gemini-2.0-flash")
        feature_str = ", ".join([f"{k}: {v}" for k, v in features.items()])
        prompt = (
            f"As a medical AI assistant, explain this sepsis prediction:\n"
            f"Patient features: {feature_str}\n"
            f"Prediction: {prediction} (probability: {probability:.2%})\n"
            f"Provide a concise, clear explanation (max 3 sentences) of what this means for the patient."
        )
        response = model_gen.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return "Explanation generation failed"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
