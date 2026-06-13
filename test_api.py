import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app, load_ml_assets

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_models():
    load_ml_assets()

@pytest.fixture
def api_key_header():
    return {"X-API-Key": "sk-sepsis-2024-dev-key"}

@pytest.fixture
def sample_patient():
    return {
        "PRG": 1.0,
        "PL": 85.0,
        "PR": 66.0,
        "SK": 29.0,
        "TS": 0.0,
        "M11": 26.6,
        "BD2": 0.351,
        "Age": 31.0,
        "Insurance": 0.0,
    }

@pytest.fixture
def invalid_patient():
    return {
        "PRG": -1.0,
        "PL": 85.0,
        "PR": 66.0,
        "SK": 29.0,
        "TS": 0.0,
        "M11": 26.6,
        "BD2": 0.351,
        "Age": 31.0,
        "Insurance": 2.0,
    }

class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "timestamp" in data

class TestPredictEndpoint:
    def test_predict_success(self, api_key_header, sample_patient):
        response = client.post("/predict", json=sample_patient, headers=api_key_header)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in ["Positive", "Negative"]
        assert "probability" in data
        assert 0 <= data["probability"] <= 1
        assert "confidence" in data
        assert "features_used" in data

    def test_predict_invalid_api_key(self, sample_patient):
        response = client.post("/predict", json=sample_patient, headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401

    def test_predict_missing_api_key(self, sample_patient):
        response = client.post("/predict", json=sample_patient)
        assert response.status_code == 401

    def test_predict_invalid_input(self, api_key_header, invalid_patient):
        response = client.post("/predict", json=invalid_patient, headers=api_key_header)
        assert response.status_code == 422

    def test_predict_missing_field(self, api_key_header):
        response = client.post("/predict", json={"PRG": 1.0}, headers=api_key_header)
        assert response.status_code == 422

    def test_predict_with_explanation(self, api_key_header, sample_patient):
        with patch("main.GEMINI_AVAILABLE", True), \
             patch("main.GOOGLE_AI_API_KEY", "test-key"), \
             patch("google.generativeai.GenerativeModel") as mock_gen:
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Patient shows elevated risk factors."
            mock_model.generate_content.return_value = mock_response
            mock_gen.return_value = mock_model
            response = client.post("/predict?explain=true", json=sample_patient, headers=api_key_header)
            assert response.status_code == 200
            data = response.json()
            assert "explanation" in data

class TestBatchEndpoint:
    @pytest.fixture
    def csv_file(self):
        csv_content = b"ID,PRG,PL,PR,SK,TS,M11,BD2,Age,Insurance\n1,1.0,85.0,66.0,29.0,0.0,26.6,0.351,31.0,0.0\n2,8.0,183.0,64.0,0.0,0.0,23.3,0.672,32.0,1.0\n"
        return ("test.csv", csv_content, "text/csv")

    def test_batch_predict_success(self, api_key_header, csv_file):
        response = client.post(
            "/predict-batch",
            files={"file": csv_file},
            headers=api_key_header,
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["count"] == 2
        assert len(data["results"]) == 2
        for result in data["results"]:
            assert "patient_id" in result
            assert "prediction" in result
            assert "probability" in result
            assert "confidence" in result

    def test_batch_predict_invalid_file_type(self, api_key_header):
        response = client.post(
            "/predict-batch",
            files={"file": ("test.txt", b"hello", "text/plain")},
            headers=api_key_header,
        )
        assert response.status_code == 400

    def test_batch_predict_invalid_api_key(self, csv_file):
        response = client.post(
            "/predict-batch",
            files={"file": csv_file},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

class TestModelInfo:
    def test_model_info(self, api_key_header):
        response = client.get("/model-info", headers=api_key_header)
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "accuracy" in data
        assert "f1_score" in data
        assert "features" in data

    def test_model_info_invalid_key(self):
        response = client.get("/model-info", headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

class TestSchemaValidation:
    def test_insurance_zero(self, api_key_header):
        patient = {
            "PRG": 1.0, "PL": 85.0, "PR": 66.0, "SK": 29.0,
            "TS": 0.0, "M11": 26.6, "BD2": 0.351,
            "Age": 31.0, "Insurance": 0.0,
        }
        response = client.post("/predict", json=patient, headers=api_key_header)
        assert response.status_code == 200

    def test_insurance_one(self, api_key_header):
        patient = {
            "PRG": 1.0, "PL": 85.0, "PR": 66.0, "SK": 29.0,
            "TS": 0.0, "M11": 26.6, "BD2": 0.351,
            "Age": 31.0, "Insurance": 1.0,
        }
        response = client.post("/predict", json=patient, headers=api_key_header)
        assert response.status_code == 200

    def test_age_out_of_range(self, api_key_header):
        patient = {
            "PRG": 1.0, "PL": 85.0, "PR": 66.0, "SK": 29.0,
            "TS": 0.0, "M11": 26.6, "BD2": 0.351,
            "Age": 150.0, "Insurance": 0.0,
        }
        response = client.post("/predict", json=patient, headers=api_key_header)
        assert response.status_code == 422

class TestRootEndpoint:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "Sepsis Prediction API" in response.text
