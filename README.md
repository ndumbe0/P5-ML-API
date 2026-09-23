# 🚀 Sepsis Prediction API

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-24.0%2B-2496ED?style=for-the-badge&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Short Summary:** A production-grade machine learning API for predicting sepsis in patients, built with FastAPI, featuring Pydantic validation, API key authentication, batch predictions, and optional Gemini AI explanations. Includes a Streamlit frontend for user-friendly interaction.

---

## 📌 Executive Summary & Business Impact

* **The Problem:** Sepsis is a life-threatening condition that requires early detection. Healthcare providers need a reliable, ML-powered tool to assess sepsis risk quickly and accurately from patient biomarkers.
* **The Solution:** A FastAPI backend serving 5 trained ML models (Logistic Regression, Random Forest, XGBoost, LightGBM, Gradient Boosting) with hyperparameter tuning and SMOTE balancing, plus a Streamlit frontend for bedside prediction with AI-generated explanations.
* **Key Metrics & Results:**
  * **Best Model:** Tuned Logistic Regression — Accuracy ≈ 72%, F1 ≈ 0.64, ROC AUC ≈ 0.80
  * **API:** 16 passing tests, Swagger/OpenAPI docs, health monitoring
  * **Frontend:** Streamlit UI with confidence bars and batch CSV upload

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    A[Patient Data CSV] --> B[train_model.py<br/>SMOTE + GridSearchCV]
    B --> C[Trained Model<br/>.joblib]
    C --> D[FastAPI Backend<br/>main.py :8000]
    D --> E[/health / /predict<br/>/predict-batch<br/>/model-info]
    D --> F[Gemini AI<br/>Explanations]
    D --> G[Streamlit Frontend<br/>frontend/app.py :8501]
    G --> D
```

---

## 🛠️ Tech Stack & Key Tools

* **Core Language:** Python 3.10+
* **ML Framework:** Scikit-learn, XGBoost, LightGBM, Imbalanced-learn (SMOTE)
* **API Framework:** FastAPI with OpenAPI/Swagger docs
* **Frontend:** Streamlit
* **Testing:** Pytest (16 tests)
* **AI / LLM:** Google Generative AI (Gemini 2.0 Flash, optional)
* **Deployment & Containerization:** Docker, Docker Compose
* **CI/CD:** GitHub Actions

---

## 📂 Repository Directory Structure

```text
P5-ML-API/
├── main.py                  # FastAPI application (backend)
├── train_model.py           # Model training pipeline with SMOTE + GridSearchCV
├── test_api.py              # Pytest test suite (16 tests)
├── requirements.txt         # Python dependencies (pinned)
├── Dockerfile               # Backend API container
├── docker-compose.yml       # Multi-service orchestration
├── .env.example             # Environment template
├── .gitignore               # Git ignore rules
├── .dockerignore            # Docker build ignore rules
├── pytest.ini               # Pytest configuration
├── LICENSE                  # MIT License
├── README.md                # This file
│
├── data/                    # Training and test datasets
│   ├── Paitients_Files_Train.csv
│   └── Paitients_Files_Test.csv
│
├── frontend/                # Streamlit frontend
│   ├── app.py               # Streamlit UI
│   ├── Dockerfile           # Frontend container
│   └── requirements.txt     # Frontend dependencies
│
├── models/                  # Trained model artifacts (gitignored)
│   ├── funding_model.joblib
│   ├── scaler.joblib
│   └── feature_columns.joblib
│
├── .github/workflows/       # CI/CD pipelines
│   ├── test.yml
│   └── docker-publish.yml
│
├── supabase/                # Supabase deployment docs
│   └── README.md
└── images/                  # Project screenshots and plots
```

---

## ⚙️ Quickstart & Local Setup Guide

### Local Python Environment Setup

```bash
git clone https://github.com/ndumbe0/P5-ML-API.git
cd P5-ML-API
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the API
uvicorn main:app --reload --port 8000
```

### Docker Setup

```bash
# Build and run all services
docker-compose up --build -d

# Access:
# API:         http://localhost:8000
# Swagger UI:  http://localhost:8000/docs
# Frontend:    http://localhost:8501
```

### API Key Setup

```bash
cp .env.example .env
# Edit .env and set a secure API_KEY
```

---

## 🛡️ Security & Quality Standards

* **API Key Authentication:** All prediction endpoints require `X-API-Key` header.
* **Secrets Management:** API keys and credentials loaded from `.env` via `python-dotenv`, never hardcoded.
* **Input Validation:** Pydantic schemas enforce field types, ranges, and business rules.
* **CORS Protection:** Restricted to known origins (`localhost:8501`, `localhost:3000`).
* **Non-Root Execution:** Both containers run as non-root `appuser`.
* **Dependency Pinning:** All packages have upper-bound version constraints.
* **Error Handling:** `try/except` blocks with structured logging on all endpoints.

---

## 🧪 Testing

```bash
pip install pytest
pytest test_api.py -v --tb=short
```

Run tests with `API_KEY` set in your environment:

```bash
export API_KEY=REPLACE_WITH_YOUR_API_KEY
pytest test_api.py -v
```

---

## 📡 API Endpoints

| Endpoint | Method | Auth | Description |
|:---------|:------:|:----:|:------------|
| `/` | GET | No | API info page |
| `/health` | GET | No | Health check with model status |
| `/docs` | GET | No | Swagger UI documentation |
| `/predict` | POST | Yes | Single prediction (JSON body) |
| `/predict?explain=true` | POST | Yes | Single prediction + Gemini explanation |
| `/predict-batch` | POST | Yes | Batch prediction (CSV/Excel upload) |
| `/model-info` | GET | Yes | Model metadata and metrics |

---

## 👤 Author & Contact

* **GitHub:** [@ndumbe0](https://github.com/ndumbe0)
* **Email:** ndumbemoses@gmail.com
* **Team Lead:** Ms. Portia Bentum
* **Organization:** Azubi Africa

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
