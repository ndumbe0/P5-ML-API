# 🏥 Sepsis Prediction API

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/FastAPI-0.136-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/ML-Logistic%20Regression-orange" alt="ML Model">
  <img src="https://img.shields.io/badge/Tests-16%20Passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

<p align="center">
  <b>A production-grade machine learning API for predicting sepsis in patients.</b><br>
  Built with FastAPI, featuring Pydantic validation, API key authentication, batch predictions,<br>
  and optional AI-powered explanations via Google Gemini.
</p>

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|:------|:--------:|:---------:|:------:|:--------:|:-------:|
| **Logistic Regression** ⭐ | **72%** | **58%** | **71%** | **0.64** | **0.80** |
| Random Forest | 69% | 55% | 64% | 0.59 | 0.79 |
| XGBoost | 69% | 55% | 64% | 0.59 | 0.78 |
| LightGBM | 65% | 50% | 57% | 0.53 | 0.77 |
| Gradient Boosting | 66% | 51% | 52% | 0.52 | 0.75 |

> **Best Model:** Tuned Logistic Regression | **Best Params:** `{'C': 0.01, 'solver': 'lbfgs'}`

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                        User / Client                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│                   (localhost:8501)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │    Single    │  │    Batch     │  │  Model Info &    │  │
│  │  Prediction  │  │  Prediction  │  │  Explanations    │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend                             │
│                   (localhost:8000)                           │
│  ┌────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │   /health  │  │    /predict    │  │  /predict-batch  │   │
│  └────────────┘  └────────────────┘  └──────────────────┘   │
│  ┌────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │ /model-info    │  │ /gemini-explain  │  │   /docs     │ │
│  └────────────────┘  └──────────────────┘  └─────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    ML Model Layer                      │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐ │  │
│  │  │   Logistic  │  │    Model    │  │    Scaler     │ │  │
│  │  │  Regression │  │ (.joblib)   │  │   (.joblib)   │ │  │
│  │  └─────────────┘  └─────────────┘  └───────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Supabase    │
                    │  (Optional)   │
                    │  PostgreSQL   │
                    └───────────────┘
```

---

## ✨ Features

- ✅ **FastAPI** with auto-generated Swagger UI at `/docs`
- ✅ **Pydantic** input validation with detailed error messages
- ✅ **API key authentication** via `X-API-Key` header
- ✅ **`/health`** endpoint for monitoring
- ✅ **`/predict`** for single JSON predictions with confidence scores
- ✅ **`/predict-batch`** for bulk CSV/Excel predictions
- ✅ **`/model-info`** endpoint for model metadata
- ✅ **Optional Gemini explanations** via `?explain=true`
- ✅ **Proper HTTP status codes** and error handling
- ✅ **Multi-model training** (Logistic Regression, Random Forest, XGBoost, LightGBM, Gradient Boosting)
- ✅ **Hyperparameter tuning** with GridSearchCV
- ✅ **SMOTE** for class imbalance handling
- ✅ **Comprehensive pytest test suite** (16 tests)
- ✅ **GitHub Actions CI/CD**
- ✅ **Streamlit frontend** with confidence bars
- ✅ **Docker Compose** for easy deployment
- ✅ **Supabase deployment documentation**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip
- (Optional) Docker & Docker Compose

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/ndumbe0/P5-ML-API.git
cd P5-ML-API

# 2. Create virtual environment
python -m venv venv

# 3. Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Train the model (if not already done)
python train_model.py

# 6. Set environment variables (optional)
cp .env.example .env
# Edit .env with your API keys

# 7. Run the API
uvicorn main:app --reload --port 8000
```

### Access the API
- **API:** http://localhost:8000
- **Swagger Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### Run Tests
```bash
pytest test_api.py -v --tb=short
```

### Docker Setup
```bash
# Build and run all services
docker-compose up --build

# Access:
# API: http://localhost:8000
# Frontend: http://localhost:8501
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

### Example: Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: sk-sepsis-2024-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "PRG": 1.0,
    "PL": 85.0,
    "PR": 66.0,
    "SK": 29.0,
    "TS": 0.0,
    "M11": 26.6,
    "BD2": 0.351,
    "Age": 31.0,
    "Insurance": 0.0
  }'
```

### Example: Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict-batch" \
  -H "X-API-Key: sk-sepsis-2024-dev-key" \
  -F "file=@patients.csv"
```

### Example: Health Check

```bash
curl http://localhost:8000/health
```

---

## 📥 Input Schema

| Field | Type | Min | Max | Description |
|:------|:----:|:---:|:---:|:------------|
| `PRG` | float | 0 | - | Plasma glucose |
| `PL` | float | 0 | - | Blood pressure |
| `PR` | float | 0 | - | Diastolic blood pressure |
| `SK` | float | 0 | - | Skin thickness |
| `TS` | float | 0 | - | Insulin |
| `M11` | float | 0 | - | BMI |
| `BD2` | float | 0 | - | Diabetes pedigree function |
| `Age` | float | 0 | 120 | Age in years |
| `Insurance` | float | 0 | 1 | Insurance (0 or 1) |

## 📤 Response Schema

```json
{
  "prediction": "Positive",
  "probability": 0.7234,
  "confidence": "High",
  "features_used": {
    "PRG": 1.0,
    "PL": 85.0,
    "PR": 66.0,
    "SK": 29.0,
    "TS": 0.0,
    "M11": 26.6,
    "BD2": 0.351,
    "Age": 31.0,
    "Insurance": 0.0
  },
  "explanation": "Patient shows elevated risk factors..."
}
```

### Confidence Levels
| Level | Probability |
|:------|:-----------:|
| 🔴 **High Risk** | > 0.80 |
| 🟡 **Medium Risk** | 0.30 – 0.80 |
| 🟢 **Low Risk** | < 0.20 |

---

## 📁 Project Structure

```
P5-ML-API/
├── main.py                 # FastAPI application
├── train_model.py          # Model training script
├── test_api.py             # Pytest tests
├── requirements.txt        # Python dependencies
├── Dockerfile              # API container
├── docker-compose.yml      # Multi-service orchestration
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
├── sepsis_model.joblib     # Trained model
├── scaler.joblib           # Feature scaler
├── feature_columns.joblib  # Feature list
├── model_metrics.json      # Model performance metrics
├── data/
│   ├── Paitients_Files_Train.csv
│   └── Paitients_Files_Test.csv
├── frontend/
│   ├── app.py              # Streamlit frontend
│   ├── requirements.txt    # Frontend dependencies
│   └── Dockerfile          # Frontend container
├── .github/
│   └── workflows/
│       └── test.yml        # CI/CD pipeline
├── supabase/
│   └── README.md           # Supabase deployment guide
├── images/                 # Project images and plots
│   ├── API 1.png
│   ├── API 2.png
│   ├── API 3.png
│   ├── API 4.png
│   ├── confusion Matrix-Logistic Regression.png
│   ├── Confusion Matrix -Random Forest.png
│   ├── age distribution by sepsis status, train dataset.png
│   ├── joint plot of PL vs PR Train.png
│   └── Scatter plot matrix Train Datset.png
└── README.md               # This file
```

---

## 🔧 Environment Variables

| Variable | Description | Required |
|:---------|:------------|:---------:|
| `API_KEY` | API key for authentication | ✅ Yes |
| `GOOGLE_AI_API_KEY` | Google AI Studio key for Gemini explanations | ❌ No |

---

## 🧪 Model Evaluation

### Confusion Matrices

| Logistic Regression | Random Forest |
|:-------------------:|:-------------------:|
| ![LR Confusion Matrix](./images/confusion%20Matrix-Logistic%20Regression.png) | ![RF Confusion Matrix](./images/Confusion%20Matrix%20-Random%20Forest.png) |

### Data Exploration

| Age Distribution | Feature Correlations |
|:----------------:|:--------------------:|
| ![Age Distribution](./images/age%20distribution%20by%20sepsis%20status,%20train%20dataset.png) | ![Scatter Plot Matrix](./images/Scatter%20plot%20matrix%20Train%20Datset.png) |

| Joint Plot |
|:----------:|
| ![Joint Plot](./images/joint%20plot%20of%20PL%20vs%20PR%20Train.png) |

---

## 🖥️ Frontend Preview

The Streamlit frontend provides an intuitive interface for interacting with the API:

| Single Prediction | Batch Prediction | Model Info |
|:-----------------:|:----------------:|:----------:|
| ![API 1](./images/API%201.png) | ![API 2](./images/API%202.png) | ![API 3](./images/API%203.png) |

![API Overview](./images/API%204.png)

---

## 🚢 Deployment

### Docker Compose (Recommended)
```bash
docker-compose up --build -d
```

### Individual Services
```bash
# API only
docker build -t sepsis-api .
docker run -p 8000:8000 sepsis-api

# Frontend only
cd frontend
docker build -t sepsis-frontend .
docker run -p 8501:8501 sepsis-frontend
```

### Supabase Deployment
See [supabase/README.md](./supabase/README.md) for Supabase deployment options.

---

## 🛠️ Development

### Retrain Model
```bash
python train_model.py
```

### Run Tests
```bash
pytest test_api.py -v --tb=short
```

### Update Requirements
```bash
pip freeze > requirements.txt
```

---

## 🔄 CI/CD

GitHub Actions workflow runs on every push to `main`:

```yaml
# .github/workflows/test.yml
- Sets up Python 3.11 and 3.12
- Installs dependencies from requirements.txt
- Trains the ML model
- Runs pytest (16 tests)
```

---

## 📊 Dataset

The model is trained on the **Paitients_Files_Train.csv** dataset containing patient records with features like plasma glucose, blood pressure, BMI, age, and insurance status. The target variable is sepsis diagnosis (Positive/Negative).

- **Training samples:** 600
- **Features:** 9 numeric features
- **Target:** Binary classification (Positive/Negative)
- **Class distribution:** Imbalanced (handled with SMOTE)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Ndumbe Moses N.**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ using FastAPI, scikit-learn, and Streamlit
</p>
