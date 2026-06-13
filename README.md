# Sepsis Prediction API

A production-grade machine learning API for predicting sepsis in patients using advanced ensemble models. Built with FastAPI, featuring Pydantic validation, API key authentication, batch predictions, and optional AI-powered explanations via Google Gemini.

## Architecture

```
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

## Project Structure

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
└── README.md               # This file
```

## Features

- ✅ FastAPI with auto-generated Swagger UI at `/docs`
- ✅ Pydantic input validation with detailed error messages
- ✅ API key authentication via `X-API-Key` header
- ✅ `/health` endpoint for monitoring
- ✅ `/predict` for single JSON predictions with confidence scores
- ✅ `/predict-batch` for bulk CSV/Excel predictions
- ✅ `/model-info` endpoint for model metadata
- ✅ Optional `/gemini-explain` via query parameter for AI explanations
- ✅ Proper HTTP status codes and error handling
- ✅ Multi-model training (Logistic Regression, Random Forest, XGBoost, LightGBM, Gradient Boosting)
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ SMOTE for class imbalance handling
- ✅ Comprehensive pytest test suite
- ✅ GitHub Actions CI/CD
- ✅ Streamlit frontend with confidence bars
- ✅ Docker Compose for easy deployment
- ✅ Supabase deployment documentation

## Model Performance

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Logistic Regression | 72% | 58% | 71% | 0.64 | 0.80 |
| Random Forest | 69% | 55% | 64% | 0.59 | 0.79 |
| XGBoost | 69% | 55% | 64% | 0.59 | 0.78 |
| LightGBM | 65% | 50% | 57% | 0.53 | 0.77 |
| Gradient Boosting | 66% | 51% | 52% | 0.52 | 0.75 |

**Best Model: Tuned Logistic Regression**
- Best params: `{'C': 0.01, 'solver': 'lbfgs'}`
- Final ROC AUC: 0.80

## Setup & Installation

### Prerequisites
- Python 3.11+
- pip
- (Optional) Docker & Docker Compose

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ndumbe0/P5-ML-API.git
   cd P5-ML-API
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Train the model** (if not already done)
   ```bash
   python train_model.py
   ```

6. **Set environment variables** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

7. **Run the API**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

8. **Access the API**
   - API: http://localhost:8000
   - Swagger Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Run Tests

```bash
pytest test_api.py -v
```

### Docker Setup

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access services**
   - API: http://localhost:8000
   - Frontend: http://localhost:8501

### Streamlit Frontend (Local)

```bash
streamlit run frontend/app.py
```

## API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | No | API info page |
| `/health` | GET | No | Health check |
| `/docs` | GET | No | Swagger UI |
| `/predict` | POST | Yes | Single prediction (JSON body) |
| `/predict?explain=true` | POST | Yes | Single prediction with Gemini explanation |
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

### Example: With Explanation

```bash
curl -X POST "http://localhost:8000/predict?explain=true" \
  -H "X-API-Key: sk-sepsis-2024-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "PRG": 6.0, "PL": 148.0, "PR": 72.0, "SK": 35.0,
    "TS": 0.0, "M11": 33.6, "BD2": 0.627,
    "Age": 50.0, "Insurance": 0.0
  }'
```

### Example: Health Check

```bash
curl http://localhost:8000/health
```

### Example: Model Info

```bash
curl http://localhost:8000/model-info \
  -H "X-API-Key: sk-sepsis-2024-dev-key"
```

## Input Schema

| Field | Type | Min | Max | Description |
|-------|------|-----|-----|-------------|
| PRG | float | 0 | - | Plasma glucose |
| PL | float | 0 | - | Blood pressure |
| PR | float | 0 | - | Diastolic blood pressure |
| SK | float | 0 | - | Skin thickness |
| TS | float | 0 | - | Insulin |
| M11 | float | 0 | - | BMI |
| BD2 | float | 0 | - | Diabetes pedigree function |
| Age | float | 0 | 120 | Age in years |
| Insurance | float | 0 | 1 | Insurance (0 or 1) |

## Response Schema

```json
{
  "prediction": "Positive",
  "probability": 0.7234,
  "confidence": "High",
  "features_used": {
    "PRG": 1.0,
    "PL": 85.0,
    ...
  },
  "explanation": "Patient shows elevated risk factors..."
}
```

### Confidence Levels
- **High**: Probability > 0.8 or < 0.2
- **Medium**: Probability between 0.3 and 0.7
- **Low**: Probability between 0.2 and 0.3 or 0.7 and 0.8

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_KEY` | API key for authentication | Yes |
| `GOOGLE_AI_API_KEY` | Google AI Studio API key for Gemini explanations | No |

## CI/CD

GitHub Actions workflow (`.github/workflows/test.yml`) runs on every push:
1. Sets up Python 3.11 and 3.12
2. Installs dependencies
3. Trains the model
4. Runs pytest

## Deployment

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
See [supabase/README.md](supabase/README.md) for Supabase deployment options.

## Development

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

## Author

Ndumbe Moses N.

## License

MIT License
