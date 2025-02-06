FROM python:3.9-slim

WORKDIR /app

# Copy training script first
COPY train_model.py .
COPY Paitients_Files_Train.csv .

# Install dependencies and train model first
RUN pip install pandas scikit-learn imbalanced-learn joblib
RUN python train_model.py

# Now copy app files
COPY requirements.txt .
COPY main.py .

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]