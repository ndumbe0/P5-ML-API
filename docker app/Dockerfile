# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Create data directory for persistent storage
RUN mkdir -p /app/data

# Copy initial training data
COPY data/Paitients_Files_Train.csv /app/data/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
