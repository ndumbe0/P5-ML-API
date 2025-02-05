# Start with official Python base image
FROM python:3.11-slim

# Install system dependencies first
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies with explicit numpy version
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir numpy==1.26.4  # Force specific numpy version

# Copy application code
COPY . /app
WORKDIR /app

# Expose port and set entrypoint
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
