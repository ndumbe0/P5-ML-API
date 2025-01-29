# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.8
FROM mcr.microsoft.com/windows/servercore:ltsc2022 AS windows_base 

FROM python:${PYTHON_VERSION}-slim-windowsservercore-ltsc2022 AS base



# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run your FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]