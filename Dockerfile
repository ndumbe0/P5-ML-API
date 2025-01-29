FROM mcr.microsoft.com/windows/servercore:ltsc2022 AS base

# Install Python
SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop';"]

ARG PYTHON_VERSION=3.12.7

WORKDIR C:\

# Download and install Python
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-embed-amd64.zip" -OutFile "python.zip"

Expand-Archive -Path "python.zip" -DestinationPath "."

# Add Python to the PATH
$env:Path = "$env:Path;C:\python-${PYTHON_VERSION}-embed-amd64"
[Environment]::SetEnvironmentVariable("Path", $env:Path, "Machine")

# Clean up
Remove-Item "python.zip"

SHELL ["cmd", "/S", "/C"]

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# *** KEY CHANGE: Fix the file path issue ***
# Create the necessary directory inside the container
WORKDIR /app  # Or wherever your app expects the data
RUN mkdir -p data  # Create the 'data' directory

# Copy the data file *into* the image
COPY data/Paitients_Files_Train.csv /app/data/Paitients_Files_Train.csv # Copy the file

# Set the working directory
WORKDIR /app

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]