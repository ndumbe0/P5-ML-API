# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Make port 8000 available to the outside world
EXPOSE 8000

# Define environment variable
ENV NAME SepsisApp

# Run main.py when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
