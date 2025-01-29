FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create a directory for the data if it doesn't exist
RUN mkdir -p data

# Copy the data into the image.  Important: Be mindful of file sizes!
COPY data/Paitients_Files_Train.csv data/

# Expose the port the app runs on
EXPOSE 8000

# Start the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]