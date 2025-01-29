# Sepsis Prediction API

## Introduction

This project implements a machine learning model to predict sepsis in patients using a Random Forest Classifier. The model is deployed as a FastAPI application, allowing users to upload patient data, retrieve patient information, and make sepsis predictions through API endpoints.

## Table of Contents

1. [Setup](#setup)
2. [Usage](#usage)
3. [API Endpoints](#api-endpoints)
4. [Author](#author)
5. [Conclusion](#conclusion)

## Setup

To set up and run this project, follow these steps:

1. Install the required dependencies:
   ```
   pip install pandas scikit-learn fastapi uvicorn joblib
   ```

2. Ensure you have the training data file `Paitients_Files_Train.csv` in the correct directory.

3. Run the script to train the model and start the FastAPI server:
   ```
   python main.py
   ```

## Usage

The application provides three main functionalities:

1. Upload patient data via CSV file
2. Retrieve patient information by ID
3. Predict sepsis for a specific patient

To use the API, you can make HTTP requests to the appropriate endpoints using tools like cURL or Postman.[1]

## API Endpoints

1. **Upload Data**
   - Endpoint: `/upload`
   - Method: POST
   - Description: Upload a CSV file containing patient data

2. **Get Patient Data**
   - Endpoint: `/patient/{patient_id}`
   - Method: GET
   - Description: Retrieve data for a specific patient by ID

3. **Predict Sepsis**
   - Endpoint: `/predict/{patient_id}`
   - Method: POST
   - Description: Make a sepsis prediction for a specific patient by ID[1]

## Author

[Ndumbe Moses N.]

## App Link

(http://127.0.0.1:8000/docs#/)

## Conclusion

This project demonstrates the implementation of a machine learning model for sepsis prediction, deployed as a user-friendly API. The Random Forest Classifier is trained on patient data and can provide predictions on new patient records. The FastAPI framework enables easy data upload, patient information retrieval, and sepsis predictions through well-defined endpoints.

The application showcases the integration of data preprocessing, model training, and API development, providing a complete solution for sepsis prediction in a clinical setting. Future improvements could include model performance optimization, additional data validation, and enhanced error handling.[1]

