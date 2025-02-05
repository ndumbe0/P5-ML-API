import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load data
data = pd.read_csv(os.path.join(DATA_DIR, "Paitients_Files_Train.csv"))
X = data.drop(['ID', 'Sepssis', 'Insurance'], axis=1)
y = data['Sepssis']

# Preprocessing
X = pd.get_dummies(X).fillna(X.mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save artifacts
joblib.dump(model, 'sepsis_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(X.columns, 'feature_columns.joblib')