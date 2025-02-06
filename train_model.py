import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from joblib import dump

# Load data
train_data = pd.read_csv("Paitients_Files_Train.csv")

# Preprocess
# Fix column name typo and encode target
train_data['Sepsis'] = train_data['Sepssis'].apply(lambda x: 1 if x == 'Positive' else 0)
features = train_data.drop(['ID', 'Sepssis', 'Sepsis'], axis=1)
target = train_data['Sepsis']

# Handle missing values (example: fill with median)
features.fillna(features.median(), inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

# Train model
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_res, y_res)

# Save model and scaler
dump(model, 'sepsis_model.joblib')
dump(scaler, 'scaler.joblib')