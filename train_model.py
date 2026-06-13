import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from joblib import dump

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Paitients_Files_Train.csv")

print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

df['Sepsis'] = df['Sepssis'].apply(lambda x: 1 if str(x).strip() == 'Positive' else 0)
features = df.drop(['ID', 'Sepssis', 'Sepsis'], axis=1)
target = df['Sepsis']

feature_cols = features.columns.tolist()
dump(feature_cols, os.path.join(BASE_DIR, 'feature_columns.joblib'))

features.fillna(features.median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

print("\nTraining models...\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }
    print(f"{name}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")

best_name = max(results, key=lambda k: results[k]['f1'])
best_model = results[best_name]['model']
print(f"\nBest model: {best_name} (F1={results[best_name]['f1']:.4f})")

if best_name == "Logistic Regression":
    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs']}
elif best_name in ["Random Forest", "Gradient Boosting"]:
    param_grid = {'n_estimators': [200, 300], 'max_depth': [5, 10, None]}
elif best_name == "XGBoost":
    param_grid = {'n_estimators': [200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1]}
elif best_name == "LightGBM":
    param_grid = {'n_estimators': [200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1]}

print(f"\nTuning {best_name}...")
grid = GridSearchCV(best_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_res, y_res)
best_model = grid.best_estimator_
print(f"Best params: {grid.best_params_}")

y_pred = best_model.predict(X_test_scaled)
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
print("\nFinal Evaluation:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

dump(best_model, os.path.join(BASE_DIR, 'sepsis_model.joblib'))
dump(scaler, os.path.join(BASE_DIR, 'scaler.joblib'))

metrics = {
    "best_model": best_name,
    "best_params": grid.best_params_,
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_prob),
    "feature_count": len(feature_cols),
    "feature_columns": feature_cols,
}
with open(os.path.join(BASE_DIR, 'model_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nModel, scaler, feature columns, and metrics saved successfully.")
