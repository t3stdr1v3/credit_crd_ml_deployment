"""
скрипт для обучения модели v2 (GradientBoosting) для A/B-тестирования
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_PATH = 'models/model_v2.pkl'
METRICS_PATH = 'models/metrics_v2.json'

print("загрузка данных")
df = pd.read_csv('data/UCI_Credit_Card.csv')
df = df.drop('ID', axis=1)

target_col = 'default.payment.next.month'
X = df.drop(target_col, axis=1)
y = df[target_col]
feature_names = X.columns.tolist()

print(f"признаков = {len(feature_names)}, данных = {X.shape}")

# train/tes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# модель v2 GradientBoosting
model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    random_state=RANDOM_STATE
)
model.fit(X_train_scaled, y_train)

# оценка
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    'accuracy': round(accuracy_score(y_test, y_pred), 4),
    'f1_score': round(f1_score(y_test, y_pred), 4),
    'precision': round(precision_score(y_test, y_pred), 4),
    'recall': round(recall_score(y_test, y_pred), 4),
    'roc_auc': round(roc_auc_score(y_test, y_prob), 4)
}

print("\nметрики модели v2")
for name, value in metrics.items():
    print(f"{name}: {value}")

# сохраненяем
model_package = {
    'model': model,
    'scaler': scaler,
    'feature_names': feature_names
}
joblib.dump(model_package, MODEL_PATH)
print(f"\nмодель v2 сохранена в {MODEL_PATH}")

with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"метрики v2 сохранены в {METRICS_PATH}")
