"""
cкрипт для обучения модели прогнозирования дефолта по кредитным картам
pагружает датасет, обучает RandomForestClassifier и сохраняет модель.
"""
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, roc_auc_score, classification_report)

# константы
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_PATH = 'models/model_v1.pkl'
FEATURES_PATH = 'models/features.json'
METRICS_PATH = 'models/metrics_v1.json'

def load_and_preprocess_data(filepath='data/UCI_Credit_Card.csv'):
    """загрузка и предобработка данных"""
    df = pd.read_csv(filepath)
    
    # удаляем ID
    df = df.drop('ID', axis=1)
    
    # разделяем на признаки и таргет
    target_col = 'default.payment.next.month'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # сохраняем названия признаков 
    feature_names = X.columns.tolist()
    
    return X, y, feature_names

def train_model(X, y):
    """обучение модели с масштабированием"""
    # train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # масштабирование 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # обучение
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # предсказания
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # расчет метрик
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'f1_score': round(f1_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall': round(recall_score(y_test, y_pred), 4),
        'roc_auc': round(roc_auc_score(y_test, y_prob), 4)
    }
    
    print("\nметрики модели на тестовой выборке")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    
    print("\nдетальный отчет")
    print(classification_report(y_test, y_pred, target_names=['Не дефолт', 'Дефолт']))
    
    return model, scaler, metrics

def save_model(model, scaler, feature_names, metrics):
    """сохранение модели, скейлера и метаданных"""
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    joblib.dump(model_package, MODEL_PATH)
    print(f"\nмодель сохранена в {MODEL_PATH}")
    
    # сохраняем список признаков отдельно (для удобной проверки)
    with open(FEATURES_PATH, 'w') as f:
        json.dump({'feature_names': feature_names}, f, indent=2)
    print(f"список признаков сохранен в {FEATURES_PATH}")
    
    # сохраняем метрики
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"метрики сохранены в {METRICS_PATH}")

if __name__ == '__main__':
    print("загрузка данных")
    X, y, feature_names = load_and_preprocess_data()
    print(f"признаков: {len(feature_names)}")
    print(f"размер данных: {X.shape}")
    print(f"распределение классов:\n{y.value_counts()}")
    
    print("\nобучение модели")
    model, scaler, metrics = train_model(X, y)
    
    print("\nсохранение артефактов")
    save_model(model, scaler, feature_names, metrics)