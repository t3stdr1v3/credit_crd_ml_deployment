"""
модуль для загрузки модели и выполнения предсказаний
поддерживает загрузку модели и скейлера из упакованного .pkl файла
"""
import json
import logging
import numpy as np
import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class ModelHandler:
    """
    класс для загрузки, валидации входных данных и инференса модели
    """

    def __init__(self, model_path: str):
        """
        загрузка модели из файла.

        args:
            model_path: путь к .pkl файлу с моделью и скейлером
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = []

        self._load_model()

    def _load_model(self):
        """загрузка модели и метаданных из файла"""
        logger.info(json.dumps({
            "event": "loading_model",
            "path": self.model_path,
        }))

        package = joblib.load(self.model_path)

        self.model = package['model']
        self.scaler = package['scaler']
        self.feature_names = package['feature_names']

        logger.info(json.dumps({
            "event": "model_loaded",
            "path": self.model_path,
            "features_count": len(self.feature_names),
        }))

    def validate_input(self, data: dict) -> None:
        """
        валидация входного JSON.

        Args:
            data: словарь с признаками клиента

        Raises:
            ValueError: если признаки отсутствуют или есть лишние
        """
        input_features = set(data.keys())
        expected_features = set(self.feature_names)

        missing = expected_features - input_features
        extra = input_features - expected_features

        errors = []
        if missing:
            errors.append(f"отсутствуют признаки: {sorted(missing)}")
        if extra:
            errors.append(f"лишние признаки: {sorted(extra)}")

        if errors:
            raise ValueError("; ".join(errors))

    def preprocess(self, data: dict) -> np.ndarray:
        """
        преобразование входного словаря в масштабированный numpy-массив

        Args:
            data: словарь с признаками клиента

        Returns:
            np.ndarray формы (1, n_features)
        """
        # извлекаем признаки
        features_df = pd.DataFrame([data], columns=self.feature_names)

        # масштабируем
        features_scaled = self.scaler.transform(features_df)

        return features_scaled

    def predict(self, data: dict) -> tuple:
        """
        предсказание класса и вероятности дефолта

        Args:
            data: словарь с признаками клиента

        Returns:
            tuple: (prediction: int, probability: float)
        """
        # Валидация
        self.validate_input(data)

        # Предобработка
        features_scaled = self.preprocess(data)

        # Предсказание
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]

        return prediction, probability