"""
тесты для Flask API сервиса прогнозирования дефолта
проверяем эндпоинты /health и /predict
"""
import json
import pytest
from app.api import app


@pytest.fixture
def client():
    """создание тестового клиента Flask"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# тестовые данные (пример из датасета)
VALID_INPUT = {
    "LIMIT_BAL": 20000.0,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -2,
    "PAY_6": -2,
    "BILL_AMT1": 3913.0,
    "BILL_AMT2": 3102.0,
    "BILL_AMT3": 689.0,
    "BILL_AMT4": 0.0,
    "BILL_AMT5": 0.0,
    "BILL_AMT6": 0.0,
    "PAY_AMT1": 0.0,
    "PAY_AMT2": 689.0,
    "PAY_AMT3": 0.0,
    "PAY_AMT4": 0.0,
    "PAY_AMT5": 0.0,
    "PAY_AMT6": 0.0
}

INVALID_INPUT_MISSING = {
    "LIMIT_BAL": 20000.0,
    "SEX": 2
}

INVALID_INPUT_EXTRA = {
    **VALID_INPUT,
    "SOME_EXTRA_FEATURE": 123.0
}


class TestHealthEndpoint:
    """тесты для GET /health"""

    def test_health_returns_200(self, client):
        """/health возвращает 200"""
        response = client.get('/health')
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """/health возвращает JSON"""
        response = client.get('/health')
        assert response.is_json

    def test_health_contains_status(self, client):
        """ответ содержит 'status': 'healthy'"""
        response = client.get('/health')
        data = response.get_json()
        assert 'status' in data
        assert data['status'] == 'healthy'

    def test_health_contains_models(self, client):
        """в ответе список доступных моделей"""
        response = client.get('/health')
        data = response.get_json()
        assert 'available_models' in data
        assert 'v1' in data['available_models']


class TestPredictEndpoint:
    """тесты для POST /predict"""

    def test_predict_valid_input(self, client):
        """проверка успешного предсказания с валидными данными"""
        response = client.post(
            '/predict',
            data=json.dumps(VALID_INPUT),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.get_json()
        assert 'prediction' in data
        assert 'probability' in data
        assert 'model_version' in data
        assert data['prediction'] in [0, 1]
        assert 0.0 <= data['probability'] <= 1.0

    def test_predict_default_version_is_v1(self, client):
        """по умолчанию используется v1"""
        response = client.post(
            '/predict',
            data=json.dumps(VALID_INPUT),
            content_type='application/json'
        )
        data = response.get_json()
        assert data['model_version'] == 'v1'

    def test_predict_v1_explicit(self, client):
        """явный запрос версии v1"""
        response = client.post(
            '/predict?version=v1',
            data=json.dumps(VALID_INPUT),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data['model_version'] == 'v1'

    def test_predict_v2_not_found(self, client):
        """проверка ошибки при запросе несуществующей версии v2"""
        response = client.post(
            '/predict?version=v2',
            data=json.dumps(VALID_INPUT),
            content_type='application/json'
        )
        assert response.status_code == 404
        data = response.get_json()
        assert 'error' in data

    def test_predict_missing_features(self, client):
        """неполные данный"""
        response = client.post(
            '/predict',
            data=json.dumps(INVALID_INPUT_MISSING),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_predict_extra_features(self, client):
        """лишние признаки"""
        response = client.post(
            '/predict',
            data=json.dumps(INVALID_INPUT_EXTRA),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_predict_no_json(self, client):
        """неттела запроса"""
        response = client.post(
            '/predict',
            content_type='application/json'
        )
        assert response.status_code == 400

    def test_predict_invalid_json(self, client):
        """невалидный JSON"""
        response = client.post(
            '/predict',
            data='not a json',
            content_type='application/json'
        )
        assert response.status_code == 400