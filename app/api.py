"""
Flask API для сервиса прогнозирования дефолта по кредитным картам
Эндпоинты:
  GET /health - проверка работоспособности
  POST /predict - прогноз дефолта по признакам клиента
"""
import json
import logging
import sys
from datetime import datetime

from flask import Flask, request, jsonify

from app.model_handler import ModelHandler

# настройка логирования в JSON-формате
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}',
    datefmt='%Y-%m-%dT%H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# загружаем модель при старте
MODEL_V1_PATH = 'models/model_v1.pkl'
MODEL_V2_PATH = 'models/model_v2.pkl'

handlers = {}

try:
    handlers['v1'] = ModelHandler(MODEL_V1_PATH)
    logger.info('{"model": "v1", "status": "loaded"}')
except Exception as e:
    logger.error(json.dumps({"model": "v1", "status": "failed", "error": str(e)}))

# пробуем загрузить v2 (если есть)
try:
    handlers['v2'] = ModelHandler(MODEL_V2_PATH)
    logger.info('{"model": "v2", "status": "loaded"}')
except FileNotFoundError:
    logger.info('{"model": "v2", "status": "not_found", "message": "model_v2.pkl не найден, используется только v1"}')
except Exception as e:
    logger.error(json.dumps({"model": "v2", "status": "failed", "error": str(e)}))


@app.route('/health', methods=['GET'])
def health():
    """проверка работоспособности сервиса."""
    available_models = list(handlers.keys())
    status_data = {
        'status': 'healthy',
        'available_models': available_models,
        'timestamp': datetime.now(tz=None).isoformat(),
    }
    logger.info(json.dumps({"endpoint": "/health", "response": status_data}))
    return jsonify(status_data), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    прогноз дефолта по кредитной карте

    ожидаемый формат JSON
    {
        "LIMIT_BAL": 20000.0,
        "SEX": 2,
        "EDUCATION": 2,
        ...
    }

    query-параметры
        ?version=v1 - использовать модель v1 (по умолчанию)
        ?version=v2 - использовать модель v2 (если доступна)

    формат ответа
    {
        "prediction": 0,
        "probability": 0.1234,
        "model_version": "v1"
    }
    """
    # определяем версию модели
    model_version = request.args.get('version', 'v1')

    if model_version not in handlers:
        logger.warning(json.dumps({
            "endpoint": "/predict",
            "model_version": model_version,
            "error": "model_not_found",
        }))
        return jsonify({
            'error': f'Модель версии {model_version} не найдена',
            'available_versions': list(handlers.keys()),
        }), 404

    handler = handlers[model_version]

    # логируем входящий запрос
    logger.info(json.dumps({
        "endpoint": "/predict",
        "model_version": model_version,
        "request": request.get_json(silent=True),
    }))

    # получаем JSON из запроса
    try:
        data = request.get_json()
        if data is None:
            raise ValueError("тело запроса должно быть в формате JSON")
    except Exception as e:
        logger.error(json.dumps({"endpoint": "/predict", "error": str(e)}))
        return jsonify({'error': 'невалидный JSON в теле запроса'}), 400

    # предсказание
    try:
        prediction, probability = handler.predict(data)
    except ValueError as e:
        logger.error(json.dumps({"endpoint": "/predict", "error": str(e)}))
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(json.dumps({"endpoint": "/predict", "error": str(e)}))
        return jsonify({'error': 'внутренняя ошибка сервера'}), 500

    # ответ
    response = {
        'prediction': int(prediction),
        'probability': round(float(probability), 4),
        'model_version': model_version,
    }

    logger.info(json.dumps({"endpoint": "/predict", "response": response}))
    return jsonify(response), 200


if __name__ == '__main__':
    logger.info('{"event": "starting_server", "port": 5000}')
    app.run(host='0.0.0.0', port=5000, debug=False)