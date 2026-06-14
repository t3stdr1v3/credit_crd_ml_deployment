# Credit Card Default ML Service
Сервис машинного обучения для прогнозирования дефолта по кредитным картам.
Реализован полный цикл: от обучения модели до контейнеризации и A/B-тестирования.

## Структура проекта
credit-card-ml-deployment/
```
├── app/
│ ├── init.py # Инициализация модуля
│ ├── api.py # Flask-приложение (эндпоинты /health, /predict)
│ └── model_handler.py # Загрузка модели, валидация, инференс
├── models/
│ ├── train_model.py # Скрипт обучения модели v1 (RandomForest)
│ ├── train_model_v2.py # Скрипт обучения модели v2 (GradientBoosting)
│ ├── model_v1.pkl # Сохранённая модель v1
│ ├── model_v2.pkl # Сохранённая модель v2
│ ├── features.json # Список признаков
│ ├── metrics_v1.json # Метрики модели v1
│ └── metrics_v2.json # Метрики модели v2
├── tests/
│ ├── init.py
│ └── test_api.py # Тесты API (12 тестов)
├── docker/
│ ├── Dockerfile # Конфигурация Docker-образа
│ └── nginx.conf # Конфиг nginx для балансировки
├── docker-compose.yml # Оркестрация сервисов
├── requirements.txt # Зависимости Python
├── ARCHITECTURE.md # Архитектурные решения и MLOps-концепты
├── ab_test_plan.md # План A/B-тестирования
└── README.md # Документация
```


## Датасет

**Default of Credit Card Clients Dataset** (UCI Machine Learning)

- 30 000 клиентов, 23 признака
- Таргет: `default.payment.next.month` (дефолт в следующем месяце, 0/1)
- Доля дефолтов: ~22.1%

Признаки: кредитный лимит, пол, образование, брак, возраст, история платежей (PAY_0–PAY_6), суммы счетов (BILL_AMT1–BILL_AMT6), суммы платежей (PAY_AMT1–PAY_AMT6).

## Модели
```
| Версия | Алгоритм | F1-score | Precision | Recall | ROC-AUC |
| v1 | RandomForest (100 деревьев, max_depth=10) | 0.4592 | 0.6605 | 0.3519 | 0.7727 |
| v2 | GradientBoosting (150 деревьев, max_depth=5) | 0.4636 | 0.6579 | 0.3580 | 0.7780 |
```

## Эндпоинты API

### `GET /health` — Проверка работоспособности

**Ответ (200):**
```
json
{
    "status": "healthy",
    "available_models": ["v1", "v2"],
    "timestamp": "2026-06-14T10:32:09"
} 
```

### POST /predict — Прогноз дефолта
Query-параметры:

?version=v1 — модель v1 (по умолчанию)

?version=v2 — модель v2

Тело запроса (JSON):
```
json
{
    "LIMIT_BAL": 20000.0,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913.0, "BILL_AMT2": 3102.0, "BILL_AMT3": 689.0,
    "BILL_AMT4": 0.0, "BILL_AMT5": 0.0, "BILL_AMT6": 0.0,
    "PAY_AMT1": 0.0, "PAY_AMT2": 689.0, "PAY_AMT3": 0.0,
    "PAY_AMT4": 0.0, "PAY_AMT5": 0.0, "PAY_AMT6": 0.0
}
```

Ответ (200):
```
 json
{
    "prediction": 1,
    "probability": 0.7105,
    "model_version": "v1"
} 
```

Ошибки:
400 — невалидный JSON или отсутствуют/лишние признаки,
404 — запрошенная версия модели не найдена,
500 — внутренняя ошибка сервера,

## Запуск
Локально
```
bash
git clone https://github.com/t3stdr1v3/credit_crd_ml_deployment.git
cd credit_crd_ml_deployment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
python models/train_model.py
python models/train_model_v2.py
python app/api.py
```

Docker
```
bash
docker build -f docker/Dockerfile -t credit-default-service .
docker run -p 5000:5000 credit-default-service
Docker Compose
bash
docker compose up -d
docker compose down
```
Тестирование
```
bash
# Windows
set PYTHONPATH=%CD% && pytest tests/test_api.py -v

# Linux/macOS
PYTHONPATH=. pytest tests/test_api.py -v
```

Примеры curl-запросов
```
bash
# проверка health
curl http://localhost:5000/health

# прогноз дефолта (v1 по умолчанию)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,"PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,"BILL_AMT1":3913,"BILL_AMT2":3102,"BILL_AMT3":689,"BILL_AMT4":0,"BILL_AMT5":0,"BILL_AMT6":0,"PAY_AMT1":0,"PAY_AMT2":689,"PAY_AMT3":0,"PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0}'

# прогноз с выбором версии v2
curl -X POST "http://localhost:5000/predict?version=v2" \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,"PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,"BILL_AMT1":3913,"BILL_AMT2":3102,"BILL_AMT3":689,"BILL_AMT4":0,"BILL_AMT5":0,"BILL_AMT6":0,"PAY_AMT1":0,"PAY_AMT2":689,"PAY_AMT3":0,"PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0}'
```

Технологии
```
Python 3.12, Flask, scikit-learn, pandas, numpy

pytest, Docker, nginx
```

Docker Hub
https://hub.docker.com/r/testdrivee/credit-default-service
