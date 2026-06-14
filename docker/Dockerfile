FROM python:3.12-slim

WORKDIR /app

# копирование и усановка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# копирование кода и модели
COPY app/ ./app/
COPY models/ ./models/

# устанавливаем PYTHONPATH для импорта модуля app
ENV PYTHONPATH=/app

# открытие порта
EXPOSE 5000

# запуск приложения
CMD ["python", "app/api.py"]