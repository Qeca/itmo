# Используем официальный образ Python (например, 3.9-slim)
FROM python:3.9-slim

# Обновляем список пакетов и устанавливаем зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения
COPY . .

# Открываем порт 8081 (тот, который используется в вашем коде)
EXPOSE 8081

# Запускаем приложение с помощью Hypercorn (убедитесь, что он указан в requirements.txt)
CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:8081"]