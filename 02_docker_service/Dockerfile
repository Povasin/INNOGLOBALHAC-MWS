# ==============================================================================
# DOCKERFILE ДЛЯ КЕЙСА MTS MWS CompactFactExtraction (Innopolis Global Hack 2026)
# ==============================================================================
# Поддерживает работу как на чистом CPU, так и на GPU (при запуске с --gpus all)
# ==============================================================================

FROM python:3.11-slim

LABEL maintainer="MTS MWS CompactFactExtraction Team"
LABEL description="Compact Fact Extraction HTTP Service with Invariant Calibration v3.0"

# Системные переменные окружения для воспроизводимости и отсутствия буферизации
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PIP_NO_CACHE_DIR=1 \
    INFERENCE_MODE=point2_05b \
    MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
    WEIGHTS_PATH=/app/weights/final_adapter \
    CATALOG_PATH=/app/catalog.json \
    PORT=8000

WORKDIR /app

# Установка минимальных системных утилит (curl для healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей Python
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Копирование кода сервиса, скриптов обучения/инференса и конфигураций
COPY . /app/

# Открываем HTTP порт сервиса
EXPOSE 8000

# Проверка состояния контейнера
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Запуск HTTP-сервера FastAPI (Uvicorn)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
