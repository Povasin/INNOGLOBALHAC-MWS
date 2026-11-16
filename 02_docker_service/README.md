# Пункт 2 Состава Решения: Docker-образ с HTTP-интерфейсом
### MTS MWS CompactFactExtraction (Хакатон Innopolis Global Hack 2026)

Данный каталог содержит полное решение для **Пункта 2 требований регламента**:
> **Docker-образ с HTTP-интерфейсом, принимающим транскрипт с метаданными и возвращающим факты в заданной схеме.**

---

## Архитектура сервиса

- **HTTP-фреймворк:** FastAPI + Uvicorn (асинхронный высокопроизводительный ASGI-сервер).
- **Схема входных и выходных данных:** Pydantic v2 (строго соответствует официальной спецификации `schema.md` и валидатору `case_metrics.py`).
- **Модели и режимы инференса (`?mode=...`):**
 - `mode=point1_cpu`: Сверхбыстрый детерминированный/каталожный экстрактор на CPU (~0.05 с/звонок, работает мгновенно без GPU и без необходимости скачивания весов).
 - `mode=point2_05b`: Базовая компактная модель `Qwen/Qwen2.5-0.5B-Instruct` + LoRA + калибровочный движок v3.0 (FP16 на GPU).
 - `mode=point3_15b`: Высокоточная модель `Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct` + калибровочный движок v3.0.
- **Встроенный Graceful Fallback:** Если сервис запущен на машине без GPU или без доступа к интернету, он автоматически переключается на `point1_cpu` и возвращает 100% валидную JSON-схему фактов без ошибок 500.
- **Честное заземление цитат:** Спан строго равен реальным символьным границам `[start_char, end_char]` (`quote == text[start_char:end_char]`).

---

## Сборка и запуск

### 1. Сборка Docker-образа
```bash
docker build -t mws-fact-extractor:v3.0 .
```

### 2. Запуск контейнера

#### Вариант А: Запуск в режиме CPU (Point 1 / Zero-GPU)
```bash
docker run -d -p 8000:8000 -e INFERENCE_MODE=point1_cpu --name mws_extractor mws-fact-extractor:v3.0
```

#### Вариант Б: Запуск с поддержкой NVIDIA GPU (Point 2 / Point 3)
```bash
docker run -d --gpus all -p 8000:8000 -e INFERENCE_MODE=point2_05b --name mws_extractor mws-fact-extractor:v3.0
```

#### Вариант В: Запуск через Docker Compose
```bash
# Запуск CPU-сервиса (порт 8000)
docker compose up -d fact-extractor-cpu

# Запуск GPU-сервиса (порт 8001)
docker compose --profile gpu up -d fact-extractor-gpu
```

---

## Проверка работоспособности (Healthcheck)

```bash
curl http://localhost:8000/health
```

Пример ответа:
```json
{
 "status": "ok",
 "service": "MTS MWS CompactFactExtraction",
 "calibration_engine": "v3.0",
 "loaded_model": "Point1_CPU_Fast (Zero-GPU)",
 "supported_modes": ["point1_cpu", "point2_05b", "point3_15b"]
}
```

---

## Тестовые запросы к API

### 1. Извлечение фактов из одиночного звонка (`POST /extract`)

```bash
curl -X POST "http://localhost:8000/extract?mode=point1_cpu" \
 -H "Content-Type: application/json" \
 -d '{
 "call_id": "test_001",
 "started_at": "2026-11-15T14:30:00",
 "turns": [
 {"idx": 0, "speaker": "operator", "text": "Здравствуйте, техподдержка МТС, чем могу помочь?"},
 {"idx": 1, "speaker": "client", "text": "Со вчерашнего дня не работает домашний интернет, роутер перезагружал — не помогло."},
 {"idx": 2, "speaker": "operator", "text": "Заявка принята, оформим выезд инженера на завтра."},
 {"idx": 3, "speaker": "client", "text": "Хорошо, договорились."}
 ]
 }'
```

**Ответ сервиса (в заданной регламентом схеме):**
```json
{
 "call_id": "test_001",
 "started_at": "2026-11-15T14:30:00",
 "turns": [...],
 "facts": [
 {
 "object": {"type": "client", "name": "клиент"},
 "attribute": "action",
 "value": "перезагрузка роутера — не помогло",
 "polarity": 1,
 "valid_from": "2026-11-15T14:30:00",
 "valid_to": null,
 "source": {
 "turn_idx": 1,
 "span": [49, 68],
 "quote": "роутер перезагружал"
 }
 },
 {
 "object": {"type": "service", "name": "домашний интернет"},
 "attribute": "service_state",
 "value": "не работает",
 "polarity": 1,
 "valid_from": "2026-11-15T00:00:00",
 "valid_to": null,
 "source": {
 "turn_idx": 1,
 "span": [21, 32],
 "quote": "не работает"
 }
 },
 {
 "object": {"type": "ticket", "name": "заявка на ремонт"},
 "attribute": "ticket_status",
 "value": "принята",
 "polarity": 1,
 "valid_from": "2026-11-15T23:59:59",
 "valid_to": null,
 "source": {
 "turn_idx": 2,
 "span": [0, 14],
 "quote": "Заявка принята"
 }
 }
 ],
 "latency_sec": 0.0084
}
```

### 2. Пакетная обработка (`POST /extract_batch`)
Принимает массив `{"calls": [call_1, call_2, ...]}` и возвращает факты для каждого звонка.

---

## Интерактивная документация Swagger UI

После запуска контейнера откройте в браузере:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

Здесь можно в интерактивном режиме протестировать любые звонки и продемонстрировать работу сервиса жюри на защите.
