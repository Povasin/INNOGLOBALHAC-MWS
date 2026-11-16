# Руководство по развёртыванию решения (Deployment Guide)

Данный документ описывает пошаговое развёртывание системы извлечения фактов **MTS MWS CompactFactExtraction** в различных окружениях: локально, в Docker, в облаке/Kubernetes и на платформе Kaggle/Colab.

---

## 1. Системные требования

### Минимальные требования (Точка 1: CPU Fast)
- **ОС:** Linux (Ubuntu 20.04+), Windows 10/11, macOS.
- **CPU:** 2 ядра (x86_64 / ARM64).
- **RAM:** 2 ГБ.
- **Диск:** 500 МБ.
- **GPU:** Не требуется!

### Рекомендуемые требования (Точки 2 и 3: GPU Инференс)
- **ОС:** Linux (Ubuntu 22.04 LTS).
- **CPU:** 4 ядра.
- **RAM:** 8 ГБ.
- **GPU:** 1x NVIDIA GPU с памятью $\ge 4$ ГБ (Tesla T4, RTX 3060/3090, A10G, V100, A100).
- **CUDA:** 11.8 или 12.1+.

---

## 2. Локальное развёртывание (Python venv)

### Шаг 1. Клонирование репозитория
```bash
git clone https://github.com/Povasin/INNOGLOBALHAC-MWS.git
cd INNOGLOBALHAC-MWS
```

### Шаг 2. Создание виртуального окружения
```bash
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1
```

### Шаг 3. Установка зависимостей
```bash
pip install -r 02_docker_service/requirements.txt
```

### Шаг 4. Быстрая валидация метрик
```bash
python -c "import case_metrics; print('Validator loaded successfully!')"
```

---

## 3. Развёртывание в Docker

```bash
cd 02_docker_service
docker build -t mws-compactfact:latest .
docker run -d --name mws-api -p 8000:8000 --restart always mws-compactfact:latest
```

Проверка статуса:
```bash
curl http://localhost:8000/health
```

---

## 4. Развёртывание в Kubernetes (K8s)

Пример манифеста `deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: compactfact-service
  namespace: mws-nlp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: compactfact-service
  template:
    metadata:
      labels:
        app: compactfact-service
    spec:
      containers:
      - name: compactfact
        image: mws-compactfact:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 15
---
apiVersion: v1
kind: Service
metadata:
  name: compactfact-svc
  namespace: mws-nlp
spec:
  type: ClusterIP
  selector:
    app: compactfact-service
  ports:
  - port: 80
    targetPort: 8000
```

---

## 5. Запуск на Kaggle и Google Colab

Для воспроизведения результатов жюри достаточно запустить ноутбук `check.ipynb` (или `01_model_and_training/notebooks/verification_and_points_check.ipynb`):
1. Открыть Kaggle / Colab.
2. Загрузить `check.ipynb` и архив данных.
3. Нажать **Run All**.
4. Время выполнения: ~40 секунд. Все метрики считаются официальным валидатором `case_metrics.py`.
