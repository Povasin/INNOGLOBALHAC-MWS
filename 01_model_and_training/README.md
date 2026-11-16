# Пункт 1 Состава Решения: Обученная Модель и Код Обучения
### MTS MWS CompactFactExtraction (Хакатон Innopolis Global Hack 2026)

Данная директория содержит полное выполнение **Пункта 1 требований организаторов**:
* Обученную модель (веса LoRA адаптера, токенизатор, параметры генерации);
* Полную конфигурацию инференса;
* Автономный код обучения с **строго фиксированными сидами** (`seed=42`).

---

## Структура директории

```
01_model_and_training/
├── train.py # Скрипт обучения модели с фиксацией сидов (SEED=42)
├── inference.py # Скрипт инференса модели с калибровочным движком v3.0
├── prepare_dataset.py # Скрипт подготовки ChatML-датасета (train.jsonl -> train_sft.jsonl)
├── training_config.json # Гиперпараметры обучения, LoRA, оптимайзера и сидов
├── inference_config.json # Конфигурация инференса (жадный поиск temp=0.0, калибровка)
├── weights/ # Веса и конфиг адаптера
│ └── adapter_config.json # Официальный PEFT LoRA конфиг (r=16, alpha=32)
├── notebooks/ # Jupyter ноутбуки для запуска на Kaggle и Colab
│ ├── verification_and_points_check.ipynb # Проверочный ноутбук с метриками всех 3 точек
│ └── training_pipeline_colab_kaggle.ipynb # Полный ноутбук дообучения нейросети на GPU
└── predictions/ # Готовые выгрузки предсказаний всех 3 точек
 ├── pred_point1_cpu.jsonl # Точка 1 (CPU Fast, 0 параметров на GPU)
 ├── pred_point2_05b.jsonl # Точка 2 (Qwen 0.5B Instruct + Калибровка)
 ├── pred_point3_15b.jsonl # Точка 3 (Vikhr 1.5B SOTA Ensemble)
 ├── pred_final.jsonl # Финальное предсказание
 ├── metrics_summary.json # Сводка официальных метрик case_metrics v2.1
 └── resource_benchmarks.json # Замеры задержки, VRAM и пропускной способности
```

---

## 1. Воспроизводимость и фиксация сидов (`train.py`)

Для соответствия критерию №2 регламента («Воспроизводимость обучения») в коде реализована функция жесткой детерминированности:

```python
def fix_all_seeds(seed: int = 42):
 os.environ["PYTHONHASHSEED"] = str(seed)
 random.seed(seed)
 np.random.seed(seed)
 torch.manual_seed(seed)
 torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic = True
 torch.backends.cudnn.benchmark = False
```

### Запуск обучения:
```bash
python train.py --model_id "Qwen/Qwen2.5-0.5B-Instruct" --train_file "train_sft.jsonl" --seed 42 --epochs 3
```

---

## 2. Конфигурация инференса (`inference.py`)

Инференс поддерживает детерминированное извлечение фактов:
* **Жадное декодирование:** `temperature = 0.0`, `do_sample = False`.
* **Калибровочный движок v3.0:**
 * Упорядочивание цепочек статусов заявок (`принята` -> `в работе` -> `выезд` -> `выполнена`);
 * Разделение тарифов по времени (исторический закрывается датой перехода, текущий открыт);
 * 100% символьное заземление цитат и спанов (`grounding_span = 1.0`).

### Запуск инференса:
```bash
python inference.py --input "val.jsonl" --output "predictions.jsonl"
```

---

## 3. Подтверждённые метрики обученных точек (case_metrics v2.1):

| Точка кривой | Параметры | Latency | F1 | Recall | Temporal Recall | State Accuracy | Grounding Span | Статус |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Point 1 (CPU Fast)** | 0M GPU | 0.05 с | **0.3766** | 0.3919 | **0.4702** | **0.5201** | **1.0000** | PASS |
| **Point 2 (Qwen 0.5B)** | 490M | 1.25 с | **0.3777** | 0.4016 | **0.4708** | **0.5201** | **1.0000** | PASS |
| **Point 3 (Vikhr 1.5B)** | 1.54B | 3.80 с | **0.3791** | **0.4396** | **0.4627** | **0.5329** | **1.0000** | PASS |
