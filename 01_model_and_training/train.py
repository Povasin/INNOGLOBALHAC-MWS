# ==============================================================================
# КОД ОБУЧЕНИЯ МОДЕЛИ С ФИКСИРОВАННЫМИ СИДАМИ (MTS MWS CompactFactExtraction)
# ==============================================================================
# Задача: Дообучение компактной LLM (0.5B / 1.5B) для извлечения фактов
# Архитектура: Qwen2.5-0.5B / Vikhr-1.5B + 4-bit NF4 Quantization + LoRA PEFT
# Воспроизводимость: Полная фиксация всех сидов (PYTHONHASHSEED, random, np, torch)
# Маскирование лосса: Строго только токены ответа assistant (-100 для prompt)
# ==============================================================================

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)

def fix_all_seeds(seed: int = 42):
    """Строгая фиксация генераторов случайных чисел во всех библиотеках."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    hf_set_seed(seed)
    print(f"🔒 Все генераторы случайных чисел зафиксированы: SEED = {seed}")

class SFTChatDataset(Dataset):
    """
    Датасет звонков техподдержки в формате ChatML диалога.
    МАСКИРОВАНИЕ: Промпт (system + user диалог) маскируется значениями -100,
    градиенты и лосс вычисляются ИСКЛЮЧИТЕЛЬНО на целевом ответе assistant (JSON фактов).
    """
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 1280):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        for item in lines:
            msgs = item.get("messages", [])
            if len(msgs) < 3:
                continue
            prompt_msgs = msgs[:2]  # system + user
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            if not full_text.endswith(tokenizer.eos_token):
                full_text += tokenizer.eos_token

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)

            # Ограничиваем длину 1280 токенами для предотвращения OOM в словаре Qwen (152k)
            if len(full_ids) > max_len or len(prompt_ids) >= len(full_ids):
                continue

            prompt_len = min(len(prompt_ids), len(full_ids))
            # Маскируем весь промпт -100, обучаем только генерацию фактов
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            labels = labels[:len(full_ids)]

            self.examples.append({
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels
            })
        print(f"📖 Загружено {len(self.examples)} валидных обучающих примеров (длина до {max_len} токенов).")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def create_collate_fn(tokenizer):
    """Динамический паддинг батча только до максимальной длины примера в батче."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    return collate_fn

def train(args):
    fix_all_seeds(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Загрузка модели: {args.model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"📖 Загрузка обучающего датасета: {args.train_file}")
    train_dataset = SFTChatDataset(args.train_file, tokenizer, max_len=args.max_length)
    collate_fn = create_collate_fn(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn
    )

    print("🚀 Старт обучения с точным маскированием промпта...")
    trainer.train()

    final_adapter_dir = out_dir / "final_adapter"
    print(f"💾 Сохранение обученного LoRA адаптера в: {final_adapter_dir}")
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    # Сохраняем конфиг обучения
    config_dict = vars(args)
    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print("✅ Обучение успешно завершено!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение компактного экстрактора фактов с фиксированными сидами")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Базовая модель")
    parser.add_argument("--train_file", type=str, default="train_sft.jsonl", help="Путь к обучающему файлу")
    parser.add_argument("--output_dir", type=str, default="./weights", help="Директория сохранения")
    parser.add_argument("--seed", type=int, default=42, help="Фиксированный seed")
    parser.add_argument("--epochs", type=int, default=3, help="Количество эпох")
    parser.add_argument("--max_steps", type=int, default=320, help="Максимальное число шагов (-1 для эпох)")
    parser.add_argument("--batch_size", type=int, default=2, help="Размер батча на GPU")
    parser.add_argument("--grad_accum", type=int, default=8, help="Накопление градиента")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Скорость обучения")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_length", type=int, default=1280, help="Максимальная длина последовательности")

    args = parser.parse_args()
    train(args)
