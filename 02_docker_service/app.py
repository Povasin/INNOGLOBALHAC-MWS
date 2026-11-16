# ==============================================================================
# HTTP-СЕРВИС ИЗВЛЕЧЕНИЯ ФАКТОВ (MTS MWS CompactFactExtraction)
# ==============================================================================
# Принимает транскрипт звонка с метаданными по HTTP (FastAPI) и возвращает
# извлеченные и откалиброванные факты в соответствии со схемой case_metrics v2.1.
# Поддерживает 3 режима работы (кривая "Ресурсы x Качество"):
#   - point1_cpu : Сверхбыстрый детерминированный/каталожный режим (CPU, ~0.05 с)
#   - point2_05b : Qwen/Qwen2.5-0.5B-Instruct + LoRA + Калибровка v3.0
#   - point3_15b : Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct + LoRA + Калибровка v3.0
# ==============================================================================

import copy
import json
import os
import re
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# 1. СХЕМЫ ДАННЫХ (Pydantic)
# ------------------------------------------------------------------------------

class TurnInput(BaseModel):
    idx: int = Field(..., description="Индекс реплики в диалоге")
    speaker: str = Field(..., description="Говорящий (client / operator)")
    text: str = Field(..., description="Текст реплики")

class CallInput(BaseModel):
    call_id: str = Field(default="call_001", description="Уникальный ID звонка")
    started_at: str = Field(default="2026-01-01T12:00:00", description="ISO дата и время начала звонка")
    turns: List[TurnInput] = Field(default_factory=list, description="Список реплик диалога")
    facts: Optional[List[Dict[str, Any]]] = Field(default=None, description="Предварительные факты (если есть)")

class BatchCallInput(BaseModel):
    calls: List[CallInput]

# ------------------------------------------------------------------------------
# 2. КОНСТАНТЫ И СЛОВАРИ КАЛИБРОВКИ (Гипотезы H1-H5)
# ------------------------------------------------------------------------------

STATUS_RANK = {
    "принята": 0,
    "в работе": 1,
    "назначен выезд специалиста": 2,
    "выполнена": 3
}

NEG_CUES = [
    "не помог", "не помогла", "не устранил", "без результат",
    "толку нет", "толку ноль", "всё равно", "все равно",
    "не сработа", "не помогло", "проблему не решил"
]

WORD_SPAN_RE = re.compile(r"[а-яёa-z0-9]+", re.I)
STOPWORDS = {
    "не", "на", "в", "во", "с", "со", "по", "за", "из", "от", "до", "для",
    "при", "без", "и", "а", "но", "или", "что", "как", "это", "все", "всё",
    "у", "меня", "мне", "мы", "вы", "он", "она", "оно", "они", "клиент",
    "оператор", "помогло"
}

SYSTEM_PROMPT = (
    "Извлекай подтверждённые факты из звонков техподдержки.\n"
    "Выведи только JSON-массив записей, каждая запись имеет 8 полей в таком порядке:\n"
    "[тип объекта, имя объекта, атрибут, значение, полярность, valid_from, valid_to, индекс реплики].\n"
    'Пример формата: [["service","Домашний интернет","service_state","не работает",1,"2026-01-01",null,3]]\n'
    "Каждая запись — массив из восьми значений. Не добавляй ключи, Markdown или пояснения.\n"
    "Типы: service, device, client, ticket, other.\n"
    "Атрибуты: service_state, tariff, ticket_status, action, agreement.\n"
    "Указывай источник по индексу реплики, где озвучен факт. Если фактов нет, выведи []."
)

DEFAULT_CATALOG = {
    "symptoms": [
        "не работает", "низкая скорость", "периодические обрывы", "пропадает сигнал",
        "нет подключения", "горит красный индикатор", "не открываются сайты",
        "зависает картинка", "рассыпается изображение", "не раздает вай-фай",
        "высокий пинг", "работает нормально", "по кабелю сигнал в норме"
    ],
    "actions": [
        "перезагрузка роутера", "перезагрузка приставки", "переподключение кабеля",
        "сброс настроек", "проверка баланса", "замена патч-корда", "прямое подключение по кабелю"
    ],
    "agreements": [
        "выезд инженера", "перерасчёт абонентской платы", "замена оборудования",
        "обратный звонок специалиста", "отправка СМС с инструкцией", "смена тарифа"
    ],
    "statuses": [
        "принята", "в работе", "назначен выезд специалиста", "выполнена"
    ],
    "tariffs": [
        "Базовый", "Плюс", "Максимальный", "Игровой", "Семейный", "Ультра", "Стандарт"
    ]
}

def load_catalog_safe(path: str = "catalog.json") -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cat = json.load(f)
                return {
                    "symptoms": cat.get("symptoms", DEFAULT_CATALOG["symptoms"]),
                    "actions": cat.get("actions", DEFAULT_CATALOG["actions"]),
                    "agreements": cat.get("agreements", DEFAULT_CATALOG["agreements"]),
                    "statuses": cat.get("statuses", DEFAULT_CATALOG["statuses"]),
                    "tariffs": cat.get("tariffs", DEFAULT_CATALOG["tariffs"]),
                }
        except Exception:
            pass
    return DEFAULT_CATALOG

CATALOG = load_catalog_safe(os.getenv("CATALOG_PATH", "catalog.json"))

# ------------------------------------------------------------------------------
# 3. ФУНКЦИИ НОРМАЛИЗАЦИИ, ЗАЗЕМЛЕНИЯ И КАЛИБРОВКИ v3.0
# ------------------------------------------------------------------------------

def _norm_w(w: str) -> str:
    return w.lower().replace("ё", "е")

def value_sim(a: str, b: str) -> float:
    a_n, b_n = _norm_w(a.strip()), _norm_w(b.strip())
    if not a_n or not b_n:
        return 0.0
    if a_n == b_n:
        return 1.0
    return SequenceMatcher(None, a_n, b_n).ratio()

def best_match(val_str: str, candidates: List[str]):
    if not val_str:
        return val_str, 0.0
    best_c, best_s = val_str, 0.0
    for c in candidates:
        s = value_sim(val_str, c)
        if s > best_s:
            best_s, best_c = s, c
            if s == 1.0:
                break
    return best_c, best_s

def clean_value(s: str) -> str:
    if not s:
        return ""
    s = s.replace("Wi-Fi", "вай-фай").replace("wi-fi", "вай-фай").replace("WiFi", "вай-фай").replace("wifi", "вай-фай")
    m = re.match(r"^(?:неисправност[ьия]|неисправен|неисправно|неисправна)\s*[:\(]\s*(.*?)\s*\)?$", s, re.IGNORECASE)
    if m:
        s = m.group(1)
    s = re.sub(r"^(?:самодиагностика|клиент выполнил[а-я]*|выполнена|клиент перезапустил[а-я]*)\s*[:\-]\s*", "", s, flags=re.IGNORECASE)
    return s.strip()

def find_best_span_in_turn(turn_text: str, value: str, obj_name: str):
    """Символьное заземление спана и цитаты в реплике."""
    if not turn_text:
        return [0, 0], ""
    turn_tokens = [(m.start(), m.end(), _norm_w(m.group(0))) for m in WORD_SPAN_RE.finditer(turn_text)]
    if not turn_tokens:
        return [0, len(turn_text)], turn_text
    query_words = [
        _norm_w(m.group(0))
        for m in WORD_SPAN_RE.finditer(f"{value} {obj_name}")
        if len(m.group(0)) >= 3 and _norm_w(m.group(0)) not in STOPWORDS
    ]
    matched_spans = []
    for s_idx, e_idx, tw in turn_tokens:
        if len(tw) < 3 or tw in STOPWORDS:
            continue
        for qw in query_words:
            if tw == qw or (len(tw) >= 4 and len(qw) >= 4 and tw[:4] == qw[:4]) or SequenceMatcher(None, tw, qw).ratio() >= 0.72:
                matched_spans.append((s_idx, e_idx))
                break
    if matched_spans:
        start_char = matched_spans[0][0]
        end_char = matched_spans[-1][1]
        return [start_char, end_char], turn_text[start_char:end_char]
    first_tok = turn_tokens[0]
    return [first_tok[0], first_tok[1]], turn_text[first_tok[0]:first_tok[1]]

def resolve_relative_date(text: str, started_at: str) -> str:
    """Детерминированная нормализация относительных дат относительно даты звонка."""
    base_str = (started_at or "2026-01-01T00:00:00")[:10]
    try:
        base_dt = datetime.fromisoformat(base_str)
    except ValueError:
        return base_str + "T00:00:00"
    t = _norm_w(text)
    if "позавчера" in t:
        return (base_dt - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00")
    if "вчера" in t:
        return (base_dt - timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
    if "завтра" in t:
        return (base_dt + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")
    if "в прошлом месяце" in t or "месяц назад" in t:
        return (base_dt - timedelta(days=30)).strftime("%Y-%m-%dT00:00:00")
    if "неделю назад" in t or "с прошлой недели" in t:
        return (base_dt - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00")
    return base_dt.strftime("%Y-%m-%dT00:00:00")

def extract_facts_cpu_fast(call_dict: dict) -> List[dict]:
    """Детерминированный экстрактор Point 1 (CPU Fast, 0 параметров GPU)."""
    turns = call_dict.get("turns", [])
    started_at = call_dict.get("started_at", "2026-01-01T00:00:00")
    extracted = []

    for turn in turns:
        idx = turn["idx"]
        speaker = turn.get("speaker", "client").lower()
        text = turn.get("text", "")
        t_low = _norm_w(text)
        vf = resolve_relative_date(text, started_at)

        # Определение базового объекта в реплике
        obj_type, obj_name = "service", "Домашний интернет"
        if any(w in t_low for w in ["роутер", "модем", "приставк", "кабел", "оборудован"]):
            obj_type = "device"
            obj_name = "роутер" if "роутер" in t_low else ("модем" if "модем" in t_low else "ТВ-приставка")
        elif any(w in t_low for w in ["тв", "телевид", "канал"]):
            obj_type, obj_name = "service", "Цифровое ТВ"

        # 1. Статусы заявок
        for st in CATALOG["statuses"]:
            if _norm_w(st) in t_low:
                extracted.append({
                    "object": {"type": "ticket", "name": "заявка"},
                    "attribute": "ticket_status",
                    "value": st,
                    "polarity": 1,
                    "valid_from": vf,
                    "valid_to": None,
                    "source": {"turn_idx": idx, "span": None, "quote": None}
                })

        # 2. Тарифы (только подтвержденные клиентом или исторические)
        for tar in CATALOG["tariffs"]:
            if _norm_w(tar) in t_low and ("тариф" in t_low or "переш" in t_low or "подключ" in t_low):
                if speaker == "operator" and any(w in t_low for w in ["предлож", "можем", "хотите"]):
                    continue
                is_old = any(w in t_low for w in ["был", "раньше", "до этого", "да нет"])
                extracted.append({
                    "object": {"type": "client", "name": "клиент"},
                    "attribute": "tariff",
                    "value": tar,
                    "polarity": 1,
                    "valid_from": vf,
                    "valid_to": started_at[:10] + "T00:00:00" if is_old else None,
                    "source": {"turn_idx": idx, "span": None, "quote": None}
                })

        # 3. Действия клиента
        if any(w in t_low for w in ["перезагруж", "перезагруз", "переподключ", "сбрасыв", "проверял"]):
            b_act, s_act = best_match(text, CATALOG["actions"])
            val = b_act if s_act >= 0.35 else "перезагрузка роутера"
            if any(c in t_low for c in NEG_CUES):
                val += " — не помогло"
            extracted.append({
                "object": {"type": "client", "name": "клиент"},
                "attribute": "action",
                "value": val,
                "polarity": 1,
                "valid_from": vf,
                "valid_to": None,
                "source": {"turn_idx": idx, "span": None, "quote": None}
            })

        # 4. Договоренности
        if any(w in t_low for w in ["инженер", "специалист", "перерасчет", "перерасчёт", "перезвон"]):
            if speaker == "operator" and any(w in t_low for w in ["предлагаю", "можем предложить"]):
                continue
            b_agr, s_agr = best_match(text, CATALOG["agreements"])
            if s_agr >= 0.35:
                extracted.append({
                    "object": {"type": "client", "name": "клиент"},
                    "attribute": "agreement",
                    "value": b_agr,
                    "polarity": 1,
                    "valid_from": vf,
                    "valid_to": None,
                    "source": {"turn_idx": idx, "span": None, "quote": None}
                })

        # 5. Состояние услуги / симптомы
        if any(w in t_low for w in ["не работа", "пропада", "обрыв", "низкая скорост", "красн", "зависа", "нет интернет"]):
            b_sym, s_sym = best_match(text, CATALOG["symptoms"])
            extracted.append({
                "object": {"type": obj_type, "name": obj_name},
                "attribute": "service_state",
                "value": b_sym if s_sym >= 0.35 else "не работает",
                "polarity": 1,
                "valid_from": vf,
                "valid_to": None,
                "source": {"turn_idx": idx, "span": None, "quote": None}
            })

    return extracted

def apply_full_postprocessing_and_calibration(call_data: dict) -> dict:
    """Полный стек Гипотез H1-H5 и калибровка инвариантов v3.0."""
    c = copy.deepcopy(call_data)
    turns_list = c.get("turns", [])
    turns_dict = {t["idx"]: t["text"] for t in turns_list}
    call_started = c.get("started_at", "2026-01-01T00:00:00")
    call_day = call_started[:10]

    symptoms = CATALOG["symptoms"] + ["работает нормально", "по кабелю сигнал в норме"]
    raw_facts = c.get("facts") or []
    facts = []

    for f in raw_facts:
        f = copy.deepcopy(f)
        f.setdefault("object", {"type": "other", "name": ""})
        f.setdefault("source", {"turn_idx": 0, "span": None, "quote": None})
        attr = f.get("attribute", "")
        raw_val = str(f.get("value") or "")
        val = clean_value(raw_val)
        tidx = f["source"].get("turn_idx", 0)
        turn_text = turns_dict.get(tidx, "").lower()

        # Регламентные типы объектов
        if attr in ("action", "agreement", "tariff"):
            f["object"]["type"] = "client"
        elif attr == "ticket_status":
            f["object"]["type"] = "ticket"

        # H1 & H2: Каталогизация и переквалификация action -> agreement
        if attr == "action":
            b_agr, s_agr = best_match(val, CATALOG["agreements"])
            b_act, s_act = best_match(val, CATALOG["actions"])
            if s_agr >= 0.75 and s_agr > s_act:
                f["attribute"] = "agreement"
                attr = "agreement"
                val = b_agr
            elif s_act >= 0.40:
                has_neg = any(w in raw_val.lower() or w in turn_text for w in NEG_CUES)
                val = b_act + (" — не помогло" if has_neg else "")
        elif attr == "agreement":
            b, s = best_match(val, CATALOG["agreements"])
            if s >= 0.40:
                val = b
        elif attr == "tariff":
            b, s = best_match(val, CATALOG["tariffs"])
            if s >= 0.40:
                val = b
        elif attr == "ticket_status":
            b, s = best_match(val, CATALOG["statuses"])
            if s >= 0.40:
                val = b
        elif attr == "service_state":
            b, s = best_match(val, symptoms)
            if s >= 0.40:
                val = b

        f["value"] = val

        # Нормализация дат и H3 (valid_to = None у событий)
        vf, vt = f.get("valid_from"), f.get("valid_to")
        if vf and len(vf) == 10:
            vf += "T00:00:00"
        if vt and len(vt) == 10:
            vt += "T00:00:00"
        if attr in ("action", "agreement"):
            vt = None
        if vf and vt and vf[:10] > vt[:10]:
            vf, vt = vt, vt
        f["valid_from"], f["valid_to"] = vf, vt

        # H5: Символьное заземление спана и цитаты
        t_txt = turns_dict.get(tidx, "")
        span, quote = find_best_span_in_turn(t_txt, f["value"], str(f["object"].get("name") or ""))
        f["source"] = {"turn_idx": tidx, "span": span, "quote": quote}
        facts.append(f)

    # Блок 2: Жизненный цикл заявок (ticket_status)
    t_facts = [
        f for f in facts
        if f.get("attribute") == "ticket_status" and f.get("polarity", 1) > 0 and f.get("value") in STATUS_RANK
    ]
    if len(t_facts) >= 2:
        t_facts.sort(key=lambda x: (STATUS_RANK[x["value"]], x.get("valid_from") or ""))
        dedup_t = []
        for tf in t_facts:
            if dedup_t and dedup_t[-1]["value"] == tf["value"]:
                if (tf.get("valid_from") or "") >= (dedup_t[-1].get("valid_from") or ""):
                    dedup_t[-1] = tf
            else:
                dedup_t.append(tf)
        if dedup_t:
            for i in range(len(dedup_t) - 1):
                next_vf = dedup_t[i + 1].get("valid_from")
                dedup_t[i]["valid_to"] = (
                    next_vf[:10] if next_vf and next_vf[:10] >= (dedup_t[i].get("valid_from") or "")[:10] else call_day
                ) + "T00:00:00"
            dedup_t[-1]["valid_to"] = None
            top_day = (dedup_t[-1].get("valid_from") or call_day)[:10]
            dedup_t[-1]["valid_from"] = top_day + "T23:59:59"
            facts = [f for f in facts if f not in t_facts] + dedup_t
    elif len(t_facts) == 1:
        t_facts[0]["valid_to"] = None
        top_day = (t_facts[0].get("valid_from") or call_day)[:10]
        t_facts[0]["valid_from"] = top_day + "T23:59:59"

    # Блок 3: Вытеснение тарифов
    tar_facts = [f for f in facts if f.get("attribute") == "tariff" and f.get("polarity", 1) > 0]
    if len(tar_facts) >= 2:
        closed_t = [f for f in tar_facts if f.get("valid_to") is not None]
        open_t = [f for f in tar_facts if f.get("valid_to") is None]
        if closed_t and open_t:
            trans_date = closed_t[-1]["valid_to"]
            for ot in open_t:
                if ot.get("valid_from") and ot["valid_from"][:10] == call_day and trans_date[:10] < call_day:
                    ot["valid_from"] = trans_date[:10] + "T23:59:59"
                ot["valid_to"] = None
    elif len(tar_facts) == 1:
        tar_facts[0]["valid_to"] = None

    # Блок 4: Защита от точечного схлопывания service_state
    for f in facts:
        if f.get("attribute") == "service_state" and f.get("polarity", 1) > 0:
            if f.get("valid_to") and f.get("valid_from") and f["valid_to"][:10] == f["valid_from"][:10]:
                f["valid_to"] = None

    c["facts"] = facts
    return c

# ------------------------------------------------------------------------------
# 4. МЕНЕДЖЕР МОДЕЛИ (Ленивая загрузка LLM + LoRA или CPU Fallback)
# ------------------------------------------------------------------------------

class FactExtractorEngine:
    def __init__(self):
        self.mode = os.getenv("INFERENCE_MODE", "auto")  # point1_cpu | point2_05b | point3_15b | auto
        self.model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
        self.weights_path = os.getenv("WEIGHTS_PATH", "./weights/final_adapter")
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.loaded_model_name = "Point1_CPU_Fast (Zero-GPU)"

    def ensure_loaded(self, requested_mode: str):
        if requested_mode == "point1_cpu":
            return
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            target_model = (
                "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct"
                if requested_mode == "point3_15b"
                else self.model_id
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            # Проверяем наличие весов локально или доступность загрузки
            source_path = self.weights_path if os.path.exists(os.path.join(self.weights_path, "config.json")) else target_model
            self.tokenizer = AutoTokenizer.from_pretrained(source_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                target_model,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            if os.path.exists(os.path.join(self.weights_path, "adapter_model.safetensors")) or os.path.exists(os.path.join(self.weights_path, "adapter_model.bin")):
                base_model = PeftModel.from_pretrained(base_model, self.weights_path)

            base_model.eval()
            self.model = base_model
            self.loaded_model_name = f"{target_model} ({self.device.upper()})"
        except Exception as e:
            print(f"[Warning] Переключение на Point1_CPU_Fast (LLM не загружена: {e})")
            self.model = None

    @staticmethod
    def unpack_answer(text: str, call_dict: dict) -> List[dict]:
        decoder = json.JSONDecoder()
        start = text.find("[")
        if start < 0:
            return []
        try:
            rows, _ = decoder.raw_decode(text[start:])
        except ValueError:
            rows, rest = [], text[start + 1:].lstrip()
            while rest.startswith(("[", "{")):
                try:
                    row, end = decoder.raw_decode(rest)
                    rows.append(row)
                    rest = rest[end:].lstrip()
                    if not rest.startswith(","):
                        break
                    rest = rest[1:].lstrip()
                except ValueError:
                    break

        facts = []
        TYPES = {"service", "device", "client", "ticket", "other"}
        ATTRIBUTES = {"service_state", "tariff", "ticket_status", "action", "agreement"}

        for row in rows:
            if isinstance(row, list) and len(row) == 8:
                kind, name, attr, val, pol, vf, vt, idx = row
                if kind in TYPES and attr in ATTRIBUTES and isinstance(pol, int) and isinstance(idx, int):
                    facts.append({
                        "object": {"type": kind, "name": str(name).strip()},
                        "attribute": attr,
                        "value": str(val).strip(),
                        "polarity": pol,
                        "valid_from": vf or call_dict.get("started_at"),
                        "valid_to": vt,
                        "source": {"turn_idx": idx, "span": None, "quote": None}
                    })
        return facts

    def process_call(self, call_dict: dict, mode: str = "point2_05b") -> dict:
        # Если на вход уже поданы сырые факты (например, режим калибровки предрасчитанных данных)
        if call_dict.get("facts"):
            return apply_full_postprocessing_and_calibration(call_dict)

        if mode != "point1_cpu":
            self.ensure_loaded(mode)

        if self.model is not None and mode != "point1_cpu":
            import torch
            turns = call_dict.get("turns", [])
            dialogue_lines = [f"Начало звонка: {call_dict.get('started_at', '2026-01-01T00:00:00')}"]
            dialogue_lines.extend(f"{t['idx']} {t['speaker']}: {t['text']}" for t in turns)
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(dialogue_lines)}
            ]
            with torch.inference_mode():
                encoded = self.tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(self.model.device)
                input_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
                if input_ids.shape[1] > 2048:
                    input_ids = input_ids[:, -2048:]
                gen_tokens = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )
                answer = self.tokenizer.decode(gen_tokens[0, input_ids.shape[1]:], skip_special_tokens=True)
                raw_facts = self.unpack_answer(answer, call_dict)
        else:
            raw_facts = extract_facts_cpu_fast(call_dict)

        call_with_raw = copy.deepcopy(call_dict)
        call_with_raw["facts"] = raw_facts
        return apply_full_postprocessing_and_calibration(call_with_raw)


ENGINE = FactExtractorEngine()

# ------------------------------------------------------------------------------
# 5. FASTAPI ПРИЛОЖЕНИЕ, КАСТОМНЫЙ WEB-ИНТЕРФЕЙС (MWS DARK UI) И ЭНДПОИНТЫ
# ------------------------------------------------------------------------------
from fastapi.responses import HTMLResponse

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="ru" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MTS MWS · CompactFactExtraction Studio v3.0</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-main: #090b10;
      --bg-card: rgba(18, 22, 31, 0.85);
      --bg-card-hover: rgba(26, 32, 44, 0.95);
      --bg-input: #0d1017;
      --border: rgba(255, 255, 255, 0.08);
      --border-focus: rgba(227, 6, 17, 0.55);
      --mts-red: #E30611;
      --mts-red-glow: rgba(227, 6, 17, 0.25);
      --text-primary: #F3F4F6;
      --text-secondary: #9CA3AF;
      --text-muted: #6B7280;
      --emerald: #10B981;
      --amber: #F59E0B;
      --blue: #3B82F6;
      --purple: #8B5CF6;
      --rose: #F43F5E;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-main);
      background-image:
        radial-gradient(circle at 12% 8%, rgba(227, 6, 17, 0.14), transparent 32%),
        radial-gradient(circle at 88% 15%, rgba(59, 130, 246, 0.10), transparent 35%);
      color: var(--text-primary);
      min-height: 100vh;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }
    code, pre, .mono {
      font-family: 'JetBrains Mono', monospace;
    }
    .container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px 28px 48px;
    }
    /* Header */
    .navbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-bottom: 22px;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
      flex-wrap: wrap;
      gap: 16px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .brand-logo {
      width: 42px;
      height: 42px;
      border-radius: 11px;
      background: linear-gradient(135deg, #E30611 0%, #990008 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 15px;
      letter-spacing: -0.5px;
      color: #fff;
      box-shadow: 0 6px 20px var(--mts-red-glow);
    }
    .brand-title {
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge-version {
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(227, 6, 17, 0.15);
      color: #ff4d57;
      border: 1px solid rgba(227, 6, 17, 0.35);
    }
    .brand-sub {
      font-size: 13px;
      color: var(--text-secondary);
    }
    .nav-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .status-pill {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 7px 14px;
      border-radius: 999px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.25);
      font-size: 12px;
      font-weight: 600;
      color: #34D399;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10B981;
      box-shadow: 0 0 8px #10B981;
    }
    .nav-link {
      text-decoration: none;
      color: var(--text-secondary);
      font-size: 13px;
      font-weight: 500;
      padding: 8px 14px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--bg-card);
      transition: all 0.18s ease;
    }
    .nav-link:hover {
      color: #fff;
      border-color: rgba(255,255,255,0.2);
      background: var(--bg-card-hover);
    }

    /* Pareto Mode Selector */
    .mode-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }
    .mode-card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px 18px;
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
      overflow: hidden;
    }
    .mode-card:hover {
      border-color: rgba(255, 255, 255, 0.2);
      transform: translateY(-1px);
    }
    .mode-card.active {
      border-color: var(--mts-red);
      background: linear-gradient(180deg, rgba(227, 6, 17, 0.12) 0%, rgba(18, 22, 31, 0.92) 100%);
      box-shadow: 0 8px 25px rgba(227, 6, 17, 0.15);
    }
    .mode-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }
    .mode-name {
      font-weight: 700;
      font-size: 14px;
      color: #fff;
    }
    .mode-tag {
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.07);
      color: var(--text-secondary);
    }
    .mode-card.active .mode-tag {
      background: var(--mts-red);
      color: #fff;
    }
    .mode-desc {
      font-size: 12px;
      color: var(--text-secondary);
      margin-bottom: 10px;
    }
    .mode-metrics {
      display: flex;
      gap: 12px;
      font-size: 11.5px;
      color: #D1D5DB;
    }
    .mode-metrics span strong {
      color: #10B981;
    }

    /* Main Workspace Layout */
    .workspace {
      display: grid;
      grid-template-columns: 1fr 1.15fr;
      gap: 24px;
      align-items: start;
    }
    @media (max-width: 1024px) {
      .mode-grid, .workspace { grid-template-columns: 1fr; }
    }
    .panel {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px 22px;
      backdrop-filter: blur(12px);
    }
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 10px;
    }
    .panel-title {
      font-size: 15px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .preset-group {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .btn-preset {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border);
      color: var(--text-secondary);
      padding: 5px 10px;
      border-radius: 7px;
      font-size: 11.5px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn-preset:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.2);
    }
    .meta-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 14px;
    }
    .field-label {
      display: block;
      font-size: 11.5px;
      font-weight: 600;
      color: var(--text-secondary);
      margin-bottom: 5px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .input-box {
      width: 100%;
      background: var(--bg-input);
      border: 1px solid var(--border);
      color: var(--text-primary);
      padding: 9px 12px;
      border-radius: 9px;
      font-size: 13px;
      outline: none;
      transition: border 0.15s;
    }
    .input-box:focus {
      border-color: var(--mts-red);
    }
    /* Dialogue Preview + JSON Toggle */
    .turns-preview {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 290px;
      overflow-y: auto;
      padding: 12px;
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 11px;
      margin-bottom: 14px;
    }
    .turn-bubble {
      padding: 9px 12px;
      border-radius: 9px;
      font-size: 13px;
      border-left: 3px solid transparent;
      background: rgba(255, 255, 255, 0.03);
    }
    .turn-bubble.client {
      border-left-color: #E30611;
      background: rgba(227, 6, 17, 0.06);
    }
    .turn-bubble.operator {
      border-left-color: #3B82F6;
      background: rgba(59, 130, 246, 0.06);
    }
    .turn-meta {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      margin-bottom: 3px;
      display: flex;
      justify-content: space-between;
      color: var(--text-secondary);
    }
    textarea.json-editor {
      width: 100%;
      height: 170px;
      background: var(--bg-input);
      border: 1px solid var(--border);
      color: #A7F3D0;
      padding: 12px;
      border-radius: 10px;
      font-size: 12px;
      resize: vertical;
      outline: none;
      margin-bottom: 14px;
    }
    textarea.json-editor:focus {
      border-color: var(--mts-red);
    }
    .btn-primary {
      width: 100%;
      background: linear-gradient(135deg, #E30611 0%, #B80009 100%);
      color: #fff;
      border: none;
      padding: 13px 20px;
      border-radius: 11px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      box-shadow: 0 6px 20px var(--mts-red-glow);
      transition: all 0.2s ease;
    }
    .btn-primary:hover {
      filter: brightness(1.1);
      transform: translateY(-1px);
    }
    .btn-primary:disabled {
      opacity: 0.6;
      cursor: wait;
    }

    /* KPI Strip */
    .kpi-strip {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-bottom: 16px;
    }
    .kpi-box {
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
    }
    .kpi-label {
      font-size: 11px;
      color: var(--text-secondary);
      margin-bottom: 2px;
    }
    .kpi-val {
      font-size: 16px;
      font-weight: 700;
      color: #fff;
    }
    .kpi-val.emerald { color: #34D399; }

    /* Fact Cards */
    .facts-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
      max-height: 520px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .fact-card {
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 16px;
      transition: border 0.15s;
    }
    .fact-card:hover {
      border-color: rgba(255, 255, 255, 0.22);
    }
    .fact-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }
    .attr-badge {
      font-size: 11px;
      font-weight: 700;
      padding: 3px 9px;
      border-radius: 6px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .attr-service_state { background: rgba(244, 63, 94, 0.15); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.35); }
    .attr-action { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.35); }
    .attr-agreement { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .attr-ticket_status { background: rgba(139, 92, 246, 0.15); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.35); }
    .attr-tariff { background: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.35); }

    .obj-pill {
      font-size: 12px;
      color: var(--text-secondary);
      font-weight: 500;
    }
    .fact-val {
      font-size: 15px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 10px;
    }
    .fact-dates {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 11.5px;
      color: var(--text-secondary);
      margin-bottom: 8px;
      padding: 6px 10px;
      background: rgba(255, 255, 255, 0.025);
      border-radius: 7px;
    }
    .fact-quote {
      font-size: 12px;
      color: #D1D5DB;
      border-left: 2px solid var(--mts-red);
      padding-left: 10px;
      font-style: italic;
    }
    .raw-json-out {
      margin-top: 14px;
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      font-size: 11.5px;
      color: #93C5FD;
      max-height: 200px;
      overflow-y: auto;
      display: none;
    }
    .toggle-btn {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-secondary);
      font-size: 11.5px;
      padding: 4px 10px;
      border-radius: 6px;
      cursor: pointer;
    }
    .toggle-btn:hover { color: #fff; border-color: rgba(255,255,255,0.25); }
  </style>
</head>
<body>
  <div class="container">
    <!-- Top Navbar -->
    <header class="navbar">
      <div class="brand">
        <div class="brand-logo">MWS</div>
        <div>
          <div class="brand-title">
            CompactFactExtraction Studio
            <span class="badge-version">Calibration Engine v3.0</span>
          </div>
          <div class="brand-sub">Компактные модели извлечения темпоральных фактов из транскриптов техподдержки · Innopolis Global Hack 2026</div>
        </div>
      </div>
      <div class="nav-actions">
        <div class="status-pill">
          <span class="status-dot"></span>
          <span id="engineStatusText">Engine Ready · Schema v2.1</span>
        </div>
        <a href="/docs" target="_blank" class="nav-link">Swagger API (/docs) ↗</a>
      </div>
    </header>

    <!-- 3 Points Pareto Selector -->
    <section class="mode-grid">
      <div class="mode-card" id="card-point1_cpu" onclick="selectMode('point1_cpu')">
        <div class="mode-header">
          <span class="mode-name"> Точка 1 · CPU Fast (Zero-GPU)</span>
          <span class="mode-tag">0M GPU · ~0.05s</span>
        </div>
        <div class="mode-desc">Детерминированный каталожный экстрактор + Калибровка v3.0</div>
        <div class="mode-metrics mono">
          <span>F1: <strong>0.3766</strong></span>
          <span>TempRec: <strong>0.4702</strong></span>
          <span>GrSpan: <strong>1.0000</strong></span>
        </div>
      </div>

      <div class="mode-card active" id="card-point2_05b" onclick="selectMode('point2_05b')">
        <div class="mode-header">
          <span class="mode-name"> Точка 2 · Qwen2.5-0.5B + LoRA</span>
          <span class="mode-tag">490M · 1.25s</span>
        </div>
        <div class="mode-desc">Оптимальный баланс цены и качества (экономия GPU 8.5x)</div>
        <div class="mode-metrics mono">
          <span>F1: <strong>0.3777</strong></span>
          <span>TempRec: <strong>0.4708</strong></span>
          <span>GrSpan: <strong>1.0000</strong></span>
        </div>
      </div>

      <div class="mode-card" id="card-point3_15b" onclick="selectMode('point3_15b')">
        <div class="mode-header">
          <span class="mode-name"> Точка 3 · Vikhr-1.5B SOTA</span>
          <span class="mode-tag">1.54B · 3.80s</span>
        </div>
        <div class="mode-desc">Максимальная полнота извлечения (+5.91 п.п. Recall к LLM)</div>
        <div class="mode-metrics mono">
          <span>F1: <strong>0.3791</strong></span>
          <span>Recall: <strong>0.4396</strong></span>
          <span>StateAcc: <strong>0.5329</strong></span>
        </div>
      </div>
    </section>

    <!-- Main Two-Column Workspace -->
    <main class="workspace">
      <!-- Left: Input Transcript -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"> Входной диалог техподдержки</div>
          <div class="preset-group">
            <button class="btn-preset" onclick="loadPreset(0)">Кейс 1: Обрыв + Выезд</button>
            <button class="btn-preset" onclick="loadPreset(1)">Кейс 2: Смена тарифа</button>
            <button class="btn-preset" onclick="loadPreset(2)">Кейс 3: Цепочка статусов</button>
          </div>
        </div>

        <div class="meta-row">
          <div>
            <label class="field-label">ID Звонка (call_id)</label>
            <input id="callIdInput" class="input-box mono" value="call_mws_demo_01" />
          </div>
          <div>
            <label class="field-label">Дата и время начала (started_at)</label>
            <input id="startedAtInput" class="input-box mono" value="2026-11-15T14:30:00" />
          </div>
        </div>

        <label class="field-label">Визуализация реплик диалога</label>
        <div id="turnsPreview" class="turns-preview"></div>

        <label class="field-label">JSON реплик (редактируемый)</label>
        <textarea id="turnsJsonInput" class="json-editor mono" oninput="renderTurnsPreview()"></textarea>

        <button id="extractBtn" class="btn-primary" onclick="runExtraction()">
          <span> Извлечь и откалибровать факты</span>
        </button>
      </div>

      <!-- Right: Extracted Facts Output -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"> Извлечённые факты (case_metrics v2.1)</div>
          <button class="toggle-btn mono" onclick="toggleRawJson()">{ } Показать Raw JSON</button>
        </div>

        <div class="kpi-strip">
          <div class="kpi-box">
            <div class="kpi-label">Найдено фактов</div>
            <div id="kpiCount" class="kpi-val mono">0</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">Latency (сервер)</div>
            <div id="kpiLatency" class="kpi-val emerald mono">—</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">Grounding Span</div>
            <div id="kpiGrounding" class="kpi-val emerald mono">100%</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">Активный режим</div>
            <div id="kpiMode" class="kpi-val mono" style="font-size:13px;">point2_05b</div>
          </div>
        </div>

        <div id="factsContainer" class="facts-list"></div>
        <pre id="rawJsonOutput" class="raw-json-out mono"></pre>
      </div>
    </main>
  </div>

  <script>
    let currentMode = 'point2_05b';

    const PRESETS = [
      {
        call_id: "call_internet_outage_01",
        started_at: "2026-11-15T14:30:00",
        turns: [
          {"idx": 0, "speaker": "operator", "text": "Здравствуйте, техподдержка МТС, чем могу помочь?"},
          {"idx": 1, "speaker": "client", "text": "Со вчерашнего дня не работает домашний интернет, роутер перезагружал уже три раза — вообще не помогло."},
          {"idx": 2, "speaker": "operator", "text": "Вижу обрыв на линии. Заявка принята, сейчас она уже в работе. Оформим выезд инженера на завтра."},
          {"idx": 3, "speaker": "client", "text": "Хорошо, выезд инженера подходит, жду."}
        ]
      },
      {
        call_id: "call_tariff_switch_02",
        started_at: "2026-11-15T18:00:00",
        turns: [
          {"idx": 0, "speaker": "operator", "text": "Добрый вечер! Подскажите, у вас сейчас тариф Базовый подключен?"},
          {"idx": 1, "speaker": "client", "text": "Да нет, раньше был Базовый, а в прошлом месяце я перешёл на тариф Плюс."},
          {"idx": 2, "speaker": "operator", "text": "Понял, также можем предложить вам пакет Ультра со скидкой."},
          {"idx": 3, "speaker": "client", "text": "Нет, спасибо, меня текущий устраивает, только сделайте перерасчёт абонентской платы за вчерашний сбой."}
        ]
      },
      {
        call_id: "call_ticket_lifecycle_03",
        started_at: "2026-11-15T11:15:00",
        turns: [
          {"idx": 0, "speaker": "client", "text": "Здравствуйте, вчера оставлял обращение, потому что на модеме горит красный индикатор и не раздает вай-фай."},
          {"idx": 1, "speaker": "operator", "text": "Проверяю: вчера ваша заявка была принята, затем переведена в статус в работе."},
          {"idx": 2, "speaker": "operator", "text": "Сегодня по ней назначен выезд специалиста после 14:00."},
          {"idx": 3, "speaker": "client", "text": "Отлично, спасибо за уточнение!"}
        ]
      }
    ];

    function selectMode(mode) {
      currentMode = mode;
      ['point1_cpu', 'point2_05b', 'point3_15b'].forEach(m => {
        document.getElementById('card-' + m).classList.toggle('active', m === mode);
      });
      document.getElementById('kpiMode').textContent = mode;
    }

    function loadPreset(index) {
      const p = PRESETS[index];
      document.getElementById('callIdInput').value = p.call_id;
      document.getElementById('startedAtInput').value = p.started_at;
      document.getElementById('turnsJsonInput').value = JSON.stringify(p.turns, null, 2);
      renderTurnsPreview();
      runExtraction();
    }

    function renderTurnsPreview() {
      const container = document.getElementById('turnsPreview');
      try {
        const turns = JSON.parse(document.getElementById('turnsJsonInput').value);
        container.innerHTML = turns.map(t => `
          <div class="turn-bubble ${t.speaker === 'client' ? 'client' : 'operator'}">
            <div class="turn-meta">
              <span>#${t.idx} · ${t.speaker === 'client' ? ' Клиент' : ' Оператор'}</span>
            </div>
            <div>${t.text}</div>
          </div>
        `).join('');
      } catch (e) {
        container.innerHTML = '<div style="color:#FB7185;font-size:12px;">Ошибка синтаксиса JSON в списке реплик</div>';
      }
    }

    function toggleRawJson() {
      const el = document.getElementById('rawJsonOutput');
      el.style.display = (el.style.display === 'block') ? 'none' : 'block';
    }

    async function runExtraction() {
      const btn = document.getElementById('extractBtn');
      btn.disabled = true;
      btn.innerHTML = '<span> Извлечение и калибровка инвариантов...</span>';

      try {
        const turns = JSON.parse(document.getElementById('turnsJsonInput').value);
        const payload = {
          call_id: document.getElementById('callIdInput').value,
          started_at: document.getElementById('startedAtInput').value,
          turns: turns
        };

        const res = await fetch(`/extract?mode=${currentMode}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        document.getElementById('kpiCount').textContent = (data.facts || []).length;
        document.getElementById('kpiLatency').textContent = (data.latency_sec ?? 0.01) + ' s';
        document.getElementById('rawJsonOutput').textContent = JSON.stringify(data, null, 2);

        const factsContainer = document.getElementById('factsContainer');
        if (!data.facts || data.facts.length === 0) {
          factsContainer.innerHTML = '<div style="color:var(--text-secondary);font-size:13px;padding:12px;">Подтверждённых фактов не обнаружено.</div>';
        } else {
          factsContainer.innerHTML = data.facts.map(f => {
            const attr = f.attribute || 'other';
            const obj = f.object || {};
            const src = f.source || {};
            const vt = f.valid_to ? `<span style="color:#FB7185;">${f.valid_to} (закрыт)</span>` : `<span style="color:#34D399;">null (открыт / актуален)</span>`;
            return `
              <div class="fact-card">
                <div class="fact-top">
                  <span class="attr-badge attr-${attr}">${attr}</span>
                  <span class="obj-pill mono">[${obj.type}: "${obj.name}"] · Реплика #${src.turn_idx}</span>
                </div>
                <div class="fact-val">${f.value}</div>
                <div class="fact-dates mono">
                  <span>valid_from: <strong>${f.valid_from || '—'}</strong></span>
                  <span>valid_to: <strong>${vt}</strong></span>
                </div>
                <div class="fact-quote">«${src.quote || ''}» <span class="mono" style="color:var(--text-muted);font-style:normal;">span: [${(src.span || []).join(', ')}]</span></div>
              </div>
            `;
          }).join('');
        }
      } catch (err) {
        alert('Ошибка запроса: ' + err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span> Извлечь и откалибровать факты</span>';
      }
    }

    window.addEventListener('DOMContentLoaded', () => loadPreset(0));
  </script>
</body>
</html>
"""

app = FastAPI(
    title="MTS MWS CompactFactExtraction API",
    description="HTTP-интерфейс компактного извлечения и калибровки фактов из транскриптов техподдержки (case_metrics v2.1)",
    version="3.0.0"
)

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def web_dashboard():
    """Красивый интерактивный Web-интерфейс для живой демонстрации жюри."""
    return HTML_DASHBOARD

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "MTS MWS CompactFactExtraction",
        "calibration_engine": "v3.0",
        "loaded_model": ENGINE.loaded_model_name,
        "supported_modes": ["point1_cpu", "point2_05b", "point3_15b"]
    }

@app.post("/extract")
@app.post("/extract_facts")
def extract_single_call(
    call: CallInput,
    mode: str = Query(default="point2_05b", description="Режим кривой: point1_cpu | point2_05b | point3_15b")
):
    t0 = time.perf_counter()
    call_dict = call.model_dump()
    result = ENGINE.process_call(call_dict, mode=mode)
    result["latency_sec"] = round(time.perf_counter() - t0, 4)
    return result

@app.post("/extract_batch")
@app.post("/extract_facts_batch")
def extract_batch_calls(
    batch: BatchCallInput,
    mode: str = Query(default="point2_05b", description="Режим кривой: point1_cpu | point2_05b | point3_15b")
):
    t0 = time.perf_counter()
    results = [ENGINE.process_call(c.model_dump(), mode=mode) for c in batch.calls]
    return {
        "n_calls": len(results),
        "total_latency_sec": round(time.perf_counter() - t0, 4),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    print(" Запуск сервера MTS MWS Fact Extraction на http://127.0.0.1:8000 ...")
    print(" Откройте в браузере Web UI: http://127.0.0.1:8000")
    print(" Swagger UI документация: http://127.0.0.1:8000/docs")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
