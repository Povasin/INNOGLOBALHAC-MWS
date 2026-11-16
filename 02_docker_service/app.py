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
    """Честное символьное заземление: границы [start_char, end_char], где turn_text[start_char:end_char] == quote."""
    if not turn_text:
        return [0, 0], ""
    turn_tokens = [(m.start(), m.end(), _norm_w(m.group(0))) for m in WORD_SPAN_RE.finditer(turn_text)]
    if not turn_tokens:
        l = min(30, len(turn_text))
        return [0, l], turn_text[:l]
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
            print(f"⚠️ Переключение на Point1_CPU_Fast (LLM не загружена: {e})")
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
# 5. FASTAPI ПРИЛОЖЕНИЕ И ЭНДПОИНТЫ
# ------------------------------------------------------------------------------

app = FastAPI(
    title="MTS MWS CompactFactExtraction API",
    description="HTTP-интерфейс компактного извлечения и калибровки фактов из транскриптов техподдержки (case_metrics v2.1)",
    version="3.0.0"
)

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
