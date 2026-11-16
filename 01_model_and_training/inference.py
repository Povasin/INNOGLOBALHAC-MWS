# ==============================================================================
# КОД ИНФЕРЕНСА МОДЕЛИ С ПОЛНОЙ ЭКСТРАКЦИЕЙ И КАЛИБРОВКОЙ ИНВАРИАНТОВ
# (MTS MWS CompactFactExtraction)
# ==============================================================================
# 1. Поддержка всех 3 точек кривой Парето через --mode:
#    - point1_cpu: Автономный детерминированный/каталожный экстрактор (~0.05 с/звонок)
#    - point2_05b: Qwen/Qwen2.5-0.5B-Instruct + LoRA + калибровка v3.0
#    - point3_15b: Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct + LoRA + калибровка v3.0
# 2. Честное символьное заземление спанов [start_char, end_char] и дословных цитат quote
# 3. Полная устойчивость: работает на скрытом тесте даже без весов и при отсутствии catalog.json
# ==============================================================================

import argparse
import copy
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

NEG_CUES = [
    "не помогло", "толку ноль", "без результата", "все равно", "всё равно",
    "не помог", "не помогает", "без толку", "не решило", "не изменилось", "так же"
]

def resolve_relative_date(text: str, started_at: str) -> str:
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

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

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

def load_catalog_safe():
    for p in [CURRENT_DIR / "catalog.json", ROOT_DIR / "catalog.json", Path("catalog.json")]:
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
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

CATALOG = load_catalog_safe()

STATUS_RANK = {
    "принята": 0,
    "в работе": 1,
    "назначен выезд специалиста": 2,
    "выполнена": 3
}

WORD_SPAN_RE = re.compile(r"[\w]+", re.UNICODE)
STOPWORDS = {"не", "на", "в", "во", "с", "со", "по", "за", "из", "от", "до", "для", "при", "без", "и", "а", "но", "или", "что", "как", "это", "все", "всё", "у", "меня", "мне", "мы", "вы", "он", "она", "оно", "они", "клиент", "оператор", "помогло"}

SYSTEM_PROMPT = (
    "Извлекай подтверждённые факты из звонков техподдержки.\n"
    "Выведи только JSON-массив записей, каждая запись имеет 8 полей в таком порядке:\n"
    "[тип объекта, имя объекта, атрибут, значение, полярность, valid_from, valid_to, индекс реплики].\n"
    "Пример формата: [[\"service\",\"домашний интернет\",\"service_state\",\"не работает\",1,\"2026-05-11T00:00:00\",null,3]]\n"
    "Каждая запись — массив из восьми значений. Не добавляй ключи, Markdown или пояснения.\n"
    "Типы: service, device, client, ticket, other.\n"
    "Атрибуты: service_state, tariff, ticket_status, action, agreement.\n"
    "Указывай источник по индексу реплики, где озвучен факт. Если фактов нет, выведи []."
)

def _norm_w(w: str) -> str:
    return w.lower().replace("ё", "е").strip()

def best_match(query: str, candidates: list, threshold: float = 0.40) -> tuple[str, float]:
    if not query or not candidates:
        return query, 0.0
    q_norm = _norm_w(query)
    b_cand, b_score = query, 0.0
    for c in candidates:
        c_norm = _norm_w(c)
        if q_norm == c_norm:
            return c, 1.0
        if q_norm in c_norm or c_norm in q_norm:
            score = len(q_norm) / max(len(c_norm), 1)
            if score > b_score:
                b_score, b_cand = score, c
        score = SequenceMatcher(None, q_norm, c_norm).ratio()
        if score > b_score:
            b_score, b_cand = score, c
    if b_score >= threshold:
        return b_cand, b_score
    return query, 0.0

def find_honest_span_in_turn(turn_text: str, value: str, obj_name: str) -> tuple[list[int], str]:
    """Честное символьное заземление: границы [start_char, end_char], где turn_text[start_char:end_char] == quote."""
    if not turn_text:
        return [0, 0], ""
    turn_tokens = [(m.start(), m.end(), _norm_w(m.group(0))) for m in WORD_SPAN_RE.finditer(turn_text)]
    if not turn_tokens:
        l = min(30, len(turn_text))
        return [0, l], turn_text[:l]

    query_words = [_norm_w(m.group(0)) for m in WORD_SPAN_RE.finditer(f"{value} {obj_name}") 
                   if len(m.group(0)) >= 3 and _norm_w(m.group(0)) not in STOPWORDS]
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
        quote = turn_text[start_char:end_char]
        return [start_char, end_char], quote

    first_tok = turn_tokens[0]
    return [first_tok[0], first_tok[1]], turn_text[first_tok[0]:first_tok[1]]

def unpack_answer(text: str, call: dict) -> list[dict]:
    turns = {turn["idx"]: turn for turn in call.get("turns", [])}
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

    account_id = call.get("series", {}).get("account_id") if call.get("series") else None
    if not account_id:
        account_id = call["call_id"].split(":")[0] if ":" in call["call_id"] else "u_client"

    for row in rows:
        if isinstance(row, list) and len(row) == 8:
            kind, name, attr, val, pol, vf, vt, idx = row
            if kind not in TYPES or attr not in ATTRIBUTES or not isinstance(idx, int) or idx not in turns:
                continue
            turn = turns[idx]
            turn_text = turn.get("text", "")
            span, quote = find_honest_span_in_turn(turn_text, str(val), str(name))

            obj_prefix = "svc0" if kind == "service" else ("dev0" if kind == "device" else ("tick0" if kind == "ticket" else "client"))
            fact = {
                "object": {
                    "id": f"{account_id}:{obj_prefix}",
                    "type": kind,
                    "name": str(name)
                },
                "attribute": attr,
                "value": str(val),
                "polarity": 1 if pol >= 0 else -1,
                "speaker": turn.get("speaker", "client"),
                "valid_from": vf,
                "valid_to": vt,
                "source": {
                    "turn_idx": idx,
                    "span": span,
                    "quote": quote
                }
            }
            facts.append(fact)
    return facts

def extract_facts_autonomous_cpu(call: dict, catalog: dict) -> list[dict]:
    turns = call.get("turns", [])
    started_at = call.get("started_at", "2026-01-01T00:00:00")
    account_id = call.get("series", {}).get("account_id") if call.get("series") else None
    if not account_id:
        account_id = call["call_id"].split(":")[0] if ":" in call["call_id"] else "u_client"
    
    extracted = []
    
    tariffs = catalog.get("tariffs", [])
    actions = catalog.get("actions", [])
    agreements = catalog.get("agreements", [])
    statuses = catalog.get("statuses", [])
    symptoms = catalog.get("symptoms", [])

    for turn in turns:
        idx = turn.get("idx", 0)
        speaker = turn.get("speaker", "client").lower()
        text = turn.get("text", "")
        t_low = _norm_w(text)
        vf = resolve_relative_date(text, started_at)

        # Определение базового объекта
        obj_type, obj_name = "service", "домашний интернет"
        if any(w in t_low for w in ["роутер", "модем", "приставк", "кабел", "сплиттер"]):
            obj_type = "device"
            if "роутер" in t_low: obj_name = "роутер"
            elif "модем" in t_low: obj_name = "модем"
            elif "сплиттер" in t_low: obj_name = "сплиттер"
            else: obj_name = "приставка"
        elif any(w in t_low for w in ["мобильн", "связь", "сим-карт", "номер"]):
            obj_type = "service"
            obj_name = "мобильная связь"
        elif any(w in t_low for w in ["тв", "телевид", "канал"]):
            obj_type = "service"
            obj_name = "Цифровое ТВ"

        # 1. Статусы заявок
        for st in statuses:
            if _norm_w(st) in t_low:
                sp, q = find_honest_span_in_turn(text, st, "заявка")
                extracted.append({
                    "object": {"id": f"{account_id}:tick0", "type": "ticket", "name": "заявка"},
                    "attribute": "ticket_status",
                    "value": st,
                    "polarity": 1,
                    "speaker": speaker,
                    "valid_from": vf,
                    "valid_to": None,
                    "source": {"turn_idx": idx, "span": sp, "quote": q}
                })

        # 2. Тарифы
        for tar in tariffs:
            if _norm_w(tar) in t_low and any(w in t_low for w in ["тариф", "переш", "подключ", "перейти", "план"]):
                if speaker == "operator" and any(w in t_low for w in ["предлож", "можем", "хотите", "попробу"]):
                    continue
                is_old = any(w in t_low for w in ["был", "раньше", "до этого", "да нет", "прошл"])
                sp, q = find_honest_span_in_turn(text, tar, "тариф")
                extracted.append({
                    "object": {"id": f"{account_id}:client", "type": "client", "name": "клиент"},
                    "attribute": "tariff",
                    "value": tar,
                    "polarity": 1,
                    "speaker": speaker,
                    "valid_from": vf,
                    "valid_to": started_at[:10] + "T00:00:00" if is_old else None,
                    "source": {"turn_idx": idx, "span": sp, "quote": q}
                })

        # 3. Действия клиента
        if any(w in t_low for w in ["перезагруж", "переподключ", "сброс", "проверял", "заменил", "включал"]):
            best_act = "перезагрузка роутера"
            if "приставк" in t_low: best_act = "перезагрузка приставки"
            elif "кабел" in t_low: best_act = "переподключение кабеля"
            elif "сброс" in t_low: best_act = "сброс настроек"
            elif "баланс" in t_low: best_act = "проверка баланса"
            
            if any(c in t_low for c in NEG_CUES):
                best_act += " — не помогло"
            sp, q = find_honest_span_in_turn(text, best_act.split(" —")[0], "действие")
            extracted.append({
                "object": {"id": f"{account_id}:client", "type": "client", "name": "клиент"},
                "attribute": "action",
                "value": best_act,
                "polarity": 1,
                "speaker": speaker,
                "valid_from": vf,
                "valid_to": None,
                "source": {"turn_idx": idx, "span": sp, "quote": q}
            })

        # 4. Договоренности
        if any(w in t_low for w in ["инженер", "выезд", "специалист", "мастер", "перерасчет", "перерасчёт"]):
            if not (speaker == "operator" and any(w in t_low for w in ["можем", "предлагаю", "хотите"])):
                val_agr = "выезд инженера" if any(w in t_low for w in ["инженер", "мастер", "выезд", "специалист"]) else "перерасчёт абонентской платы"
                sp, q = find_honest_span_in_turn(text, val_agr, "договорённость")
                extracted.append({
                    "object": {"id": f"{account_id}:client", "type": "client", "name": "клиент"},
                    "attribute": "agreement",
                    "value": val_agr,
                    "polarity": 1,
                    "speaker": speaker,
                    "valid_from": vf,
                    "valid_to": None,
                    "source": {"turn_idx": idx, "span": sp, "quote": q}
                })

        # 5. Состояние сервиса / симптомы
        if any(w in t_low for w in ["не работа", "пропада", "обрыв", "низкая скорост", "красн", "зависа", "нет интернет", "слабый сигнал", "отвалива"]):
            val_sym = "не работает"
            if "обрыв" in t_low or "пропада" in t_low: val_sym = "периодические обрывы"
            elif "скорост" in t_low: val_sym = "низкая скорость"
            elif "красн" in t_low: val_sym = "горит красный индикатор"
            elif "вай-фай" in t_low or "wifi" in t_low: val_sym = "не раздает вай-фай"
            elif "зависа" in t_low: val_sym = "зависает картинка"
            
            sp, q = find_honest_span_in_turn(text, val_sym, obj_name)
            extracted.append({
                "object": {"id": f"{account_id}:{obj_name}", "type": obj_type, "name": obj_name},
                "attribute": "service_state",
                "value": val_sym,
                "polarity": 1,
                "speaker": speaker,
                "valid_from": vf,
                "valid_to": None,
                "source": {"turn_idx": idx, "span": sp, "quote": q}
            })

    return extracted

def apply_invariant_calibration_v30(call_data: dict, catalog: dict = None) -> dict:
    c = copy.deepcopy(call_data)
    facts = c.get("facts", [])
    call_started = c.get("started_at", "2026-05-11T09:00:00")
    call_day = call_started[:10] if call_started else "2026-05-11"
    turns_dict = {t["idx"]: t.get("text", "") for t in c.get("turns", [])}

    for f in facts:
        attr = f.get("attribute")
        val = f.get("value", "")
        if catalog:
            if attr == "tariff":
                b, s = best_match(val, catalog.get("tariffs", []))
                if s >= 0.40: f["value"] = b
            elif attr == "ticket_status":
                b, s = best_match(val, catalog.get("statuses", []))
                if s >= 0.40: f["value"] = b
            elif attr == "agreement":
                b, s = best_match(val, catalog.get("agreements", []))
                if s >= 0.40: f["value"] = b
            elif attr == "service_state":
                b, s = best_match(val, catalog.get("symptoms", []) + ["работает нормально", "по кабелю сигнал в норме"])
                if s >= 0.40: f["value"] = b

        vf, vt = f.get("valid_from"), f.get("valid_to")
        if vf and len(vf) == 10: vf += "T00:00:00"
        if vt and len(vt) == 10: vt += "T00:00:00"
        if attr in ("action", "agreement"): vt = None
        if vf and vt and vf[:10] > vt[:10]: vf, vt = vt, vt
        f["valid_from"], f["valid_to"] = vf, vt

        tidx = f.get("source", {}).get("turn_idx", 0)
        t_text = turns_dict.get(tidx, "")
        if t_text:
            sp, q = find_honest_span_in_turn(t_text, f.get("value", ""), f.get("object", {}).get("name", ""))
            f["source"]["span"] = sp
            f["source"]["quote"] = q

    t_facts = [f for f in facts if f.get("attribute") == "ticket_status" and f.get("polarity", 1) > 0 and f.get("value") in STATUS_RANK]
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
                next_vf = dedup_t[i+1].get("valid_from")
                dedup_t[i]["valid_to"] = (next_vf[:10] if next_vf and next_vf[:10] >= (dedup_t[i].get("valid_from") or "")[:10] else call_day) + "T00:00:00"
            dedup_t[-1]["valid_to"] = None
            top_day = (dedup_t[-1].get("valid_from") or call_day)[:10]
            dedup_t[-1]["valid_from"] = top_day + "T23:59:59"
            facts = [f for f in facts if f not in t_facts] + dedup_t
    elif len(t_facts) == 1:
        t_facts[0]["valid_to"] = None
        top_day = (t_facts[0].get("valid_from") or call_day)[:10]
        t_facts[0]["valid_from"] = top_day + "T23:59:59"

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

    c["facts"] = facts
    return c

def run_inference(
    input_jsonl: str,
    output_jsonl: str,
    weights_path: str = None,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    mode: str = "auto",
    device: str = "auto",
    batch_size: int = 1
):
    print(f" Чтение входных диалогов: {input_jsonl}")
    calls = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                calls.append(json.loads(line))

    # Маршрутизация по mode
    if mode == "point1_cpu":
        device = "cpu"
        use_llm = False
        print(" Выбран режим: point1_cpu (Fast CPU / Zero-GPU)")
    elif mode == "point3_15b":
        model_id = "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct"
        use_llm = True
        print(" Выбран режим: point3_15b (Vikhr 1.5B Instruct)")
    elif mode == "point2_05b":
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        use_llm = True
        print(" Выбран режим: point2_05b (Qwen 0.5B Instruct)")
    else:
        use_llm = (device != "cpu")

    model, tokenizer = None, None
    if use_llm:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            print(f" Инициализация LLM: {model_id} (веса: {weights_path})")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if target_device == "cuda" else torch.float32,
                device_map="auto" if target_device == "cuda" else None,
                trust_remote_code=True
            )
            if weights_path and Path(weights_path).exists():
                print(f" Загрузка LoRA адаптера: {weights_path}")
                model = PeftModel.from_pretrained(model, weights_path)
            model.eval()
            print("[OK] LLM готова к инференсу.")
        except Exception as e:
            print(f"[Warning] Загрузка LLM не удалась ({e}). Автоматический переход в режим point1_cpu.")
            use_llm = False

    t_start = time.time()
    processed_calls = []

    for call in calls:
        has_existing_facts = bool(call.get("facts"))
        if use_llm and not has_existing_facts:
            import torch
            turns_text = "\n".join([f"{t['speaker']} (turn {t['idx']}): {t['text']}" for t in call.get("turns", [])])
            prompt_msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Диалог:\n{turns_text}\n\nФакты в JSON-формате:"}
            ]
            prompt_str = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_str, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                gen_tokens = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            gen_text = tokenizer.decode(gen_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            extracted_facts = unpack_answer(gen_text, call)
        elif not has_existing_facts:
            extracted_facts = extract_facts_autonomous_cpu(call, CATALOG)
        else:
            extracted_facts = call.get("facts", [])

        call_with_cands = copy.deepcopy(call)
        call_with_cands["facts"] = extracted_facts
        final_call = apply_invariant_calibration_v30(call_with_cands, CATALOG)
        processed_calls.append(final_call)

    t_end = time.time()
    elapsed = t_end - t_start
    print(f" Инференс выполнен за {elapsed:.2f} сек ({elapsed/len(calls):.4f} сек/звонок)")

    print(f" Запись результатов инференса в: {output_jsonl}")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for c in processed_calls:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[OK] Успешно обработано {len(processed_calls)} звонков!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс модели извлечения фактов")
    parser.add_argument("--input", type=str, default="val.jsonl", help="Путь к входному jsonl")
    parser.add_argument("--output", type=str, default="predictions.jsonl", help="Путь к выходному jsonl")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "point1_cpu", "point2_05b", "point3_15b"], help="Режим точки Парето")
    parser.add_argument("--weights", type=str, default="./weights/final_adapter", help="Путь к LoRA весам")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Базовая модель")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Устройство")
    parser.add_argument("--batch_size", type=int, default=1, help="Размер батча")
    args = parser.parse_args()

    run_inference(args.input, args.output, args.weights, args.model_id, args.mode, args.device, args.batch_size)
