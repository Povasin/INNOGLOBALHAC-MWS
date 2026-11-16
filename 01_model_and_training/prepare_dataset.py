import json
import os
from datetime import datetime
from pathlib import Path

def get_path(filename):
    if os.path.exists(filename):
        return filename
    alt = os.path.join("Данные", filename)
    if os.path.exists(alt):
        return alt
    return filename

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def iso_date(raw, fallback=None):
    if raw is None:
        return fallback
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw).isoformat()
        except ValueError:
            pass
    return fallback

def compact_facts(facts, turns_dict):
    out = []
    for f in facts:
        obj = f.get('object', {})
        src = f.get('source', {})
        idx = src.get('turn_idx', 0)
        # Проверяем, что индекс реплики реально существует в звонке
        if idx not in turns_dict:
            continue
            
        kind = obj.get('type', 'other')
        name = obj.get('name', '')
        attr = f.get('attribute', '')
        val = f.get('value', '')
        pol = f.get('polarity', 1)
        vf = iso_date(f.get('valid_from'))
        vt = iso_date(f.get('valid_to'))
        
        out.append([kind, name, attr, val, pol, vf, vt, idx])
    return out

def prepare_sft_dataset(input_jsonl='train.jsonl', output_jsonl='train_sft.jsonl'):
    input_path = get_path(input_jsonl)
    print(f"Превращаем {input_path} в формат компактного SFT обучения...")
    calls = load_jsonl(input_path)
    
    sft_examples = []
    total_facts = 0
    
    SYSTEM_PROMPT = (
        "Извлекай подтверждённые факты из звонков техподдержки.\n"
        "Выведи только JSON-массив записей, каждая запись имеет 8 полей в таком порядке:\n"
        "[тип объекта, имя объекта, атрибут, значение, полярность, valid_from, valid_to, индекс реплики].\n"
        "Пример формата: [[\"service\",\"Домашний интернет\",\"service_state\",\"не работает\",1,\"2026-01-01\",null,3]]\n"
        "Каждая запись — массив из восьми значений. Не добавляй ключи, Markdown или пояснения.\n"
        "Типы: service, device, client, ticket, other.\n"
        "Атрибуты: service_state, tariff, ticket_status, action, agreement.\n"
        "Указывай источник по индексу реплики, где озвучен факт. Если фактов нет, выведи []."
    )
    
    for call in calls:
        turns = call.get('turns', [])
        turns_dict = {t['idx']: t for t in turns}
        facts = call.get('facts', [])
        
        if not facts:
            continue
            
        # Формируем текст диалога
        dialogue_lines = [f"Начало звонка: {call.get('started_at', '2026-01-01T00:00:00')}"]
        dialogue_lines.extend(f"{t['idx']} {t['speaker']}: {t['text']}" for t in turns)
        user_content = "\n".join(dialogue_lines)
        
        # Упаковываем факты в плоский компактный массив
        target_list = compact_facts(facts, turns_dict)
        if not target_list:
            continue
            
        assistant_content = json.dumps(target_list, ensure_ascii=False, separators=(',', ':'))
        
        # Формируем структуру чата (совместимую с ChatTemplate для Qwen)
        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        sft_examples.append(example)
        total_facts += len(target_list)
        
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for ex in sft_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            
    print(f"Готово! Создано {len(sft_examples)} обучающих примеров")
    print(f"Всего упаковано фактов: {total_facts}")
    print(f"Результат сохранен в {output_jsonl}")

if __name__ == '__main__':
    prepare_sft_dataset()
