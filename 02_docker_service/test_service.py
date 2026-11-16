import json
import sys
import os

# Add directory containing this script to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, CallInput, ENGINE

sample_call = {
    "call_id": "test_001",
    "started_at": "2026-11-15T14:30:00",
    "turns": [
        {"idx": 0, "speaker": "operator", "text": "Здравствуйте, техподдержка МТС, чем могу помочь?"},
        {"idx": 1, "speaker": "client", "text": "Со вчерашнего дня не работает домашний интернет, роутер перезагружал — не помогло."},
        {"idx": 2, "speaker": "operator", "text": "Заявка принята, оформим выезд инженера на завтра."},
        {"idx": 3, "speaker": "client", "text": "Хорошо, договорились."}
    ]
}

print("=== Testing ENGINE.process_call in point1_cpu mode ===")
res = ENGINE.process_call(sample_call, mode="point1_cpu")
print("Call ID:", res.get("call_id"))
print(f"Total facts extracted: {len(res.get('facts', []))}")
for i, f in enumerate(res.get("facts", [])):
    print(f"  Fact {i+1}: attr={f.get('attribute')} | val='{f.get('value')}' | span={f.get('source', {}).get('span')} | quote='{f.get('source', {}).get('quote')}'")

assert len(res.get("facts", [])) > 0, "Error: No facts extracted!"
# Verify honest span
for f in res.get("facts", []):
    sp = f.get("source", {}).get("span")
    q = f.get("source", {}).get("quote")
    if sp and q:
        assert sp[1] - sp[0] == len(q), f"Span mismatch: span={sp}, len(quote)={len(q)}"
print("[OK] All assertions passed successfully! Honest spans verified.")
