# scripts/eval_questions.py
import json
import time
import requests
from pathlib import Path

API_URL = "http://localhost:8000/ask"
OUT_DIR = Path("ai_logs")
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "results.jsonl"

QUESTIONS = [
    "What are the main revenue drivers?",
    "What supply chain risks are mentioned?",
    "What does management expect for next quarter?",
    "Which regions contributed most to growth?",
    "How did operating margin change year over year?",
    "Summarize guidance for the upcoming quarter.",
    "What are the key risk factors mentioned?",
    "What does the company say about capital expenditures?",
    "What FX headwinds or tailwinds are noted?",
    "What segments performed best this period?"
]

def ask(q):
    t0 = time.time()
    r = requests.get(API_URL, params={"question": q}, timeout=60)
    dt = time.time() - t0
    r.raise_for_status()
    return r.json(), dt

def main():
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for q in QUESTIONS:
            try:
                res, latency = ask(q)
                row = {
                    "question": q,
                    "latency_sec": round(latency, 3),
                    "answer": res.get("answer"),
                    "citations": res.get("citations", []),
                    "confidence": res.get("confidence"),
                }
                f.write(json.dumps(row) + "\n")
                print(f"OK: {q} ({latency:.2f}s)")
            except Exception as e:
                print(f"FAIL: {q} -> {e}")

    print(f"Wrote: {OUT_PATH}")

if __name__ == "__main__":
    main()