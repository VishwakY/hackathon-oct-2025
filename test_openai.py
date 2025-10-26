# test_openai.py
import os
from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")
model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

print("Using model:", model)
if not api_key:
    print("ERROR: OPENAI_API_KEY not set in this terminal session.")
    raise SystemExit(1)

try:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say 'hello'"}],
        temperature=0.0,
    )
    print("OK. Model responded:", resp.choices[0].message.content)
except Exception as e:
    print("OpenAI call failed:", repr(e))
    raise