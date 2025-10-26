from pathlib import Path

def load_raw_docs(data_dir: str):
    docs = []
    for p in Path(data_dir).glob("*.txt"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        docs.append({"doc_id": p.stem, "text": text, "source": str(p)})
    return docs