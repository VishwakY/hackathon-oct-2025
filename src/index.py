from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150):
    # Normalize newlines
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        # Step back by overlap to create overlap with next chunk
        start = max(0, end - overlap)
    return chunks

def build_index(docs, collection_name="sec_filings", persist_dir=".chroma", model_name="all-MiniLM-L6-v2"):
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name)
    embedder = SentenceTransformer(model_name)
    ids, texts, metadatas = [], [], []
    for d in docs:
        chunks = chunk_text(d["text"])
        for i, c in enumerate(chunks):
            ids.append(f'{d["doc_id"]}_{i}')
            texts.append(c)
            metadatas.append({"doc_id": d["doc_id"], "chunk_id": i, "source": d.get("source", "")})
    embs = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    col.add(ids=ids, metadatas=metadatas, documents=texts, embeddings=embs)
    return col

import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path=".chroma", settings=Settings(allow_reset=False))

# Reset the old collection so we only have fresh chunks
try:
    client.delete_collection("sec_filings")
    print("Deleted old collection: sec_filings")
except Exception:
    pass

col = client.get_or_create_collection("sec_filings")

from pathlib import Path

def build_index():
    data_dir = Path("data")
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        print("No .txt files found in data/. Add files and run again.")
        return

    total_chunks = 0
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        doc_id = p.stem  # filename without .txt
        chunks = chunk_text(text, chunk_size=1000, overlap=150)

        if not chunks:
            print(f"Skipping {p.name}: no chunks")
            continue

        documents = []
        metadatas = []
        ids = []
        for i, ch in enumerate(chunks):
            documents.append(ch)
            metadatas.append({"doc_id": doc_id, "chunk_id": i})
            ids.append(f"{doc_id}_{i}")

        col.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Indexed {doc_id}: {len(chunks)} chunks.")
        total_chunks += len(chunks)

    print(f"Done. Total chunks indexed: {total_chunks}")

if __name__ == "__main__":
    build_index()