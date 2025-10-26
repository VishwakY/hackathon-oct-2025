# scripts/build_index.py
import os
import glob
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# scripts/build_index.py
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
DATA_DIR = os.getenv("DATA_DIR", "data")
PERSIST_DIR = os.getenv("PERSIST_DIR", ".chroma")       # <- this is the persist dir on disk
COLLECTION = os.getenv("COLLECTION", "sec_filings")     # <- this is the collection name

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+size]
        chunks.append(" ".join(chunk_words))
        i += size - overlap if (size - overlap) > 0 else size
    return chunks

def main():
    os.makedirs(PERSIST_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(allow_reset=False))
    # Create or get collection
    try:
        col = client.get_collection(COLLECTION)
    except Exception:
        col = client.create_collection(COLLECTION)

    embedder = SentenceTransformer(EMBED_MODEL)

    txt_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not txt_files:
        print(f"[build_index] No .txt files found in {DATA_DIR}")
        return

    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metas: List[Dict[str, Any]] = []

    for path in txt_files:
        doc_id = os.path.basename(path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if not text:
            print(f"[build_index] Skipping empty file: {path}")
            continue
        chunks = chunk_text(text)
        for idx, ch in enumerate(chunks):
            # Force chunk_id to int in metadata
            meta = {"doc_id": doc_id, "chunk_id": int(idx)}
            all_ids.append(f"{doc_id}:::{idx}")
            all_docs.append(ch)
            all_metas.append(meta)

    print(f"[build_index] Adding {len(all_docs)} chunks to collection '{COLLECTION}'")

    # Compute embeddings
    embs = embedder.encode(all_docs, normalize_embeddings=True)
    # Add to chroma
    col.add(documents=all_docs, embeddings=embs, metadatas=all_metas, ids=all_ids)

    print("[build_index] Done.")

if __name__ == "__main__":
    main()