# quick_check_chroma.py
import chromadb
from chromadb.config import Settings

PERSIST_DIR = ".chroma"         # <- set to whatever your app uses
COLLECTION = "sec_filings"      # <- set to whatever your app uses

client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(allow_reset=False))
try:
    col = client.get_collection(COLLECTION)
except Exception as e:
    print("Collection missing:", e)
    raise SystemExit()

count = col.count()
print("Count:", count)
if count == 0:
    print("Collection is empty.")
else:
    res = col.get(include=["metadatas", "documents", "ids"], limit=3)
    print("Sample IDs:", res.get("ids"))
    print("Sample metas:", res.get("metadatas"))
    print("Sample docs snippet:", [ (d[:120] + "…") if d and len(d)>120 else d for d in res.get("documents") ])