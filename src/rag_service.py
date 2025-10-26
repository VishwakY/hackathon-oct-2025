# src/rag_service.py
import json
import os
import re
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from FlagEmbedding import FlagReranker

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))

DEFAULT_PROMPT_FALLBACK = (
    "You are an expert assistant answering questions strictly from the provided context snippets.\n"
    "Rules:\n"
    "- Use ONLY the provided context to answer.\n"
    "- If the context is insufficient or unclear, reply exactly: \"I don't know based on the provided documents.\"\n"
    "- Do NOT add any external knowledge or assumptions.\n"
    "- Cite specific context items by their doc_id and chunk_id in an array called \"citations\".\n"
    "- Confidence should be a float in [0, 1].\n"
    "- Return only one JSON object, with no extra commentary, no Markdown, and no code fences.\n"
    "- The JSON must have exactly these keys: answer (string), citations (array of objects with doc_id and chunk_id), confidence (number).\n\n"
    "Question:\n{question}\n\n"
    "Context snippets (JSONL; each line is a JSON object with doc_id, chunk_id, text):\n{context}\n\n"
    "Now return the JSON object only."
)

def try_extract_json(txt: str) -> str:
    if not txt:
        return txt
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.S | re.I)
    if m:
        return m.group(1)
    start = txt.find("{"); end = txt.rfind("}")
    if 0 <= start < end:
        cand = txt[start:end+1].strip()
        if cand.startswith("{") and cand.endswith("}"):
            return cand
    return txt

def to_int(val, default=-1):
    try:
        return int(val)
    except Exception:
        return default

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

class RAGService:
    def __init__(self, collection_name: str = "sec_filings", persist_dir: str = ".chroma"):
        # Vector store client
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
        # Get or create collection
        try:
            self.col = self.client.get_collection(collection_name)  # or create_collection(...)
            print(f"[RAGService] Using existing Chroma collection: {collection_name}")
        except Exception:
            self.col = self.client.get_collection(collection_name)  # or create_collection(...)
            print(f"[RAGService] Created Chroma collection: {collection_name}")

        # Embeddings
        self.embedder = SentenceTransformer(EMBED_MODEL)

        # LLM (Gemini via LangChain)
        self.gemini_lc = None
        if GEMINI_API_KEY:
            try:
                self.gemini_lc = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    google_api_key=GEMINI_API_KEY,
                    temperature=LLM_TEMPERATURE,
                    max_output_tokens=LLM_MAX_TOKENS,
                )
                print(f"[RAGService] Gemini initialized: {GEMINI_MODEL}")
            except Exception as e:
                print("[RAGService] Gemini init failed:", repr(e))
        else:
            print("[RAGService] GEMINI_API_KEY not set; will use stub answers.")

        # Reranker
        self.reranker = None
        try:
            self.reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
            print("[RAGService] Reranker initialized.")
        except Exception as e:
            print("[RAGService] Reranker init failed:", repr(e))

    def retrieve(self, question: str, k: int = 12) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        res = self.col.query(query_embeddings=q_emb, n_results=k, include=["metadatas", "documents", "distances"])

        items: List[Dict[str, Any]] = []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        dists = res.get("distances") or []
        if not docs or not docs[0]:
            return items

        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            doc = doc or ""
            meta = meta or {}
            doc_id = str(meta.get("doc_id", "unknown"))
            chunk_id = to_int(meta.get("chunk_id", -1), -1)
            sim = float(1 - dist) if dist is not None else 0.0
            preview = doc.strip().replace("\n", " ")
            if len(preview) > 240:
                preview = preview[:240] + "…"
            items.append({"text": doc, "doc_id": doc_id, "chunk_id": chunk_id, "similarity": sim, "preview": preview})

        return items

    def rerank(self, question: str, retrieved: List[Dict[str, Any]], keep_top: int = 8) -> List[Dict[str, Any]]:
        if not self.reranker or not retrieved:
            return retrieved
        try:
            pairs = [(question, r["text"]) for r in retrieved]
            scores = self.reranker.compute_score(pairs, normalize=True)
            for r, s in zip(retrieved, scores):
                r["rerank"] = float(s)
            retrieved.sort(key=lambda x: x.get("rerank", 0.0), reverse=True)
            return retrieved[: min(len(retrieved), keep_top)]
        except Exception as e:
            print("[RAGService] Rerank failed:", repr(e))
            return retrieved

    def build_prompt(self, question: str, retrieved: List[Dict[str, Any]]) -> str:
        try:
            with open("prompts/answer_with_citations.txt", "r", encoding="utf-8") as f:
                tmpl = f.read()
        except Exception:
            tmpl = DEFAULT_PROMPT_FALLBACK
        ctx_lines = []
        for r in (retrieved or []):
            ctx_lines.append(json.dumps({"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "text": r["text"]}, ensure_ascii=False))
        return tmpl.format(question=question, context="\n".join(ctx_lines))

    def answer(self, question: str, k: int = 12) -> Dict[str, Any]:
        retrieved = self.retrieve(question, k=k)
        print("Top retrieved (pre-rerank):", [(r["doc_id"], r["chunk_id"], round(r["similarity"], 3)) for r in retrieved[:5]])
        retrieved = self.rerank(question, retrieved, keep_top=8)
        print("Top retrieved (post-rerank):", [(r["doc_id"], r["chunk_id"], round(r.get("similarity", 0.0), 3), round(r.get("rerank", 0.0), 3)) for r in retrieved[:5]])

        if not retrieved:
            return {"answer": "I don't know based on the provided documents.", "citations": [], "confidence": 0.0}

        prompt = self.build_prompt(question, retrieved)

        if self.gemini_lc:
            try:
                result = self.gemini_lc.invoke(prompt)
                txt = getattr(result, "content", "") or ""
                raw = try_extract_json(txt)
                if raw:
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, dict) and isinstance(obj.get("citations"), list):
                            idx = {(r["doc_id"], r["chunk_id"]): r for r in retrieved}
                            fixed = []
                            for c in obj["citations"]:
                                did = str(c.get("doc_id"))
                                cid = to_int(c.get("chunk_id", -1), -1)
                                prev = idx.get((did, cid), {}).get("preview", "")
                                fixed.append({"doc_id": did, "chunk_id": cid, "preview": prev})
                            obj["citations"] = fixed
                        if isinstance(obj, dict):
                            obj["confidence"] = safe_float(obj.get("confidence", 0.0), 0.0)
                        return obj
                    except Exception:
                        pass
                # Fallback if non-JSON or empty
                return {
                    "answer": raw.strip() or "I don't know based on the provided documents.",
                    "citations": [{"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "preview": r.get("preview", "")} for r in retrieved[:3]],
                    "confidence": 0.5 if raw.strip() else 0.0,
                }
            except Exception as e:
                print("[RAGService] Gemini call failed:", repr(e))

        # Final fallback
        return {
            "answer": "I don't know based on the provided documents.",
            "citations": [{"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "preview": r.get("preview", "")} for r in retrieved[:3]],
            "confidence": 0.0,
        }