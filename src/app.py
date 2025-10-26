# src/app.py
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from src.rag_service import RAGService

app = FastAPI(title="SEC Filing Q&A (RAG)")

# CORS (allow local UI usage)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the /ui static directory
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

# Initialize RAG service (loads Chroma, reranker, and LLM)
rag = RAGService(collection_name="sec_filings", persist_dir=".chroma")

class AskRequest(BaseModel):
    question: str
    k: Optional[int] = 12


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ask")
def ask_get(question: str = Query(..., description="Your question"), k: int = 12):
    try:
        result = rag.answer(question=question, k=k)
        # ensure only expected keys returned
        safe = {
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "confidence": result.get("confidence", 0.0),
        }
        return JSONResponse(content=safe)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask")
def ask_post(body: AskRequest):
    try:
        result = rag.answer(question=body.question, k=body.k or 12)
        safe = {
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "confidence": result.get("confidence", 0.0),
        }
        return JSONResponse(content=safe)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Optional: root redirect (nice touch) -> open UI if someone hits "/"
@app.get("/")
def root():
    return {"message": "Go to /ui to use the demo UI, or /docs for API docs."}