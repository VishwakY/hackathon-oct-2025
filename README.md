### SEC Filing Summarizer & Q&A (RAG) — Gemini-only

#### Overview
This app answers questions about SEC filings using Retrieval-Augmented Generation (RAG). It indexes local text files (excerpts) and uses:
- Retrieval: ChromaDB + Sentence Transformers (`all-MiniLM-L6-v2`)
- Reranker: BAAI/bge-reranker-base (FlagEmbedding)
- LLM: Gemini 2.0 via LangChain (`gemini-2.0-flash` or `gemini-2.0-flash-thinking-exp`)
- JSON-only answers with citations

It returns structured JSON:
- `answer` (string)
- `citations` (array of `{doc_id, chunk_id, preview?}`)
- `confidence` (0–1)

#### Prerequisites
- Windows/PowerShell (guide below uses PS)
- Python 3.12+
- Google Gemini API Key (required)

### Submission Notes
- Scope: Gemini-only RAG system that answers questions over SEC filing excerpts with citations and confidence.
- Data: Local .txt excerpts in /data (e.g., supply chain risks, guidance, next-quarter outlook, capex).
- Pipeline: ChromaDB embeddings → Cross-encoder reranker (BAAI/bge-reranker-base) → Gemini 2.0 (LangChain) with a strict JSON-only prompt.
- Output Contract: answer (string), citations (array of {doc_id, chunk_id, preview}), confidence (0–1).
- Demo UI: Served at /ui; displays answer, confidence, citations table, and raw JSON.
- Reliability: Anchors added for key queries (supply chain risks, next-quarter outlook, capex). Rebuild index after any data changes.
- Env: Requires GEMINI_API_KEY; optional GEMINI_MODEL override (default gemini-2.0-flash).
- Known Constraints: Relies on the presence and quality of the ingested text. If context lacks coverage, the model correctly returns “I don’t know…”.
- Performance: First run downloads models; subsequent queries are fast. Reranker can be disabled for speed if needed.
- Testing: scripts/eval_questions.py runs multiple queries against /ask and saves ai_logs/results.jsonl.

#### Setup (Windows / PowerShell)
```powershell
# 1) Create/activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install --upgrade pip
pip install uvicorn fastapi chromadb sentence-transformers "langchain-google-genai==2.0.8" "google-generativeai>=0.8.0,<0.9.0" "protobuf>=5.29,<6.0" FlagEmbedding requests

# 3) Run the Program:
python build_index.py (Rebuild index any time data changes)
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

# Optional: speed up Hugging Face downloads or silence symlink warning
# $env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
# pip install "huggingface_hub[hf_xet]"
