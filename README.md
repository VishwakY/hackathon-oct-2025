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

#### Setup (Windows / PowerShell)
```powershell
# 1) Create/activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install --upgrade pip
pip install uvicorn fastapi chromadb sentence-transformers "langchain-google-genai==2.0.8" "google-generativeai>=0.8.0,<0.9.0" "protobuf>=5.29,<6.0" FlagEmbedding requests

# Optional: speed up Hugging Face downloads or silence symlink warning
# $env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
# pip install "huggingface_hub[hf_xet]"
