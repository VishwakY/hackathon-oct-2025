# scripts/fetch_and_extract_filings.py
import csv
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from readability import Document

CSV_PATH = Path("kaggle_sec_filings") / "sec_filings.csv"
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

# Config
MAX_FILES = 10               # how many excerpts to write
FORMS_TO_KEEP = {"10-K", "10-Q"}  # restrict to these forms
MIN_PAR_LEN = 200            # minimum paragraph length to consider
MAX_CHARS = 2800             # max chars per excerpt file

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HackathonRAGBot/1.0; +https://example.com)"
}

def clean_text(txt: str) -> str:
    # Basic whitespace cleanup
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def html_to_text(html: str) -> str:
    # Use Readability to get main content, fallback to raw BeautifulSoup text
    try:
        doc = Document(html)
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, "html.parser")
        text = soup.get_text(separator="\n")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
    # Remove excessive blank lines, keep paragraphs
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)

def pick_paragraphs(full_text: str, limit_chars: int = MAX_CHARS) -> str:
    # Split by paragraph-like breaks
    paras = [p.strip() for p in re.split(r"\n{2,}|\r\n\r\n", full_text) if p.strip()]
    # Prefer medium-length paragraphs
    selected = []
    total = 0
    for p in paras:
        if len(p) < MIN_PAR_LEN:
            continue
        if "forward-looking" in p.lower():
            continue
        selected.append(p)
        total += len(p)
        if total >= limit_chars:
            break
    if not selected and paras:
        # fallback: take the first non-empty paragraphs until limit
        for p in paras:
            if p:
                selected.append(p)
                total += len(p)
                if total >= limit_chars:
                    break
    return "\n\n".join(selected)

def slugify(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Make sure you expanded sec-filings.zip to kaggle_sec_filings/")

    written = 0
    with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if written >= MAX_FILES:
                break

            company = (row.get("Company Name") or "").strip() or "company"
            form_type = (row.get("Form Type") or row.get("Filing Type") or "").strip()
            filing_url = (row.get("Filing URL") or "").strip()
            accession = (row.get("Accession No") or "").strip()

            if not filing_url:
                continue
            if FORMS_TO_KEEP and form_type not in FORMS_TO_KEEP:
                continue

            try:
                resp = requests.get(filing_url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                html = resp.text
                text = html_to_text(html)
                text = clean_text(text)
                excerpt = pick_paragraphs(text, limit_chars=MAX_CHARS)
                if not excerpt or len(excerpt) < 120:
                    # If nothing useful extracted, skip
                    continue

                name = f"{slugify(company)}_{slugify(form_type)}_{slugify(accession) or written}.txt"
                (OUT_DIR / name).write_text(excerpt, encoding="utf-8")
                written += 1
                print(f"Wrote: {name} ({len(excerpt)} chars)")

                # Be polite to EDGAR/mirrors
                time.sleep(1.2)
            except Exception as e:
                print(f"Skip {filing_url}: {e}")

    print(f"Done. Wrote {written} files to {OUT_DIR}/")
    if written == 0:
        print("No files written. Consider relaxing FORMS_TO_KEEP, or increase MAX_FILES, or inspect a Filing URL manually.")

if __name__ == "__main__":
    main()