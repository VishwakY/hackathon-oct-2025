# scripts/extract_kaggle_to_txt.py
import csv
from pathlib import Path

"""
This script reads a CSV from the Kaggle SEC Filings dataset and writes
a small set of short .txt excerpts into ./data for your RAG pipeline.

Adjust the CSV filename and column names below to match the dataset structure.
"""

# 1) Point to your extracted Kaggle dataset CSV here
# Example guesses: 'kaggle_sec_filings/filings.csv' or 'kaggle_sec_filings/sec_filings.csv'
CSV_PATHS = [
    Path("kaggle_sec_filings") / "sec_filings.csv",
    Path("kaggle_sec_filings") / "filings.csv",
]

# 2) Column names guess (adjust if needed after you inspect the CSV header)
CANDIDATE_TEXT_COLS = ["text", "section_text", "content", "filing_text", "body"]
CANDIDATE_COMPANY_COLS = ["company", "company_name", "ticker", "cik", "issuer"]
CANDIDATE_FORM_COLS = ["form_type", "form", "formtype"]
CANDIDATE_SECTION_COLS = ["section", "section_name", "item"]

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

def find_csv():
    for p in CSV_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find CSV. Looked for: {', '.join(map(str, CSV_PATHS))}. "
        f"Please update CSV_PATHS in this script to match your dataset."
    )

def sniff_columns(header):
    def pick(cands):
        for c in cands:
            if c in header:
                return c
        return None
    text_col = pick(CANDIDATE_TEXT_COLS)
    company_col = pick(CANDIDATE_COMPANY_COLS)
    form_col = pick(CANDIDATE_FORM_COLS)
    section_col = pick(CANDIDATE_SECTION_COLS)
    return text_col, company_col, form_col, section_col

def slugify(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")

def main():
    csv_path = find_csv()
    print(f"Reading: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        print("CSV columns:", header)

        text_col, company_col, form_col, section_col = sniff_columns(header)
        if not text_col:
            raise RuntimeError("Could not identify a text/content column. Please adjust CANDIDATE_TEXT_COLS list.")

        print(f"Using columns -> text: {text_col}, company: {company_col}, form: {form_col}, section: {section_col}")

        written = 0
        max_files = 12  # write up to 12 excerpts for quick iteration
        for row in reader:
            text = (row.get(text_col) or "").strip()
            if not text:
                continue

            company = row.get(company_col) or "company"
            form = row.get(form_col) or "form"
            section = row.get(section_col) or "section"

            # Keep excerpts short (around 1–3 paragraphs)
            excerpt = text
            if len(excerpt) > 3000:
                excerpt = excerpt[:3000] + "..."

            fname = f"{slugify(company)}_{slugify(form)}_{slugify(section)}_{written}.txt"
            (OUT_DIR / fname).write_text(excerpt, encoding="utf-8")
            written += 1

            if written >= max_files:
                break

    print(f"Wrote {written} excerpt file(s) to {OUT_DIR}/")
    if written == 0:
        print("No files were written. Verify CSV path and column names and try again.")

if __name__ == "__main__":
    main()