"""
improved_rag_local_ollama.py

Improved RAG pipeline using local Ollama (LLaMA 3.2).
- Smaller semantic chunks (400 chars, 100 overlap)
- Per-key retrieval (search per key)
- Stronger embedding model (all-mpnet-base-v2)
- Strict JSON-schema prompts (single-key extraction)
- Key normalization and deduplication
- Robust fallback heuristic extractor
- Saves final cleaned Excel

Usage:
    1) Ensure Ollama server is running locally:
         ollama pull llama3.2
         ollama serve
    2) Install dependencies:
         pip install pdfplumber pandas sentence-transformers faiss-cpu requests numpy openpyxl
    3) Run:
         python improved_rag_local_ollama.py
"""

import os
import re
import json
import time
import logging
from typing import List, Dict

import pdfplumber
import pandas as pd
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIG
# ----------------------------
PDF_PATH = "D:\Assignment\Data_Input.pdf"  # uploaded file path (will be used as file URL)
OUTPUT_XLSX = "Output_improved.xlsx"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
EMBED_MODEL = "all-mpnet-base-v2"
TOP_K = 6  # number of chunks to retrieve per key

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ----------------------------
# Utility: Extract PDF text
# ----------------------------
def extract_pdf_text(path: str) -> str:
    logging.info("Extracting text from PDF: %s", path)
    all_text = []
    with pdfplumber.open(path) as pdf:
        for p, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                # maintain page separators for context
                all_text.append(f"[Page {p}]\n" + text)
    full = "\n\n".join(all_text)
    return full

# ----------------------------
# Utility: Chunk text (overlapping)
# ----------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)
    logging.info("Created %d chunks", len(chunks))
    return chunks

# ----------------------------
# Embeddings + FAISS helpers
# ----------------------------
def build_embeddings(chunks: List[str], model_name: str = EMBED_MODEL):
    logging.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    logging.info("Computing embeddings for %d chunks...", len(chunks))
    embs = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    return embs, model

def build_faiss_index(embs: np.ndarray):
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embs)
    index.add(embs)
    logging.info("FAISS index built (dim=%d, n=%d)", dim, embs.shape[0])
    return index

def retrieve_top_chunks(query: str, model, index, chunks: List[str], top_k: int = TOP_K) -> List[Dict]:
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append({"chunk_id": int(idx), "score": float(score), "text": chunks[idx]})
    return results

# ----------------------------
# Ollama call (local)
# ----------------------------
def ollama_generate(prompt: str, model: str = OLLAMA_MODEL, timeout: int = 60) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Ollama responses may place text in different keys; try common ones:
        if isinstance(data, dict):
            # older/newer versions vary; attempt to extract content robustly
            for key in ("response", "text", "content", "output", "result"):
                if key in data:
                    return data[key]
            # sometimes nested:
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                c = data["choices"][0]
                for k in ("message", "text", "content"):
                    if k in c:
                        return c[k]
        # fallback: stringify
        return json.dumps(data)
    except Exception as e:
        logging.error("Ollama request failed: %s", e)
        raise

# ----------------------------
# Strict per-key LLM prompt (single-key JSON)
# ----------------------------
SINGLE_KEY_PROMPT_TEMPLATE = """
You are a strict extractor. Extract ONLY the requested key from the provided context and return EXACTLY one JSON object with fields: Key, Value, Comments.

Requested Key: "{requested_key}"

Rules:
- Extract ONLY the value corresponding to the Requested Key from the context. If multiple values appear, pick the most relevant (prefer exact matches).
- Comments: include the exact sentence(s) from the context that justify the extracted value. Do NOT paraphrase the comments.
- Preserve original wording and punctuation for Value and Comments where possible.
- If the key is not present, return empty string ("") for Value and Comments.
- Output STRICTLY the JSON object and nothing else. Example:
{{"Key":"{requested_key}", "Value":"...", "Comments":"..."}}

Context:
{context}
"""

# ----------------------------
# Heuristic fallback for single key (if LLM fails or returns unparsable output)
# ----------------------------
def heuristic_single_key_extract(key: str, retrieved_chunks: List[Dict]) -> Dict:
    key_lower = key.lower()
    combined = " ".join([c["text"] for c in retrieved_chunks])
    # common heuristics for dates, numbers, currency, names
    # Dates (YYYY-MM-DD or Month DD, YYYY)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", combined) or re.search(r"([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})", combined)
    if "date" in key_lower and date_match:
        val = date_match.group(1)
        # grab sentence with date
        sent = find_sentence_with_text(combined, val)
        return {"Key": key, "Value": val, "Comments": sent}
    # Currency amounts like 2,800,000 INR or 350000 INR
    if "salary" in key_lower or "salary" in key_lower:
        m = re.search(r"([\d,]{3,}\s*(?:INR|USD|EUR)?)", combined)
        if m:
            val = m.group(1).strip()
            sent = find_sentence_with_text(combined, val)
            return {"Key": key, "Value": val, "Comments": sent}
    # simple "X is Y" patterns
    m = re.search(rf"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was born on\s+([A-Za-z]+\s+\d{{1,2}},\s*\d{{4}})", combined)
    if m and ("birth" in key_lower or "date" in key_lower):
        val = m.group(2)
        sent = find_sentence_with_text(combined, val)
        return {"Key": key, "Value": val, "Comments": sent}
    # fallback: find line containing key words
    for chunk in retrieved_chunks:
        lines = chunk["text"].splitlines()
        for ln in lines:
            if key_lower.split()[0] in ln.lower():
                # use the line as value
                return {"Key": key, "Value": ln.strip(), "Comments": ln.strip()}
    # Nothing found
    return {"Key": key, "Value": "", "Comments": ""}

def find_sentence_with_text(text: str, substring: str) -> str:
    # split into sentences and find the one containing substring
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    for s in sentences:
        if substring in s:
            return s.strip()
    # fallback: return shortened context
    return substring

# ----------------------------
# Parse single-key JSON output robustly
# ----------------------------
def parse_single_key_json(raw: str, requested_key: str) -> Dict:
    raw = raw.strip()
    # try direct JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            k = obj.get("Key", requested_key)
            v = obj.get("Value", "")
            c = obj.get("Comments", "")
            return {"Key": k, "Value": v, "Comments": c}
        # if array returned unexpectedly, pick first
        if isinstance(obj, list) and obj:
            first = obj[0]
            return {"Key": first.get("Key", requested_key), "Value": first.get("Value", ""), "Comments": first.get("Comments", "")}
    except Exception:
        # attempt to extract JSON substring
        m = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                return {"Key": obj.get("Key", requested_key), "Value": obj.get("Value", ""), "Comments": obj.get("Comments", "")}
            except Exception:
                pass
    # if cannot parse, return empty (upstream will try heuristic)
    return {"Key": requested_key, "Value": None, "Comments": None}

# ----------------------------
# Top-level single-key extraction
# ----------------------------
def extract_key_value_for_key(key: str, model, index, chunks: List[str], top_k: int = TOP_K) -> Dict:
    # Build query phrased for retrieval
    query = key  # simple; can be expanded e.g., "Find {key} in document"
    retrieved = retrieve_top_chunks(query, model, index, chunks, top_k=top_k)
    if not retrieved:
        logging.warning("No chunks retrieved for key: %s", key)
        return {"Key": key, "Value": "", "Comments": ""}
    # Build context (concatenate top chunks with separators)
    context = "\n\n---\n\n".join([f"Chunk {i+1}:\n{c['text']}" for i, c in enumerate(retrieved)])
    prompt = SINGLE_KEY_PROMPT_TEMPLATE.format(requested_key=key, context=context)
    # Call Ollama
    try:
        raw = ollama_generate(prompt)
        parsed = parse_single_key_json(raw, key)
        # If parsed values are None (parsing failed), use heuristic fallback
        if parsed["Value"] is None:
            logging.warning("LLM output unparsable for key '%s' — using heuristic.", key)
            return heuristic_single_key_extract(key, retrieved)
        # If parsed fields are empty strings, treat as not found
        if parsed["Value"].strip() == "" and parsed["Comments"].strip() == "":
            # fallback to heuristic
            logging.info("LLM returned empty for key '%s' — trying heuristic", key)
            return heuristic_single_key_extract(key, retrieved)
        return parsed
    except Exception as e:
        logging.error("Error calling Ollama for key %s: %s", key, e)
        # fallback
        return heuristic_single_key_extract(key, retrieved)

# ----------------------------
# Post-processing & normalization
# ----------------------------
def normalize_key(k: str) -> str:
    return re.sub(r"\s+", " ", k.strip()).lower()

def dedupe_and_clean_records(records: List[Dict]) -> List[Dict]:
    # Normalize keys, prefer non-empty values, merge comments if duplicates found
    store = {}
    for rec in records:
        key_norm = normalize_key(rec.get("Key", "") or "")
        if key_norm == "":
            continue
        val = (rec.get("Value") or "").strip()
        com = (rec.get("Comments") or "").strip()
        if key_norm not in store:
            store[key_norm] = {"Key": key_norm, "Value": val, "Comments": com}
        else:
            # prefer existing non-empty value, otherwise take new
            if not store[key_norm]["Value"] and val:
                store[key_norm]["Value"] = val
            # combine comments
            if com:
                existing = store[key_norm]["Comments"]
                if com not in existing:
                    store[key_norm]["Comments"] = (existing + " " + com).strip()
    # convert back and title-case keys for readability
    out = []
    for k_norm, v in store.items():
        pretty_key = k_norm.title()
        out.append({"Key": pretty_key, "Value": v["Value"], "Comments": v["Comments"]})
    # sort by Key
    out = sorted(out, key=lambda x: x["Key"])
    return out

# ----------------------------
# Main pipeline
# ----------------------------
def main():
    start_time = time.time()
    # 1) Extract text
    full_text = extract_pdf_text(PDF_PATH)
    if not full_text.strip():
        logging.error("No text extracted from PDF. Exiting.")
        return

    # 2) Chunk text
    chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

    # 3) Embeddings + FAISS
    embs, emb_model = build_embeddings(chunks)
    index = build_faiss_index(embs)

    # 4) Keys to extract (expandable)
    keys = [
        "First Name", "Last Name", "Date of Birth", "Birthdate", "Birth City", "Birth State",
        "Age", "Blood Group", "Nationality", "Citizenship status",
        "Joining Date of first professional role", "Designation of first professional role",
        "Salary of first professional role", "Salary currency of first professional role",
        "Current Organization", "Current Joining Date", "Current Designation", "Current Salary", "Current Salary Currency",
        "Previous Organization", "Previous Joining Date", "Previous end year", "Previous Starting Designation",
        "High School", "12th standard pass out year", "12th overall board score",
        "Undergraduate degree", "Undergraduate college", "Undergraduate year", "Undergraduate CGPA",
        "Graduation degree", "Graduation college", "Graduation year", "Graduation CGPA",
        "Certifications 1", "Certifications 2", "Certifications 3", "Certifications 4",
        "Technical Proficiency", "Python proficiency", "SQL proficiency", "Cloud platform expertise",
        "Data visualization skills", "Academic foundation", "Work authorization", "Visa requirements",
        "Full Text"  # fallback to capture uncaptured content if needed
    ]

    records = []
    for i, key in enumerate(keys, start=1):
        logging.info("Extracting key %d/%d: %s", i, len(keys), key)
        rec = extract_key_value_for_key(key, emb_model, index, chunks, top_k=TOP_K)
        # Ensure Key present
        if not rec.get("Key"):
            rec["Key"] = key
        records.append(rec)
        # small delay to avoid overloading local server in rapid loops
        time.sleep(0.2)

    # 5) Dedupe & clean
    cleaned = dedupe_and_clean_records(records)

    # 6) Save to Excel
    df = pd.DataFrame(cleaned, columns=["Key", "Value", "Comments"])
    df.to_excel(OUTPUT_XLSX, index=False)
    logging.info("Saved final output to %s (rows=%d)", OUTPUT_XLSX, len(df))

    elapsed = time.time() - start_time
    logging.info("Pipeline finished in %.2f seconds", elapsed)

if __name__ == "__main__":
    main()
