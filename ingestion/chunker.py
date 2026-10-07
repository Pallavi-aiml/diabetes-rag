"""
ingestion/chunker.py
Splits raw documents into overlapping chunks suitable for RAG retrieval.
"""

import os
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSED_DATA_PATH


def chunk_documents(documents: list[dict],
                    chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split a list of document dicts into smaller, overlapping text chunks.
    Each chunk inherits the source metadata from its parent document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        text = doc.get("text", "").strip()
        if not text or len(text) < 50:
            continue

        splits = splitter.split_text(text)
        for i, split_text in enumerate(splits):
            if len(split_text.strip()) < 30:
                continue
            chunk_id = f"{doc['source'].replace(':', '_').replace('/', '_')}__chunk_{i:04d}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": split_text.strip(),
                "source": doc.get("source", "unknown"),
                "title": doc.get("title", ""),
                "journal": doc.get("journal", "Unknown"),
                "year": doc.get("year", 2000),
                "source_type": doc.get("source_type", "unknown"),
                "page": doc.get("page", 0),
                "pmid": doc.get("pmid", "N/A"),
                "chunk_index": i,
            })

    return chunks


def save_chunks(chunks: list[dict], filename: str = "chunks.json") -> str:
    """Persist chunks to disk for later embedding."""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    out_path = Path(PROCESSED_DATA_PATH) / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks → {out_path}")
    return str(out_path)


def load_chunks(filename: str = "chunks.json") -> list[dict]:
    """Load previously saved chunks from disk."""
    fpath = Path(PROCESSED_DATA_PATH) / filename
    if not fpath.exists():
        raise FileNotFoundError(f"Chunks file not found: {fpath}. Run ingestion first.")
    with open(fpath, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {fpath}")
    return chunks


if __name__ == "__main__":
    from loader import load_all_documents

    print("=== Loading documents ===")
    docs = load_all_documents()
    if not docs:
        print("No documents found. Run loader.py first.")
        exit(1)

    print(f"\n=== Chunking {len(docs)} documents ===")
    chunks = chunk_documents(docs)
    print(f"Generated {len(chunks)} chunks")

    save_chunks(chunks)
    print("Done.")
