"""
ingestion/embedder.py
Generates PubMedBERT embeddings and stores them in ChromaDB (primary)
and FAISS (secondary fast-search) vector stores.
"""

import os
import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import faiss

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    CHROMA_DB_PATH, COLLECTION_NAME, PROCESSED_DATA_PATH
)

# Load embedding model once at module level (reused across calls)
print(f"Loading embedding model: {EMBEDDING_MODEL}")
embed_model = SentenceTransformer(EMBEDDING_MODEL)
EMBEDDING_DIM = embed_model.get_sentence_embedding_dimension()


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

def get_chroma_collection():
    """Return (or create) the ChromaDB persistent collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def build_chroma_store(chunks: list[dict]) -> chromadb.Collection:
    """
    Embed all chunks with PubMedBERT and upsert into ChromaDB.
    Safe to re-run — existing IDs are overwritten.
    """
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    collection = get_chroma_collection()

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {k: v for k, v in c.items() if k not in ("text",) and isinstance(v, (str, int, float, bool))}
        for c in chunks
    ]

    print(f"Generating embeddings for {len(texts)} chunks (batch={EMBEDDING_BATCH_SIZE})…")
    embeddings = embed_model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    # Upsert in batches of 500 (ChromaDB limit)
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.upsert(
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  Upserted {min(end, len(chunks))}/{len(chunks)}")

    print(f"ChromaDB collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")
    return collection


# ---------------------------------------------------------------------------
# FAISS (secondary index)
# ---------------------------------------------------------------------------

FAISS_INDEX_PATH = os.path.join(PROCESSED_DATA_PATH, "faiss.index")
FAISS_META_PATH = os.path.join(PROCESSED_DATA_PATH, "faiss_meta.pkl")


def build_faiss_index(chunks: list[dict]) -> faiss.Index:
    """Build a FAISS flat inner-product index from chunk embeddings."""
    texts = [c["text"] for c in chunks]
    print(f"Building FAISS index for {len(texts)} chunks…")
    embeddings = embed_model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product = cosine on normalized vecs
    index.add(embeddings)

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save metadata alongside index
    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"FAISS index saved ({index.ntotal} vectors) → {FAISS_INDEX_PATH}")
    return index


def load_faiss_index():
    """Load FAISS index and metadata from disk."""
    if not Path(FAISS_INDEX_PATH).exists():
        raise FileNotFoundError("FAISS index not found. Run build_faiss_index() first.")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# ---------------------------------------------------------------------------
# Query helpers (used by retriever)
# ---------------------------------------------------------------------------

def embed_query(query: str) -> np.ndarray:
    """Embed a single query string, return normalized float32 vector."""
    vec = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    return vec


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.chunker import load_chunks

    print("=== Loading chunks ===")
    chunks = load_chunks()

    print("\n=== Building ChromaDB vector store ===")
    build_chroma_store(chunks)

    print("\n=== Building FAISS index ===")
    build_faiss_index(chunks)

    print("\n=== Embedding complete ===")
