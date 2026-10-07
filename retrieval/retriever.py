"""
retrieval/retriever.py
Hybrid retrieval pipeline:
  1. Dense retrieval  — ChromaDB (PubMedBERT cosine similarity)
  2. Sparse retrieval — BM25 (rank-bm25)
  3. Score fusion     — Reciprocal Rank Fusion (RRF)
  4. Reranking        — cross-encoder ms-marco
  5. Relevance filter — remove low-score and off-topic chunks
  6. Source boosting  — prioritise high-quality clinical sources
"""

import os
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import chromadb

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    CHROMA_DB_PATH, COLLECTION_NAME,
    TOP_K_RETRIEVE, TOP_K_RERANK, RERANKER_MODEL
)
from ingestion.embedder import embed_query, get_chroma_collection


# ── Load reranker once ────────────────────────────────────────────────────────
print(f"Loading reranker: {RERANKER_MODEL}")
reranker = CrossEncoder(RERANKER_MODEL, max_length=512)


# ── Diabetes-specific clinical keywords for relevance filtering ───────────────
DIABETES_KEYWORDS = [
    "diabetes", "diabetic", "diabetics",
    "glucose", "glycaemic", "glycemic",
    "insulin", "metformin", "hba1c", "a1c",
    "hyperglycaemia", "hyperglycemia",
    "hypoglycaemia", "hypoglycemia",
    "type 2", "type 1", "t2dm", "t1dm",
    "pancreas", "pancreatic", "beta cell",
    "sglt2", "glp-1", "dpp-4", "gliptin",
    "sulfonylurea", "thiazolidinedione",
    "retinopathy", "nephropathy", "neuropathy",
    "ketoacidosis", "dka",
    "blood sugar", "fasting glucose",
    "obesity", "bmi", "weight loss",
    "cardiovascular", "heart failure",
    "kidney", "renal", "egfr",
    "clinical guideline", "standards of care",
    "icmr", "ada guideline", "who diabetes",
]

# Journals that are clearly off-topic for diabetes care
OFF_TOPIC_JOURNALS = [
    "journal of infectious diseases",
    "canadian journal of infectious",
    "microbiology",
    "bactericida",
    "antimicrobial",
    "virology",
    "veterinary",
    "dentistry",
    "ophthalmology",
    "dermatology",
]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    """
    Merge multiple ranked lists of chunk dicts into one via RRF.
    Each dict must have a 'chunk_id' key.
    """
    scores:    dict[str, float] = {}
    id_to_doc: dict[str, dict]  = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            cid = doc["chunk_id"]
            id_to_doc[cid] = doc
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [id_to_doc[cid] for cid in sorted_ids]


# ── Dense retrieval ───────────────────────────────────────────────────────────

def dense_retrieve(query: str, collection: chromadb.Collection,
                   n: int = TOP_K_RETRIEVE) -> list:
    """Return top-n chunks by cosine similarity from ChromaDB."""
    query_vec = embed_query(query).tolist()
    results   = collection.query(
        query_embeddings=query_vec,
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    docs = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        docs.append({
            "chunk_id": results["ids"][0][i],
            "text":     results["documents"][0][i],
            "distance": results["distances"][0][i],
            **meta,
        })
    return docs


# ── Sparse retrieval (BM25) ───────────────────────────────────────────────────

def build_bm25_index(chunks: list) -> BM25Okapi:
    """Build a BM25 index from a list of chunk dicts."""
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def bm25_retrieve(query: str, bm25_index: BM25Okapi,
                  all_chunks: list, n: int = TOP_K_RETRIEVE) -> list:
    """Return top-n chunks by BM25 score."""
    query_tokens = query.lower().split()
    scores       = bm25_index.get_scores(query_tokens)
    top_indices  = sorted(range(len(scores)),
                          key=lambda i: scores[i], reverse=True)[:n]
    return [all_chunks[i] for i in top_indices]


# ── Cross-encoder reranker ────────────────────────────────────────────────────

def rerank(query: str, docs: list, top_k: int = TOP_K_RERANK) -> list:
    """Rerank docs using a cross-encoder model. Returns top_k."""
    if not docs:
        return []
    pairs  = [(query, doc["text"][:512]) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    result = []
    for score, doc in ranked[:top_k]:
        doc["rerank_score"] = float(score)
        result.append(doc)
    return result


# ── Relevance filter ──────────────────────────────────────────────────────────

def is_diabetes_relevant(chunk: dict) -> bool:
    """
    Returns True if the chunk is relevant to diabetes care.
    Checks text content and journal name.
    """
    text    = chunk.get("text", "").lower()
    journal = chunk.get("journal", "").lower()

    # Reject clearly off-topic journals
    for off in OFF_TOPIC_JOURNALS:
        if off in journal:
            return False

    # Accept if any diabetes keyword is in the text
    for kw in DIABETES_KEYWORDS:
        if kw in text:
            return True

    return False


def filter_relevant(chunks: list, min_rerank_score: float = -5.0) -> list:
    """
    Remove chunks that are:
    1. Below minimum rerank score threshold
    2. Clearly off-topic (not diabetes-related)

    Keeps at least 2 chunks even if all are borderline.
    """
    # First pass — remove very low rerank scores
    filtered = [
        c for c in chunks
        if c.get("rerank_score", 0.0) >= min_rerank_score
    ]

    # Second pass — remove off-topic chunks
    relevant = [c for c in filtered if is_diabetes_relevant(c)]

    # Fallback — keep top 2 if everything was filtered out
    if len(relevant) < 2:
        relevant = filtered[:2] if len(filtered) >= 2 else chunks[:2]

    return relevant


# ── Source quality boosting ───────────────────────────────────────────────────

def boost_quality_sources(chunks: list) -> list:
    """
    Re-sort chunks to prioritise high-quality clinical sources
    (guidelines, RCTs) over expert opinion and off-topic papers.
    Uses a combined score of rerank_score + source quality bonus.
    """
    SOURCE_BONUS = {
        "clinical_guideline": 3.0,
        "systematic_review":  2.5,
        "rct":                2.0,
        "cohort_study":       1.0,
        "case_report":        0.0,
        "expert_opinion":    -1.0,
        "unknown":            0.0,
    }

    def combined_score(chunk):
        rerank  = chunk.get("rerank_score", 0.0)
        bonus   = SOURCE_BONUS.get(chunk.get("source_type", "unknown"), 0.0)
        return rerank + bonus

    return sorted(chunks, key=combined_score, reverse=True)


# ── BM25 cache ────────────────────────────────────────────────────────────────

_bm25_cache: dict = {"index": None, "chunks": None}


# ── Main hybrid retrieval function ───────────────────────────────────────────

def hybrid_retrieve(
    query:      str,
    collection: chromadb.Collection,
    all_chunks: list,
    top_k:      int = TOP_K_RERANK,
) -> list:
    """
    Full hybrid retrieval pipeline:
      dense → sparse → RRF → rerank → relevance filter → quality boost → top_k

    Args:
        query:      user clinical question
        collection: ChromaDB collection
        all_chunks: all chunks (for BM25)
        top_k:      number of final chunks to return

    Returns:
        List of top_k most relevant, diabetes-specific chunks.
    """
    global _bm25_cache

    # Build BM25 index (cached per session)
    if _bm25_cache["chunks"] is not all_chunks or _bm25_cache["index"] is None:
        print("Building BM25 index…")
        _bm25_cache["index"]  = build_bm25_index(all_chunks)
        _bm25_cache["chunks"] = all_chunks

    # Step 1: Dense retrieval (top-20)
    dense_docs = dense_retrieve(query, collection, n=TOP_K_RETRIEVE)

    # Step 2: Sparse BM25 retrieval (top-20)
    bm25_docs  = bm25_retrieve(query, _bm25_cache["index"],
                               all_chunks, n=TOP_K_RETRIEVE)

    # Step 3: Fuse rankings via RRF
    fused = reciprocal_rank_fusion([dense_docs, bm25_docs])

    # Step 4: Rerank top-20 with cross-encoder
    top_fused = fused[:20]
    reranked  = rerank(query, top_fused, top_k=top_k + 3)  # get extra for filtering

    # Step 5: Filter irrelevant chunks
    relevant = filter_relevant(reranked)

    # Step 6: Boost quality sources to top
    boosted = boost_quality_sources(relevant)

    return boosted[:top_k]


# ── Entry point (quick test) ──────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle
    from ingestion.embedder import get_chroma_collection

    with open("data/processed/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    collection = get_chroma_collection()

    queries = [
        "What is the recommended HbA1c target for type 2 diabetes?",
        "What are the contraindications of metformin?",
        "What is the role of SGLT2 inhibitors in heart failure?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = hybrid_retrieve(query, collection, chunks)
        for i, r in enumerate(results):
            print(f"  [{i+1}] {r.get('journal','?')} ({r.get('year','?')}) "
                  f"| {r.get('source_type','?')} "
                  f"| rerank={r.get('rerank_score',0):.2f}")
            print(f"       {r['text'][:100]}…")