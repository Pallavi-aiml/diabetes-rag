"""
retrieval/retriever.py
Hybrid diabetes retrieval with relevance-first ranking.

Pipeline:
  dense retrieval + BM25 -> reciprocal-rank fusion -> cross-encoder reranking
  -> query-specific relevance checks -> source-quality tie-breaker.

The relevance checks are heuristics. They must not be treated as clinical validation.
"""

import re
import sys
from pathlib import Path
from collections import Counter

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import chromadb

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    TOP_K_RETRIEVE, TOP_K_RERANK, RERANKER_MODEL,
    MIN_QUERY_RELEVANCE, MIN_RERANK_SCORE,
)
from ingestion.embedder import embed_query

_reranker = None


def get_reranker() -> CrossEncoder:
    """Load the cross-encoder once, on first rerank call."""
    global _reranker
    if _reranker is None:
        print(f"Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker

DIABETES_KEYWORDS = [
    "diabetes", "diabetic", "diabetics", "glucose", "glycaemic", "glycemic",
    "insulin", "metformin", "hba1c", "a1c", "hyperglycaemia", "hyperglycemia",
    "hypoglycaemia", "hypoglycemia", "type 2", "type 1", "t2dm", "t1dm",
    "pancreas", "pancreatic", "beta cell", "sglt2", "glp-1", "dpp-4",
    "gliptin", "sulfonylurea", "thiazolidinedione", "retinopathy",
    "nephropathy", "neuropathy", "ketoacidosis", "dka", "blood sugar",
    "fasting glucose", "obesity", "bmi", "weight loss", "cardiovascular",
    "heart failure", "kidney", "renal", "egfr", "clinical guideline",
    "standards of care", "icmr", "ada guideline", "who diabetes",
]

OFF_TOPIC_JOURNALS = [
    "journal of infectious diseases", "canadian journal of infectious",
    "microbiology", "bactericida", "antimicrobial", "virology", "veterinary",
    "dentistry", "dermatology",
]

# These phrases identify common intents; they are not a medical ontology.
INTENT_TERMS = {
    "risk": {
        "risk", "risk factors", "predict", "prediction", "prevent", "prevention",
        "progression", "develop", "developing", "incidence", "screening",
        "prediabetes", "diagnostic criteria",
    },
    "diagnosis": {
        "diagnosis", "diagnostic", "criteria", "fasting", "fpg", "a1c",
        "hba1c", "oral glucose tolerance", "threshold", "cutoff",
    },
    "treatment": {
        "treatment", "treat", "therapy", "medication", "drug", "metformin",
        "insulin", "dose", "contraindication", "adverse effect", "side effect",
    },
    "complications": {
        "complication", "complications", "retinopathy", "nephropathy",
        "neuropathy", "cardiovascular", "kidney", "renal", "foot ulcer",
        "fracture",
    },
    "lifestyle": {
        "diet", "dietary", "exercise", "physical activity", "weight", "obesity",
        "smoking", "nutrition", "lifestyle",
    },
}

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]*")


def _tokens(text: str) -> list:
    return [t.lower() for t in _WORD_RE.findall(str(text or ""))]


def _query_intent(query: str) -> set:
    q = str(query or "").lower()
    found = set()
    for intent, terms in INTENT_TERMS.items():
        if any(term in q for term in terms):
            found.add(intent)
    return found


def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    """Fuse ranked document lists by reciprocal rank."""
    scores = {}
    id_to_doc = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            cid = doc.get("chunk_id") or doc.get("id")
            if cid is None:
                continue
            id_to_doc[cid] = doc
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return [id_to_doc[cid] for cid in sorted(scores, key=lambda x: scores[x], reverse=True)]



def dense_retrieve(
    query: str,
    collection: chromadb.Collection,
    n: int = TOP_K_RETRIEVE,
) -> list:
    """Return top-n chunks by cosine similarity from ChromaDB."""
    import numpy as np

    # Convert the query embedding to a NumPy array.
    query_vec = np.asarray(embed_query(query), dtype=np.float32)

    # ChromaDB expects shape (number_of_queries, embedding_dimension).
    # A single query should therefore be a 2D array.
    if query_vec.ndim == 3 and query_vec.shape[0] == 1:
        query_vec = query_vec.squeeze(axis=0)

    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    if query_vec.ndim != 2 or query_vec.shape[0] != 1:
        raise ValueError(
            "Expected one query embedding with shape (1, dimension); "
            f"received shape {query_vec.shape}."
        )

    if not np.isfinite(query_vec).all():
        raise ValueError("Query embedding contains NaN or infinite values.")

    results = collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=min(n, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, document in enumerate(documents):
        meta = metadatas[i] or {}
        docs.append({
            "chunk_id": ids[i],
            "text": document,
            "distance": distances[i],
            **meta,
        })

    return docs



def build_bm25_index(chunks: list) -> BM25Okapi:
    """Build a BM25 index from chunks, safely handling empty text."""
    tokenized = [_tokens(c.get("text", "")) for c in chunks]
    if not tokenized:
        tokenized = [["__empty__"]]
    return BM25Okapi(tokenized)


def bm25_retrieve(query: str, bm25_index: BM25Okapi, all_chunks: list, n: int = TOP_K_RETRIEVE) -> list:
    """Return sparse-retrieved chunks, excluding all-zero score results when possible."""
    if not all_chunks or n <= 0:
        return []
    scores = bm25_index.get_scores(_tokens(query))
    if not len(scores):
        return []
    top_indices = sorted(range(min(len(scores), len(all_chunks))), key=lambda i: scores[i], reverse=True)[:n]
    positive = [i for i in top_indices if scores[i] > 0]
    chosen = positive if positive else top_indices
    return [all_chunks[i] for i in chosen]


def rerank(query: str, docs: list, top_k: int = TOP_K_RERANK) -> list:
    """Cross-encoder reranking."""
    if not docs or top_k <= 0:
        return []
    pairs = [(query, str(doc.get("text", ""))[:1800]) for doc in docs]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda item: float(item[0]), reverse=True)
    result = []
    for score, doc in ranked[:top_k]:
        copied = dict(doc)
        copied["rerank_score"] = float(score)
        result.append(copied)
    return result


def is_diabetes_relevant(chunk: dict) -> bool:
    """Broad topical filter; query-specific relevance is handled separately."""
    text = str(chunk.get("text", "")).lower()
    journal = str(chunk.get("journal", "")).lower()
    if any(off in journal for off in OFF_TOPIC_JOURNALS):
        return False
    return any(keyword in text for keyword in DIABETES_KEYWORDS)


def _query_specific_score(query: str, chunk: dict) -> float:
    """Blend cross-encoder score with lexical coverage and intent overlap."""
    q_tokens = set(_tokens(query))
    d_tokens = set(_tokens(chunk.get("text", "")))
    if not q_tokens or not d_tokens:
        lexical = 0.0
    else:
        # Ignore common query words; score content-term coverage.
        stop = {"what", "are", "is", "the", "a", "an", "of", "for", "to", "in", "my", "me", "based", "on", "explain", "common"}
        content = {t for t in q_tokens if t not in stop and len(t) > 2}
        lexical = len(content & d_tokens) / max(1, len(content))

    query_intents = _query_intent(query)
    text = str(chunk.get("text", "")).lower()
    intent_hits = sum(1 for intent in query_intents if any(term in text for term in INTENT_TERMS[intent]))
    intent_score = intent_hits / max(1, len(query_intents)) if query_intents else 0.0

    # Cross-encoder logits are not probabilities; use only as a ranking signal.
    rerank_score = float(chunk.get("rerank_score", 0.0))
    # Smoothly bound rerank contribution so fixed quality bonuses cannot dominate.
    rerank_bounded = max(-1.0, min(1.0, rerank_score / 5.0))
    score = 0.60 * ((lexical + intent_score) / 2.0) + 0.40 * ((rerank_bounded + 1.0) / 2.0)
    return round(score, 5)


def filter_relevant(
    chunks: list,
    query: str = "",
    min_rerank_score: float = MIN_RERANK_SCORE,
    min_query_relevance: float = MIN_QUERY_RELEVANCE,
) -> list:
    """
    Filter low-ranking and clearly off-topic chunks. Does not reintroduce
    off-topic chunks through a fallback. If none pass, returns [].
    """
    relevant = []
    for chunk in chunks:
        try:
            rerank_score = float(chunk.get("rerank_score", 0.0))
        except (TypeError, ValueError):
            rerank_score = -999.0
        if rerank_score < min_rerank_score:
            continue
        if not is_diabetes_relevant(chunk):
            continue
        copied = dict(chunk)
        copied["query_relevance_score"] = _query_specific_score(query, copied) if query else 0.0
        if query and copied["query_relevance_score"] < min_query_relevance:
            continue
        relevant.append(copied)
    return relevant


def boost_quality_sources(chunks: list) -> list:
    """
    Source quality is a small tie-breaker after query relevance, not a way for
    an irrelevant guideline to outrank a directly relevant source.
    """
    source_bonus = {
        "clinical_guideline": 0.08,
        "systematic_review": 0.07,
        "rct": 0.06,
        "cohort_study": 0.03,
        "case_report": 0.0,
        "expert_opinion": -0.03,
        "unknown": 0.0,
    }

    def final_score(chunk):
        relevance = float(chunk.get("query_relevance_score", 0.0))
        bonus = source_bonus.get(str(chunk.get("source_type", "unknown")).lower(), 0.0)
        rerank = float(chunk.get("rerank_score", 0.0))
        return relevance + bonus + 0.02 * max(-1.0, min(1.0, rerank / 5.0))

    return sorted(chunks, key=final_score, reverse=True)


_bm25_cache = {"index": None, "fingerprint": None}


def _corpus_fingerprint(chunks: list) -> tuple:
    """Identify a chunk list without relying on object identity alone."""
    if not chunks:
        return (0, "")
    ids = tuple(str(chunk.get("chunk_id", i)) for i, chunk in enumerate(chunks))
    return (len(ids), hash(ids))


def hybrid_retrieve(
    query: str,
    collection: chromadb.Collection,
    all_chunks: list,
    top_k: int = TOP_K_RERANK,
) -> list:
    """Run hybrid retrieval and return only relevant, ranked chunks."""
    global _bm25_cache
    if not str(query or "").strip() or top_k <= 0:
        return []
    if not all_chunks:
        # Dense retrieval can still function if the caller has no local BM25 corpus.
        dense_docs = dense_retrieve(query, collection, n=TOP_K_RETRIEVE)
        return boost_quality_sources(filter_relevant(rerank(query, dense_docs, top_k=top_k), query))[:top_k]

    fingerprint = _corpus_fingerprint(all_chunks)
    if _bm25_cache["fingerprint"] != fingerprint or _bm25_cache["index"] is None:
        print("Building BM25 index…")
        _bm25_cache["index"] = build_bm25_index(all_chunks)
        _bm25_cache["fingerprint"] = fingerprint

    dense_docs = dense_retrieve(query, collection, n=TOP_K_RETRIEVE)
    bm25_docs = bm25_retrieve(query, _bm25_cache["index"], all_chunks, n=TOP_K_RETRIEVE)
    fused = reciprocal_rank_fusion([dense_docs, bm25_docs])
    reranked = rerank(
        query,
        fused[:max(20, top_k + 3)],
        top_k=min(len(fused), max(top_k * 3, top_k + 3))
    )
    relevant = filter_relevant(reranked, query=query)

    # Crucially, do not fallback to off-topic chunks when relevance filtering empties results.
    boosted = boost_quality_sources(relevant)
    return boosted[:top_k]


if __name__ == "__main__":
    print("retriever.py loaded. Run hybrid_retrieve() through the API or your project test harness.")
