"""
api/app.py
FastAPI backend for the Evidence-Verified RAG Diabetes Care System.

Endpoints:
  POST /query          — main RAG query endpoint
  POST /feedback       — user feedback collection
  GET  /health         — health check
  GET  /stats          — knowledge base statistics
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import mlflow
import chromadb

sys.path.append(str(Path(__file__).parent.parent))
#from config import CHROMA_DB_PATH, COLLECTION_NAME
from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    LOCAL_LLM_MODEL,
    OPENAI_MODEL,
)
from generation.generator import generate_answer
from ingestion.embedder import get_chroma_collection
from ingestion.chunker import load_chunks
from retrieval.retriever import hybrid_retrieve
from evidence.verifier import verify_chunks, aggregate_confidence
#from generation.generator import generate_answer

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Evidence-Verified RAG — Diabetes Care",
    description=(
        "AI-powered clinical QA system with evidence verification. "
        "Every answer is backed by credibility-scored, GRADE-classified sources."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (loaded once at startup)
_collection: Optional[chromadb.Collection] = None
_all_chunks: list[dict] = []
_feedback_log: list[dict] = []


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global _collection, _all_chunks
    print("Loading ChromaDB collection…")
    try:
        _collection = get_chroma_collection()
        print(f"  Collection loaded: {_collection.count()} chunks")
    except Exception as e:
        print(f"  Warning: ChromaDB not ready — {e}. Run ingestion first.")

    print("Loading chunks for BM25…")
    try:
        _all_chunks = load_chunks()
        print(f"  Loaded {len(_all_chunks)} chunks")
    except FileNotFoundError:
        print("  Warning: chunks.json not found. Run ingestion/chunker.py first.")

    # MLflow experiment
    mlflow.set_experiment("diabetes-rag")
    print("API ready.")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000,
                          example="What is the recommended HbA1c target for type 2 diabetes?")
    use_local_llm: bool = Field(True, description="True=Ollama(free), False=GPT-4o(paid)")
    top_k: int = Field(5, ge=1, le=10, description="Number of evidence chunks to retrieve")
    patient_context: Optional[str] = Field(None, max_length=500,
                                           description="Optional patient context (age, comorbidities)")


class CitationItem(BaseModel):
    source_num: int
    source: str
    journal: str
    year: int
    evidence_grade: str
    credibility_score: float
    source_type: str
    chunk_preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    evidence_grade: str
    faithfulness: float
    avg_credibility: float
    hallucination_risk: bool
    contradictions_found: bool
    citations: list[CitationItem]
    disclaimer: str
    llm_used: str
    timestamp: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., ge=1, le=5, description="1=poor, 5=excellent")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    message: str
    total_feedback: int


class StatsResponse(BaseModel):
    total_chunks: int
    collection_name: str
    status: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "collection_loaded": _collection is not None,
        "chunks_loaded": len(_all_chunks),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    if _collection is None:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded.")
    return StatsResponse(
        total_chunks=_collection.count(),
        collection_name=COLLECTION_NAME,
        status="ready",
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, background_tasks: BackgroundTasks):
    if _collection is None or not _all_chunks:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not ready. Run ingestion pipeline first.",
        )

    # ---- Step 1: Hybrid retrieval ----
    retrieved = hybrid_retrieve(
        query=req.question,
        collection=_collection,
        all_chunks=_all_chunks,
        top_k=req.top_k,
    )

    if not retrieved:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    # ---- Step 2: Evidence verification ----
    verified = verify_chunks(retrieved)

    # ---- Step 3: LLM generation ----
    # Prepend patient context if provided
    full_query = req.question
    if req.patient_context:
        full_query = f"Patient context: {req.patient_context}\n\nQuestion: {req.question}"

    answer = generate_answer(full_query, verified, use_local=req.use_local_llm)

    # ---- Step 4: Confidence aggregation ----
    conf_report = aggregate_confidence(verified, answer)

    # ---- Step 5: Build citations ----
    citations = [
        CitationItem(
            source_num=i + 1,
            source=c.get("source", ""),
            journal=c.get("journal", "Unknown"),
            year=int(c.get("year", 0)),
            evidence_grade=c.get("evidence_grade", "D"),
            credibility_score=c.get("credibility_score", 0.0),
            source_type=c.get("source_type", "unknown"),
            chunk_preview=c["text"][:200] + "…",
        )
        for i, c in enumerate(verified)
    ]

    # ---- Step 6: Log to MLflow (background) ----
    background_tasks.add_task(
        log_to_mlflow,
        question=req.question,
        confidence=conf_report["confidence"],
        grade=conf_report["grade"],
        faithfulness=conf_report["faithfulness"],
        llm=LOCAL_LLM_MODEL if req.use_local_llm else OPENAI_MODEL,
        top_k=req.top_k,
    )

   
    return QueryResponse(
        question=req.question,
        answer=answer,
        confidence=conf_report["confidence"],
        evidence_grade=conf_report["grade"],
        faithfulness=conf_report["faithfulness"],
        avg_credibility=conf_report["avg_credibility"],
        hallucination_risk=conf_report["hallucination_risk"],
        contradictions_found=bool(conf_report["contradictions"]),
        citations=citations,
        disclaimer="This system provides information only. Always consult a qualified clinician.",
       # llm_used=LOCAL_LLM_MODEL if req.use_local_llm else OPENAI_MODEL,
        llm_used="Groq — qwen/qwen3.8-27b",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest):
    entry = {
        "question": req.question,
        "answer_preview": req.answer[:200],
        "rating": req.rating,
        "comment": req.comment,
        "timestamp": datetime.utcnow().isoformat(),
    }
    _feedback_log.append(entry)

    # Persist to disk
    log_path = Path("./data/feedback_log.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(_feedback_log, f, indent=2)

    return FeedbackResponse(
        message="Thank you for your feedback.",
        total_feedback=len(_feedback_log),
    )


# ---------------------------------------------------------------------------
# MLflow logging (background task)
# ---------------------------------------------------------------------------

def log_to_mlflow(question, confidence, grade, faithfulness, llm, top_k):
    try:
        with mlflow.start_run(run_name="query"):
            mlflow.log_param("llm", llm)
            mlflow.log_param("top_k", top_k)
            mlflow.log_metric("confidence", confidence)
            mlflow.log_metric("faithfulness", faithfulness)
            mlflow.log_param("evidence_grade", grade)
    except Exception:
        pass  # MLflow logging is non-critical


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
