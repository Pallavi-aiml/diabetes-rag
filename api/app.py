"""
api/app.py — Updated to integrate ALL patent modules (101-115)
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

sys.path.append(str(Path(__file__).parent.parent))
from config import CHROMA_DB_PATH, COLLECTION_NAME
from ingestion.embedder import get_chroma_collection
from ingestion.chunker import load_chunks
from retrieval.retriever import hybrid_retrieve
from evidence.verifier import verify_chunks, aggregate_confidence
from generation.generator import generate_answer, get_llm_name

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Evidence-Verified RAG — Diabetes Care",
    description="Patent-aligned clinical QA system — all 15 modules (101-115).",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_collection   = None
_all_chunks   = []
_feedback_log = []


@app.on_event("startup")
async def startup_event():
    global _collection, _all_chunks
    try:
        _collection = get_chroma_collection()
        print(f"ChromaDB loaded: {_collection.count()} chunks")
    except Exception as e:
        print(f"ChromaDB not ready: {e}")
    try:
        _all_chunks = load_chunks()
        print(f"Chunks loaded: {len(_all_chunks)}")
    except FileNotFoundError:
        print("chunks.json not found — run ingestion pipeline first.")
    mlflow.set_experiment("diabetes-rag-v3")
    print("API ready.")


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class PatientParams(BaseModel):
    Pregnancies:              Optional[float] = Field(None)
    Glucose:                  Optional[float] = Field(None)
    BloodPressure:            Optional[float] = Field(None)
    SkinThickness:            Optional[float] = Field(None)
    Insulin:                  Optional[float] = Field(None)
    BMI:                      Optional[float] = Field(None)
    DiabetesPedigreeFunction: Optional[float] = Field(None)
    Age:                      Optional[float] = Field(None)


class QueryRequest(BaseModel):
    question:       str           = Field(..., min_length=5, max_length=1000)
    patient_params: Optional[PatientParams] = Field(None)
    use_local_llm:  bool          = Field(False)
    top_k:          int           = Field(5, ge=1, le=10)


class RiskResult(BaseModel):
    risk_score:    float
    risk_category: str
    risk_level:    str
    risk_color:    str
    prediction:    int


class SHAPFeature(BaseModel):
    feature:    str
    display:    str
    shap_value: float
    direction:  str
    actual:     object


class SHAPResult(BaseModel):
    top_features: list[SHAPFeature]
    explanation:  str
    base_value:   float


class CitationItem(BaseModel):
    source_num:        int
    source:            str
    journal:           str
    year:              int
    evidence_grade:    str
    credibility_score: float
    source_type:       str
    chunk_preview:     str


class QueryResponse(BaseModel):
    question:               str
    patient_specific_query: Optional[str]
    risk_result:            Optional[RiskResult]
    shap_result:            Optional[SHAPResult]
    answer:                 str
    confidence:             float
    evidence_grade:         str
    faithfulness:           float
    avg_credibility:        float
    hallucination_risk:     bool
    contradictions_found:   bool
    citations:              list[CitationItem]
    disclaimer:             str
    llm_used:               str
    timestamp:              str


class FeedbackRequest(BaseModel):
    question: str
    answer:   str
    rating:   int = Field(..., ge=1, le=5)
    comment:  Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":            "ok",
        "collection_loaded": _collection is not None,
        "chunks_loaded":     len(_all_chunks),
        "timestamp":         datetime.utcnow().isoformat(),
    }


@app.get("/stats")
async def stats():
    if _collection is None:
        raise HTTPException(503, "Knowledge base not loaded.")
    return {
        "total_chunks":    _collection.count(),
        "collection_name": COLLECTION_NAME,
        "status":          "ready",
        "llm":             get_llm_name(),
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, background_tasks: BackgroundTasks):
    if _collection is None or not _all_chunks:
        raise HTTPException(503, "Knowledge base not ready.")

    # ── Modules 101-104: Patient pipeline ─────────────────────────────────────
    risk_result_raw = None
    shap_result_raw = None
    enriched_query  = req.question

    if req.patient_params:
        # Safely convert all values to float
        patient_dict = {}
        for k, v in req.patient_params.dict().items():
            if v is not None:
                try:
                    patient_dict[k] = float(str(v).strip())
                except (ValueError, TypeError):
                    pass

        if patient_dict:
            try:
                from risk_prediction.predictor import run_patient_pipeline
                pipeline_result = run_patient_pipeline(
                    user_query=req.question,
                    patient_params=patient_dict,
                )
                enriched_query  = pipeline_result["patient_specific_query"]
                risk_result_raw = pipeline_result["risk_result"]
                shap_result_raw = pipeline_result["shap_result"]
            except Exception as e:
                print(f"Risk pipeline error (non-fatal): {e}")
                enriched_query = req.question

    # ── Modules 105-107: Hybrid retrieval ─────────────────────────────────────
    retrieved = hybrid_retrieve(
        query=enriched_query,
        collection=_collection,
        all_chunks=_all_chunks,
        top_k=req.top_k,
    )
    if not retrieved:
        raise HTTPException(404, "No relevant documents found.")

    # ── Modules 108-109: Evidence verification ────────────────────────────────
    verified = verify_chunks(retrieved)

    # ── Module 110: Grounded generation ──────────────────────────────────────
    answer = generate_answer(enriched_query, verified, use_local=req.use_local_llm)

    # ── Modules 111-113: Confidence aggregation ───────────────────────────────
    conf_report = aggregate_confidence(verified, answer)

    # ── Module 115: Build structured response ─────────────────────────────────
    citations = [
        CitationItem(
            source_num=i + 1,
            source=c.get("source", ""),
            journal=c.get("journal", "Unknown"),
            year=int(c.get("year", 0)),
            evidence_grade=c.get("evidence_grade", "D"),
            credibility_score=c.get("credibility_score", 0.0),
            source_type=c.get("source_type", "unknown"),
            chunk_preview=c["text"][:200] + "...",
        )
        for i, c in enumerate(verified)
    ]

    # Serialise risk result
    risk_resp = None
    if risk_result_raw:
        risk_resp = RiskResult(
            risk_score=risk_result_raw["risk_score"],
            risk_category=risk_result_raw["risk_category"],
            risk_level=risk_result_raw["risk_level"],
            risk_color=risk_result_raw.get("risk_color", "#6c757d"),
            prediction=risk_result_raw["prediction"],
        )

    # Serialise SHAP result
    shap_resp = None
    if shap_result_raw:
        shap_resp = SHAPResult(
            top_features=[
                SHAPFeature(
                    feature=item["feature"],
                    display=item["display"],
                    shap_value=item["shap_value"],
                    direction=item["direction"],
                    actual=item["actual"],
                )
                for item in shap_result_raw["top_features"]
            ],
            explanation=shap_result_raw["explanation"],
            base_value=shap_result_raw["base_value"],
        )

    background_tasks.add_task(
        _log_mlflow,
        question=req.question,
        confidence=conf_report["confidence"],
        grade=conf_report["grade"],
        faithfulness=conf_report["faithfulness"],
        llm=get_llm_name(),
        has_patient_params=risk_result_raw is not None,
    )

    return QueryResponse(
        question=req.question,
        patient_specific_query=enriched_query if enriched_query != req.question else None,
        risk_result=risk_resp,
        shap_result=shap_resp,
        answer=answer,
        confidence=conf_report["confidence"],
        evidence_grade=conf_report["grade"],
        faithfulness=conf_report["faithfulness"],
        avg_credibility=conf_report["avg_credibility"],
        hallucination_risk=conf_report["hallucination_risk"],
        contradictions_found=bool(conf_report["contradictions"]),
        citations=citations,
        disclaimer="This system provides information only. Always consult a qualified clinician.",
        llm_used=get_llm_name(),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    entry = {
        "question":       req.question,
        "answer_preview": req.answer[:200],
        "rating":         req.rating,
        "comment":        req.comment,
        "timestamp":      datetime.utcnow().isoformat(),
    }
    _feedback_log.append(entry)
    log_path = Path("./data/feedback_log.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(_feedback_log, f, indent=2)
    return {"message": "Feedback saved.", "total": len(_feedback_log)}


def _log_mlflow(question, confidence, grade, faithfulness, llm, has_patient_params):
    try:
        with mlflow.start_run(run_name="query"):
            mlflow.log_param("llm", llm)
            mlflow.log_param("has_patient_params", has_patient_params)
            mlflow.log_metric("confidence", confidence)
            mlflow.log_metric("faithfulness", faithfulness)
            mlflow.log_param("evidence_grade", grade)
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
