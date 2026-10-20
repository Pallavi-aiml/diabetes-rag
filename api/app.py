"""
api/app.py — Diabetes RAG API

Notes:
- Risk prediction failures are returned as API errors instead of silently falling back
  to a RAG-only answer when patient parameters were supplied.
- CORS origins are configurable through ALLOWED_ORIGINS.
- Health checks report degraded readiness when the knowledge base is unavailable.
- Feedback is appended to JSON Lines so prior feedback is not overwritten.
"""

import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import COLLECTION_NAME
from ingestion.embedder import get_chroma_collection
from ingestion.chunker import load_chunks
from retrieval.retriever import hybrid_retrieve
from evidence.verifier import verify_chunks, aggregate_confidence
from generation.generator import generate_answer, get_llm_name


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


app = FastAPI(
    title="Evidence-Verified RAG — Diabetes Care",
    description="Evidence-grounded diabetes information with optional educational risk estimation.",
    version="3.1.0",
)

# Set ALLOWED_ORIGINS to a comma-separated list in deployment.
# Defaults support common local development frontends only.
_default_origins = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8501,http://127.0.0.1:8501"
_allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

_collection = None
_all_chunks = []
_feedback_log_path = PROJECT_ROOT / "data" / "feedback_log.jsonl"


class PatientParams(BaseModel):
    Pregnancies: Optional[float] = Field(None, ge=0, le=30)
    Glucose: Optional[float] = Field(None, ge=0, le=600)
    BloodPressure: Optional[float] = Field(None, ge=0, le=350)
    SkinThickness: Optional[float] = Field(None, ge=0, le=150)
    Insulin: Optional[float] = Field(None, ge=0, le=3000)
    BMI: Optional[float] = Field(None, ge=0, le=150)
    DiabetesPedigreeFunction: Optional[float] = Field(None, ge=0, le=20)
    Age: Optional[float] = Field(None, ge=1, le=120)

    @validator("*", allow_reuse=True)
    def values_must_be_finite(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("Patient values must be finite numbers.")
        return value

    class Config:
        extra = "forbid"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    patient_params: Optional[PatientParams] = None
    use_local_llm: bool = False
    top_k: int = Field(5, ge=1, le=10)

    @validator("question")
    def question_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("Question must not be blank.")
        return value.strip()


class RiskResult(BaseModel):
    risk_score: float
    risk_category: str
    risk_level: str
    risk_color: str
    prediction: int


class SHAPFeature(BaseModel):
    feature: str
    display: str
    shap_value: float
    direction: str
    actual: object


class SHAPResult(BaseModel):
    top_features: list[SHAPFeature]
    explanation: str
    base_value: float


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
    patient_specific_query: Optional[str]
    risk_result: Optional[RiskResult]
    shap_result: Optional[SHAPResult]
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
    question: str = Field(..., min_length=1, max_length=1000)
    answer: str = Field(..., min_length=1, max_length=10000)
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = Field(None, max_length=2000)

    @validator("question", "answer")
    def feedback_text_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("This field must not be blank.")
        return value.strip()


@app.on_event("startup")
async def startup_event():
    """Load the knowledge base; keep startup alive so health can report degraded state."""
    global _collection, _all_chunks

    try:
        _collection = get_chroma_collection()
        print(f"ChromaDB loaded: {_collection.count()} chunks")
    except Exception as exc:
        _collection = None
        print(f"ChromaDB not ready: {type(exc).__name__}: {exc}")

    try:
        _all_chunks = load_chunks()
        print(f"Chunks loaded: {len(_all_chunks)}")
    except Exception as exc:
        _all_chunks = []
        print(f"Chunks not ready: {type(exc).__name__}: {exc}")

    try:
        mlflow.set_experiment("diabetes-rag-v3")
    except Exception as exc:
        # Tracking is optional; a tracking outage should not prevent API startup.
        print(f"MLflow setup unavailable: {type(exc).__name__}: {exc}")

    print("API startup completed.")


@app.get("/health")
async def health():
    ready = _collection is not None and bool(_all_chunks)
    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "collection_loaded": _collection is not None,
        "chunks_loaded": len(_all_chunks),
        "timestamp": utc_now_iso(),
    }


@app.get("/stats")
async def stats():
    if _collection is None or not _all_chunks:
        raise HTTPException(status_code=503, detail="Knowledge base not ready.")

    try:
        total_chunks = _collection.count()
    except Exception as exc:
        print(f"Could not count collection: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail="Knowledge base is unavailable.") from exc

    return {
        "total_chunks": total_chunks,
        "collection_name": COLLECTION_NAME,
        "status": "ready",
        "llm": get_llm_name(),
    }


def build_final_answer(evidence_answer: str, risk_result=None, shap_result=None) -> str:
    """Combine outputs while clearly separating model estimates from medical evidence.

    Evidence verification metrics are calculated against evidence_answer before the
    ML section is appended. This function does not validate the model clinically.
    """
    if not risk_result:
        return evidence_answer

    score = risk_result.get("risk_score", "Not available")
    category = risk_result.get("risk_category", "Not available")

    lines = [
        "## 1. ML-generated risk assessment",
        "",
        f"- **Model category:** {category}",
        f"- **Model score:** {score}",
        "- **Interpretation:** This is an output from the project's prediction model, "
        "not a diagnosis or a clinically validated probability.",
        "- **Limitation:** The clinical meaning of this score depends on appropriate "
        "validation, probability calibration, and documented decision thresholds.",
    ]

    features = (shap_result or {}).get("top_features", [])
    if features:
        lines.extend([
            "",
            "### Model feature explanations (SHAP)",
            "",
            "These features describe how the model output was influenced for this input. "
            "They do not establish causation or mean a factor is medically protective.",
        ])
        for item in features:
            display = item.get("display") or item.get("feature") or "Unknown feature"
            direction = item.get("direction", "unknown")
            shap_value = item.get("shap_value", "not available")
            actual = item.get("actual", "not available")
            lines.append(
                f"- **{display}:** supplied value = {actual}; SHAP contribution = "
                f"{shap_value}; reported direction = {direction}."
            )

        explanation = (shap_result or {}).get("explanation")
        if explanation:
            lines.extend(["", f"**Model explanation:** {explanation}"])

    lines.extend([
        "",
        "## 2. Evidence-based medical guidance",
        "",
        evidence_answer,
        "",
        "The guidance in this section is generated from retrieved sources. The cited "
        "sources do not independently validate the ML score or category unless they "
        "explicitly evaluate this exact model and its validation results.",
        "",
        "## 3. Verification scope and limitations",
        "",
        "The evidence-verification metrics in the API response apply to the "
        "evidence-grounded answer before the ML assessment was appended. They do not "
        "establish that the ML prediction is clinically valid.",
    ])
    return "\n".join(lines)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, background_tasks: BackgroundTasks):
    if _collection is None or not _all_chunks:
        raise HTTPException(status_code=503, detail="Knowledge base not ready.")

    # Modules 101–104: optional patient risk prediction and explanation.
    risk_result_raw = None
    shap_result_raw = None
    enriched_query = req.question

    if req.patient_params is not None:
        patient_dict = {
            key: value
            for key, value in req.patient_params.dict().items()
            if value is not None
        }

        if not patient_dict:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Patient parameters were supplied, but all values are empty. "
                    "Provide at least one patient parameter."
                ),
            )

        try:
            from risk_prediction.predictor import run_patient_pipeline

            pipeline_result = run_patient_pipeline(
                user_query=req.question,
                patient_params=patient_dict,
            )

            enriched_query = pipeline_result["patient_specific_query"]
            risk_result_raw = pipeline_result["risk_result"]
            shap_result_raw = pipeline_result["shap_result"]

            if not risk_result_raw:
                raise RuntimeError("Risk pipeline returned no risk result.")

        except ValueError as exc:
            print(f"Invalid patient parameters: {exc}")
            raise HTTPException(
                status_code=422,
                detail="Patient parameters are invalid. Check the supplied values.",
            ) from exc
        except Exception as exc:
            print(f"Risk pipeline failed: {type(exc).__name__}: {exc}")
            raise HTTPException(
                status_code=503,
                detail="Risk prediction is temporarily unavailable. Please try again later.",
            ) from exc

    # Modules 105–107: hybrid retrieval.
    try:
        retrieved = hybrid_retrieve(
            query=enriched_query,
            collection=_collection,
            all_chunks=_all_chunks,
            top_k=req.top_k,
        )
    except Exception as exc:
        # Full traceback appears in the Uvicorn terminal for local debugging.
        logger.exception(
            "Document retrieval failed (error_type=%s, error=%s)",
            type(exc).__name__,
            str(exc),
        )
        # Keep implementation details in server logs, not public API responses.
        raise HTTPException(
            status_code=503,
            detail="Document retrieval is temporarily unavailable. Please try again.",
        ) from exc

    if not retrieved:
        # A valid question with no supporting evidence is a normal abstention,
        # not an API failure. Keep the response schema stable for clients/evaluation.
        abstention = (
            "The available knowledge-base evidence does not adequately answer this "
            "question. I will not guess or provide an unsupported medical claim. "
            "Please consult a qualified healthcare professional or an authoritative, "
            "current clinical guideline."
        )
        return QueryResponse(
            question=req.question,
            patient_specific_query=(enriched_query if enriched_query != req.question else None),
            risk_result=(RiskResult(
                risk_score=risk_result_raw["risk_score"],
                risk_category=risk_result_raw["risk_category"],
                risk_level=risk_result_raw["risk_level"],
                risk_color=risk_result_raw.get("risk_color", "#6c757d"),
                prediction=risk_result_raw["prediction"],
            ) if risk_result_raw else None),
            shap_result=(SHAPResult(
                top_features=[SHAPFeature(**item) for item in shap_result_raw.get("top_features", [])],
                explanation=shap_result_raw.get("explanation", ""),
                base_value=shap_result_raw.get("base_value", 0.0),
            ) if shap_result_raw else None),
            answer=abstention + "\n\n" + (
                "For educational information only; not a diagnosis or a substitute "
                "for assessment by a qualified healthcare professional."
            ),
            confidence=0.0,
            evidence_grade="D",
            faithfulness=0.0,
            avg_credibility=0.0,
            hallucination_risk=False,
            contradictions_found=False,
            citations=[],
            disclaimer=(
                "For educational information only; not a diagnosis or a substitute "
                "for assessment by a qualified healthcare professional."
            ),
            llm_used="not used (insufficient retrieved evidence)",
            timestamp=utc_now_iso(),
        )

    # Modules 108–109: evidence verification.
    try:
        verified = verify_chunks(retrieved)
    except Exception as exc:
        print(f"Evidence verification failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Evidence verification is temporarily unavailable.",
        ) from exc

    # Module 110: generate the evidence-grounded answer first.
    try:
        evidence_answer = generate_answer(
            enriched_query,
            verified,
            use_local=req.use_local_llm,
        )
    except Exception as exc:
        logger.exception("Answer generation failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Answer generation is temporarily unavailable. Please try again.",
        ) from exc

    # Modules 111–113: evaluate the evidence-grounded answer BEFORE appending ML output.
    # This prevents the model-generated score/SHAP section from inflating evidence metrics.
    try:
        conf_report = aggregate_confidence(verified, evidence_answer)
    except Exception as exc:
        logger.exception("Confidence aggregation failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail="Could not evaluate the generated answer.",
        ) from exc

    # Clearly separate model output from evidence-based medical guidance.
    answer = build_final_answer(
        evidence_answer=evidence_answer,
        risk_result=risk_result_raw,
        shap_result=shap_result_raw,
    )

    # Module 115: structured citations.
    citations = []
    for index, chunk in enumerate(verified):
        text_value = str(chunk.get("text", ""))
        try:
            year = int(chunk.get("year", 0) or 0)
        except (TypeError, ValueError):
            year = 0

        try:
            credibility = float(chunk.get("credibility_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            credibility = 0.0

        citations.append(
            CitationItem(
                source_num=index + 1,
                source=str(chunk.get("source", "")),
                journal=str(chunk.get("journal", "Unknown")),
                year=year,
                evidence_grade=str(chunk.get("evidence_grade", "D")),
                credibility_score=credibility,
                source_type=str(chunk.get("source_type", "unknown")),
                chunk_preview=text_value[:200] + ("..." if len(text_value) > 200 else ""),
            )
        )

    risk_resp = None
    if risk_result_raw:
        risk_resp = RiskResult(
            risk_score=risk_result_raw["risk_score"],
            risk_category=risk_result_raw["risk_category"],
            risk_level=risk_result_raw["risk_level"],
            risk_color=risk_result_raw.get("risk_color", "#6c757d"),
            prediction=risk_result_raw["prediction"],
        )

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
                for item in shap_result_raw.get("top_features", [])
            ],
            explanation=shap_result_raw.get("explanation", ""),
            base_value=shap_result_raw.get("base_value", 0.0),
        )

    background_tasks.add_task(
        _log_mlflow,
        confidence=conf_report["confidence"],
        grade=conf_report["grade"],
        faithfulness=conf_report["faithfulness"],
        llm=get_llm_name(),
        has_patient_params=risk_result_raw is not None,
    )

    return QueryResponse(
        question=req.question,
        patient_specific_query=(
            enriched_query if enriched_query != req.question else None
        ),
        risk_result=risk_resp,
        shap_result=shap_resp,
        answer=answer,
        confidence=conf_report["confidence"],
        evidence_grade=conf_report["grade"],
        faithfulness=conf_report["faithfulness"],
        avg_credibility=conf_report["avg_credibility"],
        hallucination_risk=conf_report["hallucination_risk"],
        contradictions_found=bool(conf_report.get("contradictions")),
        citations=citations,
        disclaimer=(
            "For educational information only; not a diagnosis or a substitute "
            "for assessment by a qualified healthcare professional."
        ),
        llm_used=get_llm_name(),
        timestamp=utc_now_iso(),
    )


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """Append feedback without replacing feedback saved by earlier processes."""
    entry = {
        "question": req.question,
        "answer_preview": req.answer[:200],
        "rating": req.rating,
        "comment": req.comment,
        "timestamp": utc_now_iso(),
    }

    try:
        _feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
        with _feedback_log_path.open("a", encoding="utf-8") as feedback_file:
            feedback_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"Feedback write failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Feedback could not be saved.",
        ) from exc

    return {"message": "Feedback saved.", "storage": "jsonl"}


def _log_mlflow(confidence, grade, faithfulness, llm, has_patient_params):
    """Best-effort experiment logging; never log raw questions or patient values."""
    try:
        with mlflow.start_run(run_name="query"):
            mlflow.log_param("llm", str(llm))
            mlflow.log_param("has_patient_params", bool(has_patient_params))
            mlflow.log_param("evidence_grade", str(grade))
            mlflow.log_metric("confidence", float(confidence))
            mlflow.log_metric("faithfulness", float(faithfulness))
    except Exception as exc:
        print(f"MLflow logging failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)
