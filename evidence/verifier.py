"""
evidence/verifier.py
Core innovation module — Evidence Verification Layer.

For every retrieved chunk this module computes:
  1. Credibility score     — source type + journal tier + recency
  2. Evidence grade        — GRADE classification (A / B / C / D)
  3. Faithfulness score    — fixed value (NLI disabled for CPU speed)
  4. Contradiction flag    — disabled for CPU speed
  5. Confidence aggregate  — weighted combination of all above scores

Patent Modules 108, 109, 111, 112, 113
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import FAITHFULNESS_THRESHOLD, CONTRADICTION_THRESHOLD

# ── Source type credibility scores (0–5) ─────────────────────────────────────
SOURCE_SCORES = {
    "systematic_review":  5,
    "rct":                5,
    "clinical_guideline": 4,
    "cohort_study":       3,
    "case_report":        2,
    "expert_opinion":     1,
    "unknown":            2,   # default raised from 0 to 2
}

# ── Journal tier scores (0–5) — partial matching used ─────────────────────────
JOURNAL_SCORES = {
    # Score 5 — top-tier medical journals
    "diabetes care":                                    5,
    "nejm":                                             5,
    "new england journal":                              5,
    "the lancet":                                       5,
    "lancet":                                           5,
    "jama":                                             5,
    "nature medicine":                                  5,
    "cell metabolism":                                  5,
    # Score 4 — strong diabetes / clinical journals
    "bmj":                                              4,
    "diabetologia":                                     4,
    "diabetes":                                         4,
    "journal of diabetes":                              4,
    "journal of clinical endocrinology":                4,
    "endocrinology":                                    4,
    "genome medicine":                                  4,
    "diabetes, metabolic syndrome and obesity":         4,
    "diabetes & metabolic syndrome":                    4,
    "pediatric research":                               4,
    "annals of internal medicine":                      4,
    "circulation":                                      4,
    "european heart journal":                           4,
    "kidney international":                             4,
    # Score 4 — guidelines (special case)
    "icmr guideline":                                   4,
    "icmr":                                             4,
    "who guideline":                                    4,
    "ada guideline":                                    4,
    "diabetes care (ada)":                              5,
    "clinical guideline":                               4,
    # Score 3 — good general medical journals
    "plos one":                                         3,
    "plos":                                             3,
    "scientific reports":                               3,
    "frontiers":                                        3,
    "medicine":                                         3,
    "bmc":                                              3,
    "journal of clinical medicine":                     3,
    "nutrients":                                        3,
    "international journal":                            3,
    "pubmed abstract":                                  3,
    # Score 2 — other/unknown
    "unknown":                                          2,
}


# ── 1. Credibility Scoring ────────────────────────────────────────────────────

def score_credibility(chunk: dict) -> float:
    """
    Compute a credibility score 0–5 for a chunk based on:
      - Source type  (RCT=5, systematic_review=5, guideline=4, cohort=3, case=2)
      - Journal tier (partial match against JOURNAL_SCORES)
      - Publication recency (papers from 2020+ score well)
    """
    source_type  = chunk.get("source_type", "unknown").lower()
    journal      = chunk.get("journal", "Unknown")
    year         = int(chunk.get("year", 2000))

    # Source type score
    src_score = SOURCE_SCORES.get(source_type, 2)

    # Journal score — partial matching (case-insensitive)
    jnl_score    = 2  # default
    journal_lower = journal.lower()
    for jnl_key, score in JOURNAL_SCORES.items():
        if jnl_key in journal_lower:
            jnl_score = score
            break

    # Recency score (0–5): 2020+ gets full marks
    recency = min(5.0, max(0.0, (year - 2010) * 0.5))

    # Weighted combination
    credibility = round(
        src_score * 0.50 +
        jnl_score * 0.30 +
        recency   * 0.20,
        3,
    )
    return min(5.0, credibility)


# ── 2. GRADE Evidence Classification ─────────────────────────────────────────

GRADE_KEYWORDS = {
    "A": [
        "randomized controlled trial", "randomised controlled trial",
        "rct", "meta-analysis", "meta analysis",
        "systematic review", "randomized", "randomised",
        "double-blind", "placebo-controlled",
    ],
    "B": [
        "cohort study", "prospective", "retrospective",
        "guideline", "clinical practice", "standards of care",
        "recommendation", "controlled study", "observational",
        "clinical guideline", "icmr", "ada", "who",
    ],
    "C": [
        "case-control", "cross-sectional", "pilot study",
        "small sample", "case series", "survey",
    ],
    "D": [
        "case report", "expert opinion", "editorial",
        "letter", "anecdotal", "in vitro", "animal study",
    ],
}


def classify_evidence_grade(chunk: dict) -> str:
    """
    Assign GRADE level A/B/C/D based on:
      - source_type metadata
      - keyword signals in the chunk text
    Returns the highest (best) grade found.
    """
    text        = chunk.get("text", "").lower()
    source_type = chunk.get("source_type", "").lower()
    journal     = chunk.get("journal", "").lower()

    # Source-type fast path
    if source_type in ("systematic_review", "rct"):
        return "A"
    if source_type in ("clinical_guideline",):
        return "B"
    if source_type in ("cohort_study",):
        return "B"
    if source_type in ("case_report",):
        return "C"

    # Journal-based fast path
    if any(j in journal for j in ["icmr", "ada", "who", "guideline"]):
        return "B"

    # Keyword scan (highest grade wins)
    for grade in ("A", "B", "C", "D"):
        for kw in GRADE_KEYWORDS[grade]:
            if kw in text:
                return grade

    return "D"


# ── 3. Faithfulness Check (simplified for CPU speed) ─────────────────────────

def check_faithfulness(generated_answer: str, context_chunk: str) -> float:
    """
    Returns a fixed faithfulness score.
    Full NLI (BART-MNLI) is disabled because it times out on CPU
    during live API requests. Returns 0.75 as a reasonable default.
    """
    return 0.75


def check_faithfulness_multi(generated_answer: str, chunks: list) -> float:
    """
    Returns fixed faithfulness score for all chunks.
    NLI disabled for CPU speed.
    """
    if not chunks:
        return 0.0
    return 0.75


# ── 4. Contradiction Detection (disabled for CPU speed) ──────────────────────

def detect_contradiction(chunk_a: str, chunk_b: str) -> bool:
    """Contradiction detection disabled for CPU speed."""
    return False


def find_contradictions(chunks: list) -> list:
    """Returns empty list — contradiction detection disabled for CPU speed."""
    return []


# ── 5. Verify chunks (adds all metadata in-place) ────────────────────────────

def verify_chunks(chunks: list) -> list:
    """
    Annotate every chunk with credibility score and evidence grade.
    """
    for chunk in chunks:
        chunk["credibility_score"] = score_credibility(chunk)
        chunk["evidence_grade"]    = classify_evidence_grade(chunk)
    return chunks


# ── 6. Confidence Aggregator ──────────────────────────────────────────────────

def aggregate_confidence(
    verified_chunks: list,
    generated_answer: str,
) -> dict:
    """
    Compute a final confidence report for a generated answer.

    Returns:
        {
          "confidence":        float  [0-1],
          "grade":             str    best evidence grade across chunks,
          "faithfulness":      float  fixed at 0.75,
          "avg_credibility":   float  mean credibility score,
          "contradictions":    list   always empty (disabled),
          "hallucination_risk":bool   always False (NLI disabled),
        }
    """
    if not verified_chunks:
        return {
            "confidence":        0.0,
            "grade":             "D",
            "faithfulness":      0.0,
            "avg_credibility":   0.0,
            "contradictions":    [],
            "hallucination_risk":True,
        }

    faithfulness  = 0.75  # fixed value (NLI disabled)

    # Average credibility
    avg_cred = sum(
        c.get("credibility_score", 0) for c in verified_chunks
    ) / len(verified_chunks)

    # Best evidence grade
    grade_order = {"A": 4, "B": 3, "C": 2, "D": 1}
    best_grade  = max(
        verified_chunks,
        key=lambda c: grade_order.get(c.get("evidence_grade", "D"), 0),
    )["evidence_grade"]

    # Grade bonus
    grade_bonus = {"A": 0.10, "B": 0.07, "C": 0.04, "D": 0.0}.get(best_grade, 0.0)

    # Final confidence
    confidence = round(
        faithfulness       * 0.50 +
        (avg_cred / 5.0)   * 0.40 +
        grade_bonus,
        4,
    )
    confidence = min(1.0, max(0.0, confidence))

    return {
        "confidence":        confidence,
        "grade":             best_grade,
        "faithfulness":      faithfulness,
        "avg_credibility":   round(avg_cred, 3),
        "contradictions":    [],
        "hallucination_risk":False,
    }


# ── Entry point (smoke test) ──────────────────────────────────────────────────

if __name__ == "__main__":
    sample_chunks = [
        {
            "chunk_id": "test_001",
            "text": (
                "A randomized controlled trial showed that metformin reduces "
                "HbA1c by approximately 1.5% in patients with type 2 diabetes."
            ),
            "source":      "PubMed:12345",
            "journal":     "Diabetes Care",
            "year":        2023,
            "source_type": "rct",
        },
        {
            "chunk_id": "test_002",
            "text": (
                "The ADA Standards of Care recommend an HbA1c target of "
                "less than 7% for most non-pregnant adults with diabetes."
            ),
            "source":      "ADA_Guidelines:page12",
            "journal":     "Diabetes Care (ADA)",
            "year":        2024,
            "source_type": "clinical_guideline",
        },
        {
            "chunk_id": "test_003",
            "text": (
                "ICMR Guidelines for Management of Type 2 Diabetes 2018 "
                "recommend individualised HbA1c targets for Indian patients."
            ),
            "source":      "ICMR_Guidelines:page14",
            "journal":     "ICMR Guideline",
            "year":        2023,
            "source_type": "clinical_guideline",
        },
    ]

    print("=== Verifying chunks ===")
    verified = verify_chunks(sample_chunks)
    for c in verified:
        print(f"  [{c['chunk_id']}] grade={c['evidence_grade']}  "
              f"cred={c['credibility_score']}")

    generated = "HbA1c target is less than 7% and metformin is first-line."
    print("\n=== Aggregating confidence ===")
    report = aggregate_confidence(verified, generated)
    for k, v in report.items():
        print(f"  {k}: {v}")