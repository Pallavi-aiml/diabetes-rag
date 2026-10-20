"""Evidence scoring and lightweight answer verification.

Faithfulness uses NLI when available, with a lexical fallback. These scores
are heuristic indicators only; they are not clinical validation or calibrated
probabilities. Contradiction detection is heuristic and should be reviewed.
"""

import re
import sys
from functools import lru_cache
from pathlib import Path

# Allow imports from the project root.
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from config import FAITHFULNESS_THRESHOLD, CONTRADICTION_THRESHOLD
except Exception:
    FAITHFULNESS_THRESHOLD = 0.60
    CONTRADICTION_THRESHOLD = 0.50

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


@lru_cache(maxsize=1)
def _get_nli_pipeline():
    """Load the NLI model once and reuse it across calls."""
    if pipeline is None:
        return None
    try:
        return pipeline(
            "text-classification",
            model=NLI_MODEL_NAME,
            device=-1,
        )
    except Exception as exc:
        print(f"[NLI WARNING] Could not load model: {exc}")
        return None


SOURCE_SCORES = {
    "systematic_review": 5,
    "rct": 5,
    "clinical_guideline": 4,
    "cohort_study": 3,
    "case_report": 2,
    "expert_opinion": 1,
    "unknown": 2,
}

JOURNAL_SCORES = {
    "diabetes care (ada)": 5,
    "diabetes care": 5,
    "new england journal": 5,
    "nejm": 5,
    "the lancet": 5,
    "lancet": 5,
    "jama": 5,
    "nature medicine": 5,
    "cell metabolism": 5,
    "bmj": 4,
    "diabetologia": 4,
    "annals of internal medicine": 4,
    "icmr guideline": 4,
    "who guideline": 4,
    "clinical guideline": 4,
    "plos one": 3,
    "scientific reports": 3,
    "frontiers": 3,
    "bmc": 3,
    "unknown": 2,
}

GRADE_ORDER = {"A": 4, "B": 3, "C": 2, "D": 1}
GRADE_BONUS = {"A": 0.10, "B": 0.07, "C": 0.04, "D": 0.0}
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]*")
# Only explicit negation/absence terms are used for this very lightweight
# polarity heuristic. Directional words such as "lower" or "reduced" are not
# negations and caused false contradiction flags in earlier evaluations.
_NEGATIONS = {
    "no", "not", "never", "without", "unlikely", "absent", "neither",
}


def _safe_year(value):
    """Convert a year to an integer, using 2000 when unavailable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 2000


def score_credibility(chunk: dict) -> float:
    """Return a heuristic source credibility score between 0 and 5."""
    source_type = str(chunk.get("source_type") or "unknown").lower()
    journal = str(chunk.get("journal") or "unknown").lower()
    year = _safe_year(chunk.get("year", 2000))

    source_score = SOURCE_SCORES.get(source_type, SOURCE_SCORES["unknown"])
    journal_score = 2
    for key in sorted(JOURNAL_SCORES, key=len, reverse=True):
        if key in journal:
            journal_score = JOURNAL_SCORES[key]
            break

    recency = min(5.0, max(0.0, (year - 2010) * 0.5))
    score = source_score * 0.50 + journal_score * 0.30 + recency * 0.20
    return round(min(5.0, max(0.0, score)), 3)


def classify_evidence_grade(chunk: dict) -> str:
    """Assign a heuristic evidence grade; not a formal GRADE assessment."""
    source_type = str(chunk.get("source_type") or "unknown").lower()
    text = str(chunk.get("text") or "").lower()
    journal = str(chunk.get("journal") or "").lower()

    if source_type in ("systematic_review", "rct"):
        return "A"
    if source_type == "clinical_guideline" or any(
        term in journal for term in ("icmr", "ada", "who", "guideline")
    ):
        return "B"
    if source_type == "cohort_study":
        return "B"
    if source_type == "case_report":
        return "C"
    if any(term in text for term in (
        "systematic review", "meta-analysis", "randomized controlled trial",
        "randomised controlled trial",
    )):
        return "A"
    if any(term in text for term in (
        "cohort study", "clinical guideline", "standards of care", "recommendation",
    )):
        return "B"
    if any(term in text for term in (
        "case-control", "cross-sectional", "pilot study", "case series", "survey",
    )):
        return "C"
    return "D"


def _normalise(text: str) -> str:
    """Lowercase text and keep word-like tokens."""
    return " ".join(_WORD_RE.findall(str(text or "").lower()))


def _tokens(text: str) -> list:
    """Tokenize text while excluding common function words."""
    stop_words = {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on",
        "for", "with", "by", "as", "is", "are", "was", "were", "be",
        "been", "being", "this", "that", "these", "those", "it", "its",
        "from", "at", "which", "who", "what", "when", "where", "how",
        "can", "may", "might", "should", "could", "would", "than", "then",
        "also", "such", "their", "they",
    }
    return [
        token for token in _WORD_RE.findall(str(text or "").lower())
        if token not in stop_words and len(token) > 1
    ]


def _sentences(text: str) -> list:
    """Split text into sentence-like units, protecting common abbreviations."""
    if not text:
        return []
    text = str(text).strip()
    text = re.sub(r"\[SOURCE\s+\d+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)

    abbreviations = {
        "U.S.": "USABBR", "U.K.": "UKABBR", "e.g.": "EGABBR",
        "i.e.": "IEABBR", "Dr.": "DRABBR", "Mr.": "MRABBR",
        "Mrs.": "MRSABBR", "Ms.": "MSABBR", "vs.": "VSABBR",
        "etc.": "ETCABBR", "Prof.": "PROFABBR", "No.": "NOABBR",
    }
    for abbreviation, placeholder in abbreviations.items():
        text = re.sub(re.escape(abbreviation), placeholder, text, flags=re.IGNORECASE)

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = []
    for part in parts:
        sentence = part.strip()
        for abbreviation, placeholder in abbreviations.items():
            sentence = re.sub(placeholder, abbreviation, sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"^[-*+]\s+", "", sentence)
        sentence = re.sub(r"^\d+[.)]\s+", "", sentence).strip()
        if not sentence or sentence.startswith("#"):
            continue
        if len(_tokens(sentence)) >= 4:
            sentences.append(sentence)
    return sentences


def _support_score(sentence: str, context: str) -> float:
    """Estimate lexical token coverage; this is not semantic entailment."""
    claim_tokens = set(_tokens(sentence))
    context_tokens = set(_tokens(context))
    if not claim_tokens or not context_tokens:
        return 0.0

    def normalize_token(token: str) -> str:
        token = token.lower()
        for suffix in ("ies", "ing", "ed", "es", "s"):
            if len(token) > len(suffix) + 3 and token.endswith(suffix):
                if suffix == "ies":
                    token = token[:-3] + "y"
                else:
                    token = token[:-len(suffix)]
                break
        return token

    normalized_claim = {normalize_token(t) for t in claim_tokens}
    normalized_context = {normalize_token(t) for t in context_tokens}
    overlap = len(normalized_claim & normalized_context) / len(normalized_claim)
    claim_phrase = " ".join(normalize_token(t) for t in _tokens(sentence))
    context_phrase = " ".join(normalize_token(t) for t in _tokens(context))
    phrase_bonus = 0.10 if len(claim_phrase) >= 5 and claim_phrase in context_phrase else 0.0
    return round(min(1.0, overlap + phrase_bonus), 4)


def check_faithfulness(generated_answer: str, context_chunk: str) -> float:
    """Estimate answer support against one context string."""
    return check_faithfulness_multi(generated_answer, [{"text": context_chunk}])


def _entailment_probability(result, nli=None):
    """Extract entailment score from flat or nested Transformers outputs.

    Some Transformers versions return one list of label dictionaries per input;
    others return a flat list. Resolve LABEL_n using the model's own id2label
    mapping instead of assuming a particular class index.
    """
    items = []
    def flatten(value):
        if isinstance(value, dict):
            items.append(value)
        elif isinstance(value, (list, tuple)):
            for child in value:
                flatten(child)
    flatten(result)

    id2label = {}
    try:
        id2label = getattr(nli.model.config, "id2label", {}) or {}
    except Exception:
        pass
    normalized_id2label = {}
    for key, value in id2label.items():
        normalized_id2label[str(key).upper()] = str(value).lower()
        try:
            normalized_id2label[f"LABEL_{int(key)}"] = str(value).lower()
        except (TypeError, ValueError):
            pass

    for item in items:
        label = str(item.get("label", "")).strip().lower()
        if label.startswith("label_"):
            mapped = normalized_id2label.get(label.upper(), "")
            if not mapped:
                # A bare LABEL_n is not enough to identify entailment reliably.
                continue
            label = mapped
        normalized = label.replace("-", "_").replace(" ", "_")
        if "entail" in normalized:
            try:
                score = float(item.get("score", 0.0))
                return min(1.0, max(0.0, score))
            except (TypeError, ValueError):
                return None
    # Unknown output labels should trigger the lexical fallback, not a false zero.
    return None


def check_faithfulness_multi(generated_answer: str, chunks: list) -> float:
    """Score answer sentences against retrieved evidence using NLI.

    Each evidence sentence is treated as the premise and each answer sentence
    as the hypothesis. Scores are heuristic indicators, not calibrated
    probabilities or clinical validation.
    """
    if not chunks or not generated_answer:
        return 0.0

    contexts = [
        str(chunk.get("text", "")).strip()
        for chunk in chunks
        if isinstance(chunk, dict) and str(chunk.get("text", "")).strip()
    ]
    if not contexts:
        return 0.0

    answer = re.sub(
        r"\[SOURCE\s+\d+\]", "", str(generated_answer), flags=re.IGNORECASE
    )
    sentences = []
    for line in answer.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        sentences.extend(_sentences(line))

    def is_factual_claim(sentence: str) -> bool:
        text = sentence.strip().lower()
        if not text:
            return False

        excluded_phrases = (
            "this information is for educational purposes",
            "does not constitute medical advice",
            "consult a qualified healthcare professional",
            "before making any clinical decisions",
        )
        if any(phrase in text for phrase in excluded_phrases):
            return False

        headings = {
            "major risk factors for type 2 diabetes",
            "lifestyle changes to reduce risk",
            "lifestyle changes",
            "major risk factors",
        }
        if text.lstrip("# ").rstrip(":") in headings:
            return False
        return len(_tokens(text)) >= 4

    sentences = [s for s in sentences if is_factual_claim(s)]
    if not sentences:
        print("[FAITHFULNESS DEBUG] No factual sentences detected.")
        return 0.0

    nli = _get_nli_pipeline()
    scores = []

    # Normalize PDF line wraps and split evidence into sentence-like units.
    evidence_units = []
    for context in contexts:
        normalized_context = re.sub(r"-\s*\n\s*", "", context)
        normalized_context = re.sub(r"\s+", " ", normalized_context).strip()
        units = re.split(r"(?<=[.!?])\s+", normalized_context)
        evidence_units.extend(
            unit.strip() for unit in units if len(_tokens(unit)) >= 4
        )
    if not evidence_units:
        evidence_units = contexts

    for sentence in sentences:
        sentence_score = None
        method = "nli"
        ranked_evidence = []

        if nli is not None:
            try:
                # Ignore a short leading bullet heading (for example,
                # "Physical Inactivity:") so NLI scores the factual claim.
                claim_text = re.sub(
                    r"^[A-Za-z][A-Za-z0-9 /()_-]{1,60}:\s*",
                    "",
                    sentence,
                ).strip()
                if not claim_text:
                    claim_text = sentence.strip()

                pairs = [
                    {
                        "text": evidence[:512],
                        "text_pair": claim_text[:512],
                    }
                    for evidence in evidence_units
                ]
                predictions = nli(
                    pairs,
                    batch_size=8,
                    truncation=True,
                    top_k=None,
                )
                entailment_scores = [
                    _entailment_probability(prediction, nli=nli)
                    for prediction in predictions
                ]
                if entailment_scores:
                    best_index = max(
                        range(len(entailment_scores)),
                        key=lambda index: entailment_scores[index],
                    )
                    sentence_score = float(entailment_scores[best_index])
                    ranked_evidence = sorted(
                        zip(evidence_units, entailment_scores),
                        key=lambda item: item[1],
                        reverse=True,
                    )
            except Exception as exc:
                print(f"[NLI WARNING] Scoring failed: {exc}")
                sentence_score = None

        if sentence_score is None:
            method = "lexical_fallback"
            sentence_score = max(
                (_support_score(sentence, context) for context in contexts),
                default=0.0,
            )

        scores.append(float(sentence_score))
        print(
            f"[FAITHFULNESS DEBUG] method={method} "
            f"score={sentence_score:.4f} sentence={sentence[:180]!r}"
        )

        if sentence_score < 0.60:
            print("\n--- LOW FAITHFULNESS DEBUG ---")
            print("CLAIM:", sentence)
            print(f"BEST SCORE: {sentence_score:.4f}")
            if ranked_evidence:
                for evidence, score in ranked_evidence[:3]:
                    print("EVIDENCE:", evidence[:300])
                    print(f"ENTAILMENT: {score:.4f}")
            else:
                print("No ranked NLI evidence available.")
            print("--- END DEBUG ---\n")

    overall_score = sum(scores) / len(scores)
    print(
        f"[FAITHFULNESS DEBUG] sentences={len(scores)}, "
        f"mean={overall_score:.4f}"
    )
    return round(overall_score, 4)


def detect_contradiction(chunk_a: str, chunk_b: str) -> bool:
    """Conservatively flag likely polarity conflicts for manual review.

    This is not semantic contradiction detection. It only flags sentences with
    strong shared topical overlap and differing explicit negation polarity.
    """
    sentences_a = _sentences(str(chunk_a or ""))
    sentences_b = _sentences(str(chunk_b or ""))

    def has_explicit_negation(sentence: str) -> bool:
        return any(token in _NEGATIONS for token in _tokens(sentence))

    for sentence_a in sentences_a:
        words_a = set(_tokens(sentence_a))
        for sentence_b in sentences_b:
            words_b = set(_tokens(sentence_b))
            overlap = words_a & words_b
            distinctive = {token for token in overlap if len(token) >= 5}
            # Require substantial shared wording to reduce false positives.
            if len(overlap) < 5 or len(distinctive) < 3:
                continue
            if has_explicit_negation(sentence_a) != has_explicit_negation(sentence_b):
                return True
    return False


def find_contradictions(chunks: list) -> list:
    """Return pairs of chunks flagged by the contradiction heuristic."""
    found = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            chunk_a = chunks[i] if isinstance(chunks[i], dict) else {"text": str(chunks[i])}
            chunk_b = chunks[j] if isinstance(chunks[j], dict) else {"text": str(chunks[j])}
            text_a = str(chunk_a.get("text", ""))
            text_b = str(chunk_b.get("text", ""))
            if detect_contradiction(text_a, text_b):
                found.append({
                    "chunk_a": chunk_a.get("chunk_id", f"chunk_{i}"),
                    "chunk_b": chunk_b.get("chunk_id", f"chunk_{j}"),
                    "message": "Potential polarity conflict flagged by a heuristic; manual review is required.",
                })
    return found


def verify_chunks(chunks: list) -> list:
    """Add heuristic credibility scores and evidence grades to chunks."""
    for chunk in chunks:
        chunk["credibility_score"] = score_credibility(chunk)
        chunk["evidence_grade"] = classify_evidence_grade(chunk)
    return chunks


def aggregate_confidence(verified_chunks: list, generated_answer: str) -> dict:
    """Aggregate heuristic evidence-quality indicators, not calibrated probability."""
    if not verified_chunks:
        return {
            "confidence": 0.0,
            "grade": "D",
            "faithfulness": 0.0,
            "avg_credibility": 0.0,
            "contradictions": [],
            "hallucination_risk": True,
        }

    faithfulness = check_faithfulness_multi(generated_answer, verified_chunks)
    safe_chunks = [chunk for chunk in verified_chunks if isinstance(chunk, dict)]
    if not safe_chunks:
        return {
            "confidence": 0.0,
            "grade": "D",
            "faithfulness": 0.0,
            "avg_credibility": 0.0,
            "contradictions": [],
            "hallucination_risk": True,
        }
    verified_chunks = safe_chunks
    credibility_scores = [
        float(chunk.get("credibility_score", score_credibility(chunk)))
        for chunk in verified_chunks
    ]
    avg_credibility = sum(credibility_scores) / len(credibility_scores)
    best_grade = max(
        (str(chunk.get("evidence_grade", "D")) for chunk in verified_chunks),
        key=lambda grade: GRADE_ORDER.get(grade, 0),
    )
    contradictions = find_contradictions(verified_chunks)
    confidence = (
        faithfulness * 0.50
        + (avg_credibility / 5.0) * 0.40
        + GRADE_BONUS.get(best_grade, 0.0)
    )
    confidence = round(min(1.0, max(0.0, confidence)), 4)
    faithfulness_threshold = float(globals().get("FAITHFULNESS_THRESHOLD", 0.60))
    hallucination_risk = faithfulness < faithfulness_threshold or bool(contradictions)

    return {
        "confidence": confidence,
        "grade": best_grade,
        "faithfulness": faithfulness,
        "avg_credibility": round(avg_credibility, 3),
        "contradictions": contradictions,
        "hallucination_risk": hallucination_risk,
    }


if __name__ == "__main__":
    test_chunks = [
        {
            "chunk_id": "guideline_1",
            "text": "Overweight and obesity increase the risk of type 2 diabetes.",
            "journal": "Diabetes Care (ADA)",
            "year": 2024,
            "source_type": "clinical_guideline",
        },
        {
            "chunk_id": "guideline_2",
            "text": "Smoking increases the risk of type 2 diabetes.",
            "journal": "WHO Guideline",
            "year": 2023,
            "source_type": "clinical_guideline",
        },
    ]
    test_answer = (
        "Overweight and obesity increase the risk of type 2 diabetes. "
        "Smoking increases the risk of type 2 diabetes."
    )
    verified = verify_chunks(test_chunks)
    print("Verification result:")
    print(aggregate_confidence(verified, test_answer))
    print("\nContradiction test:")
    print(detect_contradiction(
        "Physical activity reduces the risk of type 2 diabetes.",
        "Physical activity does not reduce the risk of type 2 diabetes.",
    ))
