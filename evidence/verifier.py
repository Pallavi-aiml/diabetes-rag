"""Evidence scoring and answer verification.

Pipeline: credibility score + heuristic grade per chunk, then CLAIM-LEVEL
faithfulness of the generated answer. Each answer sentence (claim) is checked
by NLI against windows of 1..N consecutive evidence sentences, preferring the
source(s) the answer actually cited via [SOURCE n].

All scores are heuristic indicators, not calibrated probabilities or clinical
validation. Contradiction detection is a polarity heuristic for manual review.
"""

import logging
import re
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from config import (
        FAITHFULNESS_THRESHOLD, CONTRADICTION_THRESHOLD, NLI_MODEL,
        NLI_MAX_EVIDENCE_UNITS, FAITHFULNESS_DEBUG, ABSTAIN_FAITHFULNESS_THRESHOLD,
        NLI_WINDOW_SENTENCES, CLAIM_SUPPORT_THRESHOLD, SUPPORTED_RATIO_MIN,
        RISKY_CONFIDENCE_CAP, INSUFFICIENT_CONFIDENCE_CAP,
    )
except Exception:  # keeps the module importable stand-alone
    FAITHFULNESS_THRESHOLD = 0.60
    CONTRADICTION_THRESHOLD = 0.50
    NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
    NLI_MAX_EVIDENCE_UNITS = 8
    FAITHFULNESS_DEBUG = False
    ABSTAIN_FAITHFULNESS_THRESHOLD = 0.15
    NLI_WINDOW_SENTENCES = 3
    CLAIM_SUPPORT_THRESHOLD = 0.50
    SUPPORTED_RATIO_MIN = 0.70
    RISKY_CONFIDENCE_CAP = 0.50
    INSUFFICIENT_CONFIDENCE_CAP = 0.30

try:
    from transformers import pipeline
except ImportError:  # lexical fallback is used when transformers is missing
    pipeline = None

logger = logging.getLogger(__name__)
NLI_MODEL_NAME = NLI_MODEL
# Matches [SOURCE 2], [SOURCE 1, SOURCE 3], [SOURCE 1, 3], [SOURCE 2 and 4]  (models cite all of these ways)
_CITE_RE = re.compile(r"\[\s*SOURCES?\s+\d+(?:\s*(?:,|;|and|&)\s*(?:SOURCES?\s*)?\d+)*\s*\]", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+")
NLI_WINDOWS_PER_SOURCE = 4

# Entity grounding: NLI models barely notice a swapped drug name ("metformin" -> "glimepiride"), but a wrong drug is
# exactly the error that matters clinically. Every drug / drug class named in a claim must appear somewhere in the
# retrieved evidence, otherwise the claim is capped as unsupported. Set ENTITY_GROUNDING=0 to disable.
import os as _os
ENTITY_GROUNDING = _os.getenv("ENTITY_GROUNDING", "1").strip().lower() not in ("0", "false", "no")
ENTITY_UNGROUNDED_SCORE = 0.2
NUMBER_GROUNDING = _os.getenv("NUMBER_GROUNDING", "1").strip().lower() not in ("0", "false", "no")
_NUM_UNIT = (r"(?:%|\s?mL(?:/min)?|\s?mg/dL|\s?mmol/L|\s?mmol/mol|\s?mmHg|\s?mg|\s?g\b|\s?kg|\s?years?|\s?months?|"
             r"\s?weeks?|\s?days?|\s?minutes?|\s?min\b|\s?hours?|\s?units?|\s?IU)")
_CLINICAL_NUMBER_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)(?=" + _NUM_UNIT + r"|(?:/\d))|(?<=[<>\u2264\u2265])\s?(\d+(?:\.\d+)?)", re.IGNORECASE)
_ANY_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")


def _clinical_numbers(claim: str) -> set:
    """Numbers in a claim that carry a unit, a % or a </> threshold (never the 2 in 'type 2')."""
    found = set()
    for m in _CLINICAL_NUMBER_RE.finditer(str(claim or "")):
        group = 1 if m.group(1) else 2
        before = claim[:m.start(group)].lower().rstrip()
        if before.endswith(("type", "stage", "source", "grade", "glp-", "sglt", "class")):
            continue
        found.add(float(m.group(group)))
    return found


def _all_numbers(text: str) -> set:
    return {float(x) for x in _ANY_NUMBER_RE.findall(str(text or ""))}


_DRUG_CLASSES = {
    "metformin": r"metformin",
    "sulfonylurea": r"sulfonylureas?|glimepiride|glipizide|glyburide|glibenclamide|gliclazide|tolbutamide",
    "insulin": r"insulins?|glargine|detemir|degludec|lispro|aspart",
    "sglt2": r"sglt-?2|empagliflozin|dapagliflozin|canagliflozin|ertugliflozin|gliflozins?",
    "glp1": r"glp-?1|semaglutide|liraglutide|dulaglutide|exenatide|lixisenatide|tirzepatide",
    "dpp4": r"dpp-?4|gliptins?|sitagliptin|linagliptin|saxagliptin|alogliptin|vildagliptin",
    "tzd": r"thiazolidinediones?|tzds?|pioglitazone|rosiglitazone",
    "statin": r"statins?|atorvastatin|rosuvastatin|simvastatin|pravastatin",
    "fibrate": r"fibrates?|fenofibrate|gemfibrozil",
    "ezetimibe": r"ezetimibe",
    "acei": r"ace inhibitors?|aceis?|angiotensin[- ]converting|lisinopril|enalapril|ramipril",
    "arb": r"arbs?|angiotensin receptor blockers?|losartan|valsartan|irbesartan",
    "ccb": r"calcium channel blockers?|ccbs?|amlodipine|nifedipine|diltiazem|verapamil",
    "betablocker": r"beta[- ]blockers?|metoprolol|atenolol|carvedilol",
    "diuretic": r"diuretics?|thiazides?|chlorthalidone|furosemide",
    "aspirin": r"aspirin",
    "gabapentinoid": r"gabapentin|pregabalin",
    "snri": r"duloxetine|venlafaxine",
    "tca": r"amitriptyline|nortriptyline|tricyclic",
    "mra": r"finerenone|spironolactone|eplerenone",
    "acarbose": r"acarbose|alpha-glucosidase",
}
_DRUG_RES = {name: re.compile(r"\b(?:" + pattern + r")\b", re.IGNORECASE) for name, pattern in _DRUG_CLASSES.items()}


def _drug_classes(text: str) -> set:
    """Canonical drug classes mentioned in a text (members map to their class: glimepiride -> sulfonylurea)."""
    text = str(text or "")
    return {name for name, rx in _DRUG_RES.items() if rx.search(text)}

# Framing that makes a SUPPORTED sentence look unsupported to an NLI model ("Additionally, ...",
# "The WHO guideline notes that ...", "Based on the provided evidence, ...").
_LEADING_CONNECTIVE_RE = re.compile(
    r"^\s*(additionally|moreover|furthermore|however|also|specifically|in addition|further|notably|importantly|"
    r"therefore|thus|while)\s*,?\s+", re.IGNORECASE)
_LEADING_BASED_ON_RE = re.compile(
    r"^\s*based on (the )?(provided |retrieved |available )?(evidence|sources?|context|text)\s*,?\s*", re.IGNORECASE)
_ATTRIBUTION_RE = re.compile(
    r"^\s*(the\s+)?(provided\s+)?(evidence|text|sources?|context|who|icmr|ada|american diabetes association|"
    r"guidelines?|study|review|report)(\s+guidelines?)?\s+"
    r"(also\s+)?(notes?|states?|indicates?|suggests?|describes?|reports?|shows?|mentions?|identifies?|recommends?)"
    r"\s+(that\s+)?", re.IGNORECASE)
_PRONOUN_START_RE = re.compile(r"^\s*(this|these|it|they|that|those|its|their)\b", re.IGNORECASE)


def _normalize_claim_for_nli(claim: str) -> str:
    """Remove discourse framing so the NLI model judges the factual content only."""
    text = str(claim or "").strip()
    for _ in range(2):
        before = text
        text = _LEADING_CONNECTIVE_RE.sub("", text)
        text = _LEADING_BASED_ON_RE.sub("", text)
        text = _ATTRIBUTION_RE.sub("", text)
        if text == before:
            break
    text = text.strip()
    return (text[:1].upper() + text[1:]) if text else str(claim or "").strip()


def _cited_numbers(text: str) -> list:
    return sorted({int(n) for block in _CITE_RE.findall(str(text or "")) for n in _NUM_RE.findall(block)})
_CURRENT_YEAR = datetime.now().year

# Sentences where the answer itself says the evidence is missing/limited are HEDGES,
# not factual assertions: scoring them with NLI against the sources is meaningless
# (a source cannot "entail" that it lacks something) and used to produce false alarms.
_HEDGE_RE = re.compile(
    r"(does|do|did|is|are)\s+not\s+(explicitly\s+|clearly\s+|directly\s+)?"
    r"(state|specify|specif|define|establish|provide|address|mention|contain|report|include|"
    r"determine|identify|give|offer)"
    r"|not\s+(enough|sufficient)\s+(evidence|information|data)"
    r"|(evidence|information|data|sources?)\s+(is|are)\s+(insufficient|inadequate|limited|lacking|unclear)"
    r"|insufficient\s+(evidence|to\s)|cannot\s+be\s+(determined|established|concluded)"
    r"|unable\s+to\s+(determine|establish|conclude)|no\s+(clear|explicit|direct)\s+(evidence|statement|recommendation)",
    re.IGNORECASE,
)


def is_hedge_sentence(sentence: str) -> bool:
    return bool(_HEDGE_RE.search(str(sentence or "")))


@lru_cache(maxsize=1)
def _get_nli_pipeline():
    """Load the NLI model once and reuse it across calls."""
    if pipeline is None:
        return None
    try:
        return pipeline("text-classification", model=NLI_MODEL_NAME, device=-1)
    except Exception as exc:
        logger.warning("Could not load NLI model %s: %s", NLI_MODEL_NAME, exc)
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
    """Return a plausible publication year, else 2000 (treated as unknown/old)."""
    try:
        year = int(value)
    except (TypeError, ValueError):
        return 2000
    return year if 1950 <= year <= _CURRENT_YEAR + 1 else 2000


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
    text = _CITE_RE.sub("", text)
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
    return _label_probability(result, nli, "entail")


def _label_probability(result, nli=None, needle="entail"):
    """Extract the probability of the label containing `needle` (entail / contradict) from flat or nested Transformers outputs.

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
        if needle in normalized:
            try:
                score = float(item.get("score", 0.0))
                return min(1.0, max(0.0, score))
            except (TypeError, ValueError):
                return None
    # Unknown output labels should trigger the lexical fallback, not a false zero.
    return None


def _split_answer_claims(answer: str) -> list:
    """Split an answer into factual claim sentences, keeping [SOURCE n] citations.

    Returns a list of {"claim": text-without-citations, "cited": [ints]}.
    Disclaimers, headings and very short fragments are dropped.
    """
    excluded = (
        "this information is for educational purposes",
        "does not constitute medical advice",
        "consult a qualified healthcare professional",
        "please consult",
        "consult a healthcare",
        "consult your"  ,
        "before making any clinical decisions",
        "for educational information only",
    )
    answer = re.sub(r"<think>.*?</think>", "", str(answer or ""), flags=re.DOTALL | re.IGNORECASE)
    claims = []
    for line in answer.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        for sentence in _split_keep_citations(line):
            cited = _cited_numbers(sentence)
            clean = _CITE_RE.sub("", sentence).strip()
            clean = re.sub(r"\s+", " ", clean)
            low = clean.lower()
            if any(phrase in low for phrase in excluded):
                continue
            if clean.endswith(":") or len(_tokens(clean)) < 4:
                continue
            claims.append({"claim": clean, "cited": cited, "hedge": is_hedge_sentence(clean)})
    return claims


def _split_keep_citations(text: str) -> list:
    """Sentence-split like _sentences() but without deleting [SOURCE n] markers."""
    placeholders = {}
    def _stash(match):
        key = f"CITEPH{len(placeholders)}X"
        placeholders[key] = match.group(0)
        return key
    stashed = _CITE_RE.sub(_stash, text)
    pieces = []
    for sentence in _sentences(stashed):
        for key, original in placeholders.items():
            sentence = sentence.replace(key, original)
        pieces.append(sentence)
    # _sentences() drops nothing but placeholders for text that had only citations
    return pieces


def _evidence_windows(chunk_text: str, window: int) -> list:
    """Sliding windows of 1..window consecutive sentences from one chunk."""
    normalized = re.sub(r"-\s*\n\s*", "", str(chunk_text or ""))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    units = [u.strip() for u in re.split(r"(?<=[.!?])\s+", normalized) if len(_tokens(u)) >= 4]
    if not units:
        return [normalized] if normalized else []
    windows = []
    for size in range(1, max(1, window) + 1):
        for i in range(0, len(units) - size + 1):
            windows.append(" ".join(units[i:i + size]))
    return windows


def _score_claim(claim: str, premises: list, nli, limit: int = None) -> tuple:
    """Return (score, method, best_premise) for one claim against candidate premises."""
    if not premises:
        return 0.0, "no_evidence", ""
    claim = _normalize_claim_for_nli(claim)
    claim_tokens = set(_tokens(claim))

    def overlap(premise: str) -> float:
        return len(claim_tokens & set(_tokens(premise))) / max(1, len(claim_tokens))

    # Cheap lexical pre-ranking keeps NLI cost bounded (NLI_MAX_PREMISES per claim).
    ranked = sorted(premises, key=overlap, reverse=True)[:max(1, int(limit or NLI_MAX_EVIDENCE_UNITS or 8))]

    if nli is not None:
        try:
            claim_text = re.sub(r"^[A-Za-z][A-Za-z0-9 /()_-]{1,60}:\s*", "", claim).strip() or claim
            pairs = [{"text": p[:512], "text_pair": claim_text[:512]} for p in ranked]
            predictions = nli(pairs, batch_size=8, truncation=True, top_k=None)
            scores = [_entailment_probability(pred, nli=nli) for pred in predictions]
            if scores and all(sc is not None for sc in scores):
                best = max(range(len(scores)), key=lambda i: scores[i])
                return float(scores[best]), "nli", ranked[best]
        except Exception as exc:  # fall through to lexical
            logger.warning("NLI scoring failed (%s); using lexical fallback", exc)

    best_premise, best_score = "", 0.0
    for premise in ranked:
        score = _support_score(claim, premise)
        if score > best_score:
            best_premise, best_score = premise, score
    return float(best_score), "lexical_fallback", best_premise


def _top_windows(claim: str, windows: list, k: int) -> list:
    """The k evidence windows of ONE source that share the most words with the claim."""
    claim_tokens = set(_tokens(claim))
    return sorted(windows, key=lambda w: len(claim_tokens & set(_tokens(w))), reverse=True)[:k]


def verify_answer(generated_answer: str, chunks: list) -> dict:
    """Claim-level faithfulness report.

    For every factual sentence: if it cites [SOURCE n], it is checked against that
    source first (a citation to a source that does not support it counts against the
    answer); otherwise against all retrieved chunks.

    Returns
      faithfulness        mean per-claim support score (0..1)
      supported_ratio     share of claims with score >= CLAIM_SUPPORT_THRESHOLD
      claims              per-claim details (score, method, cited, citation_ok)
      unsupported_claims  list of claim strings below threshold
      citation_coverage   share of claims that carry a valid [SOURCE n]
    """
    empty = {"faithfulness": 0.0, "supported_ratio": 0.0, "claims": [],
             "unsupported_claims": [], "citation_coverage": 0.0,
             "insufficient_evidence": False, "hedge_sentences": 0}
    chunks = [c for c in (chunks or []) if isinstance(c, dict) and str(c.get("text", "")).strip()]
    claims = _split_answer_claims(generated_answer)
    if not chunks or not claims:
        return empty

    all_claims = claims
    hedges = [c for c in all_claims if c.get("hedge")]
    claims = [c for c in all_claims if not c.get("hedge")]
    # The answer is mostly "the evidence doesn't say" -> treat as an insufficient-evidence
    # answer (a soft abstention), not as a hallucination.
    insufficient = len(hedges) >= max(1, len(all_claims) / 2)
    if not claims:
        out = dict(empty)
        out.update({"insufficient_evidence": insufficient, "hedge_sentences": len(hedges)})
        return out

    nli = _get_nli_pipeline()
    windows_by_source = {
        i: _evidence_windows(c.get("text", ""), NLI_WINDOW_SENTENCES)
        for i, c in enumerate(chunks, start=1)
    }
    all_windows = [w for ws in windows_by_source.values() for w in ws]

    evidence_classes = _drug_classes(" ".join(str(c.get("text", "")) for c in chunks))
    evidence_numbers = _all_numbers(" ".join(str(c.get("text", "")) for c in chunks))
    spread_limit = min(24, max(int(NLI_MAX_EVIDENCE_UNITS or 8), 3 * len(windows_by_source)))

    def spread_windows(claim_text: str) -> list:
        """Best 3 windows from EVERY retrieved source (instead of the global top-N by word overlap)."""
        return [w for ws in windows_by_source.values() for w in _top_windows(claim_text, ws, 3)]

    details, valid_cites = [], 0
    prev_claim = ""
    for item in claims:
        cited_valid = [n for n in item["cited"] if n in windows_by_source]
        if cited_valid:
            valid_cites += 1
            # Take the best windows PER cited source, so a claim citing 5 sources is not judged on the
            # 8 highest word-overlap windows overall (which can all come from the same decoy chunk).
            premises = [w for n in cited_valid
                        for w in _top_windows(item["claim"], windows_by_source[n], NLI_WINDOWS_PER_SOURCE)]
            limit = min(24, max(int(NLI_MAX_EVIDENCE_UNITS or 8), NLI_WINDOWS_PER_SOURCE * len(cited_valid)))
            score, method, best = _score_claim(item["claim"], premises, nli, limit=limit)
            scope = "cited"
            # A wrong citation is a defect even if another chunk happens to support it.
            if score < CLAIM_SUPPORT_THRESHOLD:
                alt_score, alt_method, alt_best = _score_claim(item["claim"], spread_windows(item["claim"]), nli,
                                                               limit=spread_limit)
                if alt_score > score:
                    score, method, best = alt_score * 0.85, alt_method, alt_best
                    scope = "cited_wrong_source"
        else:
            score, method, best = _score_claim(item["claim"], spread_windows(item["claim"]), nli, limit=spread_limit)
            scope = "uncited"
        # "This target applies to..." cannot be judged without the sentence it refers to: also try the
        # previous claim + this one, and keep the better score.
        ungrounded = []
        if ENTITY_GROUNDING:
            ungrounded = sorted(_drug_classes(item["claim"]) - evidence_classes)
        ungrounded_numbers = sorted(_clinical_numbers(item["claim"]) - evidence_numbers) if NUMBER_GROUNDING else []
        if prev_claim and _PRONOUN_START_RE.match(item["claim"]) and score < CLAIM_SUPPORT_THRESHOLD and not ungrounded:
            src_windows = ([w for n in cited_valid
                            for w in _top_windows(item["claim"], windows_by_source[n], NLI_WINDOWS_PER_SOURCE)]
                           if cited_valid else spread_windows(item["claim"]))
            ctx_score, ctx_method, ctx_best = _score_claim(f"{prev_claim} {item['claim']}", src_windows, nli,
                                                           limit=spread_limit)
            if ctx_score > score:
                score, method, best = ctx_score, f"{ctx_method}+ctx", ctx_best
        prev_claim = item["claim"]
        if ungrounded and score > ENTITY_UNGROUNDED_SCORE:
            score, method = ENTITY_UNGROUNDED_SCORE, f"{method}+entity"
        if ungrounded_numbers and score > ENTITY_UNGROUNDED_SCORE:
            score, method = ENTITY_UNGROUNDED_SCORE, f"{method}+number"
        details.append({
            "claim": item["claim"], "score": round(score, 4), "method": method,
            "ungrounded_entities": ungrounded,
            "ungrounded_numbers": ungrounded_numbers,
            "cited": item["cited"], "scope": scope,
            "supported": score >= CLAIM_SUPPORT_THRESHOLD,
            "best_evidence": best[:300],
        })

    if FAITHFULNESS_DEBUG:
        for d in details:
            print(f"[FAITHFULNESS DEBUG] {d['method']}/{d['scope']} score={d['score']:.3f} "
                  f"cited={d['cited']} claim={d['claim'][:140]!r}")
    n = len(details)
    mean_score = sum(d["score"] for d in details) / n
    supported = sum(1 for d in details if d["supported"])
    return {
        "faithfulness": round(mean_score, 4),
        "supported_ratio": round(supported / n, 4),
        "claims": details,
        "unsupported_claims": [d["claim"] for d in details if not d["supported"]],
        "citation_coverage": round(valid_cites / n, 4),
        "insufficient_evidence": insufficient,
        "hedge_sentences": len(hedges),
    }


def check_faithfulness_multi(generated_answer: str, chunks: list) -> float:
    """Backward-compatible wrapper: mean claim-level faithfulness (0..1)."""
    return verify_answer(generated_answer, chunks)["faithfulness"]


def detect_contradiction(chunk_a: str, chunk_b: str) -> bool:
    """Conservatively flag likely polarity conflicts for manual review.

    This is not semantic contradiction detection. It only flags sentences with
    strong shared topical overlap and differing explicit negation polarity.
    """
    sentences_a = _sentences(str(chunk_a or ""))
    sentences_b = _sentences(str(chunk_b or ""))

    def has_explicit_negation(sentence: str) -> bool:
        return any(token in _NEGATIONS for token in _tokens(sentence))

    candidates = []
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
                candidates.append((len(overlap), sentence_a, sentence_b))
    if not candidates:
        return False

    # The polarity heuristic alone is trigger-happy ("metformin remains first-line ... not ..." vs
    # "metformin is first-line"), and a flagged pair can make the API refuse the whole answer.
    # When an NLI model is available, only keep pairs it also labels as a contradiction.
    nli = _get_nli_pipeline()
    if nli is None:
        return True
    candidates.sort(key=lambda item: item[0], reverse=True)
    inputs = []
    for _, sa, sb in candidates[:6]:
        inputs.append({"text": sa[:512], "text_pair": sb[:512]})
        inputs.append({"text": sb[:512], "text_pair": sa[:512]})
    try:
        predictions = nli(inputs, batch_size=8, truncation=True, top_k=None)
    except Exception as exc:
        logger.warning("NLI contradiction check failed (%s); using polarity heuristic", exc)
        return True
    for pred in predictions:
        probability = _label_probability(pred, nli, "contradict")
        if probability is not None and probability >= float(CONTRADICTION_THRESHOLD):
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
    """Aggregate heuristic evidence-quality indicators (not a calibrated probability).

    confidence = w_f * faithfulness + w_c * (avg credibility / 5) + grade bonus,
    capped at RISKY_CONFIDENCE_CAP when the answer is flagged as at-risk, so strong
    sources can no longer mask poorly supported claims.
    """
    safe_chunks = [c for c in (verified_chunks or []) if isinstance(c, dict)]
    if not safe_chunks:
        return {
            "confidence": 0.0, "grade": "D", "faithfulness": 0.0, "avg_credibility": 0.0,
            "contradictions": [], "hallucination_risk": True, "supported_ratio": 0.0,
            "unsupported_claims": [], "citation_coverage": 0.0, "claims": [],
            "insufficient_evidence": False,
        }

    report = verify_answer(generated_answer, safe_chunks)
    faithfulness = report["faithfulness"]
    # Grade and credibility describe the evidence the answer actually RELIES ON (the sources it
    # cites). An irrelevant high-grade paper that happens to be retrieved must not lift them.
    text_chunks = [c for c in safe_chunks if str(c.get("text", "")).strip()]
    cited_idx = sorted({n for cl in report["claims"] for n in cl.get("cited", []) if 1 <= n <= len(text_chunks)})
    basis = [text_chunks[n - 1] for n in cited_idx] or safe_chunks
    grade_basis = "cited" if cited_idx else "retrieved"
    credibility_scores = [
        float(c.get("credibility_score", score_credibility(c))) for c in basis
    ]
    avg_credibility = sum(credibility_scores) / len(credibility_scores)
    best_grade = max(
        (str(c.get("evidence_grade", "D")) for c in basis),
        key=lambda g: GRADE_ORDER.get(g, 0),
    )
    contradictions = find_contradictions(safe_chunks)

    insufficient = report.get("insufficient_evidence", False)
    no_claims = not report["claims"] and not insufficient
    hallucination_risk = (
        no_claims
        or (not insufficient and report["supported_ratio"] < SUPPORTED_RATIO_MIN)
        or (not insufficient and faithfulness < float(FAITHFULNESS_THRESHOLD) * 0.75)  # mean-score safety net
        or (not insufficient and faithfulness < float(ABSTAIN_FAITHFULNESS_THRESHOLD))
        or bool(contradictions)
    )

    # Source prestige may only SCALE faithfulness; it can never replace it.
    cred_norm = min(1.0, max(0.0, avg_credibility / 5.0))
    quality = 0.85 + 0.15 * cred_norm
    confidence = faithfulness * quality
    if faithfulness >= float(FAITHFULNESS_THRESHOLD):
        confidence = min(1.0, confidence + GRADE_BONUS.get(best_grade, 0.0) * 0.5)
    else:
        confidence = min(confidence, faithfulness)
    if hallucination_risk:
        confidence = min(confidence, RISKY_CONFIDENCE_CAP)
    if insufficient:
        # The model told the user the evidence is not enough; confidence in an ANSWER is low.
        confidence = min(confidence, INSUFFICIENT_CONFIDENCE_CAP)
    confidence = round(min(1.0, max(0.0, confidence)), 4)

    return {
        "confidence": confidence,
        "grade": best_grade,
        "faithfulness": faithfulness,
        "avg_credibility": round(avg_credibility, 3),
        "contradictions": contradictions,
        "hallucination_risk": hallucination_risk,
        "supported_ratio": report["supported_ratio"],
        "unsupported_claims": report["unsupported_claims"],
        "citation_coverage": report["citation_coverage"],
        "claims": report["claims"],
        "insufficient_evidence": insufficient,
        "grade_basis": grade_basis,
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
