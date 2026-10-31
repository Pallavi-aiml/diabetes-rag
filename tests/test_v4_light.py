"""
Lightweight v4 tests — run in seconds with NO torch / chromadb / model downloads.

Heavy third-party modules are stubbed only when they are not installed, so the same
file also runs unchanged on a full local install. Covers the behaviour that changed in
v4: claim-level verification, citation checks, abstention gate, output cleaning,
API hardening, loader fixes.
"""
import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _stub(name, **attrs):
    try:
        importlib.import_module(name)
    except Exception:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod


# Stubs for heavy / optional deps (only used if the real package is missing).
_stub("sentence_transformers", CrossEncoder=MagicMock(), SentenceTransformer=MagicMock())
_stub("chromadb", Collection=object, PersistentClient=MagicMock())
_stub("faiss")
_stub("mlflow", set_experiment=lambda *a, **k: None)
_stub("pymupdf")
_stub("langchain_text_splitters", RecursiveCharacterTextSplitter=MagicMock())

# Use the real embedder when its dependencies exist (it lazy-loads the model, so import is cheap);
# otherwise (e.g. lightweight CI) stub it.
try:
    importlib.import_module("ingestion.embedder")
except Exception:
    _emb = types.ModuleType("ingestion.embedder")
    _emb.embed_query = lambda q: [[0.0]]
    _emb.get_chroma_collection = lambda: MagicMock()
    sys.modules["ingestion.embedder"] = _emb


GUIDELINE = {
    "chunk_id": "g1",
    "text": ("Most patients with type 2 diabetes should aim to keep their HbA1c below 7%. "
             "Treatment targets should be individualised for each patient. "
             "Metformin is the preferred initial pharmacological agent."),
    "source": "ADA:1", "journal": "Diabetes Care (ADA)", "year": 2024,
    "source_type": "clinical_guideline",
}
RCT = {
    "chunk_id": "r1",
    "text": "Regular physical activity reduces the risk of type 2 diabetes in adults with prediabetes.",
    "source": "PubMed:1", "journal": "BMJ", "year": 2022, "source_type": "rct",
}


@pytest.fixture
def lexical_only(monkeypatch):
    """Force the deterministic lexical fallback so tests never need an NLI model."""
    from evidence import verifier
    monkeypatch.setattr(verifier, "_get_nli_pipeline", lambda: None)
    return verifier


# ── verifier ────────────────────────────────────────────────────────────────

class TestClaimSplitting:
    def test_disclaimer_and_headings_are_not_claims(self, lexical_only):
        answer = ("## Targets\nMost patients aim for HbA1c below 7% [SOURCE 1].\n\n"
                  "⚠️ This information is for educational purposes only and does not "
                  "constitute medical advice. Always consult a qualified healthcare professional.")
        claims = lexical_only._split_answer_claims(answer)
        assert len(claims) == 1
        assert claims[0]["cited"] == [1]
        assert "[SOURCE" not in claims[0]["claim"]

    def test_think_blocks_ignored(self, lexical_only):
        answer = "<think>Let me reason about this carefully now.</think>Metformin is the preferred initial therapy for type 2 diabetes [SOURCE 1]."
        claims = lexical_only._split_answer_claims(answer)
        assert len(claims) == 1 and "reason" not in claims[0]["claim"]


class TestVerifyAnswer:
    def test_supported_claim_scores_high(self, lexical_only):
        r = lexical_only.verify_answer(
            "Most patients with type 2 diabetes should aim for an HbA1c below 7% [SOURCE 1].", [GUIDELINE])
        assert r["supported_ratio"] == 1.0 and r["unsupported_claims"] == []
        assert r["citation_coverage"] == 1.0

    def test_fabricated_claim_is_flagged(self, lexical_only):
        answer = ("Most patients with type 2 diabetes should aim for an HbA1c below 7% [SOURCE 1]. "
                  "Metformin causes severe weight gain in most patients [SOURCE 2].")
        r = lexical_only.verify_answer(answer, [GUIDELINE, RCT])
        assert 0 < r["supported_ratio"] < 1
        assert any("weight gain" in c for c in r["unsupported_claims"])

    def test_invalid_citation_does_not_count_as_valid(self, lexical_only):
        r = lexical_only.verify_answer("HbA1c should be kept below 7% for most patients [SOURCE 9].", [GUIDELINE])
        assert r["citation_coverage"] == 0.0           # SOURCE 9 does not exist
        assert r["claims"][0]["scope"] == "uncited"    # falls back to all evidence

    def test_empty_inputs(self, lexical_only):
        assert lexical_only.verify_answer("", [GUIDELINE])["faithfulness"] == 0.0
        assert lexical_only.verify_answer("Some claim about diabetes here.", [])["claims"] == []


class TestConfidence:
    def test_strong_sources_cannot_mask_bad_answer(self, lexical_only):
        chunks = lexical_only.verify_chunks([dict(GUIDELINE), dict(RCT)])
        bad = "Insulin glargine cures type 1 diabetes permanently within two weeks [SOURCE 1]."
        r = lexical_only.aggregate_confidence(chunks, bad)
        assert r["hallucination_risk"] is True
        assert r["confidence"] <= 0.5

    def test_good_answer_not_flagged(self, lexical_only):
        chunks = lexical_only.verify_chunks([dict(GUIDELINE)])
        good = "Most patients with type 2 diabetes should aim to keep their HbA1c below 7% [SOURCE 1]."
        r = lexical_only.aggregate_confidence(chunks, good)
        assert r["hallucination_risk"] is False and r["confidence"] > 0.5

    def test_no_chunks(self, lexical_only):
        r = lexical_only.aggregate_confidence([], "anything")
        assert r["confidence"] == 0.0 and r["hallucination_risk"] is True


def test_bogus_years_are_not_rewarded(lexical_only):
    base = dict(GUIDELINE)
    far_future = dict(GUIDELINE, year=2099)
    assert lexical_only.score_credibility(far_future) <= lexical_only.score_credibility(base)


# ── generator ───────────────────────────────────────────────────────────────

class TestGenerator:
    def test_clean_output_strips_think_and_fake_citations(self):
        from generation.generator import clean_model_output
        out = clean_model_output("<think>secret plan</think>HbA1c <7% [SOURCE 1] and X [SOURCE 7].", 3)
        assert "secret" not in out and "[SOURCE 1]" in out and "[SOURCE 7]" not in out

    def test_no_chunks_means_abstention(self):
        from generation.generator import generate_answer, NO_EVIDENCE_ANSWER
        assert generate_answer("What is HbA1c?", []).startswith(NO_EVIDENCE_ANSWER[:40])

    @pytest.mark.parametrize("q,expected", [
        ("I am having a seizure and feel very low blood sugar", True),
        ("What is DKA and how is it diagnosed?", False),
        ("My father collapsed and is not responding, he has diabetes", True),
    ])
    def test_emergency_detection(self, q, expected):
        from generation.generator import is_emergency_query
        assert is_emergency_query(q) is expected


# ── loader ──────────────────────────────────────────────────────────────────

class TestLoader:
    def test_unlabeled_paper_is_unknown_not_cohort(self):
        from ingestion.loader import detect_source_type
        assert detect_source_type([], "We describe the biochemical properties of an enzyme.") == "unknown"

    def test_rct_and_review_detection(self):
        from ingestion.loader import detect_source_type
        assert detect_source_type(["randomized controlled trial"], "x") == "rct"
        assert detect_source_type(["meta-analysis"], "x") == "systematic_review"

    @pytest.mark.parametrize("xml,year", [
        ("<A><Journal><JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue></Journal></A>", 2019),
        ("<A><Journal><JournalIssue><PubDate><MedlineDate>2018 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal></A>", 2018),
        ("<A><Journal><JournalIssue><PubDate></PubDate></JournalIssue></Journal></A>", 2000),
    ])
    def test_year_parsing(self, xml, year):
        import xml.etree.ElementTree as ET
        from ingestion.loader import _parse_year
        assert _parse_year(ET.fromstring(xml)) == year


# ── API ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient
    import api.app as app_module

    monkeypatch.setattr(app_module, "_collection", MagicMock())
    monkeypatch.setattr(app_module, "_all_chunks", [GUIDELINE])
    monkeypatch.setattr(app_module, "API_KEY", "")
    monkeypatch.setattr(app_module, "RATE_LIMIT_PER_MIN", 1000)
    app_module._hits.clear()
    return TestClient(app_module.app), app_module


def _prime(app_module, monkeypatch, retrieved, answer, report):
    monkeypatch.setattr(app_module, "hybrid_retrieve", lambda **kw: retrieved)
    monkeypatch.setattr(app_module, "verify_chunks", lambda c: c)
    monkeypatch.setattr(app_module, "find_contradictions", lambda c: [])
    monkeypatch.setattr(app_module, "generate_answer", lambda *a, **k: answer)
    monkeypatch.setattr(app_module, "aggregate_confidence", lambda *a, **k: report)
    monkeypatch.setattr(app_module, "get_llm_name", lambda: "test-llm")


def test_query_returns_claim_level_fields(client, monkeypatch):
    c, app_module = client
    chunk = dict(GUIDELINE, credibility_score=4.5, evidence_grade="B", query_relevance_score=0.66)
    _prime(app_module, monkeypatch, [chunk], "HbA1c below 7% [SOURCE 1].", {
        "confidence": 0.8, "grade": "B", "faithfulness": 0.9, "avg_credibility": 4.5,
        "contradictions": [], "hallucination_risk": False, "supported_ratio": 1.0,
        "unsupported_claims": [], "citation_coverage": 1.0, "claims": []})
    r = c.post("/query", json={"question": "What is the HbA1c target?"})
    assert r.status_code == 200
    body = r.json()
    assert body["abstained"] is False and body["supported_ratio"] == 1.0
    assert body["citations"][0]["text"].startswith("Most patients")


def test_insufficient_evidence_answer_becomes_flagged_abstention(client, monkeypatch):
    c, app_module = client
    chunk = dict(GUIDELINE, credibility_score=4.5, evidence_grade="B", query_relevance_score=0.66)
    _prime(app_module, monkeypatch, [chunk], "The evidence does not specify a target.", {
        "confidence": 0.0, "grade": "B", "faithfulness": 0.0, "avg_credibility": 4.5,
        "contradictions": [], "hallucination_risk": False, "supported_ratio": 0.0,
        "unsupported_claims": [], "citation_coverage": 0.0, "claims": [],
        "insufficient_evidence": True})
    body = c.post("/query", json={"question": "What is the HbA1c target?"}).json()
    assert body["abstained"] is True and body["insufficient_evidence"] is True


def test_no_evidence_is_a_clean_abstention(client, monkeypatch):
    c, app_module = client
    monkeypatch.setattr(app_module, "hybrid_retrieve", lambda **kw: [])
    r = c.post("/query", json={"question": "What is the capital of France today?"})
    assert r.status_code == 200
    assert r.json()["abstained"] is True and r.json()["citations"] == []


def test_api_key_enforced_when_configured(client, monkeypatch):
    c, app_module = client
    monkeypatch.setattr(app_module, "API_KEY", "secret")
    assert c.post("/query", json={"question": "What is HbA1c?"}).status_code == 401
    monkeypatch.setattr(app_module, "hybrid_retrieve", lambda **kw: [])
    ok = c.post("/query", json={"question": "What is HbA1c?"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200


def test_rate_limit(client, monkeypatch):
    c, app_module = client
    monkeypatch.setattr(app_module, "RATE_LIMIT_PER_MIN", 2)
    monkeypatch.setattr(app_module, "hybrid_retrieve", lambda **kw: [])
    codes = [c.post("/query", json={"question": "What is HbA1c?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_health_degraded_without_kb(monkeypatch):
    from fastapi.testclient import TestClient
    import api.app as app_module
    monkeypatch.setattr(app_module, "_collection", None)
    monkeypatch.setattr(app_module, "_all_chunks", [])
    assert TestClient(app_module.app).get("/health").json()["status"] == "degraded"


# ── v4.1: hedging answers, PDF cleaning, synonyms ───────────────────────────

HEDGING_ANSWER = (
    "The provided evidence does not explicitly state a single, universal recommended HbA1c target for all adults "
    "with type 2 diabetes. While [SOURCE 1] mentions that only half of individuals achieve the recommended target, "
    "it does not specify if this applies to all adults or specific subgroups. "
    "Therefore, the evidence is insufficient to definitively establish the standard recommended target."
)


class TestInsufficientEvidenceAnswers:
    def test_hedging_answer_is_not_a_hallucination(self, lexical_only):
        chunks = lexical_only.verify_chunks([dict(RCT)])
        r = lexical_only.aggregate_confidence(chunks, HEDGING_ANSWER)
        assert r["insufficient_evidence"] is True
        assert r["hallucination_risk"] is False
        assert r["confidence"] <= 0.30
        assert r["unsupported_claims"] == []

    def test_real_fabrication_still_flagged(self, lexical_only):
        chunks = lexical_only.verify_chunks([dict(GUIDELINE)])
        r = lexical_only.aggregate_confidence(chunks, "Insulin glargine cures type 1 diabetes permanently within two weeks [SOURCE 1].")
        assert r["insufficient_evidence"] is False and r["hallucination_risk"] is True

    def test_one_hedge_among_facts_is_not_insufficient(self, lexical_only):
        ans = ("Most patients with type 2 diabetes should aim for an HbA1c below 7% [SOURCE 1]. "
               "Metformin is the preferred initial pharmacological agent [SOURCE 1]. "
               "The sources do not specify a target for older adults.")
        r = lexical_only.verify_answer(ans, [GUIDELINE])
        assert r["insufficient_evidence"] is False and r["supported_ratio"] == 1.0


class TestPdfCleaning:
    def test_ligatures_hyphens_control_chars(self):
        from ingestion.text_cleaning import normalize_pdf_text
        out = normalize_pdf_text("glyce- mic \ufb02uctuated erythro-\ncyte (\x01120 days) A1C of \x017%")
        assert "glycemic" in out and "fluctuated" in out and "erythrocyte" in out
        assert "\x01" not in out and "A1C of 7%" in out

    def test_empty(self):
        from ingestion.text_cleaning import normalize_pdf_text
        assert normalize_pdf_text("") == "" and normalize_pdf_text(None) == ""


class TestSynonyms:
    def test_hba1c_matches_a1c(self):
        from retrieval.retriever import _tokens
        assert _tokens("HbA1c target") == _tokens("A1C target")
        assert _tokens("glycaemic control") == _tokens("glycemic control")
        assert _tokens("A1C goal") == _tokens("A1C target")

    def test_recommendation_intent(self):
        from retrieval.retriever import has_recommendation_intent
        assert has_recommendation_intent("What is the recommended HbA1c target?")
        assert not has_recommendation_intent("How does insulin work?")

    def test_guideline_bonus_is_off_by_default_and_works_when_enabled(self, monkeypatch):
        import retrieval.retriever as R
        def mk():
            return [{"source_type": "clinical_guideline", "rerank_score": 0.0, "query_relevance_score": 0.40, "chunk_id": "g"},
                    {"source_type": "cohort_study", "rerank_score": 0.0, "query_relevance_score": 0.50, "chunk_id": "c"}]
        q = "What is the recommended A1C target?"
        monkeypatch.setattr(R, "GUIDELINE_INTENT_BONUS", 0.0)
        assert R.boost_quality_sources(mk(), q)[0]["chunk_id"] == "c"      # more relevant paper wins
        monkeypatch.setattr(R, "GUIDELINE_INTENT_BONUS", 0.12)
        assert R.boost_quality_sources(mk(), q)[0]["chunk_id"] == "g"      # bonus lifts guideline
        assert R.boost_quality_sources(mk(), "How does insulin work?")[0]["chunk_id"] == "c"  # no intent -> no bonus


class TestPopulationPenalty:
    def _chunks(self):
        return [
            {"chunk_id": "peds", "source_type": "clinical_guideline", "rerank_score": 0.0, "query_relevance_score": 0.80,
             "text": "A lower A1C goal in youth with type 2 diabetes is recommended."},
            {"chunk_id": "adult", "source_type": "clinical_guideline", "rerank_score": 0.0, "query_relevance_score": 0.70,
             "text": "An A1C goal for many nonpregnant adults of <7% is appropriate."},
        ]

    def test_penalises_population_the_question_did_not_ask_about(self):
        from retrieval.retriever import boost_quality_sources
        ordered = boost_quality_sources(self._chunks(), "What is the recommended A1C target for adults with type 2 diabetes?")
        assert ordered[0]["chunk_id"] == "adult"          # "nonpregnant" must NOT count as pregnancy

    def test_no_penalty_when_question_names_the_population(self):
        from retrieval.retriever import boost_quality_sources
        ordered = boost_quality_sources(self._chunks(), "What is the A1C goal for youth with type 2 diabetes?")
        assert ordered[0]["chunk_id"] == "peds"

    def test_type1_only_chunk_penalised_for_type2_question(self):
        from retrieval.retriever import population_mismatch_penalty
        c = {"text": "Donislecel is indicated for adults with type 1 diabetes."}
        assert population_mismatch_penalty("A1C target in type 2 diabetes", c) < 0
        assert population_mismatch_penalty("A1C target in type 1 diabetes", c) == 0


class TestGradeFromCitedSources:
    def test_irrelevant_high_grade_source_does_not_lift_grade(self, lexical_only):
        irrelevant = {"chunk_id": "x", "text": "Gene variants influence GLP-1 levels in patients.", "source": "PubMed:1",
                      "journal": "Genome medicine", "year": 2026, "source_type": "systematic_review"}
        cited = dict(GUIDELINE)            # clinical_guideline -> grade B
        chunks = lexical_only.verify_chunks([irrelevant, cited])
        assert chunks[0]["evidence_grade"] == "A" and chunks[1]["evidence_grade"] == "B"
        ans = "Most patients with type 2 diabetes should aim for an HbA1c below 7% [SOURCE 2]."
        r = lexical_only.aggregate_confidence(chunks, ans)
        assert r["grade"] == "B" and r["grade_basis"] == "cited"

    def test_falls_back_to_all_sources_when_nothing_is_cited(self, lexical_only):
        chunks = lexical_only.verify_chunks([dict(GUIDELINE), dict(RCT)])
        r = lexical_only.aggregate_confidence(chunks, "Most patients with type 2 diabetes should aim for an HbA1c below 7%.")
        assert r["grade_basis"] == "retrieved"


class TestV425:
    def test_unclosed_think_block_never_reaches_user(self):
        from generation.generator import clean_model_output
        assert clean_model_output("<think>Let me work through the guideline text and decide", 3) == ""
        assert clean_model_output("<think>plan</think>Answer [SOURCE 1].", 3) == "Answer [SOURCE 1]."

    def test_retry_delay_backs_off_and_honours_retry_after(self):
        from generation.generator import _retry_delay
        assert _retry_delay(RuntimeError("x"), 0) == 1.5 and _retry_delay(RuntimeError("x"), 1) == 6.0
        class R:  headers = {"retry-after": "3"}
        class E(Exception): response = R()
        assert _retry_delay(E("rate"), 0) == 3.0

    def test_non_pregnant_is_not_a_pregnancy_mention(self):
        from retrieval.retriever import _populations, population_mismatch_penalty
        assert "pregnancy" not in _populations("target for most non-pregnant adults with type 2 diabetes")
        assert "pregnancy" not in _populations("An A1C goal for many nonpregnant adults of <7%")
        assert "pregnancy" in _populations("glycemic goals in pregnancy")
        chunk = {"text": "Glycemic goals for preexisting diabetes in pregnancy are lower."}
        assert population_mismatch_penalty("recommended target for most non-pregnant adults with type 2 diabetes", chunk) < 0


class TestContradictionConfirmation:
    A = "Metformin remains the first-line agent for the treatment of type 2 diabetes, although a second-line agent is not always needed."
    B = "Metformin is the first-line oral antihyperglycemic agent for the treatment of type 2 diabetes."

    def test_heuristic_fires_without_nli(self, lexical_only):
        assert lexical_only.detect_contradiction(self.A, self.B) is True     # old behaviour when no NLI model

    def test_agreeing_pair_rejected_when_nli_says_no_contradiction(self, monkeypatch):
        from evidence import verifier as V
        class FakeNLI:
            model = type("M", (), {"config": type("C", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}})()})()
            def __call__(self, inputs, **kw):
                return [[{"label": "contradiction", "score": 0.05}, {"label": "entailment", "score": 0.80}, {"label": "neutral", "score": 0.15}] for _ in inputs]
        monkeypatch.setattr(V, "_get_nli_pipeline", lambda: FakeNLI())
        assert V.detect_contradiction(self.A, self.B) is False

    def test_real_contradiction_kept_when_nli_agrees(self, monkeypatch):
        from evidence import verifier as V
        class FakeNLI:
            model = type("M", (), {"config": type("C", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}})()})()
            def __call__(self, inputs, **kw):
                return [[{"label": "contradiction", "score": 0.93}, {"label": "entailment", "score": 0.02}, {"label": "neutral", "score": 0.05}] for _ in inputs]
        monkeypatch.setattr(V, "_get_nli_pipeline", lambda: FakeNLI())
        assert V.detect_contradiction(self.A, self.B) is True


class TestCitationFormatsAndPremiseSelection:
    def test_multi_source_citations_are_parsed_and_stripped(self, lexical_only):
        for tag, expected in [("[SOURCE 1, SOURCE 3]", [1, 3]), ("[SOURCE 1, 3]", [1, 3]), ("[SOURCE 2 and 4]", [2, 4]), ("[SOURCE 2]", [2])]:
            claims = lexical_only._split_answer_claims(f"Treatment targets should be individualized for each patient {tag}.")
            assert claims[0]["cited"] == expected, tag
            assert "SOURCE" not in claims[0]["claim"], tag

    def test_generator_cleans_multi_citations(self):
        from generation.generator import clean_model_output
        assert clean_model_output("A [SOURCE 1, SOURCE 2].", 3) == "A [SOURCE 1, SOURCE 2]."
        assert clean_model_output("A [SOURCE 1, SOURCE 9] and B [SOURCE 8].", 3) == "A [SOURCE 1] and B ."

    def test_each_cited_source_contributes_premises(self, monkeypatch):
        """A claim citing two sources must be checked against the best window of EACH,
        not only the globally highest word-overlap windows (which can all come from one decoy chunk)."""
        from evidence import verifier as V
        seen = []
        class FakeNLI:
            model = type("M", (), {"config": type("C", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}})()})()
            def __call__(self, inputs, **kw):
                out = []
                for item in inputs:
                    seen.append(item["text"])
                    hit = "KEYFACT" in item["text"]
                    out.append([{"label": "entailment", "score": 0.97 if hit else 0.01},
                                {"label": "contradiction", "score": 0.01}, {"label": "neutral", "score": 0.02}])
                return out
        monkeypatch.setattr(V, "_get_nli_pipeline", lambda: FakeNLI())
        monkeypatch.setattr(V, "NLI_MAX_EVIDENCE_UNITS", 2)      # tiny cap to expose the old behaviour
        decoys = " ".join(f"Metformin is the first-line treatment for type 2 diabetes in decoy sentence number {i}." for i in range(12))
        chunks = [{"chunk_id": "a", "text": decoys},
                  {"chunk_id": "b", "text": "Guidelines recommend an agent KEYFACT as initial therapy for patients."}]
        r = V.verify_answer("Metformin is the first-line treatment for type 2 diabetes [SOURCE 1, SOURCE 2].", chunks)
        assert any("KEYFACT" in s for s in seen)             # source 2 was actually examined
        assert r["claims"][0]["score"] > 0.9


class TestLabelParsing:
    def test_excel_numbers_and_zero_are_not_blank(self):
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location("slc", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "score_labeled_claims.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        assert [mod.parse(x) for x in (0, 1, 0.0, 1.0, "0", "1", "yes", "no")] == [0, 1, 0, 1, 0, 1, 1, 0]
        assert [mod.parse(x) for x in (None, "", "  ", "?", 2)] == [None] * 5


class TestClaimNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("Additionally, IFG is defined as fasting plasma glucose levels from 100 to 125 mg/dL.", "IFG is defined as fasting plasma glucose levels from 100 to 125 mg/dL."),
        ("The WHO guideline notes that ideally, HbA1c should be measured twice a year.", "Ideally, HbA1c should be measured twice a year."),
        ("Based on the provided evidence, metformin is contraindicated with eGFR below 30.", "Metformin is contraindicated with eGFR below 30."),
        ("However, treatment targets need to be individualized.", "Treatment targets need to be individualized."),
        ("Metformin is first-line therapy.", "Metformin is first-line therapy."),
    ])
    def test_framing_is_removed(self, raw, expected):
        from evidence.verifier import _normalize_claim_for_nli
        assert _normalize_claim_for_nli(raw) == expected

    def test_advice_sentences_are_not_claims(self, lexical_only):
        claims = lexical_only._split_answer_claims("Metformin is first-line therapy for type 2 diabetes. For specific, individualized activity prescriptions, please consult a healthcare clinician.")
        assert [c["claim"] for c in claims] == ["Metformin is first-line therapy for type 2 diabetes."]

    def test_pronoun_claim_is_scored_with_previous_claim_as_context(self, monkeypatch):
        from evidence import verifier as V
        seen = []
        class FakeNLI:
            model = type("M", (), {"config": type("C", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}})()})()
            def __call__(self, inputs, **kw):
                out = []
                for item in inputs:
                    seen.append(item["text_pair"])
                    ok = "hba1c" in item["text_pair"].lower() and "target" in item["text_pair"].lower() and item["text_pair"].lower().startswith("most")
                    out.append([{"label": "entailment", "score": 0.95 if ok else 0.02}, {"label": "contradiction", "score": 0.01}, {"label": "neutral", "score": 0.03}])
                return out
        monkeypatch.setattr(V, "_get_nli_pipeline", lambda: FakeNLI())
        chunks = [{"chunk_id": "a", "text": "The recommended HbA1c target for most adults is below 7 percent and should be individualized for each person."}]
        ans = "Most adults should aim for a HbA1c target below 7 percent [SOURCE 1]. This applies to individuals without hypoglycemia [SOURCE 1]."
        r = V.verify_answer(ans, chunks)
        assert any(s.startswith("Most adults") and "This applies" in s for s in seen)       # combined hypothesis was tried
        assert r["claims"][1]["method"].endswith("+ctx") and r["claims"][1]["supported"]


class TestPerturbations:
    def _mod(self):
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location("pt", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "perturbation_test.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

    def test_corruptions_change_the_claim(self):
        m = self._mod()
        assert m.perturb_number("Aim for an HbA1c below 7% in adults.") == "Aim for an HbA1c below 10% in adults."
        assert m.perturb_negate("SGLT2 inhibitors are recommended in heart failure.") == "SGLT2 inhibitors are not recommended in heart failure."
        assert m.perturb_negate("Metformin should not be used with eGFR below 30.") is None     # already negated
        assert m.perturb_drug("Metformin is first-line therapy.") == "glimepiride is first-line therapy."
        assert m.perturb_number("Metformin is first-line therapy.") is None

    def test_number_perturbation_ignores_type_2_and_changes_the_threshold(self):
        m = self._mod()
        assert m.perturb_number("Most patients with type 2 diabetes should aim for an HbA1c below 7%.") == "Most patients with type 2 diabetes should aim for an HbA1c below 10%."
        assert m.perturb_number("Blood pressure should be <130/80 mmHg.") == "Blood pressure should be <260/80 mmHg."
        assert m.perturb_number("Metformin is contraindicated when eGFR is below 30 mL/min.") is not None
        assert m.perturb_number("People with type 2 diabetes use SGLT2 inhibitors.") is None


class TestEntityGrounding:
    def test_drug_classes_map_members_to_classes(self):
        from evidence.verifier import _drug_classes
        assert _drug_classes("Glimepiride and metformin were compared.") == {"sulfonylurea", "metformin"}
        assert _drug_classes("SGLT2 inhibitors, empagliflozin and GLP-1 receptor agonists") == {"sglt2", "glp1"}
        assert _drug_classes("ACE inhibitors or ARBs; calcium channel blockers") == {"acei", "arb", "ccb"}
        assert _drug_classes("Blood glucose targets") == set()

    def _fake(self, monkeypatch):
        from evidence import verifier as V
        class FakeNLI:   # accepts everything with 0.99, like the real model did for swapped drugs
            model = type("M", (), {"config": type("C", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}})()})()
            def __call__(self, inputs, **kw):
                return [[{"label": "entailment", "score": 0.99}, {"label": "contradiction", "score": 0.005}, {"label": "neutral", "score": 0.005}] for _ in inputs]
        monkeypatch.setattr(V, "_get_nli_pipeline", lambda: FakeNLI())
        return V

    def test_swapped_drug_is_caught_even_when_nli_accepts(self, monkeypatch):
        V = self._fake(monkeypatch)
        chunks = [{"chunk_id": "a", "text": "Metformin is the preferred initial pharmacological agent for type 2 diabetes in most adults."}]
        good = V.verify_answer("Metformin is the preferred initial agent for type 2 diabetes [SOURCE 1].", chunks)
        bad = V.verify_answer("Glimepiride is the preferred initial agent for type 2 diabetes [SOURCE 1].", chunks)
        assert good["claims"][0]["supported"] is True and good["claims"][0]["ungrounded_entities"] == []
        assert bad["claims"][0]["supported"] is False and bad["claims"][0]["ungrounded_entities"] == ["sulfonylurea"]
        assert bad["claims"][0]["method"].endswith("+entity")

    def test_class_member_in_evidence_grounds_the_class_name(self, monkeypatch):
        V = self._fake(monkeypatch)
        chunks = [{"chunk_id": "a", "text": "Empagliflozin reduced hospitalisation for heart failure in adults with type 2 diabetes."}]
        r = V.verify_answer("SGLT2 inhibitors reduce hospitalisation for heart failure in type 2 diabetes [SOURCE 1].", chunks)
        assert r["claims"][0]["supported"] is True

    def test_can_be_disabled(self, monkeypatch):
        V = self._fake(monkeypatch)
        monkeypatch.setattr(V, "ENTITY_GROUNDING", False)
        chunks = [{"chunk_id": "a", "text": "Metformin is the preferred initial pharmacological agent for type 2 diabetes in most adults."}]
        r = V.verify_answer("Glimepiride is the preferred initial agent for type 2 diabetes [SOURCE 1].", chunks)
        assert r["claims"][0]["supported"] is True


class TestNumberGroundingAndPrompt:
    def _fake(self, monkeypatch):
        from evidence import verifier as V
        class FakeNLI:
            model = type("M", (), {"config": type("C", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}})()})()
            def __call__(self, inputs, **kw):
                return [[{"label": "entailment", "score": 0.99}, {"label": "contradiction", "score": 0.005}, {"label": "neutral", "score": 0.005}] for _ in inputs]
        monkeypatch.setattr(V, "_get_nli_pipeline", lambda: FakeNLI())
        return V

    def test_clinical_numbers_ignore_type_2(self):
        from evidence.verifier import _clinical_numbers
        assert _clinical_numbers("Patients with type 2 diabetes should keep HbA1c below 7% and BP <130/80 mmHg.") == {7.0, 130.0, 80.0}
        assert _clinical_numbers("People with type 2 diabetes use metformin.") == set()

    def test_changed_number_is_caught_even_when_nli_accepts(self, monkeypatch):
        V = self._fake(monkeypatch)
        chunks = [{"chunk_id": "a", "text": "Insulin should be considered in patients with severe hyperglycemia, defined as an A1C greater than 10%."}]
        ok = V.verify_answer("Insulin should be considered in patients with an A1C greater than 10.0% [SOURCE 1].", chunks)
        bad = V.verify_answer("Insulin should be considered in patients with an A1C greater than 13% [SOURCE 1].", chunks)
        assert ok["claims"][0]["supported"] is True
        assert bad["claims"][0]["supported"] is False and bad["claims"][0]["ungrounded_numbers"] == [13.0]
        assert bad["claims"][0]["method"].endswith("+number")

    def test_number_check_can_be_disabled(self, monkeypatch):
        V = self._fake(monkeypatch)
        monkeypatch.setattr(V, "NUMBER_GROUNDING", False)
        chunks = [{"chunk_id": "a", "text": "Insulin should be considered with an A1C greater than 10%."}]
        assert V.verify_answer("Insulin should be considered with an A1C greater than 13% [SOURCE 1].", chunks)["claims"][0]["supported"] is True

    def test_prompt_asks_for_self_contained_sentences(self):
        from generation.generator import build_rag_prompt
        p = build_rag_prompt("What is the HbA1c target?", [{"text": "HbA1c below 7%.", "source": "ADA", "journal": "Diabetes Care", "year": 2024}])
        assert 'never start a sentence with "This", "It"' in p
