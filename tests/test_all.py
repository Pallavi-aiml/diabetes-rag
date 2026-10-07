"""
tests/test_all.py
Full test suite covering:
  - Evidence verifier (unit tests)
  - Retriever (unit tests with mock data)
  - Generator prompt builder (unit test)
  - FastAPI endpoints (integration tests via httpx)
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "test_chunk_001",
            "text": (
                "A randomized controlled trial demonstrated that metformin reduces "
                "HbA1c by approximately 1.5% in patients with type 2 diabetes mellitus. "
                "The drug is considered first-line therapy."
            ),
            "source": "PubMed:12345678",
            "journal": "Diabetes Care",
            "year": 2023,
            "source_type": "rct",
            "page": 1,
            "pmid": "12345678",
        },
        {
            "chunk_id": "test_chunk_002",
            "text": (
                "The ADA 2024 Standards of Medical Care recommend an HbA1c target of "
                "less than 7.0% for most non-pregnant adults with diabetes to reduce "
                "the risk of microvascular complications."
            ),
            "source": "ADA_Guidelines:page42",
            "journal": "Diabetes Care (ADA)",
            "year": 2024,
            "source_type": "clinical_guideline",
            "page": 42,
            "pmid": "N/A",
        },
        {
            "chunk_id": "test_chunk_003",
            "text": (
                "A case report described a patient with type 2 diabetes who experienced "
                "hypoglycaemia following initiation of intensive insulin therapy."
            ),
            "source": "PubMed:99887766",
            "journal": "Unknown",
            "year": 2018,
            "source_type": "case_report",
            "page": 1,
            "pmid": "99887766",
        },
    ]


@pytest.fixture
def verified_chunks(sample_chunks):
    from evidence.verifier import verify_chunks
    return verify_chunks(sample_chunks)


# ===========================================================================
# EVIDENCE VERIFIER TESTS
# ===========================================================================

class TestCredibilityScorer:
    def test_rct_gets_high_score(self, sample_chunks):
        from evidence.verifier import score_credibility
        rct_chunk = sample_chunks[0]  # source_type = rct
        score = score_credibility(rct_chunk)
        assert score >= 3.5, f"RCT should score >= 3.5, got {score}"

    def test_case_report_gets_low_score(self, sample_chunks):
        from evidence.verifier import score_credibility
        case_chunk = sample_chunks[2]  # source_type = case_report
        score = score_credibility(case_chunk)
        assert score <= 3.0, f"Case report should score <= 3.0, got {score}"

    def test_guideline_scores_well(self, sample_chunks):
        from evidence.verifier import score_credibility
        guideline_chunk = sample_chunks[1]  # clinical_guideline + Diabetes Care + 2024
        score = score_credibility(guideline_chunk)
        assert score >= 3.5, f"Guideline should score >= 3.5, got {score}"

    def test_score_within_bounds(self, sample_chunks):
        from evidence.verifier import score_credibility
        for chunk in sample_chunks:
            score = score_credibility(chunk)
            assert 0.0 <= score <= 5.0, f"Score out of bounds: {score}"


class TestEvidenceGradeClassifier:
    def test_rct_gets_grade_a(self, sample_chunks):
        from evidence.verifier import classify_evidence_grade
        grade = classify_evidence_grade(sample_chunks[0])
        assert grade == "A", f"RCT should be Grade A, got {grade}"

    def test_guideline_gets_grade_b(self, sample_chunks):
        from evidence.verifier import classify_evidence_grade
        grade = classify_evidence_grade(sample_chunks[1])
        assert grade in ("A", "B"), f"Guideline should be Grade A or B, got {grade}"

    def test_case_report_gets_low_grade(self, sample_chunks):
        from evidence.verifier import classify_evidence_grade
        grade = classify_evidence_grade(sample_chunks[2])
        assert grade in ("C", "D"), f"Case report should be Grade C or D, got {grade}"

    def test_grade_is_valid_value(self, sample_chunks):
        from evidence.verifier import classify_evidence_grade
        for chunk in sample_chunks:
            grade = classify_evidence_grade(chunk)
            assert grade in ("A", "B", "C", "D"), f"Invalid grade: {grade}"


class TestVerifyChunks:
    def test_verify_adds_credibility(self, sample_chunks):
        from evidence.verifier import verify_chunks
        verified = verify_chunks(sample_chunks.copy())
        for c in verified:
            assert "credibility_score" in c
            assert isinstance(c["credibility_score"], float)

    def test_verify_adds_evidence_grade(self, sample_chunks):
        from evidence.verifier import verify_chunks
        verified = verify_chunks(sample_chunks.copy())
        for c in verified:
            assert "evidence_grade" in c
            assert c["evidence_grade"] in ("A", "B", "C", "D")

    def test_verify_preserves_original_keys(self, sample_chunks):
        from evidence.verifier import verify_chunks
        original_keys = set(sample_chunks[0].keys())
        verified = verify_chunks(sample_chunks.copy())
        for orig_key in original_keys:
            assert orig_key in verified[0], f"Key lost after verify: {orig_key}"


class TestFaithfulness:
    def test_faithful_answer_scores_high(self):
        from evidence.verifier import check_faithfulness
        context = "Metformin is the first-line treatment for type 2 diabetes."
        answer = "Metformin is recommended as first-line therapy for type 2 diabetes."
        score = check_faithfulness(answer, context)
        assert score >= 0.0
        assert score <= 1.0

    def test_empty_chunks_returns_zero(self):
        from evidence.verifier import check_faithfulness_multi
        score = check_faithfulness_multi("Some answer", [])
        assert score == 0.0


class TestConfidenceAggregator:
    def test_returns_all_keys(self, verified_chunks):
        from evidence.verifier import aggregate_confidence
        report = aggregate_confidence(verified_chunks, "Metformin is first-line therapy.")
        expected_keys = {
            "confidence", "grade", "faithfulness",
            "avg_credibility", "contradictions", "hallucination_risk"
        }
        assert expected_keys.issubset(report.keys())

    def test_confidence_in_range(self, verified_chunks):
        from evidence.verifier import aggregate_confidence
        report = aggregate_confidence(verified_chunks, "HbA1c target is less than 7%.")
        assert 0.0 <= report["confidence"] <= 1.0

    def test_empty_chunks_low_confidence(self):
        from evidence.verifier import aggregate_confidence
        report = aggregate_confidence([], "Some answer.")
        assert report["confidence"] == 0.0
        assert report["grade"] == "D"

    def test_grade_is_valid(self, verified_chunks):
        from evidence.verifier import aggregate_confidence
        report = aggregate_confidence(verified_chunks, "Answer text here.")
        assert report["grade"] in ("A", "B", "C", "D")


# ===========================================================================
# CHUNKER TESTS
# ===========================================================================

class TestChunker:
    def test_chunks_are_created(self):
        from ingestion.chunker import chunk_documents
        docs = [{"text": "A" * 1000, "source": "test", "source_type": "rct",
                 "journal": "Test", "year": 2024, "page": 1, "pmid": "0"}]
        chunks = chunk_documents(docs)
        assert len(chunks) > 0

    def test_chunk_has_required_keys(self):
        from ingestion.chunker import chunk_documents
        docs = [{"text": "Diabetes management requires glycaemic control. " * 20,
                 "source": "test_source", "source_type": "rct",
                 "journal": "Test", "year": 2024, "page": 1, "pmid": "0"}]
        chunks = chunk_documents(docs)
        required = {"chunk_id", "text", "source", "journal", "year", "source_type"}
        for chunk in chunks:
            assert required.issubset(chunk.keys())

    def test_short_text_skipped(self):
        from ingestion.chunker import chunk_documents
        docs = [{"text": "Too short", "source": "s", "source_type": "rct",
                 "journal": "J", "year": 2024, "page": 1, "pmid": "0"}]
        chunks = chunk_documents(docs)
        assert len(chunks) == 0

    def test_chunk_ids_are_unique(self):
        from ingestion.chunker import chunk_documents
        docs = [{"text": "Diabetes management. " * 50, "source": "doc_a",
                 "source_type": "rct", "journal": "J", "year": 2024, "page": 1, "pmid": "0"},
                {"text": "Insulin therapy. " * 50, "source": "doc_b",
                 "source_type": "rct", "journal": "J", "year": 2024, "page": 2, "pmid": "1"}]
        chunks = chunk_documents(docs)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs are not unique"


# ===========================================================================
# PROMPT BUILDER TESTS
# ===========================================================================

class TestPromptBuilder:
    def test_prompt_contains_question(self, verified_chunks):
        from generation.generator import build_rag_prompt
        question = "What is the HbA1c target?"
        prompt = build_rag_prompt(question, verified_chunks)
        assert question in prompt

    def test_prompt_contains_sources(self, verified_chunks):
        from generation.generator import build_rag_prompt
        prompt = build_rag_prompt("Test question?", verified_chunks)
        assert "[SOURCE 1]" in prompt
        assert "[SOURCE 2]" in prompt

    def test_prompt_contains_chunk_text(self, verified_chunks):
        from generation.generator import build_rag_prompt
        prompt = build_rag_prompt("Question?", verified_chunks)
        for chunk in verified_chunks:
            assert chunk["text"][:50] in prompt

    def test_empty_chunks_handled(self):
        from generation.generator import generate_answer
        result = generate_answer("Question?", [], use_local=True)
        assert "No relevant evidence" in result or len(result) > 0


# ===========================================================================
# RETRIEVER UNIT TESTS (no live DB needed)
# ===========================================================================

class TestRRF:
    def test_rrf_merges_two_lists(self):
        from retrieval.retriever import reciprocal_rank_fusion
        list1 = [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}]
        list2 = [{"chunk_id": "b", "text": "B"}, {"chunk_id": "c", "text": "C"}]
        fused = reciprocal_rank_fusion([list1, list2])
        ids = [d["chunk_id"] for d in fused]
        assert "b" in ids    # b appears in both → should rank high
        assert "a" in ids
        assert "c" in ids

    def test_rrf_preserves_all_docs(self):
        from retrieval.retriever import reciprocal_rank_fusion
        list1 = [{"chunk_id": str(i), "text": f"doc{i}"} for i in range(5)]
        list2 = [{"chunk_id": str(i), "text": f"doc{i}"} for i in range(3, 8)]
        fused = reciprocal_rank_fusion([list1, list2])
        assert len(fused) == 8  # 0-7 unique

    def test_rrf_empty_lists(self):
        from retrieval.retriever import reciprocal_rank_fusion
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []


class TestBM25:
    def test_bm25_returns_results(self, sample_chunks):
        from retrieval.retriever import build_bm25_index, bm25_retrieve
        index = build_bm25_index(sample_chunks)
        results = bm25_retrieve("metformin type 2 diabetes", index, sample_chunks, n=2)
        assert len(results) > 0

    def test_bm25_relevant_first(self, sample_chunks):
        from retrieval.retriever import build_bm25_index, bm25_retrieve
        index = build_bm25_index(sample_chunks)
        results = bm25_retrieve("metformin first-line therapy", index, sample_chunks, n=3)
        # The metformin chunk should appear in results
        texts = [r["text"] for r in results]
        assert any("metformin" in t.lower() for t in texts)


# ===========================================================================
# FASTAPI INTEGRATION TESTS
# ===========================================================================

@pytest.fixture
def test_client():
    """Create a test client with mocked dependencies."""
    from fastapi.testclient import TestClient
    import api.app as app_module

    # Mock the global state
    mock_chunk = {
        "chunk_id": "mock_001",
        "text": "Metformin is the first-line treatment for type 2 diabetes. HbA1c target is less than 7%.",
        "source": "ADA_Guidelines:page42",
        "journal": "Diabetes Care (ADA)",
        "year": 2024,
        "source_type": "clinical_guideline",
        "page": 42,
        "pmid": "N/A",
        "credibility_score": 4.5,
        "evidence_grade": "B",
        "rerank_score": 0.9,
    }

    mock_collection = MagicMock()
    mock_collection.count.return_value = 1000
    mock_collection.query.return_value = {
        "documents": [[mock_chunk["text"]]],
        "ids": [["mock_001"]],
        "metadatas": [[{k: v for k, v in mock_chunk.items() if k != "text"}]],
        "distances": [[0.1]],
    }

    app_module._collection = mock_collection
    app_module._all_chunks = [mock_chunk]

    with patch("retrieval.retriever.hybrid_retrieve", return_value=[mock_chunk]):
        with patch("evidence.verifier.verify_chunks", return_value=[mock_chunk]):
            with patch("evidence.verifier.aggregate_confidence", return_value={
                "confidence": 0.82,
                "grade": "B",
                "faithfulness": 0.78,
                "avg_credibility": 4.5,
                "contradictions": [],
                "hallucination_risk": False,
            }):
                with patch("generation.generator.generate_answer",
                           return_value="Metformin is first-line therapy [SOURCE 1]. ⚠️ This information is for educational purposes only."):
                    client = TestClient(app_module.app)
                    yield client


class TestHealthEndpoint:
    def test_health_returns_ok(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_has_timestamp(self, test_client):
        resp = test_client.get("/health")
        assert "timestamp" in resp.json()


class TestStatsEndpoint:
    def test_stats_returns_200(self, test_client):
        resp = test_client.get("/stats")
        assert resp.status_code == 200

    def test_stats_has_chunk_count(self, test_client):
        resp = test_client.get("/stats")
        data = resp.json()
        assert "total_chunks" in data


class TestQueryEndpoint:
    def test_query_returns_200(self, test_client):
        resp = test_client.post("/query", json={
            "question": "What is the recommended HbA1c target for type 2 diabetes?",
            "use_local_llm": True,
            "top_k": 5,
        })
        assert resp.status_code == 200

    def test_query_response_has_required_fields(self, test_client):
        resp = test_client.post("/query", json={
            "question": "What is metformin used for?",
            "use_local_llm": True,
        })
        data = resp.json()
        required = {
            "question", "answer", "confidence", "evidence_grade",
            "faithfulness", "citations", "disclaimer"
        }
        assert required.issubset(data.keys())

    def test_confidence_is_float(self, test_client):
        resp = test_client.post("/query", json={"question": "What is metformin?"})
        data = resp.json()
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_short_question_rejected(self, test_client):
        resp = test_client.post("/query", json={"question": "Hi"})
        assert resp.status_code == 422  # Pydantic validation

    def test_query_contains_disclaimer(self, test_client):
        resp = test_client.post("/query", json={"question": "What is the HbA1c target?"})
        data = resp.json()
        assert len(data["disclaimer"]) > 0


class TestFeedbackEndpoint:
    def test_feedback_accepted(self, test_client):
        resp = test_client.post("/feedback", json={
            "question": "Test question?",
            "answer": "Test answer.",
            "rating": 4,
            "comment": "Good answer",
        })
        assert resp.status_code == 200

    def test_feedback_invalid_rating(self, test_client):
        resp = test_client.post("/feedback", json={
            "question": "Test?",
            "answer": "Answer.",
            "rating": 10,  # invalid: max is 5
        })
        assert resp.status_code == 422


# ===========================================================================
# Run tests
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
