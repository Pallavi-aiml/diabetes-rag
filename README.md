# Evidence-Verified RAG System for Diabetes Care

A retrieval-augmented question-answering system for diabetes information that **verifies its own answers**.
Every response comes with source citations, a heuristic evidence grade, and a **claim-level faithfulness check**
that tells you which sentences of the answer are *not* supported by the retrieved evidence. When the knowledge base
doesn't contain enough relevant evidence, the system **abstains** instead of guessing.

> ⚠️ Educational project. Not a medical device, not a diagnosis tool. All scores are heuristic indicators,
> not calibrated probabilities or clinical validation.

---

## Architecture

```mermaid
flowchart LR
    Q[Question + optional patient params] --> R[Risk model<br/>XGBoost + TreeSHAP]
    Q --> H
    R -->|enriched query| H
    subgraph Retrieval
      H[Hybrid retrieval<br/>PubMedBERT dense + BM25] --> F[Reciprocal Rank Fusion]
      F --> X[Cross-encoder rerank]
      X --> G[Relevance filter (abstains if nothing relevant)]
    end
    G -->|no evidence| A[Abstain]
    G --> V1[Credibility + GRADE-style grade]
    V1 --> L[LLM: Groq primary / OpenAI fallback<br/>grounded prompt, citations required]
    L --> C[Clean output<br/>strip think-traces, invalid citations]
    C --> V2[Claim-level NLI verification<br/>checks each claim against the source it cites]
    V2 --> O[Answer + confidence + unsupported claims + citations]
```

| Stage | What it does |
|---|---|
| Ingestion | PubMed E-utilities (evidence-filtered: SR / meta-analysis / RCT / guideline) + ADA / WHO / ICMR PDFs → 800-char chunks |
| Retrieval | Dense (ChromaDB) + BM25 → RRF → cross-encoder rerank → relevance filter (`MIN_QUERY_RELEVANCE`, `MIN_RERANK_SCORE`) → out-of-scope + conflict refusals |
| Evidence layer | Credibility score (0–5), GRADE-style grade (A–D), contradiction *heuristic* |
| Generation | Grounded prompt, evidence treated as untrusted data, emergency-query safety filter |
| Verification | **Per-claim NLI** over 1–3 sentence evidence windows, citation-aware, lexical fallback |
| Serving | FastAPI (`/query`, `/feedback`, `/health`, `/stats`), Streamlit UI, MLflow logging |

## What's new in v4

See [CHANGELOG.md](CHANGELOG.md) for the full list. Highlights:

- **Claim-level verification** — answers are split into claims; each is checked against the source it cites. The API returns `unsupported_claims`, `supported_ratio`, `citation_coverage`.
- **Confidence can't be bought with good sources** — it is capped when claims are unsupported.
- **Safer refusals** (from the hardening commit): out-of-scope questions, conflicting sources and answers with near-zero faithfulness are withheld with an explicit abstention.
- **Cleaner model output** — reasoning traces (`<think>`) and fabricated `[SOURCE n]` citations are stripped before scoring/serving.
- **Data fixes** — evidence-biased PubMed query, robust year parsing, no more silent "everything is a cohort study".
- **Ops fixes** — Groq key now reaches Docker, Dockerfile has a `CMD` + healthcheck, `requirements.txt` includes xgboost/scikit-learn/pandas, optional API-key auth + rate limit, no committed `mlflow.db`.

## Quick start

```bash
git clone https://github.com/Pallavi-aiml/diabetes-rag && cd diabetes-rag
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # add GROQ_API_KEY (and/or OPENAI_API_KEY) + PUBMED_EMAIL
```

Optional: put `ada_2024.pdf`, `who_diabetes.pdf`, `icmr_diabetes.pdf` in `data/raw/`.

```bash
python run_pipeline.py --all          # fetch PubMed + chunk + embed (20–40 min first time)
uvicorn api.app:app --port 8000       # API   → http://localhost:8000/docs
streamlit run frontend/app.py         # UI    → http://localhost:8501
```

> The repo ships `data/processed/chunks.json` built with the **old** ingestion settings. To benefit from the
> v4 data fixes run `python run_pipeline.py --all` again (it rebuilds chunks and vectors).

**Docker:** `docker compose up --build` → API :8000, UI :8501, MLflow :5000.

## Example

```bash
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"question": "What is the recommended HbA1c target for type 2 diabetes?", "top_k": 5}'
```

Key response fields: `answer`, `confidence`, `evidence_grade`, `faithfulness`, `supported_ratio`,
`unsupported_claims`, `citation_coverage`, `hallucination_risk`, `abstained`, `insufficient_evidence`, `citations[].text`.

## Configuration

Everything is in `config.py` and overridable via `.env` (see `.env.example`):
`GROQ_MODEL`, `OPENAI_MODEL`, `NLI_MODEL`, `CLAIM_SUPPORT_THRESHOLD`, `SUPPORTED_RATIO_MIN`,
`GUIDELINE_INTENT_BONUS`, `MIN_QUERY_RELEVANCE`, `API_KEY`, `RATE_LIMIT_PER_MIN`, `OLLAMA_MODEL`, …

## Evaluation

```bash
python evaluation/ragas_eval.py                  # 50 questions: 40 answerable, 5 unsupported, 5 unanswerable
python evaluation/ragas_eval.py --skip-ragas --limit 20
python scripts/debug_retrieval.py "your question" --expect "keyword"   # see dense / BM25 / final ranking
mlflow ui --port 5000
```

Results of earlier runs (kept in `evaluation/results/`) and an honest reading of them are in
[docs/EVALUATION.md](docs/EVALUATION.md). **Read that before quoting numbers in your report.**

## Tests

```bash
pytest tests/test_v4_light.py -v     # ~0.5 s, no GPU/model downloads (also runs in CI)
pytest tests -v                      # full suite
```

## Models

| Purpose | Model |
|---|---|
| Embedding | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` (PubMedBERT) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| NLI (faithfulness) | `cross-encoder/nli-deberta-v3-small` (configurable via `NLI_MODEL`) |
| LLM | Groq (primary), OpenAI (fallback) — model IDs set in `.env` |
| Risk model | XGBoost + TreeSHAP on the Pima Indians dataset (educational only) |

## Known limitations

- Heuristic grade/credibility are **not** a formal GRADE assessment; contradiction detection is a negation-polarity heuristic.
- The risk model is trained on the small, single-population Pima dataset — educational, not clinically valid.
- NLI models are weak on numbers and multi-sentence reasoning; thresholds should be re-tuned when you change `NLI_MODEL`.
- Knowledge base is PubMed abstracts + three guideline PDFs, not full texts.
