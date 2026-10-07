# Evidence-Verified RAG System for Diabetes Care

A clinical AI system that retrieves and verifies medical evidence before generating answers. Every response includes credibility scores, GRADE evidence levels, NLI-based faithfulness checks, and cited sources.

---

## Project Structure

```
diabetes_rag/
├── config.py                   # All config (models, paths, thresholds)
├── run_pipeline.py             # Master pipeline runner
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── raw/                    # Place ADA/WHO/ICMR PDFs here
│   ├── processed/              # chunked JSON files, FAISS index
│   └── chroma_db/              # ChromaDB persistent vector store
│
├── ingestion/
│   ├── loader.py               # PubMed API fetcher + PDF parser
│   ├── chunker.py              # Recursive text splitter
│   └── embedder.py             # PubMedBERT embeddings, ChromaDB, FAISS
│
├── retrieval/
│   └── retriever.py            # BM25 + dense + RRF + cross-encoder reranker
│
├── evidence/                   ← CORE INNOVATION
│   └── verifier.py             # Credibility score, GRADE, NLI faithfulness,
│                               # contradiction detection, confidence aggregator
│
├── generation/
│   └── generator.py            # Prompt builder, Ollama/GPT-4o, safety filter
│
├── api/
│   └── app.py                  # FastAPI: /query, /feedback, /health, /stats
│
├── frontend/
│   └── app.py                  # Streamlit UI with evidence badges
│
├── evaluation/
│   └── ragas_eval.py           # RAGAS evaluation + 50-question golden dataset
│
└── tests/
    └── test_all.py             # Full pytest suite (unit + integration)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo>
cd diabetes_rag
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env: add your OPENAI_API_KEY and PUBMED_EMAIL
```

### 3. Add guideline PDFs (optional but recommended)

Download these free PDFs and place in `./data/raw/`:
- ADA Standards of Care 2024 → name it `ada_2024.pdf`
- WHO Diabetes Guidelines → name it `who_diabetes.pdf`
- ICMR Guidelines → name it `icmr_diabetes.pdf`

### 4. Install local LLM (free)

```bash
# Install Ollama from https://ollama.com
ollama pull llama3
ollama serve          # keep this running in a separate terminal
```

### 5. Run the ingestion pipeline

```bash
# Full pipeline (fetch PubMed + chunk + embed) — takes 20-40 mins first time
python run_pipeline.py --all

# Or step by step:
python run_pipeline.py --ingest     # fetch PubMed abstracts
python run_pipeline.py --chunk      # split into chunks
python run_pipeline.py --embed      # build ChromaDB + FAISS
python run_pipeline.py --test-query # quick smoke test
```

### 6. Start the API

```bash
uvicorn api.app:app --reload --port 8000
# Swagger docs: http://localhost:8000/docs
```

### 7. Start the frontend

```bash
streamlit run frontend/app.py
# Opens at http://localhost:8501
```

---

## Models Used

| Purpose | Model | Size | Source |
|---|---|---|---|
| Embedding | PubMedBERT | 110M | HuggingFace |
| Reranker | ms-marco-MiniLM-L-6-v2 | 22M | HuggingFace |
| Hallucination / NLI | facebook/bart-large-mnli | 400M | HuggingFace |
| LLM (dev/free) | LLaMA-3 8B via Ollama | 8B | ollama.com |
| LLM (final demo) | GPT-4o | — | OpenAI API |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/query` | Main RAG query — returns answer + evidence report |
| POST | `/feedback` | Submit rating and comment |
| GET | `/health` | Health check |
| GET | `/stats` | Knowledge base statistics |
| GET | `/docs` | Swagger UI |

### Sample `/query` request

```json
{
  "question": "What is the recommended HbA1c target for type 2 diabetes?",
  "use_local_llm": true,
  "top_k": 5,
  "patient_context": "65-year-old with CKD stage 3"
}
```

### Sample `/query` response

```json
{
  "question": "What is the recommended HbA1c target for type 2 diabetes?",
  "answer": "The ADA 2024 guidelines recommend an HbA1c target of less than 7.0% for most non-pregnant adults [SOURCE 1]...",
  "confidence": 0.84,
  "evidence_grade": "A",
  "faithfulness": 0.79,
  "avg_credibility": 4.5,
  "hallucination_risk": false,
  "contradictions_found": false,
  "citations": [...],
  "disclaimer": "This system provides information only. Always consult a qualified clinician.",
  "llm_used": "llama3"
}
```

---

## Evaluation

```bash
# Run full evaluation against 50-question golden dataset
python evaluation/ragas_eval.py

# Fast mode (skip RAGAS, just internal metrics)
python evaluation/ragas_eval.py --skip-ragas --limit 20

# View results
ls evaluation/
```

RAGAS metrics reported:
- **Faithfulness** — is the answer supported by retrieved context?
- **Answer Relevancy** — does the answer address the question?
- **Context Precision** — are retrieved chunks relevant?
- **Context Recall** — were all relevant chunks found?

View experiment logs:
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

---

## Run Tests

```bash
pytest tests/test_all.py -v --tb=short
```

---

## Docker Deployment

```bash
docker-compose up --build
# API    → http://localhost:8000
# UI     → http://localhost:8501
# MLflow → http://localhost:5000
```

---

## Evidence Verification Layer (Core Innovation)

Each retrieved chunk is processed through 4 stages before the LLM sees it:

1. **Credibility Scorer** — assigns 0–5 score based on source type (RCT=5, guideline=4, case report=2), journal tier (Diabetes Care=5, NEJM=5), and publication recency
2. **GRADE Classifier** — assigns A/B/C/D evidence level using study design keywords and source metadata
3. **NLI Faithfulness Check** — uses `facebook/bart-large-mnli` to measure whether the generated answer is entailed by retrieved passages (detects hallucinations)
4. **Contradiction Detector** — pairwise NLI check across chunks flags conflicting evidence

Final confidence score = faithfulness×0.5 + normalised_credibility×0.4 + grade_bonus×0.1

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Embedding | PubMedBERT (sentence-transformers) |
| Vector store | ChromaDB + FAISS |
| Sparse retrieval | rank-bm25 |
| Reranker | cross-encoder/ms-marco-MiniLM |
| NLI model | facebook/bart-large-mnli |
| LLM | LLaMA-3 (Ollama) / GPT-4o |
| Orchestration | LangChain + LlamaIndex |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Evaluation | RAGAS + MLflow |
| Testing | pytest + httpx |

---

## Dataset Sources

| Dataset | Purpose |
|---|---|
| PubMed (via E-utilities API) | Research abstracts — knowledge base |
| ADA Standards of Care 2024 | Clinical guidelines — knowledge base |
| WHO / ICMR Guidelines | Clinical guidelines — knowledge base |
| MedMCQA (HuggingFace) | Evaluation benchmark |
| Golden dataset (50 Q&A) | Custom evaluation set |

---

## Acknowledgements

- American Diabetes Association (ADA) for open guidelines
- NCBI/PubMed for free API access
- HuggingFace for open-source model hosting
- RAGAS team for the evaluation framework
