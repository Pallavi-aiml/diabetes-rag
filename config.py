import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
API_KEY = os.getenv("API_KEY", "")  # optional: if set, /query & /feedback require an X-API-Key header

# Paths
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
RAW_DATA_PATH = "./data/raw"
PROCESSED_DATA_PATH = "./data/processed"

# PubMed
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "yourname@email.com")
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Bias the knowledge base towards higher-level evidence instead of "whatever matched".
PUBMED_EVIDENCE_FILTER = (
    "(systematic review[pt] OR meta-analysis[pt] OR randomized controlled trial[pt] "
    "OR practice guideline[pt] OR guideline[pt])"
)
PUBMED_MIN_YEAR = int(os.getenv("PUBMED_MIN_YEAR", "2015"))

# Embedding
# Override with EMBEDDING_MODEL in .env to try a retrieval-trained model (requires a full re-embed).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
EMBEDDING_BATCH_SIZE = 32

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval
TOP_K_RETRIEVE = 15
TOP_K_RERANK = 5
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MIN_QUERY_RELEVANCE = 0.20
MIN_RERANK_SCORE = -2.0
# Extra ranking bonus for guideline chunks when the question asks for a recommendation /
# target / first-line choice ("what is the recommended ..."). 0 (default) disables. NOTE: values
# around 0.1 let irrelevant guideline chunks outrank relevant papers; keep it small if you use it.
# How many fused (dense+BM25) candidates the cross-encoder scores, and how many survive to the
# relevance filter. The old limits (20 / 15) cut real guideline chunks that BM25 ranked ~12th.
RERANK_POOL = int(os.getenv("RERANK_POOL", "40"))
RERANK_KEEP = int(os.getenv("RERANK_KEEP", "25"))
# Ranking penalty (per mismatched population, max 2) for chunks about a population the question
# does not mention: children/youth, pregnancy, older adults, type 1-only. 0 disables.
POPULATION_MISMATCH_PENALTY = float(os.getenv("POPULATION_MISMATCH_PENALTY", "0.15"))
GUIDELINE_INTENT_BONUS = float(os.getenv("GUIDELINE_INTENT_BONUS", "0.0"))

# Evidence verification
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
FAITHFULNESS_THRESHOLD = 0.6
ABSTAIN_FAITHFULNESS_THRESHOLD = 0.15
CONTRADICTION_THRESHOLD = 0.6
NLI_MAX_EVIDENCE_UNITS = 8
FAITHFULNESS_DEBUG = os.getenv("FAITHFULNESS_DEBUG", "").strip() in {"1", "true", "True"}
# Claim-level verification (v4): each answer sentence is checked against windows of
# 1..NLI_WINDOW_SENTENCES consecutive evidence sentences, preferring the cited source.
NLI_WINDOW_SENTENCES = int(os.getenv("NLI_WINDOW_SENTENCES", "3"))
CLAIM_SUPPORT_THRESHOLD = float(os.getenv("CLAIM_SUPPORT_THRESHOLD", "0.50"))   # claim counts as supported
SUPPORTED_RATIO_MIN = float(os.getenv("SUPPORTED_RATIO_MIN", "0.70"))           # below -> hallucination_risk
RISKY_CONFIDENCE_CAP = 0.50    # confidence never exceeds this when hallucination_risk is True
INSUFFICIENT_CONFIDENCE_CAP = 0.30

# LLM
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
LOCAL_LLM_MODEL = GROQ_MODEL
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# Chroma collection
COLLECTION_NAME = "diabetes_knowledge"

# Evidence scoring
SOURCE_SCORES = {
    "systematic_review": 5,
    "rct": 5,
    "clinical_guideline": 4,
    "cohort_study": 3,
    "case_report": 2,
    "expert_opinion": 1,
    "unknown": 0,
}

JOURNAL_SCORES = {
    "Diabetes Care": 5,
    "NEJM": 5,
    "The Lancet": 5,
    "JAMA": 5,
    "BMJ": 4,
    "Diabetologia": 4,
    "PubMed Abstract": 3,
    "Unknown": 2,
}

PUBMED_QUERIES = [
    "type 2 diabetes treatment management",
    "diabetes mellitus complications prevention",
    "insulin therapy type 1 type 2 diabetes",
    "metformin type 2 diabetes efficacy",
    "HbA1c glycemic control diabetes",
    "diabetic neuropathy management",
    "diabetic nephropathy treatment",
    "diabetic retinopathy screening",
    "diabetes diet lifestyle intervention",
    "SGLT2 inhibitors diabetes cardiovascular",
    "GLP-1 receptor agonists diabetes",
    "blood pressure control diabetes hypertension",
    "diabetes pregnancy gestational",
    "hypoglycemia diabetes management",
    "diabetes self management education",
]
