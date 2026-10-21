import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Paths
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
RAW_DATA_PATH = "./data/raw"
PROCESSED_DATA_PATH = "./data/processed"

# PubMed
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "yourname@email.com")
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Embedding
EMBEDDING_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
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

# Evidence verification
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
FAITHFULNESS_THRESHOLD = 0.6
ABSTAIN_FAITHFULNESS_THRESHOLD = 0.15
CONTRADICTION_THRESHOLD = 0.6
NLI_MAX_EVIDENCE_UNITS = 8
FAITHFULNESS_DEBUG = os.getenv("FAITHFULNESS_DEBUG", "").strip() in {"1", "true", "True"}

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
