"""
generation/generator.py

LLM generation module — updated to use Groq API (primary, FREE).
Falls back to OpenAI GPT-4o if GROQ_API_KEY is not set.

Groq advantages for this project:
  - Completely free (14,400 requests/day)
  - Runs LLaMA-3.3 70B — strong medical text quality
  - 1-2 second response time (faster than OpenAI free tier)
  - Same API format as OpenAI — minimal code change

Patent Module 110 — Grounded Generation Module
Patent Module 114 — Safety Module
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))
from config import LLM_TEMPERATURE, LLM_MAX_TOKENS

# ── Model configuration ────────────────────────────────────────────────────────

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Groq model — LLaMA-3.3 70B is excellent for medical text
#GROQ_MODEL   = "llama-3.3-70b-versatile"
#GROQ_MODEL = "llama3-70b-8192"
#GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MODEL = "qwen/qwen3.8-27b"
OPENAI_MODEL = "gpt-4o"

# ── Safety keywords for emergency detection ────────────────────────────────────
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "unconscious",
    "seizure", "severe hypoglycemia", "diabetic coma",
    "ketoacidosis", "dka", "hyperglycemic crisis",
    "difficulty breathing", "loss of consciousness",
    "not responding", "collapsed", "emergency",
]

DISCLAIMER = (
    "\n\n⚠️ This information is for educational purposes only and does not "
    "constitute medical advice. Always consult a qualified healthcare "
    "professional before making any clinical decisions."
)


def is_emergency_query(query: str) -> bool:
    q_lower = query.lower()
    return any(kw in q_lower for kw in EMERGENCY_KEYWORDS)


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_rag_prompt(query: str, verified_chunks: list) -> str:
    """
    Build a structured, grounded RAG prompt with verified evidence blocks.
    Each block includes source metadata so the LLM can cite correctly.
    """
    context_blocks = ""
    for i, chunk in enumerate(verified_chunks, start=1):
        context_blocks += (
            f"\n[SOURCE {i}]\n"
            f"Journal     : {chunk.get('journal', 'Unknown')}\n"
            f"Year        : {chunk.get('year', 'N/A')}\n"
            f"Source type : {chunk.get('source_type', 'unknown')}\n"
            f"Evidence    : Grade {chunk.get('evidence_grade', 'N/A')} "
            f"| Credibility {chunk.get('credibility_score', 0):.1f}/5\n"
            f"Content     :\n{chunk['text']}\n"
            f"{'─' * 60}\n"
        )

    prompt = f"""You are a clinical information assistant specialising in diabetes care.

STRICT RULES:
1. Answer ONLY using the evidence provided below — never add outside knowledge.
2. Cite every factual claim as [SOURCE N].
3. If sources do not contain sufficient information, say:
   "The provided evidence does not clearly address this question."
4. Never speculate, invent statistics, or add information not in the sources.
5. Keep the answer concise, structured, and clinically accurate.

{'=' * 60}
EVIDENCE CONTEXT
{'=' * 60}
{context_blocks}
{'=' * 60}

CLINICAL QUESTION:
{query}

ANSWER (cite [SOURCE N] for every claim):"""

    return prompt


# ── Groq LLM (Primary — FREE) ──────────────────────────────────────────────────

def generate_with_groq(prompt: str) -> str:
    """
    Call Groq API with LLaMA-3.3 70B.
    FREE — 14,400 requests per day. 1-2 second response time.
    Sign up at console.groq.com
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found in .env file.\n"
            "Get your free key at: console.groq.com\n"
            "Then add to .env: GROQ_API_KEY=gsk_xxxxxxxx"
        )
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical assistant for diabetes care. "
                        "Answer strictly from provided evidence. "
                        "Never hallucinate or add outside information."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        raise ImportError(
            "groq package not installed. Run: pip install groq"
        )
    except Exception as e:
        raise RuntimeError(f"Groq API error: {e}")


# ── OpenAI (Fallback — Paid) ───────────────────────────────────────────────────

def generate_with_openai(prompt: str) -> str:
    """
    Call OpenAI GPT-4o API.
    Fallback option if Groq is not configured.
    Requires billing at platform.openai.com.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not found in .env file.\n"
            "Get your key at: platform.openai.com/api-keys"
        )
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical assistant for diabetes care. "
                        "Answer strictly from provided evidence. "
                        "Never hallucinate."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")


# ── Safety filter ──────────────────────────────────────────────────────────────

def apply_safety_filter(query: str, answer: str) -> str:
    """
    Patent Module 114 — Safety Module.
    Detect emergency keywords and prepend emergency alert.
    Append clinical disclaimer to every response.
    """
    if is_emergency_query(query):
        emergency_msg = (
            "🚨 EMERGENCY ALERT: This query describes symptoms that may "
            "require immediate medical attention. Please call emergency "
            "services (112 / 108 in India) or go to the nearest emergency "
            "department immediately.\n\n"
        )
        return emergency_msg + answer + DISCLAIMER

    return answer + DISCLAIMER


# ── Auto-detect which LLM to use ──────────────────────────────────────────────

def detect_llm() -> str:
    """
    Returns 'groq' if GROQ_API_KEY is set, else 'openai'.
    Groq is preferred — free and fast.
    """
    if GROQ_API_KEY:
        return "groq"
    elif OPENAI_API_KEY:
        return "openai"
    else:
        raise ValueError(
            "No LLM API key found in .env file.\n"
            "Add one of:\n"
            "  GROQ_API_KEY=gsk_xxxxx   (free — console.groq.com)\n"
            "  OPENAI_API_KEY=sk-xxxxx  (paid — platform.openai.com)"
        )


# ── Main generation pipeline ───────────────────────────────────────────────────

def generate_answer(
    query: str,
    verified_chunks: list,
    use_local: bool = False,   # ignored — kept for API compatibility
    force_llm: str = None,     # "groq" or "openai" to override auto-detect
) -> str:
    """
    Full generation pipeline:
      1. Build grounded RAG prompt with verified evidence
      2. Call LLM (Groq primary, OpenAI fallback)
      3. Apply safety filter and disclaimer

    Args:
        query:           user clinical question
        verified_chunks: evidence-verified retrieved chunks
        use_local:       ignored (kept for backward compatibility)
        force_llm:       "groq" or "openai" to override auto-detection

    Returns:
        Final answer string with citations and disclaimer.
    """
    if not verified_chunks:
        return (
            "No relevant evidence was found in the knowledge base for this query. "
            "Please consult a qualified clinician." + DISCLAIMER
        )

    # Emergency check before any LLM call
    if is_emergency_query(query):
        emergency_response = (
            "🚨 EMERGENCY ALERT: This query describes symptoms that may require "
            "immediate medical attention. Please call emergency services "
            "(112 / 108 in India) or go to the nearest emergency department "
            "immediately. Do not rely on this system in an emergency."
        )
        return emergency_response + DISCLAIMER

    # Build prompt
    prompt = build_rag_prompt(query, verified_chunks)

    # Determine which LLM to call
    llm = force_llm or detect_llm()

    if llm == "groq":
        raw_answer = generate_with_groq(prompt)
        llm_used   = GROQ_MODEL
    else:
        raw_answer = generate_with_openai(prompt)
        llm_used   = OPENAI_MODEL

    # Apply safety filter + disclaimer
    final_answer = apply_safety_filter(query, raw_answer)
    return final_answer


def get_llm_name() -> str:
    """Returns the name of the currently active LLM for display in UI."""
    if GROQ_API_KEY:
        return f"Groq — {GROQ_MODEL}"
    elif OPENAI_API_KEY:
        return f"OpenAI — {OPENAI_MODEL}"
    return "No LLM configured"


# ── Entry point (quick test) ───────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Active LLM: {get_llm_name()}")
    print(f"Groq key set  : {'Yes' if GROQ_API_KEY else 'No'}")
    print(f"OpenAI key set: {'Yes' if OPENAI_API_KEY else 'No'}")

    sample_chunks = [
        {
            "chunk_id": "test_001",
            "text": (
                "The ADA 2024 Standards of Medical Care recommend an HbA1c "
                "target of less than 7.0% (53 mmol/mol) for most non-pregnant "
                "adults with type 2 diabetes to reduce microvascular complications."
            ),
            "source": "ADA_Guidelines:page42",
            "journal": "Diabetes Care (ADA)",
            "year": 2024,
            "source_type": "clinical_guideline",
            "evidence_grade": "A",
            "credibility_score": 4.8,
        },
        {
            "chunk_id": "test_002",
            "text": (
                "Metformin remains the preferred initial pharmacological agent "
                "for type 2 diabetes management due to its efficacy, safety, "
                "and low cost. It is contraindicated in patients with eGFR < 30."
            ),
            "source": "PubMed:34567890",
            "journal": "Diabetes Care",
            "year": 2023,
            "source_type": "rct",
            "evidence_grade": "A",
            "credibility_score": 4.5,
        },
    ]

    query = "What is the HbA1c target for type 2 diabetes and what is first-line medication?"
    print(f"\nQuery: {query}\n")
    print("Generating answer...\n")

    answer = generate_answer(query, sample_chunks)
    print(answer)
