# Evaluation — what the numbers actually say

Source: `evaluation/results/` (8 runs, 2026-10-06 → 2026-10-20). Latest complete run: `20261020_145651`
(50 questions: 40 answerable, 5 unsupported, 5 unanswerable; 0 API errors).

## Latest run (pre-v4 code)

| Metric | Value | How to read it |
|---|---|---|
| Answered with retrieved evidence | 48 / 50 | |
| Correct abstentions (unsupported + unanswerable) | **2 / 10 (20 %)** | 8 questions the KB cannot answer still got an answer |
| Mean internal faithfulness (answered) | **0.39** | very low |
| Hallucination flag rate (answered) | **75 %** (36/48) | flag fires on most answers |
| Contradiction flag rate | 10.4 % (5/48) | heuristic |
| Mean confidence (answered) | 0.59 | not consistent with faithfulness 0.39 |
| Avg citations per answer | 4.6 | |
| RAGAS metrics | **empty in all 4 JSON summaries** | RAGAS never produced scores |

Earlier runs on 19 Oct had 0.45–0.56 faithfulness and 2–14 API errors (rate limits / provider errors).

## Why the numbers look bad — and what v4 changes

1. **Faithfulness scored one evidence sentence at a time.** A correct answer that synthesises two sentences gets a low
   entailment score. v4 scores each claim against windows of 1–3 consecutive sentences and prefers the cited source.
2. **Flag = mean score < 0.6.** One strong and one weak sentence average out. v4 flags on the *share of unsupported claims*
   and reports which claims they are.
3. **Confidence was dominated by source prestige** (`0.4 × credibility`), so weakly supported answers from good journals
   still scored ~0.6. v4 caps confidence when the answer is flagged.
4. **No abstention logic in retrieval.** Weak matches were still passed to the LLM. v4 adds a relevance gate; tune it with
   `MIN_QUERY_RELEVANCE` / `MIN_RERANK_SCORE` (added in your hardening commit).
5. **RAGAS received 200-character previews** as `contexts`. v4 sends the full chunk text.
6. **Corpus metadata skew.** ~96 % of PubMed records were dated 2026 and ~80 % were typed `cohort_study` by default.
   Recency and source-type scores were therefore nearly constant. v4 fixes the query/parsing; **re-run ingestion** to rebuild.

## What I could NOT verify

The v4 changes were tested with a fast unit/integration suite (verifier logic via the lexical fallback, API, loader,
generator cleaning, relevance filter). I did **not** run the full pipeline: the NLI model, embeddings, reranker, Groq and
PubMed were not available in my environment. So:

- The new thresholds (`CLAIM_SUPPORT_THRESHOLD=0.50`, `SUPPORTED_RATIO_MIN=0.70`, `GUIDELINE_INTENT_BONUS=0.12`) are **starting points**.
- Expect the hallucination flag rate to change; re-run `evaluation/ragas_eval.py` and compare with the table above.
- Do not quote v4 numbers in your report until you have run it yourself.

## Suggested next evaluation steps

1. Re-ingest (`python run_pipeline.py --all`), start the API, run `python evaluation/ragas_eval.py`.
2. Tune `MIN_QUERY_RELEVANCE` in `config.py` using `scripts/debug_retrieval.py`, then re-run the eval.
3. Hand-label ~30 answers (supported / not supported) and compare against the NLI verdict → report precision/recall of the
   verifier itself. This is the strongest single addition for a "verification" project.
4. Report RAGAS faithfulness / answer relevancy / context precision / recall once they produce numbers; investigate
   RAGAS errors if they stay empty (usually an LLM-provider / API key issue for the RAGAS judge).
