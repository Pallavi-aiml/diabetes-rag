# Changelog

## v4.3.5
- Perturbation test after v4.3.4: 70/73 (96%) synthetic hallucinations detected. Misses: two negations of pronoun-start
  sentences ("It is not a key feature ...": the NLI score went UP, 0.34 -> 0.99) and one changed number (10% -> 13%).
- **Number grounding**: numbers that carry a unit, % or < / > threshold must appear in the retrieved evidence, else the claim
  is capped at 0.2 (`+number`, `ungrounded_numbers`). `NUMBER_GROUNDING=0` disables it. On the labeled sample: 0 of 14
  supported claims with such numbers flagged; 13 of 14 changed numbers caught.
- **Prompt**: the model is told to write self-contained sentences (no "This/It/These/Additionally" openings, one cited
  fact per sentence). Pronoun openings were the main source of both false alarms and undetected negations.

## v4.3.4 - drug-entity grounding
- Perturbation test (real NLI, 57 supported claims): numbers changed -> 93% detected, negations -> 95%, but swapped drugs
  (metformin -> glimepiride, statin -> fibrate ...) only 50%.
- New check: every drug / drug class named in a claim must appear in the retrieved evidence (members map to classes,
  e.g. empagliflozin grounds "SGLT2 inhibitors"); otherwise the claim is capped at 0.2 (`method` ends with `+entity`,
  `ungrounded_entities` lists the classes). `ENTITY_GROUNDING=0` disables it.
- On the labeled sample it flags 0 of 57 supported claims. Its 16/16 on synthetic swaps is true by construction (the swap
  targets are in the lexicon); coverage on real hallucinations depends on how complete the drug lexicon is.

## v4.3.3
- `scripts/perturbation_test.py`: corrupts supported claims (number / negation / drug swap) and reports how many the
  verifier still accepts - a measurable (optimistic) detection rate that does not need human-labeled negatives.
- `score_labeled_claims.py --errors` lists the false alarms and missed hallucinations.
- On the same 59 labeled claims the false-alarm rate fell from 25% to 16% after v4.3.2 (in-sample: the fix was
  designed after looking at these claims, so validate on a fresh sample).

## v4.3.2 - from labeling 59 claims (labels by Claude, to be spot-checked by you)
- 14/57 supported claims were flagged by the verifier (25% false alarms). 11 of them began with a connective
  ("Additionally, ..."), a pronoun ("This/It/These ..."), or an attribution ("The WHO guideline notes that ...").
- Claims are now normalised before NLI (framing removed), and a pronoun-start claim that scores low is re-scored together
  with the previous claim (best score kept; method gets a `+ctx` suffix).
- Advice sentences such as "please consult a healthcare clinician" are no longer treated as factual claims.
- `score_labeled_claims.py` stops claiming "the NLI model is too weak" when the sample has < 10 unsupported claims.

## v4.3.1 - from the per-claim NLI log
- **Multi-source citations** such as `[SOURCE 1, SOURCE 3]`, `[SOURCE 1, 3]`, `[SOURCE 2 and 4]` were not parsed: the
  bracket text stayed inside the claim and the claim was scored as uncited. Fixed in the verifier and the generator's
  citation cleaner.
- **Premise selection is now per cited source** (best 4 windows of each cited source, up to 24 in total) instead of the
  global top `NLI_MAX_EVIDENCE_UNITS` (8) by word overlap. A claim citing five sources could previously be judged on
  eight windows that all came from one decoy chunk (e.g. "metformin is first-line" scored 0.18).
- `export_claims_for_labeling.py` no longer overwrites an existing CSV and can carry labels over (`--labels-from`).

## v4.2.5 - found by the 28 Oct evaluation
- LLM calls retry up to 3 times with backoff (honouring Retry-After); 6/50 questions failed with 503 after a single
  0.8 s retry.
- An unclosed `<think>` block (model ran out of tokens) is discarded instead of being shown as the answer.
- "non-pregnant" no longer counts as a pregnancy mention in the population penalty (the eval's first question uses it,
  so pregnancy chunks escaped the penalty).
- `ragas_eval.py --delay N` pauses between questions; `scripts/eval_breakdown.py` shows why each answerable question
  was refused.

## v4.2.4
- Evidence grade and average credibility now come from the sources the answer CITES (new field `grade_basis`),
  not from every retrieved chunk. An irrelevant Grade-A paper no longer lifts the grade of an answer that only
  relies on Grade-B guideline chunks. Falls back to all retrieved chunks if the answer cites nothing.

## v4.2.3 - ranking fixes found with scripts/debug_retrieval.py
- **Wider reranker pool** (`RERANK_POOL=40`, `RERANK_KEEP=25`): the ADA recommendation chunk (6.5a, "A1C goal for many
  nonpregnant adults of <7%") was BM25 rank 12 but was cut before reranking because only 20 fused candidates were scored.
- **Population-mismatch penalty** (`POPULATION_MISMATCH_PENALTY=0.15`): chunks about youth, pregnancy, older adults or
  type 1 only rank lower when the question does not mention that population.
- `GUIDELINE_INTENT_BONUS` now defaults to 0 (0.12 let irrelevant guideline chunks take 4 of 5 slots).
- Added goal/target synonym; debug script now shows every reranker candidate and why it was kept or cut.

## v4.2.0 - merge with the local hardening commit (`ba0c5f6`)
Your commit already provided: relevance filter + abstention, out-of-scope and conflict refusals, Ollama support,
prestige-can't-rescue-faithfulness confidence, BM25 cache fingerprint, lazy embedder/reranker, patient-param handling,
`unknown` default source type and the 800-char chunking. Those are preserved.
v4/v4.1 additions layered on top are listed below. One fix to your code: `filter_relevant` now drops chunks that score
*exactly* at `MIN_QUERY_RELEVANCE` (your own test `test_low_lexical_overlap_is_dropped` failed on that equality edge case).

## v4.1.0 (from first real-world run)
- **Insufficient-evidence answers**: when the model itself says the sources don't answer the question, the verifier no longer scores those hedging sentences as unsupported claims. New field `insufficient_evidence`; no false hallucination warning; confidence capped at 0.30; eval counts it as an abstention.
- **Retrieval**: clinical synonym normalisation (HbA1c = A1C, glycaemic = glycemic, …) for BM25 and lexical scoring; small configurable ranking bonus for guideline chunks on "recommended / target / first-line" questions (`GUIDELINE_INTENT_BONUS`).
- **PDF text cleaning** (`ingestion/text_cleaning.py`): ligatures, broken hyphenation, control characters. `scripts/clean_existing_chunks.py` applies it to the existing knowledge base and re-embeds only changed chunks.
- `scripts/debug_retrieval.py`: prints dense / BM25 / final results so you can see whether a miss is a recall or a ranking problem.

## v4.0.0
**Verification**
- `evidence/verifier.py`: claim-level NLI (`verify_answer`) over 1–3-sentence evidence windows; checks each claim against the `[SOURCE n]` it cites; wrong citations are penalised; lexical fallback kept; debug `print`s → `logging`.
- Hallucination flag now uses `supported_ratio` (share of supported claims) instead of the mean score alone; confidence is capped when the flag is raised.
- Implausible publication years no longer inflate credibility.
- NLI model is configured in one place (`config.NLI_MODEL`); README/config previously said BART-MNLI while the code ran DeBERTa-v3-small.

**Retrieval**
- (Abstention gate, lazy reranker and blocklist removal: already provided by the hardening commit `ba0c5f6`; not duplicated.)

**Generation**
- Strip `<think>…</think>` and out-of-range `[SOURCE n]` from model output.
- Ollama / `use_local_llm` support from the hardening commit is kept.

**API / UI**
- New response fields: `abstained`, `supported_ratio`, `citation_coverage`, `unsupported_claims`, `insufficient_evidence` (citations keep the hardening commit's `text` field).
- Optional `X-API-Key` auth, per-client rate limit, pydantic v2 validators, lifespan startup.
- Streamlit shows unsupported claims and abstention; `API_URL`/`API_KEY` from env (was hard-coded to localhost, breaking Docker).

**Data**
- PubMed query biased to SR/meta-analysis/RCT/guidelines, `sort=relevance`, min-year window.
- Robust year parsing (`MedlineDate`, `ArticleDate`); unlabeled papers are `unknown` (were silently `cohort_study`, ~80% of PubMed chunks).

**Evaluation**
- RAGAS contexts use the full chunk text (were 200-char previews, which likely contributed to the empty RAGAS results); records `supported_ratio`, `insufficient_evidence`.

**Ops**
- Docker: Groq key + `.env` passed through, `CMD`, healthcheck, pinned MLflow image, volumes.
- `requirements.txt` adds xgboost, scikit-learn, pandas, numpy, langchain-text-splitters; CI workflow + `requirements-ci.txt`.
- `mlflow.db` and the old `feedback_log.json` removed from version control; `.env.example` added.
