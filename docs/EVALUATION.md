# Evaluation

All numbers below come from `evaluation/ragas_eval.py` (50 questions: 40 answerable, 5 unsupported, 5 unanswerable) run
against the local API, plus two verifier checks (`scripts/perturbation_test.py`, `scripts/score_labeled_claims.py`).
They are **heuristic indicators, not clinical validation**.

## 1. End-to-end runs

| Run (UTC date_time) | API errors | Answered | Abstained | Correct refusals (of 10) | Mean faithfulness | Flagged at-risk | Mean confidence |
|---|---|---|---|---|---|---|---|
| 20261020_145651 (baseline) | 0 | 48 | 2 | 2 | 0.39 | 36/48 (75%) | 0.59 |
| 20261028_174040 | 6 | 28 | 16 | 9 | 0.70 | 15/28 (54%) | 0.64 |
| 20261029_081219 | 0 | 34 | 16 | 9 | 0.68 | 20/34 (59%) | 0.58 |
| 20261029_190327 | 0 | 37 | 13 | 9 | 0.68 | 22/37 (59%) | 0.61 |
| 20261030_083514 | 0 | 36 | 14 | 9 | 0.66 | 21/36 (58%) | 0.62 |
| 20261031_181537 | 0 | 38 | 12 | 9 | 0.86 | 6/38 (16%) | 0.83 |
| 20261031_184036 | 0 | 37 | 13 | 9 | 0.84 | 9/37 (24%) | 0.80 |
| 20261031_185639 | 0 | 38 | 12 | 9 | 0.83 | 8/38 (21%) | 0.78 |

What changed between runs is listed in `CHANGELOG.md` (hardening commit, retry/backoff, NLI-confirmed contradictions,
citation parsing, per-source evidence selection, claim normalisation, drug/number grounding, self-contained-sentence prompt).
The three runs on 31 Oct use the same code: the spread (flag rate 16-24%, faithfulness 0.83-0.86) is run-to-run
variation of the LLM. **Report means and ranges over several runs, not a single run.**
"Mean confidence" is a heuristic score (faithfulness scaled by source quality), not a probability.

## 2. Verifier checks

**Perturbation test** (57 claims labelled supported; each is corrupted and re-scored with the real NLI model):

| Corruption | Claims | Detected |
|---|---|---|
| Number changed | 14 | 100% |
| Verb negated | 43 | 95% |
| Drug / drug class swapped | 16 | 100%* |
| **All** | **73** | **97%** |

Correct (uncorrupted) claims accepted: 82%. *The drug-swap detection is true by construction: the swap targets are in the drug
lexicon used by the entity check. Synthetic corruptions are easier than real hallucinations, so treat 97% as an upper bound.

**Labelled claims** (59 claims from the 30 Oct run): the labels were produced by an AI assistant (Claude) from the cited
evidence and have **not** been independently verified by a clinician. Only 2 claims were unsupported, so the missed-hallucination
rate cannot be estimated. False alarms (supported claims flagged): 25% before the v4.3.2 claim normalisation, 16% after - measured on
the same claims the fix was designed on, so this is an in-sample figure. A fresh, human-checked sample is still needed.

## 3. Known limitations

- Three answerable questions were still refused in run 20261031_181537 (HbA1c monitoring frequency, target BMI, first-line
  neuropathy treatment): two failed the faithfulness gate, one was a retrieval miss.
- RAGAS metrics were not computed (`--skip-ragas`).
- Sentences that begin with a pronoun ("It is not ...") can be mis-scored by the NLI model; the answer prompt now asks for self-contained sentences.
- About 0.5% of ADA chunks contain words glued together by PDF extraction.
- The dense index and `chunks.json` were built at different times (different PubMed papers); rebuilding both is planned.
- The risk model is trained on the Pima dataset (768 women of Pima heritage, outcome defined by WHO criteria that include the
  2-hour glucose value); it uses one stratified hold-out split, no calibration and no external validation. Its inputs are
  2-hour OGTT glucose and 2-hour insulin, not fasting values. SHAP values explain the model's score, not causation, and are not
  used for retrieval.

## 4. Reproduce

```
uvicorn api.app:app --port 8000
python evaluation/ragas_eval.py --skip-ragas --delay 4
python scripts/eval_breakdown.py
python scripts/perturbation_test.py claims_v3.xlsx
```
