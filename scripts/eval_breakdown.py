"""
scripts/eval_breakdown.py — turn the newest evaluation run into an actionable list.

Prints which questions (1) hit API errors, (2) were wrongly refused although answerable,
(3) should have been refused but were answered, (4) were answered but flagged as at-risk.

Usage:
    python scripts/eval_breakdown.py                       # newest raw_results_*.json
    python scripts/eval_breakdown.py path\to\raw_results_X.json
"""
import glob
import json
import os
import sys


def newest():
    files = sorted(glob.glob(os.path.join("evaluation", "raw_results_*.json")) +
                   glob.glob(os.path.join("evaluation", "results", "raw_results_*.json")),
                   key=os.path.getmtime)
    if not files:
        sys.exit("No raw_results_*.json found under evaluation\\")
    return files[-1]


def short(text, n=95):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def section(title, rows, fmt):
    print(f"\n── {title} ({len(rows)}) " + "─" * max(1, 55 - len(title)))
    for r in rows:
        print("  " + fmt(r))
    if not rows:
        print("  (none)")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else newest()
    rows = json.load(open(path, encoding="utf-8"))
    print(f"Using {path}  ({len(rows)} questions)")

    ok = [r for r in rows if r.get("http_status") == 200]
    errors = [r for r in rows if r.get("http_status") != 200]
    answerable = [r for r in ok if r.get("category") == "answerable"]
    should_refuse = [r for r in ok if r.get("category") in ("unsupported", "unanswerable")]
    refused = lambda r: bool(r.get("abstained"))

    section("API errors", errors,
            lambda r: f"[{r.get('category')}] {short(r.get('question'))} -> {r.get('http_status')} {short(r.get('error'), 60)}")
    def why(r):
        f = float(r.get("faithfulness_internal") or 0.0)
        c = int(r.get("citation_count") or 0)
        ans = " ".join(str(r.get("answer") or "").split()).lower()
        if r.get("contradictions_found") or "appear to conflict" in ans:
            return f"CONFLICT refusal (contradiction heuristic fired on {c} retrieved chunks)"
        if c == 0:
            return "nothing relevant retrieved (relevance filter) - retrieval problem"
        if r.get("insufficient_evidence"):
            return f"model itself said the {c} retrieved chunks don't answer it - retrieval problem"
        return f"answer generated from {c} chunks but verifier supported <15% of claims - verifier/answer problem"

    section("WRONGLY REFUSED: answerable but the system abstained", [r for r in answerable if refused(r)],
            lambda r: f"{short(r.get('question'), 70)}\n        -> {why(r)} | answer starts: {short(r.get('answer'), 70)}")
    section("MISSED: should refuse but answered", [r for r in should_refuse if not refused(r)],
            lambda r: f"[{r.get('category')}] {short(r.get('question'))}")
    flagged = [r for r in answerable if not refused(r) and r.get("hallucination_risk")]
    section("Answered but flagged at-risk (answerable only)", flagged,
            lambda r: f"faith={float(r.get('faithfulness_internal') or 0):.2f} conf={float(r.get('confidence') or 0):.2f}  {short(r.get('question'), 70)}")

    n_ans = len(answerable)
    wrong = sum(1 for r in answerable if refused(r))
    print(f"\nSummary: answerable reached={n_ans}/{sum(1 for r in rows if r.get('category')=='answerable')}  "
          f"wrongly refused={wrong}  flagged-at-risk={len(flagged)}  "
          f"correct refusals={sum(1 for r in should_refuse if refused(r))}/{len(should_refuse)}")


if __name__ == "__main__":
    main()
