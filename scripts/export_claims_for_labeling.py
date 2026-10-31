r"""
scripts/export_claims_for_labeling.py — build a CSV you can label by hand to measure the verifier.

Why: the verifier says ~58% of answers are "at risk". Nobody knows how many of those are real
hallucinations and how many are false alarms until a human checks. This script takes the answers from
your newest evaluation run, re-scores every claim with the real NLI model, and writes a sample of claims
to a CSV. You fill in the `label` column; scripts/score_labeled_claims.py then measures the verifier.

Usage (API does NOT need to be running):
    python scripts\export_claims_for_labeling.py                 # newest raw_results_*.json, 60 claims
    python scripts\export_claims_for_labeling.py --max-claims 80 --out claims_to_label.csv

Labeling rule (be consistent!): read the claim and the evidence shown.
    label = 1  if the evidence shown supports the claim (even if worded differently)
    label = 0  if the evidence does NOT support it (missing, wrong, or goes beyond the evidence)
    leave blank if you cannot decide
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _norm_key(question, claim):
    """Match claims across exports even if citation brackets / spacing changed between verifier versions."""
    import re
    claim = re.sub(r"\[\s*SOURCES?[^\]]*\]", "", str(claim or ""), flags=re.IGNORECASE)
    return (" ".join(str(question or "").lower().split()), " ".join(claim.lower().split()).rstrip(" ."))


def _read_table(path):
    if str(path).lower().endswith(".xlsx"):
        from openpyxl import load_workbook
        it = load_workbook(path, data_only=True).active.iter_rows(values_only=True)
        header = [str(h) for h in next(it)]
        return [dict(zip(header, r)) for r in it]
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def newest_raw():
    files = sorted(glob.glob(os.path.join("evaluation", "raw_results_*.json")) +
                   glob.glob(os.path.join("evaluation", "results", "raw_results_*.json")), key=os.path.getmtime)
    if not files:
        sys.exit("No raw_results_*.json found under evaluation\\")
    return files[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?", default=None)
    ap.add_argument("--max-claims", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="claims_to_label.csv")
    ap.add_argument("--force", action="store_true", help="overwrite --out even if it exists")
    ap.add_argument("--only-from", default=None,
                    help="CSV/XLSX of already-labeled claims: re-score EXACTLY those claims (matched by question+claim) "
                         "with the current verifier and carry their labels/notes over")
    ap.add_argument("--labels-from", default=None,
                    help="CSV with labels you already filled in; they are copied (matched by question+claim) "
                         "so you can re-score the SAME claims after changing the verifier")
    args = ap.parse_args()

    if Path(args.out).exists() and not args.force:
        sys.exit(f"{args.out} already exists (it may contain your labels). Use --out other.csv, or --force to overwrite.")
    labels = {}
    label_file = args.only_from or args.labels_from
    if label_file:
        for r in _read_table(label_file):
            lab = r.get("label")
            lab = "" if lab is None else (str(int(lab)) if isinstance(lab, (int, float)) else str(lab))
            labels[_norm_key(r["question"], r["claim"])] = (lab, r.get("notes") or "")

    from evidence.verifier import verify_answer  # heavy import (loads NLI model on first use)

    path = args.raw or newest_raw()
    rows = json.load(open(path, encoding="utf-8"))
    print(f"Using {path}")

    pool = {"flagged": [], "clean": []}
    for q_idx, r in enumerate(rows, 1):
        if r.get("category") != "answerable" or r.get("http_status") != 200 or r.get("abstained"):
            continue
        contexts = [c for c in (r.get("contexts") or []) if c and c != "No context retrieved"]
        if not contexts or not r.get("answer"):
            continue
        chunks = [{"text": c, "chunk_id": f"c{i}"} for i, c in enumerate(contexts, 1)]
        report = verify_answer(r["answer"], chunks)
        bucket = "flagged" if r.get("hallucination_risk") else "clean"
        for d in report["claims"]:
            shown = [f"[SOURCE {n}] {' '.join(contexts[n - 1].split())[:700]}" for n in d["cited"] if 1 <= n <= len(contexts)]
            pool[bucket].append({
                "question": r["question"], "claim": d["claim"], "cited_sources": ",".join(map(str, d["cited"])),
                "nli_score": d["score"], "verifier_says_supported": int(d["supported"]),
                "answer_flagged": int(bucket == "flagged"), "method": d["method"],
                "evidence_shown": " || ".join(shown) if shown else f"(no citation) best match: {' '.join(d['best_evidence'].split())}",
            })
        print(f"  scored Q{q_idx}: {len(report['claims'])} claims")

    if args.only_from:
        everything = pool["flagged"] + pool["clean"]
        pick = [c for c in everything if _norm_key(c["question"], c["claim"]) in labels]
        missing = len(labels) - len(pick)
        print(f"Matched {len(pick)} of {len(labels)} claims from {args.only_from}"
              + (f" ({missing} not found: the verifier now splits or drops some sentences)" if missing else ""))
    else:
        rng = random.Random(args.seed)
        half = args.max_claims // 2
        pick = []
        for name in ("flagged", "clean"):
            items = pool[name][:]
            rng.shuffle(items)
            pick += items[:half]
        rest = [c for name in ("flagged", "clean") for c in pool[name] if c not in pick]
        rng.shuffle(rest)
        pick += rest[:max(0, args.max_claims - len(pick))]
        rng.shuffle(pick)

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        cols = ["id", "question", "claim", "cited_sources", "evidence_shown", "nli_score",
                "verifier_says_supported", "answer_flagged", "method", "label", "notes"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        carried = 0
        for i, c in enumerate(pick, 1):
            lab, note = labels.get(_norm_key(c["question"], c["claim"]), ("", ""))
            carried += 1 if lab != "" else 0
            w.writerow({"id": i, **c, "label": lab, "notes": note})
    if label_file:
        print(f"Carried over {carried} labels from {label_file}")
    print(f"\nWrote {len(pick)} claims to {args.out}. Fill the `label` column (1 supported / 0 not), then run:\n"
          f"    python scripts\\score_labeled_claims.py {args.out}")


if __name__ == "__main__":
    main()
