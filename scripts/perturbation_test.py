r"""
scripts/perturbation_test.py — can the verifier catch a claim that has been deliberately corrupted?

Human labels gave us only ~2 unsupported claims, so the verifier's ability to catch hallucinations cannot be
measured from them. This script manufactures unsupported claims from your SUPPORTED ones:
    number  : a number in the claim is changed          ("below 7%"        -> "below 10%")
    negate  : the claim's verb is negated               ("is recommended"  -> "is not recommended")
    drug    : a drug/class is swapped for another one   ("metformin"       -> "glimepiride")
and checks how many corrupted claims the verifier still accepts. Original claims are scored the same way as a control.

Usage (API not needed):
    python scripts\perturbation_test.py claims_v3.xlsx            # uses rows labeled 1
    python scripts\perturbation_test.py claims_v3.xlsx --threshold 0.5

Caveat: synthetic corruptions are usually EASIER to catch than natural hallucinations, so treat the detection rate
as an optimistic upper bound and report it as such.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DRUG_SWAPS = [
    (r"\bmetformin\b", "glimepiride"), (r"\bGLP-1 receptor agonists?\b", "DPP-4 inhibitors"),
    (r"\bSGLT2 inhibitors?\b", "thiazolidinediones"), (r"\bACE inhibitors?\b", "calcium channel blockers"),
    (r"\bstatins?\b", "fibrates"), (r"\binsulin\b", "sulfonylureas"), (r"\bpregabalin\b", "amitriptyline"),
]
NEG_VERBS = r"\b(is|are|should|can|may|must|does|do|was|were|has|have|will)\b"


_UNIT = r"(?:%|\s?mL(?:/min)?|\s?mg/dL|\s?mmol/L|\s?mmol/mol|\s?mmHg|\s?mg|\s?g\b|\s?kg|\s?years?|\s?months?|\s?weeks?|\s?days?|\s?minutes?|\s?min\b|\s?hours?|\s?units?|\s?IU)"
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?=" + _UNIT + r"|(?:/\d))|(?<=[<>\u2264\u2265])\s?(\d+(?:\.\d+)?)", re.IGNORECASE)


def perturb_number(claim):
    """Change a clinically meaningful number (with a unit, % or a </> threshold); never the '2' in 'type 2'."""
    for m in _NUMBER.finditer(claim):
        group = 1 if m.group(1) else 2
        before = claim[:m.start(group)].lower().rstrip()
        if before.endswith(("type", "stage", "source", "grade", "glp-", "sglt", "class")):
            continue
        value = float(m.group(group))
        new = value + 3 if value <= 10 else value * 2
        return claim[:m.start(group)] + f"{new:g}" + claim[m.end(group):]
    return None


def perturb_negate(claim):
    if re.search(r"\b(not|no|never|without|n't)\b", claim, re.IGNORECASE):
        return None
    m = re.search(NEG_VERBS, claim, re.IGNORECASE)
    if not m:
        return None
    return claim[:m.end()] + " not" + claim[m.end():]


def perturb_drug(claim):
    for pattern, repl in DRUG_SWAPS:
        if re.search(pattern, claim, re.IGNORECASE):
            return re.sub(pattern, repl, claim, count=1, flags=re.IGNORECASE)
    return None


PERTURBATIONS = {"number": perturb_number, "negate": perturb_negate, "drug": perturb_drug}


def read_rows(path):
    if str(path).lower().endswith(".xlsx"):
        from openpyxl import load_workbook
        it = load_workbook(path, data_only=True).active.iter_rows(values_only=True)
        header = [str(h) for h in next(it)]
        return [dict(zip(header, r)) for r in it]
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    from config import CLAIM_SUPPORT_THRESHOLD
    from evidence.verifier import verify_answer
    threshold = args.threshold if args.threshold is not None else float(CLAIM_SUPPORT_THRESHOLD)

    def score(claim, chunks):
        cites = " ".join(f"[SOURCE {i}]" for i in range(1, len(chunks) + 1))
        report = verify_answer(f"{claim.rstrip('.')} {cites}.", chunks)
        return report["claims"][0]["score"] if report["claims"] else None

    stats = {k: {"n": 0, "accepted": 0} for k in ["original", *PERTURBATIONS]}
    examples = []
    used = 0
    for r in read_rows(args.file):
        if str(r.get("label")).strip() not in ("1", "1.0"):
            continue
        segments = [re.sub(r"^\[SOURCE \d+\]\s*", "", s).strip() for s in str(r.get("evidence_shown") or "").split(" || ")]
        chunks = [{"text": s, "chunk_id": f"e{i}"} for i, s in enumerate(segments, 1) if s and not s.startswith("(no citation)")]
        claim = " ".join(str(r.get("claim") or "").split())
        if not chunks or not claim:
            continue
        used += 1
        base = score(claim, chunks)
        if base is None:
            continue
        stats["original"]["n"] += 1
        stats["original"]["accepted"] += int(base >= threshold)
        for name, fn in PERTURBATIONS.items():
            bad = fn(claim)
            if not bad or bad == claim:
                continue
            s = score(bad, chunks)
            if s is None:
                continue
            stats[name]["n"] += 1
            stats[name]["accepted"] += int(s >= threshold)
            if s >= threshold and len(examples) < 8:
                examples.append((name, s, base, claim, bad))

    print(f"Supported claims used: {used}   (threshold {threshold:.2f})\n")
    print(f"{'type':<10}{'claims':>7}{'accepted':>10}{'rate':>8}   meaning")
    o = stats["original"]
    print(f"{'original':<10}{o['n']:>7}{o['accepted']:>10}{o['accepted'] / max(1, o['n']):>8.0%}   control: correct claims accepted (= 1 - false-alarm rate)")
    caught_total = tot = 0
    for name in PERTURBATIONS:
        s = stats[name]
        rate = s["accepted"] / s["n"] if s["n"] else 0.0
        print(f"{name:<10}{s['n']:>7}{s['accepted']:>10}{rate:>8.0%}   corrupted claims WRONGLY accepted ({1 - rate:.0%} detected)")
        caught_total += s["n"] - s["accepted"]
        tot += s["n"]
    if tot:
        print(f"\nOverall detection of synthetic hallucinations: {caught_total}/{tot} = {caught_total / tot:.0%}")
    if examples:
        print("\nCorruptions the verifier still ACCEPTED (original score -> corrupted score):")
        for name, s, base, claim, bad in examples:
            print(f"  [{name}] {base:.2f} -> {s:.2f}\n     orig: {claim[:110]}\n     bad : {bad[:110]}")
    print("\nSynthetic corruptions are easier than real hallucinations: report this as an optimistic upper bound.")


if __name__ == "__main__":
    main()
