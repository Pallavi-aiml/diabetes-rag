r"""
scripts/score_labeled_claims.py — how good is the verifier, according to YOUR labels?

Reads the CSV from export_claims_for_labeling.py after you filled the `label` column
(1 = evidence supports the claim, 0 = it does not; blank rows are ignored) and reports
  * false alarms         : claims you labeled 1 that the verifier called unsupported (annoying)
  * missed hallucinations: claims you labeled 0 that the verifier called supported  (dangerous)
  * a threshold sweep, and a suggested CLAIM_SUPPORT_THRESHOLD.

Usage:  python scripts\score_labeled_claims.py claims_to_label.xlsx [--threshold 0.5]   (.csv also works)
"""
import argparse
import csv


def read_rows(path):
    """Yield dict rows from a .csv (utf-8) or .xlsx file."""
    if str(path).lower().endswith(".xlsx"):
        from openpyxl import load_workbook
        ws = load_workbook(path, data_only=True).active
        it = ws.iter_rows(values_only=True)
        header = [str(h) for h in next(it)]
        for row in it:
            yield dict(zip(header, row))
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:
            yield from csv.DictReader(f)


def parse(v):
    """1 / 0 from 1, 0, 1.0, 0.0, "1", "yes" ...; None for blank or undecided. (Excel hands back numbers, so a
    typed 0 arrives as the integer 0 - it must not be mistaken for a blank cell.)"""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v) if v in (0, 1) else None
    v = str(v).strip().lower()
    if v in ("1", "1.0", "y", "yes", "true", "supported"):
        return 1
    if v in ("0", "0.0", "n", "no", "false", "unsupported"):
        return 0
    return None


def metrics(rows, t):
    tp = sum(1 for s, h in rows if h == 1 and s >= t)
    fn = sum(1 for s, h in rows if h == 1 and s < t)      # false alarm
    fp = sum(1 for s, h in rows if h == 0 and s >= t)     # missed hallucination
    tn = sum(1 for s, h in rows if h == 0 and s < t)
    n = len(rows)
    acc = (tp + tn) / n if n else 0.0
    false_alarm = fn / (tp + fn) if tp + fn else 0.0
    missed = fp / (fp + tn) if fp + tn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, acc=acc, false_alarm=false_alarm, missed=missed, f1=f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--threshold", type=float, default=0.50, help="current CLAIM_SUPPORT_THRESHOLD")
    ap.add_argument("--errors", action="store_true", help="list the false alarms and missed hallucinations at --threshold")
    ap.add_argument("--max-missed", type=float, default=0.10, help="largest acceptable missed-hallucination rate")
    a = ap.parse_args()

    rows, detail = [], []
    for r in read_rows(a.csv):
        h = parse(r.get("label"))
        if h is not None:
            rows.append((float(r["nli_score"]), h))
            detail.append((r.get("id"), float(r["nli_score"]), h, str(r.get("claim") or "")))
    if len(rows) < 10:
        raise SystemExit(f"Only {len(rows)} labeled rows - label at least 30 for meaningful numbers.")

    supported = sum(h for _, h in rows)
    print(f"Labeled claims: {len(rows)}  (you judged {supported} supported, {len(rows) - supported} unsupported)")
    m = metrics(rows, a.threshold)
    print(f"\nAt the current threshold {a.threshold:.2f}:")
    print(f"  accuracy {m['acc']:.0%} | false alarms {m['fn']}/{m['tp'] + m['fn']} ({m['false_alarm']:.0%}) "
          f"| missed hallucinations {m['fp']}/{m['fp'] + m['tn']} ({m['missed']:.0%})")

    if a.errors:
        print("\nFALSE ALARMS (you: supported, verifier: not supported):")
        for i, sc, h, claim in sorted(detail, key=lambda d: d[1]):
            if h == 1 and sc < a.threshold:
                print(f"  #{i} nli={sc:.3f}  {' '.join(claim.split())[:130]}")
        print("MISSED HALLUCINATIONS (you: unsupported, verifier: supported):")
        for i, sc, h, claim in sorted(detail, key=lambda d: -d[1]):
            if h == 0 and sc >= a.threshold:
                print(f"  #{i} nli={sc:.3f}  {' '.join(claim.split())[:130]}")
    print(f"\n{'threshold':>9} {'accuracy':>9} {'false alarm':>12} {'missed halluc.':>15} {'F1':>6}")
    negatives = len(rows) - supported
    if negatives < 10:
        print(f"\n!! Only {negatives} claims were judged UNSUPPORTED. The missed-hallucination rate below is not "
              f"meaningful (it is {negatives} claim(s) deep). Treat it as unknown, and judge the verifier by its "
              f"false-alarm rate only. To measure missed hallucinations you need >= 10-15 unsupported claims, e.g. "
              f"by labeling claims from answers to adversarial/unsupported questions.")
    ok = []
    for k in range(1, 20):
        t = k * 0.05
        mm = metrics(rows, t)
        print(f"{t:>9.2f} {mm['acc']:>9.0%} {mm['false_alarm']:>12.0%} {mm['missed']:>15.0%} {mm['f1']:>6.2f}")
        if mm["missed"] <= a.max_missed:
            ok.append((t, mm))
    best = None
    if ok:
        lowest = min(mm["false_alarm"] for _, mm in ok)
        tied = [(t, mm) for t, mm in ok if mm["false_alarm"] <= lowest + 1e-9]
        best = tied[len(tied) // 2]        # middle of the equally-good range, not an extreme edge
    if negatives < 10:
        best = None
        print("\nNo threshold suggestion: too few unsupported claims in the sample (see warning above).")
    elif best:
        print(f"\nSuggested CLAIM_SUPPORT_THRESHOLD={best[0]:.2f} (fewest false alarms with missed hallucinations "
              f"<= {a.max_missed:.0%}). Put it in .env, restart, re-run the evaluation.")
    elif negatives >= 10:
        print(f"\nNo threshold keeps missed hallucinations <= {a.max_missed:.0%}: the NLI model itself is too weak "
              f"for this text. Try a larger NLI_MODEL, then re-export and re-label a small sample.")
    print("\nWith ~60 labels these are rough estimates; report them as such.")


if __name__ == "__main__":
    main()
