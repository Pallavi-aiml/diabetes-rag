r"""
scripts/debug_retrieval.py — see WHY a question retrieved what it did.

Shows (1) dense and BM25 candidates, then (2) every candidate that enters the reranker with
its scores and the exact reason it was kept or cut. A "✔" marks chunks whose text matches
--expect (a regex), so you can follow the chunk you care about through every stage.

Usage:
    python scripts/debug_retrieval.py "What is the recommended HbA1c target for adults with type 2 diabetes?" \
        --expect "goal of 7%|A1C.{0,30}<\s*7"
    python scripts/debug_retrieval.py "..." --expect "..." --bonus 0     # override GUIDELINE_INTENT_BONUS
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TOP_K_RETRIEVE, MIN_RERANK_SCORE, MIN_QUERY_RELEVANCE, RERANK_POOL, RERANK_KEEP  # noqa: E402
from ingestion.chunker import load_chunks  # noqa: E402
from ingestion.embedder import get_chroma_collection  # noqa: E402
from retrieval import retriever as R  # noqa: E402


def mark(c, expect):
    return "✔" if expect and re.search(expect, c.get("text", ""), re.IGNORECASE) else " "


def show(title, chunks, expect, n=8):
    print(f"\n── {title} " + "─" * max(1, 60 - len(title)))
    for i, c in enumerate(chunks[:n], 1):
        print(f"{i:>2}{mark(c, expect)} {c.get('source_type','?')[:12]:<12} {c.get('source','?'):<22} "
              f"{c.get('text','')[:80].replace(chr(10), ' ')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--expect", default="", help='regex the right chunk should contain, e.g. "goal of 7%%"')
    ap.add_argument("--bonus", type=float, default=None, help="override GUIDELINE_INTENT_BONUS for this run")
    ap.add_argument("--top-k", type=int, default=5)
    a = ap.parse_args()
    if a.bonus is not None:
        R.GUIDELINE_INTENT_BONUS = a.bonus

    col = get_chroma_collection()
    chunks = load_chunks()
    dense = R.dense_retrieve(a.question, col, n=TOP_K_RETRIEVE)
    bm25 = R.bm25_retrieve(a.question, R.build_bm25_index(chunks), chunks, n=TOP_K_RETRIEVE)
    show("dense candidates", dense, a.expect)
    show("BM25 candidates", bm25, a.expect)

    # Replicate hybrid_retrieve() step by step.
    fused = R.reciprocal_rank_fusion([dense, bm25])
    cand = fused[:max(RERANK_POOL, a.top_k + 3)]
    reranked_all = R.rerank(a.question, cand, top_k=len(cand))          # all candidates, scored
    keep_n = min(len(cand), max(a.top_k * 3, a.top_k + 3, RERANK_KEEP))
    survivors = R.filter_relevant(reranked_all[:keep_n], query=a.question)
    order = R.boost_quality_sources(survivors, a.question)
    pos = {c["chunk_id"]: i for i, c in enumerate(order, 1)}
    surv_ids = {c["chunk_id"] for c in survivors}
    rel = {c["chunk_id"]: c.get("query_relevance_score", 0.0) for c in survivors}

    print(f"\n── every reranker candidate ({len(cand)} chunks) ─────────────────────")
    print("rr# = rerank order | rerank = cross-encoder score | relev = query relevance | fate")
    for i, c in enumerate(reranked_all, 1):
        rs = c["rerank_score"]
        cid = c["chunk_id"]
        if i > keep_n:
            fate = f"cut: outside rerank top-{keep_n}"
        elif rs < MIN_RERANK_SCORE:
            fate = f"cut: rerank {rs:.2f} < {MIN_RERANK_SCORE}"
        elif not R.is_diabetes_relevant(c):
            fate = "cut: not diabetes-relevant"
        elif cid not in surv_ids:
            fate = f"cut: relevance <= {MIN_QUERY_RELEVANCE}"
        elif pos[cid] <= a.top_k:
            fate = f"KEPT -> final #{pos[cid]}"
        else:
            fate = f"outranked (final #{pos[cid]})"
        print(f"{i:>3}{mark(c, a.expect)} {c.get('source_type','?')[:10]:<10} {c.get('source','?'):<20} "
              f"rerank={rs:>6.2f} relev={rel.get(cid, 0.0):.2f}  {fate}")

    final = order[:a.top_k]
    show("FINAL (what the LLM sees)", final, a.expect, n=a.top_k)
    if a.expect and not any(re.search(a.expect, c.get("text", ""), re.IGNORECASE) for c in final):
        print("\n⚠ none of the final chunks match --expect. Look at the ✔ rows above: the 'fate' column "
              "says where each was lost. If no ✔ row exists, it never reached the reranker (recall problem).")


if __name__ == "__main__":
    main()
