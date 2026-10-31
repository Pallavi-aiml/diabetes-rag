"""
scripts/check_index_sync.py — is ChromaDB (dense search) in sync with chunks.json (BM25)?

Dense retrieval reads ChromaDB; BM25 reads data/processed/chunks.json. If their chunk IDs
differ, some chunks are only findable by one of the two searches.

Usage:
    python scripts/check_index_sync.py          # report only
    python scripts/check_index_sync.py --fix    # embed chunks missing from Chroma, delete extras
Stop the API before using --fix.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def diff_ids(json_ids: set, chroma_ids: set):
    return sorted(json_ids - chroma_ids), sorted(chroma_ids - json_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()

    from ingestion.embedder import get_chroma_collection

    chunks = json.loads(Path("data/processed/chunks.json").read_text(encoding="utf-8"))
    by_id = {c["chunk_id"]: c for c in chunks}
    col = get_chroma_collection()
    chroma_ids = set(col.get(include=[])["ids"])

    missing, extra = diff_ids(set(by_id), chroma_ids)
    print(f"chunks.json: {len(by_id)} | ChromaDB: {len(chroma_ids)}")
    print(f"in chunks.json but NOT in Chroma (invisible to dense search): {len(missing)}")
    print(f"in Chroma but NOT in chunks.json (invisible to BM25, stale):  {len(extra)}")
    for cid in missing[:3]:
        print("  missing e.g.", cid, "|", by_id[cid]["text"][:70].replace("\n", " "))
    for cid in extra[:3]:
        print("  extra   e.g.", cid)

    if not args.fix:
        print("\nRe-run with --fix to repair.")
        return
    if missing:
        from ingestion.embedder import build_chroma_store
        build_chroma_store([by_id[c] for c in missing])
    for i in range(0, len(extra), 500):
        col.delete(ids=extra[i:i + 500])
    print(f"\nFixed. ChromaDB now has {col.count()} chunks.")


if __name__ == "__main__":
    main()
