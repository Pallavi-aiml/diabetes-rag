"""
scripts/clean_existing_chunks.py — fix PDF artefacts in your EXISTING knowledge base
without re-parsing PDFs or re-embedding everything.

What it does
  1. Backs up data/processed/chunks.json
  2. Normalises text of clinical_guideline chunks (ligatures, broken hyphenation,
     control characters) using ingestion/text_cleaning.py
  3. Re-embeds and upserts ONLY the chunks whose text changed into ChromaDB

Usage:
    python scripts/clean_existing_chunks.py --dry-run     # report only
    python scripts/clean_existing_chunks.py               # apply (a few minutes on CPU)
Restart the API afterwards so it reloads chunks.json.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.text_cleaning import normalize_pdf_text  # noqa: E402

CHUNKS = Path("data/processed/chunks.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    changed = []
    for c in chunks:
        if c.get("source_type") != "clinical_guideline":
            continue
        new = normalize_pdf_text(c.get("text", ""))
        if new and new != c["text"]:
            c["text"] = new
            changed.append(c)

    print(f"{len(chunks)} chunks total, {len(changed)} guideline chunks cleaned")
    if args.dry_run or not changed:
        return

    shutil.copy(CHUNKS, CHUNKS.with_suffix(".json.bak"))
    CHUNKS.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"Saved cleaned chunks (backup: {CHUNKS.with_suffix('.json.bak')})")

    from ingestion.embedder import build_chroma_store  # heavy import, only when applying
    build_chroma_store(changed)                         # upsert by chunk_id
    print(f"Re-embedded {len(changed)} chunks. Restart the API.")


if __name__ == "__main__":
    main()
