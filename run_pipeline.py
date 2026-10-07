"""
run_pipeline.py
Master script — runs the entire ingestion pipeline in order.

Usage:
  python run_pipeline.py --all          # Full pipeline (fetch + chunk + embed)
  python run_pipeline.py --ingest       # Only fetch PubMed + parse PDFs
  python run_pipeline.py --chunk        # Only chunk loaded documents
  python run_pipeline.py --embed        # Only build vector stores
  python run_pipeline.py --test-query   # Quick retrieval smoke test
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def step_ingest():
    print("\n" + "=" * 60)
    print("STEP 1 — DATA INGESTION")
    print("=" * 60)
    from ingestion.loader import fetch_all_pubmed, load_all_pdfs

    print("\n[1a] Fetching PubMed abstracts…")
    t0 = time.time()
    pubmed_records = fetch_all_pubmed()
    print(f"     Done in {time.time()-t0:.1f}s — {len(pubmed_records)} records")

    print("\n[1b] Parsing guideline PDFs…")
    print("     Place ADA/WHO/ICMR PDFs in ./data/raw/ before running.")
    pdf_records = load_all_pdfs()
    print(f"     Done — {len(pdf_records)} pages")

    return len(pubmed_records) + len(pdf_records)


def step_chunk():
    print("\n" + "=" * 60)
    print("STEP 2 — CHUNKING")
    print("=" * 60)
    from ingestion.loader import load_all_documents
    from ingestion.chunker import chunk_documents, save_chunks

    docs = load_all_documents()
    if not docs:
        print("No documents found — run --ingest first.")
        return 0

    print(f"Chunking {len(docs)} documents…")
    chunks = chunk_documents(docs)
    save_chunks(chunks)
    print(f"Generated {len(chunks)} chunks.")
    return len(chunks)


def step_embed():
    print("\n" + "=" * 60)
    print("STEP 3 — EMBEDDING + VECTOR STORE")
    print("=" * 60)
    from ingestion.chunker import load_chunks
    from ingestion.embedder import build_chroma_store, build_faiss_index

    chunks = load_chunks()
    if not chunks:
        print("No chunks found — run --chunk first.")
        return

    print(f"Building ChromaDB store for {len(chunks)} chunks…")
    build_chroma_store(chunks)

    print(f"\nBuilding FAISS index…")
    build_faiss_index(chunks)
    print("Vector stores ready.")


def step_test_query():
    print("\n" + "=" * 60)
    print("SMOKE TEST — Quick retrieval check")
    print("=" * 60)
    try:
        from ingestion.chunker import load_chunks
        from ingestion.embedder import get_chroma_collection
        from retrieval.retriever import hybrid_retrieve
        from evidence.verifier import verify_chunks, aggregate_confidence

        chunks = load_chunks()
        collection = get_chroma_collection()

        query = "What is the recommended HbA1c target for type 2 diabetes?"
        print(f"Query: {query}\n")

        results = hybrid_retrieve(query, collection, chunks, top_k=3)
        verified = verify_chunks(results)

        print(f"Retrieved {len(verified)} chunks:\n")
        for i, r in enumerate(verified, 1):
            print(f"  [{i}] {r.get('journal','?')} ({r.get('year','?')}) "
                  f"| Grade {r['evidence_grade']} | Cred {r['credibility_score']:.1f}/5")
            print(f"       {r['text'][:150]}…\n")

        dummy_answer = "The HbA1c target for type 2 diabetes is less than 7%."
        report = aggregate_confidence(verified, dummy_answer)
        print(f"Confidence: {report['confidence']:.3f} | Grade: {report['grade']} | "
              f"Faithfulness: {report['faithfulness']:.3f}")
        print("\nSmoke test passed ✅")
    except Exception as e:
        print(f"Smoke test failed ❌: {e}")


def main():
    parser = argparse.ArgumentParser(description="Diabetes RAG pipeline runner")
    parser.add_argument("--all",        action="store_true", help="Run full pipeline")
    parser.add_argument("--ingest",     action="store_true", help="Run ingestion only")
    parser.add_argument("--chunk",      action="store_true", help="Run chunking only")
    parser.add_argument("--embed",      action="store_true", help="Run embedding only")
    parser.add_argument("--test-query", action="store_true", help="Run smoke test")
    args = parser.parse_args()

    if args.all:
        step_ingest()
        step_chunk()
        step_embed()
        step_test_query()
    elif args.ingest:
        step_ingest()
    elif args.chunk:
        step_chunk()
    elif args.embed:
        step_embed()
    elif args.test_query:
        step_test_query()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
