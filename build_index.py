#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chicago_budget_rag.engine import RAGEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAG index for Chicago budget PDFs")
    parser.add_argument("--pdf-dir", type=Path, default=Path("."), help="Directory containing PDF files")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"), help="Directory to write index.json")
    parser.add_argument("--max-tokens", type=int, default=800, help="Max tokens per chunk")
    parser.add_argument("--overlap-tokens", type=int, default=120, help="Token overlap between chunks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.pdf_dir}")

    engine = RAGEngine(args.index_dir)
    index = engine.build(pdfs, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens)
    print(f"Index written to {engine.index_file}")
    print(f"Chunks: {index['stats']['chunk_count']}")
    if index["stats"]["embedding_model"]:
        print(f"Embeddings model: {index['stats']['embedding_model']}")
    else:
        print("Embeddings: disabled (set OPENAI_API_KEY to enable)")


if __name__ == "__main__":
    main()
