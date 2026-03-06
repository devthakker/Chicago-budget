#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chicago_budget_rag.engine import RAGEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Chicago budget RAG index")
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--bm25-weight", type=float, default=None, help="Override BM25 weight")
    parser.add_argument("--vector-weight", type=float, default=None, help="Override vector weight")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = RAGEngine(args.index_dir)
    payload = engine.answer(
        args.query,
        top_k=args.top_k,
        bm25_weight=args.bm25_weight,
        vector_weight=args.vector_weight,
    )

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("\nAnswer\n------")
    print(payload["answer"])
    print("\nTop Sources\n-----------")
    for row in payload["results"]:
        print(
            f"- {row['source_file']} p.{row['page_start']}-{row['page_end']} "
            f"(score={row['score']:.3f}, toc_like={row.get('toc_like', False)})"
        )


if __name__ == "__main__":
    main()
