#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chicago_budget_rag.engine import RAGEngine


@dataclass
class QueryEval:
    query_id: str
    query: str
    hit: bool
    first_hit_rank: int | None
    reciprocal_rank: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and tune RAG retrieval settings")
    parser.add_argument("--questions-file", type=Path, default=Path("eval/questions.sample.json"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--bm25-weight", type=float, default=0.85)
    parser.add_argument("--vector-weight", type=float, default=0.15)
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    parser.add_argument("--show-queries", action="store_true", help="Show per-query result summary")

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run grid search across BM25 weights (vector weight is 1-bm25)",
    )
    parser.add_argument(
        "--bm25-grid",
        type=str,
        default="0.60,0.70,0.75,0.80,0.85,0.90,0.95",
        help="Comma-separated BM25 weights for tuning",
    )
    parser.add_argument(
        "--objective",
        choices=["hit_rate", "mrr"],
        default="mrr",
        help="Metric used to pick best config during tuning",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Questions file not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise SystemExit("Questions file must contain a non-empty list")
    return data


def has_target_match(result: dict[str, Any], targets: list[dict[str, Any]]) -> bool:
    for t in targets:
        if result.get("source_file") != t.get("source_file"):
            continue
        p_start = int(result.get("page_start", 0))
        p_end = int(result.get("page_end", 0))
        t_min = int(t.get("page_min", 0))
        t_max = int(t.get("page_max", 0))
        if p_start <= t_max and p_end >= t_min:
            return True
    return False


def evaluate_once(
    engine: RAGEngine,
    questions: list[dict[str, Any]],
    top_k: int,
    bm25_weight: float,
    vector_weight: float,
) -> dict[str, Any]:
    per_query: list[QueryEval] = []

    for idx, q in enumerate(questions, start=1):
        qid = str(q.get("id", f"q{idx}"))
        query = str(q["query"])
        targets = q.get("targets", [])
        if not isinstance(targets, list) or not targets:
            raise SystemExit(f"Question {qid} missing targets")

        payload = engine.answer(
            query,
            top_k=top_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )
        results = payload.get("results", [])

        first_hit_rank = None
        for rank, r in enumerate(results, start=1):
            if has_target_match(r, targets):
                first_hit_rank = rank
                break

        hit = first_hit_rank is not None
        rr = 0.0 if first_hit_rank is None else 1.0 / first_hit_rank

        per_query.append(
            QueryEval(
                query_id=qid,
                query=query,
                hit=hit,
                first_hit_rank=first_hit_rank,
                reciprocal_rank=rr,
            )
        )

    total = len(per_query)
    hits = sum(1 for row in per_query if row.hit)
    mrr = sum(row.reciprocal_rank for row in per_query) / total
    mean_first_hit_rank = statistics.mean([row.first_hit_rank for row in per_query if row.first_hit_rank is not None]) if hits else math.inf

    report = {
        "total_queries": total,
        "top_k": top_k,
        "bm25_weight": bm25_weight,
        "vector_weight": vector_weight,
        "hit_rate": hits / total,
        "mrr": mrr,
        "mean_first_hit_rank": None if not math.isfinite(mean_first_hit_rank) else mean_first_hit_rank,
        "per_query": [
            {
                "id": row.query_id,
                "query": row.query,
                "hit": row.hit,
                "first_hit_rank": row.first_hit_rank,
                "reciprocal_rank": row.reciprocal_rank,
            }
            for row in per_query
        ],
    }
    return report


def parse_grid(grid: str) -> list[float]:
    vals = []
    for part in grid.split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise SystemExit("--bm25-grid produced no values")
    return vals


def tune(
    engine: RAGEngine,
    questions: list[dict[str, Any]],
    top_k: int,
    bm25_grid: list[float],
    objective: str,
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    best = None

    for w in bm25_grid:
        w = max(0.0, min(1.0, w))
        report = evaluate_once(
            engine=engine,
            questions=questions,
            top_k=top_k,
            bm25_weight=w,
            vector_weight=(1.0 - w),
        )
        trials.append(report)

        score = report[objective]
        tie_break = report["hit_rate"] if objective == "mrr" else report["mrr"]

        if best is None:
            best = (score, tie_break, report)
        else:
            if score > best[0] or (score == best[0] and tie_break > best[1]):
                best = (score, tie_break, report)

    assert best is not None
    return {
        "objective": objective,
        "top_k": top_k,
        "trials": trials,
        "best": best[2],
    }


def print_single_report(report: dict[str, Any], show_queries: bool) -> None:
    print("\nEvaluation Report")
    print("-----------------")
    print(f"Queries: {report['total_queries']}")
    print(f"Top-k: {report['top_k']}")
    print(f"Weights: bm25={report['bm25_weight']:.2f}, vector={report['vector_weight']:.2f}")
    print(f"Hit Rate: {report['hit_rate']:.3f}")
    print(f"MRR: {report['mrr']:.3f}")
    mean_rank = report.get("mean_first_hit_rank")
    print(f"Mean First Hit Rank: {mean_rank if mean_rank is not None else 'n/a'}")

    if show_queries:
        print("\nPer Query")
        print("---------")
        for row in report["per_query"]:
            status = "hit" if row["hit"] else "miss"
            rank = row["first_hit_rank"] if row["first_hit_rank"] is not None else "-"
            print(f"- {row['id']} [{status}] rank={rank} :: {row['query']}")


def print_tuning_report(report: dict[str, Any]) -> None:
    print("\nTuning Report")
    print("-------------")
    print(f"Objective: {report['objective']}")
    print(f"Top-k: {report['top_k']}")
    print("\nTrials")
    print("------")
    for t in sorted(report["trials"], key=lambda x: x["bm25_weight"]):
        print(
            f"- bm25={t['bm25_weight']:.2f}, vector={t['vector_weight']:.2f} "
            f"| hit_rate={t['hit_rate']:.3f} | mrr={t['mrr']:.3f}"
        )

    best = report["best"]
    print("\nBest")
    print("----")
    print(
        f"bm25={best['bm25_weight']:.2f}, vector={best['vector_weight']:.2f}, "
        f"hit_rate={best['hit_rate']:.3f}, mrr={best['mrr']:.3f}"
    )


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions_file)
    engine = RAGEngine(args.index_dir)

    if args.tune:
        grid = parse_grid(args.bm25_grid)
        report = tune(
            engine=engine,
            questions=questions,
            top_k=args.top_k,
            bm25_grid=grid,
            objective=args.objective,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_tuning_report(report)
        return

    report = evaluate_once(
        engine=engine,
        questions=questions,
        top_k=args.top_k,
        bm25_weight=args.bm25_weight,
        vector_weight=args.vector_weight,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_single_report(report, show_queries=args.show_queries)


if __name__ == "__main__":
    main()
