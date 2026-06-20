"""

Computes Hit Rate@K and Precision@K for the Expert Finder retrieval pipeline
against evaluation_answer_key.json, broken down by hop level (0/1/2/3).

Usage:
    python score_hit_rate.py --run-name baseline --pipeline baseline
    python score_hit_rate.py --run-name template1 --pipeline template1
    python score_hit_rate.py --run-name template1 --pipeline template1 --filter-ids Q01,Q02,Q03,Q07,Q09,Q11,Q12,Q16

Each run's detailed results are saved to results/<run_name>_results.json so
multiple pipeline variants (baseline, template1, +expansion, +decomposition)
can be compared later in the ablation study.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

ANSWER_KEY_PATH = Path("eval/evaluation_answer_key.json")
RESULTS_DIR = Path("results")


def get_pipeline_fn(pipeline_name: str, top_k: int):
    """
    Dispatch to the right retrieval pipeline based on --pipeline.
    Each branch returns a callable: question (str) -> list[dict] of rows,
    so the rest of the scoring logic never needs to know which pipeline
    produced the rows.
    """
    if pipeline_name == "baseline":
        from text2cypher import ask
        return ask

    elif pipeline_name == "template1":
        from template1_find_experts import find_experts_by_topic
        # template1's top_n should match the scoring top_k so the comparison
        # against baseline is apples-to-apples (same K on both sides)
        return lambda question: find_experts_by_topic(question, top_n=top_k)
    
    elif pipeline_name == "template2":
        from template2_find_projects import find_projects_by_topic_with_experts
        return lambda question: find_projects_by_topic_with_experts(question, top_n=top_k)
    
    elif pipeline_name == "baseline_expanded":
        from text2cypher_expanded import ask_with_expansion
        return ask_with_expansion

    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")


def normalize(s: str) -> str:
    """Lowercase + strip for robust string comparison."""
    return str(s).strip().lower()


def extract_strings(record) -> list:
    """
    Pull every string-like value out of a single result row, regardless of
    field name (e.g. p.cfPersNameEN, o.cfNameEN, fund.cfNameEN, ea.cfURI).
    Handles dicts, neo4j Record objects, and lists/tuples inside a row
    (e.g. collect(...) results).
    """
    if record is None:
        return []

    if hasattr(record, "keys") and not isinstance(record, dict):
        try:
            record = dict(record)
        except Exception:
            pass

    if isinstance(record, dict):
        iterable = record.values()
    elif isinstance(record, (list, tuple)):
        iterable = record
    else:
        iterable = [record]

    values = []
    for v in iterable:
        if isinstance(v, str):
            values.append(v)
        elif isinstance(v, (list, tuple)):
            values.extend(item for item in v if isinstance(item, str))
    return values


def row_is_relevant(ground_truth: list, row) -> bool:
    """True if this single result row contains any ground-truth entity."""
    strings = [normalize(s) for s in extract_strings(row)]
    for gt in ground_truth:
        gt_norm = normalize(gt)
        if any(gt_norm in s or s in gt_norm for s in strings):
            return True
    return False


def hit_rate_at_k(ground_truth: list, rows: list, k: int) -> bool:
    """True if ANY of the top-k rows contains a ground-truth entity."""
    return any(row_is_relevant(ground_truth, row) for row in rows[:k])


def precision_at_k(ground_truth: list, rows: list, k: int) -> float:
    """Fraction of the top-k rows that are individually relevant."""
    top_k = rows[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for row in top_k if row_is_relevant(ground_truth, row))
    return relevant / len(top_k)


def run_evaluation(run_name: str, top_k: int, pipeline_name: str, filter_ids: list = None):
    with open(ANSWER_KEY_PATH, encoding="utf-8") as f:
        answer_key = json.load(f)

    questions = answer_key["questions"]
    if filter_ids:
        questions = [q for q in questions if q["id"] in filter_ids]
        if not questions:
            print(f"No questions matched filter-ids: {filter_ids}")
            return

    pipeline_fn = get_pipeline_fn(pipeline_name, top_k)

    detailed_results = []
    hits_by_hop = defaultdict(list)
    precision_by_hop = defaultdict(list)

    for q in questions:
        qid, question_text, hop_level, ground_truth = (
            q["id"], q["question"], q["hop_level"], q["ground_truth"]
        )
        print(f"[{qid}] (hop {hop_level}) {question_text}")

        try:
            rows = pipeline_fn(question_text)
        except Exception as e:
            print(f"  -> ERROR: {e}")
            rows = []

        hit = hit_rate_at_k(ground_truth, rows, top_k)
        precision = precision_at_k(ground_truth, rows, top_k)
        top_k_preview = [extract_strings(r) for r in rows[:top_k]]

        hits_by_hop[hop_level].append(1 if hit else 0)
        precision_by_hop[hop_level].append(precision)

        print(f"  -> {'HIT ' if hit else 'MISS'} | top-{top_k}: {top_k_preview}")

        detailed_results.append({
            "id": qid,
            "question": question_text,
            "hop_level": hop_level,
            "ground_truth": ground_truth,
            "returned_top_k": top_k_preview,
            "hit": hit,
            "precision_at_k": round(precision, 3),
        })

    # ── Aggregate ──
    all_hits = [h for v in hits_by_hop.values() for h in v]
    all_precision = [p for v in precision_by_hop.values() for p in v]
    overall_hit_rate = sum(all_hits) / len(all_hits) if all_hits else 0.0
    overall_precision = sum(all_precision) / len(all_precision) if all_precision else 0.0

    print("\n" + "=" * 60)
    print(f"RUN: {run_name}  (pipeline: {pipeline_name}, n={len(questions)})")
    print("=" * 60)
    print(f"Overall Hit Rate@{top_k}:  {overall_hit_rate:.1%}  ({sum(all_hits)}/{len(all_hits)})")
    print(f"Overall Precision@{top_k}: {overall_precision:.1%}")
    print("-" * 60)
    print(f"{'Hop Level':<12}{'Hit Rate':<12}{'Precision':<12}{'n':<5}")
    for hop in sorted(hits_by_hop.keys()):
        hits, precisions = hits_by_hop[hop], precision_by_hop[hop]
        hr = sum(hits) / len(hits) if hits else 0.0
        pr = sum(precisions) / len(precisions) if precisions else 0.0
        hr_str, pr_str = f"{hr:.1%}", f"{pr:.1%}"
        print(f"{hop:<12}{hr_str:<12}{pr_str:<12}{len(hits):<5}")

    # ── Save for cross-run comparison ──
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"{run_name}_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_name": run_name,
            "pipeline": pipeline_name,
            "top_k": top_k,
            "n_questions": len(questions),
            "overall_hit_rate": overall_hit_rate,
            "overall_precision": overall_precision,
            "hit_rate_by_hop": {h: sum(v) / len(v) for h, v in hits_by_hop.items()},
            "precision_by_hop": {h: sum(v) / len(v) for h, v in precision_by_hop.items()},
            "detailed_results": detailed_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score retrieval accuracy against the answer key.")
    parser.add_argument("--run-name", default="baseline", help="e.g. baseline, template1, +expansion")
    parser.add_argument("--pipeline", default="baseline", choices=["baseline", "template1", "template2","baseline_expanded"],
                         help="Which retrieval pipeline to score")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--filter-ids", default=None,
                         help="Comma-separated question IDs to run, e.g. Q01,Q02,Q03 (default: all)")
    args = parser.parse_args()

    filter_ids = args.filter_ids.split(",") if args.filter_ids else None
    run_evaluation(args.run_name, args.top_k, args.pipeline, filter_ids)