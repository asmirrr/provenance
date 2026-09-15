"""Batch dense diagnostics against explicit evidence groups, including draft labels."""

import argparse
import json
from pathlib import Path

from src.benchmark import Benchmark, bind_benchmark
from src.dense import load_model, runtime_metadata, search_many
from src.evidence import verify_source_pdf
from src.ingestion.pipeline import ProcessedFiling


def coverage(item, ranked_ids):
    """Best group recall; complete support never mixes partial alternatives."""
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Duplicate ranked IDs")
    if not item["answerable"]:
        return None
    groups = [set(g["supporting_chunk_ids"]) for g in item["evidence_groups"]]
    if not groups or any(not g for g in groups):
        raise ValueError("Answerable items need nonempty evidence groups")
    union = set.union(*groups)
    first = next((i for i, cid in enumerate(ranked_ids, 1) if cid in union), None)
    output = {"reciprocal_rank_at_10": 1 / first if first and first <= 10 else 0.0}
    for k in (1, 3, 5, 10):
        selected = set(ranked_ids[:k])
        output[f"best_group_recall_at_{k}"] = max(len(g & selected) / len(g) for g in groups)
        output[f"complete_group_at_{k}"] = float(any(g <= selected for g in groups))
    return output


def evaluate(document, benchmark, model, *, encoding, allow_draft=False):
    if benchmark.status != "human_reviewed" and not allow_draft:
        raise ValueError("Benchmark is pending human review; use --allow-draft for development diagnostics")
    bound = bind_benchmark(document, benchmark)
    runs = search_many(document, [i.question for i in benchmark.items], model, top_k=10, encoding=encoding)
    rows = []
    for item, run in zip(bound["items"], runs, strict=True):
        rows.append({"item_id": item["id"], "category": item["category"],
                     "metrics": coverage(item, [r["chunk"]["chunk_id"] for r in run["results"]]),
                     "retrieval": run})
    measured = [r["metrics"] for r in rows if r["metrics"] is not None]
    return {"schema_version": 1, "benchmark_version": benchmark.version,
            "benchmark_model_sha256": bound["benchmark_model_sha256"],
            "document_model_sha256": bound["document_model_sha256"],
            "benchmark_status": benchmark.status,
            "evaluation_status": "draft_diagnostic" if benchmark.status != "human_reviewed" else "reviewed_labels",
            "answerable_denominator": len(measured), "unsupported_excluded": len(rows) - len(measured),
            "aggregate": {key: sum(m[key] for m in measured) / len(measured) for key in measured[0]} if measured else {},
            "items": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--encoding", choices=["prefix", "window-mean"], default="window-mean")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--source-pdf", type=Path)
    args = parser.parse_args()
    inputs = [args.document, args.benchmark] + ([args.source_pdf] if args.source_pdf else [])
    if args.output.resolve() in {p.resolve() for p in inputs}:
        parser.error("Output must not overwrite an input")
    document = ProcessedFiling.model_validate_json(args.document.read_text(encoding="utf-8"))
    benchmark = Benchmark.model_validate_json(args.benchmark.read_text(encoding="utf-8"))
    if benchmark.status != "human_reviewed" and not args.allow_draft:
        parser.error("Use --allow-draft for this pending benchmark")
    # Validate source anchors before model loading or output writes.
    bind_benchmark(document, benchmark)
    verification = verify_source_pdf(document, args.source_pdf) if args.source_pdf else {"status": "not_checked"}
    result = evaluate(document, benchmark, load_model(args.local_files_only),
                      encoding=args.encoding, allow_draft=args.allow_draft)
    result.update(runtime_metadata())
    result["source_pdf_verification"] = verification
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["evaluation_status"], "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
