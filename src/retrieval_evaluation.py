"""Batch retrieval diagnostics against explicit evidence groups, including draft labels."""

import argparse
import json
from pathlib import Path

from src.benchmark import Benchmark, bind_benchmark
from src import bm25, hybrid
from src.dense import load_model, runtime_metadata, search_many
from src.evidence import verify_source_pdf, select_page_context
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


def page_coverage(document, item, ranked_ids, max_chars=8000, page_policy="strict"):
    """Select context without labels, then check exact anchored evidence coverage."""
    selection = select_page_context(document, ranked_ids, max_chars=max_chars, policy=page_policy)
    bundle = selection["bundle"]
    groups = []
    if bundle:
        for group in item["evidence_groups"]:
            if group["evidence"] and all(any(
                span["page"] == anchor["page"]
                and span["raw_start"] <= anchor["raw_start"] < anchor["raw_end"] <= span["raw_end"]
                and span["text"][anchor["raw_start"] - span["raw_start"]:anchor["raw_end"] - span["raw_start"]] == anchor["quote"]
                for span in bundle["spans"]) for anchor in group["evidence"]):
                groups.append(group["group_id"])
    return {**selection, "complete_group_ids": groups if item["answerable"] else None,
            "complete_group": bool(groups) if item["answerable"] else None}


def summarize_runs(document, benchmark, bound, runs, retriever, max_chars=8000, page_policy="strict"):
    if len(runs) != len(bound["items"]):
        raise ValueError("Retrieval run item count mismatch")
    chunks = {c.chunk_id: c.model_dump(mode="json") for c in document.chunks}
    rows = []
    for item, run in zip(bound["items"], runs, strict=True):
        if (run["question"] != item["question"]
                or run["document_model_sha256"] != bound["document_model_sha256"]
                or run["source_sha256"] != document.source_sha256
                or run["requested_top_k"] != 10 or len(run["results"]) > 10):
            raise ValueError("Retrieval question, corpus, source or depth mismatch")
        ids = []
        for rank, result in enumerate(run["results"], 1):
            cid = result["chunk"]["chunk_id"]
            if result["rank"] != rank or chunks.get(cid) != result["chunk"] or cid in ids:
                raise ValueError("Invalid retrieved rank, duplicate or citation")
            ids.append(cid)
        rows.append({"item_id": item["id"], "category": item["category"],
                     "metrics": coverage(item, ids),
                     "page_context": {str(k): page_coverage(document, item, ids[:k], max_chars, page_policy) for k in (1, 3, 5, 10)},
                     "retrieval": run})
    measured = [r["metrics"] for r in rows if r["metrics"] is not None]
    supported = [r for r in rows if r["metrics"] is not None]
    context = {}
    for k in (1, 3, 5, 10):
        values = [r["page_context"][str(k)] for r in supported]
        context[str(k)] = {"complete_group_count": sum(v["complete_group"] for v in values),
                           "answerable_denominator": len(values),
                           "over_budget_count": sum(v["status"] == "over_budget" for v in values),
                           "requests_with_omissions": sum(bool(v["omitted_pages"]) for v in values),
                           "complete_group_rate": sum(v["complete_group"] for v in values) / len(values) if values else None}
    return {"schema_version": 3, "retriever": retriever, "benchmark_version": benchmark.version,
            "benchmark_model_sha256": bound["benchmark_model_sha256"],
            "document_model_sha256": bound["document_model_sha256"],
            "benchmark_status": benchmark.status,
            "evaluation_status": "draft_diagnostic" if benchmark.status != "human_reviewed" else "reviewed_labels",
            "answerable_denominator": len(measured), "unsupported_excluded": len(rows) - len(measured),
            "aggregate": {key: sum(m[key] for m in measured) / len(measured) for key in measured[0]} if measured else {},
            "page_context_policy": {"name": "all-selected-pages-or-error-v1" if page_policy == "strict" else "ranked-whole-pages-fit-v1", "max_chars": max_chars},
            "page_context_aggregate": context, "items": rows}


def evaluate(document, benchmark, model=None, *, encoding="window-mean", allow_draft=False, retriever="dense", max_chars=8000, page_policy="strict"):
    if max_chars < 1 or page_policy not in {"strict", "ranked-fit"}:
        raise ValueError("Require positive budget and a known page policy")
    if benchmark.status != "human_reviewed" and not allow_draft:
        raise ValueError("Benchmark is pending human review; use --allow-draft for development diagnostics")
    bound = bind_benchmark(document, benchmark)
    questions = [i.question for i in benchmark.items]
    if retriever == "bm25":
        runs = bm25.search_many(document, questions, top_k=10)
    elif retriever == "dense":
        runs = search_many(document, questions, model, top_k=10, encoding=encoding)
    elif retriever == "hybrid":
        runs = hybrid.search_many(document, questions, model, top_k=10, encoding=encoding)
    else:
        raise ValueError("Unknown retriever")
    return summarize_runs(document, benchmark, bound, runs, retriever, max_chars, page_policy)


def replay(document, benchmark, artifact, *, allow_draft=False, max_chars=8000, page_policy="strict"):
    if benchmark.status != "human_reviewed" and not allow_draft:
        raise ValueError("Use --allow-draft for pending benchmark labels")
    bound = bind_benchmark(document, benchmark)
    for key in ("benchmark_model_sha256", "document_model_sha256"):
        if artifact[key] != bound[key]:
            raise ValueError("Saved retrieval hashes do not match current inputs")
    if [i["item_id"] for i in artifact["items"]] != [i["id"] for i in bound["items"]]:
        raise ValueError("Saved retrieval item IDs/order mismatch")
    return summarize_runs(document, benchmark, bound, [i["retrieval"] for i in artifact["items"]],
                          artifact.get("retriever", "dense"), max_chars, page_policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--encoding", choices=["prefix", "window-mean"])
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--source-pdf", type=Path)
    parser.add_argument("--retrieval-run", type=Path, help="Replay saved rankings without model inference")
    parser.add_argument("--max-context-chars", type=int, default=8000)
    parser.add_argument("--page-policy", choices=["strict", "ranked-fit"], default="strict")
    args = parser.parse_args()
    if args.retrieval_run and (args.retriever or args.encoding or args.local_files_only):
        parser.error("Replay uses saved retrieval settings; omit retriever, encoding and model-loading options")
    args.retriever = args.retriever or "dense"
    if args.max_context_chars < 1:
        parser.error("Context budget must be positive")
    if args.retriever == "bm25" and (args.encoding or args.local_files_only):
        parser.error("Encoding and model-loading options apply only to dense retrieval")
    inputs = [args.document, args.benchmark] + ([args.source_pdf] if args.source_pdf else [])
    if args.retrieval_run:
        inputs.append(args.retrieval_run)
    if args.output.resolve() in {p.resolve() for p in inputs}:
        parser.error("Output must not overwrite an input")
    document = ProcessedFiling.model_validate_json(args.document.read_text(encoding="utf-8"))
    benchmark = Benchmark.model_validate_json(args.benchmark.read_text(encoding="utf-8"))
    if benchmark.status != "human_reviewed" and not args.allow_draft:
        parser.error("Use --allow-draft for this pending benchmark")
    # Validate source anchors before model loading or output writes.
    bind_benchmark(document, benchmark)
    verification = verify_source_pdf(document, args.source_pdf) if args.source_pdf else {"status": "not_checked"}
    if args.retrieval_run:
        import hashlib
        data = args.retrieval_run.read_bytes()
        result = replay(document, benchmark, json.loads(data), allow_draft=args.allow_draft,
                        max_chars=args.max_context_chars, page_policy=args.page_policy)
        result["replayed_from"] = {"path": args.retrieval_run.resolve().as_posix(),
                                   "sha256": hashlib.sha256(data).hexdigest()}
    else:
        model = load_model(args.local_files_only) if args.retriever != "bm25" else None
        result = evaluate(document, benchmark, model, retriever=args.retriever,
                          encoding=args.encoding or "window-mean", allow_draft=args.allow_draft,
                          max_chars=args.max_context_chars, page_policy=args.page_policy)
        result.update(runtime_metadata() if args.retriever != "bm25" else bm25.runtime_metadata())
    result["source_pdf_verification"] = verification
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["evaluation_status"], "aggregate": result["aggregate"]}))


if __name__ == "__main__":
    main()
