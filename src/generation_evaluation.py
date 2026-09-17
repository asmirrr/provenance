"""Offline diagnostics for saved generation runs; never calls a provider or scores truth."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from src.answer import Answer, query_digest, validate_answer, validate_query
from src.benchmark import Benchmark, bind_benchmark
from src.evidence import verify_source_pdf
from src.generation import build_request, validate_response
from src.ingestion.pipeline import ProcessedFiling


def evaluate(document, benchmark, runs, *, allow_draft=False):
    """runs contains (benchmark item ID, query artifact, generation artifact) tuples.

    Repeated IDs are rejected so repeated samples cannot silently change denominators.
    Saved success flags are never used to establish citation validity.
    """
    if benchmark.status != "human_reviewed" and not allow_draft:
        raise ValueError("Draft benchmark requires --allow-draft; results are diagnostics only")
    bound = bind_benchmark(document, benchmark)
    items = {item.id: item for item in benchmark.items}
    seen, rows = set(), []
    for item_id, query, generation in runs:
        if item_id not in items or item_id in seen:
            raise ValueError("Require unique known benchmark item IDs")
        seen.add(item_id)
        item = items[item_id]
        selection = validate_query(document, query)
        if query["question"] != item.question:
            raise ValueError(f"Query question mismatch: {item_id}")
        if (generation.get("schema_version") != 1 or generation.get("provider") != "anthropic" or
                generation.get("prompt_version") != "evidence-only-v1" or
                generation.get("query_sha256") != query_digest(query)):
            raise ValueError(f"Generation binding/version mismatch: {item_id}")
        request = generation.get("request", {})
        expected = build_request(document, query, model=request.get("model", ""),
                                 max_tokens=request.get("max_tokens", 0))
        if request != expected:
            raise ValueError(f"Saved request differs from delivered evidence/prompt: {item_id}")
        status = generation.get("status")
        raw = generation.get("raw_response")
        validation, failure = None, None
        if raw is not None:
            if status not in {"citation_validated", "invalid_response"}:
                raise ValueError("Raw response has inconsistent generation status")
            try:
                validation = validate_response(document, query, raw)
                outcome = "citation_validated"
            except ValueError as exc:
                outcome, failure = "invalid_response", str(exc)
        elif status == "local_abstention":
            if selection["bundle"] is not None:
                raise ValueError("Local abstention requires empty delivered evidence")
            abstention = Answer(schema_version=1, query_sha256=query_digest(query), status="abstained",
                                claims=[], abstention_reason="No evidence was delivered for this query.")
            validation = validate_answer(document, query, abstention)
            outcome = "local_abstention"
        elif status in {"dry_run", "provider_error"}:
            outcome = status
        else:
            raise ValueError("Missing raw response for generation outcome")
        answer = validation["answer"] if validation else None
        rows.append({"item_id": item_id, "question": item.question, "category": item.category,
                     "expected_answerable": item.answerable, "review_status": item.review_status,
                     "outcome": outcome, "saved_status": status, "error": failure,
                     "answer_status": answer["status"] if answer else None,
                     "validation": validation, "model": request["model"],
                     "query_sha256": query_digest(query), "selection_status": selection["status"],
                     "human_review": "pending"})
    if not rows:
        raise ValueError("Provide at least one saved run")
    valid = [row for row in rows if row["validation"] is not None]
    behavior = {label: {"selected": sum(r["expected_answerable"] == expected for r in rows),
                       "answered": sum(r["expected_answerable"] == expected and r["answer_status"] == "answered" for r in valid),
                       "abstained": sum(r["expected_answerable"] == expected and r["answer_status"] == "abstained" for r in valid)}
                for label, expected in (("answerable", True), ("unsupported", False))}
    return {"schema_version": 1, "kind": "generation_diagnostics", "benchmark_status": benchmark.status,
            "benchmark_sha256": hashlib.sha256(benchmark.model_dump_json().encode()).hexdigest(),
            "source_sha256": document.source_sha256, "pending_benchmark_review": bound["pending_review_count"],
            "selected_count": len(rows), "benchmark_count": len(items),
            "unevaluated_item_ids": [key for key in items if key not in seen],
            "outcomes": dict(Counter(r["outcome"] for r in rows)), "response_behavior": behavior,
            "claim_accuracy": "not_scored", "abstention_accuracy": "not_scored", "items": rows,
            "interpretation": "Mechanical integrity and response behavior only. Failures/dry runs are not abstentions. Saved artifacts are not authenticated provider records."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("manifest", type=Path, help="JSON list of item_id, query, generation paths; paths relative to manifest")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--source-pdf", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    try:
        document = ProcessedFiling.model_validate_json(args.document.read_text(encoding="utf-8"))
        benchmark = Benchmark.model_validate_json(args.benchmark.read_text(encoding="utf-8"))
        verification = verify_source_pdf(document, args.source_pdf) if args.source_pdf else {"status": "not_checked"}
        entries = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(entries, list) or any(not isinstance(e, dict) or set(e) != {"item_id", "query", "generation"} for e in entries):
            raise ValueError("Manifest must list item_id, query and generation for each run")
        runs, sources = [], []
        for entry in entries:
            paths = [args.manifest.parent / entry[key] for key in ("query", "generation")]
            data = [p.read_bytes() for p in paths]
            runs.append((entry["item_id"], *[json.loads(raw) for raw in data]))
            sources.append({"item_id": entry["item_id"], **{
                key: {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
                for key, path, raw in zip(("query", "generation"), paths, data)}})
        report = evaluate(document, benchmark, runs, allow_draft=args.allow_draft)
        report.update(source_pdf_verification=verification, artifacts=sources)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"selected": report["selected_count"], "outcomes": report["outcomes"],
                      "claim_accuracy": "not_scored", "output": str(args.output)}))


if __name__ == "__main__":
    main()
