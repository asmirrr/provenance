"""Offline evidence-context diagnostics; this does not measure answer accuracy."""

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.ingestion.pipeline import ProcessedFiling
from src.evidence import resolve_evidence, validate_document
from src.benchmark import Benchmark, bind_benchmark


def evaluate_benchmark(document: ProcessedFiling, benchmark: Benchmark) -> dict:
    """Expand each explicit evidence group into a context case, without retrieval.

    Alternative support paths are separate cases, not independent questions.
    Ground-truth mappings are draft annotations until reviewed by a human.
    """
    bound = bind_benchmark(document, benchmark)
    rows = []
    for item in bound["items"]:
        for group in item["evidence_groups"]:
            anchors = group["evidence"]
            ids = group["supporting_chunk_ids"]
            target = anchors[-1]
            target_ids = target["chunk_ids"]
            def contains(bundle):
                return all(any(s["page"] == a["page"] and s["raw_start"] <= a["raw_start"]
                               and s["raw_end"] >= a["raw_end"] for s in bundle["spans"])
                           for a in anchors)
            try:
                target_complete = contains(resolve_evidence(document, target_ids, context="page", max_chars=8000))
            except ValueError:
                target_complete = False
            try:
                bundle = resolve_evidence(document, ids, context="page", max_chars=8000)
                explicit_complete, error = contains(bundle), None
            except ValueError as exc:
                explicit_complete, error = False, str(exc)
            single = [c.chunk_id for c in document.chunks if all(
                c.page == a["page"] and c.raw_start <= a["raw_start"] and c.raw_end >= a["raw_end"]
                for a in anchors)]
            rows.append({"case_id": f"{item['id']}:{group['group_id']}", "item_id": item["id"],
                         "group_id": group["group_id"], "question": item["question"],
                         "expected_answer": item["expected_answer"], "review_status": item["review_status"],
                         "supporting_pages": group["supporting_pages"], "supporting_chunk_ids": ids,
                         "evidence": anchors, "single_chunk_complete": bool(single),
                         "complete_chunk_ids": single,
                         "target_page_complete": target_complete,
                         "explicit_pages_complete": explicit_complete,
                         "page_context_chars": group["page_context_chars"], "context_error": error})
    # Exclude local source paths so the same PDF/configuration can reproduce this on another machine.
    corpus = {"source_sha256": document.source_sha256, "metadata": document.metadata.model_dump(mode="json"),
              "extractor": document.extractor, "chunking": document.chunking, "max_chars": document.max_chars,
              "pages": [p.model_dump() for p in document.pages],
              "chunks": [c.model_dump(exclude={"source_path"}) for c in document.chunks]}
    return {"diagnostic_version": "2.0.0", "benchmark_version": benchmark.version,
            "benchmark_model_sha256": bound["benchmark_model_sha256"],
            "corpus_sha256": hashlib.sha256(json.dumps(corpus, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "source_sha256": document.source_sha256, "extractor": document.extractor,
            "chunking": document.chunking, "max_chars": document.max_chars,
            "page_count": len(document.pages), "chunk_count": len(document.chunks),
            "benchmark_status": benchmark.status, "question_count": len(benchmark.items),
            "answerable_question_count": bound["answerable_count"], "case_count": len(rows),
            "unsupported_item_ids": [i.id for i in benchmark.items if not i.answerable],
            "single_chunk_complete": sum(r["single_chunk_complete"] for r in rows),
            "target_page_complete": sum(r["target_page_complete"] for r in rows),
            "explicit_pages_complete": sum(r["explicit_pages_complete"] for r in rows),
            "context_max_chars": 8000,
            "interpretation": "Oracle-selected context availability, not retrieval or answer accuracy. Alternative groups are separate cases. Human review pending for draft labels.",
            "results": rows}


class EvidenceCase(BaseModel):
    case_id: str
    description: str
    page: int = Field(ge=1)
    section: str | None
    # Ordered, verbatim fragments selected from the raw extraction. Together
    # these include the units/year/header needed to interpret the target value.
    evidence: list[str] = Field(min_length=1)


class EvidenceCases(BaseModel):
    source_sha256: str
    review_method: str
    cases: list[EvidenceCase] = Field(min_length=1)


def evaluate(document: ProcessedFiling, cases: EvidenceCases) -> dict:
    if document.source_sha256 != cases.source_sha256:
        raise ValueError("Evaluation cases refer to a different source PDF")
    if len({c.case_id for c in cases.cases}) != len(cases.cases):
        raise ValueError("Duplicate evaluation case IDs")
    validate_document(document)
    pages = {p.page: p for p in document.pages}
    results = []
    for case in cases.cases:
        if case.page not in pages:
            raise ValueError(f"Missing PDF page for {case.case_id}")
        raw = pages[case.page].raw_text
        locations = []
        for quote in case.evidence:
            start = raw.find(quote)
            if not quote or start < 0 or raw.find(quote, start + 1) >= 0:
                raise ValueError(f"Evidence must occur exactly once on its page: {case.case_id}")
            locations.append((start, start + len(quote)))
        if locations != sorted(locations) or any(a[1] > b[0] for a, b in zip(locations, locations[1:])):
            raise ValueError(f"Evidence fragments must be ordered and disjoint: {case.case_id}")
        start, end = locations[0][0], locations[-1][1]
        relevant = [c for c in document.chunks if c.page == case.page
                    and c.raw_start < end and c.raw_end > start]
        complete = [c.chunk_id for c in relevant if c.raw_start <= start and c.raw_end >= end]
        # Page completeness is a diagnostic upper bound, not a retrieval score.
        cleaned = pages[case.page].text
        # Seed with the chunk containing the final (target) fragment, then use
        # the actual resolver. Do not relabel page availability as chunk quality.
        target_start, target_end = locations[-1]
        seeds = [c for c in relevant if c.raw_start <= target_start and c.raw_end >= target_end]
        expanded_complete = False
        context_chars = 0
        expansion_error = None
        if seeds:
            try:
                bundle = resolve_evidence(document, [seeds[0].chunk_id], context="page")
                expanded_complete = all(q in bundle["spans"][0]["text"] for q in case.evidence)
                context_chars = bundle["context_chars"]
            except ValueError as error:
                expansion_error = str(error)
        results.append({
            "case_id": case.case_id, "description": case.description,
            "page": case.page, "printed_page": pages[case.page].printed_page,
            "raw_start": start, "raw_end": end,
            "single_chunk_complete": bool(complete), "complete_chunk_ids": complete,
            "overlapping_chunk_ids": [c.chunk_id for c in relevant],
            "page_context_complete": all(q in cleaned for q in case.evidence),
            "section_correct": bool(relevant) and all(c.section == case.section for c in relevant),
            "page_expanded_complete": expanded_complete,
            "expansion_seed_chunk_id": seeds[0].chunk_id if seeds else None,
            "expanded_context_chars": context_chars, "expansion_error": expansion_error,
        })
    return {
        "source_sha256": document.source_sha256, "extractor": document.extractor,
        "chunking": document.chunking,
        "max_chars": document.max_chars, "chunk_count": len(document.chunks),
        "review_method": cases.review_method, "case_count": len(results),
        "single_chunk_complete": sum(r["single_chunk_complete"] for r in results),
        "page_context_complete": sum(r["page_context_complete"] for r in results),
        "section_correct": sum(r["section_correct"] for r in results),
        "context_policy": "page-v1", "context_max_chars": 8000,
        "page_expanded_complete": sum(r["page_expanded_complete"] for r in results),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark", action="store_true", help="Use versioned benchmark evidence groups as context cases")
    args = parser.parse_args()
    if args.output and args.output.resolve() in {args.document.resolve(), args.cases.resolve()}:
        parser.error("Output must not overwrite the document or evaluation cases")
    document = ProcessedFiling.model_validate_json(args.document.read_text(encoding="utf-8"))
    cases = (Benchmark if args.benchmark else EvidenceCases).model_validate_json(args.cases.read_text(encoding="utf-8"))
    report = json.dumps((evaluate_benchmark if args.benchmark else evaluate)(document, cases), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
