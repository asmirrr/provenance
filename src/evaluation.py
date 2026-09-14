"""Offline evidence-context diagnostics; this does not measure answer accuracy."""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.ingestion.pipeline import ProcessedFiling
from src.evidence import resolve_evidence, validate_document


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
    args = parser.parse_args()
    if args.output and args.output.resolve() in {args.document.resolve(), args.cases.resolve()}:
        parser.error("Output must not overwrite the document or evaluation cases")
    document = ProcessedFiling.model_validate_json(args.document.read_text(encoding="utf-8"))
    cases = EvidenceCases.model_validate_json(args.cases.read_text(encoding="utf-8"))
    report = json.dumps(evaluate(document, cases), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
