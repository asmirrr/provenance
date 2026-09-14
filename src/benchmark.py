"""Bind a draft question set to exact evidence and produce a human review packet."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.evidence import resolve_evidence, validate_document
from src.ingestion.pipeline import ProcessedFiling


class EvidenceAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)


class BenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: Literal["factual", "numerical", "comparative", "table", "temporal",
                      "section_specific", "multi_hop", "unsupported", "contradiction", "abstention"]
    answerable: bool
    expected_answer: str | None
    expected_claims: list[str]
    difficulty: Literal["easy", "medium", "hard"]
    notes: str = Field(min_length=1)
    evidence: list[EvidenceAnchor]
    review_status: Literal["pending", "human_reviewed"] = "pending"
    reviewer: str | None = None
    reviewed_on: date | None = None

    @model_validator(mode="after")
    def consistent_labels(self):
        if self.answerable:
            if not self.expected_answer or not self.expected_answer.strip() or not self.evidence or not self.expected_claims:
                raise ValueError("Answerable items require an answer, claims and evidence")
            if any(not claim.strip() for claim in self.expected_claims):
                raise ValueError("Expected claims must not be blank")
        elif self.expected_answer is not None or self.expected_claims or self.evidence:
            raise ValueError("Unanswerable items must not carry an answer, claims or supporting evidence")
        if self.review_status == "human_reviewed" and (
                not self.reviewer or not self.reviewer.strip() or self.reviewed_on is None):
            raise ValueError("Human review requires a named reviewer and review date")
        return self


class Benchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    status: Literal["draft_pending_human_review", "human_reviewed"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_method: str
    items: list[BenchmarkItem] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_review(self):
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("Duplicate benchmark item IDs")
        if self.status == "human_reviewed" and any(item.review_status != "human_reviewed" for item in self.items):
            raise ValueError("A reviewed benchmark cannot contain pending items")
        return self


def bind_benchmark(document: ProcessedFiling, benchmark: Benchmark) -> dict:
    """Resolve quote anchors anew when chunking changes; never reuse stale IDs."""
    if benchmark.source_sha256 != document.source_sha256:
        raise ValueError("Benchmark source PDF digest does not match document")
    validate_document(document)
    pages = {p.page: p for p in document.pages}
    rows = []
    for item in benchmark.items:
        anchors = []
        supporting_ids = []
        previous = (0, 0)
        for anchor in item.evidence:
            if anchor.page not in pages:
                raise ValueError(f"Missing evidence page: {item.id}")
            page = pages[anchor.page]
            start = page.raw_text.find(anchor.quote)
            if not anchor.quote.strip() or start < 0 or page.raw_text.find(anchor.quote, start + 1) >= 0:
                raise ValueError(f"Quote must occur exactly once on page {anchor.page}: {item.id}")
            end = start + len(anchor.quote)
            if (anchor.page, start) < previous:
                raise ValueError(f"Evidence must be ordered and disjoint: {item.id}")
            previous = (anchor.page, end)
            matches = sorted((c for c in document.chunks if c.page == anchor.page
                              and c.raw_start < end and c.raw_end > start), key=lambda c: c.raw_start)
            # Every non-whitespace character must be covered, even when a quote
            # straddles chunks. An overlap alone is not a complete citation.
            cursor = start
            for chunk in matches:
                if page.raw_text[cursor:min(chunk.raw_start, end)].strip():
                    raise ValueError(f"Evidence is missing from chunks: {item.id}")
                cursor = max(cursor, min(chunk.raw_end, end))
            if not matches or page.raw_text[cursor:end].strip():
                raise ValueError(f"Evidence is missing from chunks: {item.id}")
            ids = [c.chunk_id for c in matches]
            supporting_ids.extend(ids)
            anchors.append({**anchor.model_dump(), "printed_page": page.printed_page,
                            "raw_start": start, "raw_end": end, "chunk_ids": ids,
                            "sections": list(dict.fromkeys(c.section for c in matches))})
        supporting_ids = list(dict.fromkeys(supporting_ids))
        context_chars = 0
        context_error = None
        if supporting_ids:
            try:
                context_chars = resolve_evidence(document, supporting_ids, context="page")["context_chars"]
            except ValueError as error:
                context_chars = None
                context_error = str(error)
        rows.append({**item.model_dump(mode="json"), "evidence": anchors,
                     "supporting_pages": list(dict.fromkeys(a["page"] for a in anchors)),
                     "supporting_chunk_ids": supporting_ids,
                     "page_context_chars": context_chars, "page_context_error": context_error})
    return {
        "version": benchmark.version, "status": benchmark.status,
        "ready_for_scored_evaluation": benchmark.status == "human_reviewed",
        "review_method": benchmark.review_method,
        "source_sha256": document.source_sha256, "source_path": document.source_path,
        "source_url": document.metadata.source_url, "sec_url": document.metadata.sec_url,
        # Hash canonical model JSON, not arbitrary input indentation or newlines.
        "benchmark_model_sha256": hashlib.sha256(benchmark.model_dump_json().encode()).hexdigest(),
        "document_model_sha256": hashlib.sha256(document.model_dump_json().encode()).hexdigest(),
        "chunking": document.chunking, "max_chars": document.max_chars, "extractor": document.extractor,
        "item_count": len(rows), "answerable_count": sum(i.answerable for i in benchmark.items),
        "pending_review_count": sum(i.review_status == "pending" for i in benchmark.items),
        "items": rows,
    }


def review_markdown(bound: dict) -> str:
    lines = [f"# Provenance benchmark {bound['version']} — review packet", "",
             f"Status: **{bound['status']}**. Pending: {bound['pending_review_count']} / {bound['item_count']}.",
             "", bound["review_method"], "",
             f"[Source PDF]({bound['source_url']}) · [SEC filing]({bound['sec_url']})", "",
             f"Source SHA-256: `{bound['source_sha256']}`", "",
             f"Chunking: {bound['chunking']}, {bound['max_chars']} characters.", "",
             "For each item, check the original PDF, answer, units, fiscal period, evidence sufficiency,",
             "alternative supporting passages and answerability. Unsupported items require an absence/scope",
             "review of the filing; no quote match can establish absence. Record human review in the",
             "source benchmark JSON using review_status, reviewer and reviewed_on; this packet is generated.", "",
             "Chunk IDs below are mechanical mappings of the selected quotes, not adjudicated relevance labels.", ""]
    for item in bound["items"]:
        lines.extend([f"## {item['id']}: {item['question']}", "",
                      f"Category: {item['category']} | Difficulty: {item['difficulty']} | Review: {item['review_status']}", "",
                      f"Expected answer: {item['expected_answer'] if item['answerable'] else 'Abstain — unsupported by this corpus.'}", "",
                      f"Notes: {item['notes']}", ""])
        for claim in item["expected_claims"]:
            lines.extend([f"- Expected claim: {claim}"])
        lines.append("")
        for anchor in item["evidence"]:
            lines.extend([f"PDF page {anchor['page']} (printed {anchor['printed_page']}), raw characters "
                          f"{anchor['raw_start']}:{anchor['raw_end']}.", "",
                          *["> " + line for line in anchor["quote"].splitlines()], "",
                          "Chunk IDs: " + ", ".join(f"`{cid}`" for cid in anchor["chunk_ids"]), ""])
        if item["page_context_error"]:
            lines.extend([f"Context limitation: {item['page_context_error']}", ""])
        lines.extend(["Human review notes: _pending_" if item["review_status"] == "pending"
                      else f"Reviewed by {item['reviewer']} on {item['reviewed_on']}.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    inputs = {args.document.resolve(), args.benchmark.resolve()}
    outputs = {args.output.resolve(), args.review.resolve()}
    if inputs & outputs or len(outputs) != 2:
        parser.error("Outputs must be distinct and must not overwrite inputs")
    document = ProcessedFiling.model_validate_json(args.document.read_text(encoding="utf-8"))
    benchmark = Benchmark.model_validate_json(args.benchmark.read_text(encoding="utf-8"))
    bound = bind_benchmark(document, benchmark)
    for path, content in ((args.output, json.dumps(bound, ensure_ascii=False, indent=2)),
                          (args.review, review_markdown(bound))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(json.dumps({k: bound[k] for k in ("version", "item_count", "answerable_count",
                                          "pending_review_count", "ready_for_scored_evaluation")}))


if __name__ == "__main__":
    main()
