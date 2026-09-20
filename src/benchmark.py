"""Bind a draft question set to exact evidence and produce a human review packet."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.evidence import resolve_evidence, validate_document, verify_source_pdf
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
    # AND within each group, OR across primary and alternative groups.
    alternative_evidence: list[list[EvidenceAnchor]] = Field(default_factory=list)
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
        elif self.expected_answer is not None or self.expected_claims or self.evidence or self.alternative_evidence:
            raise ValueError("Unanswerable items must not carry an answer, claims or supporting evidence")
        groups = [self.evidence, *self.alternative_evidence] if self.answerable else []
        identities = [tuple((a.page, a.quote) for a in group) for group in groups]
        if any(not group for group in groups) or len(set(identities)) != len(identities):
            raise ValueError("Evidence groups must be nonempty and distinct")
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


class AIReviewFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: str = Field(min_length=1)
    verdict: Literal["confirmed", "correction_required", "inconclusive"]
    pages: list[int] = Field(min_length=1)
    findings: str = Field(min_length=1)


class AIReview(BaseModel):
    """Separate provenance record; never changes human review or benchmark labels."""
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    review_type: Literal["ai"]
    reviewer: str = Field(min_length=1)
    reviewed_on: date
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    method: str = Field(min_length=1)
    items: list[AIReviewFinding] = Field(min_length=1)


def validate_ai_review(bound: dict, review: AIReview) -> None:
    if review.source_sha256 != bound["source_sha256"] or review.benchmark_model_sha256 != bound["benchmark_model_sha256"]:
        raise ValueError("AI review is stale or belongs to another source/benchmark")
    items = {i["id"]: i for i in bound["items"]}
    seen = set()
    if not review.reviewer.strip() or not review.method.strip():
        raise ValueError("AI review requires a reviewer and method")
    for finding in review.items:
        if finding.item_id not in items or finding.item_id in seen:
            raise ValueError("AI review requires unique known item IDs")
        seen.add(finding.item_id)
        if not finding.findings.strip() or len(set(finding.pages)) != len(finding.pages):
            raise ValueError("AI review requires findings and unique inspected pages")
        # Inspected pages may extend beyond selected evidence (e.g. absence review),
        # but they must be real source pages. Binding lists source page count below.
        if any(p < 1 or p > bound["page_count"] for p in finding.pages):
            raise ValueError("AI review cites a nonexistent source page")


def _bind_group(document: ProcessedFiling, evidence: list[EvidenceAnchor], item_id: str) -> dict:
    """Resolve one jointly required evidence set, independently of alternatives."""
    pages = {p.page: p for p in document.pages}
    anchors, supporting_ids = [], []
    previous = (0, 0)
    for anchor in evidence:
        if anchor.page not in pages:
            raise ValueError(f"Missing evidence page: {item_id}")
        page = pages[anchor.page]
        start = page.raw_text.find(anchor.quote)
        if not anchor.quote.strip() or start < 0 or page.raw_text.find(anchor.quote, start + 1) >= 0:
            raise ValueError(f"Quote must occur exactly once on page {anchor.page}: {item_id}")
        end = start + len(anchor.quote)
        if (anchor.page, start) < previous:
            raise ValueError(f"Evidence must be ordered and disjoint: {item_id}")
        previous = (anchor.page, end)
        matches = sorted((c for c in document.chunks if c.page == anchor.page
                          and c.raw_start < end and c.raw_end > start), key=lambda c: c.raw_start)
        cursor = start
        for chunk in matches:
            if page.raw_text[cursor:min(chunk.raw_start, end)].strip():
                raise ValueError(f"Evidence is missing from chunks: {item_id}")
            cursor = max(cursor, min(chunk.raw_end, end))
        if not matches or page.raw_text[cursor:end].strip():
            raise ValueError(f"Evidence is missing from chunks: {item_id}")
        ids = [c.chunk_id for c in matches]
        supporting_ids.extend(ids)
        anchors.append({**anchor.model_dump(), "printed_page": page.printed_page,
                        "raw_start": start, "raw_end": end, "chunk_ids": ids,
                        "sections": list(dict.fromkeys(c.section for c in matches))})
    supporting_ids = list(dict.fromkeys(supporting_ids))
    context_chars, context_error = 0, None
    if supporting_ids:
        try:
            context_chars = resolve_evidence(document, supporting_ids, context="page")["context_chars"]
        except ValueError as error:
            context_chars, context_error = None, str(error)
    return {"evidence": anchors,
            "supporting_pages": list(dict.fromkeys(a["page"] for a in anchors)),
            "supporting_chunk_ids": supporting_ids,
            "page_context_chars": context_chars, "page_context_error": context_error}


def bind_benchmark(document: ProcessedFiling, benchmark: Benchmark) -> dict:
    """Resolve quote anchors anew when chunking changes; never reuse stale IDs."""
    if benchmark.source_sha256 != document.source_sha256:
        raise ValueError("Benchmark source PDF digest does not match document")
    validate_document(document)
    rows = []
    for item in benchmark.items:
        primary = _bind_group(document, item.evidence, item.id)
        groups = [{"group_id": "primary", **primary}] if item.answerable else []
        groups.extend({"group_id": f"alternative-{i}", **_bind_group(document, group, item.id)}
                      for i, group in enumerate(item.alternative_evidence, 1))
        rows.append({**item.model_dump(mode="json", exclude={"alternative_evidence"}),
                     **primary, "evidence_groups": groups})
    categories = {}
    for category in sorted({item.category for item in benchmark.items}):
        items = [item for item in benchmark.items if item.category == category]
        categories[category] = {
            "total": len(items),
            "reviewed": sum(item.review_status == "human_reviewed" for item in items),
            "pending": sum(item.review_status == "pending" for item in items),
            "unsupported": sum(not item.answerable for item in items),
        }
    pending = [item for item in benchmark.items if item.review_status == "pending"]
    # Absence/scope labels need explicit attention; keep original order within
    # answerability groups. This is a review convenience, not a difficulty score.
    queue = [item.id for item in sorted(pending, key=lambda item: item.answerable)]
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
        "page_count": len(document.pages),
        "pending_review_count": sum(i.review_status == "pending" for i in benchmark.items),
        "evidence_semantics": "any_complete_group",
        "items_with_alternatives": sum(bool(i.alternative_evidence) for i in benchmark.items),
        "review_progress": {"by_category": categories, "pending_item_ids": queue},
        "items": rows,
    }


def complete_evidence_groups(item: dict, selected_chunk_ids: list[str]) -> list[str]:
    """Diagnostic raw-chunk coverage, not answer correctness or a benchmark score.

    A whole group must be present. Partial evidence from unrelated alternatives
    is never combined; page-context expansion does not count as chunk retrieval.
    """
    selected = set(selected_chunk_ids)
    return [group["group_id"] for group in item["evidence_groups"]
            if group["supporting_chunk_ids"] and set(group["supporting_chunk_ids"]) <= selected]


def review_markdown(bound: dict, *, item_ids: list[str] | None = None,
                    limit: int | None = None, ai_review: AIReview | None = None) -> str:
    """Export a review batch without changing labels or the full bound artifact."""
    if item_ids is not None and limit is not None:
        raise ValueError("Choose explicit review items or a pending batch limit, not both")
    items = bound["items"]
    ai_findings = {}
    if ai_review is not None:
        validate_ai_review(bound, ai_review)
        ai_findings = {i.item_id: i for i in ai_review.items}
    if item_ids is not None:
        by_id = {item["id"]: item for item in items}
        if not item_ids or len(set(item_ids)) != len(item_ids) or any(i not in by_id for i in item_ids):
            raise ValueError("Review item IDs must be nonempty, unique and known")
        items = [by_id[i] for i in item_ids]
    elif limit is not None:
        if limit < 1:
            raise ValueError("Review limit must be positive")
        # Absence labels first, then cross-page support, preserving source order within ties.
        pending = [i for i in items if i["review_status"] == "pending"]
        items = sorted(pending, key=lambda i: (
            i["answerable"], not any(len(g["supporting_pages"]) > 1 for g in i["evidence_groups"])))[:limit]
    lines = [f"# Provenance benchmark {bound['version']} — review packet", "",
             f"Status: **{bound['status']}**. Pending: {bound['pending_review_count']} / {bound['item_count']}.",
             "", bound["review_method"], "",
             f"[Source PDF]({bound['source_url']}) · [SEC filing]({bound['sec_url']})", "",
             f"Source SHA-256: `{bound['source_sha256']}`", "",
             f"Benchmark model SHA-256: `{bound['benchmark_model_sha256']}`", "",
             f"Document model SHA-256: `{bound['document_model_sha256']}`", "",
             f"This packet includes {len(items)} of {bound['item_count']} questions; counts below cover the full benchmark.", "",
             f"Chunking: {bound['chunking']}, {bound['max_chars']} characters.", "",
             "For each item, check the original PDF, answer, units, fiscal period, evidence sufficiency,",
             "alternative supporting passages and answerability. Unsupported items require an absence/scope",
             "review of the filing; no quote match can establish absence. Record human review in the",
             "source benchmark JSON using review_status, reviewer and reviewed_on; this packet is generated.", "",
             "Chunk IDs below are mechanical mappings of the selected quotes, not adjudicated relevance labels.", ""]
    progress = bound["review_progress"]
    lines.extend(["## Review progress", "",
                  "| Category | Total | Reviewed | Pending | Unsupported |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for category, counts in progress["by_category"].items():
        lines.append(f"| {category} | {counts['total']} | {counts['reviewed']} | "
                     f"{counts['pending']} | {counts['unsupported']} |")
    lines.extend(["", "Pending queue (unsupported questions first, for absence/scope review):", "",
                  ", ".join(f"`{item_id}`" for item_id in progress["pending_item_ids"])
                  or "No pending items. Overall benchmark status must still be finalized explicitly.", ""])
    if not items:
        lines.extend(["No pending items remain for this batch. No review status was changed.", ""])
    for item in items:
        lines.extend([f"## {item['id']}: {item['question']}", "",
                      f"Category: {item['category']} | Difficulty: {item['difficulty']} | Review: {item['review_status']}", "",
                      f"Expected answer: {item['expected_answer'] if item['answerable'] else 'Abstain — unsupported by this corpus.'}", "",
                      f"Notes: {item['notes']}", ""])
        if item["id"] in ai_findings:
            finding = ai_findings[item["id"]]
            lines.extend([f"AI review only: **{finding.verdict}** by {ai_review.reviewer} on {ai_review.reviewed_on}.",
                          "Human review status is unchanged.", "", finding.findings, "",
                          f"Inspected PDF pages: {', '.join(map(str, finding.pages))}.", ""])
        for claim in item["expected_claims"]:
            lines.extend([f"- Expected claim: {claim}"])
        lines.append("")
        for group in item["evidence_groups"]:
            lines.extend([f"### Evidence group: {group['group_id']}", "",
                          "All excerpts below are jointly required; another complete group is an alternative.", ""])
            for anchor in group["evidence"]:
                printed = anchor['printed_page'] if anchor['printed_page'] is not None else 'unknown'
                page_url = bound['source_url'].split('#', 1)[0] + f"#page={anchor['page']}"
                lines.extend([f"[PDF page {anchor['page']}]({page_url}) (printed {printed}), raw characters "
                              f"{anchor['raw_start']}:{anchor['raw_end']}.", "",
                              *["> " + line for line in anchor["quote"].splitlines()], "",
                              "Chunk IDs: " + ", ".join(f"`{cid}`" for cid in anchor["chunk_ids"]), ""])
            if group["page_context_error"]:
                lines.extend([f"Context limitation: {group['page_context_error']}", ""])
        lines.extend(["Review checklist: answer and units; fiscal period; all required evidence; alternative groups; scope/absence if unsupported.", "",
                      "Human review notes: _pending_" if item["review_status"] == "pending"
                      else f"Reviewed by {item['reviewer']} on {item['reviewed_on']}.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--source-pdf", type=Path, help="Verify the original PDF's SHA-256 before export")
    batch = parser.add_mutually_exclusive_group()
    batch.add_argument("--review-item", action="append", help="Export only this item in Markdown; repeat for a batch")
    batch.add_argument("--review-limit", type=int, help="Export up to N pending questions, unsupported then cross-page first")
    parser.add_argument("--ai-review", type=Path, help="Attach a separately recorded AI review without changing labels")
    args = parser.parse_args()
    inputs = {args.document.resolve(), args.benchmark.resolve()}
    if args.source_pdf:
        inputs.add(args.source_pdf.resolve())
    if args.ai_review:
        inputs.add(args.ai_review.resolve())
    outputs = {args.output.resolve(), args.review.resolve()}
    if inputs & outputs or len(outputs) != 2:
        parser.error("Outputs must be distinct and must not overwrite inputs")
    document = ProcessedFiling.model_validate_json(args.document.read_text(encoding="utf-8"))
    verification = verify_source_pdf(document, args.source_pdf) if args.source_pdf else {"status": "not_checked"}
    benchmark = Benchmark.model_validate_json(args.benchmark.read_text(encoding="utf-8"))
    bound = bind_benchmark(document, benchmark)
    bound["source_pdf_verification"] = verification
    try:
        ai_review = AIReview.model_validate_json(args.ai_review.read_text(encoding="utf-8")) if args.ai_review else None
        review = review_markdown(bound, item_ids=args.review_item, limit=args.review_limit, ai_review=ai_review)
    except ValueError as exc:
        parser.error(str(exc))
    for path, content in ((args.output, json.dumps(bound, ensure_ascii=False, indent=2)),
                          (args.review, review)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(json.dumps({k: bound[k] for k in ("version", "item_count", "answerable_count",
                                          "items_with_alternatives", "pending_review_count",
                                          "ready_for_scored_evaluation")}))


if __name__ == "__main__":
    main()
