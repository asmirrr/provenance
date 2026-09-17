"""Structured answer contract and citation integrity checks; no claim verification."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.evidence import select_chunk_context, select_page_context, validate_document, verify_source_pdf
from src.ingestion.pipeline import ProcessedFiling

Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    page: int = Field(ge=1)
    raw_start: int = Field(ge=0)
    raw_end: int = Field(gt=0)
    quote: Text


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: Text
    citations: list[Citation] = Field(min_length=1)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    query_sha256: Digest
    status: Literal["answered", "abstained"]
    claims: list[Claim]
    abstention_reason: Text | None = None

    @model_validator(mode="after")
    def check_status(self):
        if self.status == "answered" and (not self.claims or self.abstention_reason is not None):
            raise ValueError("Answered output requires claims and no abstention reason")
        if self.status == "abstained" and (self.claims or self.abstention_reason is None):
            raise ValueError("Abstained output requires a reason and no claims")
        return self


def query_digest(run: dict) -> str:
    """Bind the response to the complete query artifact, independent of JSON layout."""
    return hashlib.sha256(json.dumps(run, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_query(document: ProcessedFiling, run: dict) -> dict:
    """Reconstruct delivered context against the supplied processed document."""
    validate_document(document)
    try:
        if run["schema_version"] != 1 or run["kind"] != "retrieved_evidence":
            raise ValueError("Unsupported query artifact")
        retrieval = run["retrieval"]
        digest = hashlib.sha256(document.model_dump_json().encode()).hexdigest()
        if (retrieval["document_model_sha256"] != digest or
                retrieval["source_sha256"] != document.source_sha256 or
                retrieval["question"] != run["question"] or not run["question"].strip()):
            raise ValueError("Query corpus or question mismatch")
        chunks = {c.chunk_id: c.model_dump(mode="json") for c in document.chunks}
        depth = retrieval["requested_top_k"]
        if type(depth) is not int or not 1 <= depth <= 10 or len(retrieval["results"]) > depth:
            raise ValueError("Invalid candidate depth")
        ids = []
        for rank, row in enumerate(retrieval["results"], 1):
            cid = row["chunk"]["chunk_id"]
            if row["rank"] != rank or row["chunk"] != chunks.get(cid):
                raise ValueError("Invalid retrieved citation or rank")
            ids.append(cid)
        context = run["context"]
        if context not in {"chunk", "page"}:
            raise ValueError("Unknown context policy")
        selector = select_chunk_context if context == "chunk" else select_page_context
        selection = selector(document, ids, max_chars=run["selection"]["max_chars"])
        if selection != run["selection"]:
            raise ValueError("Delivered context does not match ranked selection")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed query artifact") from exc
    return selection


def validate_answer(document: ProcessedFiling, run: dict, answer: Answer) -> dict:
    """Check citation integrity, not semantic support or source authenticity."""
    # Revalidate instances constructed/copied without Pydantic validation.
    answer = Answer.model_validate(answer.model_dump())
    if answer.query_sha256 != query_digest(run):
        raise ValueError("Answer belongs to a different query artifact")
    selection = validate_query(document, run)
    spans = selection["bundle"]["spans"] if selection["bundle"] else []
    count = 0
    for claim in answer.claims:
        seen = set()
        for citation in claim.citations:
            key = (citation.page, citation.raw_start, citation.raw_end)
            if key in seen:
                raise ValueError("Duplicate citation within a claim")
            seen.add(key)
            if not any(
                span["page"] == citation.page and
                span["raw_start"] <= citation.raw_start < citation.raw_end <= span["raw_end"] and
                span["text"][citation.raw_start - span["raw_start"]:
                             citation.raw_end - span["raw_start"]] == citation.quote
                for span in spans
            ):
                raise ValueError("Citation must quote an exact range within one delivered span")
            count += 1
    return {"schema_version": 1, "answer": answer.model_dump(mode="json"),
            "citation_integrity": "passed", "checked_citations": count,
            "claim_support": "not_checked", "answer_completeness": "not_checked",
            "abstention_correctness": "not_checked", "source_sha256": document.source_sha256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("query_run", type=Path)
    parser.add_argument("answer", type=Path)
    parser.add_argument("--source-pdf", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = [args.document, args.query_run, args.answer] + ([args.source_pdf] if args.source_pdf else [])
    if any(args.output.resolve() == p.resolve() or
           (args.output.exists() and p.exists() and args.output.samefile(p)) for p in inputs):
        parser.error("Output must not overwrite an input")
    try:
        document = ProcessedFiling.model_validate_json(args.document.read_text(encoding="utf-8"))
        verification = verify_source_pdf(document, args.source_pdf) if args.source_pdf else {"status": "not_checked"}
        run = json.loads(args.query_run.read_text(encoding="utf-8"))
        answer = Answer.model_validate_json(args.answer.read_text(encoding="utf-8"))
        result = validate_answer(document, run, answer)
        result["source_pdf_verification"] = verification
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"citation_integrity": "passed", "checked_citations": result["checked_citations"],
                      "claim_support": "not_checked", "output": str(args.output)}))


if __name__ == "__main__":
    main()
