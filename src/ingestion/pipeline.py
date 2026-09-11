"""Page-local ingestion. Offsets cite raw extracted text, not PDF coordinates."""

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pdfplumber
from pydantic import BaseModel, Field

from .pdf_parser import extract_pages
from .schema import Chunk


class FilingMetadata(BaseModel):
    company: str
    ticker: str
    document_type: str = "10-K"
    fiscal_year: int = Field(ge=1900, le=2100)
    filing_date: date
    source_url: str
    sec_url: str


class PageText(BaseModel):
    page: int
    raw_text: str
    text: str
    raw_start: int
    printed_page: int | None = None


class ProcessedFiling(BaseModel):
    schema_version: int = 1
    extractor: str
    max_chars: int
    metadata: FilingMetadata
    source_path: str
    source_sha256: str
    pages: list[PageText]
    chunks: list[Chunk]
    warnings: list[str]


def clean_page(raw: str, metadata: FilingMetadata, page: int) -> PageText:
    # Only remove the observed, anchored report footer. Do not guess at tables,
    # standalone numbers, repeated prose, or word-break hyphens.
    footer = re.search(
        rf"(?m)^{re.escape(metadata.company)}\s*\|\s*"
        rf"{metadata.fiscal_year} Form {re.escape(metadata.document_type)}\s*\|\s*(\d+)\s*\Z",
        raw,
    )
    body = raw[:footer.start()] if footer else raw
    start = len(body) - len(body.lstrip())
    return PageText(page=page, raw_text=raw, text=body.strip(), raw_start=start,
                    printed_page=int(footer[1]) if footer else None)


def spans(text: str, start: int, end: int, max_chars: int):
    """Prefer line boundaries, then whitespace; hard-split only oversized tokens."""
    while start < end:
        while start < end and text[start].isspace():
            start += 1
        if start == end:
            break
        stop = min(start + max_chars, end)
        if stop < end:
            boundary = text.rfind("\n", start + max_chars // 2, stop + 1)
            if boundary < 0:
                boundaries = list(re.finditer(r"\s", text[start:stop + 1]))
                boundary = start + boundaries[-1].start() if boundaries else -1
            if boundary > start:
                stop = boundary
        trimmed = stop
        while trimmed > start and text[trimmed - 1].isspace():
            trimmed -= 1
        if trimmed > start:
            yield start, trimmed
        start = stop


def ingest(pdf_path: str | Path, metadata: FilingMetadata, max_chars: int = 1800) -> ProcessedFiling:
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    path = Path(pdf_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pages = [clean_page(p["text"], metadata, p["page"]) for p in extract_pages(path)]
    if not any(p.text for p in pages):
        raise ValueError("No extractable text; OCR is not supported")
    chunks = []
    section = None
    in_exhibits = False
    for page in pages:
        text = page.text
        # TOC entries are not section boundaries. After signatures, exhibits are
        # deliberately unclassified rather than inheriting the last SEC item.
        headings = [] if "TABLE OF CONTENTS" in text or in_exhibits else list(
            re.finditer(r"(?m)^(?:Item\s+\d{1,2}[A-C]?\.\s+[^\n]+|SIGNATURES)\s*$", text)
        )
        cursor = 0
        segments = []
        for heading in headings:
            segments.append((cursor, heading.start(), section))
            section = heading[0].strip()
            if section == "SIGNATURES":
                in_exhibits = True
            cursor = heading.start()
        segments.append((cursor, len(text), section))
        for start, end, label in segments:
            for left, right in spans(text, start, end, max_chars):
                raw_start, raw_end = page.raw_start + left, page.raw_start + right
                identity = f"{digest}:{page.page}:{raw_start}:{raw_end}"
                chunks.append(Chunk(
                    **metadata.model_dump(mode="json", exclude={"sec_url"}),
                    chunk_id=hashlib.sha256(identity.encode()).hexdigest(),
                    text=text[left:right], section=label, page=page.page,
                    printed_page=page.printed_page, source_path=path.as_posix(),
                    source_sha256=digest, raw_start=raw_start, raw_end=raw_end,
                ))
        if in_exhibits:
            section = None
    return ProcessedFiling(
        extractor=f"pdfplumber {pdfplumber.__version__}", max_chars=max_chars,
        metadata=metadata, source_path=path.as_posix(), source_sha256=digest,
        pages=pages, chunks=chunks,
        warnings=[f"PDF page {p.page} has no extractable text; OCR was not run" for p in pages if not p.text],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-chars", type=int, default=1800)
    args = parser.parse_args()
    metadata = FilingMetadata.model_validate_json(args.metadata.read_text(encoding="utf-8"))
    result = ingest(args.pdf, metadata, args.max_chars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({"pages": len(result.pages), "chunks": len(result.chunks), "warnings": result.warnings}))


if __name__ == "__main__":
    main()
