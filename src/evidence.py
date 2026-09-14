"""Resolve selected chunks into exact, bounded source context without retrieval."""

import argparse
import hashlib
import json
from pathlib import Path

from src.ingestion.pipeline import ProcessedFiling


def validate_document(document: ProcessedFiling) -> None:
    """Check internal citation consistency, not authenticity of the source PDF."""
    pages = {p.page: p for p in document.pages}
    if len(pages) != len(document.pages):
        raise ValueError("Duplicate PDF page numbers")
    if len({c.chunk_id for c in document.chunks}) != len(document.chunks):
        raise ValueError("Duplicate chunk IDs")
    for page in document.pages:
        end = page.raw_start + len(page.text)
        if (page.page < 1 or not 0 <= page.raw_start <= end <= len(page.raw_text)
                or page.raw_text[page.raw_start:end] != page.text):
            raise ValueError(f"Invalid cleaned text citation for page {page.page}")
    expected_metadata = document.metadata.model_dump(mode="json")
    for chunk in document.chunks:
        page = pages.get(chunk.page)
        if (page is None or not 0 <= chunk.raw_start < chunk.raw_end <= len(page.raw_text)
                or page.raw_text[chunk.raw_start:chunk.raw_end] != chunk.text
                or chunk.source_sha256 != document.source_sha256):
            raise ValueError(f"Invalid raw citation for chunk {chunk.chunk_id}")
        expected_id = hashlib.sha256(
            f"{document.source_sha256}:{chunk.page}:{chunk.raw_start}:{chunk.raw_end}".encode()
        ).hexdigest()
        if chunk.chunk_id != expected_id:
            raise ValueError(f"Invalid chunk ID: {chunk.chunk_id}")
        if not page.raw_start <= chunk.raw_start < chunk.raw_end <= page.raw_start + len(page.text):
            raise ValueError(f"Chunk lies outside cleaned page: {chunk.chunk_id}")
        for field in ("company", "ticker", "document_type", "fiscal_year", "filing_date", "source_url"):
            if getattr(chunk, field) != expected_metadata[field]:
                raise ValueError(f"Chunk metadata mismatch: {field}")
        if chunk.source_path != document.source_path or chunk.printed_page != page.printed_page:
            raise ValueError("Chunk source path or printed page mismatch")


def resolve_evidence(document: ProcessedFiling, chunk_ids: list[str], *,
                     context: str = "chunk", max_chars: int = 8000) -> dict:
    """Explicit page expansion never infers that an adjacent page continues a table.

    The budget counts the returned text once per span. Oversized evidence fails
    rather than silently truncating a heading, unit, year, or financial row.
    """
    if context not in {"chunk", "page"} or max_chars < 1:
        raise ValueError("Use context 'chunk' or 'page' and a positive character budget")
    if not chunk_ids:
        raise ValueError("Select at least one chunk")
    validate_document(document)
    chunks = {c.chunk_id: c for c in document.chunks}
    selected = list(dict.fromkeys(chunk_ids))
    missing = set(selected) - chunks.keys()
    if missing:
        raise ValueError(f"Unknown chunk IDs: {sorted(missing)}")
    pages = {p.page: p for p in document.pages}
    spans = []
    seen_pages = set()
    for chunk_id in selected:
        chunk = chunks[chunk_id]
        page = pages[chunk.page]
        if context == "page":
            if page.page in seen_pages:
                continue
            seen_pages.add(page.page)
            text, start = page.text, page.raw_start
        else:
            text, start = chunk.text, chunk.raw_start
        spans.append({
            "page": page.page, "printed_page": page.printed_page,
            "raw_start": start, "raw_end": start + len(text), "text": text,
            # Page context can cross SEC Item boundaries, so do not inherit the
            # selected chunk's section as if it described the entire page.
            "section": chunk.section if context == "chunk" else None,
            "selected_chunk_ids": [cid for cid in selected if chunks[cid].page == page.page]
            if context == "page" else [chunk_id],
        })
    total = sum(len(span["text"]) for span in spans)
    if total > max_chars:
        raise ValueError(f"Evidence needs {total} characters; budget is {max_chars}. Nothing was truncated.")
    return {
        "context_policy": f"{context}-v1", "source_sha256": document.source_sha256,
        "source_path": document.source_path, "metadata": document.metadata.model_dump(mode="json"),
        "selected_chunk_ids": selected, "context_chars": total, "max_chars": max_chars,
        "spans": spans,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("chunk_ids", nargs="+")
    parser.add_argument("--context", choices=["chunk", "page"], default="chunk")
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.resolve() == args.document.resolve():
        parser.error("Output must not overwrite the processed document")
    document = ProcessedFiling.model_validate_json(args.document.read_text(encoding="utf-8"))
    result = resolve_evidence(document, args.chunk_ids, context=args.context, max_chars=args.max_chars)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
