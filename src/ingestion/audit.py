"""Compare a persisted corpus with fresh ingestion of its verified source PDF."""

import argparse
import hashlib
import json
from pathlib import Path

from src.evidence import validate_document, verify_source_pdf
from src.ingestion.pipeline import ProcessedFiling, ingest


def audit(document: ProcessedFiling, source_pdf: str | Path) -> dict:
    """Use stored metadata/configuration, but independently extract all PDF pages.

    Matching establishes reproducibility with the current parser, not visual
    correctness or authenticity of user-supplied company/filing metadata.
    """
    verification = verify_source_pdf(document, source_pdf)
    validate_document(document)
    fresh = ingest(source_pdf, document.metadata, max_chars=document.max_chars, chunking=document.chunking)
    validate_document(fresh)
    old_pages = {p.page: p.model_dump() for p in document.pages}
    new_pages = {p.page: p.model_dump() for p in fresh.pages}
    changed_pages = sorted(p for p in old_pages.keys() & new_pages.keys() if old_pages[p] != new_pages[p])
    old_chunks = {c.chunk_id: c.model_dump(exclude={"source_path"}) for c in document.chunks}
    new_chunks = {c.chunk_id: c.model_dump(exclude={"source_path"}) for c in fresh.chunks}
    differences = {
        "missing_pages": sorted(new_pages.keys() - old_pages.keys()),
        "unexpected_pages": sorted(old_pages.keys() - new_pages.keys()),
        "changed_pages": changed_pages,
        "page_order_changed": [p.page for p in document.pages] != [p.page for p in fresh.pages],
        "missing_chunk_ids": sorted(new_chunks.keys() - old_chunks.keys()),
        "unexpected_chunk_ids": sorted(old_chunks.keys() - new_chunks.keys()),
        "changed_chunk_ids": sorted(cid for cid in old_chunks.keys() & new_chunks.keys() if old_chunks[cid] != new_chunks[cid]),
        "warnings_changed": document.warnings != fresh.warnings,
        "extractor_changed": document.extractor != fresh.extractor,
        "schema_version_changed": document.schema_version != fresh.schema_version,
    }
    return {"schema_version": 1, "status": "mismatch" if any(differences.values()) else "matched",
            "source_pdf_verification": verification,
            "document_model_sha256": hashlib.sha256(document.model_dump_json().encode()).hexdigest(),
            "stored_page_count": len(document.pages), "fresh_page_count": len(fresh.pages),
            "stored_chunk_count": len(document.chunks), "fresh_chunk_count": len(fresh.chunks),
            "stored_extractor": document.extractor, "fresh_extractor": fresh.extractor,
            "chunking": document.chunking, "max_chars": document.max_chars, "differences": differences,
            "limitations": "Reproduces current text extraction using supplied metadata; does not verify visual table interpretation, metadata truth, or OCR."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("source_pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    try:
        document = ProcessedFiling.model_validate_json(args.document.read_text(encoding="utf-8"))
        report = audit(document, args.source_pdf)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": report["status"], "pages": report["fresh_page_count"],
                      "chunks": report["fresh_chunk_count"], "output": str(args.output)}))
    if report["status"] != "matched":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
