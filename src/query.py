"""Retrieve and export bounded cited evidence for one question, without a benchmark."""

import argparse
import json
from pathlib import Path

from src import bm25, dense, hybrid
from src.evidence import select_chunk_context, select_page_context, validate_document, verify_source_pdf
from src.ingestion.pipeline import ProcessedFiling


def query(document, question, *, retriever="bm25", context="chunk", top_k=10,
          max_chars=8000, encoding=None, local_files_only=False, source_pdf=None):
    """Keep candidate rankings separate from context actually delivered under budget."""
    if not question.strip() or not 1 <= top_k <= 10 or max_chars < 1:
        raise ValueError("Require a nonblank question, top_k between 1 and 10, and positive budget")
    if retriever not in {"bm25", "dense", "hybrid"} or context not in {"chunk", "page"}:
        raise ValueError("Unknown retriever or context policy")
    if retriever == "bm25":
        if encoding is not None or local_files_only:
            raise ValueError("Encoding and model-cache options apply only to dense or hybrid retrieval")
    elif encoding not in {"reject", "prefix", "window-mean"}:
        raise ValueError("Dense and hybrid retrieval require an explicit encoding policy")
    validate_document(document)
    verification = verify_source_pdf(document, source_pdf) if source_pdf else {"status": "not_checked"}
    if retriever == "bm25":
        retrieval = bm25.search_many(document, [question], top_k=top_k)[0]
        runtime = bm25.runtime_metadata()
    else:
        model = dense.load_model(local_files_only=local_files_only)
        search = dense.search_many if retriever == "dense" else hybrid.search_many
        retrieval = search(document, [question], model, top_k=top_k, encoding=encoding)[0]
        runtime = dense.runtime_metadata()
        if retriever == "hybrid":
            runtime["bm25"] = bm25.runtime_metadata()
    ids = [row["chunk"]["chunk_id"] for row in retrieval["results"]]
    selector = select_chunk_context if context == "chunk" else select_page_context
    selection = selector(document, ids, max_chars=max_chars)
    return {"schema_version": 1, "kind": "retrieved_evidence", "question": question,
            "retriever": retriever, "context": context, "runtime": runtime,
            "source_pdf_verification": verification, "retrieval": retrieval,
            "selection": selection,
            "interpretation": "Selection status describes delivery only, not answerability or claim support."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("question")
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--context", choices=["chunk", "page"], default="chunk")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--encoding", choices=["reject", "prefix", "window-mean"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--source-pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.document] + ([args.source_pdf] if args.source_pdf else [])
    if any(args.output.resolve() == p.resolve() or
           (args.output.exists() and p.exists() and args.output.samefile(p)) for p in inputs):
        parser.error("Output must not overwrite an input")
    try:
        document = ProcessedFiling.model_validate_json(args.document.read_text(encoding="utf-8"))
        result = query(document, args.question, retriever=args.retriever, context=args.context,
                       top_k=args.top_k, max_chars=args.max_chars, encoding=args.encoding,
                       local_files_only=args.local_files_only, source_pdf=args.source_pdf)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    selection = result["selection"]
    print(json.dumps({"status": selection["status"], "retrieved_chunks": len(result["retrieval"]["results"]),
                      "delivered_chars": selection["delivered_chars"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
