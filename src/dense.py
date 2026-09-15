"""Small, exact CPU dense retrieval baseline; returns evidence, never answers."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from time import perf_counter

from src.evidence import validate_document, verify_source_pdf
from src.ingestion.pipeline import ProcessedFiling

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def cosine_ranking(ids, vectors, query, top_k):
    """Exact cosine ordering with chunk ID as a stable tie-breaker."""
    if top_k < 1 or not ids or len(ids) != len(vectors) or len(set(ids)) != len(ids):
        raise ValueError("Require positive top_k and unique IDs matching nonempty vectors")

    def unit(vector):
        if not vector or any(not math.isfinite(x) for x in vector):
            raise ValueError("Embeddings must be finite and nonempty")
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Embedding norm must be finite and positive")
        return [x / norm for x in vector]

    query = unit(query)
    scores = []
    for cid, vector in zip(ids, vectors, strict=True):
        if len(vector) != len(query):
            raise ValueError("Embedding dimensions differ")
        score = math.fsum(a * b for a, b in zip(unit(vector), query, strict=True))
        scores.append((cid, max(-1.0, min(1.0, score))))
    return sorted(scores, key=lambda row: (-row[1], row[0]))[:top_k]


def search(document, question, model, *, top_k=5, allow_truncation=False):
    """Encode raw chunk text only; benchmark labels never enter model input.

    The caller supplies the pinned model. Each run embeds the corpus anew so
    there is no stale cache or vector-to-chunk mapping to invalidate.
    """
    if not question.strip() or top_k < 1:
        raise ValueError("Require a nonblank question and positive top_k")
    validate_document(document)
    if not document.chunks:
        raise ValueError("Document has no chunks")
    texts = [c.text for c in document.chunks]
    started = perf_counter()
    # Include special tokens, matching the model's input sequence limit.
    tokens = model.tokenizer(texts + [question], truncation=False, padding=False)["input_ids"]
    counts = [len(t) for t in tokens]
    limit = model.max_seq_length
    truncated_ids = [c.chunk_id for c, count in zip(document.chunks, counts[:-1], strict=True)
                     if count > limit]
    query_truncated = counts[-1] > limit
    if (truncated_ids or query_truncated) and not allow_truncation:
        raise ValueError(f"Model limit {limit} tokens: {len(truncated_ids)} chunks exceed it; "
                         f"query exceeds limit: {query_truncated}. Use --allow-truncation "
                         "to run the explicitly recorded prefix-only baseline.")
    tokenization_seconds = perf_counter() - started
    started = perf_counter()
    vectors = model.encode(texts, batch_size=32, convert_to_numpy=True,
                           show_progress_bar=False).tolist()
    corpus_seconds = perf_counter() - started
    started = perf_counter()
    query = model.encode([question], convert_to_numpy=True, show_progress_bar=False).tolist()[0]
    query_seconds = perf_counter() - started
    started = perf_counter()
    ranking = cosine_ranking([c.chunk_id for c in document.chunks], vectors, query, top_k)
    ranking_seconds = perf_counter() - started
    chunks = {c.chunk_id: c for c in document.chunks}
    token_counts = dict(zip(chunks, counts[:-1], strict=True))
    return {
        "schema_version": 1, "question": question, "requested_top_k": top_k,
        "source_sha256": document.source_sha256,
        "document_model_sha256": hashlib.sha256(document.model_dump_json().encode()).hexdigest(),
        "chunking": document.chunking, "chunk_count": len(texts),
        "embedding_dimensions": len(query), "similarity": "cosine",
        "tie_breaker": "chunk_id_ascending", "embedding_input": "chunk_text_only",
        "truncation": {"policy": "prefix" if allow_truncation else "reject",
                       "max_sequence_tokens": limit, "chunk_ids": truncated_ids,
                       "query_tokens": counts[-1], "query_truncated": query_truncated},
        "timings_seconds": {"tokenization": tokenization_seconds, "corpus_encoding": corpus_seconds,
                            "query_encoding": query_seconds, "ranking": ranking_seconds},
        "results": [{"rank": rank, "score": score,
                     "embedding_input_tokens": token_counts[cid],
                     "embedding_truncated": cid in truncated_ids,
                     "chunk": chunks[cid].model_dump(mode="json")}
                    for rank, (cid, score) in enumerate(ranking, 1)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-pdf", type=Path)
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", help="Use downloaded model without network")
    args = parser.parse_args()
    inputs = {args.document.resolve()}
    if args.source_pdf:
        inputs.add(args.source_pdf.resolve())
    if args.output.resolve() in inputs:
        parser.error("Output must not overwrite an input")
    if args.top_k < 1 or not args.question.strip():
        parser.error("Require positive top_k and a nonblank question")
    document = ProcessedFiling.model_validate_json(args.document.read_text(encoding="utf-8"))
    validate_document(document)
    verification = verify_source_pdf(document, args.source_pdf) if args.source_pdf else {"status": "not_checked"}
    # Lazy imports keep ingestion and offline unit tests independent of model loading.
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = perf_counter()
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu", trust_remote_code=False,
                                local_files_only=args.local_files_only)
    loading_seconds = perf_counter() - started
    result = search(document, args.question, model, top_k=args.top_k,
                    allow_truncation=args.allow_truncation)
    result.update({"created_at": datetime.now(timezone.utc).isoformat(),
                   "model": {"name": MODEL, "revision": REVISION, "device": "cpu", "threads": 1},
                   "versions": {name: version(name) for name in ("sentence-transformers", "torch", "transformers")},
                   "source_pdf_verification": verification})
    result["timings_seconds"]["model_loading"] = loading_seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"results": len(result["results"]), "truncated_chunks": len(result["truncation"]["chunk_ids"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
