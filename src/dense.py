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


def text_windows(text, tokenizer, limit):
    """Partition all characters into bounded windows; never decode/rewrite text."""
    def split(start, end):
        count = len(tokenizer([text[start:end]], truncation=False, padding=False)["input_ids"][0])
        if count <= limit:
            return [{"start": start, "end": end, "tokens": count}]
        if end - start <= 1:
            raise ValueError("A single character exceeds the model token limit")
        middle = (start + end) // 2
        # Prefer nearby whitespace, but always make progress.
        candidates = [i for i in range(max(start + 1, middle - 40), min(end, middle + 40))
                      if text[i].isspace()]
        cut = min(candidates, key=lambda i: abs(i - middle)) if candidates else middle
        return split(start, cut) + split(cut, end)
    return split(0, len(text))


def mean_windows(vectors, windows):
    """Character-weighted mean of unit window embeddings, then cosine at ranking."""
    result = [0.0] * len(vectors[0])
    total = sum(w["end"] - w["start"] for w in windows)
    for vector, window in zip(vectors, windows, strict=True):
        norm = math.hypot(*vector)
        if not norm or not math.isfinite(norm) or len(vector) != len(result):
            raise ValueError("Invalid window embedding")
        weight = (window["end"] - window["start"]) / total
        for i, value in enumerate(vector):
            result[i] += weight * value / norm
    return result


def search(document, question, model, *, top_k=5, allow_truncation=False, encoding="reject"):
    return search_many(document, [question], model, top_k=top_k,
                       allow_truncation=allow_truncation, encoding=encoding)[0]


def search_many(document, questions, model, *, top_k=5, allow_truncation=False, encoding="reject"):
    """Encode the corpus once per batch. No labels or answers enter embeddings."""
    if not questions or any(not q.strip() for q in questions) or top_k < 1:
        raise ValueError("Require nonblank questions and positive top_k")
    if encoding not in {"reject", "prefix", "window-mean"}:
        raise ValueError("Unknown encoding policy")
    if allow_truncation:
        if encoding == "window-mean":
            raise ValueError("Window encoding cannot request truncation")
        encoding = "prefix"
    validate_document(document)
    if not document.chunks:
        raise ValueError("Document has no chunks")
    texts = [c.text for c in document.chunks]
    started = perf_counter()
    counts = [len(t) for t in model.tokenizer(texts + questions, truncation=False,
                                             padding=False)["input_ids"]]
    limit = model.max_seq_length
    n = len(texts)
    oversized_ids = [c.chunk_id for c, count in zip(document.chunks, counts[:n], strict=True) if count > limit]
    if encoding == "reject" and any(count > limit for count in counts):
        raise ValueError(f"Model limit {limit} tokens exceeded. Use --encoding window-mean "
                         "or --allow-truncation for the prefix-only baseline.")
    manifests = [text_windows(text, model.tokenizer, limit) if encoding == "window-mean"
                 else [{"start": 0, "end": len(text), "tokens": count}]
                 for text, count in zip(texts + questions, counts, strict=True)]
    tokenization_seconds = perf_counter() - started

    def encode_group(group_texts, group_windows):
        inputs = [text[w["start"]:w["end"]] for text, ws in zip(group_texts, group_windows, strict=True) for w in ws]
        embeddings = model.encode(inputs, batch_size=32, convert_to_numpy=True, show_progress_bar=False).tolist()
        vectors, cursor = [], 0
        for ws in group_windows:
            selected = embeddings[cursor:cursor + len(ws)]
            vectors.append(mean_windows(selected, ws) if encoding == "window-mean" else selected[0])
            cursor += len(ws)
        return vectors

    started = perf_counter()
    vectors = encode_group(texts, manifests[:n])
    corpus_seconds = perf_counter() - started
    started = perf_counter()
    queries = encode_group(questions, manifests[n:])
    query_seconds = perf_counter() - started
    chunks = {c.chunk_id: c for c in document.chunks}
    token_counts = dict(zip(chunks, counts[:n], strict=True))
    truncated_ids = oversized_ids if encoding == "prefix" else []
    corpus_hash = hashlib.sha256(document.model_dump_json().encode()).hexdigest()
    results = []
    for index, (question, query) in enumerate(zip(questions, queries, strict=True)):
        started = perf_counter()
        ranking = cosine_ranking(list(chunks), vectors, query, top_k)
        results.append({
            "schema_version": 2, "question": question, "requested_top_k": top_k,
            "source_sha256": document.source_sha256, "document_model_sha256": corpus_hash,
            "chunking": document.chunking, "chunk_count": n,
            "embedding_dimensions": len(query), "similarity": "cosine",
            "tie_breaker": "chunk_id_ascending", "embedding_input": "chunk_text_only",
            "encoding": {"policy": encoding, "pooling": "character_weighted_unit_mean" if encoding == "window-mean" else None,
                         "chunk_windows": dict(zip(chunks, manifests[:n], strict=True)),
                         "query_windows": manifests[n + index]},
            "truncation": {"policy": encoding, "max_sequence_tokens": limit,
                           "chunk_ids": truncated_ids, "query_tokens": counts[n + index],
                           "query_truncated": encoding == "prefix" and counts[n + index] > limit},
            "timings_seconds": {"tokenization_batch": tokenization_seconds, "corpus_encoding": corpus_seconds,
                                "query_encoding_batch": query_seconds, "ranking": perf_counter() - started},
            "results": [{"rank": rank, "score": score, "embedding_input_tokens": token_counts[cid],
                         "embedding_truncated": cid in truncated_ids, "chunk": chunks[cid].model_dump(mode="json")}
                        for rank, (cid, score) in enumerate(ranking, 1)],
        })
    return results


def load_model(local_files_only=False):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return SentenceTransformer(MODEL, revision=REVISION, device="cpu", trust_remote_code=False,
                               local_files_only=local_files_only)


def runtime_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(),
            "model": {"name": MODEL, "revision": REVISION, "device": "cpu", "threads": 1},
            "versions": {name: version(name) for name in ("sentence-transformers", "torch", "transformers")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-pdf", type=Path)
    parser.add_argument("--allow-truncation", action="store_true")
    parser.add_argument("--encoding", choices=["reject", "prefix", "window-mean"], default="reject")
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
    started = perf_counter()
    model = load_model(args.local_files_only)
    loading_seconds = perf_counter() - started
    result = search(document, args.question, model, top_k=args.top_k,
                    allow_truncation=args.allow_truncation, encoding=args.encoding)
    result.update(runtime_metadata())
    result["source_pdf_verification"] = verification
    result["timings_seconds"]["model_loading"] = loading_seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"results": len(result["results"]), "truncated_chunks": len(result["truncation"]["chunk_ids"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
