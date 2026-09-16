"""Dependency-free BM25 baseline over complete chunk text; no answer generation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
from time import perf_counter

from src.evidence import validate_document, verify_source_pdf
from src.ingestion.pipeline import ProcessedFiling


def tokenize(text):
    """Casefold words; normalize numeric grouping, retaining decimal points.

    No stop list, stemming, synonyms, units conversion, or sign interpretation.
    Citation text is never modified by this search-only normalization.
    """
    return [token.replace(",", "") for token in
            re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?|[^\W\d_]+", text.casefold())]


class BM25Index:
    def __init__(self, ids, texts, *, k1=1.2, b=0.75):
        if not ids or len(ids) != len(texts) or len(set(ids)) != len(ids):
            raise ValueError("Require unique IDs matching a nonempty corpus")
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Require finite k1 > 0 and b between 0 and 1")
        self.ids, self.k1, self.b = list(ids), k1, b
        self.terms = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(terms.values()) for terms in self.terms]
        self.average_length = sum(self.lengths) / len(ids)
        self.df = Counter(term for terms in self.terms for term in terms)

    def rank(self, question, top_k=5):
        if not question.strip() or top_k < 1:
            raise ValueError("Require a nonblank question and positive top_k")
        # Repeated query words do not add weight. Sort for stable summation.
        query = sorted(set(tokenize(question)))
        scores = []
        for cid, terms, length in zip(self.ids, self.terms, self.lengths, strict=True):
            normalization = self.k1 * (1 - self.b + self.b * length / self.average_length) if self.average_length else self.k1
            score = math.fsum(
                math.log1p((len(self.ids) - self.df[t] + 0.5) / (self.df[t] + 0.5))
                * terms[t] * (self.k1 + 1) / (terms[t] + normalization)
                for t in query if terms[t])
            # Never fill top-k with arbitrary zero-overlap chunks.
            if score > 0:
                scores.append((cid, score))
        return sorted(scores, key=lambda row: (-row[1], row[0]))[:top_k]


def search_many(document, questions, *, top_k=5):
    if not questions or any(not q.strip() for q in questions) or top_k < 1:
        raise ValueError("Require nonblank questions and positive top_k")
    validate_document(document)
    chunks = {c.chunk_id: c for c in document.chunks}
    started = perf_counter()
    index = BM25Index(list(chunks), [c.text for c in document.chunks])
    indexing_seconds = perf_counter() - started
    digest = hashlib.sha256(document.model_dump_json().encode()).hexdigest()
    runs = []
    for question in questions:
        started = perf_counter()
        ranked = index.rank(question, top_k)
        elapsed = perf_counter() - started
        runs.append({"schema_version": 1, "retriever": "bm25", "question": question,
                     "requested_top_k": top_k, "source_sha256": document.source_sha256,
                     "document_model_sha256": digest, "chunking": document.chunking,
                     "chunk_count": len(chunks), "tie_breaker": "chunk_id_ascending",
                     "configuration": {"k1": index.k1, "b": index.b,
                                       "idf": "log(1 + (N-df+0.5)/(df+0.5))",
                                       "tokenizer": "words-numeric-v1", "query_terms": "unique",
                                       "input": "full_chunk_text", "zero_scores": "excluded"},
                     "query_tokens": tokenize(question),
                     "timings_seconds": {"indexing_batch": indexing_seconds, "query_and_ranking": elapsed},
                     "results": [{"rank": n, "score": score, "chunk": chunks[cid].model_dump(mode="json")}
                                 for n, (cid, score) in enumerate(ranked, 1)]})
    return runs


def runtime_metadata():
    return {"created_at": datetime.now(timezone.utc).isoformat(),
            "retriever": "bm25", "implementation": "bm25-v1", "python": platform.python_version()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-pdf", type=Path)
    args = parser.parse_args()
    inputs = [args.document] + ([args.source_pdf] if args.source_pdf else [])
    if args.output.resolve() in {p.resolve() for p in inputs}:
        parser.error("Output must not overwrite an input")
    document = ProcessedFiling.model_validate_json(args.document.read_text(encoding="utf-8"))
    verification = verify_source_pdf(document, args.source_pdf) if args.source_pdf else {"status": "not_checked"}
    result = search_many(document, [args.question], top_k=args.top_k)[0]
    result.update(runtime_metadata())
    result["source_pdf_verification"] = verification
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"results": len(result["results"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
