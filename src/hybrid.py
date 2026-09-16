"""Fixed reciprocal-rank fusion of dense and BM25 candidate lists."""

from time import perf_counter

from src import bm25, dense

RRF_K = 60
CANDIDATE_K = 10


def fuse(dense_results, bm25_results, *, top_k=10):
    """RRF with equal weights; absent candidates contribute zero, not a guessed rank."""
    if top_k < 1:
        raise ValueError("Require positive top_k")
    candidates = {}
    for name, results in (("dense", dense_results), ("bm25", bm25_results)):
        seen = set()
        for rank, result in enumerate(results, 1):
            cid = result["chunk"]["chunk_id"]
            if cid in seen or result["rank"] != rank:
                raise ValueError("Require unique candidates in consecutive rank order")
            seen.add(cid)
            candidate = candidates.setdefault(cid, {"chunk": result["chunk"], "components": {}})
            if candidate["chunk"] != result["chunk"]:
                raise ValueError("Component citations disagree")
            candidate["components"][name] = {"rank": rank, "score": result["score"],
                                              "rrf_contribution": 1 / (RRF_K + rank)}
    for candidate in candidates.values():
        candidate["score"] = sum(c["rrf_contribution"] for c in candidate["components"].values())
    ranked = sorted(candidates.values(), key=lambda c: (-c["score"], c["chunk"]["chunk_id"]))
    return [{"rank": rank, **candidate} for rank, candidate in enumerate(ranked[:top_k], 1)]


def search_many(document, questions, model, *, top_k=10, encoding="prefix"):
    if not 1 <= top_k <= CANDIDATE_K:
        raise ValueError("Hybrid top_k must be between 1 and the fixed candidate depth 10")
    started = perf_counter()
    dense_runs = dense.search_many(document, questions, model, top_k=CANDIDATE_K, encoding=encoding)
    bm25_runs = bm25.search_many(document, questions, top_k=CANDIDATE_K)
    retrieval_seconds = perf_counter() - started
    runs = []
    for left, right in zip(dense_runs, bm25_runs, strict=True):
        for field in ("question", "document_model_sha256", "source_sha256"):
            if left[field] != right[field]:
                raise ValueError("Component runs must use identical questions and corpus")
        started = perf_counter()
        results = fuse(left["results"], right["results"], top_k=top_k)
        elapsed = perf_counter() - started
        runs.append({"schema_version": 1, "retriever": "hybrid-rrf",
                     "question": left["question"], "source_sha256": left["source_sha256"],
                     "document_model_sha256": left["document_model_sha256"],
                     "chunking": document.chunking, "chunk_count": len(document.chunks),
                     "requested_top_k": top_k, "tie_breaker": "chunk_id_ascending",
                     "configuration": {"rrf_k": RRF_K, "candidate_k_per_retriever": CANDIDATE_K,
                                       "weights": {"dense": 1, "bm25": 1}},
                     # Preserve candidates, settings, truncation and original scores for auditing.
                     "component_runs": {"dense": left, "bm25": right},
                     "timings_seconds": {"component_retrieval_batch": retrieval_seconds, "fusion": elapsed},
                     "results": results})
    return runs
