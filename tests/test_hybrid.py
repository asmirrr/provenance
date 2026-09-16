"""Synthetic fusion rankings; no model or financial data required."""

from copy import deepcopy

import pytest

from src.hybrid import fuse


def rows(ids):
    return [{"rank": i, "score": 100 - i, "chunk": {"chunk_id": cid, "text": cid}}
            for i, cid in enumerate(ids, 1)]


def test_fusion_counts_agreement_and_preserves_component_scores():
    results = fuse(rows(["a", "b"]), rows(["c", "b"]))
    assert [r["chunk"]["chunk_id"] for r in results] == ["b", "a", "c"]
    assert results[0]["score"] == pytest.approx(2 / 62)
    assert results[0]["components"]["dense"] == {"rank": 2, "score": 98, "rrf_contribution": 1 / 62}
    assert "bm25" not in results[1]["components"]


def test_fusion_is_independent_of_score_scale_and_does_not_mutate_inputs():
    left, right = rows(["a", "b"]), rows(["b", "c"])
    original = deepcopy((left, right))
    before = fuse(left, right)
    assert (left, right) == original
    for r in left:
        r["score"] *= 1000
    after = fuse(left, right)
    assert [(r["chunk"], r["score"]) for r in before] == [(r["chunk"], r["score"]) for r in after]


def test_empty_component_and_cutoff():
    assert fuse([], []) == []
    assert [r["chunk"]["chunk_id"] for r in fuse(rows(["b", "a"]), [], top_k=1)] == ["b"]
    with pytest.raises(ValueError):
        fuse([], [], top_k=0)


@pytest.mark.parametrize("corruption", ["duplicate", "rank", "citation"])
def test_invalid_components_fail(corruption):
    left, right = rows(["a", "b"]), rows(["a"])
    if corruption == "duplicate":
        left[1]["chunk"] = left[0]["chunk"]
    elif corruption == "rank":
        left[1]["rank"] = 3
    else:
        right[0]["chunk"]["text"] = "changed"
    with pytest.raises(ValueError):
        fuse(left, right)


@pytest.mark.parametrize("field", ["question", "document_model_sha256", "source_sha256"])
def test_batch_refuses_mismatched_component_identity(monkeypatch, field):
    from src import hybrid
    left = {"question": "query", "document_model_sha256": "corpus", "source_sha256": "pdf"}
    right = {**left, field: "different"}
    monkeypatch.setattr(hybrid.dense, "search_many", lambda *a, **k: [left])
    monkeypatch.setattr(hybrid.bm25, "search_many", lambda *a, **k: [right])
    with pytest.raises(ValueError, match="identical"):
        hybrid.search_many(None, ["query"], None)


def test_candidate_budget_is_explicit():
    from src.hybrid import search_many
    with pytest.raises(ValueError, match="candidate depth"):
        search_many(None, ["query"], None, top_k=11)
