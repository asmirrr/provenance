"""Offline synthetic retrieval tests; fixtures are not financial evidence."""

import pytest

from src.dense import cosine_ranking, search, search_many, text_windows, mean_windows
from src.ingestion import pipeline


def test_cosine_normalizes_and_breaks_ties_by_id():
    result = cosine_ranking(["b", "c", "a"], [[2, 0], [0, 1], [10, 0]], [3, 0], 9)
    assert result == [("a", 1.0), ("b", 1.0), ("c", 0.0)]
    assert cosine_ranking(["a", "b"], [[-1, 0], [0, 1]], [1, 0], 1) == [("b", 0.0)]


@pytest.mark.parametrize("ids,vectors,query,k", [
    ([], [], [1], 1), (["a", "a"], [[1], [1]], [1], 1),
    (["a"], [], [1], 1), (["a"], [[1]], [1], 0),
    (["a"], [[1, 2]], [1], 1), (["a"], [[0]], [1], 1),
    (["a"], [[float("nan")]], [1], 1), (["a"], [[1]], [float("inf")], 1),
])
def test_invalid_vectors_fail(ids, vectors, query, k):
    with pytest.raises(ValueError):
        cosine_ranking(ids, vectors, query, k)


@pytest.fixture
def document(tmp_path, monkeypatch):
    pdf = tmp_path / "fixture.pdf"
    pdf.write_bytes(b"synthetic fixture")
    monkeypatch.setattr(pipeline, "extract_pages", lambda _: [
        {"page": 1, "text": "First synthetic passage."},
        {"page": 2, "text": "Second synthetic passage."},
    ])
    return pipeline.ingest(pdf, pipeline.FilingMetadata(
        company="Fixture", ticker="TEST", fiscal_year=2024, filing_date="2024-11-01",
        source_url="https://example.org/fixture.pdf", sec_url="https://example.org/fixture"))


class FakeModel:
    max_seq_length = 10

    def __init__(self, counts):
        self.counts = counts
        self.calls = []

    def tokenizer(self, texts, **kwargs):
        assert kwargs == {"truncation": False, "padding": False}
        assert len(texts) == len(self.counts)
        return {"input_ids": [[0] * n for n in self.counts]}

    def encode(self, texts, **kwargs):
        self.calls.append(texts)

        class Array:
            def tolist(self):
                return [[1, 0], [0, 1]] if len(texts) == 2 else [[1, 0]]

        return Array()


def test_search_returns_exact_citations_and_only_encodes_text(document):
    model = FakeModel([10, 5, 4])
    result = search(document, "Synthetic query", model, top_k=1)
    assert model.calls == [[c.text for c in document.chunks], ["Synthetic query"]]
    assert result["results"][0]["chunk"] == document.chunks[0].model_dump(mode="json")
    assert not result["results"][0]["embedding_truncated"]
    assert result["truncation"]["chunk_ids"] == []
    assert result["source_sha256"] == document.source_sha256


@pytest.mark.parametrize("counts", [[11, 5, 4], [5, 5, 11]])
def test_truncation_requires_opt_in_before_encoding(document, counts):
    model = FakeModel(counts)
    with pytest.raises(ValueError, match="allow-truncation"):
        search(document, "Query", model)
    assert not model.calls
    result = search(document, "Query", model, allow_truncation=True)
    assert result["truncation"]["query_truncated"] == (counts[-1] > 10)
    assert result["results"][0]["embedding_truncated"] == (counts[0] > 10)
    assert result["results"][0]["chunk"]["text"] == document.chunks[0].text


def test_invalid_input_fails_before_encoding(document):
    model = FakeModel([5, 5, 4])
    with pytest.raises(ValueError, match="nonblank"):
        search(document, " ", model)
    document.chunks[0].text = "tampered"
    with pytest.raises(ValueError, match="citation"):
        search(document, "Query", model)
    assert not model.calls


def test_cli_rejects_input_overwrite_before_loading_model(document, tmp_path, monkeypatch):
    from src.dense import main
    path = tmp_path / "document.json"
    original = document.model_dump_json()
    path.write_text(original, encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["dense", str(path), "Query", "--output", str(path)])
    with pytest.raises(SystemExit):
        main()
    assert path.read_text(encoding="utf-8") == original


def test_cli_rejects_wrong_pdf_without_changing_output(document, tmp_path, monkeypatch):
    from src.dense import main
    path = tmp_path / "document.json"
    path.write_text(document.model_dump_json(), encoding="utf-8")
    wrong = tmp_path / "wrong.pdf"
    wrong.write_bytes(b"different synthetic source")
    output = tmp_path / "result.json"
    output.write_text("previous result", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["dense", str(path), "Query", "--output", str(output),
                                    "--source-pdf", str(wrong)])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        main()
    assert output.read_text(encoding="utf-8") == "previous result"


class WindowModel:
    max_seq_length = 10

    def __init__(self):
        self.calls = []

    def tokenizer(self, texts, **kwargs):
        return {"input_ids": [[0] * (len(text) + 2) for text in texts]}

    def encode(self, texts, **kwargs):
        assert all(len(text) + 2 <= self.max_seq_length for text in texts)
        self.calls.append(texts)

        class Array:
            def tolist(self):
                return [[len(text), 1] for text in texts]
        return Array()


@pytest.mark.parametrize("text", ["abcdefgh", "a" * 50, "first sentence. second sentence!", "a\n b  c\t" * 8])
def test_windows_cover_every_character_without_overflow(text):
    model = WindowModel()
    windows = text_windows(text, model.tokenizer, model.max_seq_length)
    assert "".join(text[w["start"]:w["end"]] for w in windows) == text
    assert all(w["tokens"] <= 10 for w in windows)
    assert windows[0]["start"] == 0 and windows[-1]["end"] == len(text)
    assert all(a["end"] == b["start"] for a, b in zip(windows, windows[1:]))


def test_weighted_pooling_does_not_overweight_small_tail():
    assert mean_windows([[2, 0], [0, 3]], [{"start": 0, "end": 3}, {"start": 3, "end": 4}]) == [0.75, 0.25]


def test_batch_encodes_corpus_once_and_keeps_citations(document):
    model = WindowModel()
    runs = search_many(document, ["query one long", "query two long"], model, encoding="window-mean")
    assert len(model.calls) == 2
    assert "".join(model.calls[0]) == "".join(c.text for c in document.chunks)
    assert "".join(model.calls[1]) == "query one longquery two long"
    for run in runs:
        assert run["truncation"]["chunk_ids"] == []
        assert not run["truncation"]["query_truncated"]
        assert {r["chunk"]["chunk_id"] for r in run["results"]} == {c.chunk_id for c in document.chunks}


def test_conflicting_encoding_options_rejected(document):
    with pytest.raises(ValueError, match="cannot request"):
        search(document, "query", WindowModel(), allow_truncation=True, encoding="window-mean")


def test_group_metrics_do_not_mix_alternatives_or_score_unsupported():
    from src.retrieval_evaluation import coverage
    item = {"answerable": True, "evidence_groups": [
        {"supporting_chunk_ids": ["a", "b"]}, {"supporting_chunk_ids": ["c", "d"]}]}
    metrics = coverage(item, ["x", "a", "c"])
    assert metrics["reciprocal_rank_at_10"] == 0.5
    assert metrics["best_group_recall_at_3"] == 0.5
    assert metrics["complete_group_at_3"] == 0
    assert coverage(item, ["c", "d"])["complete_group_at_3"] == 1
    assert coverage({"answerable": False}, ["a"]) is None
    with pytest.raises(ValueError, match="Duplicate"):
        coverage(item, ["a", "a"])


def test_draft_evaluation_requires_explicit_opt_in(document):
    from types import SimpleNamespace
    from src.retrieval_evaluation import evaluate
    with pytest.raises(ValueError, match="allow-draft"):
        evaluate(document, SimpleNamespace(status="draft_pending_human_review"), None, encoding="window-mean")


@pytest.mark.parametrize("retriever", ["dense", "bm25", "hybrid"])
def test_batch_diagnostic_excludes_unsupported_and_keeps_draft_status(document, retriever):
    from src.benchmark import Benchmark
    from src.retrieval_evaluation import evaluate
    benchmark = Benchmark.model_validate({
        "version": "0.1.0", "status": "draft_pending_human_review",
        "source_sha256": document.source_sha256, "review_method": "Synthetic fixture",
        "items": [
            {"id": "supported", "question": "first?", "category": "factual", "answerable": True,
             "expected_answer": "Synthetic answer", "expected_claims": ["Synthetic claim"],
             "difficulty": "easy", "notes": "Unit fixture", "evidence": [
                 {"page": 1, "quote": document.chunks[0].text}]},
            {"id": "unsupported", "question": "future?", "category": "unsupported", "answerable": False,
             "expected_answer": None, "expected_claims": [], "difficulty": "easy",
             "notes": "Unit fixture", "evidence": []},
        ]})
    result = evaluate(document, benchmark, WindowModel() if retriever != "bm25" else None,
                      encoding="window-mean", allow_draft=True, retriever=retriever)
    assert result["evaluation_status"] == "draft_diagnostic"
    assert result["answerable_denominator"] == 1 and result["unsupported_excluded"] == 1
    assert result["items"][1]["metrics"] is None
    assert result["aggregate"]["complete_group_at_10"] == 1
    assert benchmark.status == "draft_pending_human_review"
