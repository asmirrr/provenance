"""Offline synthetic retrieval tests; fixtures are not financial evidence."""

import pytest

from src.dense import cosine_ranking, search
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
