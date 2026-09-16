"""Synthetic scoring checks; no model, network, or real financial assertions."""

import math

import pytest

from src.bm25 import BM25Index, tokenize, search_many
from src.ingestion import pipeline


def test_score_matches_hand_calculation():
    index = BM25Index(["a", "b"], ["cash cash debt", "debt"])
    # N=2, df(cash)=1, tf=2, length=3, average length=2.
    expected = math.log(2) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
    assert index.rank("cash") == [("a", pytest.approx(expected))]
    assert index.rank("cash cash") == index.rank("cash")


def test_ties_oov_empty_tokens_and_length_normalization():
    index = BM25Index(["b", "a", "c"], ["cash", "cash", "cash other other"])
    ranked = index.rank("cash", 20)
    assert [cid for cid, _ in ranked] == ["a", "b", "c"]
    assert index.rank("absent") == []
    assert index.rank("!!!") == []
    assert BM25Index(["a"], ["!!!"]).rank("cash") == []


def test_tokenization_preserves_years_decimals_and_normalizes_grouping():
    assert tokenize("CASH $391,035 2024 6.08 (1,953) 24.6%") == [
        "cash", "391035", "2024", "6.08", "1953", "24.6"]
    assert tokenize("391035") == tokenize("391,035")
    assert tokenize("2023") != tokenize("2024")


@pytest.mark.parametrize("k1,b", [(0, .75), (-1, .75), (float("nan"), .75),
                                    (1.2, -1), (1.2, 2), (1.2, float("inf"))])
def test_invalid_parameters_fail(k1, b):
    with pytest.raises(ValueError):
        BM25Index(["a"], ["text"], k1=k1, b=b)


def test_invalid_identity_and_query_fail():
    for ids, texts in [([], []), (["a", "a"], ["a", "b"]), (["a"], [])]:
        with pytest.raises(ValueError):
            BM25Index(ids, texts)
    index = BM25Index(["a"], ["text"])
    with pytest.raises(ValueError):
        index.rank(" ")
    with pytest.raises(ValueError):
        index.rank("text", 0)


@pytest.fixture
def document(tmp_path, monkeypatch):
    pdf = tmp_path / "fixture.pdf"
    pdf.write_bytes(b"synthetic fixture")
    monkeypatch.setattr(pipeline, "extract_pages", lambda _: [
        {"page": 1, "text": "Cash amount 2024 1,234"},
        {"page": 2, "text": "Separate debt disclosure"}])
    return pipeline.ingest(pdf, pipeline.FilingMetadata(
        company="Fixture", ticker="TEST", fiscal_year=2024, filing_date="2024-11-01",
        source_url="https://example.org/fixture.pdf", sec_url="https://example.org/fixture"))


def test_batch_citations_and_repeatability(document):
    before = document.model_dump_json()
    runs = search_many(document, ["cash 1234", "debt", "missing"])
    assert runs[0]["results"][0]["chunk"] == document.chunks[0].model_dump(mode="json")
    assert runs[1]["results"][0]["chunk"] == document.chunks[1].model_dump(mode="json")
    assert runs[2]["results"] == []
    assert runs[0]["results"] == search_many(document, ["cash 1234"])[0]["results"]
    assert document.model_dump_json() == before
    document.chunks[0].text = "altered"
    with pytest.raises(ValueError, match="citation"):
        search_many(document, ["cash"])


def test_cli_guards_inputs_and_source_before_writing(document, tmp_path, monkeypatch):
    from src.bm25 import main
    source = tmp_path / "document.json"
    source.write_text(document.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["bm25", str(source), "cash", "--output", str(source)])
    with pytest.raises(SystemExit):
        main()
    wrong = tmp_path / "wrong.pdf"
    wrong.write_bytes(b"wrong")
    output = tmp_path / "output.json"
    output.write_text("previous", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["bm25", str(source), "cash", "--output", str(output), "--source-pdf", str(wrong)])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        main()
    assert output.read_text(encoding="utf-8") == "previous"
