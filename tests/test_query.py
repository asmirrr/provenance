"""Query integration using explicitly synthetic source text."""

import json
import sys

import pytest

from src import query
from src.ingestion import pipeline


@pytest.fixture
def document(tmp_path, monkeypatch):
    pdf = tmp_path / "synthetic.pdf"
    pdf.write_bytes(b"synthetic PDF fixture")
    monkeypatch.setattr(pipeline, "extract_pages", lambda _: [
        {"page": 1, "text": "Cash units millions.\n" + "Cash balance. " * 20},
        {"page": 2, "text": "Debt footnote."}])
    return pipeline.ingest(pdf, pipeline.FilingMetadata(
        company="Fixture", ticker="TEST", fiscal_year=2024, filing_date="2024-11-01",
        source_url="https://example.org/pdf", sec_url="https://example.org/filing"), max_chars=100)


def test_bm25_delivery_and_exact_citations_without_model(document, monkeypatch):
    monkeypatch.setattr(query.dense, "load_model", lambda **_: pytest.fail("BM25 loaded a model"))
    result = query.query(document, "Cash", max_chars=100, source_pdf=document.source_path)
    assert result["source_pdf_verification"]["status"] == "matched"
    selection = result["selection"]
    assert selection["status"] == "partial"
    assert selection["omitted_chunks"] and selection["delivered_chars"] <= 100
    for span in selection["bundle"]["spans"]:
        assert span["text"] == document.pages[span["page"] - 1].raw_text[span["raw_start"]:span["raw_end"]]
    assert "benchmark" not in result and "answer" not in result


def test_pages_empty_and_budget_failure(document):
    result = query.query(document, "Cash", context="page")
    assert result["selection"]["bundle"]["spans"][0]["text"] == document.pages[0].text
    for question, budget, status in [("unknownword", 8000, "no_results"), ("Cash", 1, "over_budget")]:
        result = query.query(document, question, max_chars=budget)
        assert result["selection"]["status"] == status
        assert result["selection"]["bundle"] is None


@pytest.mark.parametrize("options", [{"top_k": 0}, {"top_k": 11}, {"max_chars": 0},
    {"retriever": "other"}, {"context": "other"}, {"encoding": "prefix"},
    {"local_files_only": True}, {"retriever": "dense"}, {"retriever": "hybrid"}])
def test_options_fail_before_loading(document, monkeypatch, options):
    monkeypatch.setattr(query.dense, "load_model", lambda **_: pytest.fail("Loaded before validation"))
    with pytest.raises(ValueError):
        query.query(document, "Cash", **options)


@pytest.mark.parametrize("retriever", ["dense", "hybrid"])
def test_model_dispatch_preserves_encoding_and_rankings(document, monkeypatch, retriever):
    model = object()
    monkeypatch.setattr(query.dense, "load_model", lambda *, local_files_only: model if local_files_only else None)
    run = query.bm25.search_many(document, ["Cash"], top_k=3)[0]
    def search(doc, questions, received, *, top_k, encoding):
        assert doc is document and questions == ["Cash"] and received is model
        assert top_k == 3 and encoding == "window-mean"
        return [run]
    monkeypatch.setattr(query.dense if retriever == "dense" else query.hybrid, "search_many", search)
    monkeypatch.setattr(query.dense, "runtime_metadata", lambda: {"model": "synthetic"})
    result = query.query(document, "Cash", retriever=retriever, encoding="window-mean", local_files_only=True, top_k=3)
    assert result["retrieval"] == run
    assert result["source_pdf_verification"]["status"] == "not_checked"


def test_cli_export_input_guard_and_mismatched_pdf(document, tmp_path, monkeypatch):
    source = tmp_path / "document.json"
    source.write_text(document.model_dump_json(), encoding="utf-8")
    output = tmp_path / "result.json"
    def invoke(destination, *extra):
        monkeypatch.setattr(sys, "argv", ["query", str(source), "Cash", "--output", str(destination), *extra])
        query.main()
    invoke(output)
    assert json.loads(output.read_text(encoding="utf-8"))["selection"]["bundle"]
    original = source.read_bytes()
    with pytest.raises(SystemExit):
        invoke(source)
    assert source.read_bytes() == original
    before = output.read_bytes()
    bad = tmp_path / "wrong.pdf"
    bad.write_bytes(b"wrong")
    with pytest.raises(SystemExit):
        invoke(output, "--source-pdf", str(bad))
    assert output.read_bytes() == before
