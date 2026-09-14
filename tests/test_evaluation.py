"""Synthetic fixtures test scoring logic; the real-filing cases live in data/evaluation."""

import pytest

from src.evaluation import EvidenceCase, EvidenceCases, evaluate
from src.ingestion import pipeline
from src.ingestion.pipeline import FilingMetadata, PageText, ingest


@pytest.fixture
def documents(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.pdf"
    path.write_bytes(b"synthetic fixture; parser mocked")
    text = "Header 2024\n" + "Body sentence.\n" * 25 + "Amount $ 12"
    monkeypatch.setattr(pipeline, "extract_pages", lambda _: [{"page": 1, "text": text}])
    metadata = FilingMetadata(company="Test", ticker="TEST", fiscal_year=2024,
                              filing_date="2024-01-01", source_url="https://example.org/test",
                              sec_url="https://example.org/test")
    whole = ingest(path, metadata)
    split = ingest(path, metadata, max_chars=100)
    cases = EvidenceCases(source_sha256=whole.source_sha256, review_method="synthetic test",
                          cases=[EvidenceCase(case_id="test", description="Header and amount",
                                              page=1, section=None, evidence=["Header 2024", "Amount $ 12"])])
    return whole, split, cases


def test_complete_chunk_and_fragmented_context_are_distinct(documents):
    whole, split, cases = documents
    assert evaluate(whole, cases)["single_chunk_complete"] == 1
    report = evaluate(split, cases)
    assert report["single_chunk_complete"] == 0
    assert report["page_context_complete"] == report["section_correct"] == 1
    assert len(report["results"][0]["overlapping_chunk_ids"]) > 1
    assert report == evaluate(split, cases)


@pytest.mark.parametrize("change", ["source", "missing", "ambiguous", "empty", "order", "duplicate", "page"])
def test_invalid_cases_fail_instead_of_scoring(documents, change):
    whole, _, cases = documents
    if change == "source":
        cases.source_sha256 = "different"
    elif change == "missing":
        cases.cases[0].evidence = ["not in source"]
    elif change == "ambiguous":
        cases.cases[0].evidence = ["Body sentence."]
    elif change == "empty":
        cases.cases[0].evidence = [""]
    elif change == "order":
        cases.cases[0].evidence.reverse()
    elif change == "duplicate":
        cases.cases.append(cases.cases[0])
    else:
        cases.cases[0].page = 2
    with pytest.raises(ValueError):
        evaluate(whole, cases)


@pytest.mark.parametrize("change", ["text", "offset", "source", "page"])
def test_corrupt_citations_fail(documents, change):
    whole, _, cases = documents
    chunk = whole.chunks[0]
    if change == "text":
        chunk.text = "fabricated"
    elif change == "offset":
        chunk.raw_start = -1
    elif change == "source":
        chunk.source_sha256 = "different"
    else:
        chunk.page = 99
    with pytest.raises(ValueError, match="Invalid raw citation"):
        evaluate(whole, cases)


def test_section_mismatch_is_reported(documents):
    whole, _, cases = documents
    cases.cases[0].section = "Item 8. Financial Statements"
    report = evaluate(whole, cases)
    assert report["single_chunk_complete"] == 1
    assert report["section_correct"] == 0


@pytest.mark.parametrize("change", ["text", "offset"])
def test_corrupt_cleaned_page_fails(documents, change):
    whole, _, cases = documents
    if change == "text":
        whole.pages[0].text = "\n".join(cases.cases[0].evidence)
    else:
        whole.pages[0].raw_start = -1
    with pytest.raises(ValueError, match="Invalid cleaned text citation"):
        evaluate(whole, cases)


def test_valid_empty_page_is_allowed(documents):
    whole, _, cases = documents
    whole.pages.append(PageText(page=2, raw_text=" \n", text="", raw_start=2))
    assert evaluate(whole, cases)["single_chunk_complete"] == 1


def test_overlapping_quote_occurrences_are_ambiguous(documents, monkeypatch):
    whole, _, cases = documents
    monkeypatch.setattr(pipeline, "extract_pages", lambda _: [{"page": 1, "text": "aaa"}])
    document = ingest(whole.source_path, whole.metadata)
    cases.cases[0].evidence = ["aa"]
    with pytest.raises(ValueError, match="exactly once"):
        evaluate(document, cases)
