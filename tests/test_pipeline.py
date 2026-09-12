"""Synthetic strings below are unit fixtures, never research observations."""

from pathlib import Path

import pytest

from src.ingestion import pipeline
from src.ingestion.pipeline import FilingMetadata, ProcessedFiling, clean_page, ingest, spans


@pytest.fixture
def metadata():
    return FilingMetadata(company="Apple Inc.", ticker="AAPL", fiscal_year=2024,
                          filing_date="2024-11-01", source_url="https://example.org/test.pdf",
                          sec_url="https://example.org/test")


def test_cleaning_preserves_numbers_hyphens_and_raw_offsets(metadata):
    raw = "\n Item 8. Statements\nSales $ 1,234 (56)\nlong-\nterm\nApple Inc. | 2024 Form 10-K | 29\n"
    page = clean_page(raw, metadata, 32)
    assert page.printed_page == 29
    assert page.text == raw[page.raw_start:page.raw_start + len(page.text)]
    assert "long-\nterm" in page.text
    assert "Sales $ 1,234 (56)" in page.text
    assert "Form 10-K" not in page.text
    assert clean_page("Revenue\n29", metadata, 1).text == "Revenue\n29"


@pytest.mark.parametrize("text", ["word " * 300, "x" * 350, "\n " * 200, "a\n" * 200])
def test_spans_bounded_and_cover_all_nonwhitespace(text):
    parts = [text[a:b] for a, b in spans(text, 0, len(text), 100)]
    assert all(0 < len(part) <= 100 for part in parts)
    assert "".join("".join(parts).split()) == "".join(text.split())


def test_ingestion_sections_citations_roundtrip_and_determinism(tmp_path, monkeypatch, metadata):
    pdf = tmp_path / "fixture.pdf"
    pdf.write_bytes(b"unit test stub; extraction mocked")
    raw = [
        {"page": 1, "text": "TABLE OF CONTENTS\nItem 1. Business 1"},
        {"page": 2, "text": "Preface\nItem 1. Business\n" + "Business line.\n" * 30 + "Item 1A. Risk Factors\nRisk."},
        {"page": 3, "text": "Risk continuation.\nSIGNATURES\nSigned."},
        {"page": 4, "text": "Exhibit\nItem 1. Not a report section"},
        {"page": 5, "text": ""},
    ]
    monkeypatch.setattr(pipeline, "extract_pages", lambda _: raw)
    result = ingest(pdf, metadata, 100)
    assert result == ingest(pdf, metadata, 100)
    assert ProcessedFiling.model_validate_json(result.model_dump_json()) == result
    assert len({c.chunk_id for c in result.chunks}) == len(result.chunks)
    assert result.chunks[0].section is None
    assert next(c for c in result.chunks if c.text == "Preface").section is None
    assert next(c for c in result.chunks if c.text.startswith("Item 1A.")).section == "Item 1A. Risk Factors"
    assert next(c for c in result.chunks if c.page == 3).section == "Item 1A. Risk Factors"
    assert all(c.section is None for c in result.chunks if c.page == 4)
    assert len(result.warnings) == 1
    for c in result.chunks:
        assert c.text == raw[c.page - 1]["text"][c.raw_start:c.raw_end]
        assert c.ticker == "AAPL" and c.filing_date == "2024-11-01"
    for p in result.pages:
        rebuilt = "".join(c.text for c in result.chunks if c.page == p.page)
        assert "".join(rebuilt.split()) == "".join(p.text.split())
    pdf.write_bytes(b"changed source")
    assert ingest(pdf, metadata, 100).chunks[0].chunk_id != result.chunks[0].chunk_id


def test_empty_document_rejected(tmp_path, monkeypatch, metadata):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(pipeline, "extract_pages", lambda _: [{"page": 1, "text": ""}])
    with pytest.raises(ValueError, match="No extractable text"):
        ingest(path, metadata)
    with pytest.raises(ValueError, match="at least 100"):
        ingest(path, metadata, 0)


def test_real_filing(monkeypatch):
    """Offline integration test; download the documented source to enable it."""
    root = Path(__file__).resolve().parents[1]
    pdf = root / "data/raw/aapl-2024-10k.pdf"
    if not pdf.exists():
        pytest.skip("Download the real filing using the README instructions")
    metadata = FilingMetadata.model_validate_json(
        (root / "data/raw/aapl-2024-10k.metadata.json").read_text(encoding="utf-8")
    )
    result = ingest(pdf, metadata)
    assert len(result.pages) == 121
    assert not result.warnings
    assert result.pages[31].printed_page == 29
    assert "Total net sales 391,035 383,285 394,328" in result.pages[31].text
    assert "SIGNATURES" in result.pages[59].text
    for chunk in result.chunks:
        assert chunk.text == result.pages[chunk.page - 1].raw_text[chunk.raw_start:chunk.raw_end]
        assert len(chunk.text) <= 1800
    for page in result.pages:
        rebuilt = "".join(c.text for c in result.chunks if c.page == page.page)
        assert "".join(rebuilt.split()) == "".join(page.text.split())
    assert all(c.section is None for c in result.chunks if c.page > 60)
    assert all(c.section == "Item 8. Financial Statements and Supplementary Data"
               for c in result.chunks if c.page == 32)
    # Compare strategies on exactly the same real extracted pages, without
    # repeating PDF extraction or pinning case locations to generated chunks.
    from src.evaluation import EvidenceCases, evaluate
    cases = EvidenceCases.model_validate_json(
        (root / "data/evaluation/aapl-2024-10k.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(pipeline, "extract_pages", lambda _: [
        {"page": p.page, "text": p.raw_text} for p in result.pages
    ])
    baseline = ingest(pdf, metadata, chunking="line-v1")
    before, after = evaluate(baseline, cases), evaluate(result, cases)
    assert before["single_chunk_complete"] == 3
    assert after["single_chunk_complete"] == 7
    assert after["section_correct"] == after["page_context_complete"] == 8
    assert [r["case_id"] for r in after["results"] if not r["single_chunk_complete"]] == ["cash-tax-payment"]


def test_sentence_boundary_preserves_following_heading():
    text = "Prior discussion " + "word " * 10 + ".\nNext Heading\n" + "following " * 8
    parts = [text[a:b] for a, b in spans(text, 0, len(text), 100)]
    assert parts[0].endswith(".")
    assert parts[1].startswith("Next Heading\n")
    assert "".join("".join(parts).split()) == "".join(text.split())


@pytest.mark.parametrize("abbreviation", ["U.S.", "Inc.", "Corp."])
def test_abbreviations_are_not_sentence_boundaries(abbreviation):
    text = "Intro " + "word " * 10 + abbreviation + " market continued with growth\nand more words after that."
    first = next(spans(text, 0, len(text), 100))
    assert not text[first[0]:first[1]].endswith(abbreviation)
