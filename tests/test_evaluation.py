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


def test_expanded_cases_preserve_group_boundaries_and_budget(tmp_path, monkeypatch):
    from src.benchmark import Benchmark
    from src.evaluation import evaluate_benchmark
    pdf = tmp_path / 'synthetic.pdf'
    pdf.write_bytes(b'synthetic fixture')
    raw = [{'page': 1, 'text': '2024 units millions\nAmount 10'},
           {'page': 2, 'text': '2023 units millions\nAmount 20'}]
    monkeypatch.setattr(pipeline, 'extract_pages', lambda _: raw)
    doc = ingest(pdf, FilingMetadata(company='Fixture', ticker='TEST', fiscal_year=2024,
                 filing_date='2024-11-01', source_url='https://example.org/pdf', sec_url='https://example.org/sec'))
    bench = Benchmark.model_validate({'version': '0.1.0', 'status': 'draft_pending_human_review',
        'source_sha256': doc.source_sha256, 'review_method': 'Synthetic test', 'items': [{
        'id': 'comparison', 'question': 'Compare years', 'category': 'comparative', 'answerable': True,
        'expected_answer': 'Synthetic', 'expected_claims': ['Synthetic'], 'difficulty': 'easy', 'notes': 'Synthetic',
        'evidence': [{'page': 1, 'quote': raw[0]['text']}, {'page': 2, 'quote': raw[1]['text']}]}]})
    result = evaluate_benchmark(doc, bench)
    assert result['case_count'] == 1
    assert result['single_chunk_complete'] == result['target_page_complete'] == 0
    assert result['explicit_pages_complete'] == 1
    case = result['results'][0]
    assert case['supporting_pages'] == [1, 2] and len(case['supporting_chunk_ids']) == 2
    before = result['corpus_sha256']
    doc.source_path = 'different/machine.pdf'
    for chunk in doc.chunks:
        chunk.source_path = doc.source_path
    assert evaluate_benchmark(doc, bench)['corpus_sha256'] == before
    doc.pages[0].raw_text = 'corrupt'
    with pytest.raises(ValueError):
        evaluate_benchmark(doc, bench)


def test_fresh_filing_matches_versioned_ground_truth_and_preserves_pages(monkeypatch):
    import json
    from pathlib import Path
    from src.benchmark import Benchmark
    from src.evaluation import evaluate_benchmark
    from src.ingestion.pdf_parser import extract_pages
    root = Path(__file__).resolve().parents[1]
    pdf = root / 'data/raw/aapl-2024-10k.pdf'
    if not pdf.exists():
        pytest.skip('Download the documented real filing to check the pinned diagnostic')
    raw = extract_pages(pdf)
    assert [p['page'] for p in raw] == list(range(1, 122))
    # Reuse fresh extraction for a second deterministic ingestion, not stored processed JSON.
    monkeypatch.setattr(pipeline, 'extract_pages', lambda _: raw)
    metadata = FilingMetadata.model_validate_json((root/'data/raw/aapl-2024-10k.metadata.json').read_text(encoding='utf-8'))
    doc = ingest(pdf, metadata)
    assert doc.model_dump_json() == ingest(pdf, metadata).model_dump_json()
    assert len(doc.pages) == 121 and len(doc.chunks) == 314
    for page, extracted in zip(doc.pages, raw, strict=True):
        assert page.raw_text == extracted['text']
        chunks = [c for c in doc.chunks if c.page == page.page]
        cursor = page.raw_start
        for chunk in chunks:
            assert chunk.raw_start >= cursor
            assert not page.raw_text[cursor:chunk.raw_start].strip()
            assert chunk.text == page.raw_text[chunk.raw_start:chunk.raw_end]
            cursor = chunk.raw_end
        assert not page.raw_text[cursor:page.raw_start + len(page.text)].strip()
    bench = Benchmark.model_validate_json((root/'data/benchmark/aapl-2024-10k.v1.json').read_text(encoding='utf-8'))
    # Validate review provenance against fresh ingestion, without treating an AI
    # verdict as human approval or inventing evidence for an absent answer.
    from src.benchmark import AIReview, bind_benchmark, review_markdown, validate_ai_review
    review = AIReview.model_validate_json(
        (root/'data/benchmark/aapl-2024-10k.ai-review.v3.json').read_text(encoding='utf-8'))
    bound = bind_benchmark(doc, bench)
    validate_ai_review(bound, review)
    packet = review_markdown(bound, ai_review=review, item_ids=['aapl24-023', 'aapl24-024'])
    assert 'confirmed' in packet
    assert bound['pending_review_count'] == 28
    assert not bound['ready_for_scored_evaluation']
    for item in bench.items:
        if item.id in {'aapl24-023', 'aapl24-024'}:
            assert not item.answerable and item.expected_answer is None
            assert item.evidence == [] and item.review_status == 'pending'
    report = evaluate_benchmark(doc, bench)
    snapshot = json.loads((root/'data/evaluation/aapl-2024-10k.context-v2.json').read_text(encoding='utf-8'))
    assert report == snapshot
    assert report['case_count'] == 35 and report['explicit_pages_complete'] == 35
    assert report['target_page_complete'] == 30 and report['single_chunk_complete'] == 27
    assert len(report['unsupported_item_ids']) == 5
    for row in report['results']:
        if row['item_id'] in {'aapl24-026', 'aapl24-027', 'aapl24-028'}:
            assert row['supporting_pages'] == [39, 40]
            assert not row['target_page_complete'] and row['explicit_pages_complete']
