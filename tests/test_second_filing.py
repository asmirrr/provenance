"""Real second-filing integration; skips explicitly when its source is unavailable."""
import json
from pathlib import Path

import pytest

from src.benchmark import Benchmark
from src.evaluation import evaluate_benchmark
from src.evidence import validate_document
from src.ingestion.pipeline import FilingMetadata, ingest


def test_apple_2023_fresh_ingestion_and_pinned_evidence():
    root = Path(__file__).resolve().parents[1]
    pdf = root / 'data/raw/aapl-2023-10k.pdf'
    if not pdf.exists():
        pytest.skip('Download the documented FY2023 PDF to enable this real-filing check')
    metadata = FilingMetadata.model_validate_json((root/'data/raw/aapl-2023-10k.metadata.json').read_text(encoding='utf-8'))
    doc = ingest(pdf, metadata)
    validate_document(doc)
    assert [p.page for p in doc.pages] == list(range(1, 81))
    assert len(doc.chunks) == 206 and doc.warnings == []
    assert all(c.fiscal_year == 2023 and c.filing_date == '2023-11-03' for c in doc.chunks)
    assert [doc.pages[p-1].printed_page for p in (31,33,35)] == [28,30,32]
    benchmark = Benchmark.model_validate_json((root/'data/benchmark/aapl-2023-10k.v1.json').read_text(encoding='utf-8'))
    report = evaluate_benchmark(doc, benchmark)
    expected = json.loads((root/'data/evaluation/aapl-2023-10k.context-v2.json').read_text(encoding='utf-8'))
    assert report == expected
    assert report['single_chunk_complete'] == 4
    assert report['explicit_pages_complete'] == 5
    other = Benchmark.model_validate_json((root/'data/benchmark/aapl-2024-10k.v1.json').read_text(encoding='utf-8'))
    with pytest.raises(ValueError, match='digest'):
        evaluate_benchmark(doc, other)
    other_snapshot = json.loads((root/'data/evaluation/aapl-2024-10k.context-v2.json').read_text(encoding='utf-8'))
    old_ids = {cid for row in other_snapshot['results'] for cid in row['supporting_chunk_ids']}
    assert old_ids.isdisjoint(c.chunk_id for c in doc.chunks)


def test_microsoft_2024_fresh_ingestion_and_index_regression():
    root = Path(__file__).resolve().parents[1]
    pdf = root / 'data/raw/msft-2024-10k.pdf'
    if not pdf.exists():
        pytest.skip('Download the documented Microsoft FY2024 PDF to enable this check')
    metadata = FilingMetadata.model_validate_json((root/'data/raw/msft-2024-10k.metadata.json').read_text(encoding='utf-8'))
    doc = ingest(pdf, metadata)
    validate_document(doc)
    assert [p.page for p in doc.pages] == list(range(1, 169))
    assert len(doc.chunks) == 401
    assert doc.pages[-1].text == ''
    assert doc.warnings == ['PDF page 168 has no extractable text; OCR was not run']
    assert not any(c.page == 168 for c in doc.chunks)
    assert all(c.section is None for c in doc.chunks if c.page == 2)
    assert all(c.section == 'ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA'
               for c in doc.chunks if c.page == 59)
    assert all(c.section is None for c in doc.chunks if c.page > 111)
    # Bare footer numbers are retained, never guessed to be financial values or removed.
    assert doc.pages[58].text.endswith('\n59') and doc.pages[58].printed_page is None
    benchmark = Benchmark.model_validate_json((root/'data/benchmark/msft-2024-10k.v1.json').read_text(encoding='utf-8'))
    report = evaluate_benchmark(doc, benchmark)
    expected = json.loads((root/'data/evaluation/msft-2024-10k.context-v2.json').read_text(encoding='utf-8'))
    assert report == expected
    assert report['single_chunk_complete'] == 4 and report['explicit_pages_complete'] == 5
    chunks = {c.chunk_id:c for c in doc.chunks}
    for row in report['results']:
        assert all(chunks[cid].section == 'ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA'
                   for cid in row['supporting_chunk_ids'])
